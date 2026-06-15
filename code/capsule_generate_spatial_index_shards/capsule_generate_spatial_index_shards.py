import sys
import logging
import os
import glob
from timeit import default_timer
from collections import defaultdict, Counter
import shutil
import tarfile
import random

import shared.annotations as anno
import shared.sharding as sharding
import shared.utilities as utilities

from shared.util import *
from shared.ram_data_pond import *
from shared.aws_storage import *

import capsule_generate_spatial_index_shards.finalize_annotations as fa

def read_shardworker_file():
    shard_worker_desc_files = list(glob.glob(f"{data_loc}shard_worker*txt"))
    assert len(shard_worker_desc_files) == 1
    shard_worker_desc_file_path = shard_worker_desc_files[0]
    shard_worker_desc_filename = os.path.basename(shard_worker_desc_file_path)
    shard_worker_desc_file_hash = shard_worker_desc_filename[:shard_worker_desc_filename.rindex('.')].split('_')[-1]
    logging.info(f"shard_worker_desc_file_hash parsed from shard worker description filename: {shard_worker_desc_file_hash}")
    if not os.path.exists(shard_worker_desc_file_path):
        raise RuntimeError(f"Expected input file not found: {shard_worker_desc_file_path}")
    with open(shard_worker_desc_file_path) as f:
        shard_worker_desc = f.read().strip()
    logging.info(f"shard_worker_desc_file_hash, shard_worker_desc: {shard_worker_desc_file_hash} {shard_worker_desc}")
    assigned_shards = set(shard_worker_desc.split('_'))
    # Add all versions of assigned shards with leading 0s stripped off
    additional_assigned_shards = set()
    for assigned_shard in assigned_shards:
        while assigned_shard and assigned_shard[0] == '0':
            assigned_shard = assigned_shard[1:]
            if assigned_shard:
                additional_assigned_shards.add(assigned_shard)
    assigned_shards.update(additional_assigned_shards)
    logging.info(f"Shard worker assigned shards: {assigned_shards}")
    logging.info("\n")

    # Copy the shard worker file to the output
    shutil.copy(shard_worker_desc_file_path, f"{results_loc}{shard_worker_desc_filename}")

    return shard_worker_desc_file_hash, assigned_shards

def download_data_from_bucket():
    if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
        if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] != "internal":
            st = default_timer()
            filename_filter = f"shard_worker-{shard_worker_desc_file_hash}"
            logging.info(f"A filename_filter: {filename_filter}")
            if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "gcp":
                raise RuntimeError("GCP bucket no longer supported due to possible financial cost if done incorrectly!")
                logging.info("\nDownloading files from Google storage")
                download_files_from_gcp(f"{config['TIMESTAMP']}/spatial_index", data_loc, filename_filter, config['GCP_BUCKET'], config['GCP_SCRATCH_BLOB_PATH'])#, dryrun=True)
                ets = default_timer() - st
                logging.info(f"\nGCP download elapsed time: {seconds_to_hms(ets)}")

                logging.info(f"\nMoving GCP downloads to {data_loc}")
                downloaded_files = glob.glob(f"{data_loc}{config['GCP_SCRATCH_BLOB_PATH']}/{config['TIMESTAMP']}/spatial_index/*")
                for downloaded_file in downloaded_files:
                    os.rename(downloaded_file, f"{data_loc}{os.path.basename(downloaded_file)}")
            elif config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "aws":
                logging.info("\nDownloading files from Amazon storage")
                download_folder_relative_path = f"{config['TIMESTAMP']}/spatial_index/tree_builder/"
                download_folder_from_aws(download_folder_relative_path, f"{data_loc}aws_downloads/", filename_filter, config['AWS_BUCKET'], config['AWS_PROJECT_PATH'], dryrun=False)
                ets = default_timer() - st
                logging.info(f"\nAWS download elapsed time: {seconds_to_hms(ets)}")

                logging.info(f"\nMoving AWS downloads to {data_loc}")
                downloaded_files = glob.glob(f"{data_loc}aws_downloads/*")
                for downloaded_file in downloaded_files:
                    logging.info(f"Moving {downloaded_file} -> {data_loc}{os.path.basename(downloaded_file)}")
                    os.rename(downloaded_file, f"{data_loc}{os.path.basename(downloaded_file)}")

            data_loc_contents = sorted(os.listdir(data_loc))
            data_loc_contents = [v for v in data_loc_contents if "placeholder" not in v]
            logging.info(f"\n{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):")
            logging.info('  ' + '\n  '.join(data_loc_contents[:30]).strip() + '\n')
            
            timestamps.append(("download_files_from_external_storage", default_timer()))
        else:
            logging.info(f"\n{data_loc}PASS_DATA_BETWEEN_CAPSULES_METHOD indicates Code Ocean. Results won't be downloaded from external bucket.")
    else:
        logging.info(f"\n{data_loc}DEBUG_FLAG.txt file found. Results won't be downloaded from GCP.")

def extract_input_files_before_shard_loop(ARCHIVE_MEMORY_STORE):
    if ARCHIVE_MEMORY_STORE and config['ARCHIVE_MEMORY_STORE_VIA_CUSTOM_METHOD']:
        if not config['ARCHIVE_COMPLETED_TREECELLS_WITH_SHARD_GROUPING']:
            logging.info("Dearchiving custom archives without shard grouping")
            input_files = list(glob.glob(f"{data_loc}*archive.txt"))
            file_shard_filters = [""]  # set([f"__shard-{shard}_" for shard in assigned_shards])
            total_num_subarchives_written, total_num_subarchive_groups_written, total_subarchive_size_written = 0, 0, 0
            st = default_timer()
            for input_file_i, input_file in enumerate(input_files):
                num_subarchives_written, num_subarchive_groups_written, total_subarchives_len = RAMDataPond.dearchive_file__group_by_treelevel_and_shard(input_file_i, data_loc, input_file, file_shard_filters)
                total_num_subarchives_written += num_subarchives_written
                total_num_subarchive_groups_written += num_subarchive_groups_written
                total_subarchive_size_written += total_subarchives_len
            logging.info(f"\nTotal dearchival time: {default_timer() - st:.3f}s")
            logging.info(f"Total num & size of subarchives dearchived, groups written, and bytes written: {total_num_subarchives_written}, {total_num_subarchive_groups_written}, {total_subarchive_size_written/1000000:.1f} MB")
    else:
        input_files = []
        if not ARCHIVE_MEMORY_STORE:
            if not config['ARCHIVE_COMPLETED_TREECELLS_WITH_SHARD_GROUPING']:
                logging.info("Dearchiving tar archives without shard grouping")
                split_sub_dir = ""  # "split-*/"
                input_files = list(glob.glob(f"{data_loc}{split_sub_dir}completed_treecells__*.tar*"))
                logging.info(f"\nInput completed_treecells archive files:\n  {'\n  '.join(input_files)}\n")
        else:  # not config['ARCHIVE_MEMORY_STORE_VIA_CUSTOM_METHOD']:
            logging.info("Dearchiving memory-store tar archives")
            input_files = list(glob.glob(f"{data_loc}split-*.tar*"))

        for input_file in input_files:
            mode = "r:gz" if config['COMPRESS_ARCHIVE'] else "r"
            with tarfile.open(input_file, mode) as tar:
                tar.extractall(path=f"{data_loc}")

def extract_input_files_by_shardworker_before_shard_loop(ARCHIVE_MEMORY_STORE):
    # Extract archives grouped by shard worker (not necessarily the same thing as a shard, since a worker can be responsible for multiple shards)
    found_shardworker_files = False
    if ARCHIVE_MEMORY_STORE and config['ARCHIVE_MEMORY_STORE_VIA_CUSTOM_METHOD'] and config['ARCHIVE_COMPLETED_TREECELLS_WITH_SHARD_GROUPING']:
        logging.info("Dearchiving custom archives with shard worker grouping")
        # Only process the input files assigned to this shard worker
        split_sub_dir = ""  # "split-*/"
        input_files = list(glob.glob(f"{data_loc}{split_sub_dir}split-*__shard_worker-{shard_worker_desc_file_hash}__archive.txt"))
        logging.info(f"\nInput completed_treecells archive files for this shard worker:\n  {'\n  '.join(input_files)}\n")
        if input_files:
            found_shardworker_files = True
            total_num_subarchives_written, total_num_subarchive_groups_written, total_subarchive_size_written = 0, 0, 0
            st = default_timer()
            for input_file_i, input_file in enumerate(input_files):
                num_subarchives_written, num_subarchive_groups_written, total_subarchives_len = RAMDataPond.dearchive_file__group_by_treelevel_and_shard(input_file_i, data_loc, input_file)
                total_num_subarchives_written += num_subarchives_written
                total_num_subarchive_groups_written += num_subarchive_groups_written
                total_subarchive_size_written += total_subarchives_len
            logging.info(f"\nTotal dearchival time: {default_timer() - st:.3f}s")
            logging.info(f"Total num & size of subarchives dearchived, groups written, and bytes written: {total_num_subarchives_written}, {total_num_subarchive_groups_written}, {total_subarchive_size_written/1000000:.1f} MB")
            dearchives = sorted(list(glob.glob(f"{data_loc}annotations_one_treecell*.csv")))
            logging.info(f"Results after dearchiving shard worker archives (first 20 shown):\n  {'\n  '.join(dearchives[:20])}")
            dearchives = sorted(list(glob.glob(f"{data_loc}annotations_one_treecell*/*")))
            logging.info(f"Results after dearchiving shard worker archives (first 20 shown):\n  {'\n  '.join(dearchives[:20])}")
        else:
            logging.warning("\n\nWARNING! No input files after extraction. Theoretically, this could happen if there are no files for the associated shards, but it is generally unlikely to occur. For the time being, this process was abort so any potential error can be investigated, but this might turn out to be false alarm.\n\n")
            # raise RuntimeError("\n\nNo input files after extraction. Theoretically, this could happen if there are no files for the associated shards, but it is generally unlikely to occur. For the time being, this process was abort so any potential error can be investigated, but this might turn out to be false alarm.\n\n")
    
    logging.info(f"found_shardworker_files: {found_shardworker_files}")

    return found_shardworker_files

def extract_input_files_inside_shard_loop(shard_hex, found_shardworker_files):
    if not found_shardworker_files:
        # Grouping by shard instead of shard worker will soon become deprecated, once the shard worker approach is validated
        logging.info("Proceeding to look for and extract per-shard input files (presumably no per-shardworker files were found)")
        if ARCHIVE_MEMORY_STORE and config['ARCHIVE_MEMORY_STORE_VIA_CUSTOM_METHOD']:
            if config['ARCHIVE_COMPLETED_TREECELLS_WITH_SHARD_GROUPING']:
                logging.info("Dearchiving custom archives with shard grouping")
                # Only process the input files assigned to this shard
                split_sub_dir = ""  # "split-*/"
                input_files = sorted(list(glob.glob(f"{data_loc}{split_sub_dir}split-*__shard-{shard_hex}__archive.txt")))
                logging.info(f"\nInput completed_treecells archive files for this shard:\n  {'\n  '.join(input_files)}\n")

            file_shard_filters = [""]  # set([f"__shard-{shard}_" for shard in assigned_shards])
            total_num_subarchives_written, total_num_subarchive_groups_written, total_subarchive_size_written = 0, 0, 0
            st = default_timer()
            for input_file_i, input_file in enumerate(input_files):
                num_subarchives_written, num_subarchive_groups_written, total_subarchives_len = RAMDataPond.dearchive_file__group_by_treelevel_and_shard(input_file_i, data_loc, input_file, file_shard_filters)
                total_num_subarchives_written += num_subarchives_written
                total_num_subarchive_groups_written += num_subarchive_groups_written
                total_subarchive_size_written += total_subarchives_len
            logging.info(f"\nTotal dearchival time: {default_timer() - st:.3f}s")
            logging.info(f"Total num & size of subarchives dearchived, groups written, and bytes written: {total_num_subarchives_written}, {total_num_subarchive_groups_written}, {total_subarchive_size_written/1000000:.1f} MB")
        else:
            input_files = []
            if not ARCHIVE_MEMORY_STORE:
                if config['ARCHIVE_COMPLETED_TREECELLS_WITH_SHARD_GROUPING']:
                    logging.info("Dearchiving tar archives with shard grouping")
                    # Only process the input files assigned to this shard
                    split_sub_dir = ""  # "split-*/"
                    input_files = sorted(list(glob.glob(f"{data_loc}{split_sub_dir}completed_treecells__shard-{shard_hex}_*.tar*")))
                    logging.info(f"\nInput completed_treecells archive files for this shard:\n  {'\n  '.join(input_files)}\n")
                else:
                    pass
            else:  # not config['ARCHIVE_MEMORY_STORE_VIA_CUSTOM_METHOD']:
                pass

            for input_file in input_files:
                mode = "r:gz" if config['COMPRESS_ARCHIVE'] else "r"
                with tarfile.open(input_file, mode) as tar:
                    tar.extractall(path=f"{data_loc}")

def process_input_files(shard_hex, input_files, max_tree_level):
    total_line_count = None
    max_tree_level_this_shard = 0
    subdirs_this_treelevel_shard = set()
    for input_file_i, input_file in enumerate(input_files):
        input_file_name = os.path.basename(input_file)
        input_file_name_no_ext = input_file_name[:-4]

        if input_file_i < 5:
            logging.info(f"\nProcessing file (only first 5 of {len(input_files)} shown): {input_file}")
            
        if False:  # DEBUG
            # Determining the line count requires scanning the file, which is otherwise unnecessary and potentially time-consuming.
            # Only do this for debugging, not production.
            total_line_count = 0
            with open(input_file) as f:
                line_count = 0
                for line_count, line in enumerate(f):
                    pass  # no-op
                line_count += 1
                # logging.info(f"Num lines (rows) in file: {line_count}")
                total_line_count += line_count

        pcs = input_file_name_no_ext.split('__')
        # logging.info(pcs)
        split_n_all = pcs[1].split('-')[1]
        split_id, num_splits = (int(v) for v in split_n_all.split('@'))
        tree_level = int(pcs[2].split('-')[1])
        # cell_id = pcs[3]

        if input_file_i < 5:
            logging.info(f"  Num splits, Split id, Tree level: {num_splits}, {split_id}, {tree_level}")
        
        if tree_level > max_tree_level_this_shard:
            logging.info(f"  New max tree level for this shard: {tree_level}")
            max_tree_level_this_shard = tree_level

        # if False:  # DEBUG
        #     tree_level_cell_id = [int(v) for v in cell_id.split('-')[1].split(',')]
        #     grid_dim = 2 ** tree_level
        #     grid_shape = (grid_dim, grid_dim, grid_dim)
        #     morton_code = utilities.compressed_morton_code(tree_level_cell_id, grid_shape)

        #     sharding_spec = anno.ShardingSpec(
        #         hash=config['SPATIAL_SHARDING_HASH'],
        #         preshift_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['preshift_bits'],
        #         shard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits'],
        #         minishard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['minishard_bits'])

        #     shard_num = sharding_spec.get_shard_number(morton_code)
        #     shard_hex2 = sharding.get_shard_hex(shard_num, config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits'])
        #     # logging.info(f"  Tree level cell id Morton code, shard num, hex: {morton_code} {shard_num} {shard_hex}")

        #     assert shard_hex2 == shard_hex

        # subdir = f"treelevel-{tree_level:02}__{cell_id}__shard-{shard_hex}/"
        subdir = f"treelevel-{tree_level:02}__shard-{shard_hex}/"
        subdirs_this_treelevel_shard.add(subdir)
        assert not os.path.exists(f"{results_loc}{subdir}/{input_file_name}")
        os.makedirs(f"{results_loc}{subdir}", exist_ok=True)
        shutil.copy(input_file, f"{results_loc}{subdir}/{input_file_name}")

    # Only happens if enabled above
    if total_line_count is not None:
        logging.info(f"\nTotal num lines (rows) processed: {total_line_count}")

    logging.info(f"Max tree level this shard: {max_tree_level_this_shard}")
    if max_tree_level_this_shard > max_tree_level:
        max_tree_level = max_tree_level_this_shard

    logging.info(f"Subdirs created for this shard (i.e., all tree levels for this shard):\n  {'\n  '.join(subdirs_this_treelevel_shard)}")

    return max_tree_level, subdirs_this_treelevel_shard

def merge_splits(subdirs_this_treelevel_shard):
    # This next section incorporates the entire Combiner capsule into this capsule, thereby obviating the Combiner.
    # Once this section's behavior is validated, the Combiner capsule can be removed from the pipeline,
    # leaving the Conglomerator as the next capsule after this one.

    for subdir_this_treelevel_shard in subdirs_this_treelevel_shard:
        logging.info(f"Combining all splits in subdir {subdirs_this_treelevel_shard}")
        assert subdir_this_treelevel_shard[-1] == '/'
        pcs = subdir_this_treelevel_shard[:-1].split('__')
        tree_level = pcs[0].split('-')[1]
        shard_hex = pcs[1].split('-')[1]
        logging.info(f"  Subdir tree level & shard hex: {tree_level} {shard_hex}")
        subdir_files = sorted(list(glob.glob(f"{results_loc}{subdir_this_treelevel_shard}*")))
        logging.info(f"  Files in subdir (first 5 shown):\n    {'\n    '.join(subdir_files[:5])}")
        merged_file = ""
        for subdir_file_i, subdir_file in enumerate(subdir_files):
            with open(subdir_file) as f:
                file_content = f.read()
            if file_content[-1] != '\n':
                file_content += '\n'
            if subdir_file_i == 0:
                logging.info(f"  First subdir_file first 500 chars:\n{file_content[:500]}\n")
            merged_file += file_content
        logging.info(f"  Merged file len: {len(merged_file)/1000000:.1f} MB")
        merged_filepath = f"{results_loc}{subdir_this_treelevel_shard}annotations_one_treecell__treelevel-{tree_level:02}__shard-{shard_hex}.csv"
        logging.info(f"  Writing merged file to {merged_filepath}")
        with open(merged_filepath, 'w') as f:
            f.write(merged_file)
        if not PRESERVE_ALL_FIlES:
            for sdfi, subdir_file in enumerate(subdir_files):
                if sdfi <= 3:
                    logging.info(f"  Removing single-split file {sdfi+1} of {len(subdir_files)} (first 3 shown): {subdir_file}")
                os.remove(subdir_file)

def archive_merged_splits(shard_hex):
    # This was copied from the Combiner,
    # but I think the original archiving code below is fine.
    # I just need to alter the Conglomerator to expect that tar file name instead.
    # if config['ARCHIVE_COMBINED_OUTPUT']:
    #     for shard_hex in assigned_shards:
    #         logging.info(f"Archiving merged shard {shard_hex}")
    #         output_files_one_shard = list(glob.glob(f"{results_loc}treelevel-*__shard-{shard_hex}"))
    #         logging.info(f"Found {len(output_files_one_shard)} output files for all tree levels, shard {shard_hex}")
    #         if config['COMBINE_BY_TREELEVEL']:
    #             tree_levels = sorted(list(set(os.path.basename(output_file).split('__')[0].split('-')[1] for output_file in output_files_one_shard)))
    #             logging.info(f"Output file tree levels: {tree_levels}")
    #             for tree_level in tree_levels:
    #                 output_files_one_tree_level = list(glob.glob(f"{results_loc}treelevel-{tree_level}__shard-{shard_hex}"))
    #                 logging.info(f"\nTarring and compressing {len(output_files_one_tree_level)} output files for tree level {tree_level}, shard {shard_hex}")
    #                 ext, mode = (".tar.gz", "w:gz") if config['COMPRESS_ARCHIVE'] else (".tar", "w")
    #             with tarfile.open(f"{results_loc}treelevel-{tree_level}__shard-{shard_hex}{ext}", mode) as tar:
    #                 for output_file in output_files_one_tree_level:
    #                     # logging.info(f"  Adding split shard file to tar: {output_file}")
    #                     tar.add(output_file, arcname=os.path.basename(output_file))
    #         else:
    #             assert False
    #     timestamps.append(("archive_merged_splits", default_timer()))
    
    if not found_shardworker_files:
        logging.info(f"archive_merged_splits() found_shardworker_files is false. Proceeding to archive one shard.")
        if config['ARCHIVE_REGROUPED_OUTPUT']:
            # Tar and compress the results
            # results_loc_contents = sorted(list(glob.glob(f"{results_loc}treelevel-*__treelevelcellid-*__shard-{shard_hex}")))
            results_loc_contents = sorted(list(glob.glob(f"{results_loc}treelevel-*__shard-{shard_hex}")))
            logging.info(f"Results contents (first 20 shown):\n  {'\n  '.join(results_loc_contents[:20])}\n")
            if results_loc_contents:
                ext, mode = (".tar.gz", "w:gz") if config['COMPRESS_ARCHIVE'] else (".tar", "w")
                with tarfile.open(f"{results_loc}regrouper__tree_cell_shards__shard-{shard_hex}{ext}", mode) as tar:
                    for results_loc_content in results_loc_contents:
                        if results_loc_content[-1] == '/':
                            results_loc_content = treecell_dir[:-1]
                        # logging.info(f"  Adding treecell dir to tar: {results_loc_content}")
                        tar.add(results_loc_content, arcname=os.path.basename(results_loc_content))
                if not PRESERVE_ALL_FIlES and not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
                    for results_loc_content in results_loc_contents:
                        shutil.rmtree(results_loc_content)
            timestamps.append(("archive_merged_splits", default_timer()))
            
def process_shards(found_shardworker_files):
    max_tree_level = 0
    
    for shard_hex in assigned_shards:
        logging.info("\n" + "*" * 100 + "\n")
        timestamps.append(("shard_loop_top", default_timer()))

        logging.info(f"Processing shard {shard_hex}")

        extract_input_files_inside_shard_loop(shard_hex, found_shardworker_files)
        timestamps.append(("extract_input_files_inside_shard_loop", default_timer()))

        # input_files = sorted(list(glob.glob(f"{data_loc}annotations_one_treecell*/*.csv")))
        # input_files = sorted(list(glob.glob(f"{data_loc}annotations_one_treecell*/*shard-{shard_hex}.csv")))
        input_files = sorted(list(glob.glob(f"{data_loc}annotations_one_treecell*__shard-{shard_hex}.csv")))
        # logging.info(f"\nInput annotations_one_treecell files after extraction ({len(input_files)}) (first 5 shown):\n  {'\n  '.join(input_files[:5])}\n")
        logging.info(f"\nInput annotations_one_treecell files for this shard after extraction ({len(input_files)}) (first 5 shown):\n  {'\n  '.join(input_files[:5])}\n")

        timestamps.append(("gather_input_files_after_extraction", default_timer()))

        # DEBUG
        # data_loc_contents = sorted(list(glob.glob(f'{data_loc}*')))
        # logging.info(f"\n{data_loc}* (first 5 shown):\n  {'\n  '.join(data_loc_contents[:5])}")
        # annotations_one_treecell_contents = sorted(list(glob.glob(f'{data_loc}annotations_one_treecell*/*')))
        # logging.info(f"\n{data_loc}annotations_one_treecell*/* (first 5 shown):\n  {'\n  '.join(annotations_one_treecell_contents[:5])}")
        # logging.info("")
        # timestamps.append(("debug_output____disable_this_block_if_it_takes_a_long_time", default_timer()))

        # Process each input file for this shard.
        # All that this "processing" consists of is reorganizing the files on disk from their received input organization
        # into an organization that groups them into subdirectories by tree level and shard
        # (this effectively groups multiple splits' outputs together, but keeps files separated by shard treelevel and shard).

        max_tree_level, subdirs_this_treelevel_shard = process_input_files(shard_hex, input_files, max_tree_level)
        timestamps.append(("regroup_shard_files", default_timer()))
        
        # This next section incorporates the entire Combiner capsule into this capsule, thereby obviating the Combiner.
        # Once this section's behavior is validated, the Combiner capsule can be removed from the pipeline,
        # leaving the Conglomerator as the next capsule after this one.

        merge_splits(subdirs_this_treelevel_shard)
        timestamps.append(("merge_splits", default_timer()))

        archive_merged_splits(shard_hex)
    
    return max_tree_level

def archive_results(found_shardworker_files):
    if found_shardworker_files:
        logging.info("\n" + "* " * 50 + "\n")
        if config['ARCHIVE_REGROUPED_OUTPUT']:
            logging.info(f"archive_results() found_shardworker_files is true. Proceeding to archive this shard worker.")
            # Tar and compress the results
            # results_loc_contents = sorted(list(glob.glob(f"{results_loc}treelevel-*__treelevelcellid-*__shard-*")))
            results_loc_contents = sorted(list(glob.glob(f"{results_loc}treelevel-*__shard-*")))
            logging.info(f"Results contents (first 20 shown):\n  {'\n  '.join(results_loc_contents[:20])}\n")
            if results_loc_contents:
                ext, mode = (".tar.gz", "w:gz") if config['COMPRESS_ARCHIVE'] else (".tar", "w")
                with tarfile.open(f"{results_loc}regrouper__tree_cell_shards__shard_worker-{shard_worker_desc_file_hash}{ext}", mode) as tar:
                    for results_loc_content in results_loc_contents:
                        if results_loc_content[-1] == '/':
                            results_loc_content = treecell_dir[:-1]
                        logging.info(f"  Adding treecell dir to tar: {results_loc_content}")
                        tar.add(results_loc_content, arcname=os.path.basename(results_loc_content))
                if not PRESERVE_ALL_FIlES and not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
                    for results_loc_content in results_loc_contents:
                        shutil.rmtree(results_loc_content)
            timestamps.append(("archive_output", default_timer()))

def upload_results_to_bucket():
    if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
        if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] != "internal":
            files_to_upload_to_scratch = glob.glob(f"{results_loc}regrouper__tree_cell_shards__*.tar*")
            logging.info("\nUploading files to AWS storage")
            upload_folder_relative_path = f"aws_upload/{config['TIMESTAMP']}/spatial_index/regrouper/"
            os.makedirs(f"{results_loc}{upload_folder_relative_path}", exist_ok=True)
            for file in files_to_upload_to_scratch:
                # logging.info(f"Move {file} -> {results_loc}{upload_folder_relative_path}{os.path.basename(file)}")
                shutil.move(file, f"{results_loc}{upload_folder_relative_path}{os.path.basename(file)}")
            upload_folder_to_aws(f"{results_loc}aws_upload/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])#, dryrun=True)
            query_folder_on_aws(f"{config['TIMESTAMP']}/spatial_index/regrouper/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])
            for file in files_to_upload_to_scratch:
                file_dir = '/'.join(file.split('/')[:-1])
                # Move the files back so we can delete them from the original location below
                # logging.info(f"Move {results_loc}{upload_folder_relative_path}{os.path.basename(file)} -> {file}")
                shutil.move(f"{results_loc}{upload_folder_relative_path}{os.path.basename(file)}", file)
            if not PRESERVE_ALL_FIlES:
                shutil.rmtree(f"{results_loc}aws_upload/")
                for file in files_to_upload_to_scratch:
                    logging.info(f"Deleting result file after uploading to external bucket: {file}")
                    os.remove(file)
            timestamps.append(("upload_results_to_scratch_bucket", default_timer()))
        else:
            logging.info(f"\n{data_loc}PASS_DATA_BETWEEN_CAPSULES_METHOD indicates Code Ocean. Results won't be uploaded externally.")
    else:
        logging.info(f"\n{data_loc}DEBUG_FLAG.txt file found. Results won't be uploaded externally.")

if __name__ == "__main__":
    PRESERVE_ALL_FIlES = False

    data_loc = "../data/"
    results_loc = "../results/"

    logging_uid = hex(int(random.random()*1000000000000))[2:]
    # os.makedirs(f"{results_loc}logs/", exist_ok=True)

    logging.basicConfig(level=logging.CRITICAL, handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{results_loc}log_regroup_spatial_oct_tree_worker_outputs_{logging_uid}.log", mode="a")
        ], format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("REGROUP SPATIAL INDEX OCT TREE SPLITS")

    config = read_config(["id", "relation", "spatial"])
    logging.basicConfig(level=get_logging_level_from_desc(config['LOGGING_LEVEL']), handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{results_loc}log_regroup_spatial_oct_tree_worker_outputs_{logging_uid}.log", mode="a")
        ], format=config['LOGGING_FORMAT'], force=True)
    
    timestamps = []
    timestamps.append(("start", default_timer()))

    if config['SPATIAL_INDEX_ENABLED']:
        data_loc_contents = sorted(os.listdir(data_loc))
        data_loc_contents = [v for v in data_loc_contents if "placeholder" not in v]
        logging.info(f"{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):")
        logging.info('  ' + '\n  '.join(data_loc_contents[:30]).strip() + '\n')
        
        logging.info(f"{data_loc} subcontents ({len(list(glob.glob(f"{data_loc}*/*")))}) (first 5 shown):\n  {'\n  '.join(sorted(list(glob.glob(f"{data_loc}*/*"))[:5])).strip()}")
        logging.info("\n")

        # Look for input files that indicate the tree-building daisy chain isn't long enough.
        # There are two categories of output produced by the daisy chain, i.e., by the 'build spatial index oct tree' capsule:
        #   - split...subtree...
        #   - synapses_one_treecell...
        # The first type is produced throughout the daisy chain process to pass unprocessed subtrees to the next stage of the daisy chain.
        # The second is produced when a cell of the tree is complete. This output holds all the annotations for one tree cell, ready for finalization (split-combination, sharding, etc.)
        # By the time the daisy chain has built the entire tree, there should be no remaining subtree outputs. They are acceptable throughout the daisy chain process, but should be gone by the time the process has bottomed out.
        # Therefore, the presence of any subtree outputs at this stage of processing indicates that the daisy chain was not long enough for the original input data size.
        # In such a case, the Code Ocean pipeline needs to be altered to add additional capsules to the daisy chain.

        subtree_input_dirs = list(glob.glob(f"{data_loc}split-*/split*subtree*")) + list(glob.glob(f"{results_loc}split-*/subtrees.tar*"))
        logging.info(f"subtree_input_dirs (SHOULD BE EMPTY): {subtree_input_dirs}")
        if subtree_input_dirs:
            raise RuntimeError("There are remaining unprocessed subtrees. The daisy chain is not long enough for the input data size. You must either increase the MAX_DATA_ROWS_PER_TREE_CELL config parameter or add additional capsules to the daisy chain in the pipeline.")
        logging.info("\n")

        shard_worker_desc_file_hash, assigned_shards = read_shardworker_file()
    
        # Copy upstream logs from input to output
        if "0" in assigned_shards:  # To avoid CodeOcean name collisions, only do this from one capsule
            logs = sorted(list(glob.glob(f"{data_loc}log*.log")))
            for log in logs:
                # logging.info(f"Copying log from {data_loc} to {results_loc}logs/: {log}")
                # shutil.copy(log, f"{results_loc}logs/{os.path.basename(log)}")
                logging.info(f"Copying log from {data_loc} to {results_loc}: {log}")
                shutil.copy(log, f"{results_loc}{os.path.basename(log)}")
        
        timestamps.append(("read_shardworker_file", default_timer()))
    
        if config['ARCHIVE_OUTPUT']:
            input_files = list(glob.glob(f"{data_loc}split-*/annotations_one_treecell*/*.csv"))
            logging.info(f"Input annotations_one_treecell directory files (SHOULD BE EMPTY): {input_files}")
            if input_files:
                logging.error("\nERROR! There should be no completed tree cell directories. There should only .tar.gz files!\n")
        
        # Prior to the external bucket method, the incoming files from the tree building capsules would be grouped into split-specific subdirectories. The external bucket method doesn't however. The files all live in a single directory. One solution would be to adapt the newer external bucket method to maintain the split subdirectory organization, but it isn't really necessary. So instead, let's move all files, regardless of method out of any split subdirectories.
        split_subdir_files = glob.glob(f"{data_loc}split-*/*")
        for split_subdir_file in split_subdir_files:
            shutil.move(split_subdir_file, data_loc)
        
        timestamps.append(("move_files_out_of_split_subdirs", default_timer()))
        
        download_data_from_bucket()
        
        # Keep this value aligned with the previous capsule.
        # TODO: put these in pipeline-level config parameters.
        ARCHIVE_MEMORY_STORE = True  # Pack RAM data pond into a tar buffer and write a single tar file to disk. Else, write each RAM data pond file to a separate file on disk (which could be 1000s and impede CodeOcean performance).

        extract_input_files_before_shard_loop(ARCHIVE_MEMORY_STORE)
        timestamps.append(("extract_input_files_before_shard_loop", default_timer()))
        
        found_shardworker_files = extract_input_files_by_shardworker_before_shard_loop(ARCHIVE_MEMORY_STORE)
        timestamps.append(("extract_input_files_by_shardworker_before_shard_loop", default_timer()))
            
        max_tree_level = process_shards(found_shardworker_files)
        
        # We don't need to archive the results or upload them to a bucket if we are running the conglomerator right here in this capsule momentarily (below).
        # archive_results(found_shardworker_files)
        # upload_results_to_bucket()
        # Instead, move the files from the results loc to the data loc so the conglomerator can fine them
        results_loc_contents = sorted(list(glob.glob(f"{results_loc}treelevel-*__shard-*")))
        for results_loc_content in results_loc_contents:
            file_name = os.path.basename(results_loc_content)
            logging.info(f"Moving:\n  {results_loc_content} to\n  {data_loc}{file_name}")
            shutil.move(results_loc_content, f"{data_loc}{file_name}")

        logging.info("\n" + "* " * 50 + "\n")
    
        logging.info(f"Max tree level all shards: {max_tree_level}")
        with open(f"{results_loc}max_tree_level-{max_tree_level:02}__shard_worker-{shard_worker_desc_file_hash}.txt", 'w') as f:
            f.write(f"{'-'.join(sorted(assigned_shards))}\n")
            f.write(f"{max_tree_level}\n")
    
    finalize_results(results_loc)
    timestamps.append(("finalize_results", default_timer()))
    
    logging.error("\nElapsed timestamps:")
    accum_elapsed_times = Counter()
    for ti, time in enumerate(timestamps):
        if ti > 0:
            elap_t = time[1] - timestamps[ti-1][1]
            accum_elapsed_times[time[0]] += elap_t
            logging.error(f"  {seconds_to_hms(elap_t)} {time[0]}")
        
    logging.error("Accumulated elapsed timestamps:")
    for label, elap_t in accum_elapsed_times.items():
        logging.error(f"  {seconds_to_hms(elap_t)} {label}")

    print("\n\n\n")
    print("/\\__" * 25)
    print("/\\__" * 25)
    print("/\\__" * 25)
    print("\nBeginning sharded precomputed file generation and bucket uploading")
    print("\n\n\n")
    fa.main()

logging.info("\nDone")
process_running_time()
