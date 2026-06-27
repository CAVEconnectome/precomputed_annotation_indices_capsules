import sys
import logging
import os
import glob
import math
from timeit import default_timer
import pandas as pd
import json
import random
import shutil

from shared.google_storage import *

import shared.simple_writer_no_spatial_indexing as simple_writer_no_spatial_indexing
import shared.annotations as anno
import shared.utilities as utilities

from shared.util import *

def generate_info_files(max_tree_level_all_shards, annotation_type):
    targets = ["unsharded", "sharded"] if config['SPATIAL_INDEX_UNSHARDED_ENABLED'] else ["sharded"]
    for sharded in targets:
        dimensions = config['DATA_CONFIG']['dimensions']
        if "docstring" in dimensions:
            del dimensions["docstring"]
        cell_bounds_low = config['DATA_CONFIG']['volume_bounds'][0]
        cell_bounds_high = config['DATA_CONFIG']['volume_bounds'][1]
        logging.info(f"DIMENSIONS: {dimensions}")
        logging.info(f"BOUNDS LOW: {cell_bounds_low}")
        logging.info(f"BOUNDS HIGH: {cell_bounds_high}")
        
        writer = simple_writer_no_spatial_indexing.SimpleWriter(annotation_type, dimensions, cell_bounds_low, cell_bounds_high)
        
        if config['ID_SHARDING']:
            writer.by_id_sharding = anno.ShardingSpec(
                hash=config['ID_SHARDING_HASH'],
                preshift_bits=config['ID_PRESHIFT_BITS'],
                shard_bits=config['ID_SHARDING_BITS'],
                minishard_bits=config['ID_MINISHARDING_BITS']
            )
        
        data_properties = config['DATA_CONFIG']['properties']
        for property_name, property_info in data_properties.items():
            if property_name != "vector":
                writer.property_specs.append(anno.PropertySpec(property_name, property_info['type'], property_name, property_info['enum_values'], property_info['enum_labels']))
            else:
                for dim in ["x", "y", "z"]:
                    writer.property_specs.append(anno.PropertySpec(f"vector_{dim}", "float32", f"vector_{dim}", None, None))

        bounds_range = [cell_bounds_high[d] - cell_bounds_low[d] for d in range(3)]
        voxel_extent = [
            bounds_range[0],
            bounds_range[1],
            bounds_range[2],
        ]
        for tree_level_ve in range(max_tree_level_all_shards+1):
            spatial_sharding_spec = anno.ShardingSpec(
                hash=config['SPATIAL_SHARDING_HASH'],
                preshift_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level_ve]['preshift_bits'],
                shard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level_ve]['shard_bits'],
                minishard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level_ve]['minishard_bits']
            ) if sharded == "sharded" else None
            
            limit = config['MAX_DATA_ROWS_PER_TREE_CELL']
            writer.spatial_specs.append(anno.SpatialEntry(voxel_extent, [2**tree_level_ve, 2**tree_level_ve, 2**tree_level_ve], f"spatial{tree_level_ve}", limit, sharding=spatial_sharding_spec))
            voxel_extent = [v/2 for v in voxel_extent]
        
        relation_sharding_spec = anno.ShardingSpec(
            hash=config['RELATION_SHARDING_HASH'],
            preshift_bits=config['RELATION_PRESHIFT_BITS'],
            shard_bits=config['RELATION_SHARDING_BITS'],
            minishard_bits=config['RELATION_MINISHARDING_BITS']
        ) if config['RELATION_SHARDING'] else None
        for relation in config['DATA_CONFIG']['relations']:
            writer.relationships.append(anno.Relationship(relation, sharding=relation_sharding_spec))
        
        # The unsharded info file gets pushed into a subdirectory out of the way.
        # The sharded info file is put at the top-level so as to reflect the Neuroglancer expected directory structure.
        dst_path = f"{results_loc}" if sharded == "sharded" else f"{results_loc}spatial_index__{sharded}"
        info_file_path = utilities.path_join(dst_path, "info")
        
        info_content = writer.format_info()
        utilities.write_bytes(info_file_path, info_content.encode("utf-8"))

def reorganize_id_index():
    logging.info("." * 100)
    logging.info("\nreorganize_id_index")

    # shard_dirs = list(glob.glob(f"{data_loc}id_index__shard*"))
    # shards = [shard_dir.split('/')[-1].split('-')[1] for shard_dir in shard_dirs]

    shard_files = list(glob.glob(f"{data_loc}id_index_*.shard"))
    logging.info(f"Shard files: {shard_files}")
    
    options = ["unsharded", "sharded"] if config['SPATIAL_INDEX_UNSHARDED_ENABLED'] else ["sharded"]
    targets = ["unsharded", "sharded"]
    for sharded in targets:
        os.makedirs(f"{results_loc}spatial_index__{sharded}__tmp/by_id/", exist_ok=True)

        # for shard_dir in shard_dirs:
        for shard_file in shard_files:
            # logging.info(f"  Shard dir:  {shard_dir}")
            # shard = shard_dir.split('/')[-1].split('-')[1]
            shard = shard_file.split('/')[-1].split('_')[2].split('.')[0]
            # logging.info(f"  Shard:      {shard}")
            shard_filename = f"{shard}.shard"
            # shard_file = f"{shard_dir}/id_index/{shard_filename}"
            # logging.info(f"  Shard file: {shard_file}")
            shutil.copy(shard_file, f"{results_loc}spatial_index__{sharded}__tmp/by_id/{shard_filename}")

def reorganize_relation_index():
    logging.info("." * 100)
    logging.info("\nreorganize_relation_index")

    # relation_shard_subdirs = list(glob.glob(f"{data_loc}/relation_indices__*__shard-*"))
    # logging.info(f"Relation shard subdirs: {relation_shard_subdirs}")

    relation_shard_files = list(glob.glob(f"{data_loc}/relation_indices__*__*.shard"))
    logging.info(f"Relation shard files: {relation_shard_files}")

    targets = ["unsharded", "sharded"]
    for sharded in targets:
        # for relation_shard_subdir in relation_shard_subdirs:
        for shard_file in relation_shard_files:
            # logging.info(f"\n  Relation shard subdir: {relation_shard_subdir}")
            # relation_shard_dirname = relation_shard_subdir.split('/')[-1]
            # relation = relation_shard_dirname.split('__')[1]
            relation = shard_file.split('/')[-1].split('__')[1]
            shard = shard_file.split('/')[-1].split('__')[2].split('.')[0]
            # logging.info(f"  Relation: {relation}")
            # logging.info(f"  Shard: {shard}")
                
            os.makedirs(f"{results_loc}spatial_index__{sharded}__tmp/{relation}/", exist_ok=True)

            shard_filename = f"{shard}.shard"
            # shard_file = f"{relation_shard_subdir}/{relation}/{shard_filename}"
            # logging.info(f"  Shard file: {shard_file}")
            shutil.copy(shard_file, f"{results_loc}spatial_index__{sharded}__tmp/{relation}/{shard_filename}")

def reorganize_spatial_unsharded_index():
    logging.info("." * 100)
    logging.info("\nreorganize_spatial_unsharded_index")

    tree_subdirs = list(glob.glob(f"{data_loc}spatial_index__unsharded__*"))
    logging.info(f"Unsharded tree subdirs: {tree_subdirs}")
    
    max_tree_level = 0
    for tree_subdir in tree_subdirs:
        logging.info("\n")
        logging.info(f"  Tree subdir:  {tree_subdir}")
        tree_subdirname = tree_subdir.split('/')[-1]
        tree_level = int(tree_subdirname.split('_')[5].split('-')[1])
        logging.info(f"  Tree level: {tree_level}")
        # shard = tree_subdirname.split('_')[6].split('-')[1]
        # logging.info(f"  Shard: {shard}")

        if tree_level > max_tree_level:
            max_tree_level = tree_level

        tree_cell_files = list(glob.glob(f"{tree_subdir}/spatial{tree_level}/*"))
        if tree_cell_files:
            os.makedirs(f"{results_loc}spatial_index_unsharded/spatial{tree_level}/", exist_ok=True)
            logging.info(f"  Tree cell files: {tree_cell_files}")
            for tree_cell_file in tree_cell_files:
                logging.info(f"  Tree cell file: {tree_cell_file}")
                shutil.copy(f"{tree_cell_file}", f"{results_loc}spatial_index_unsharded/spatial{tree_level}/")
    
    # On the off chance the upper levels of the tree didn't receive any points, go ahead and create empty directories for them.
    # Note that Code Ocean drops empty directories on the floor though (they don't appear in the results/) so we have to put a placeholder in each one as well.
    # for tree_level in range(max_tree_level+1):
    #     logging.info(f"Creating {results_loc}spatial_index_unsharded/spatial{tree_level}/")
    #     os.makedirs(f"{results_loc}spatial_index_unsharded/spatial{tree_level}/", exist_ok=True)
    #     with open(f"{results_loc}spatial_index_unsharded/spatial{tree_level}/placeholder.txt", 'w') as f:
    #         f.write("Placeholder")

def reorganize_spatial_sharded_index():
    logging.info("." * 100)
    logging.info("\nreorganize_spatial_sharded_index")

    tree_subdirs = list(glob.glob(f"{data_loc}spatial_index__sharded__*"))
    logging.info(f"Sharded tree subdirs: {tree_subdirs}")
    
    max_tree_level = 0
    for tree_subdir in tree_subdirs:
        if tree_subdir.endswith("csv"):
            continue
        logging.info(f"  Tree subdir:  {tree_subdir}")
        tree_subdirname = tree_subdir.split('/')[-1]
        tree_level = int(tree_subdirname.split('_')[5].split('-')[1])
        # logging.info(f"  Tree level: {tree_level}")
        shard = tree_subdirname.split('_')[6].split('-')[1]
        # logging.info(f"  Shard: {shard}")

        if tree_level > max_tree_level:
            max_tree_level = tree_level

        tree_level_subdirs = list(glob.glob("{tree_subdir}/spatial{tree_level}"))
        if tree_level_subdirs:
            os.makedirs(f"{results_loc}spatial_index_sharded/spatial{tree_level}/", exist_ok=True)
            shutil.copy(f"{tree_subdir}/spatial{tree_level}/{shard}.shard", f"{results_loc}spatial_index_sharded/spatial{tree_level}/{shard}.shard")
    
    # On the off chance the upper levels of the tree didn't receive any points, go ahead and create empty directories for them.
    # Note that Code Ocean drops empty directories on the floor though (they don't appear in the results/) so we have to put a placeholder in each one as well.
    # for tree_level in range(max_tree_level+1):
    #     logging.info(f"Creating {results_loc}spatial_index_sharded/spatial{tree_level}/")
    #     os.makedirs(f"{results_loc}spatial_index_sharded/spatial{tree_level}/", exist_ok=True)
    #     with open(f"{results_loc}spatial_index_sharded/spatial{tree_level}/placeholder.txt", 'w') as f:
    #         f.write("Placeholder")

def reorganize_spatial_index():
    # logging.info("." * 100)
    # logging.info("\nreorganize_spatial_index")

    reorganize_spatial_unsharded_index()
    reorganize_spatial_sharded_index()

if __name__ == "__main__":
    data_loc = "../data/"
    results_loc = "../results/"

    logging_uid = hex(int(random.random()*1000000000000))[2:]
    os.makedirs(f"{results_loc}logs/", exist_ok=True)
    
    # Don't copy any upstream logs through to the output.
    # They might be received, say from the spatial regroupers, but they would be duplicates coming from the
    # last spatial index capsule and would then cause name collisions.
    
    logging.basicConfig(level=logging.CRITICAL, handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{results_loc}/logs/log_reorganize_directory_structure_{logging_uid}.log", mode="a")
        ], format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("REORGANIZE DIRECTORY STRUCTURE")

    config = read_config(["id", "relation", "spatial"])

    # Debug
    config['SPATIAL_INDEX_UNSHARDED_ENABLED'] = False
    config['SPATIAL_SHARDING_HASH'] = "murmurhash3_x86_128"

    logging.basicConfig(level=get_logging_level_from_desc(config['LOGGING_LEVEL']), handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{results_loc}/logs/log_reorganize_directory_structure_{logging_uid}.log", mode="a")
        ], format=config['LOGGING_FORMAT'], force=True)

    data_loc_contents = sorted(os.listdir(data_loc))
    data_loc_contents = [v for v in data_loc_contents if "placeholder" not in v]
    logging.info(f"{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):")
    logging.info('  ' + '\n  '.join(data_loc_contents[:30]).strip() + '\n')
    
    # It's possible that *not* transferring the log files to the final results bucket might decrease the total CO job time even if any given capsule isn't too badly affected (or perhaps it affects this one capsule perhaps)
    COPY_LOGS_TO_RESULTS = False
    if COPY_LOGS_TO_RESULTS:
        # Copy upstream logs from input to output
        logging.info(f"\nCopying log files to logs/ output")
        logs = sorted(list(glob.glob(f"{data_loc}log*.log")))
        for log in logs:
            logging.info(f"  Copying log from {data_loc} to {results_loc}logs/: {log}")
            shutil.copy(log, f"{results_loc}logs/{os.path.basename(log)}")
    
    # Consolidate the spatial tree level directories
    logging.info(f"\nConsolidating spatial directories")
    spatial_dirs = list(glob.glob(f"{data_loc}spatial*__shard_worker-*"))
    for spatial_dir in spatial_dirs:
        spatial_dir_no_shard = os.path.basename(spatial_dir).split('__')[0]
        logging.info(f"  Spatial dir without shard worker suffix: {spatial_dir_no_shard}")
        os.makedirs(f"{results_loc}{spatial_dir_no_shard}", exist_ok=True)
        spatial_shard_files = list(glob.glob(f"{spatial_dir}/*.shard"))
        for spatial_shard_file in spatial_shard_files:
            spatial_shard_file_moved = f"{results_loc}{spatial_dir_no_shard}/{os.path.basename(spatial_shard_file)}"
            logging.info(f"  Moving spatial file from {spatial_shard_file} to {spatial_shard_file_moved}")
            shutil.move(spatial_shard_file, spatial_shard_file_moved)

    # Write the pipeline config to the output
    with open(f"{results_loc}pipeline_config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # Determine the annotation type
    annotation_type = None
    if "point_annotation_config" in config['DATA_CONFIG']:
        annotation_type = "POINT"
    elif "line_annotation_config" in config['DATA_CONFIG']:
        annotation_type = "LINE"
    elif "polyline_annotation_config" in config['DATA_CONFIG']:
        annotation_type = "POLYLINE"
    assert annotation_type

    # Find the max tree level across all shards
    max_tree_level_files = list(glob.glob(f"{data_loc}max_tree_level*.txt"))
    logging.info(f"Max tree level files: {max_tree_level_files}")
    max_tree_level_all_shards = 0
    for max_tree_level_file in max_tree_level_files:
        with open(max_tree_level_file) as f:
            assigned_shards = f.readline().strip()
            max_tree_level_some_shards = int(f.readline().strip())
            logging.info(f"Max tree level some shards: {os.path.basename(max_tree_level_file)}    {assigned_shards}    {max_tree_level_some_shards}")
            if max_tree_level_some_shards > max_tree_level_all_shards:
                max_tree_level_all_shards = max_tree_level_some_shards
    logging.info(f"Max tree level all shards: {max_tree_level_all_shards}")

    # Generate the info file(s)
    generate_info_files(max_tree_level_all_shards, annotation_type)

    # Reorganize the indices
    if config['ID_INDEX_ENABLED']:
        reorganize_id_index()
    if config['RELATION_INDEX_ENABLED']:
        reorganize_relation_index()
    if config['SPATIAL_INDEX_ENABLED']:
        reorganize_spatial_index()
    logging.info("." * 100)

    # Only do this if GCP uploading is enabled since, if it is disabled, the user may explicitly wish to retrieve the results to use them some other nonGCP way.
    if config['UPLOAD_RESULTS_TO_GCP']:
        # Upload the results to the bucket for Neuroglancer to access
        logging.info("\nUploading files to Google Storage")
        st = default_timer()
        upload_files_to_gcp(results_loc, ["info", "pipeline_config.json"], config["TIMESTAMP"], config['GCP_BUCKET'], config['GCP_RESULTS_BLOB_PATH'])#, dryrun=True)
        ets = default_timer() - st
        logging.info(f"GCP upload elapsed time: {seconds_to_hms(ets)}")
    else:
        logging.info("UPLOAD_RESULTS_TO_GCP setting is false. Results won't be uploaded to GCP.")
    
    finalize_results(results_loc)

logging.info("\nDone")
process_running_time()
