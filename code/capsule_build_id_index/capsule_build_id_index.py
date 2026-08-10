import sys
import logging
import os
import glob
import math
import numpy as np
from collections import defaultdict, Counter
from timeit import default_timer
import pandas as pd
import csv
import random
import shutil
import tarfile

from shared.util import *
from shared.ram_data_pond import *
from shared.google_storage import *
from shared.aws_storage import *

import shared.simple_writer_no_spatial_indexing as simple_writer_no_spatial_indexing
import shared.annotations as anno

def get_shard_hex(shard_number: int, shard_bits: int, force_str: bool) -> str:
    """Convert shard number to zero-padded lowercase hex string.

    :param shard_number: The shard number to convert
    :param shard_bits: Number of bits for the shard
    :return: Zero-padded lowercase hex string
    """
    padding = math.ceil(shard_bits / 4)

    # Optionally force the shard hex to a string to prevent leading 0s from being lost during possible int conversions in pandas, parquet, csv, etc.
    # There is code in the 'generate id index shards' capsule, function extract_archives(), that needs to know which of these two methods is used
    # return f"{'"' if force_str else ''}{shard_number:0{padding}x}{'"' if force_str else ''}"
    return f"{'_' if force_str else ''}{shard_number:0{padding}x}"

def determine_row_shard_hex(row, sharding_spec, id_column, split_id_start, force_str: bool):
    if id_column is not None:
        id_ = int(row[id_column])  # The default numpy datatype will cause trouble in the sharding functions, so convert it to an int
    else:
        id_ = split_id_start + row.name
    shard_num = sharding_spec.get_shard_number(id_)
    # minishard_num = sharding_spec.get_minishard_number(id_)
    shard_hex = get_shard_hex(shard_num, sharding_spec.shard_bits, force_str)
    return shard_hex

def process_text_line(line_i, line, header_reverse_map, id_column, split_id_start, sharding_spec, shard_lines, force_str: bool):
    global all_ids, num_dup_ids

    reader = csv.reader(io.StringIO(line))
    fields = next(reader)
    if id_column is not None:
        id_ = int(fields[header_reverse_map[id_column]])
    else:
        id_ = split_id_start + line_i
    shard_num = sharding_spec.get_shard_number(id_)
    # minishard_num = sharding_spec.get_minishard_number(id_)
    shard_hex = get_shard_hex(shard_num, sharding_spec.shard_bits, force_str)
    # logging.info(f"One line id, shard, minishard, shardhex: {id_:>10} {shard_num:>2} {minishard_num:>3} {shard_hex:>2}")

    if id_ in all_ids:
        num_dup_ids += 1
    all_ids.add(id_)

    line = line.strip() + f",{shard_hex}\n"

    if not force_str:
        shard_lines[shard_hex].append(line)
    else:
        # shard_lines[shard_hex[1:-1]].append(line)
        shard_lines[shard_hex[1:]].append(line)

def process_input_file(input_file=None, num_splits=None, split_id=None):
    logging.info(f"\nProcessing an input file: {input_file}\n")
    timestamps.append(("process_input_file() top", default_timer()))

    file_format = None
    if not input_file:
        # No test file was passed in, so search for a pipeline input
        input_files = glob.glob(f"{data_loc}*.csv")
        if len(input_files) > 0:
            file_format = "csv"
        if len(input_files) == 0:
            input_files = glob.glob(f"{data_loc}*.parquet")
            if len(input_files) > 0:
                file_format = "parquet"
        assert len(input_files) == 1
        input_file = input_files[0]



        # input_files = set()  # glob.glob(f"{data_loc}*.{extension}")
        # for extension in ["csv", "parquet"]:
        #     data_sizes = list(config['DATA_CONFIG']['data_sizes'].keys())
        #     data_sizes.remove('docstring')
        #     for data_src in data_sizes:
        #         # I added the materialization version to the config json after creating the test data assets so now the labels don't match the asset names and have to be manually altered a bit (without reuploading the test sets). This only affects testing.
        #         # test_materialization_version = "__v1412"
        #         # data_src_files = list(glob.glob(f"{data_loc}{data_src[:-len(test_materialization_version)]}*.{extension}"))
        #         data_src_files = glob.glob(f"{data_loc}*.{extension}")
        #         assert len(data_src_files) <= 1
        #         if len(data_src_files) == 1:
        #             logging.info(f"Found input file:    {data_src}    {data_src_files[0]}")
        #             assert os.path.isfile(data_src_files[0])
        #             input_files.add(data_src_files[0])
        #     if len(input_files) > 0:
        #         file_format = extension
        #         break



        # We should receive precisely one input
        if len(input_files) != 1:
            raise RuntimeError(f"Expected exactly 1 input file: {input_files}")

        input_file = input_files.pop()  # Safe since we know there is precisely one item in the set
    else:
        if input_file.endswith(".csv"):
            file_format = "csv"
        elif input_file.endswith(".parquet"):
            file_format = "parquet"

    file_size_bytes = os.path.getsize(input_file)
    logging.info(f"file_format: {file_format}")
    logging.info(f"Top-level split input file: {input_file}")
    logging.info(f"Top-level split input file size: {file_size_bytes} bytes")

    input_filename = os.path.basename(input_file)
    if not split_id:
        pcs = input_filename[:input_filename.rindex('.')].split("__")
        logging.info(f"Split input file pcs: {pcs}")
        for pc in pcs:
            if "split-" in pc:
                splitnm = pc.split('-')[1]
                split_id, num_splits = (int(v) for v in splitnm.split('@'))
                break
        logging.info(f"Num splits, Split id: {num_splits}, {split_id}")
    if not split_id:
        raise ValueError("Input split file does not contain a split id")
    logging.info(f"Split id: {split_id}")

    # Debugging: confirm that the header row is or is not present based on the script's circumstances.
    # We don't expect a header from the test file, but do under all other circumstances.
    if file_format == "csv":
        try:
            num_preview_lines = 2
            logging.info(f"\nBeginning of input file (first {num_preview_lines} lines):")
            with open(input_file) as f:
                for i in range(num_preview_lines):
                    logging.info(f"  Line {i+1:>2}: " + f.readline().strip())
        except Exception as e:
            logging.info(e)

    sharding_spec = anno.ShardingSpec(hash=config['ID_SHARDING_HASH'], preshift_bits=config['ID_PRESHIFT_BITS'], shard_bits=config['ID_SHARDING_BITS'], minishard_bits=config['ID_MINISHARDING_BITS'])

    header = None

    if 'id_src' in config['DATA_CONFIG']:
        raise RuntimeError("id_src support (Wan-Qing's swc data) should have hit process_input_dir(), not process_input_file()")
    assert 'id_column' in config['DATA_CONFIG']

    columns = config['DATA_CONFIG']['columns']
    id_column = config['DATA_CONFIG']['id_column']
    # logging.info(f"id_column: {id_column}")
    if id_column is None:
        logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {columns[0]}.")

    split_size = config['DATA_CONFIG']['data_size'][3]
    split_id_start = (split_id - 1) * split_size + 1
    logging.info(f"split_id_start: {split_id_start} (only used if config id_column is null)")

    timestamps.append(("process_input_file() init", default_timer()))

    FILE_PROCESSING_METHOD = "dataframe"  # dataframe or text
    logging.info(f"FILE_PROCESSING_METHOD: {FILE_PROCESSING_METHOD}")
    if FILE_PROCESSING_METHOD == "dataframe":
        if file_format == "csv":
            lines = RAMDataPond.read_nlines_from_disk(input_file, 1)
            # header_present = lines[0].startswith(columns[0 if id_column is not None else 1])
            cols = lines[0].split(',')
            header_present = cols[0] == columns[0] or cols[1] == columns[1]  # Sometimes pandas leaves the first column in the header row
            logging.info(f"header_present: {header_present}")

            if not header_present:
                # We shouldn't need a header in a pipeline because the previous capsule should have added it.
                df = pd.read_csv(input_file, names=config['DATA_CONFIG']['columns'], index_col=False)  # Header is explicitly passed in
            else:
                df = pd.read_csv(input_file, index_col=False)  # Header will be inferred from first line of file
                # logging.info(f"Inferred header: {df.columns}")
        elif file_format == "parquet":
            df = pd.read_parquet(input_file, engine=config['PARQUET_ENGINE'])

        timestamps.append(("process_input_file() read input", default_timer()))

        logging.info(f"\nNum lines (rows) in split: {len(df)}")

        pd.set_option('display.max_columns', None)
        if id_column is None:
            logging.info("Adding ID column since one wasn't configured: {columns[0]}")
            df.insert(0, columns[0], np.arange(split_id_start, split_id_start+len(df)))

        timestamps.append(("process_input_file() insert ID column", default_timer()))

        # Add a new column 'Category' using the apply function
        df['shard_hex'] = df.apply(determine_row_shard_hex, axis=1, args=(sharding_spec, id_column, split_id_start, True,))
        timestamps.append(("process_input_file() add shard_hex column", default_timer()))
        logging.info(f"df['shard_hex'] type: {df.dtypes['shard_hex']}")
        header = df.columns
        shard_hexes = sorted(list(df['shard_hex'].unique()))
        num_shard_hexes = len(df['shard_hex'].unique())
        total_shard_lines = 0
        # for shi, shard_hex in enumerate(shard_hexes):
        for shi, (shard_hex, shard_lines_one_shard_hex) in enumerate(df.groupby('shard_hex')):
            timestamps.append(("process_input_file() shard loop top", default_timer()))
            # shard_hex_no_quotes = shard_hex[1:-1] if shard_hex[0] == '"' else shard_hex  # Strip quotes off
            shard_hex_no_quotes = shard_hex[1:] if shard_hex[0] == '_' else shard_hex  # Strip underscore off
            shard_worker_desc_file_hash, shard_worker_desc = shard_worker_lookup[shard_hex_no_quotes]
            subdir = f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/"
            # timestamps.append(("process_input_file() shard loop point A", default_timer()))
            # shard_lines_one_shard_hex = df[df['shard_hex']==shard_hex]
            # timestamps.append(("process_input_file() shard loop sub-DataFrame extraction", default_timer()))
            total_shard_lines += len(shard_lines_one_shard_hex)
            if shi < 5:
                logging.info(f"  Shard {shard_hex_no_quotes} ({shi+1} of {num_shard_hexes}) num rows (first 5 shown): {len(shard_lines_one_shard_hex)}")
            filename = f"split-{split_id:03}@{num_splits}__shard-{shard_hex_no_quotes}__{input_filename}"
            if filename.endswith(".parquet"):  # If the input was parquet, make sure we output csv at this stage of processing
                filename = filename[:filename.rindex('.')] + ".csv"
            timestamps.append(("process_input_file() shard loop point B", default_timer()))
            shard_lines_one_shard_hex.to_csv(f"{subdir}{filename}", index=False, header=False)
            timestamps.append(("process_input_file() shard loop write CSV", default_timer()))
        timestamps.append(("process_input_file() shard loop done", default_timer()))
        logging.info(f"  Total shard lines accumulated across all shards: {total_shard_lines:,}")
    elif FILE_PROCESSING_METHOD == "text":
        assert file_format == "csv"  # Parquet is not supported for FILE_PROCESSING_METHOD=="text" method yet

        # Even though the file is a CSV, and therefore amenable to Pandas processing, we can process it much more effciently line by line as a text file

        header_reverse_map = {col: i for i, col in enumerate(config['DATA_CONFIG']['columns'])}

        shard_lines = defaultdict(list)
        line_count = 0
        with open(input_file) as f:
            for line_count, line in enumerate(f):
                if not line:
                    break
                if line_count == 0 and line.startswith(columns[0]):
                    # The first line is the header
                    data_header = line.strip().split(',')
                    if data_header != config['DATA_CONFIG']['columns']:
                        raise ValueError("Data CSV header doesn't match header specified in data configuration:\n{data_header}\ndoesn't equal\n{config['DATA_CONFIG']['columns']}")
                    header_reverse_map = {col: i for i, col in enumerate(data_header)}
                else:
                    process_text_line(line_count, line, header_reverse_map, id_column, split_id_start, sharding_spec, shard_lines, True)

            line_count += 1

        timestamps.append(("process_input_file() read input and process lines", default_timer()))

        header = config['DATA_CONFIG']['columns'] + ['shard_hex']

        if id_column is None:
            raise RuntimeError("Support for FILE_PROCESSING_METHOD == 'text' and inferred index not implemented yet")

        shard_hexes = sorted(list(shard_lines.keys()))
        logging.info(f"\nNum lines (rows) in split: {line_count}")

        total_shard_lines = 0
        for shi, (shard_hex, lines) in enumerate(shard_lines.items()):
            if shi < 5:
                logging.info(f"  Shard {shard_hex} ({shi} of {len(shard_hexes)}) num rows (first 5 shown): {len(lines)}")
            total_shard_lines += len(lines)
            shard_worker_desc_file_hash, shard_worker_desc = shard_worker_lookup[shard_hex]
            subdir = f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/"
            # if True:
                # This cannot be combined with a "Collect" connection to the next stage, as it will cause overlapping input errors
            #     subdir += f"shard-{shard_hex}/"
            #     os.makedirs(subdir, exist_ok=True)
            filename = f"split-{split_id:03}@{num_splits}__shard-{shard_hex}__{input_filename}"
            with open(f"{subdir}{filename}", 'w') as f:
                f.writelines(lines)
        logging.info(f"  Total shard lines accumulated across all shards: {total_shard_lines}")

        timestamps.append(("process_input_file() write lines", default_timer()))

    return split_id, num_splits, shard_hexes, header

def process_test_file():
    # Look for a test file that indicates we should process a test scenario
    input_test_filepath = None
    TEST_FILE_FORMAT = None  # None for production run. Else "csv" or "parquet" for test.
    if TEST_FILE_FORMAT:
        for test in range(1, 10):
            input_test_subdir = f"synapses_pni_2_v1_filtered_view__test{test}{'' if TEST_FILE_FORMAT == 'csv' else '__parquet'}/"
            if os.path.exists(f"{data_loc}{input_test_subdir}"):
                logging.info(f"Found test file {input_test_subdir}. Using that instead of production file.")
                input_test_filepath = glob.glob(f"{data_loc}{input_test_subdir}*.{TEST_FILE_FORMAT}")[0]
                input_format = TEST_FILE_FORMAT
                break

    if input_test_filepath:
        logging.info("Found test file. Using that instead of production file.")
        input_file = f"{data_loc}{input_test_filepath}"

        num_splits = 2
        split_id = 2

        if TEST_FILE_FORMAT == "csv":
            df = pd.read_csv(f"{data_loc}{input_test_filepath}", names=config['DATA_CONFIG']['columns'], index_col=False)
        elif TEST_FILE_FORMAT == "parquet":
            df = pd.read_parquet(f"{data_loc}{input_test_filepath}", engine=config['PARQUET_ENGINE'])
        split_size = len(df) // num_splits
        split_start = (split_id - 1) * split_size
        split_end = (split_id * split_size) if split_id < num_splits else len(df)
        split_rows = df.iloc[split_start : split_end]
        split_filename = f"{os.path.basename(input_test_filepath)}__split-{split_id:03}@{num_splits}__rows-{split_start}-{split_end-1}.{TEST_FILE_FORMAT}"
        split_filepath = f"{data_loc}{split_filename}"
        logging.info(f"Split test file for split {split_id} to produce split file: {split_filepath}")
        if TEST_FILE_FORMAT == "csv":
            split_rows.to_csv(split_filepath, index=False, header=False)
        elif TEST_FILE_FORMAT == "parquet":
            split_rows.to_parquet(split_filepath, engine=config['PARQUET_ENGINE'])

        split_id, num_splits, shard_hexes, header = process_input_file(split_filepath, num_splits, split_id)

        return split_id, num_splits, shard_hexes, header

    return None, None, None, None

def process_input_dir(input_dir=None, num_splits=None, split_id=None):
    logging.info(f"\nProcessing an input directory: {input_dir}\n")

    if not input_dir:
        logging.info("No input directory was specified. Proceeding to use config's data_size to indicate filenames to search for.")
        input_dirs = set()
        # data_sizes = list(config['DATA_CONFIG']['data_sizes'].keys())
        # data_sizes.remove('docstring')
        data_sizes = [config['DATA_CONFIG']['data_size'][0]]
        for data_src in data_sizes:
            data_src_files = list(glob.glob(f"{data_loc}{data_src}*"))
            assert len(data_src_files) <= 1
            if len(data_src_files) == 1:
                logging.info(f"Found input file for data_src '{data_src}' at {data_src_files[0]}")
                assert os.path.isdir(data_src_files[0])
                input_dirs.add(data_src_files[0])

        # We should receive precisely one input
        if len(input_dirs) != 1:
            raise RuntimeError(f"Expected exactly 1 input dir: {input_dirs}")

        input_dir = input_dirs.pop()  # Safe since we know there is precisely one item in the set

    logging.info(f"Top-level split input dir: {input_dir}")

    input_dirname = os.path.basename(input_dir)
    if not split_id:
        pcs = input_dirname.split("__")
        logging.info(f"Split input dir pcs: {pcs}")
        for pc in pcs:
            if "split-" in pc:
                splitnm = pc.split('-')[1]
                split_id, num_splits = (int(v) for v in splitnm.split('@'))
                break
        logging.info(f"Num splits, Split id: {num_splits}, {split_id}")
    logging.info(f"Split id: {split_id}")

    sharding_spec = anno.ShardingSpec(hash=config['ID_SHARDING_HASH'], preshift_bits=config['ID_PRESHIFT_BITS'], shard_bits=config['ID_SHARDING_BITS'], minishard_bits=config['ID_MINISHARDING_BITS'])

    extension = "swc"  # For now, just support Wan-Qing's SWC data
    input_files = list(glob.glob(f"{input_dir}/*.{extension}"))
    logging.info(f"Input files:\n  {'\n  '.join(input_files)}")

    # TODO: Generalize this as new data is provided for development
    assert 'id_src' in config['DATA_CONFIG']
    assert config['DATA_CONFIG']['id_src'] == 'file_basename'

    shard_hexes = set()
    for input_file in input_files:
        logging.info(f"\nProcessing input file: {input_file}")
        input_filename = os.path.basename(input_file)
        input_file_id = int(input_filename[:input_filename.rindex('.')])
        shard_num = sharding_spec.get_shard_number(input_file_id)
        shard_hex = get_shard_hex(shard_num, sharding_spec.shard_bits, True)
        shard_hex_no_quotes = shard_hex[1:] if shard_hex[0] == '_' else shard_hex
        shard_worker_desc_file_hash, shard_worker_desc = shard_worker_lookup[shard_hex_no_quotes]
        subdir = f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/"
        input_filename2 = f"split-{split_id:03}@{num_splits}__shard-{shard_hex_no_quotes}__{input_filename}"
        logging.info(f"File id, shard num, shard hex, shard hex w/o leading '_'s, shard worker: {input_file_id} {shard_num} {shard_hex} {shard_hex_no_quotes} {shard_worker_desc}")
        logging.info(f"Moving and renaming file: {input_filename} -> {subdir}{input_filename2}")
        shutil.move(input_file, f"{subdir}{input_filename2}")
        shard_hexes.add(shard_hex)
    logging.info("")

    files = sorted(list(glob.glob(f"{input_dir}/*")))
    logging.info(f"Input files after moving and renaming (should be empty):\n  {'\n  '.join(files)}")
    files = sorted(list(glob.glob(f"{results_loc}shard_worker-*/*")))
    logging.info(f"Shard worker files after moving and renaming:\n  {'\n  '.join(files)}")

    shard_hexes = sorted(list(shard_hexes))

    return split_id, num_splits, shard_hexes, config['DATA_CONFIG']['columns']

def archive_results(split_id, num_splits, header):
    # Code Ocean might shovel data between capsules (or to/from S3) faster if we keep our file-count smaller, even if the total data size isn't particularly large.
    # At first glance, it might not seem that tarring or compressing these outputs should matter. There are only as many outputs as shards, which is currently configured to 16.
    # However, that number becomes multipled by the number of file splits. If there at 10 splits, there are 160 files.
    # Furthermore, the connection to the next capsule is 'Collect', which will distribute copies of those 160 files to all of the next-layer capsules,
    # which themselves number as many shards, 16. So that implies a total of (16*10*16) or 2560 files.
    # This still many not seem prohibitive, but experiments both ways confirm that tarring and/or compressing these 16 outputs into 1 is beneficial overall.
    if config['ARCHIVE_FORMAT']:
        if not config['ARCHIVE_WITH_SHARD_GROUPING']:
            output_files = sorted(list(glob.glob(f"{results_loc}shard_worker-*/split*.csv")))
            if len(output_files) == 0:
                output_files = sorted(list(glob.glob(f"{results_loc}shard_worker-*/split*.parquet")))
            if output_files:
                if config['ARCHIVE_FORMAT'] == "tar":
                    logging.info(f"\nTarring {len(output_files)} output files")
                    ext, mode = (".tar.gz", "w:gz") if config['COMPRESS_ARCHIVE'] else (".tar", "w")
                    with tarfile.open(f"{results_loc}split-{split_id:03}@{num_splits}{ext}", mode) as tar:
                        for output_file in output_files:
                            # logging.info(f"  Adding split shard file to tar: {output_file}")
                            tar.add(output_file, arcname=os.path.basename(output_file))
                elif "parquet" in config['ARCHIVE_FORMAT']:
                    raise NotImplementedError("ARCHIVE_FORMAT 'parquet' without ARCHIVE_WITH_SHARD_GROUPING not yet implemented")
                elif config['ARCHIVE_FORMAT'] == "custom":
                    logging.info(f"\nArchiving {len(output_files)} output files")

                    if False:  # Archive to a text file
                        with open(f"{results_loc}split-{split_id:03}@{num_splits}__archive.txt", 'w') as fin:
                            for output_file in output_files:
                                assert output_file.endswith(".csv")
                                with open(output_file, 'r') as fin:
                                    file_contents = fin.read()
                                RAMDataPond.archive_str_data(output_file, file_contents, results_loc, fin)
                    else:  # Archive to an indexed binary file (only binary files facilitate seeking for "range-read"-like behavior)
                        file_index = []
                        with open(f"{results_loc}split-{split_id:03}@{num_splits}__archive.bin", 'wb') as fout:
                            for output_file in output_files:
                                with open(output_file, 'rb') as fin:
                                    file_contents_bytes = fin.read()
                                RAMDataPond.archive_strbin_data(output_file, file_contents_bytes, results_loc, fout, file_index)
                        offset = 0
                        for i in range(len(file_index)):
                            file_index[i][1] = offset
                            offset += file_index[i][2]
                        with open(f"{results_loc}split-{split_id:03}@{num_splits}__archive_idx.txt", 'w') as fout:
                            # for filepath, filebytes_len, file_offset in file_index:
                            #     fout.write(f"{filepath}\t{filebytes_len}\t{file_offset}\n")
                            fout.write(str(file_index))
                else:
                    raise ValueError(f"Illegal ARCHIVE_FORMAT: {config['ARCHIVE_FORMAT']}")

                for output_file in output_files:
                    os.remove(output_file)
        else:  # ARCHIVE_WITH_SHARD_GROUPING
            if config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row":
                output_extension = "csv"
            if config['DATA_CONFIG']['structure'] == "one_annotation_per_file__one_point_per_row":
                logging.info("Overriding archive format for structure one_annotation_per_file__one_point_per_row. Setting archive format to 'tar'")
                config['ARCHIVE_FORMAT'] = "tar"
                output_extension = "swc"  # For now, just support Wan-Qing's SWC data

            for shard_worker_desc_i, (shard_worker_desc_file_hash, shard_worker_desc) in enumerate(shard_worker_descs):
                shard_worker_desc_str = '_'.join(shard_worker_desc)  # Unused now
                logging.info(f"Archiving shard group: {shard_worker_desc_str}")

                if config['ARCHIVE_FORMAT'] == "tar":
                    ext, mode = (".tar.gz", "w:gz") if config['COMPRESS_ARCHIVE'] else (".tar", "w")
                    with tarfile.open(f"{results_loc}split-{split_id:03}@{num_splits}__shard_worker-{shard_worker_desc_file_hash}{ext}", mode) as tar:
                        for shard_hex in shard_worker_desc:
                            output_files = list(glob.glob(f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/split*shard-{shard_hex}*.{output_extension}"))
                            if len(output_files) == 0:
                                output_files = sorted(list(glob.glob(f"{results_loc}shard_worker-*/split*.parquet")))
                            logging.info(f"Files to be tarred: {output_files}")
                            if output_files:
                                logging.info(f"Tarring {len(output_files)} output files for shard {shard_hex}")

                                for output_file in output_files:
                                    logging.info(f"  Adding split shard file to tar: {output_file}")
                                    tar.add(output_file, arcname=os.path.basename(output_file))

                                for output_file in output_files:
                                    os.remove(output_file)
                elif "parquet" in config['ARCHIVE_FORMAT']:
                    merged_df = None
                    for shard_hex_i, shard_hex in enumerate(shard_worker_desc):
                        output_files = list(glob.glob(f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/split*shard-{shard_hex}*.csv"))
                        if len(output_files) == 0:
                            output_files = sorted(list(glob.glob(f"{results_loc}shard_worker-*/split*.parquet")))
                        if output_files:
                            if shard_worker_desc_i < 3 and shard_hex_i < 3:
                                logging.info(f"Output files (first 3 shown):\n  {'\n  '.join(output_files[:3])}")

                            # Produce one parquet file per shard (This is inefficient. It is better to group all shards per shard worker.)
                            # for output_file in output_files:
                                # if output_file.endswith(".csv"):
                                #     output_parquet_filename = os.path.basename(output_file).replace(".csv", ".parquet")
                                #     df = pd.read_csv(output_file)
                                #     df.to_parquet(f"{results_loc}{output_parquet_filename}", engine=config['PARQUET_ENGINE'])

                            # Produce one parquet file per shard worker (This is better. It groups all shards per shard worker into a single file.)
                            for output_file in output_files:
                                if output_file.endswith(".csv"):
                                    df = pd.read_csv(output_file, names=header, index_col=False)
                                elif output_file.endswith(".parquet"):
                                    df = pd.read_parquet(output_file, engine=config['PARQUET_ENGINE'])
                                if shard_worker_desc_i < 3 and shard_hex_i < 3:
                                    logging.info(f"df['shard_hex'] type: {df.dtypes['shard_hex']}")
                                if merged_df is None:
                                    logging.info(f"Initializing merged shards file from first shard: {shard_hex}")
                                    merged_df = df
                                else:
                                    if shard_worker_desc_i < 3 and shard_hex_i < 3:
                                        logging.info(f"Merging in another shard: {shard_hex}")
                                    merged_df = pd.concat([merged_df, df])

                            for output_file in output_files:
                                os.remove(output_file)

                    # SUCCESS

                    if merged_df is not None:
                        # logging.info(f"merged_df['shard_hex'] type: {merged_df.dtypes['shard_hex']}")
                        merged_df['shard_hex'] = merged_df['shard_hex'].astype(str)
                        # logging.info(f"merged_df['shard_hex'] type: {merged_df.dtypes['shard_hex']}")

                        # SUCCESS

                        merged_df.to_parquet(f"{results_loc}split-{split_id:03}@{num_splits}__shard_worker-{shard_worker_desc_file_hash}.parquet", engine=config['PARQUET_ENGINE'])

                        # FAIL AND the aaa.txt file doesn't appear in the results/ so to_parquet() is hanging
                elif config['ARCHIVE_FORMAT'] == "custom":
                    with open(f"{results_loc}split-{split_id:03}@{num_splits}__shard_worker-{shard_worker_desc_file_hash}__archive.txt", 'w') as fout:
                        for shard_hex in shard_worker_desc:
                            output_files = list(glob.glob(f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/split*shard-{shard_hex}*.csv"))
                            if len(output_files) == 0:
                                output_files = sorted(list(glob.glob(f"{results_loc}shard_worker-*/split*.parquet")))
                            if output_files:
                                # logging.info(f"Archiving {len(output_files)} output files for shard {shard_hex}")

                                for output_file in output_files:
                                    assert output_file.endswith(".csv")
                                    with open(output_file, 'r') as fin:
                                        file_contents = fin.read()
                                    RAMDataPond.archive_str_data(output_file, file_contents, f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/", fout)

                                for output_file in output_files:
                                    os.remove(output_file)
                else:
                    raise ValueError(f"Illegal ARCHIVE_FORMAT: {config['ARCHIVE_FORMAT']}")

    # FAIL

    # We have to delete the empty directory so the results directory will look empty to force the placeholder file to be added
    shard_worker_dirs = sorted(list(glob.glob(f"{results_loc}shard_worker-*")))
    for shard_worker_dir in shard_worker_dirs:
        # logging.info(f"Deleting shard worker directory after archiving contents {shard_worker_dir}")
        if os.listdir(shard_worker_dir) != []:
            logging.info(f"ERROR! Shard worker directory {shard_worker_dir} contents (should be empty): {os.listdir(shard_worker_dir)}")
        os.rmdir(shard_worker_dir)

def upload_results_to_bucket():
    if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] != "internal":
        st = default_timer()
        files_to_upload_to_scratch = sorted(list(glob.glob(f"{results_loc}split*")))
        logging.info(f"files_to_upload_to_scratch (first 30 shown):\n  {'\n  '.join(files_to_upload_to_scratch[:30])}")
        if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "gcp":
            raise RuntimeError("GCP bucket no longer supported due to possible financial cost if done incorrectly!")
            logging.info("\nUploading files to Google storage")
            filenames_to_upload_to_scratch = [os.path.basename(filepath) for filepath in files_to_upload_to_scratch]
            upload_files_to_gcp(results_loc, filenames_to_upload_to_scratch, f"{config['TIMESTAMP']}/id_index", config['GCP_BUCKET'], config['GCP_SCRATCH_BLOB_PATH'])#, dryrun=True)
        elif config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "aws":
            logging.info("\nUploading files to AWS storage")
            upload_folder_relative_path = f"aws_upload/{config['TIMESTAMP']}/id_index/"
            os.makedirs(f"{results_loc}{upload_folder_relative_path}", exist_ok=True)
            for file in files_to_upload_to_scratch:
                # logging.info(f"Move {file} -> {results_loc}{upload_folder_relative_path}{os.path.basename(file)}")
                shutil.move(file, f"{results_loc}{upload_folder_relative_path}{os.path.basename(file)}")
            upload_folder_to_aws(f"{results_loc}aws_upload/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])#, dryrun=True)
            query_folder_on_aws(f"{config['TIMESTAMP']}/id_index/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])
            for file in files_to_upload_to_scratch:
                # Move the files back so we can delete them from the original location below
                # logging.info(f"Move {results_loc}{upload_folder_relative_path}{os.path.basename(file)} -> {results_loc}{file}")
                shutil.move(f"{results_loc}{upload_folder_relative_path}{os.path.basename(file)}", f"{results_loc}{file}")
            shutil.rmtree(f"{results_loc}aws_upload/")

        t1 = default_timer()
        logging.info(f"External bucket upload elapsed time: {seconds_to_hms(t1 - st)}")
        timestamps.append(("upload_files_to_external_storage", t1))

        logging.info("")
        for file in files_to_upload_to_scratch:
            # logging.info(f"Deleting result file after uploading to external bucket: {file}")
            os.remove(file)

        timestamps.append(("delete_output_files", default_timer()))
    else:
        logging.info(f"\n{data_loc}PASS_DATA_BETWEEN_CAPSULES_METHOD indicates Code Ocean. Results won't be uploaded externally.")

def test_aws_interactions():
    # Make sure this subpipeline's config is loaded last so it can override any other config values
    config = read_config(["relation", "spatial", "id"])
    logging.basicConfig(stream=sys.stdout, level=get_logging_level_from_desc(config['LOGGING_LEVEL']), format=config['LOGGING_FORMAT'], force=True)

    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('boto3').setLevel(logging.INFO)
    logging.getLogger('botocore').setLevel(logging.INFO)
    logging.getLogger('aws-cli').setLevel(logging.INFO)
    logging.getLogger('cloudfiles').setLevel(logging.INFO)

    logging.info("aws_interactions()")

    query_folder_on_aws("subfolder/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])

    logging.info("\n\n" + "_" * 100 + "\nUPLOADING")
    upload_folder_to_aws("../data/test_folder/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])
    query_folder_on_aws("subfolder/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])

    logging.info("\n\n" + "_" * 100 + "\nDOWLOADING")
    download_folder_from_aws("subfolder/", "../data/aws_files/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])

    logging.info("\n\n" + "_" * 100 + "\nDELETING")
    delete_folder_from_aws("subfolder/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])
    # delete_folder_from_aws("test_folder/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])
    query_folder_on_aws("subfolder/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])

    logging.info("\n\n" + "_" * 100 + "\nRESULTS")
    logging.info(f"\n../data/ contents: {'\n'.join(os.listdir('../data/'))}")
    logging.info(f"\n../data/aws_files contents: {'\n'.join(os.listdir('../data/aws_files'))}")
    shutil.copytree("../data/aws_files", "../results/aws_files")

def test_aws_interactions2():
    # Make sure this subpipeline's config is loaded last so it can override any other config values
    config = read_config(["relation", "spatial", "id"])
    logging.basicConfig(stream=sys.stdout, level=get_logging_level_from_desc(config['LOGGING_LEVEL']), format=config['LOGGING_FORMAT'], force=True)

    delete_folder_from_aws2("subfolder/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])
    delete_folder_from_aws2("emptyFolder/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])
    delete_folder_from_aws2("subfolder/test_file2.txt", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])

    query_folder_on_aws("subfolder/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])

if __name__ == "__main__":
    # test_aws_interactions2()
    # logging.info("\nDone")
    # sys.exit(0)

    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("BUILD ID INDEX")

    analyze_memory_usage()

    data_loc = "../data/"
    results_loc = "../results/"

    # Make sure this subpipeline's config is loaded last so it can override any other config values
    config = read_config(["relation", "spatial", "id"])
    logging.basicConfig(stream=sys.stdout, level=get_logging_level_from_desc(config['LOGGING_LEVEL']), format=config['LOGGING_FORMAT'], force=True)

    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('boto3').setLevel(logging.INFO)
    logging.getLogger('botocore').setLevel(logging.INFO)
    logging.getLogger('s3transfer').setLevel(logging.INFO)
    logging.getLogger('aws-cli').setLevel(logging.INFO)
    logging.getLogger('cloudfiles').setLevel(logging.INFO)

    logging.getLogger('simple_writer_no_spatial_indexing').setLevel(
        get_logging_level_from_desc(config['PRECOMPUTED_FILE_WRITER_LOGGING_LEVEL']))
    logging.getLogger('sharding').setLevel(
        get_logging_level_from_desc(config['PRECOMPUTED_FILE_WRITER_LOGGING_LEVEL']))
    logging.getLogger('annotations').setLevel(
        get_logging_level_from_desc(config['PRECOMPUTED_FILE_WRITER_LOGGING_LEVEL']))

    all_ids, num_dup_ids = set(), 0

    if config['ID_INDEX_ENABLED']:
        timestamps = []
        timestamps.append(("start", default_timer()))

        data_loc_contents = sorted(os.listdir(data_loc))
        data_loc_contents = [v for v in data_loc_contents if "placeholder" not in v]
        logging.info(f"{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):")
        logging.info('  ' + '\n  '.join(data_loc_contents[:30]).strip() + '\n')

        shard_worker_descs = set()
        shard_worker_lookup = {}
        shard_worker_desc_files = list(glob.glob(f"{data_loc}shard_worker*txt"))
        for shard_worker_desc_file in shard_worker_desc_files:
            shard_worker_desc_filename = os.path.basename(shard_worker_desc_file)
            shard_worker_desc_file_hash = shard_worker_desc_filename[:shard_worker_desc_filename.rindex('.')].split('_')[-1]
            with open(shard_worker_desc_file) as f:
                shard_worker_desc = f.read()
            os.makedirs(f"{results_loc}shard_worker-{shard_worker_desc_file_hash}", exist_ok=True)
            assigned_shards = tuple(shard_worker_desc.split('_'))
            for shard in assigned_shards:
                shard_worker_lookup[shard] = (shard_worker_desc_file_hash, shard_worker_desc)
            logging.info(f"Shard worker {shard_worker_desc_file_hash} assigned shards: {assigned_shards}")
            shard_worker_descs.add((shard_worker_desc_file_hash, assigned_shards))
        logging.info("\n")

        timestamps.append(("read shard worker descriptions", default_timer()))

        split_id, num_splits, header = None, None, None

        # Detect an empty input directory or the presence of the no-op file, indicating that this capsule isn't being used in the current pipeline.
        data_loc_contents_set = set(data_loc_contents)
        if data_loc_contents_set == set(["job_config.py"]) \
            or data_loc_contents_set == set(["job_config.py", "no_op.txt"]):
            logging.info("Empty data input directory or no-op file found. Presumably, this capsule isn't being used in the current pipeline.")
            with open(f"{results_loc}no_op.txt", 'w') as f:
                f.write("no_op\n""Empty input directory or no-op file found. Presumably, this capsule isn't being used in the current pipeline.")
        else:
            split_id, num_splits, shard_hexes, header = process_test_file()
            if not shard_hexes:
                if config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row":
                    split_id, num_splits, shard_hexes, header = process_input_file()
                elif config['DATA_CONFIG']['structure'] == "one_annotation_per_file__one_point_per_row":
                    raise RuntimeError(f"Structure {config['DATA_CONFIG']['structure']} should have been converted in an earlier capsule.")
                    split_id, num_splits, shard_hexes, header = process_input_dir()
                elif config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row_in_one_field":
                    split_id, num_splits, shard_hexes, header = process_input_file()
            # shard_hexes = [shard_hex[1:-1] if shard_hex[0] == '"' else shard_hex for shard_hex in shard_hexes]
            shard_hexes = [shard_hex[1:] if shard_hex[0] == '_' else shard_hex for shard_hex in shard_hexes]
            # logging.info(f"Input all shard hexes: {shard_hexes}")

        logging.info(f"\nNum duplicate ids: {num_dup_ids}")

        analyze_memory_usage()
        timestamps.append(("process input file", default_timer()))

        # COMMENTING OUT archive_results() SUCCESS
        archive_results(split_id, num_splits, header)
        analyze_memory_usage()
        timestamps.append(("archive results", default_timer()))

        if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
            upload_results_to_bucket()
        else:
            logging.info(f"\n{data_loc}DEBUG_FLAG.txt file found. Results won't be uploaded externally.")

        if split_id == 1:  # To avoid CodeOcean name collisions, only do this from one capsule
            logging.info("Copying config files to results for next capsule")
            for f in glob.glob(f"{data_loc}*config*.py"):
                shutil.copy(f, f"{results_loc}{os.path.basename(f)}")

        logging.error("\nElapsed timestamps:")
        accum_elapsed_times = Counter()
        for ti, time in enumerate(timestamps):
            if ti > 0:
                elap_t = time[1] - timestamps[ti-1][1]
                accum_elapsed_times[time[0]] += elap_t
                # logging.error(f"  {seconds_to_hms(elap_t)} {time[0]}")

        logging.error("Accumulated elapsed timestamps:")
        for label, elap_t in accum_elapsed_times.items():
            logging.error(f"  {seconds_to_hms(elap_t)} {label}")

    finalize_results(results_loc)

    analyze_memory_usage()

logging.info("\nDone")
process_running_time()
