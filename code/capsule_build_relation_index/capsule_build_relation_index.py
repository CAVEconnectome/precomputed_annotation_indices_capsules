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
import re
import string
import random
import shutil
import tarfile
import ast

from shared.util import *
from shared.ram_data_pond import *
from shared.google_storage import *
from shared.aws_storage import *

import shared.annotations as anno

def convert_relation_key(relationship):
    # The following conversions are copied from Joe Strout's code
    # Convert to lowercase
    relation_key = relationship.lower()
    # Remove all punctuation
    relation_key = relation_key.translate(str.maketrans("", "", string.punctuation))
    # Replace spaces (and any other whitespace) with underscores
    relation_key = re.sub(r"\s+", "_", relation_key)
    return relation_key

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

def convert_non_int_relation_via_enum_property(row_idx, relation_val, relationship_column_name):
    if row_idx == 0:
        logging.info(f"Relation column {relationship} doesn't contain 'int' data. Looking in the data config for a corresponding enumerated property description.")
    found_it = False
    for prop_lbl, prop_info in config['DATA_CONFIG']['properties'].items():
        if prop_info['id'] == relationship_column_name:
            if prop_info['enum_values'] is not None:
                if row_idx == 0:
                    logging.info(f"Found an enum property config")
                if relation_val in prop_info['enum_labels']:
                    if row_idx == 0:
                        logging.info(f"Found the enum value")
                    enum_label_idx = prop_info['enum_labels'].index(relation_val)
                    enum_value = prop_info['enum_values'][enum_label_idx]
                else:
                    if row_idx == 0:
                        logging.info(f"Found a missing enum value: {relation_val} {prop_info}")
                    missing_enum_labels.add(relation_val)
                    enum_value = -1
                relation_val = enum_value
                found_it = True
    if not found_it:
        raise TypeError(f"Relation column '{relationship}' does not contain 'int' data and has no associated enumerated property description from which to derive an 'int' value.")
    return relation_val

def coerce_relation_field(relation_field):
    if isinstance(relation_field, int):
        relation_list = [relation_field]
    elif isinstance(relation_field, str):
        # It might a str of an int, e.g.,: 864691135568681196 but the column might encode it as '864691135568681196'.
        # Or it might not be an int, e.g.,: Jacob Quon's gene_names merscope data.
        # If it's an int, we need to convert to a list of int.
        # If it's a str, we need to convert to a list of str.
        try:
            literal_val = ast.literal_eval(relation_field)
            if isinstance(literal_val, int):
                relation_list = [literal_val]
            elif isinstance(literal_val, tuple):
                relation_list = list(literal_val)
            elif isinstance(literal_val, list):
                relation_list = literal_val
        except:
            # At this point, we will assume the relation data is some sort of enum label, like Jacob Quon's gene_names merscope data, so just wrap it in brackets.
            relation_list = [relation_field]
    else:
        raise ValueError(f"{relation_field}: {type(relation_field)}")
    assert isinstance(relation_list, list)
    return relation_list

def process_row(row_idx, fields, header_reverse_map, sharding_spec, shard_lines, force_str: bool):
    # logging.info(f"process_row() id: {row_idx}")

    for relationship, relationship_info in config['DATA_CONFIG']['relations'].items():
        relationship_column_name = relationship_info['id']
        relation_field = fields[header_reverse_map[relationship_column_name]]
        if row_idx < 3:  # 10 or (row_idx % 10 == 0 and row_idx < 100):
            logging.info(f"Row {row_idx:10} relation_field:   {relationship_column_name:15}  =>  {relation_field}")

            logging.info(f"Relation field and type: {relation_field} {type(relation_field)}")
        
        relation_list = coerce_relation_field(relation_field)

        for relation_val in relation_list:
            # This is a bit of a hack.
            # Originally, relations were only supported as ints, as per the spec.
            # However, we need to support scenarios in which a column is used as both a relation and an enumerated property.
            # So, if the relation value isn't an int, before we bail with an error, let's check if there is enum property info we can apply.
            if not isinstance(relation_val, int):
                relation_val = convert_non_int_relation_via_enum_property(row_idx, relation_val, relationship_column_name)

            shard_num = sharding_spec.get_shard_number(relation_val)
            # minishard_num = sharding_spec.get_minishard_number(relation_val)
            shard_hex = get_shard_hex(shard_num, sharding_spec.shard_bits, force_str)
            
            fields_w_shard_hex = fields + [shard_hex]
            for i, field in enumerate(fields_w_shard_hex):
                if ',' in field:
                    fields_w_shard_hex[i] = f'"{field}"'
            if row_idx < 3:  # < 10 or (row_idx % 10 == 0 and row_idx < 100):
                logging.info(f"{relation_val} fields_w_shard_hex: {fields_w_shard_hex}")
            relation_line = ','.join(fields_w_shard_hex) + "\n"

            if not force_str:
                shard_lines[relationship][shard_hex].append(relation_line)
            else:
                # shard_lines[relationship][shard_hex[1:-1]].append(relation_line)
                shard_lines[relationship][shard_hex[1:]].append(relation_line)

def process_df_row(row_idx, df_row, header_reverse_map, sharding_spec, shard_lines, force_str: bool):
    fields = [str(v) for v in list(df_row)]
    if row_idx < 3:  # < 10 or (row_idx % 10 == 0 and row_idx < 100):
        logging.info(f"Row ({type(df_row)}): {df_row}")
        logging.info(f"Row list: {list(df_row)}")
        logging.info(f"Fields: {fields}")
    process_row(row_idx, fields, header_reverse_map, sharding_spec, shard_lines, force_str)
    if row_idx < 3:  # < 10 or (row_idx % 10 == 0 and row_idx < 100):
        logging.info("")

def process_text_line(line_idx, line, header_reverse_map, sharding_spec, shard_lines, force_str: bool):
    reader = csv.reader(io.StringIO(line))
    fields = next(reader)
    process_row(line_idx, fields, header_reverse_map, sharding_spec, shard_lines, force_str)

def process_input_file(input_file=None, num_splits=None, split_id=None):
    logging.info("\n")
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
    
    sharding_spec = anno.ShardingSpec(hash=config['RELATION_SHARDING_HASH'], preshift_bits=config['RELATION_PRESHIFT_BITS'], shard_bits=config['RELATION_SHARDING_BITS'], minishard_bits=config['RELATION_MINISHARDING_BITS'])
    
    header = None

    if 'id_src' in config['DATA_CONFIG']:
        id_src = config['DATA_CONFIG']['id_src']
        logging.info(f"id_src: {id_src}")
        raise RuntimeError("id_src support (Wan-Qing's swc data) is not implemented yet")
    assert 'id_column' in config['DATA_CONFIG']
    
    columns = config['DATA_CONFIG']['columns']
    id_column = config['DATA_CONFIG']['id_column']
    # logging.info(f"id_column: {id_column}")
    if id_column is None:
        logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {columns[0]}.")
    
    split_size = config['DATA_CONFIG']['data_size'][3]
    split_id_start = (split_id - 1) * split_size + 1
    logging.info(f"split_id_start: {split_id_start} (only used if config id_column is null)")

    header_reverse_map = {col: i for i, col in enumerate(config['DATA_CONFIG']['columns'])}
    shard_lines = defaultdict(lambda: defaultdict(list))  # Dict of relation names to dict of shards to list of lines (rows)
    
    line_count = 0
    
    timestamps.append(("process_input_file() init", default_timer()))

    FILE_PROCESSING_METHOD = "dataframe"  # dataframe or text
    logging.info(f"FILE_PROCESSING_METHOD: {FILE_PROCESSING_METHOD}")
    if FILE_PROCESSING_METHOD == "dataframe":
        if file_format == "csv":
            lines = RAMDataPond.read_nlines_from_disk(input_file, 1)
            header_present = lines[0].startswith(columns[0 if id_column is not None else 1])
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
        # logging.info(f"AAA\n{df}")
        if id_column is None:
            logging.info("Adding ID column since one wasn't configured: {columns[0]}")
            df.insert(0, columns[0], np.arange(split_id_start, split_id_start+len(df)))
        # logging.info(f"BBB\n{df}")
        # logging.info(f"CCC\n{df[columns[0]]}")

        timestamps.append(("process_input_file() insert ID column", default_timer()))

        # Debug
        pd.set_option('display.max_columns', None)
        logging.info(f"Input DataFrame:\n{df}")

        # Add a new column 'Category' using the apply function
        for row_idx, row in enumerate(df.itertuples(index=False)):
            process_df_row(row_idx, row, header_reverse_map, sharding_spec, shard_lines, True)
        line_count += len(df)

        timestamps.append(("process_input_file() process rows", default_timer()))
    elif FILE_PROCESSING_METHOD == "text":
        assert file_format == "csv"  # Parquet is not supported for FILE_PROCESSING_METHOD=="text" method yet

        if id_column is None:
            raise RuntimeError("Support for FILE_PROCESSING_METHOD == 'text' and inferred index not implemented yet")

        # Even though the file is a CSV, and therefore amenable to Pandas processing, we can process it much more effciently line by line as a text file
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
                    process_text_line(line_count, line, header_reverse_map, sharding_spec, shard_lines, True)
    
            line_count += 1
        
    header = config['DATA_CONFIG']['columns'] + ['shard_hex']

    logging.info(f"\nNum lines (rows) in split: {line_count}")

    logging.info(f"shard_lines keys: {shard_lines.keys()}")
    
    shard_hexes = set()
    for relationship in shard_lines:
        relationship_key = relationship_keys[relationship]
        logging.info(f"\nGenerating output for relation    {relationship}    {relationship_key}")

        logging.info(f"shard_lines[{relationship}] keys: {shard_lines[relationship].keys()}")
        
        total_shard_lines = 0
        for shi, (shard_hex, lines) in enumerate(shard_lines[relationship].items()):
            # Strip off shard_hex prefix that forced to string to preserve leading 0s
            if shard_hex[0] == '"':
                shard_hex = shard_hex[1:-1]
            elif shard_hex[0] == '_':
                shard_hex = shard_hex[1:]
            
            shard_hexes.add(shard_hex)
            shard_worker_desc_file_hash, shard_worker_desc = shard_worker_lookup[shard_hex]

            subdir = f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/split-{split_id:03}@{num_splits}__{relationship_key}/"
            os.makedirs(subdir, exist_ok=True)
            
            total_shard_lines += len(lines)
            if shi < 5:
                logging.info(f"  Shard {shard_hex} ({shi+1} of {len(shard_lines[relationship].keys())}) num rows (first 5 shown): {len(lines):>11,}")
            
            # dir_ = f"{results_loc}"#worker_{shard_hex}/"
            # os.makedirs(dir_, exist_ok=True)
            file_subdir_path = f"{subdir}{relationship_key}__shard-{shard_hex}__{input_filename}"
            # logging.info(f"  Writing to file_subdir_path: {file_subdir_path}")
            #     # for line in lines:
            #     #     if line.strip().split(',')[-1] == "864691135885111664":
            #     #         logging.info(f"  {shard_hex} 864691135885111664")
            assert not os.path.exists(f"{results_loc}{file_subdir_path}")
            with open(f"{results_loc}{file_subdir_path}", 'w') as f:
                f.writelines(lines)
        logging.info(f"  Total shard lines accumulated across all shards: {total_shard_lines:,}")
    logging.info("")
    
    return split_id, num_splits, sorted(list(shard_hexes)), header

def process_test_file():
    # Look for a test file that indicates we should process a test scenario
    input_test_filepath = None
    TEST_FILE_FORMAT = "csv"  # None for production run. Else "csv" or "parquet" for test.
    if TEST_FILE_FORMAT:
        for test in range(1, 10):
            input_test_subdir = f"synapses_pni_2_v1_filtered_view__test{test}{'' if TEST_FILE_FORMAT == 'csv' else '__parquet'}/"
            if os.path.exists(f"{data_loc}{input_test_subdir}"):
                logging.info(f"Found test file {input_test_subdir}. Using that instead of production file.")
                input_test_filepath = glob.glob(f"{data_loc}{input_test_subdir}*.{TEST_FILE_FORMAT}")[0]
                input_format = TEST_FILE_FORMAT
                break
            if test == 1:
                input_test_subdir = f"synapses_pni_2_v1_filtered_view__v1412__test1_w_fake_enums_and_multirelations{'' if TEST_FILE_FORMAT == 'csv' else '__parquet'}/"
                if os.path.exists(f"{data_loc}{input_test_subdir}"):
                    logging.info(f"Found test file {input_test_subdir}. Using that instead of production file.")
                    input_test_filepath = glob.glob(f"{data_loc}{input_test_subdir}*.{TEST_FILE_FORMAT}")[0]
                    input_format = TEST_FILE_FORMAT
                    break

    if input_test_filepath:
        logging.info("Found test file. Using that instead of production file.")
        input_file = f"{data_loc}{input_test_filepath}"

        num_splits = 2
        split_id = 1

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
        data_sizes = list(config['DATA_CONFIG']['data_sizes'].keys())
        data_sizes.remove('docstring')
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

    sharding_spec = anno.ShardingSpec(hash=config['RELATION_SHARDING_HASH'], preshift_bits=config['RELATION_PRESHIFT_BITS'], shard_bits=config['RELATION_SHARDING_BITS'], minishard_bits=config['RELATION_MINISHARDING_BITS'])

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

        # This relationship-looping code differs from the analogous code in the ID pipeline that this code was extended from.
        # The only polyline data I have at the moment lacks relations (Wan-Qing's axon data) so I can't really test polyline relation support yet.
        # This code is not at all vetted and its development should be considered incomplete for now.
        for rki, relationship_key in enumerate(relationship_keys):
            subdir = f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/split-{split_id:03}@{num_splits}__{relationship_key}/"
            os.makedirs(subdir, exist_ok=True)

            input_filename2 = f"split-{split_id:03}@{num_splits}__shard-{shard_hex_no_quotes}__{input_filename}"
            logging.info(f"File id, shard num, shard hex, shard hex w/o leading '_'s, shard worker: {input_file_id} {shard_num} {shard_hex} {shard_hex_no_quotes} {shard_worker_desc}")
            logging.info(f"Moving and renaming file: {input_filename} -> {subdir}{input_filename2}")
            
            # Copy the file for all but the last relation. The move it for the last one so it doesn't remain in the original location.
            if rki < len(relationship_keys) - 1:
                shutil.copy(input_file, f"{subdir}{input_filename2}")
            else:
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
    # See note in the 'Build ID Index' capsule about this option.
    # The advantage/disadvantage in the case of the relation index appears to be a bit of a wash.
    # It doesn't make too much difference one way or the other.
    if config['ARCHIVE_FORMAT']:
        if not config['ARCHIVE_WITH_SHARD_GROUPING']:
            if config['ARCHIVE_FORMAT'] == "tar":
                output_dirs = sorted(list(glob.glob(f"{results_loc}shard_worker-*/split-*__*")))
                if output_dirs:
                    logging.info(f"\nTarring {len(output_dirs)} output files")
                    ext, mode = (".tar.gz", "w:gz") if config['COMPRESS_ARCHIVE'] else (".tar", "w")
                    with tarfile.open(f"{results_loc}split-{split_id:03}@{num_splits}{ext}", mode) as tar:
                        for output_dir in output_dirs:
                            # logging.info(f"  Adding split shard file to tar: {output_file}")
                            tar.add(output_dir, arcname=os.path.basename(output_file))
            elif "parquet" in config['ARCHIVE_FORMAT']:
                    raise NotImplementedError("ARCHIVE_FORMAT 'parquet' without ARCHIVE_WITH_SHARD_GROUPING not yet implemented")
            elif config['ARCHIVE_FORMAT'] == "custom":
                output_files = sorted(list(glob.glob(f"{results_loc}shard_worker-*/split-*__*/*")))
                if output_files:
                    # logging.info(f"output_files\n{'\n'.join(output_files)}")
                    logging.info(f"\nArchiving {len(output_files)} output files")
                    if False:  # Archive to a text file
                        with open(f"{results_loc}split-{split_id:03}@{num_splits}__archive.txt", 'w') as fout:
                            for output_file in output_files:
                                with open(output_file, 'r') as fin:
                                    file_contents = fin.read()
                                RAMDataPond.archive_str_data(output_file, file_contents, results_loc, fout)
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
                        for shard_hex in shard_hexes:
                            # output_dir = f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/"
                            # logging.info(f"\nTarring {output_dir}")
                            # # logging.info(f"  Adding split shard dir to tar: {output_dir}")
                            # tar.add(output_dir, arcname=os.path.basename(output_dir))
                            # shutil.rmtree(output_dir)

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
                    for relationship in config['DATA_CONFIG']['relations']:
                        relation_key = convert_relation_key(relationship)
                        merged_df = None
                        for shi, shard_hex in enumerate(shard_worker_desc):
                            output_files = list(glob.glob(f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/split-*__*/{relation_key}*__shard-{shard_hex}__*"))
                            if output_files:
                                # Produce one parquet file per shard (This is inefficient. It is better to group all shards per shard worker.)
                                # for output_file in output_files:
                                #     output_parquet_filename = os.path.basename(output_file).replace(".csv", ".parquet").replace("__shard-", f"__split-{split_id:03}@{num_splits}__shard-")
                                #     df = pd.read_csv(output_file, index_col=False)
                                #     df.to_parquet(f"{results_loc}{output_parquet_filename}", engine=config['PARQUET_ENGINE'])
                                
                                # Produce one parquet file per shard worker (This is better. It groups all shards per shard worker into a single file.)
                                for output_file in output_files:
                                    with open(output_file) as f:
                                        if shi <= 1:
                                            logging.info(f"Single archive read back one line:\n{f.readline().strip()}")
                                    df = pd.read_csv(output_file, names=header, index_col=False)

                                    # Debug, check index and column alignment
                                    pd.set_option('display.max_columns', None)
                                    # logging.info(f"Single archive read back:\n{df}")
                                    
                                    if merged_df is None:
                                        if shi <= 1:
                                            logging.info(f"Initializing merged shards file from first shard: {shard_hex}")
                                        merged_df = df
                                    else:
                                        if shi <= 1:
                                            logging.info(f"Merging in another shard: {shard_hex}")
                                        merged_df = pd.concat([merged_df, df])

                                for output_file in output_files:
                                    os.remove(output_file)
                        
                        if merged_df is not None:
                            logging.info(f"merged_df['shard_hex'] type: {merged_df.dtypes['shard_hex']}")
                            merged_df['shard_hex'] = merged_df['shard_hex'].astype(str)
                            logging.info(f"merged_df['shard_hex'] type: {merged_df.dtypes['shard_hex']}")
                            merged_df.to_parquet(f"{results_loc}{relation_key}__split-{split_id:03}@{num_splits}__shard_worker-{shard_worker_desc_file_hash}.parquet", engine=config['PARQUET_ENGINE'])

                            # Debug, check index and column alignment
                            pd.set_option('display.max_columns', None)
                            df2 = pd.read_parquet(f"{results_loc}{relation_key}__split-{split_id:03}@{num_splits}__shard_worker-{shard_worker_desc_file_hash}.parquet", engine=config['PARQUET_ENGINE'])
                            # logging.info(f"Merged archive read back:\n{df2}")
                elif config['ARCHIVE_FORMAT'] == "custom":
                    with open(f"{results_loc}split-{split_id:03}@{num_splits}__shard_worker-{shard_worker_desc_file_hash}__archive.txt", 'w') as fout:
                        for shard_hex in shard_worker_desc:
                            output_files = list(glob.glob(f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/split-*__*/*__shard-{shard_hex}__*"))
                            if output_files:
                                # logging.info(f"Archiving {len(output_files)} output files for shard {shard_hex}")
                            
                                for output_file in output_files:
                                    with open(output_file, 'r') as fin:
                                        file_contents = fin.read()
                                    RAMDataPond.archive_str_data(output_file, file_contents, f"{results_loc}shard_worker-{shard_worker_desc_file_hash}/", fout)
                                
                                for output_file in output_files:
                                    os.remove(output_file)
                
        output_dirs = sorted(list(glob.glob(f"{results_loc}shard_worker-*")))
        for output_dir in output_dirs:
            shutil.rmtree(output_dir)

def upload_results_to_bucket():
    if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] != "internal":
        st = default_timer()
        files_to_upload_to_scratch = sorted(list(glob.glob(f"{results_loc}*split*")))
        logging.info(f"files_to_upload_to_scratch (first 30 shown): {files_to_upload_to_scratch[:30]}")
        if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "gcp":
            raise RuntimeError("GCP bucket no longer supported due to possible financial cost if done incorrectly!")
            logging.info("\nUploading files to Google storage")
            filenames_to_upload_to_scratch = [os.path.basename(filepath) for filepath in files_to_upload_to_scratch]
            upload_files_to_gcp(results_loc, filenames_to_upload_to_scratch, f"{config['TIMESTAMP']}/relation_index", config['GCP_BUCKET'], config['GCP_SCRATCH_BLOB_PATH'])#, dryrun=True)
        elif config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "aws":
            logging.info("\nUploading files to AWS storage")
            upload_folder_relative_path = f"aws_upload/{config['TIMESTAMP']}/relation_index/"
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

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("BUILD RELATION INDEX")
    
    analyze_memory_usage()

    data_loc = "../data/"
    results_loc = "../results/"

    config = read_config(["id", "spatial", "relation"])
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

    missing_enum_labels = set()
    
    if config['RELATION_INDEX_ENABLED']:
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
        
        relationship_keys = {}
        for relationship in config['DATA_CONFIG']['relations']:
            relationship_keys[relationship] = convert_relation_key(relationship)
        logging.info(f"Relationship keys: {relationship_keys}")

        timestamps.append(("read relationships", default_timer()))

        split_id, num_splits, shard_hexes, header = process_test_file()
        if not shard_hexes:
            if config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row":
                split_id, num_splits, shard_hexes, header = process_input_file()
            elif config['DATA_CONFIG']['structure'] == "one_annotation_per_file__one_point_per_row":
                raise RuntimeError(f"Structure {config['DATA_CONFIG']['structure']} should have been converted in an earlier capsule.")
                split_id, num_splits, shard_hexes, header = process_input_dir()
            elif config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row_in_one_field":
                split_id, num_splits, shard_hexes, header = process_input_file()
        # logging.info(f"Input all shard hexes: {shard_hexes}")

        analyze_memory_usage()
        timestamps.append(("process input file", default_timer()))

        archive_results(split_id, num_splits, header)
        analyze_memory_usage()
        timestamps.append(("archive results", default_timer()))

        if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
            upload_results_to_bucket()
        else:
            logging.info(f"\n{data_loc}DEBUG_FLAG.txt file found. Results won't be uploaded to externally.")

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
    
    if missing_enum_labels:
        logging.error(f"Missing enum labels: {missing_enum_labels}")
        raise ValueError(f"Missing enum labels: {missing_enum_labels}")
    
    finalize_results(results_loc)
    
    analyze_memory_usage()

logging.info("\nDone")
process_running_time()
