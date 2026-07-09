import sys
import logging
import os
import io
import glob
import math
from timeit import default_timer
from collections import Counter
import pandas as pd
import json
import string
import re
import random
import shutil
import tarfile
import ast

import shared.utilities as utilities

from shared.util import *
from shared.ram_data_pond import *
from shared.google_storage import *
from shared.aws_storage import *

import shared.simple_writer_no_spatial_indexing as simple_writer_no_spatial_indexing
import shared.annotations as anno

def get_shard_hex(shard_number: int, shard_bits: int) -> str:
    """Convert shard number to zero-padded lowercase hex string.

    :param shard_number: The shard number to convert
    :param shard_bits: Number of bits for the shard
    :return: Zero-padded lowercase hex string
    """

    # THIS FUNCTION IS ONLY CALLED BY A DEBUGGING ROUTINE

    padding = math.ceil(shard_bits / 4)
    return f"{shard_number:0{padding}x}"

def extract_archives():
    logging.info("Extracting archives")
    # Decompress and dearchive any compressed files

    # Look for tar files. If any are found, this indicates that the upstream capsule used tar archiving.
    # Look for parquet files. If any are found, this indicates that the upstream capsule used parquet archiving.
    # If none are found, this indicates that the upstream capsule used custom archiving.
    tar_files = list(glob.glob(f"{data_loc}split*.tar*"))
    parquet_files = list(glob.glob(f"{data_loc}*split*.parquet"))
    
    if not config['ARCHIVE_WITH_SHARD_GROUPING']:
        if tar_files:
            logging.info(f"Input tar files:\n  {'\n  '.join(sorted(tar_files)).strip()}" + '\n')
            for tarred_file in tar_files:
                mode = "r:gz" if config['COMPRESS_ARCHIVE'] else "r"
                with tarfile.open(tarred_file, mode) as tar:
                    tar.extractall(path=data_loc)
            logging.info(f"{data_loc} contents after extraction ({len(os.listdir(data_loc))})")  # :\n  {'\n  '.join(sorted(os.listdir(data_loc))).strip()}")
            logging.info("\n")
        elif parquet_files:
            raise NotImplementedError("Parquet support not yet implemented")
        else:  # Custom archiving
            file_shard_filters = set([f"__shard-{shard_hex}_" for shard_hex in assigned_shards])
            if False:  # Dearchive from a text file
                archive_files = list(glob.glob(f"{data_loc}split*__archive.txt"))
                logging.info(f"Input archive files:\n  {'\n  '.join(sorted(archive_files)).strip()}" + '\n')
                for archive_file in archive_files:
                    RAMDataPond.dearchive_file(data_loc, archive_file, file_shard_filters)
            else:  # Dearchive from an indexed binary file
                archive_files = list(glob.glob(f"{data_loc}split*__archive.bin"))
                for archive_file in archive_files:
                    try:
                        archive_index_file = archive_file.replace("__archive.bin", "__archive_idx.txt")
                        with open(archive_index_file) as f:
                            archive_index_file_content = f.read()
                        archive_index = ast.literal_eval(archive_index_file_content)
                        # logging.info(f"Archive index:\n{archive_index}")
                        for filepath, file_offset, filebytes_len in archive_index:
                            filename = os.path.basename(filepath)
                            # logging.info(f"\nArchive inner filepath: {filepath}")
                            for keep_filter in file_shard_filters:
                                if keep_filter in filename:
                                    # logging.info(f"Filter: {keep_filter}, Dearchiving this inner file: {filename}")
                                    RAMDataPond.dearchive_bin_file(data_loc, archive_file, filepath, file_offset, filebytes_len)
                                    break
                                # else:
                                #     logging.info(f"Filter: {keep_filter}, Skipping this inner file")
                    except Exception as e:
                        logging.critical(f"Failed to dearchive index: {archive_index_file}\n")
                        logging.critical(e)
                        logging.critical(f"\nArchive index content ({len(archive_index_file_content)}):\n'''\n{archive_index_file_content[:100]}\n'''\n...to...\n{archive_index_file_content[-100:]}\n'''\n'''\n")
                        raise e
            
            # Move files up out of the shard worker subdirectories
            shard_worker_relation_dirs = glob.glob(f"{data_loc}shard_worker-*/*")
            for shard_worker_relation_dir in shard_worker_relation_dirs:
                path_pcs = shard_worker_relation_dir[len(data_loc):].split('/')
                new_shard_worker_relation_dir = data_loc + '/'.join(path_pcs[1:])
                logging.info(f"Moving\n  {shard_worker_relation_dir} to\n  {new_shard_worker_relation_dir}")
                shutil.move(shard_worker_relation_dir, new_shard_worker_relation_dir)
    else:  # ARCHIVE_WITH_SHARD_GROUPING
        if tar_files:
            logging.info(f"Input tar files:\n  {'\n  '.join(sorted(tar_files)).strip()}" + '\n')
            for tarred_file in tar_files:
                mode = "r:gz" if config['COMPRESS_ARCHIVE'] else "r"
                with tarfile.open(tarred_file, mode) as tar:
                    tar.extractall(path=data_loc)
            logging.info(f"{data_loc} contents after extraction ({len(os.listdir(data_loc))})")  # :\n  {'\n  '.join(sorted(os.listdir(data_loc))).strip()}")
            logging.info("\n")
        elif parquet_files:
            logging.info(f"Input parquet files:\n  {'\n  '.join(sorted(parquet_files)).strip()}" + '\n')
            for parquet_file in parquet_files:
                # logging.info(f"Input parquet file: {parquet_file}")
                pcs = os.path.basename(parquet_file).split('__')
                df = pd.read_parquet(parquet_file, engine=config['PARQUET_ENGINE'])
                logging.info(f"Read Parquet file of length {len(df)} containing shards {sorted(list(df['shard_hex'].unique()))}")
                pd.set_option('display.max_columns', None)
                # logging.info(f"Input parquet file:\n{df}")
                for shard_hex in assigned_shards:
                    # df_one_shard = df[df['shard_hex']==f'"{shard_hex}"']  # The shard hexes are quoted to force them to strings so they won't lose their leading 0s
                    df_one_shard = df[df['shard_hex']==f'_{shard_hex}']  # The shard hexes are underscore-prefixed to force them to strings so they won't lose their leading 0s
                    if len(df_one_shard) > 0:
                        logging.info(f"Filtered Parquet file to shard {shard_hex}, yielding {len(df_one_shard)} rows")
                        # In the following filename, the double-underscore after the shard hex is important
                        df_one_shard.to_csv(f"{data_loc}{pcs[0]}__{pcs[1]}__shard-{shard_hex}__.csv", index=False, header=False)
                        # df2 = pd.read_csv(f"{data_loc}{pcs[0]}__{pcs[1]}__shard-{shard_hex}__.csv", names=header)
        else:  # Custom archiving
            archive_files = list(glob.glob(f"{data_loc}split*__shard_worker-{shard_worker_desc_file_hash}__archive.txt"))
            logging.info(f"Input archive files:\n  {'\n  '.join(sorted(archive_files)).strip()}" + '\n')
            # file_shard_filters = set([f"__shard-{shard_hex}" for shard_hex in assigned_shards])
            # logging.info(f"file_shard_filters: {file_shard_filters}")
            for archive_file in archive_files:
                RAMDataPond.dearchive_file(data_loc, archive_file) #, file_shard_filters)

def merge_csv_splits(assigned_shard_hex, this_shard_input_files):
    logging.info(f"\nmerge_csv_splits() Shard {assigned_shard_hex}")

    merged_relation_shard_dfs = {}
    for fi, input_file in enumerate(this_shard_input_files):
        # if fi <= 1:
            # logging.info(f"\nInput file: {os.path.basename(input_file)}")

        # with open(input_file) as f:
        #     for i in range(2):
        #         logging.info(f"merge_splits() Input file first few lines: {f.readline().strip()}")

        df = pd.read_csv(input_file, names=header)

        df_unique_shards = list(df['shard_hex'].unique())
        if len(df_unique_shards) != 1:
            raise ValueError(f"Input file unique shards: {df_unique_shards} (should contain only {assigned_shard_hex})")
            df_unique_shard = df_unique_shards[0][df_unique_shards[0].rindex('_')+1:] if '_' in df_unique_shards[0] else df_unique_shards[0]
            if df_unique_shard != assigned_shard_hex:
                raise ValueError(f"Input file unique shards: {df_unique_shards} (should contain only {assigned_shard_hex} (with potential underscore prefix))")
        
        pcs = input_file.split('/')[-1].split('__')
        relation_key = pcs[0]
        shard_hex2 = pcs[2].split('-')[1]
        assert shard_hex2 == assigned_shard_hex
        if fi <= 1:
            # logging.info(f"  Num rows, relation key, shard hex:    {len(df)}   {relation_key}    {assigned_shard_hex}")
            logging.info(f"Input file {os.path.basename(input_file)} ({fi+1} of {len(this_shard_input_files)}): {len(df)} rows, relation-key '{relation_key}', shard-hex {assigned_shard_hex}")

        if relation_key not in merged_relation_shard_dfs:
            merged_relation_shard_dfs[relation_key] = {}
        
        if assigned_shard_hex not in merged_relation_shard_dfs[relation_key]:
            if True:  # fi <= 1:
                logging.info(f"  Initializing merged shard file for relation_key, shard:    {relation_key}    {assigned_shard_hex} from first shard file")
            merged_relation_shard_dfs[relation_key][assigned_shard_hex] = df
        else:
            curr_len = len(merged_relation_shard_dfs[relation_key][assigned_shard_hex])
            # logging.info(f"  Merging in another shard file for relation_key, shard:    {relation_key}    {assigned_shard_hex} onto df of current len {curr_len}")
            merged_relation_shard_dfs[relation_key][assigned_shard_hex] = pd.concat([merged_relation_shard_dfs[relation_key][assigned_shard_hex], df])
            if fi <= 1:
                logging.info(f"    Merged in another shard, accumulating len from {curr_len} to {len(merged_relation_shard_dfs[relation_key][assigned_shard_hex])}")
    
    logging.info(f"\nMerged tables for shard {assigned_shard_hex}:")
    for relation_key, merged_relation_shard_dfs_one_relation in merged_relation_shard_dfs.items():
        for shard_hex, df in merged_relation_shard_dfs_one_relation.items():
            logging.info(f"  Merged shard:    Relation {relation_key:20}    Shard {shard_hex}    DF len {len(df)}")
    
    return merged_relation_shard_dfs

def save_shard_data_as_csv(rows_this_level_df, subdir, relation_key, relation_column_name, shard_hex):
    if OUTPUT_STYLE == "capsule":
        filepath = f"{results_loc}annotations_one_shard__shard-{shard_hex}.csv"
    elif OUTPUT_STYLE == "results":
        filepath = f"{results_loc}{subdir}relation_indices__{relation_key}__{shard_hex}.csv"
    logging.info(f"\nWriting shard's CSV file with {len(rows_this_level_df)} rows: {filepath}")
    rows_this_level_df.to_csv(filepath, index=False, header=False)

def convert_relation_fields(row, data_relation_col_indices):
    # logging.info(f"convert_relation_fields(): {row}")
    
    # Convert relation fields to either an int or a list of ints
    relation_fields = {lbl: row[col_idx] for lbl, col_idx in data_relation_col_indices}
    for lbl, relation_field_val in relation_fields.items():
        # The column might already be an int, not a string, so check for that first
        if isinstance(relation_field_val, int):
            relation_field = relation_field_val
        else:
            # Casting a float to an int won't raise an exception. We have to check for a float explicitly.
            if '.' in relation_field_val:
                raise ValueError(f"Relation field must be int or list of ints: {relation_field_val}")
            
            try:
                # Try casting the field as an int (we have already established it isn't a float above)
                relation_field = int(relation_field_val)
            except:
                try:
                    # Try casting the field as an list (we have already established it isn't a float above)
                    if relation_field_val[0] != '[':
                        relation_field_val = '[' + relation_field_val
                    if relation_field_val[-1] != ']':
                        relation_field_val += ']'
                    relation_field = ast.literal_eval(relation_field_val)
                    if not isinstance(relation_field, list):
                        raise ValueError(f"Relation field must be int or list of ints: {relation_field_val}")
                    
                    # Ensure that every item in the list is an int
                    for v in relation_field:
                        if not isinstance(v, int):
                            raise ValueError(f"Relation field must be int or list of ints: {relation_field_val}")
                except:
                    raise ValueError(f"Relation field must be int or list of ints: {relation_field_val}")
        
        relation_fields[lbl] = relation_field
    
    return relation_fields

def build_annotation(annotation_description):
    if 'point_annotation_config' in config['DATA_CONFIG']:
        return anno.PointAnnotation(
            id=annotation_description["id"],
            position=annotation_description["position"],
            properties=annotation_description["properties"],
            relations=annotation_description.get("relations", {})
        )
    elif 'line_annotation_config' in config['DATA_CONFIG']:
        return anno.LineAnnotation(
            id=annotation_description["id"],
            start=annotation_description["start"],
            end=annotation_description["end"],
            properties=annotation_description["properties"],
            relations=annotation_description.get("relations", {})
        )
    elif 'polyline_annotation_config' in config['DATA_CONFIG']:
        return anno.PolyLineAnnotation(
            id=annotation_description["id"],
            num_points=len(annotation_description["points"]),
            points=annotation_description["points"],
            properties=annotation_description["properties"],
            relations=annotation_description.get("relations", {})
        )

def build_annotation_description__one_annotation_per_row__multiple_points_per_row(row, columns, pt_positions, data_property_col_indices, data_relation_col_index):
    relation_fields = convert_relation_fields(row, [data_relation_col_index])

    desc = {
        "id": row[columns.index(config['DATA_CONFIG']['id_column'])],
        "properties": {},  # {lbl: row[col_idx] for (lbl, col_idx) in data_property_col_indices},
        "relations": relation_fields,
    }

    col_index_map = {col_name: i for i, col_name in enumerate(header)}

    for prop_lbl, prop_info in config['DATA_CONFIG']['properties'].items():
        prop_id = prop_lbl  # prop_info['id']
        col_idx = col_index_map[prop_info['id']] if prop_info['id'] is not None else None
        field = row[col_idx] if col_idx is not None else None

        if prop_info['type'] == "vector":
            vec = calculate_annotation_vector(pt_positions)
            if vec:
                desc["properties"]['vector_x'] = vec[0]
                desc["properties"]['vector_y'] = vec[1]
                desc["properties"]['vector_z'] = vec[2]
        elif prop_info['type'] == "rgb":
            if field[0] == '#':
                desc["properties"][prop_id] = hex_to_rgb(field)
            else:
                raise ValueError(f"Only Hex colors are currently supported: {field}")
        elif prop_info['type'] == "rgba":
            if field[0] == '#':
                desc["properties"][prop_id] = hex_to_rgba(field)
            else:
                raise ValueError(f"Only Hex colors are currently supported: {field}")
        elif prop_info['enum_values'] is not None:
            # This will raise a ValueError if the field value isn't in the property info's enum_labels list
            if field in prop_info['enum_labels']:
                enum_label_idx = prop_info['enum_labels'].index(field)
                enum_value = prop_info['enum_values'][enum_label_idx]
            else:
                missing_enum_labels.add(field)
                enum_value = -1
            desc["properties"][prop_id] = enum_value
        else:
            desc["properties"][prop_id] = field
        
        if debug:
            logging.info(f"Anno prop: {desc['properties'][prop_id]}")
    
    if 'point_annotation_config' in config['DATA_CONFIG']:
        desc["position"] = pt_positions[config['DATA_CONFIG']["point_annotation_config"]["pt_column_label"]]
    elif 'line_annotation_config' in config['DATA_CONFIG']:
        desc["start"] = pt_positions[config['DATA_CONFIG']["line_annotation_config"]["start_pt_column_label"]]
        desc["end"] = pt_positions[config['DATA_CONFIG']["line_annotation_config"]["end_pt_column_label"]]
    elif 'polyline_annotation_config' in config['DATA_CONFIG']:
        raise RuntimeError("Not implemented yet")
    
    return desc

def save_annotations_as_precomputed(df, data_properties, data_property_by_col_idx, relation_id, relation_col_idx, shard_hex_debug):
    columns = list(df.columns)
    assert columns == header
    header_reverse_map = {col: i for i, col in enumerate(header)}

    spatial_pt_columns = config['DATA_CONFIG']['spatial_pt_columns']

    pt_pos_col_idxs = {
        pt_desc: [
            header_reverse_map[pt_pos['x']],
            header_reverse_map[pt_pos['y']],
            header_reverse_map[pt_pos['z']],
        ] \
        for pt_desc, pt_pos in spatial_pt_columns.items()
    }
    
    annotation_descriptions = []
    # for row_i, row in df.iterrows():  # Pandas Dataframe iteration, slower than itertuples()
    for row_i, row in enumerate(df.itertuples(index=False)):  # Pandas Dataframe iteration, faster than iterrows()
        debug = row_i < 3 or row_i % 100000 == 0

        if debug:
            pt_positions_old_method = {
                pt_desc: [
                        float(row[header_reverse_map[pt_pos['x']]]),
                        float(row[header_reverse_map[pt_pos['y']]]),
                        float(row[header_reverse_map[pt_pos['z']]]),
                    ] \
                    for pt_desc, pt_pos in spatial_pt_columns.items()
            }
            logging.info(f"pt_positions {row_i:>8} OLD: {pt_positions_old_method}")
        pt_positions = {
            pt_desc: [
                    float(row[pt_pos_col_idxs[pt_desc][0]]),
                    float(row[pt_pos_col_idxs[pt_desc][1]]),
                    float(row[pt_pos_col_idxs[pt_desc][2]]),
                ] \
                for pt_desc, pt_pos in spatial_pt_columns.items()
        }
        if debug:
            logging.info(f"pt_positions {row_i:>8} NEW: {pt_positions}")
            if pt_positions != pt_positions_old_method:
                raise ValueError("pt_positions: {pt_positions} != {pt_positions_old_method}")
        
        annotation_description = build_annotation_description__one_annotation_per_row__multiple_points_per_row(row, columns, pt_positions, data_property_by_col_idx, relation_col_idx)
        annotation_descriptions.append(annotation_description)

        if debug:
            sharding_spec = anno.ShardingSpec(hash=config['RELATION_SHARDING_HASH'], preshift_bits=config['RELATION_PRESHIFT_BITS'], shard_bits=config['RELATION_SHARDING_BITS'], minishard_bits=config['RELATION_MINISHARDING_BITS'])
            relation_fields = convert_relation_fields(row, [relation_col_idx])
            for lbl, relation_vals in relation_fields.items():
                # logging.info(f"Relation list: {lbl} {type(relation_vals)} {relation_vals}")
                if not isinstance(relation_vals, list):
                    relation_vals = [relation_vals]
                found_it = False
                shard_hexes2 = []
                for relation_val in relation_vals:
                    shard_num, minishard_num = sharding_spec.get_shard_number(relation_val, True)
                    shard_hex2 = get_shard_hex(shard_num, sharding_spec.shard_bits)

                    # if relation_val == 864691135568681196:
                    #     logging.info(f"AAA {relation_val} {sharding_spec.shard_bits} {shard_num} {shard_hex2}")
                        
                    shard_hexes2.append(shard_hex2)
                    if shard_hex2 == shard_hex_debug:
                        found_it = True
                        break
                if not found_it:
                    logging.info(f"ERROR! No relation values match shard hex: {lbl} {relation_vals}    {shard_hex_debug:>4} not in {shard_hexes2}")
    
    timestamps.append(("gather annotation_descriptions from table", default_timer()))
    
    writer = simple_writer_no_spatial_indexing.SimpleWriter("LINE")
    writer_profile = None

    # Define properties
    for property_name, property_info in data_properties.items():
        writer.property_specs.append(
            anno.PropertySpec(property_name, property_info['type'], property_name, property_info['enum_values'], property_info['enum_labels']))
    
    # Define relationships
    relation_sharding_spec = None
    if config['RELATION_SHARDING']:
        relation_sharding_spec = anno.ShardingSpec(hash=config['RELATION_SHARDING_HASH'], preshift_bits=config['RELATION_PRESHIFT_BITS'], shard_bits=config['RELATION_SHARDING_BITS'], minishard_bits=config['RELATION_MINISHARDING_BITS'])  # Comment out this line (or don't conditionally don't call it) to generate a non-sharded id index, but be aware that every relation will get a separate file!
    elif len(annotation_descriptions) > 100:
        raise ValueError("Sharding is disabled for the relation index, but the number of annotations is high. This will produce a lot of individual files.")
    
    relation = anno.Relationship(relation_id, sharding=relation_sharding_spec)
    writer.relationships.append(relation)
    
    timestamps.append(("init writer", default_timer()))

    for i, annotation_description in enumerate(annotation_descriptions):
        annotation = build_annotation(annotation_description)
        if i < 2:# or (i % 10 == 0 and i < 100):
            logging.info(f"Anno {i:10} description and object:\n{annotation_description}\n{annotation}\n")
        # if i < 2:# or (i % 10 == 0 and i < 1000):
        #     logging.info(f"Anno {i:10} description and object:    {annotation_description['id']:10}    {annotation.id:10}")
        writer.annotations.append(annotation)
    
    timestamps.append(("append annotation_description annotations to writer", default_timer()))

    return writer, relation

def save_csv_shard_data_as_precomputed(df, subdir, relation_id, relation_key, relation_column_name, shard_hex):
    timestamps.append(("save_csv_shard_data_as_precomputed() start", default_timer()))

    logging.info(f"\nWriting a precomputed file for relation '{relation_id}', shard {shard_hex}, of len {len(df)} rows to: {results_loc + subdir}")

    USE_RAM_BUFFER = True
    logging.info(f"USE_RAM_BUFFER: {USE_RAM_BUFFER}")

    assert list(df.columns) == header

    col_index_map = {col_name: i for i, col_name in enumerate(df.columns)}

    data_properties = config['DATA_CONFIG']['properties']
    data_properties_cols = [(prop_lbl, prop_info['id']) for prop_lbl, prop_info in data_properties.items()]
    data_property_col_indices = [(prop_col, col_index_map[prop_col]) for prop_lbl, prop_col in data_properties_cols]

    data_relation_col_idx = (relation_id, col_index_map[relation_column_name])

    writer, relation = save_annotations_as_precomputed(df, data_properties, data_property_col_indices, relation_id, data_relation_col_idx, shard_hex)

    # Direct the writer to write its contents out
    if not USE_RAM_BUFFER:
        logging.info("Writing precomputed file without RAM buffer")
        writer._write_related_index(utilities.path_join(results_loc, subdir), relation)
        timestamps.append(("write precomputed to file", default_timer()))
    else:
        logging.info("Writing precomputed file with RAM buffer")
        # The following lines are copied out of simple_writer_no_spatial_index.py and sharding.py
        rel_dir_path = utilities.path_join(results_loc, relation.key)
        logging.info(f"  rel_dir_path: {rel_dir_path}")
        dir_path = os.path.expanduser(rel_dir_path)
        logging.info(f"  dir_path: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

        timestamps.append(("prepare precomputed writer filepath", default_timer()))
        
        logging.info(f"Calling writer._writef_related_index() with relation of id '{relation.id}'")
        file_buffer_bytes, writer_profile = writer._writef_related_index(relation, shard_number=int(shard_hex, 16))
        timestamps.append(("write precomputed to buffer", default_timer()))
        
        sharding_spec = anno.ShardingSpec(hash=config['RELATION_SHARDING_HASH'], preshift_bits=config['RELATION_PRESHIFT_BITS'], shard_bits=config['RELATION_SHARDING_BITS'], minishard_bits=config['RELATION_MINISHARDING_BITS'])
        
        logging.info(f"File buffer len for shard {shard_hex}: {len(file_buffer_bytes)}")
        
        filepath = utilities.path_join(dir_path, f"{shard_hex}.shard")
        logging.info(f"  filepath: {filepath}")
        file_path = os.path.expanduser(filepath)
        logging.info(f"  filepath: {filepath}")
        assert not os.path.exists(filepath)
        with open(filepath, "wb") as f_disk:
            f_disk.write(file_buffer_bytes)
        timestamps.append(("write precomputed buffer to file", default_timer()))
        logging.info(f"Shard file size for shard {shard_hex}: {os.path.getsize(filepath)}")

    # Remove the extraneous files and move the target file
    relation_files = list(glob.glob(f"{results_loc}{subdir}{relation_key}/*.shard"))
    # logging.info(f"Relation files: {relation_files}")
    if OUTPUT_STYLE == "results":
        os.makedirs(f"{results_loc}{subdir}", exist_ok=True)
    for relation_file in relation_files:
        relation_filename = relation_file.split('/')[-1]
        if relation_filename != f"{shard_hex}.shard":
            os.remove(relation_file)
        else:
            if OUTPUT_STYLE == "capsule":
                shutil.move(relation_file, f"{results_loc}relation_indices__{relation_key}__{shard_hex}.shard")
            elif OUTPUT_STYLE == "results":
                shutil.move(relation_file, f"{results_loc}{subdir}{shard_hex}.shard")
    
    timestamps.append(("save_csv_shard_data_as_precomputed() end", default_timer()))

    return writer_profile

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("GENERATE RELATION INDEX SHARDS")
    
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

    # Pick one: changing this requires altering the pipeline topology
    # OUTPUT_STYLE = "capsule"  # Pipeline must connect this capsule to the 'reorganize directory structure' capsule with 'Collect' type
    OUTPUT_STYLE = "results"  # Pipeline must connect this capsule to the 'results'

    FILE_PROCESSING_METHOD = "dataframe"  # dataframe or text
    header = [v for v in config['DATA_CONFIG']['columns']]  # Copy, not just a reference, so we can alter it
    if 'id_column' in config['DATA_CONFIG']:
        id_column = config['DATA_CONFIG']['id_column']
        # logging.info(f"id_column: {id_column}")
        if id_column is None:
            logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {header[0]}.")
        if FILE_PROCESSING_METHOD == "dataframe":
            if config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row":
                header.append('shard_hex')
        logging.info(f"Header: {header}")
    else:
        assert 'id_src' in config['DATA_CONFIG']

    missing_enum_labels = set()
    
    if config['RELATION_INDEX_ENABLED']:
        timestamps = []
        timestamps.append(("start", default_timer()))

        writer_profiles = []

        data_loc_contents = sorted(os.listdir(data_loc))
        data_loc_contents = [v for v in data_loc_contents if "placeholder" not in v]
        logging.info(f"{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):")
        logging.info('  ' + '\n  '.join(data_loc_contents[:30]).strip() + '\n')

        # The shard worker file is only used when the upstream capsule is "build relation index" (8542974)
        shard_worker_desc_files = list(glob.glob(f"{data_loc}shard_worker*txt"))
        assert len(shard_worker_desc_files) == 1
        shard_worker_desc_file_path = shard_worker_desc_files[0]
        shard_worker_desc_filename = shard_worker_desc_file_path.split('/')[-1]
        shard_worker_desc_file_hash = shard_worker_desc_filename[:shard_worker_desc_filename.rindex('.')].split('_')[-1]
        with open(shard_worker_desc_file_path) as f:
            shard_worker_desc = f.read()
            logging.info(f"shard_worker_desc_file_hash, shard_worker_desc: {shard_worker_desc_file_hash} {shard_worker_desc}")
        assigned_shards = shard_worker_desc.split('_')
        # Add all versions of assigned shards with leading 0s stripped off
        # On second thought, this doesn't make any sense for the relation index. It only has practical applications in the spatial index.
        # additional_assigned_shards = []
        # for assigned_shard in assigned_shards:
        #     while assigned_shard and assigned_shard[0] == '0':
        #         assigned_shard = assigned_shard[1:]
        #         if assigned_shard:
        #             additional_assigned_shards.append(assigned_shard)
        # assigned_shards.extend(additional_assigned_shards)
        logging.info(f"Shard worker assigned shards: {assigned_shards}")
        logging.info("\n")

        timestamps.append(("read_shard_worker_desc", default_timer()))
        
        if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
            if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] != "internal":
                if not config['ARCHIVE_WITH_SHARD_GROUPING']:
                    raise ValueError("PASS_DATA_BETWEEN_CAPSULES_METHOD!=internal requires ARCHIVE_WITH_SHARD_GROUPING. Otherise large amounts of GCP/AWS egress can occur, which is inefficient at best and can be very expensive at worst.")
                st = default_timer()
                filename_filter = f"shard_worker-{shard_worker_desc_file_hash}"
                logging.info(f"A filename_filter: {filename_filter}")
                if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "gcp":
                    raise RuntimeError("GCP bucket no longer supported due to possible financial cost if done incorrectly!")
                    logging.info("\nDownloading files from Google storage")
                    download_files_from_gcp(f"{config['TIMESTAMP']}/relation_index", data_loc, filename_filter, config['GCP_BUCKET'], config['GCP_SCRATCH_BLOB_PATH'])#, dryrun=True)
                    ets = default_timer() - st
                    logging.info(f"\nGCP download elapsed time: {seconds_to_hms(ets)}")

                    logging.info(f"\nMoving GCP downloads to {data_loc}")
                    downloaded_files = glob.glob(f"{data_loc}{config['GCP_SCRATCH_BLOB_PATH']}/{config['TIMESTAMP']}/relation_index/*")
                    for downloaded_file in downloaded_files:
                        os.rename(downloaded_file, f"{data_loc}{os.path.basename(downloaded_file)}")
                elif config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "aws":
                    logging.info("\nDownloading files from Amazon storage")
                    download_folder_relative_path = f"{config['TIMESTAMP']}/relation_index/"
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
            logging.info(f"\n{data_loc}DEBUG_FLAG.txt file found. Results won't be downloaded from bucket.")
        
        extract_archives()

        data_loc_contents = sorted(os.listdir(data_loc))
        data_loc_contents = [v for v in data_loc_contents if "placeholder" not in v]
        logging.info(f"\n{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):")
        logging.info('  ' + '\n  '.join(data_loc_contents[:30]).strip() + '\n')

        analyze_memory_usage()
        
        timestamps.append(("extract_input_files", default_timer()))

        input_extension = "csv"
        if config['DATA_CONFIG']['structure'] == "one_annotation_per_file__one_point_per_row":
            input_extension = "swc"  # For now, just support Wan-Qing's SWC data

        # The previous capsule (Combine Relation Index Splits) could be configured to group its outputs in one of two ways.
        # That distinction governs the nature of the input to this capsule, so we need to detect those two cases and handle them appropriately.
        # We will first look for shard files in the top-level input directory.
        # Finding them will indicate one of the two scenarios.
        # Their absense will imply that the shard files are stored one level down, in subdirectories of the input directory, thereby indicating the other scenario.

        all_relation_keys = set()
        for ashi, assigned_shard_hex in enumerate(assigned_shards):
            logging.info("\n" + "*" * 100 + "\n")
            timestamps.append(("shard_loop_top", default_timer()))

            this_shard_input_files = sorted(glob.glob(f"{data_loc}*shard-{assigned_shard_hex}_*.{input_extension}"))
            if ashi <= 0:
                logging.info(f"Shard files for shard {assigned_shard_hex} ({ashi} of {len(assigned_shards)}) (at top level):\n  {'\n  '.join(this_shard_input_files)}")
            else:
                logging.info(f"Shard files for shard {assigned_shard_hex} ({ashi} of {len(assigned_shards)}) (at top level) (first 5 shown):\n  {'\n  '.join(this_shard_input_files[:5])}")

            if not this_shard_input_files:
                logging.info("No shard files found in the top-level input directory. The input is presumably grouped into relation subdirectories.")
                this_shard_input_files = sorted(glob.glob(f"{data_loc}split*/*shard-{assigned_shard_hex}_*.csv"))
                logging.info(f"Shard files (in subdirs) ({len(this_shard_input_files)}) (first 30 shown):\n  {'\n  '.join(this_shard_input_files[:30])}")
            
            if input_extension == "csv":
                logging.info("Archives are expected to be CSV files")
                merged_relation_shard_dfs = merge_csv_splits(assigned_shard_hex, this_shard_input_files)

                timestamps.append(("merge_splits", default_timer()))

                for relation_key, merged_shard_dfs in merged_relation_shard_dfs.items():
                    logging.info(f"\nSaving merged shards for relation {relation_key}, shard {assigned_shard_hex}")
                    all_relation_keys.add(relation_key)
                    for config_relation, config_relation_info in config['DATA_CONFIG']['relations'].items():
                        config_relation_column_name = config_relation_info['id']
                        # The following conversions are copied from Joe Strout's code
                        # Convert to lowercase
                        config_relation_key = config_relation.lower()
                        # Remove all punctuation
                        config_relation_key = config_relation_key.translate(str.maketrans("", "", string.punctuation))
                        # Replace spaces (and any other whitespace) with underscores
                        config_relation_key = re.sub(r"\s+", "_", config_relation_key)
                        # logging.info(f"Testing {relation_key} vs. {config_relation}, {config_relation_key}")
                        if config_relation_key == relation_key:
                            relation, relation_column_name = config_relation, config_relation_column_name
                            break
                    # logging.info(f"Relationship:  {relation}, {relation_key}, {relation_column_name}")

                    timestamps.append(("generate_relation_label", default_timer()))

                    for merged_shard_hex, merged_df in merged_shard_dfs.items():
                        logging.info(f"  Merged shard-{merged_shard_hex} num rows: {len(merged_df)}")
                        if OUTPUT_STYLE == "capsule":
                            subdir_shard = f"relation_indices__{relation_key}/"
                        elif OUTPUT_STYLE == "results":
                            subdir_shard = f"{relation_key}/"
                        os.makedirs(f"{results_loc}{subdir_shard}", exist_ok=True)
                        writer_profile = save_csv_shard_data_as_precomputed(merged_df, subdir_shard, relation, relation_key, relation_column_name, merged_shard_hex)
                        writer_profiles.append((merged_shard_hex, writer_profile))
                        
                        timestamps.append(("save_as_precomputed", default_timer()))

                        if False:
                            if OUTPUT_STYLE == "capsule":
                                subdir_csv = f"relation_indices__{relation_key}__csv/"
                            elif OUTPUT_STYLE == "results":
                                subdir_csv = f"{relation_key}__csv/"
                            os.makedirs(f"{results_loc}{subdir_csv}", exist_ok=True)
                            save_shard_data_as_csv(merged_df, subdir_csv, relation_key, relation_column_name, merged_shard_hex)
                    
                            timestamps.append(("save_as_csv", default_timer()))
                
                        if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
                            if config['UPLOAD_RESULTS_TO_GCP']:
                                logging.info(f"\n  Uploading files for relation {relation_key} to Google Storage")
                                st = default_timer()
                                upload_directory_to_gcp(results_loc, f"{relation_key}/", config["TIMESTAMP"], config['GCP_BUCKET'], config['GCP_RESULTS_BLOB_PATH'])#, dryrun=True)
                                ets = default_timer() - st
                                logging.info(f"  GCP upload for relation {relation_key} elapsed time: {seconds_to_hms(ets)}")
                            
                                timestamps.append(("upload_to_gcp", default_timer()))
                            else:
                                logging.info("UPLOAD_RESULTS_TO_GCP setting is false. Results won't be uploaded to GCP.")
                        else:
                            logging.info(f"\n{data_loc}DEBUG_FLAG.txt file found. Results won't be uploaded to GCP.")

                        if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
                            # Only do this if GCP uploading is enabled since, if it is disabled, the user may explicitly wish to retrieve the results to use them some other nonGCP way.
                            if config['UPLOAD_RESULTS_TO_GCP']:
                                # To reduce CO storage, there is no need to save the results after uploading them to GCP
                                # Note that deleting this outputs and thereby avoiding copying them to the final results
                                # doesn't make the capsule run any faster.
                                logging.info(f"\nDeleting result files after uploading to GCP")
                                if os.path.exists(f"{results_loc}{relation_key}"):
                                    shutil.rmtree(f"{results_loc}{relation_key}")
                                timestamps.append(("delete results", default_timer()))

                    logging.info(f"Done processing merged shards for relation {relation_key}")
                    analyze_memory_usage()
        
                logging.info(f"Done processing relation keys for shard {assigned_shard_hex}")
                analyze_memory_usage()
            elif input_extension == "swc":
                logging.info("Archives are expected to be SWC files")

                # NOT IMPLEMENTED YET
                writer_profile = None  # save_swc_shard_data_as_precomputed(...NOT IMPLEMENTED YET)
                writer_profiles.append((assigned_shard_hex, writer_profile))

                timestamps.append(("save_as_precomputed", default_timer()))

        logging.info(f"Done processing all assigned_shards")
        logging.info("\n" + "* " * 50 + "\n")

        if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
            # Only do this if GCP uploading is enabled since, if it is disabled, the user may explicitly wish to retrieve the results to use them some other nonGCP way.
            if config['UPLOAD_RESULTS_TO_GCP']:
                # To reduce CO storage, there is no need to save the results after uploading them to GCP
                # Note that deleting this outputs and thereby avoiding copying them to the final results
                # doesn't make the capsule run any faster...I think! I'm not sure.
                logging.info(f"\nDeleting result files after uploading to GCP")
                for relation_key in all_relation_keys:
                    if os.path.exists(f"{results_loc}{relation_key}"):
                        shutil.rmtree(f"{results_loc}{relation_key}")
                timestamps.append(("delete results", default_timer()))

        # logging.error("\nWriter profiles:")
        # for shard_hex, writer_profile in writer_profiles:
        #     logging.error(f"  Shard hex {shard_hex}: {json.dumps(writer_profile, indent=2)}")
        
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

    # finalize_results() works for all other capsules (although for 'generate id index shards' it needs a modified path, as shown here as well),
    # but for this capsule ('generate relation index shards') it also needs an extra glob check and makedirs, as shown here.
    # See similar situation in the 'finalize spatial index unsharded' capsule.
    if not glob.glob(f"{results_loc}*"):
        os.makedirs(f"{results_loc}placeholder_relation/", exist_ok=True)
        finalize_results(f"{results_loc}placeholder_relation/")
    
    analyze_memory_usage()
    
logging.info("\nDone")
process_running_time()
