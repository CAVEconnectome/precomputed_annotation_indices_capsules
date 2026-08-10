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
import random
import shutil
import tarfile

import shared.utilities as utilities

from shared.util import *
from shared.ram_data_pond import *
from shared.google_storage import *
from shared.aws_storage import *

import shared.simple_writer_no_spatial_indexing as simple_writer_no_spatial_indexing
import shared.annotations as anno

def extract_archives():
    logging.info("Extracting archives")

    # Look for tar files. If any are found, this indicates that the upstream capsule used tar archiving.
    # Look for parquet files. If any are found, this indicates that the upstream capsule used parquet archiving.
    # If none are found, this indicates that the upstream capsule used custom archiving.
    tar_files = list(glob.glob(f"{data_loc}split*.tar*"))
    parquet_files = list(glob.glob(f"{data_loc}split*.parquet"))

    if tar_files:
        logging.info(f"Input tar files:\n  {'\n  '.join(sorted(tar_files)).strip()}" + '\n')
        for tarred_file in tar_files:
            mode = "r:gz" if config['COMPRESS_ARCHIVE'] else "r"
            with tarfile.open(tarred_file, mode) as tar:
                tar.extractall(path=data_loc)
        logging.info(f"{data_loc} contents after extraction ({len(os.listdir(data_loc))}) (first 50 shown):\n  {'\n  '.join(sorted(os.listdir(data_loc))[:50]).strip()}")
        # logging.info(f"{data_loc} contents after extraction ({len(os.listdir(data_loc))})")
        logging.info("\n")
    else:
        if not config['ARCHIVE_WITH_SHARD_GROUPING']:
            if parquet_files:
                raise NotImplementedError("ARCHIVE_FORMAT 'parquet' without ARCHIVE_WITH_SHARD_GROUPING not yet implemented")
            else:  # Custom archiving
                file_shard_filters = set([f"__shard-{shard}_" for shard in assigned_shards])
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
                                        # logging.info(f"Filter: {keep_filter}, Dearchiving this inner file")
                                        RAMDataPond.dearchive_bin_file(data_loc, archive_file, filepath, file_offset, filebytes_len)
                                        break
                                    # else:
                                    #     logging.info(f"Filter: {keep_filter}, Skipping this inner file")
                        except Exception as e:
                            logging.critical(f"Failed to dearchive index: {archive_index_file}\n")
                            logging.critical(e)
                            logging.critical(f"\nArchive index content:\n{archive_index_file_content}\n")
                            raise e

                # Move files up out of the shard worker subdirectories
                shard_worker_files = glob.glob(f"{data_loc}shard_worker-*/*csv")
                for shard_worker_file in shard_worker_files:
                    path_pcs = shard_worker_file[len(data_loc):].split('/')
                    new_shard_worker_file = data_loc + '/'.join(path_pcs[1:])
                    logging.info(f"Moving\n  {shard_worker_file} to\n  {new_shard_worker_file}")
                    os.rename(shard_worker_file, new_shard_worker_file)
        else:  # ARCHIVE_WITH_SHARD_GROUPING
            if parquet_files:
                # Read one parquet file per shard (This is inefficient. It is better to group all shards per shard worker.)
                # for shard_hex in assigned_shards:
                #     logging.info(f"Input parquet files:\n  {'\n  '.join(sorted(parquet_files)).strip()}" + '\n')
                #     for parquet_file in parquet_files:
                #         df = pd.read_parquet(parquet_file, engine=config['PARQUET_ENGINE'])
                #         df.to_csv(parquet_file.replace(".parquet", ".csv"), index=False, header=False)

                # Read one parquet file per shard worker (This is better. It groups all shards per shard worker into a single file.)
                this_shard_worker_parquet_files = [parquet_file for parquet_file in parquet_files if f"shard_worker-{shard_worker_desc_file_hash}." in parquet_file]
                # print("AAA", parquet_files)
                # print("BBB", this_shard_worker_parquet_files)
                split_desc = os.path.basename(this_shard_worker_parquet_files[0]).split('__')[0]
                # print("CCC", split_desc)
                num_splits = (int)(split_desc.split('-')[1].split('@')[1])
                # print("DDD", len(this_shard_worker_parquet_files), num_splits, type(num_splits))
                assert len(this_shard_worker_parquet_files) == num_splits
                for parquet_file in this_shard_worker_parquet_files:
                    # logging.info(f"Input parquet file: {parquet_file}")
                    split_desc = os.path.basename(parquet_file).split('__')[0]
                    # print("EEE", split_desc)
                    df = pd.read_parquet(parquet_file, engine=config['PARQUET_ENGINE'])
                    logging.info(f"Read Parquet file of length {len(df)} containing shards {sorted(list(df['shard_hex'].unique()))}")
                    pd.set_option('display.max_columns', None)
                    # logging.info(f"Input parquet file:\n{df}")
                    for shard_hex in assigned_shards:
                        # df_one_shard = df[df['shard_hex']==f'"{shard_hex}"']  # The shard hexes are quoted to force them to strings so they won't lose their leading 0s
                        df_one_shard = df[df['shard_hex']==f'_{shard_hex}']  # The shard hexes are underscore-prefixed to force them to strings so they won't lose their leading 0s
                        logging.info(f"Filtered Parquet file to shard {shard_hex}, yielding {len(df_one_shard)} rows")
                        df_one_shard.to_csv(f"{data_loc}{split_desc}__shard-{shard_hex}_.csv", index=False, header=False)
                        this_shard_input_files = glob.glob(f"{data_loc}split*shard-{shard_hex}_*.csv")
            else:  # Custom archiving
                archive_files = list(glob.glob(f"{data_loc}split*__shard_worker-{shard_worker_desc_file_hash}__archive.txt"))
                logging.info(f"Input archive files:\n  {'\n  '.join(sorted(archive_files)).strip()}" + '\n')
                # file_shard_filters = set([f"__shard-{shard}" for shard in assigned_shards])
                # logging.info(f"file_shard_filters: {file_shard_filters}")
                for archive_file in archive_files:
                    RAMDataPond.dearchive_file(data_loc, archive_file)#, file_shard_filters)

def merge_csv_splits():
    timestamps.append(("merge_csv_splits() start", default_timer()))

    merged_df = None
    for csv_i, shard_csv in enumerate(this_shard_input_files):
        timestamps.append(("input_file_loop_top", default_timer()))
        if csv_i <= 1:
            logging.info(f"\nMerging in file {shard_csv} ({csv_i+1} of {len(this_shard_input_files)})")
        shard_one_df = pd.read_csv(shard_csv, names=header, index_col=False)
        timestamps.append(("input_file_loop (read_input_file)", default_timer()))
        if csv_i <= 1:
            logging.info(f"  One shard len: {len(shard_one_df)}")

        # DEBUG
        if 'id_column' in config['DATA_CONFIG']:
            id_column = config['DATA_CONFIG']['id_column']
            # logging.info(f"id_column: {id_column}")
            if id_column is None:
                logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {header[0]}.")
        elif 'id_src' in config['DATA_CONFIG']:
            id_src = config['DATA_CONFIG']['id_src']
            logging.info(f"id_src: {id_src}")
            raise RuntimeError("id_src support (Wan-Qing's swc data) is not implemented yet")

        id_column_determined = id_column if id_column is not None else header[0]

        if csv_i <= 1:
            logging.info(f"  One input file duplicate id count: {shard_one_df[id_column_determined].duplicated().sum()}")
        timestamps.append(("input_file_loop (check_input_file_dups)", default_timer()))

        for row_idx, row in shard_one_df.iterrows():
            if row_idx >= 2:
                break
            if csv_i <= 1:
                logging.info(f"  One input file row {row_idx:>2}: {list(row)}")

        if merged_df is None:
            logging.info("  Initializing merged shard file from first shard file")
            merged_df = shard_one_df
        else:
            logging.info("  Merging in another shard file")
            merged_df = pd.concat([merged_df, shard_one_df])
        timestamps.append(("input_file_loop (concat_input_file)", default_timer()))

        # DEBUG
        logging.info(f"  Merged duplicate id count: {merged_df[id_column_determined].duplicated().sum()}")
        timestamps.append(("input_file_loop (check_merged_file_dups)", default_timer()))

    timestamps.append(("merge_csv_splits() end", default_timer()))

    return merged_df

def save_shard_data_as_csv(rows_this_level_df, subdir, shard_hex):
    if OUTPUT_STYLE == "capsule":
        filepath = f"{results_loc}annotations_one_shard__shard-{shard_hex}.csv"
    elif OUTPUT_STYLE == "results":
        filepath = f"{results_loc}{subdir}id_index_{shard_hex}.csv"
    logging.info(f"\nWriting shard's CSV file with {len(rows_this_level_df)} rows: {filepath}")
    rows_this_level_df.to_csv(filepath, index=False, header=False)

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

def convert_non_int_relation_via_enum_property(relation_val, relationship_column_name):
    found_it = False
    for prop_lbl, prop_info in config['DATA_CONFIG']['properties'].items():
        if prop_info['id'] == relationship_column_name:
            if prop_info['enum_values'] is not None:
                if relation_val in prop_info['enum_labels']:
                    enum_label_idx = prop_info['enum_labels'].index(relation_val)
                    enum_value = prop_info['enum_values'][enum_label_idx]
                else:
                    missing_enum_labels.add(relation_val)
                    enum_value = -1
                relation_val = enum_value
                found_it = True
    if not found_it:
        raise TypeError(f"Relation column '{relationship_column_name}' does not contain 'int' data and has no associated enumerated property description from which to derive an 'int' value.")
    return relation_val

def convert_relation_fields(row, data_relation_col_indices):
    # logging.info(f"convert_relation_fields(): {row}")

    # Convert relation fields to either an int or a list of ints
    relation_fields = {lbl: (col, row[col_idx]) for lbl, col, col_idx in data_relation_col_indices}
    for lbl, (col, relation_field_val) in relation_fields.items():
        relation_list = coerce_relation_field(relation_field_val)

        relation_fields[lbl] = []
        for relation_val in relation_list:
            # See note in Relation index builder (search for 'enumerated property')
            if not isinstance(relation_val, int):
                relation_val = convert_non_int_relation_via_enum_property(relation_val, col)
            relation_fields[lbl].append(relation_val)

        # # The column might already be an int, not a string, so check for that first
        # if isinstance(relation_field_val, int):
        #     relation_field = relation_field_val
        # else:
        #     relation_field_val = str(convert_non_int_relation_via_enum_property(relation_field_val, col))

        #     # Casting a float to an int won't raise an exception. We have to check for a float explicitly.
        #     if '.' in relation_field_val:
        #         raise ValueError(f"Relation field must be int or list of ints: {relation_field_val}")

        #     try:
        #         # Try casting the field as an int (we have already established it isn't a float above)
        #         relation_field = int(relation_field_val)
        #     except:
        #         try:
        #             # Try casting the field as an list (we have already established it isn't a float above)
        #             if relation_field_val[0] != '[':
        #                 relation_field_val = '[' + relation_field_val
        #             if relation_field_val[-1] != ']':
        #                 relation_field_val += ']'
        #             relation_field = ast.literal_eval(relation_field_val)
        #             if not isinstance(relation_field, list):
        #                 raise ValueError(f"Relation field must be int or list of ints: {relation_field_val}")

        #             # Ensure that every item in the list is an int
        #             for v in relation_field:
        #                 if not isinstance(v, int):
        #                     raise ValueError(f"Relation field must be int or list of ints: {relation_field_val}")
        #         except:
        #             raise ValueError(f"Relation field must be int or list of ints: {relation_field_val}")

        # relation_fields[lbl] = relation_field

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

def write_sharded_precomputed_file(writer, subdir, shard_hex):
    timestamps.append(("write_sharded_precomputed_file() start", default_timer()))

    # See description in 'conglomerate spatial index by shard' capsule
    USE_RAM_BUFFER = True
    logging.info(f"USE_RAM_BUFFER: {USE_RAM_BUFFER}")

    # Direct the writer to write its contents out
    if not USE_RAM_BUFFER:
        logging.info("Writing precomputed file without RAM buffer")
        writer._write_by_id_index(utilities.path_join(results_loc + subdir, "by_id"))
        timestamps.append(("write precomputed to file", default_timer()))
    else:
        logging.info("Writing precomputed file with RAM buffer")
        # The following lines are copied out of simple_writer_no_spatial_index.py and sharding.py
        dir_path = utilities.path_join(results_loc + subdir, "by_id")
        # logging.info(f"  dir_path: {dir_path}")
        dir_path = os.path.expanduser(dir_path)
        # logging.info(f"  dir_path: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        filepath = utilities.path_join(dir_path, f"{shard_hex}.shard")
        # logging.info(f"  filepath: {filepath}")
        file_path = os.path.expanduser(filepath)
        logging.info(f"  filepath: {filepath}")

        timestamps.append(("prepare precomputed writer filepath", default_timer()))

        file_buffer = io.BytesIO()
        with file_buffer as f_buf:
            writer_profile = writer._writef_by_id_index(f_buf, shard_num=int(shard_hex, 16))  # Remove 'shard_num' param to disable some debugging/validation tests
            timestamps.append(("write precomputed to buffer", default_timer()))

            with open(filepath, "wb") as f_disk:
                f_disk.write(file_buffer.getbuffer())
            timestamps.append(("write precomputed buffer to file", default_timer()))
            logging.info(f"Shard file size for shard {shard_hex}: {os.path.getsize(filepath)}")

    # Remove the extraneous files and move the target file
    if OUTPUT_STYLE == "results":
        os.makedirs(f"{results_loc}{subdir}", exist_ok=True)
    if config['ID_SHARDING']:
        id_files = list(glob.glob(f"{results_loc}{subdir}by_id/*.shard"))
        logging.info(f"ID files ({len(id_files)}):\n  {'\n  '.join(id_files)}\n")
        for id_file in id_files:
            id_filename = os.path.basename(id_file)
            if id_filename != f"{shard_hex}.shard":
                assert not USE_RAM_BUFFER
                os.remove(id_file)
            else:
                logging.info(f"Moving results file to correct subdirectory: {id_file}")
                if OUTPUT_STYLE == "capsule":
                    shutil.move(id_file, f"{results_loc}id_index_{shard_hex}.shard")
                elif OUTPUT_STYLE == "results":
                    shutil.move(id_file, f"{results_loc}{subdir}{shard_hex}.shard")
    else:
        id_files = list(glob.glob(f"{results_loc}{subdir}by_id/*"))
        logging.info(f"ID files ({len(id_files)}):\n  {'\n  '.join(id_files)}\n")
        for id_file in id_files:
            if OUTPUT_STYLE == "capsule":
                shutil.move(id_file, f"{results_loc}")
            elif OUTPUT_STYLE == "results":
                shutil.move(id_file, f"{results_loc}{subdir}")

    timestamps.append(("write_sharded_precomputed_file() end", default_timer()))

    return writer_profile

def calculate_annotation_vector(points):
    """
    At the time of this writing, vectors are only implemented against a single CSV field containing a semi-colon delimited list of comma-delimited points.
    """
    if len(points) == 0:
        return 0, 0, 0
    if len(points) == 1:
        return points[0][0], points[0][1], points[0][2]

    vx_sum, vy_sum, vz_sum = 0, 0, 0
    prev_pt = None
    for pt in points.values():
        if prev_pt:
            vx_sum += pt[0] - prev_pt[0]
            vy_sum += pt[1] - prev_pt[1]
            vz_sum += pt[2] - prev_pt[2]
        prev_pt = pt
    vx_mean = vx_sum / (len(points) - 1)
    vy_mean = vy_sum / (len(points) - 1)
    vz_mean = vz_sum / (len(points) - 1)

    return vx_mean, vy_mean, vz_mean

def build_annotation_description__one_annotation_per_row__multiple_points_per_row(row_i, row, columns, pt_positions, data_relation_col_indices, col_index_map):
    debug = False  # row_i < 1000

    id_column = config['DATA_CONFIG']['id_column']
    if id_column is None:
        # If no id column was provided, then it was explicitly added at the left end of the table earlier in the process
        id_column = columns[0]
    id_ = row[columns.index(id_column)]
    if row_i < 3:
        logging.info(f"build_anno_desc(): Row id: {id_}")

    relation_fields = convert_relation_fields(row, data_relation_col_indices)

    desc = {
        "id": id_,
        # If this line is reverted to needing data_property_col_indices again, it was previously passed in to this function
        "properties": {},  # {lbl: row[col_idx] for (lbl, col_idx) in data_property_col_indices},
        "relations": relation_fields,
    }

    # REF:
    # data_properties = config['DATA_CONFIG']['properties']
    # data_properties_cols = [(prop_lbl, prop_info['id']) for prop_lbl, prop_info in data_properties.items()]
    # data_property_col_indices = [(prop_col, col_index_map[prop_col]) for prop_lbl, prop_col in data_properties_cols]

    for prop_lbl, prop_info in config['DATA_CONFIG']['properties'].items():
        if debug:
            logging.info(f"\nProp: {prop_lbl} {prop_info}")

        prop_id = prop_lbl  # prop_info['id']
        col_idx = col_index_map[prop_info['id']] if prop_info['id'] is not None else None
        field = row[col_idx] if col_idx is not None else None

        if debug:
            logging.info(f"Anno prop field: {field}")

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
        desc["points"] = list(pt_positions.values())

    if debug:
        logging.info(f"Anno desc: {desc}\n")

    return desc

def read_single_field_point_list(row):
    """
    Duplicated in spatial index pipeline
    """
    points = row[config['DATA_CONFIG']['columns'].index('Points')]
    points = points.split(';')
    points = [pt.split(',') for pt in points]
    points = [[float(v) for v in pt] for pt in points]
    points = {f"Point_{i:0>12}": pt for i, pt in enumerate(points)}
    return points

def save_csv_annotations_as_precomputed(df, data_properties, data_relation_col_indices, col_index_map):
    columns = list(df.columns)
    assert columns == header
    header_reverse_map = {col: i for i, col in enumerate(header)}
    logging.info(f"header_reverse_map: {header_reverse_map}")

    spatial_pt_columns = config['DATA_CONFIG']['spatial_pt_columns']

    annotation_descriptions = []
    # for row_i, row in df.iterrows():  # Pandas Dataframe iteration, slower than itertuples()
    for row_i, row in enumerate(df.itertuples(index=False)):  # Pandas Dataframe iteration, faster than iterrows()
        # if row_i % 10 == 0 and row_i < 100:
        #     logging.info(f"Row {row_i:10}: {row}")
        if isinstance(spatial_pt_columns, dict):
            pt_positions = {
                pt_desc: [
                        float(row[header_reverse_map[pt_pos['x']]]),
                        float(row[header_reverse_map[pt_pos['y']]]),
                        float(row[header_reverse_map[pt_pos['z']]]),
                    ] \
                    for pt_desc, pt_pos in spatial_pt_columns.items()
            }
        elif spatial_pt_columns == "single_field_list":
            pt_positions = read_single_field_point_list(row)
        annotation_description = build_annotation_description__one_annotation_per_row__multiple_points_per_row(row_i, row, columns, pt_positions, data_relation_col_indices, col_index_map)
        annotation_descriptions.append(annotation_description)

    timestamps.append(("gather annotation_descriptions from table", default_timer()))

    writer = simple_writer_no_spatial_indexing.SimpleWriter("LINE")

    if config['ID_SHARDING']:
        writer.by_id_sharding = anno.ShardingSpec(hash=config['ID_SHARDING_HASH'], preshift_bits=config['ID_PRESHIFT_BITS'], shard_bits=config['ID_SHARDING_BITS'], minishard_bits=config['ID_MINISHARDING_BITS'])  # Comment out this line (or don't conditionally don't call it) to generate a non-sharded id index, but be aware that every annotation will get a separate file!
    elif len(annotation_descriptions) > 100:
        raise ValueError("Sharding is disabled for the ID index, but the number of annotations is high. This will produce a lot of individual files.")

    # Define properties
    data_properties_2 = {}
    for property_name, property_info in data_properties.items():
        if property_name != 'vector':
            data_properties_2[property_name] = property_info
        else:
            data_properties_2['vector_x'] = {
                'id': 'vector_x',
                'type': 'float32',
                'enum_labels': None,
                'enum_values': None,
            }
            data_properties_2['vector_y'] = {
                'id': 'vector_y',
                'type': 'float32',
                'enum_labels': None,
                'enum_values': None,
            }
            data_properties_2['vector_z'] = {
                'id': 'vector_z',
                'type': 'float32',
                'enum_labels': None,
                'enum_values': None,
            }

    for property_name, property_info in data_properties_2.items():
        property_spec = anno.PropertySpec(property_name, property_info['type'], property_name, property_info['enum_values'], property_info['enum_labels'])
        logging.info(f"Adding PropertySpec to writer: {property_spec}")
        writer.property_specs.append(property_spec)

    # Define relationships (yes, id indexing needs this)
    for relation in config['DATA_CONFIG']['relations']:
        relationship = anno.Relationship(relation)
        logging.info(f"Adding Relationship to writer: {relationship}")
        writer.relationships.append(relationship)

    timestamps.append(("init writer", default_timer()))

    for i, annotation_description in enumerate(annotation_descriptions):
        annotation = build_annotation(annotation_description)
        if i < 3:
            logging.info(f"Annotation {i:10}: {annotation}")
        writer.annotations.append(annotation)

    timestamps.append(("append annotation_description annotations to writer", default_timer()))

    return writer

def save_csv_shard_data_as_precomputed(df, subdir, shard_hex, data_properties, data_relations, data_relation_col_indices):
    timestamps.append(("save_csv_shard_data_as_precomputed() start", default_timer()))

    logging.info(f"\nWriting precomputed file for shard hex {shard_hex} to: {results_loc + subdir}")

    assert list(df.columns) == header

    col_index_map = {col_name: i for i, col_name in enumerate(df.columns)}

    # data_properties = config['DATA_CONFIG']['properties']
    # data_properties_cols = [(prop_lbl, prop_info['id']) for prop_lbl, prop_info in data_properties.items()]
    # data_property_col_indices = [(prop_col, col_index_map[prop_col]) for prop_lbl, prop_col in data_properties_cols]

    # data_relations = config['DATA_CONFIG']['relations']
    # data_relations_cols = [(rel_lbl, rel_info['id']) for rel_lbl, rel_info in data_relations.items()]
    # data_relation_col_indices = [(rel_lbl, rel_col, col_index_map[rel_col]) for rel_lbl, rel_col in data_relations_cols]

    writer = save_csv_annotations_as_precomputed(df, data_properties, data_relation_col_indices, col_index_map)

    writer_profile = write_sharded_precomputed_file(writer, subdir, shard_hex)

    timestamps.append(("save_csv_shard_data_as_precomputed() end", default_timer()))

    return writer_profile

def build__annotation_description__one_annotation_per_file__one_point_per_row(id_, df, columns, pt_columns, data_properties_cols, data_relations):
    xs = df[pt_columns['x']]
    ys = df[pt_columns['y']]
    zs = df[pt_columns['z']]
    pt_positions = []
    for x, y, z in zip(xs, ys, zs):
        pt_positions.append([x, y, z])

    prop_vals = {}
    for property_lbl, property_col in data_properties_cols:
        prop_vals[property_col] = df[property_col]

    relation_vals = []
    for relation_col in data_relations:
        relation_vals = df[relation_col]

    if 'point_annotation_config' in config['DATA_CONFIG']:
        raise RuntimeError("Not implemented yet")
    elif 'line_annotation_config' in config['DATA_CONFIG']:
        raise RuntimeError("Not implemented yet")
    elif 'polyline_annotation_config' in config['DATA_CONFIG']:
        return {
            "id": id_,
            "points": pt_positions,
            "properties": prop_vals,
            "relations": relation_vals,
        }

def save_swc_annotations_as_precomputed(data_properties, data_properties_cols, data_relations):
    header_reverse_map = {col: i for i, col in enumerate(header)}
    logging.info(f"header_reverse_map: {header_reverse_map}")
    pt_column_label = config['DATA_CONFIG']['polyline_annotation_config']['pt_column_label']
    spatial_pt_columns = config['DATA_CONFIG']['spatial_pt_columns']
    pt_columns = spatial_pt_columns[pt_column_label]

    annotation_descriptions = []

    for fi, shard_swc in enumerate(this_shard_input_files):
        filename = os.path.basename(shard_swc)
        id_ = int(filename[filename.rindex('__')+2:filename.rindex('.')])

        if fi == 0:
            logging.info(f"Input file {fi+1} of {len(this_shard_input_files)}, basename, id:\n  {shard_swc}    {filename}    {id_}")
        df = pd.read_csv(shard_swc, index_col=False)
        if fi == 0:
            logging.info(f"  SWC columns: {df.columns}")

        columns = list(df.columns)
        assert columns == header

        annotation_description = build__annotation_description__one_annotation_per_file__one_point_per_row(id_, df, columns, pt_columns, data_properties_cols, data_relations)
        annotation_descriptions.append(annotation_description)

    timestamps.append(("gather annotation_descriptions from table", default_timer()))

    writer = simple_writer_no_spatial_indexing.SimpleWriter("POLYLINE")

    if config['ID_SHARDING']:
        writer.by_id_sharding = anno.ShardingSpec(hash=config['ID_SHARDING_HASH'], preshift_bits=config['ID_PRESHIFT_BITS'], shard_bits=config['ID_SHARDING_BITS'], minishard_bits=config['ID_MINISHARDING_BITS'])  # Comment out this line (or don't conditionally don't call it) to generate a non-sharded id index, but be aware that every annotation will get a separate file!
    elif len(annotation_descriptions) > 100:
        raise ValueError("Sharding is disabled for the ID index, but the number of annotations is high. This will produce a lot of individual files.")

    # Define properties
    for property_name, property_info in data_properties.items():
        writer.property_specs.append(
            anno.PropertySpec(property_name, property_info['type'], property_name, property_info['enum_values'], property_info['enum_labels']))

    # Define relationships (yes, id indexing needs this)
    for relation in config['DATA_CONFIG']['relations']:
        writer.relationships.append(anno.Relationship(relation))

    timestamps.append(("init writer", default_timer()))

    for i, annotation_description in enumerate(annotation_descriptions):
        annotation = build_annotation(annotation_description)
        writer.annotations.append(annotation)

    timestamps.append(("append annotation_description annotations to writer", default_timer()))

    return writer

def save_swc_shard_data_as_precomputed(subdir, shard_hex, data_properties, data_properties_cols, data_relations):
    timestamps.append(("save_swc_shard_data_as_precomputed() start", default_timer()))

    logging.info(f"\nWriting precomputed file to: {results_loc + subdir}")

    # col_index_map = {col_name: i for i, col_name in enumerate(header)}
    # data_properties_cols = [v['id'] for k, v in data_properties.items()]
    # data_property_by_col_idx = {col: col_index_map[col] for col in data_properties_cols}
    # data_relations = config['DATA_CONFIG']['relations']
    # data_relation_by_col_idx = {col: col_index_map[v['id']] for col, v in data_relations.items()}

    writer = save_swc_annotations_as_precomputed(data_properties, data_properties_cols, data_relations)

    writer_profile = write_sharded_precomputed_file(writer, subdir, shard_hex)

    timestamps.append(("save_swc_shard_data_as_precomputed() end", default_timer()))

    return writer_profile

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("GENERATE ID INDEX SHARDS")

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

    # Pick one: changing this requires altering the pipeline topology
    # OUTPUT_STYLE = "capsule"  # Pipeline must connect this capsule to the 'reorganize directory structure' capsule with 'Collect' type
    OUTPUT_STYLE = "results"  # Pipeline must connect this capsule to the 'results'

    if OUTPUT_STYLE == "capsule":
        subdir_shard = f"id_index/"
        subdir_csv = f"id_index_csv/"
    elif OUTPUT_STYLE == "results":
        subdir_shard = f"by_id/"
        subdir_csv = f"by_id_csv/"
    os.makedirs(f"{results_loc}{subdir_shard}", exist_ok=True)
    SAVE_SHARD_DATA_AS_CSV = False
    if SAVE_SHARD_DATA_AS_CSV:
        os.makedirs(f"{results_loc}{subdir_csv}", exist_ok=True)

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

    EXPECT_SHARD_DIRS = False  # If not, then expect a flat list of files

    missing_enum_labels = set()

    if config['ID_INDEX_ENABLED']:
        timestamps = []
        timestamps.append(("start", default_timer()))

        writer_profiles = []

        data_loc_contents = sorted(os.listdir(data_loc))
        data_loc_contents = [v for v in data_loc_contents if "placeholder" not in v]
        logging.info(f"{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):")
        logging.info('  ' + '\n  '.join(data_loc_contents[:30]).strip() + '\n')

        if not EXPECT_SHARD_DIRS:
            # The shard worker file is only used when the upstream capsule is "build id index" (7252776)
            shard_worker_desc_files = list(glob.glob(f"{data_loc}shard_worker*txt"))
            assert len(shard_worker_desc_files) == 1
            shard_worker_desc_file_path = shard_worker_desc_files[0]
            shard_worker_desc_filename = os.path.basename(shard_worker_desc_file_path)
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

        # Detect an empty input directory or the presence of the no-op file, indicating that this capsule isn't being used in the current pipeline.
        # data_loc_contents_set = set(data_loc_contents)
        # if data_loc_contents_set == set(["job_config.py"]) \
        #     or data_loc_contents_set == set(["job_config.py", "no_op.txt"]):
        #     logging.info("Empty data input directory or no-op file found. Presumably, this capsule isn't being used in the current pipeline.")
        #     with open(f"{results_loc}no_op.txt", 'w') as f:
        #         f.write("no_op\n""Empty input directory or no-op file found. Presumably, this capsule isn't being used in the current pipeline.")
        # else:
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
                    download_files_from_gcp(f"{config['TIMESTAMP']}/id_index", data_loc, filename_filter, config['GCP_BUCKET'], config['GCP_SCRATCH_BLOB_PATH'])#, dryrun=True)
                    ets = default_timer() - st
                    logging.info(f"\nGCP download elapsed time: {seconds_to_hms(ets)}")

                    logging.info(f"\nMoving GCP downloads to {data_loc}")
                    downloaded_files = glob.glob(f"{data_loc}{config['GCP_SCRATCH_BLOB_PATH']}/{config['TIMESTAMP']}/id_index/*")
                    for downloaded_file in downloaded_files:
                        os.rename(downloaded_file, f"{data_loc}{os.path.basename(downloaded_file)}")
                elif config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "aws":
                    logging.info("\nDownloading files from Amazon storage")
                    download_folder_relative_path = f"{config['TIMESTAMP']}/id_index/"
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
        analyze_memory_usage()

        timestamps.append(("extract_input_files", default_timer()))

        input_extension = "csv"
        if config['DATA_CONFIG']['structure'] == "one_annotation_per_file__one_point_per_row":
            input_extension = "swc"  # For now, just support Wan-Qing's SWC data
            raise RuntimeError("One-annotation-per-file, one-point-per-row SWC support is on hold, potentially indefinitely. Instead, the data spliter converts such data to a CSV file of the original format: one-annotation-per-row, multi-points-in-single-field-list.")

        for shard_hex in assigned_shards:
            logging.info("*" * 100)
            timestamps.append(("shard_loop_top", default_timer()))

            if EXPECT_SHARD_DIRS:
                # Assume the upstream capsule is "group shard splits" (4496009), which outputs shard directories containing individual files
                shard_dir = f"{data_loc}shard-{shard_hex}/"
                logging.info(f"Shard dir, shard hex: {os.path.basename(shard_dir)} {shard_hex}")
                this_shard_input_files = glob.glob(f"{shard_dir}*.{input_extension}")
            else:
                # Assume the upstream capsule is "build id index" (7252776), which outputs individual files
                this_shard_input_files = glob.glob(f"{data_loc}split*shard-{shard_hex}_*.{input_extension}")

            logging.info(f"Input files for shard {shard_hex}:\n  {'\n  '.join(this_shard_input_files)}\n")

            col_index_map = {col_name: i for i, col_name in enumerate(config['DATA_CONFIG']['columns'])}

            data_properties = config['DATA_CONFIG']['properties']
            data_properties_cols = [(prop_lbl, prop_info['id']) for prop_lbl, prop_info in data_properties.items()]
            # data_property_col_indices = [(prop_col, col_index_map[prop_col]) for prop_lbl, prop_col in data_properties_cols]

            data_relations = config['DATA_CONFIG']['relations']
            data_relations_cols = [(rel_lbl, rel_info['id']) for rel_lbl, rel_info in data_relations.items()]
            data_relation_col_indices = [(rel_lbl, rel_col, col_index_map[rel_col]) for rel_lbl, rel_col in data_relations_cols]

            if this_shard_input_files:
                if input_extension == "csv":
                    logging.info("Archives are expected to be CSV files")
                    merged_df = merge_csv_splits()

                    timestamps.append(("merge_splits", default_timer()))

                    logging.info(f"Merged shard len: {len(merged_df)}")

                    # DEBUG
                    if 'id_column' in config['DATA_CONFIG']:
                        id_column = config['DATA_CONFIG']['id_column']
                        # logging.info(f"id_column: {id_column}")
                        if id_column is None:
                            logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {header[0]}.")
                    elif 'id_src' in config['DATA_CONFIG']:
                        raise RuntimeError("id_src support (Wan-Qing's swc data) is not implemented yet")
                    logging.info(f"Final merged duplicate id count: {merged_df[id_column if id_column is not None else header[0]].duplicated().sum()}")

                    writer_profile = save_csv_shard_data_as_precomputed(merged_df, subdir_shard, shard_hex, data_properties, data_relations, data_relation_col_indices)
                    writer_profiles.append((shard_hex, writer_profile))

                    timestamps.append(("save_as_precomputed", default_timer()))

                    if SAVE_SHARD_DATA_AS_CSV:
                        save_shard_data_as_csv(merged_df, subdir_csv, shard_hex)

                        timestamps.append(("save_as_csv", default_timer()))
                elif input_extension == "swc":
                    logging.info("Archives are expected to be SWC files")
                    raise RuntimeError("One-annotation-per-file, one-point-per-row SWC support is on hold, potentially indefinitely. Instead, the data spliter converts such data to a CSV file of the original format: one-annotation-per-row, multi-points-in-single-field-list.")

                    writer_profile = save_swc_shard_data_as_precomputed(subdir_shard, shard_hex, data_properties, data_properties_cols, data_relations)
                    writer_profiles.append((shard_hex, writer_profile))

                    timestamps.append(("save_as_precomputed", default_timer()))

            else:
                logging.info(f"No input files found for shard {shard_hex}")

            analyze_memory_usage()
            timestamps.append(("shard_loop_bottom", default_timer()))

        logging.info("\n" + "* " * 50 + "\n")

        if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
            if config['UPLOAD_RESULTS_TO_GCP']:
                logging.info("\nUploading files to Google Storage")
                st = default_timer()
                upload_directory_to_gcp(results_loc, "by_id/", config["TIMESTAMP"], config['GCP_BUCKET'], config['GCP_RESULTS_BLOB_PATH'])#, dryrun=True)
                ets = default_timer() - st
                logging.info(f"GCP upload elapsed time: {seconds_to_hms(ets)}")

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
                # doesn't make the capsule run any faster...I think! I'm not sure.
                logging.info(f"\nDeleting result files after uploading to GCP")
                if os.path.exists(f"{results_loc}by_id"):
                    shutil.rmtree(f"{results_loc}by_id")
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

    os.makedirs(f"{results_loc}by_id/", exist_ok=True)
    finalize_results(f"{results_loc}by_id/")

    analyze_memory_usage()

logging.info("\nDone")
process_running_time()
