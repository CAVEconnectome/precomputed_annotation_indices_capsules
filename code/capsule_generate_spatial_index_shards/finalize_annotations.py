import sys
import logging
import os
import io
import glob
import math
from timeit import default_timer
from collections import Counter, defaultdict
import pandas as pd
import csv
import random
import shutil
import tarfile

from shared.google_storage import *
 
import shared.simple_writer_no_spatial_indexing as simple_writer_no_spatial_indexing
import shared.annotations as anno
import shared.sharding as sharding
import shared.utilities as utilities

from shared.shard_reader import *
from shared.util import *
from shared.aws_storage import *

data_loc, results_loc, config, timestamps, dimensions, missing_enum_labels = None, None, None, None, None, None

# Pick one: changing this requires altering the pipeline topology
# OUTPUT_STYLE = "capsule"  # Pipeline must connect this capsule to the 'reorganize directory structure' capsule with 'Collect' type
# OUTPUT_STYLE = "results"  # Pipeline must connect this capsule to the 'results'
OUTPUT_STYLE = "results_for_ng"  # Pipeline must connect this capsule to the 'results'. The resulting layout will drop into place for Neuroglancer without any further moving around.

def read_shardworker_file():
    # Read the shard worker description
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

    return shard_worker_desc_file_hash, assigned_shards

def download_data_from_bucket(shard_worker_desc_file_hash):
    # Download the input data from external storage (opposed to receiving it through Code Ocean)
    if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
        if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] != "codeocean":
            st = default_timer()
            filename_filter = f"shard_worker-{shard_worker_desc_file_hash}"
            logging.info(f"A filename_filter: {filename_filter}")
            if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "gcp":
                raise RuntimeError("GCP bucket no longer supported due to possible financial cost if done incorrectly!")
                logging.info("\nDownloading files from Google storage")
                download_files_from_gcp(f"{config['TIMESTAMP']}/spatial_index/regrouper", data_loc, filename_filter, config['GCP_BUCKET'], config['GCP_SCRATCH_BLOB_PATH'])#, dryrun=True)
                ets = default_timer() - st
                logging.info(f"\nGCP download elapsed time: {seconds_to_hms(ets)}")

                logging.info(f"\nMoving GCP downloads to {data_loc}")
                downloaded_files = glob.glob(f"{data_loc}{config['GCP_SCRATCH_BLOB_PATH']}/{config['TIMESTAMP']}/spatial_index/regrouper/*")
                for downloaded_file in downloaded_files:
                    os.rename(downloaded_file, f"{data_loc}{os.path.basename(downloaded_file)}")
            elif config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "aws":
                logging.info("\nDownloading files from Amazon storage")
                download_folder_relative_path = f"{config['TIMESTAMP']}/spatial_index/regrouper/"
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

def save_shard_data_as_csv(table, header_str, subdir, tree_level, shard_hex):
    # timestamps.append(("save_shard_data_as_csv() top", default_timer()))

    filepath = f"{results_loc}{subdir}annotations_one_shard__treelevel-{tree_level:02}__shard-{shard_hex}.csv"
    if isinstance(table, pd.DataFrame):
        logging.info(f"\nWriting shard's CSV file with {len(table)} rows: {filepath}")
        table.to_csv(filepath, index=False, header=False)
    elif isinstance(table, str):
        with open(filepath, 'w') as f:
            f.write(header_str + '\n')
            f.write(table)
    
    # timestamps.append(("save_shard_data_as_csv() bottom", default_timer()))

def get_shard_hex(tree_level, tree_level_cell_id, verbose=False):
    """
    See implementation in 'generate spatial index config' capsule for description
    """
    grid_dim = 2 ** tree_level
    grid_shape = (grid_dim, grid_dim, grid_dim)
    morton_code = utilities.compressed_morton_code(tree_level_cell_id, grid_shape)
    sharding_spec = anno.ShardingSpec(
        hash=config['SPATIAL_SHARDING_HASH'],
        preshift_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['preshift_bits'],
        shard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits'],
        minishard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['minishard_bits'])
    shard_num = sharding_spec.get_shard_number(morton_code)
    shard_hex = sharding.get_shard_hex(shard_num, config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits'])
    if verbose:
        logging.info(f"get_shard_hex({tree_level}, {tree_level_cell_id}) -> mc {morton_code}, shard_num {shard_num}, shard_hex {shard_hex}")
    
    return shard_hex

def confirm_point_annotation_location_treecellid_shard(point_annotation, tree_level, cell_bounds_low, cell_bounds_high, file_shard_hex):
    """
    This probably slows this process down significantly. Remove it once correct behavior is confirmed.
    """
    position = point_annotation["position"]
    tree_level_cell_id = [int(v) for v in point_annotation["treecell_index"].split('_')]

    position_in_bounds = position[0] >= cell_bounds_low[0] and position[0] <= cell_bounds_high[0] \
        and position[1] >= cell_bounds_low[1] and position[1] <= cell_bounds_high[1] \
        and position[2] >= cell_bounds_low[2] and position[2] <= cell_bounds_high[2]
    assert position_in_bounds

    shard_hex = get_shard_hex(tree_level, tree_level_cell_id)
    if shard_hex != file_shard_hex:
        print("clalts ERROR!", shard_hex, file_shard_hex, tree_level, tree_level_cell_id)
        get_shard_hex(tree_level, tree_level_cell_id, True)
    assert shard_hex == file_shard_hex

def confirm_line_annotation_location_treecellid_shard(line_annotation, tree_level, cell_bounds_low, cell_bounds_high, file_shard_hex):
    """
    This probably slows this process down significantly. Remove it once correct behavior is confirmed.
    """
    start = line_annotation["start"]
    end = line_annotation["end"]
    tree_level_cell_id = [int(v) for v in line_annotation["treecell_index"].split('_')]

    start_in_bounds = start[0] >= cell_bounds_low[0] and start[0] <= cell_bounds_high[0] \
        and start[1] >= cell_bounds_low[1] and start[1] <= cell_bounds_high[1] \
        and start[2] >= cell_bounds_low[2] and start[2] <= cell_bounds_high[2]
    end_in_bounds = end[0] >= cell_bounds_low[0] and end[0] <= cell_bounds_high[0] \
        and end[1] >= cell_bounds_low[1] and end[1] <= cell_bounds_high[1] \
        and end[2] >= cell_bounds_low[2] and end[2] <= cell_bounds_high[2]
    assert start_in_bounds or end_in_bounds

    shard_hex = get_shard_hex(tree_level, tree_level_cell_id)
    if shard_hex != file_shard_hex:
        print("clalts ERROR!", shard_hex, file_shard_hex, tree_level, tree_level_cell_id)
        get_shard_hex(tree_level, tree_level_cell_id, True)
    assert shard_hex == file_shard_hex

def confirm_polyline_annotation_location_treecellid_shard(polyline_annotation, tree_level, cell_bounds_low, cell_bounds_high, file_shard_hex):
    """
    This probably slows this process down significantly. Remove it once correct behavior is confirmed.
    """
    # raise RuntimeError("Not implemented yet")

def validate_shard_file(filepath, tree_level, property_specs):
    shardfile_reader = ShardFileReader(
        filepath=filepath,
        preshift_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['preshift_bits'],
        shard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits'],
        minishard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['minishard_bits'],
        property_specs=property_specs,
        relationships=None,
    )
    shardfile_reader.read()
    logging.info(f"Shard file is valid: {filepath}")

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

def convert_relation_fields(relation_fields):
    # logging.info(f"convert_relation_fields(): {relation_fields}")

    # Convert relation fields to either an int or a list of ints
    relation_fields_out = {}
    for lbl, (col, relation_field_str) in relation_fields.items():
        relation_list = coerce_relation_field(relation_field_str)

        relation_fields_out[lbl] = []
        for relation_val in relation_list:
            # See note in Relation index builder (search for 'enumerated property')
            if not isinstance(relation_val, int):
                relation_val = convert_non_int_relation_via_enum_property(relation_val, relationship_column_name)
            relation_fields_out[lbl].append(relation_val)
        
        # # Casting a float to an int won't raise an exception. We have to check for a float explicitly.
        # if '.' in relation_field_str:
        #     raise ValueError(f"Relation field must be int or list of ints: {relation_field_str}")
        
        # try:
        #     # Try casting the field as an int (we have already established it isn't a float above)
        #     relation_field = int(relation_field_str)
        # except:
        #     try:
        #         relation_field_str = str(convert_non_int_relation_via_enum_property(relation_field_str, col))
                
        #         # Try casting the field as a list (we have already established it isn't a float above)
        #         if relation_field_str[0] != '[':
        #             relation_field_str = '[' + relation_field_str
        #         if relation_field_str[-1] != ']':
        #             relation_field_str += ']'
        #         relation_field = ast.literal_eval(relation_field_str)
        #         if not isinstance(relation_field, list):
        #             raise ValueError(f"Relation field must be int or list of ints: {relation_field_str}")
                
        #         # Ensure that every item in the list is an int
        #         for v in relation_field:
        #             if not isinstance(v, int):
        #                 raise ValueError(f"Relation field must be int or list of ints: {relation_field_str}")
        #     except:
        #         raise ValueError(f"Relation field must be int or list of ints: {relation_field_str}")
        
        # relation_fields_out[col] = relation_field
    
    return relation_fields_out

def calculate_annotation_vector(points):
    """
    At the time of this writing, vectors are only implemented against a single CSV field containing a semi-colon delimited list of comma-delimited points.
    """
    vx_sum, vy_sum, vz_sum = 0, 0, 0
    prev_pt = None
    for pt in points.values():
        if prev_pt:
            vx_sum += pt[0] - prev_pt[0]
            vy_sum += pt[1] - prev_pt[1]
            vz_sum += pt[2] - prev_pt[2]
        prev_pt = pt
    vx_mean = vx_sum - (len(points) - 1)
    vy_mean = vy_sum - (len(points) - 1)
    vz_mean = vz_sum - (len(points) - 1)

    return vx_mean, vy_mean, vz_mean

def build_annotation_description_from_row_tuple__one_annotation_per_row__multiple_points_per_row(row, columns, pt_positions, data_property_by_col_idx, data_relation_by_col_idx):
    relation_fields = {col: (col, row[col_idx]) for col, col_idx in data_relation_by_col_idx.items()}
    relation_fields = convert_relation_fields(relation_fields)
    
    id_column = config['DATA_CONFIG']['id_column']
    # logging.info(f"id_column: {id_column}")
    if id_column is not None:
        id_column_idx = columns.index(id_column)
    else:
        id_column_idx = 0
        logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {columns[0]}.")
    logging.info(f"id_column_idx: {id_column_idx}")
    
    desc = {
        "id": row[id_column_idx],
        "treecell_index": row[treecell_index],  # [int(v) for v in row.treecell_index.split('_')],
        "properties": {},  # {col: row[col_idx] for col, col_idx in data_property_by_col_idx.items()},
        "relations": relation_fields,
    }

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
    
    # logging.info(f"Annotation description: {desc}")
    
    return desc

def build_annotation_description_from_str__one_annotation_per_row__multiple_points_per_row(id_, pt_positions, treecell_index, properties, relation_fields):
    relation_fields = convert_relation_fields(relation_fields)
    
    desc = {
        "id": id_,
        "treecell_index": treecell_index,
        "properties": properties,
        "relations": relation_fields,
    }
    
    if 'point_annotation_config' in config['DATA_CONFIG']:
        desc["position"] = pt_positions[config['DATA_CONFIG']["point_annotation_config"]["pt_column_label"]]
    elif 'line_annotation_config' in config['DATA_CONFIG']:
        desc["start"] = pt_positions[config['DATA_CONFIG']["line_annotation_config"]["start_pt_column_label"]]
        desc["end"] = pt_positions[config['DATA_CONFIG']["line_annotation_config"]["end_pt_column_label"]]
    elif 'polyline_annotation_config' in config['DATA_CONFIG']:
        desc["points"] = list(pt_positions.values())
    
    # logging.info(f"Annotation description: {desc}")
    
    return desc

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

def save_point_as_precomputed(table, columns, cell_bounds_low, cell_bounds_high, tree_level, shard_hex, data_properties, data_property_by_col_idx, data_relations, data_relation_by_col_idx, debug, debug2):
    raise RuntimeError("Not implemented yet")

def read_single_field_point_list(row):
    """
    Duplicated in ID index pipeline
    """
    points = row[config['DATA_CONFIG']['columns'].index('Points')]
    points = points.split(';')
    points = [pt.split(',') for pt in points]
    points = [[float(v) for v in pt] for pt in points]
    points = {f"Point_{i:0>12}": pt for i, pt in enumerate(points)}
    return points

def calculate_annotation_vector(row):
    """
    At the time of this writing, vectors are only implemented against a single CSV field containing a semi-colon delimited list of comma-delimited points.
    """
    points = read_single_field_point_list(row)
    if len(points) == 0:
        return None

    vx_sum, vy_sum, vz_sum = 0, 0, 0
    prev_pt = None
    for pt in points.values():
        if prev_pt:
            vx_sum += pt[0] - prev_pt[0]
            vy_sum += pt[1] - prev_pt[1]
            vz_sum += pt[2] - prev_pt[2]
        prev_pt = pt
    vx_mean = vx_sum - (len(points) - 1)
    vy_mean = vy_sum - (len(points) - 1)
    vz_mean = vz_sum - (len(points) - 1)

    return vx_mean, vy_mean, vz_mean

def save_annotations_as_precomputed(table, columns, cell_bounds_low, cell_bounds_high, tree_level, shard_hex, data_properties, data_property_by_col_idx, data_relations, data_relation_by_col_idx, debug, debug2):
    timestamps.append(("save_annotations_as_precomputed() top", default_timer()))

    spatial_pt_columns = config['DATA_CONFIG']['spatial_pt_columns']
    
    # I believe we don't care about size-restricting shard files. We are presuming the size-restrictions on the tree cells properly restrict the resulting shard files.
    # So, we don't need the following test, but I'm leaving in place for now, for clarity of the circumstance the test represents (keeping files to a reasonable size).
    # if isinstance(table, pd.DataFrame):
    #     if len(table) > max_data_rows_per_tree_cell:
    #         raise ValueError(f"Tree cell will contain too many data rows: {len(table)} > {max_data_rows_per_tree_cell}. Increase configured 'TREE_LEVELS' and try again.")

    annotation_descriptions = []
    if isinstance(table, pd.DataFrame):
        assert False  # I don't think this code is used anymore
        # for row_i, row in table.iterrows():  # Slower than itertuples()
        for row in table.itertuples():  # Faster than iterrows()
            # TODO: I DON'T THINK THE ROW ACCESSES BY COLUMN NAME IN THE CODE BELOW ARE GOING TO WORK, BUT CURRENT USAGE ONLY HITS THE 'STR' CASE, BELOW, NOT THIS DATAFRAME CASE.
            pt_positions = {
                pt_desc: [float(row[pt_pos['x']]), float(row[pt_pos['y']]), float(row[pt_pos['z']])] \
                    for pt_desc, pt_pos in spatial_pt_columns.items()
            }
            annotation_desc = build_annotation_description_from_row_tuple__one_annotation_per_row__multiple_points_per_row(row, columns, pt_positions, data_property_by_col_idx, data_relation_by_col_idx)
            annotation_descriptions.append(annotation_desc)
    elif isinstance(table, str):
        id_column_idx = None
        if 'id_column' in config['DATA_CONFIG']:
            id_column = config['DATA_CONFIG']['id_column']
            # logging.info(f"id_column: {id_column}")
            if id_column is not None:
                id_column_idx = columns.index(id_column)
            else:
                id_column_idx = 0
                logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {columns[0]}.")
            logging.info(f"id_column_idx: {id_column_idx}")
        elif 'id_src' in config['DATA_CONFIG']:
            raise RuntimeError("id_src support (Wan-Qing's swc data) is not implemented yet")

        relation_col_indices = {k: (v['id'], columns.index(v['id'])) for k, v in data_relations.items()}
        treecell_index_col_idx = columns.index('treecell_index')
        
        if config['DATA_CONFIG']['structure'] == 'one_annotation_per_row__multiple_points_per_row':
            assert isinstance(spatial_pt_columns, dict)

            spatial_pt_col_idxs = {}
            for pt_desc, pt_pos in spatial_pt_columns.items():
                spatial_pt_col_idxs[pt_desc] = [
                    columns.index(pt_pos['x']),
                    columns.index(pt_pos['y']),
                    columns.index(pt_pos['z']),
                ]
        elif config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row_in_one_field":
            assert spatial_pt_columns == "single_field_list"
        
        col_index_map = {col_name: i for i, col_name in enumerate(columns)}

        lines = table.strip().split('\n')
        logging.info(f"Num merged rows in shard: {len(lines)} (from table of len {len(table):,} B)")
        treecell_index_counts = Counter()
        for line_i, line in enumerate(lines):
            if line:
                if line_i <= 1:
                    logging.info(f"Line {line_i}: {line}")

                # fields = line.split(',')
                reader = csv.reader(io.StringIO(line))
                fields = next(reader)

                if line_i <= 1:
                    logging.info(f"fields: {fields}")
                try:
                    id_ = int(fields[id_column_idx])
                except ValueError as e:
                    logging.info(f"CCC {e}\n{line}\n")
                
                
                if isinstance(spatial_pt_columns, dict):
                    pt_positions = {
                        pt_desc: [float(fields[pt_x_col_idx]), float(fields[pt_y_col_idx]), float(fields[pt_z_col_idx])] \
                            for pt_desc, [pt_x_col_idx, pt_y_col_idx, pt_z_col_idx] in spatial_pt_col_idxs.items()
                    }
                elif spatial_pt_columns == "single_field_list":
                    pt_positions = read_single_field_point_list(fields)

                treecell_index = fields[treecell_index_col_idx] if treecell_index_col_idx < len(fields) else None
                if treecell_index not in treecell_index_counts and len(treecell_index_counts) < 3:
                    logging.info(f"New treecell_index (only first 3 are shown): {treecell_index}")
                treecell_index_counts[treecell_index] += 1

                properties = {}  # {k: fields[v] for k, v in data_property_by_col_idx.items()}
                for prop_lbl, prop_info in config['DATA_CONFIG']['properties'].items():
                    prop_id = prop_lbl  # prop_info['id']
                    prop_col_idx = col_index_map[prop_info['id']] if prop_info['id'] is not None else None
                    field = fields[prop_col_idx] if prop_col_idx is not None else None

                    if prop_info['type'] == "vector":
                        vec = calculate_annotation_vector(fields)
                        if vec:
                            properties[f"{prop_id}_x"] = vec[0]
                            properties[f"{prop_id}_y"] = vec[1]
                            properties[f"{prop_id}_z"] = vec[2]
                    elif prop_info['type'] == "rgb":
                        if field[0] == '#':
                            properties[prop_id] = hex_to_rgb(field)
                        else:
                            raise ValueError(f"Only Hex colors are currently supported: {field}")
                    elif prop_info['type'] == "rgba":
                        if field[0] == '#':
                            properties[prop_id] = hex_to_rgba(field)
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
                        properties[prop_id] = enum_value
                    else:
                        properties[prop_id] = field
                    
                relation_fields = {k: (c, fields[v]) for k, (c, v) in relation_col_indices.items()}
                # logging.info(f"relation_fields: {relation_fields}")
                # The following annotation description is specific to LineAnnotations.
                # TODO: Generalize or dynimcally build or polymorphically populate other annotation types here.
                annotation_desc = build_annotation_description_from_str__one_annotation_per_row__multiple_points_per_row(id_, pt_positions, treecell_index, properties, relation_fields)
                annotation_descriptions.append(annotation_desc)
        
        logging.info(f"treecell_index_counts (top 5): {treecell_index_counts.most_common(5)}")

    logging.info(f"Num gathered merged annotation_descriptions: {len(annotation_descriptions)}")

    if len(annotation_descriptions) != len(lines):
        raise ValueError(f"Num gathered annotations != num merged lines of input: {len(annotation_descriptions)} != {len(lines)}")
    
    timestamps.append(("gather annotation_descriptions from table", default_timer()))
    
    writer = simple_writer_no_spatial_indexing.SimpleWriter("LINE", dimensions, cell_bounds_low, cell_bounds_high, tree_level)
    
    # writer.by_id_sharding = anno.ShardingSpec(hash=config['ID_SHARDING_HASH'], preshift_bits=config['ID_PRESHIFT_BITS'], shard_bits=config['ID_SHARDING_BITS'], minishard_bits=config['ID_MINISHARDING_BITS'])

    bounds_range = [cell_bounds_high[d] - cell_bounds_low[d] for d in range(3)]
    if debug:
        logging.info(f"bounds_low, bounds_high, bounds_range: {cell_bounds_low} {cell_bounds_high} {bounds_range}")
        logging.info(f"dimensions: {dimensions}")
    voxel_extent = [
        bounds_range[0] * dimensions['x'][0],
        bounds_range[1] * dimensions['y'][0],
        bounds_range[2] * dimensions['z'][0],
    ]
    spatial_sharding_spec = anno.ShardingSpec(
        hash=config['SPATIAL_SHARDING_HASH'],
        preshift_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['preshift_bits'],
        shard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits'],
        minishard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['minishard_bits'])
    writer.spatial_sharding = spatial_sharding_spec  # Comment out this line (or don't conditionally don't call it) to generate a non-sharded spatial index. However, the input to this function is already a merged shard of multiple tree cells, so choosing to not shard at this point is too late. Instead see the other capsule named "finalize spatial index unsharded".
    for tree_level_ve in range(tree_level+1):
        if debug:
            logging.info(f"tree level, voxel_extent: {tree_level_ve} {voxel_extent[0]:10.1f} {voxel_extent[1]:10.1f} {voxel_extent[2]:10.1f}")
        spatial_sharding_spec = anno.ShardingSpec(
            hash=config['SPATIAL_SHARDING_HASH'],
            preshift_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level_ve]['preshift_bits'],
            shard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level_ve]['shard_bits'],
            minishard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level_ve]['minishard_bits'])
        writer.spatial_specs.append(anno.SpatialEntry(voxel_extent, [2**tree_level_ve, 2**tree_level_ve, 2**tree_level_ve], f"spatial{tree_level_ve}", 1, sharding=spatial_sharding_spec))
        voxel_extent = [v/2 for v in voxel_extent]

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
        writer.property_specs.append(
            anno.PropertySpec(property_name, property_info['type'], property_name, property_info['enum_values'], property_info['enum_labels']))

    # Define relationships (yes, spatial indexing needs this)
    for relation, column_name in data_relations.items():
        writer.relationships.append(anno.Relationship(relation, sharding=anno.ShardingSpec()))
    
    timestamps.append(("init writer", default_timer()))
    
    for i, annotation_description in enumerate(annotation_descriptions):
        # The confirmation performed here probably slows this process down significantly. Remove it once correct behavior is confirmed.
        if 'point_annotation_config' in config['DATA_CONFIG']:
            confirm_point_annotation_location_treecellid_shard(annotation_description, tree_level, cell_bounds_low, cell_bounds_high, shard_hex)
        elif 'line_annotation_config' in config['DATA_CONFIG']:
            confirm_line_annotation_location_treecellid_shard(annotation_description, tree_level, cell_bounds_low, cell_bounds_high, shard_hex)
        if 'polyline_annotation_config' in config['DATA_CONFIG']:
            confirm_polyline_annotation_location_treecellid_shard(annotation_description, tree_level, cell_bounds_low, cell_bounds_high, shard_hex)

        annotation = build_annotation(annotation_description)
        if i < 3:
            logging.info(f"Annotation {i:10}: {annotation}")
        
        # QUESTION: When I had a bug that always wrote the first cell index into all chunks of the shard file, why didn't sharding.py:write_shard_file():"if expected_shard != shard_number:" not trigger an exception? I'm trying to force it back to that erroneous state here by hardcoding a single cell index in for all chunks in the shard file, and as before, I'm not seeing that exception arise, but why not?!
        if debug2 and annotation_description['treecell_index'] is not None and annotation_description['treecell_index'] != "0_0_0":
            logging.info(f"Overriding cell index {annotation_description['treecell_index']} -> 0,0,0, shard {get_shard_hex(tree_level, [int(v) for v in annotation_description['treecell_index'].split('_')])} -> {get_shard_hex(tree_level, [0, 0, 0])}")
            writer.cell_annotations["0_0_0"].append(annotation)
        else:
            # assert annotation_description["treecell_index"] == "0_0_0"
            # logging.info(f"Adding line annotation {i} {treecell_index} to writer's cell annotations list")
            writer.cell_annotations[annotation_description["treecell_index"]].append(annotation)
    
    for tci_i, treecell_index in enumerate(writer.cell_annotations):
        if tci_i >= 3:
            break
        logging.info(f"Num writer.cell_annotations[annotation_description['{treecell_index}']] (first 3 shown): {len(writer.cell_annotations[treecell_index])}")

    # DEBUG
    # for i, annotation in enumerate(writer.cell_annotations[annotation_description["treecell_index"]]):
    #     logging.info(f"Writer cell annotation {i}: {annotation.id} {annotation.start} {annotation.end}")
    
    timestamps.append(("append annotation_descriptions to writer", default_timer()))

    return writer, len(annotation_descriptions)

def save_polyline_as_precomputed(table, columns, cell_bounds_low, cell_bounds_high, tree_level, shard_hex, data_properties, data_property_by_col_idx, data_relations, data_relation_by_col_idx, debug, debug2):
    raise RuntimeError("Not implemented yet")

def save_shard_data_as_precomputed(table, columns, cell_bounds_low, cell_bounds_high, subdir, tree_level, shard_hex, debug=False, debug2=False):
    timestamps.append(("save_shard_data_as_precomputed() top", default_timer()))

    logging.info("\n" + "=" * 100 + "\n")
    logging.info(f"Writing this cell's precomputed file to: {results_loc + subdir}")
    
    # Joe's original code would write a precomputed file in a series of numerous small file write operations.
    # The following parameter offers the option of generating the entire precomputed file via
    # write operations to a memory buffer first, and then dumping the entire buffer to disk in a single
    # disk I/O operation, which I theorized and hoped would be faster.
    # Testing suggests that this approach works spectacularly, but I leave it as an optional
    # parameter for future comparison and validation. 
    USE_RAM_BUFFER = True
    logging.info(f"USE_RAM_BUFFER: {USE_RAM_BUFFER}")

    if debug:
        logging.info(f"Cell bounds: {cell_bounds_low} {cell_bounds_high}")

    logging.info(f"Table type: {type(table)}")
    # logging.info(f"Table:\n{table}\n")

    if isinstance(table, pd.DataFrame):
        assert columns == list(table.columns)
    #     col_index_map = {col_name: i for i, col_name in enumerate(table.columns)}
    # elif isinstance(table, str):
    #     col_index_map = {col_name: i for i, col_name in enumerate(columns)}
    col_index_map = {col_name: i for i, col_name in enumerate(columns)}
    
    data_properties = config['DATA_CONFIG']['properties']
    data_properties_cols = [v['id'] for k, v in data_properties.items()]
    data_property_by_col_idx = {col: col_index_map[col] if col else None for col in data_properties_cols}
    data_relations = config['DATA_CONFIG']['relations']
    data_relation_by_col_idx = {col: col_index_map[v['id']] for col, v in data_relations.items()}

    writer, num_annotations = save_annotations_as_precomputed(table, columns, cell_bounds_low, cell_bounds_high, tree_level, shard_hex, data_properties, data_property_by_col_idx, data_relations, data_relation_by_col_idx, debug, debug2)

    # Direct the writer to write its contents out
    if not USE_RAM_BUFFER:
        logging.info("Writing precomputed file without RAM buffer")
        writer.write(results_loc + subdir)
        timestamps.append(("write precomputed to file", default_timer()))
    else:
        logging.info("Writing precomputed file with RAM buffer")
        # The following lines are copied out of simple_writer_no_spatial_index.py and sharding.py
        dir_path = results_loc + subdir
        # logging.info(f"  dir_path: {dir_path}")
        task_spec = writer.spatial_specs[writer.tree_level]
        # logging.info(f"  task_spec.key: {task_spec.key}")
        shard_dir = utilities.path_join(dir_path, task_spec.key)
        # logging.info(f"  shard_dir: {shard_dir}")
        shard_dir = os.path.expanduser(shard_dir)
        # logging.info(f"  shard_dir: {shard_dir}")
        os.makedirs(shard_dir, exist_ok=True)
        filepath = utilities.path_join(shard_dir, f"{shard_hex}.shard")
        # logging.info(f"  filepath: {filepath}")
        filepath = os.path.expanduser(filepath)
        logging.info(f"  filepath: {filepath}")
    
        timestamps.append(("prepare precomputed writer filepath", default_timer()))
        
        file_buffer = io.BytesIO()
        with file_buffer as f_buf:
            writer.writef(f_buf, shard_number=int(shard_hex, 16))  # Remove 'shard_num' param to disable some debugging/validation tests
            timestamps.append(("write precomputed to buffer", default_timer()))
            
            with open(filepath, "wb") as f_disk:
                f_disk.write(file_buffer.getbuffer())
            timestamps.append(("write precomputed buffer to file", default_timer()))
            logging.info(f"Shard file size for shard {shard_hex}: {os.path.getsize(filepath):,} B")

            validate_shard_file(filepath, tree_level, writer.property_specs)
            timestamps.append(("validate shard file", default_timer()))

    timestamps.append(("write precomputed file", default_timer()))

    if not USE_RAM_BUFFER:
        # Remove the extraneous directories
        all_dirs = list(glob.glob(f"{results_loc}{subdir}*"))
        for dir_ in all_dirs:
            if os.path.isdir(dir_):
                dir_name = os.path.basename(dir_)
                if not dir_name.startswith("spatial"):
                    if debug:
                        logging.info(f"Deleting non-spatial-index dir: {dir_}")
                    shutil.rmtree(dir_)
        
        # Remove the extraneous files and validate the target file
        spatial_shard_files = list(glob.glob(f"{results_loc}{subdir}spatial{tree_level}/*.shard"))
        if debug:
            logging.info(f"Spatial files {results_loc}{subdir}spatial{tree_level}/*.shard ({len(spatial_shard_files)}):\n  {'\n  '.join(spatial_shard_files)}\n")
        for shard_file in spatial_shard_files:
            shard_filename = os.path.basename(shard_file)
            if shard_filename != f"{shard_hex}.shard":
                os.remove(shard_file)
            else:
                validate_shard_file(shard_file, tree_level, writer.property_specs)

        # The global info file will be created by a different capsule.
        # We don't need this local info file.
        if os.path.exists(f"{results_loc}{subdir}info"):
            os.remove(f"{results_loc}{subdir}info")

    timestamps.append(("save_shard_data_as_precomputed() bottom", default_timer()))

    return num_annotations

def parse_treelevel_shard_file_by_treecellid(treelevel_shard_file):
    logging.info(f"parse_treelevel_shard_file_by_treecellid() {treelevel_shard_file}")

    per_treecellid_lines = defaultdict(list)

    with open(treelevel_shard_file) as f:
        lines = f.readlines()
        for line_i, line in enumerate(lines):
            line = line.strip()
            if line_i < 5:
                logging.info(f"Line {line_i} (first 5 shown): {line}")
            fields = line.split(',')
            treecellid = fields[-1]
            if line_i < 5:
                logging.info(f"Line {line_i} (first 5 shown) treecellid: {treecellid}")
            per_treecellid_lines[treecellid].append(line)
    
    logging.info(f"All tree level cell ids: {per_treecellid_lines.keys()}")

    return per_treecellid_lines

def process_treelevel_shard_file(treelevel_shard_file, tree_level, MERGE_FORMAT, columns, merged_table, debug):
    # logging.info("\n" + "*" * 100 + "\n")
    logging.info(f"Processing tree level shard file: {treelevel_shard_file}")

    timestamps.append(("treelevel_shard_file_loop_top", default_timer()))

    per_treecellid_lines = parse_treelevel_shard_file_by_treecellid(treelevel_shard_file)

    merged_tables = []
    for tree_level_cell_id_str, treecellid_lines in per_treecellid_lines.items():
        tree_level_cell_id = [int(v) for v in tree_level_cell_id_str.split('_')]
        logging.info(f"  Tree level cell id: {tree_level_cell_id_str} {tree_level_cell_id}")
        
        if debug:
            grid_dim = 2 ** tree_level
            grid_shape = (grid_dim, grid_dim, grid_dim)
            morton_code = utilities.compressed_morton_code(tree_level_cell_id, grid_shape)
            logging.info(f"  Tree level cell id Morton code: {morton_code}")
        
        # Only used by MERGE_FORMAT == "str_join"
        files_contents = []

        for shard_csv in shard_csvs:
            timestamps.append(("shard_csv_loop_top", default_timer()))
            
            # logging.info(f"  Shard CSV size {os.path.getsize(shard_csv)}B")
            timestamps.append(("shard_csv_loop (get csv size)", default_timer()))

            if MERGE_FORMAT == "dataframe":
                assert False
            elif MERGE_FORMAT == "str_sum" or MERGE_FORMAT == "str_join":
                with open(shard_csv) as f:
                    file_content = f.read()
                # logging.info(f"    One shard len: {len(file_content)} B from file of size {os.path.getsize(shard_csv)} B")
                timestamps.append(("shard_csv_loop (read csv)", default_timer()))
                
                # if file_content[-1] != '\n':
                #     file_content += '\n'
                # timestamps.append(("shard_csv_loop (append endline)", default_timer()))
                
                file_content_lines = file_content.split('\n')
                file_content_lines2 = []
                for file_content_line in file_content_lines:
                    if file_content_line:
                        file_content_line += f",{tree_level_cell_id_str}\n"
                        file_content_lines2.append(file_content_line)
                file_content = "".join(file_content_lines2)
                timestamps.append(("shard_csv_loop (append tree cell index)", default_timer()))

                if MERGE_FORMAT == "str_sum":
                    if merged_table is None:
                        merged_table = ""
                    merged_table += file_content
                    timestamps.append(("shard_csv_loop (append content)", default_timer()))
                elif MERGE_FORMAT == "str_join":
                    files_contents.append(file_content)
                    timestamps.append(("shard_csv_loop (stash content)", default_timer()))
            
            timestamps.append(("shard_csv_loop_bottom (merge tables)", default_timer()))

            if MERGE_FORMAT == "str_join":
                if merged_table is None:
                    merged_table = ""
                merged_table += "".join(files_contents)
                # logging.info(f"    Merged shard len: {len(merged_table)} B")
                timestamps.append(("shard_csv_loop_bottom (join table strs)", default_timer()))

        timestamps.append(("treelevel_shard_file_loop_bottom (merge tables)", default_timer()))

        return merged_table

def process_treelevel_shard_dirs(shard_dirs):
    logging.info("\n" + "#" * 100 + "\n")
    logging.info("Processing shard directories")

    MERGE_FORMAT = "str_join"  # dataframe, str_sum, or str_join
    summed_dfs_len = 0
    summed_csvstrs_len = 0
    num_annotations_all_shard_dirs = 0
    for shard_dir_i, shard_dir in enumerate(shard_dirs):
        timestamps.append(("shard_dir_loop_top", default_timer()))

        logging.info("\n" + "*" * 100 + "\n")
        shard_dirname = os.path.basename(shard_dir)

        # DEBUG - copy subdir from data/ to results/ so I can inspect it
        # shutil.copytree(shard_dir, shard_dir.replace(data_loc, results_loc))

        tree_level = int(shard_dirname.split('__')[0].split('-')[1])
        shard_hex = shard_dirname.split('__')[1].split('-')[1]
        logging.info(f"Shard dir, tree level, and shard hex: {shard_dirname} {tree_level} {shard_hex}")

        treelevel_shard_files = glob.glob(f"{shard_dir}/*.csv")
        logging.info(f"Tree levels shard files ({len(treelevel_shard_files)}) (first 30 shown):\n  {'\n  '.join(treelevel_shard_files[:30])}\n")

        columns = config['DATA_CONFIG']['columns'] + ['treecell_index']

        id_column = config['DATA_CONFIG']['id_column']
        # logging.info(f"id_column: {id_column}")
        if id_column is None:
            logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {columns[0]}.")

        merged_table = None
        logging.info("Only the first 5 subdirs will be logged...")

        timestamps.append(("shard_dir_loop_init", default_timer()))
        
        # for treelevel_shard_files_i, treelevel_shard_file in enumerate(treelevel_shard_files):
        #     merged_tables = process_treelevel_shard_file(treelevel_shard_file, tree_level, MERGE_FORMAT, columns, merged_table, treelevel_shard_files_i < 5)
        
        # if MERGE_FORMAT == "dataframe":
        #     logging.info(f"Merged shard len: {len(merged_table)}")
        #     summed_dfs_len += len(merged_table)
        # elif MERGE_FORMAT == "str_sum" or MERGE_FORMAT == "str_join":
        #     logging.info(f"Merged shard len: {len(merged_table)/1000000}M")
        #     summed_csvstrs_len += len(merged_table)

        # Perhaps these should be assigned as the tight bounds around the shard's tree cells?!
        # Looking at the code, they appear to only affect the metadata for the shard, not the actual annotations.
        cell_bounds_low = config['DATA_CONFIG']['volume_bounds'][0]
        cell_bounds_high = config['DATA_CONFIG']['volume_bounds'][1]

        if OUTPUT_STYLE == "capsule":
            subdir_csv = f"spatial{tree_level}__sharded__csv__{shard_dirname}/"
            subdir = f"spatial_index__sharded__{shard_dirname}/"
        elif OUTPUT_STYLE == "results":
            subdir_csv = f"spatial{tree_level}__sharded__csv/"
            subdir = f"spatial_index__sharded/"
        elif OUTPUT_STYLE == "results_for_ng":
            subdir_csv = f"spatial{tree_level}__sharded__csv/"
            subdir = ""

        os.makedirs(f"{results_loc}{subdir}", exist_ok=True)
        # num_annotations = save_shard_data_as_precomputed(merged_table, columns, cell_bounds_low, cell_bounds_high, subdir, tree_level, shard_hex, shard_dir_i==0)
        # num_annotations_all_shard_dirs += num_annotations
    
        for treelevel_shard_files_i, treelevel_shard_file in enumerate(treelevel_shard_files):
            logging.info(f"Reading treelevel shard file {treelevel_shard_file} with size {os.path.getsize(treelevel_shard_file):,} B")
            with open(treelevel_shard_file) as f:
                file_contents = f.read()
            # logging.info(f"\nTreelevel shard file content (first 500 chars of {len(file_contents):,}):\n{file_contents[:500]}")
            # logging.info(f"\nTreelevel shard file content (last 500 chars of {len(file_contents):,}):\n{file_contents[-500:]}")
            num_annotations = save_shard_data_as_precomputed(file_contents, columns, cell_bounds_low, cell_bounds_high, subdir, tree_level, shard_hex, shard_dir_i==0 and treelevel_shard_files_i==0)
            num_annotations_all_shard_dirs += num_annotations
        
        logging.info(f"\nTotal num annotations written: {num_annotations_all_shard_dirs}")

        timestamps.append(("save_precomputed", default_timer()))

        if config['SAVE_CSV']:
            os.makedirs(f"{results_loc}{subdir_csv}", exist_ok=True)
            save_shard_data_as_csv(merged_table, header_str, subdir_csv, tree_level, shard_hex)

        timestamps.append(("save_csv", default_timer()))
        
        # if subdir:
        #     shutil.rmtree(f"{results_loc}{subdir}")

        # filepath = f"{results_loc}{subdir}oct_tree_info__treelevel-{tree_level:0>2}__shard-{shard_hex}.txt"
        # logging.info(f"Writing one cell info file (for debugging): {filepath}")
        # with open(filepath, 'w') as f:
        #     f.write(f"Tree level:    {tree_level}\n")
        #     f.write(f"Shard hex:     {shard_hex}\n")
        #     f.write(f"Num rows:      {len(merged_df)}\n")
        #     f.write(f"Cell bounds:   {[cell_bounds_low, cell_bounds_high]}\n")

        timestamps.append(("shard_dir_loop_bottom (merge tables & save precomputed)", default_timer()))

    if MERGE_FORMAT == "dataframe":
        logging.info(f"\nSummed total num data rows: {summed_dfs_len}")
    elif MERGE_FORMAT == "str_sum" or MERGE_FORMAT == "str_join":
        logging.info(f"\nSummed total data size: {summed_csvstrs_len/1000000}M")
    
    logging.info("\n" + "#" * 100 + "\n")

def main():
    """
    RECEIVE INPUT WITH THE FOLLOWING LAYOUT:
 
    ../data/ contents:
        job_config.py
        treelevel-04_shard-1
        treelevel-04_shard-2
        treelevel-04_shard-3
 
    PRODUCE OUTPUT WITH THE FOLLOWING LAYOUT:
 
    ../results/ contents:
        treelevel-04_shard-1/synapses_one_shard__treelevel-04__shard-1.csv
        treelevel-04_shard-2/synapses_one_shard__treelevel-04__shard-2.csv
        treelevel-04_shard-3/synapses_one_shard__treelevel-04__shard-3.csv
    """
    global data_loc, results_loc, config, timestamps, dimensions, missing_enum_labels

    data_loc = "../data/"
    results_loc = "../results/"

    # Instead of generating a new log id, reuse the id from the upstream capsule
    # logging_uid = hex(int(random.random()*1000000000000))[2:]
    # upstream_log = sorted(list(glob.glob(f"{data_loc}logs/log*regroup*.log")))
    upstream_log = sorted(list(glob.glob(f"{results_loc}log*regroup*.log")))
    # assert len(upstream_log) == 1
    logging.info(f"upstream_log: {upstream_log}")
    upstream_log = upstream_log[0]
    logging_uid = upstream_log.split('_')[-1].split('.')[0]
    # os.makedirs(f"{results_loc}logs/", exist_ok=True)
    
    logging.basicConfig(level=logging.CRITICAL, handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{results_loc}log_conglomerate_spatial_index_by_shard_{logging_uid}.log", mode="a")
        ], format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("CONGLOMERATE SPATIAL INDEX BY SHARD")

    config = read_config(["id", "relation", "spatial"])
    logging.basicConfig(level=get_logging_level_from_desc(config['LOGGING_LEVEL']), handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{results_loc}log_conglomerate_spatial_index_by_shard_{logging_uid}.log", mode="a")
        ], format=config['LOGGING_FORMAT'], force=True)

    for module in ['simple_writer_no_spatial_indexing', 'sharding', 'annotations']:
        logging.getLogger(module).setLevel(get_logging_level_from_desc(config['PRECOMPUTED_FILE_WRITER_LOGGING_LEVEL']))
        logging.getLogger(module).addHandler(logging.StreamHandler(sys.stdout))
        logging.getLogger(module).addHandler(logging.FileHandler(f"{results_loc}log_conglomerate_spatial_index_by_shard_{logging_uid}.log", mode="a"))
    
    # Pick one: changing this requires altering the pipeline topology
    # OUTPUT_STYLE = "capsule"  # Pipeline must connect this capsule to the 'reorganize directory structure' capsule with 'Collect' type
    # OUTPUT_STYLE = "results"  # Pipeline must connect this capsule to the 'results'
    # OUTPUT_STYLE = "results_for_ng"  # Pipeline must connect this capsule to the 'results'. The resulting layout will drop into place for Neuroglancer without any further moving around.
    
    timestamps = []
    timestamps.append(("start", default_timer()))

    missing_enum_labels = set()
    
    if config['SPATIAL_INDEX_ENABLED']:
        max_data_rows_per_tree_cell = config['MAX_DATA_ROWS_PER_TREE_CELL']
 
        dimensions = config['DATA_CONFIG']['dimensions']
        dimensions_lst = [v[0] for v in dimensions.values()]
 
        data_loc_contents = sorted(os.listdir(data_loc))
        data_loc_contents = [v for v in data_loc_contents if "placeholder" not in v]
        logging.info(f"{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):")
        logging.info('  ' + '\n  '.join(data_loc_contents[:30]).strip() + '\n')
        
        timestamps.append(("init", default_timer()))

        shard_worker_desc_file_hash, assigned_shards = read_shardworker_file()

        # We don't need to transfer upstream logs when the conglomerator is merged into the regrouper.
        # Copy upstream logs from input to output
        # if "0" in assigned_shards:  # To avoid CodeOcean name collisions, only do this from one capsule
        #     logs = sorted(list(glob.glob(f"{data_loc}log*.log")))
        #     for log in logs:
        #         logging.info(f"Copying log from {data_loc} to {results_loc}: {log}")
        #         shutil.copy(log, f"{results_loc}{os.path.basename(log)}")
        
        # We don't need to download the results from a bucket or dearchive them if we are running the conglomerator right here in this capsule immediately after running the regrouped/combiner.
        # download_data_from_bucket(shard_worker_desc_file_hash)

        # Decompress and untar any compressed files
        # tared_files = list(glob.glob(f"{data_loc}treelevel-*__shard-*.tar*"))
        # tared_files = list(glob.glob(f"{data_loc}*tree_cell_shards__shard_worker-*.tar*"))
        # logging.info(f"Input tar files:\n  {'\n  '.join(sorted(tared_files)).strip()}" + '\n')
        # for tared_file in tared_files:
        #     mode = "r:gz" if config['COMPRESS_ARCHIVE'] else "r"
        #     with tarfile.open(tared_file, mode) as tar:
        #         tar.extractall(path=f"{data_loc}")
        #     os.remove(tared_file)
        # logging.info(f"{data_loc} contents after extraction ({len(os.listdir(data_loc))}) (first 50 shown):\n  {'\n  '.join(sorted(os.listdir(data_loc))[:50]).strip()}\n")
        
        # timestamps.append(("extract_input_files", default_timer()))

        treelevel_shard_dirs = sorted(glob.glob(f"{data_loc}treelevel-*__shard-*"))
        treelevel_shard_dirs = [treelevel_shard_dir for treelevel_shard_dir in treelevel_shard_dirs if ".tar" not in treelevel_shard_dir]
        logging.info(f"Shard directories (first 50 shown):\n  {'\n  '.join(sorted(treelevel_shard_dirs)[:50]).strip()}\n")

        treelevel_shard_subdirs = sorted(glob.glob(f"{data_loc}treelevel-*__shard-*/*"))
        treelevel_shard_subdirs = [treelevel_shard_dir for treelevel_shard_dir in treelevel_shard_subdirs if ".tar" not in treelevel_shard_dir]
        logging.info(f"Shard subdirectories (first 50 shown):\n  {'\n  '.join(sorted(treelevel_shard_subdirs)[:50]).strip()}\n")

        timestamps.append(("prep_treelevel_shard_dirs", default_timer()))
 
        process_treelevel_shard_dirs(treelevel_shard_dirs)

        if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
            if config['UPLOAD_RESULTS_TO_GCP']:
                logging.info("\nUploading files to Google Storage")
                st = default_timer()
                spatial_dirs = list(glob.glob(f"{results_loc}spatial*"))
                for spatial_dir in spatial_dirs:
                    logging.info(f"Uploading {spatial_dir}    {os.path.basename(spatial_dir)}")
                    upload_directory_to_gcp(results_loc, os.path.basename(spatial_dir) + '/', config["TIMESTAMP"], config['GCP_BUCKET'], config['GCP_RESULTS_BLOB_PATH'])#, dryrun=True)
                ets = default_timer() - st
                logging.info(f"GCP upload elapsed time: {seconds_to_hms(ets)}")

                timestamps.append(("upload_files_to_gcp", default_timer()))
            else:
                logging.info("UPLOAD_RESULTS_TO_GCP setting is false. Results won't be uploaded to GCP.")
        else:
            logging.info(f"\n{data_loc}DEBUG_FLAG.txt file found. Results won't be uploaded to GCP.")

        if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
            # Only do this if GCP uploading is enabled since, if it is disabled, the user may explicitly wish to retrieve the results to use them some other nonGCP way.
            if config['UPLOAD_RESULTS_TO_GCP']:
                # To reduce CO storage, there is no need to save the results after uploading them to GCP
                # Note that deleting these outputs and thereby avoiding copying them to the final results
                # doesn't make the capsule run any faster.
                # Also note that when the conglomerator is merged into the regrouper, we *must* either delete these directories or rename them with shard-worker specific suffixes to avoid directory name collisions in the final capsule.
                DELETE_OR_MOVE_RESULTS_DIRS = "delete"  # delete, move
                spatial_dirs = list(glob.glob(f"{results_loc}spatial*"))
                if DELETE_OR_MOVE_RESULTS_DIRS == "delete":
                    logging.info(f"\nDeleting result directory after uploading to GCP")
                    for spatial_dir in spatial_dirs:
                        shutil.rmtree(f"{results_loc}{spatial_dir}")
                    timestamps.append(("delete results", default_timer()))
                else:
                    logging.info(f"\nRenaming result directory after uploading to GCP")
                    for spatial_dir in spatial_dirs:
                        os.rename(f"{results_loc}{spatial_dir}", f"{results_loc}{spatial_dir}__shard_worker-{shard_worker_desc_file_hash}")
                    timestamps.append(("rename results", default_timer()))
    
    if missing_enum_labels:
        logging.error(f"Missing enum labels: {missing_enum_labels}")
        raise ValueError(f"Missing enum labels: {missing_enum_labels}")

    finalize_results(results_loc)
    timestamps.append(("finalize_results", default_timer()))
    
    # logging.error("\nElapsed timestamps:")
    accum_elapsed_times = Counter()
    for ti, time in enumerate(timestamps):
        if ti > 0:
            elap_t = time[1] - timestamps[ti-1][1]
            accum_elapsed_times[time[0]] += elap_t
            # logging.error(f"  {seconds_to_hms(elap_t)} {time[0]}")
        
    logging.error("Accumulated elapsed timestamps:")
    for label, elap_t in accum_elapsed_times.items():
        logging.error(f"  {seconds_to_hms(elap_t)} {label}")

if __name__ == "__main__":
    main()
 
logging.info("\nDone")
process_running_time()
