import sys
from dataclasses import dataclass
from abc import abstractmethod
import logging
import os
import glob
from timeit import default_timer
from collections import defaultdict, Counter
import math
# import copy
import zipfile
# import numpy as np
import pandas as pd
import csv
import re
import random
import shutil
import tarfile
import io

from shared.sharding_spec_calculations import *
import shared.annotations as anno
import shared.sharding as sharding
import shared.utilities as utilities

from shared.util import *
from shared.nested_profiler import *
from shared.ram_data_pond import *
from shared.raw_table import *
from shared.aws_storage import *

from shared.test_sharding import *
from shared.test_morton_code_and_shardhex import *

@dataclass
class Annotation:
    def __init__(self, id_, raw_data):
        self.id_ = id_
        self.raw_data = raw_data  # Perhaps a CSV row from the original input file

    @abstractmethod
    def get_all_points(self):
        assert False
        pass

    def add_treecell_index(self, treecell_index, format):
        last_field = self.raw_data.strip().split(',')[-1]
        if len(last_field.split('_')) == 3:
            logging.error(f"ERROR!  Annotation is about to receive a second appended treecell_index:   {last_field}   {treecell_index}")
            logging.info(self.raw_data)
            assert False

        if format == "csv":
            self.raw_data = self.raw_data.strip() + f",{treecell_index}"
        else:
            raise ValueError(f"Unknown data format when adding tree cell index: {format}")

@dataclass
class PointAnnotation(Annotation):
    def __init__(self, id_, position, raw_data):
        super().__init__(id_, raw_data)
        self.position = position

    def copy(self):
        return PointAnnotation(self.id_, self.position, self.raw_data)

    def get_all_points(self):
        return [self.position]

@dataclass
class LineAnnotation(Annotation):
    def __init__(self, id_, start, end, raw_data):
        super().__init__(id_, raw_data)
        self.start = start
        self.end = end

    def copy(self):
        return LineAnnotation(self.id_, self.start, self.end, self.raw_data)

    def get_all_points(self):
        return [self.start, self.end]

@dataclass
class PolyLineAnnotation(Annotation):
    def __init__(self, id_, points, raw_data):
        super().__init__(id_, raw_data)
        self.points = points

    def copy(self):
        new_points = [v for v in self.points]
        return PolyLineAnnotation(self.id_, new_points, self.raw_data)

    def get_all_points(self):
        return self.points

def get_shard_hex(tree_level, tree_level_cell_id, verbose=False):
    """
    Given a tree level and a tree cell id:
      - Determine the grid dimension
      - Determine the grid shape from the grid dimension
      - Determine the Morton code from the tree cell id and the grid shape
      - Determine the shard number from the Morton code
      - Determine the shard hex from the shard number
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

    if verbose:
        logging.info(f"get_shard_hex(): Tree level, cell id, morton code, shard bits, shard range, shard num, shard hex: {tree_level} {tree_level_cell_id} {morton_code:>2} {config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits']} {2**config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits']} {shard_num} {shard_hex}")

        logging.info(f"get_shard_hex(): {tree_level:>10} [{tree_level_cell_id[0]:3}, {tree_level_cell_id[1]:3}, {tree_level_cell_id[2]:3}] {morton_code:>12} {config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits']:>11} {2**config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits']:>12} {shard_num:>10} {shard_hex:>10}")

    return shard_hex

def determine_capsule_tree_level(src_loc):
    """
    Determine the capsule's tree level by investigating top-level input files, and subtree and treecell directories.
    If there are any split files, this is the top of the tree, i.e., level 0.
    If there are any subtree directories, they determine the capsule's working tree level.
    Otherwise, this capsule is only passing completed tree cells through and there is no meaningful capsule tree level.
    """

    start_timeblock("determine_capsule_tree_level()")

    tree_level = None

    if config['DATA_CONFIG']['structure'] == 'one_annotation_per_row__multiple_points_per_row' or \
        config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row_in_one_field":
        # raw_input_files = glob.glob(f"{src_loc}*split-*_rows-*.csv")
        raw_input_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}*split-*.csv", src_loc==data_loc)
        logging.info(f"AAA raw_input_files: {src_loc}*split-*.csv: {raw_input_files}")
        if len(raw_input_files) == 0:
            raw_input_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}*split-*.parquet", src_loc==data_loc)
        if len(raw_input_files) == 0:
            raw_input_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}*split-*.zip", src_loc==data_loc)
        if raw_input_files:
            if len(raw_input_files) > 1:
                logging.info(raw_input_files)
                raise RuntimeError(f"Expected exactly 1 input file: {raw_input_files}")
            logging.info("Top-level split file found. Processing tree level 0.")
            tree_level = 0
        if tree_level is None:
            # Since no top-level split files were found, look for info files in the subdirectory.
            # info_files = glob.glob(f"{src_loc}subtrees/*/oct_tree__subtree__info.txt")
            # subtree_levels = set()
            # for fi, info_file in enumerate(info_files):
            #     with open(info_file) as f:
            #         for line in f:
            #             # We are looking for the child tree level since we are looking at files
            #             # written by the parent (the previous capsule). The "current" tree level
            #             # in the file was written by the previous capsule.
            #             if line.startswith("Tree child level:"):
            #                 subtree_level = int(line.split()[-1])
            #                 if fi == 0 or (subtree_levels and subtree_level not in subtree_levels):
            #                     logging.info(f"Read one subtree level: {subtree_level}")
            #                 subtree_levels.add(subtree_level)
            #                 tree_level = subtree_level
            #                 break
            info_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}subtrees/*/oct_tree__subtree__info.txt", src_loc==data_loc)
            subtree_levels = set()
            for fi, info_file in enumerate(info_files):
                lines = ram_data_pond.read_splitlines_from_disk_or_ram_data_pond(info_file, None, None, src_loc==data_loc)
                for line in lines:
                    # We are looking for the child tree level since we are looking at files
                    # written by the parent (the previous capsule). The "current" tree level
                    # in the file was written by the previous capsule.
                    if line.startswith("Tree child level:"):
                        subtree_level = int(line.split()[-1])
                        if fi == 0 or (subtree_levels and subtree_level not in subtree_levels):
                            logging.info(f"Read one subtree level: {subtree_level}")
                        subtree_levels.add(subtree_level)
                        tree_level = subtree_level
                        break
    elif config['DATA_CONFIG']['structure'] == 'one_annotation_per_file__one_point_per_row':
        raise RuntimeError(f"Structure {config['DATA_CONFIG']['structure']} should have been converted in an earlier capsule.")
        raw_input_dirs = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}*split-*", src_loc==data_loc)
        if raw_input_dirs:
            if len(raw_input_dirs) > 1:
                logging.info(raw_input_dirs)
                raise RuntimeError(f"Expected exactly 1 input directory: {raw_input_dirs}")
            logging.info("Top-level split directory found. Processing tree level 0.")
            tree_level = 0
        if tree_level is None:
            # This section is identical to the section above (at the time of this writing at any rate)
            info_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}subtrees/*/oct_tree__subtree__info.txt", src_loc==data_loc)
            subtree_levels = set()
            for fi, info_file in enumerate(info_files):
                lines = ram_data_pond.read_splitlines_from_disk_or_ram_data_pond(info_file, None, None, src_loc==data_loc)
                for line in lines:
                    # We are looking for the child tree level since we are looking at files
                    # written by the parent (the previous capsule). The "current" tree level
                    # in the file was written by the previous capsule.
                    if line.startswith("Tree child level:"):
                        subtree_level = int(line.split()[-1])
                        if fi == 0 or (subtree_levels and subtree_level not in subtree_levels):
                            logging.info(f"Read one subtree level: {subtree_level}")
                        subtree_levels.add(subtree_level)
                        tree_level = subtree_level
                        break
    else:
        raise ValueError(f"Unknown structure: {config['DATA_CONFIG']['structure']}")

    logging.info(f"Determined tree level: {tree_level}\n")
    assert tree_level is not None

    end_timeblock("determine_capsule_tree_level()")

    return tree_level

def determine_subtree_bounds(cell_bounds_low, cell_bounds_mid, cell_bounds_high, child_cell_octant):
    '''
    See note in determine_pt_subtreecell() about X,Y,Z bit order.
    '''
    start_timeblock("determine_subtree_bounds()")

    x_bounds = [cell_bounds_mid[0], cell_bounds_high[0]] if child_cell_octant[0] == 1 else [cell_bounds_low[0], cell_bounds_mid[0]]
    y_bounds = [cell_bounds_mid[1], cell_bounds_high[1]] if child_cell_octant[1] == 1 else [cell_bounds_low[1], cell_bounds_mid[1]]
    z_bounds = [cell_bounds_mid[2], cell_bounds_high[2]] if child_cell_octant[2] == 1 else [cell_bounds_low[2], cell_bounds_mid[2]]

    end_timeblock("determine_subtree_bounds()")

    return [[x_bounds[0], y_bounds[0], z_bounds[0]], [x_bounds[1], y_bounds[1], z_bounds[1]]]

def determine_pt_subtreecell(cell_bounds_low, cell_bounds_mid, cell_bounds_high, pt, debug_label=None):
    """
    Given a cell bounds and a point, determine which of eight subcell children the point resides in.
    The eight children consist of halving the cell counts on each of three axes.
    Put differently, determine which quadrant (or octant, strictly speaking) the point resides in.
    """
    start_timeblock("determine_pt_subtreecell()")

    # The potentially problematic situation detected here should have been caught by similar checks in earlier steps of the overall process
    # if pt[0] < cell_bounds_low[0] or pt[0] > cell_bounds_high[0] or \
    #     pt[1] < cell_bounds_low[1] or pt[1] > cell_bounds_high[1] or \
    #     pt[2] < cell_bounds_low[2] or pt[2] > cell_bounds_high[2]:
    #     logging.error(f"ERROR! Point resides outside tree cell bounds: {pt} outside {cell_bounds_low}, {cell_bounds_high} (this is okay if the opposing point is inside the bounds)")

    child_cell_local_id = [1 if pt[i] > cell_bounds_mid[i] else 0 for i in range(3)]

    if debug_label:
        cell_bounds_w = [
            cell_bounds_high[i] - cell_bounds_low[i] for i in range(3)
        ]
        logging.info(f"Point subtree determination for {debug_label} at {[float(v) for v in pt]}:")
        logging.info(f"  Cell bounds low/mid/high & determined subtree: {cell_bounds_low}  {cell_bounds_mid}  {cell_bounds_high}\n    @extent:  {cell_bounds_w}  =>  {child_cell_local_id}")
        # logging.info(f"  {float(pt[0]):15,.1f} >? {cell_bounds_mid[0]:15,.1f}:    {bool(child_cell_local_id[0])}")
        # logging.info(f"  {float(pt[1]):15,.1f} >? {cell_bounds_mid[1]:15,.1f}:    {bool(child_cell_local_id[1])}")
        # logging.info(f"  {float(pt[2]):15,.1f} >? {cell_bounds_mid[2]:15,.1f}:    {bool(child_cell_local_id[2])}")

    end_timeblock("determine_pt_subtreecell()")

    return child_cell_local_id

def select_holdout_rows_and_subdivide_subtree_rows(src_loc, num_splits, num_subsplits, annotations, tree_level, cell_bounds_low, cell_bounds_mid, cell_bounds_high, verbose):
    '''
    Save the rows intended to be held within this tree cell.
    For the rest of the rows, determine which of eight children they belong to and save them to the corresponding output.

    annos_this_level and annos_children_levels were initially implemented as DataFrames,
    but the concat operation is quite slow.
    So, this function now offers two options: DataFrames or lists.
    '''
    start_timeblock("select_holdout_rows_and_subdivide_subtree_rows()")

    # verbose = False
    if verbose:
        logging.info("\n\nBEWARE! verbose is True in select_holdout_rows_and_subdivide_subtree_rows()\n\n")

    # The concat_method is partially independent of whether the 'annotations' parameter is a DataFrame or a RAM file from the RAM data pond.
    # If annotations is a DataFrame, both concat_method 'dataframe' and 'list' are supported.
    # But if annotations is a RAM file, then only concat_method 'list' is supported.
    concat_method = "list"
    if concat_method == "dataframe":
        assert False, "concat_method dataframe no longer supported"
        assert isinstance(annotations, pd.DataFrame)
        annos_children_levels = [
            [
                [pd.DataFrame(columns=annotations.columns), pd.DataFrame(columns=annotations.columns)],
                [pd.DataFrame(columns=annotations.columns), pd.DataFrame(columns=annotations.columns)],
            ],
            [
                [pd.DataFrame(columns=annotations.columns), pd.DataFrame(columns=annotations.columns)],
                [pd.DataFrame(columns=annotations.columns), pd.DataFrame(columns=annotations.columns)],
            ],
        ]
    elif concat_method == "list":
        annos_children_levels = [
            [
                [[], []],
                [[], []],
            ],
            [
                [[], []],
                [[], []],
            ],
        ]

    if verbose:
        logging.info(f"\nHeld row count:            {config['MAX_DATA_ROWS_PER_TREE_CELL'] // num_splits:>11,} max-rows-per-tree-cell-split ({config['MAX_DATA_ROWS_PER_TREE_CELL']} max-rows-per-tree-cell // {num_splits} splits)")

    start_timeblock("extract_sample")

    # Determine how many rows to hold in this tree cell.
    # Then select a random sample of that many rows from the data.
    # Then remove the selected rows from the working set; the working set will be sent down to the next tree level.
    max_num_rows_to_hold = math.floor(config['MAX_DATA_ROWS_PER_TREE_CELL'] / (num_splits * num_subsplits))

    # PANDAS_SAMPLE_RANDOM_STATE = 0
    # logging.critical("WARNING!   " * 5)
    # logging.critical(f"PANDAS_SAMPLE_RANDOM_STATE set to hard-coded {PANDAS_SAMPLE_RANDOM_STATE}")

    if isinstance(annotations, pd.DataFrame):
        assert False
        annos_this_level = annotations.sample(n=min(max_num_rows_to_hold, len(annotations)))  #, random_state=PANDAS_SAMPLE_RANDOM_STATE)
        indices_to_drop = annos_this_level.index
        annotations_passed_on = annotations.drop(indices_to_drop)
    elif isinstance(annotations, RawTable):
        annos_this_level, indices_to_drop = annotations.sample(n=min(max_num_rows_to_hold, len(annotations)))  #, random_state=PANDAS_SAMPLE_RANDOM_STATE)
        annotations_passed_on = annotations.drop(indices_to_drop)
    elif isinstance(annotations, list):  # A list of Annotation subclass objects
        anno_indices = random.sample(list(np.arange(len(annotations))), min(max_num_rows_to_hold, len(annotations)))
        anno_indices_set = set(anno_indices)
        assert len(anno_indices_set) == len(anno_indices)  # Since the sample was built from an arange, this assertion is hardly necessary, but let's make sure anyway
        annos_this_level = [annotations[anno_idx] for anno_idx in anno_indices_set]
        annotations_passed_on = [anno for anno_idx, anno in enumerate(annotations) if anno_idx not in anno_indices_set]
    else:
        assert False, f"Unknown annotations type: {type(annotations)}"

    if verbose:
        logging.info(f"annotations len:           {len(annotations):>11,}")
        logging.info(f"max_num_rows_to_hold:      {max_num_rows_to_hold:>11,}")
        logging.info(f"annos_this_level len:      {len(annos_this_level):>11,}")
        logging.info(f"annotations_passed_on len: {len(annotations_passed_on):>11,}")

    end_start_timeblocks("extract_sample", "subdivide_subtree")

    num_cell_dups, num_cell_nondups = 0, 0

    # For the rows intended to send down the tree for further processing,
    # determine which of the eight children will receive each row and group them accordingly.

    if isinstance(annotations_passed_on, pd.DataFrame):
        assert False, "Development has moved from Pandas DataFrames to RawTable and the DataFrame design has not continued to advance. It is not longer reliable."
        # for row_idx, row in annotations_passed_on.iterrows():  # Pandas Dataframe iteration, slower than itertuples()
        for row_idx, row in enumerate(annotations_passed_on.itertuples(index=False)):  # Pandas Dataframe iteration, faster than iterrows()
            # Each annotation *might* be emitted twice, once for the PRE location and once for POST,
            # but only if the two locations fall into different child cells of the oct tree.
            # As such, the final index will have a size: DATA_SIZE <= INDEX_SIZE <= DATA_SIZE*2
            # In general, tiny annotations like annotations will rarely span the boundary between two tree cells,
            # so the final index size should be just a tiny bit larger than the orginal data size.

            # Furthermore, if the two locations fall into different tree cells, we must also pass the annotation down to any intervening cells between the end points.
            # This situation can get almost arbitrarily complicated from a geometric perspective, but most cases will be fairly simple. Consider the following scenarios:

            # Add the annotation to all cells between the endpoints. This will almost never happen, with most annotations residing in a single cell, as shown here:
            # +--------+
            # |  END   |
            # | o----o |
            # | POINTS |
            # +--------+
            # Furthermore, most of the rare two-cell annotations will merely cross the boundary between abutting tree cells, as shown here:
            # +--------+--------+
            # |  END   |  END   |
            # |    o---|---o    |
            # | POINT  | POINT  |
            # +--------+--------+
            # However, if the tree subdivides small enough, it is theoretically possible for one or more cells to "open up" between the endpoints.
            # +--------+--------+--------+
            # |        | INTER- |        |
            # |    o---|-VENING-|---o    |
            # |        |  CELL  |        |
            # +--------+--------+--------+
            # Or:
            # +--------+--------+--------+--------+
            # |        | INTER- |        |        |
            # |    o---|-VENING-|--------|---o    |
            # |        |  CELLS |        |        |
            # +--------+--------+--------+--------+
            # One way to handle this situation is to traverse the line segment connecting the endpoints at an interval that fits within the current cell size, testing points along the segment.
            # For any point between the endpoints that falls into a new child cell, add the annotation to that child as well as the two endpoints initially processed (the PRE and POST locations).
            # However, there are idiosyncratic geometric situations in which interval testing can fail to discover an intervening cell, such as when the line segment cuts across the corner or edge of a cell.
            # Such a cut could be arbitrarily small such that no discrete interval is guaranteed to find such intersections. Observe:
            # +--------+--------+
            # |    o   | MISSED |
            # |      \ |  CELL  |
            # |        \        |
            # +--------+-\------+
            # |        |   \    |
            # |        |     o  |
            # |        |        |
            # +--------+--------+
            # A smoothly integrated solution involving more "complex math" (ooga booga) would have to be applied to catch these more problematic cases.
            # It's worth asking how important this is.

            debug = False  # row_idx < 1

            spatial_pt_columns = config['DATA_CONFIG']['spatial_pt_columns']
            pt_tree_level_child_cell_local_ids = set()
            for pt_desc, location in spatial_pt_columns.items():
                pt_tree_level_child_cell_local_id = (
                    1 if row[location['x']] > cell_bounds_mid[0] else 0,
                    1 if row[location['y']] > cell_bounds_mid[1] else 0,
                    1 if row[location['z']] > cell_bounds_mid[2] else 0,
                )

                if pt_tree_level_child_cell_local_id in pt_tree_level_child_cell_local_ids:
                    num_cell_dups += 1
                else:
                    num_cell_nondups += 1
                    pt_tree_level_child_cell_local_ids.add(pt_tree_level_child_cell_local_id)
                    x, y, z = pt_tree_level_child_cell_local_id
                    if concat_method == "dataframe":
                        annos_children_levels[x][y][z] = pd.concat([annos_children_levels[x][y][z], row.to_frame().T])
                    elif concat_method == "list":
                        # annos_children_levels[x][y][z].append(row.to_list())
                        annos_children_levels[x][y][z].append(list(row))
    elif isinstance(annotations_passed_on, RawTable):
        assert concat_method == "list"

        # Optimization: Get column indexes in advance (here) and use them directly instead of calling get_row_field_val() (see below)
        start_timeblock("preretrieval_optimizations")

        # spatial_pt_columns = config['DATA_CONFIG']['spatial_pt_columns']
        # spatial_pt_col_idxs = {}
        # for pt_desc, pt_pos in spatial_pt_columns.items():
        #     spatial_pt_col_idxs[pt_desc] = [
        #         annotations_passed_on.get_col_idx(pt_pos['x']),
        #         annotations_passed_on.get_col_idx(pt_pos['y']),
        #         annotations_passed_on.get_col_idx(pt_pos['z'])
        #     ]
        columns = config['DATA_CONFIG']['columns']
        spatial_pt_columns = config['DATA_CONFIG']['spatial_pt_columns']
        spatial_pt_col_idxs = {}
        for pt_desc, pt_pos in spatial_pt_columns.items():
            spatial_pt_col_idxs[pt_desc] = [
                columns.index(pt_pos['x']),
                columns.index(pt_pos['y']),
                columns.index(pt_pos['z']),
            ]

        # Optimization: avoid continual list accesses by preretrieving the values
        cblx = cell_bounds_low[0]
        cbly = cell_bounds_low[1]
        cblz = cell_bounds_low[2]
        cbhx = cell_bounds_high[0]
        cbhy = cell_bounds_high[1]
        cbhz = cell_bounds_high[2]
        end_timeblock("preretrieval_optimizations")

        for row_idx, line in enumerate(annotations_passed_on.iterlines()):  # RawTable iteration
            start_start_timeblocks("annotations_passed_on_loop_body", "split_line")
            # row = line.split(',')
            reader = csv.reader(io.StringIO(line))
            row = next(reader)

            debug = False  # row_idx < 1

            end_start_timeblocks("split_line", "get_row_field_vals")

            pt_positions = {
                pt_desc: [float(row[pt_x_col_idx]), float(row[pt_y_col_idx]), float(row[pt_z_col_idx])] \
                    for pt_desc, [pt_x_col_idx, pt_y_col_idx, pt_z_col_idx] in spatial_pt_col_idxs.items()
            }

            end_start_timeblocks("get_row_field_vals", "check_bounds")

            # this_anno_pre_pt_in_bounds = True
            # this_anno_post_pt_in_bounds = True
            this_anno_pts_in_bounds = {pt_desc: True for pt_desc in pt_positions}

            # It's okay for some of the annotation's points to be outside the cell bounds.
            # Depending on design, it may be okay for all points to be outside the cell bounds,
            # such as when a line (segment) extends all the way through a cell, ending outside the cell on opposite sides.
            # At the current time, that design is not supported and at least one point should be within the cell bounds.
            for pt_desc, [pt_pos_x, pt_pos_y, pt_pos_z] in pt_positions.items():
                if pt_pos_x < cblx or pt_pos_x > cbhx or \
                    pt_pos_y < cbly or pt_pos_y > cbhy or \
                    pt_pos_z < cblz or pt_pos_z > cbhz:
                    this_anno_pts_in_bounds[pt_desc] = False

            # if pre_pt_pos_x < cblx or pre_pt_pos_x > cbhx or \
            #     pre_pt_pos_y < cbly or pre_pt_pos_y > cbhy or \
            #     pre_pt_pos_z < cblz or pre_pt_pos_z > cbhz:
            #     this_anno_pre_pt_in_bounds = False
                # The following warning message is too numerous, so just count the occurrences and report them after the loop
                # logging.warning(f"WARNING! Row {row_idx}. Pre point resides outside tree cell bounds:\n  [{pre_pt_pos_x}, {pre_pt_pos_y}, {pre_pt_pos_z}] outside\n  {cell_bounds_low}, {cell_bounds_high}\n  (this is okay if the opposing point is inside the bounds)")

            # if post_pt_pos_x < cblx or post_pt_pos_x > cbhx or \
            #     post_pt_pos_y < cbly or post_pt_pos_y > cbhy or \
            #     post_pt_pos_z < cblz or post_pt_pos_z > cbhz:
            #     this_anno_post_pt_in_bounds = False
                # The following warning message is too numerous, so just count the occurrences and report them after the loop
                # logging.warning(f"WARNING! Row {row_idx}. Post point resides outside tree cell bounds:\n  [{post_pt_pos_x}, {post_pt_pos_y}, {post_pt_pos_z}] outside\n  {cell_bounds_low}, {cell_bounds_high}\n  (this is okay if the opposing point is inside the bounds)")

            num_pts_in_bounds = sum([1 if v else 0 for v in this_anno_pts_in_bounds.values()])
            if num_pts_in_bounds == 0:
                logging.error(f"ERROR! (BB shrassr()) Row {row_idx}. All annotation spatial indexing points reside outside tree cell bounds: {cell_bounds_low}, {cell_bounds_high}")
                raise RuntimeError(f"BB shrassr()")

            # if not this_anno_pre_pt_in_bounds and not this_anno_post_pt_in_bounds:
            #     logging.error(f"ERROR! (BB shrassr()) Row {row_idx}. Both Pre and Post points reside outside tree cell bounds:\n  [{pre_pt_pos_x}, {pre_pt_pos_y}, {pre_pt_pos_z}] and [{post_pt_pos_x}, {post_pt_pos_y}, {post_pt_pos_z}] outside\n  {cell_bounds_low}, {cell_bounds_high}")
            #     raise RuntimeError(f"BB shrassr()")

            # end_timeblock("check_bounds")
            end_start_timeblocks("check_bounds", "process_pre_point")

            this_anno_pts_propagated_to_child = {pt_desc: False for pt_desc in pt_positions}
            tree_level_child_cell_local_ids = set()

            for pt_desc, [pt_pos_x, pt_pos_y, pt_pos_z] in pt_positions.items():
                # Only process annotation points that are inside the current cell. The remaining points will be handled within their own cells.
                if this_anno_pts_in_bounds[pt_desc]:
                    pt_tree_level_child_cell_local_id = (
                        1 if pt_pos_x > cell_bounds_mid[0] else 0,
                        1 if pt_pos_y > cell_bounds_mid[1] else 0,
                        1 if pt_pos_z > cell_bounds_mid[2] else 0,
                    )

                    if pt_tree_level_child_cell_local_id in tree_level_child_cell_local_ids:
                        # Don't propagate an annotation to the same child cell multiple times if its multiple points fall within the same child cell
                        num_cell_dups += 1
                    else:
                        start_timeblock("check_subtree_bounds_DEBUG")

                        tree_level_child_cell_local_ids.add(pt_tree_level_child_cell_local_id)

                        subtree_cell_bounds = determine_subtree_bounds(
                            cell_bounds_low, cell_bounds_mid, cell_bounds_high,
                            [pt_tree_level_child_cell_local_id[0], pt_tree_level_child_cell_local_id[1], pt_tree_level_child_cell_local_id[2]]
                        )
                        if pt_pos_x < subtree_cell_bounds[0][0] or pt_pos_x > subtree_cell_bounds[1][0] or \
                            pt_pos_y < subtree_cell_bounds[0][1] or pt_pos_y > subtree_cell_bounds[1][1] or \
                            pt_pos_z < subtree_cell_bounds[0][2] or pt_pos_z > subtree_cell_bounds[1][2]:
                            logging.error("ERROR!")
                            # start_timeblock("call_determine_pt_subtreecell_4")
                            # determine_pt_subtreecell(
                            #     cell_bounds_low, cell_bounds_mid, cell_bounds_high,
                            #     [pre_pt_pos_x, pre_pt_pos_y, pre_pt_pos_z],
                            #     f"{id_} {post_pt_root_id} PRE")
                            # end_timeblock("call_determine_pt_subtreecell_4")

                            raise RuntimeError(f"shrassr(): Point about to be assigned to wrong child cell (or it's outside the parent bounds, which should have been caught earlier):\n  Lo: {cell_bounds_low}\n  Md: {cell_bounds_mid}\n  Hi: {cell_bounds_high}\n  Pt: [{pt_pos_x}, {pt_pos_y}, {pt_pos_z}]\n  ST: {subtree_cell_bounds}")

                        end_timeblock("check_subtree_bounds_DEBUG")
                        # end_start_timeblocks("check_subtree_bounds_DEBUG", "concatenate_dataframe")

                        annos_children_levels[pt_tree_level_child_cell_local_id[0]][pt_tree_level_child_cell_local_id[1]][pt_tree_level_child_cell_local_id[2]].append(line)
                        # end_timeblock("concatenate_dataframe")

                        this_anno_pts_propagated_to_child[pt_desc] = True

            end_start_timeblocks("process_pre_point", "process_post_point")

            end_timeblock("process_post_point")

            num_pts_propagated_to_child = sum([1 if v else 0 for v in this_anno_pts_propagated_to_child.values()])
            if num_pts_propagated_to_child == len(pt_positions):
                num_cell_nondups += 1

            # if pre_point_propagated_to_child and post_point_propagated_to_child:
            #     num_cell_nondups += 1

            if num_pts_propagated_to_child == 0:
                raise RuntimeError(f"No annotation spatial indexing points were propagated to a child")

            # if not pre_point_propagated_to_child and not post_point_propagated_to_child:
            #     logging.info(f"post_pt_tree_level_child_cell_local_id ==? pre_pt_tree_level_child_cell_local_id: {post_pt_tree_level_child_cell_local_id} ==? {pre_pt_tree_level_child_cell_local_id}: {post_pt_tree_level_child_cell_local_id == pre_pt_tree_level_child_cell_local_id}")
            #     raise RuntimeError(f"Neither the PRE nor the POST point was propagated to a child: [{pre_pt_pos_x}, {pre_pt_pos_y}, {pre_pt_pos_z}], [{post_pt_pos_x}, {post_pt_pos_y}, {post_pt_pos_z}]")

            end_timeblock("annotations_passed_on_loop_body")
    elif isinstance(annotations_passed_on, list):  # A list of Annotation subclass objects
        cblx = cell_bounds_low[0]
        cbly = cell_bounds_low[1]
        cblz = cell_bounds_low[2]
        cbhx = cell_bounds_high[0]
        cbhy = cell_bounds_high[1]
        cbhz = cell_bounds_high[2]

        num_deep_copies, num_ref_copies = 0, 0
        for anno_idx, annotation in enumerate(annotations_passed_on):
            start_start_timeblocks("annotations_loop", "gather_in_bounds")
            pt_positions = annotation.get_all_points()
            # logging.info(f"Annotation {anno_idx} points: {pt_positions}")
            this_anno_pts_in_bounds = [True] * len(pt_positions)
            for pt_idx, [pt_pos_x, pt_pos_y, pt_pos_z] in enumerate(pt_positions):
                if pt_pos_x < cblx or pt_pos_x > cbhx or \
                    pt_pos_y < cbly or pt_pos_y > cbhy or \
                    pt_pos_z < cblz or pt_pos_z > cbhz:
                    this_anno_pts_in_bounds[pt_idx] = False
            # logging.info(f"Annotation {anno_idx} points in bounds: {this_anno_pts_in_bounds}")

            num_pts_in_bounds = sum([1 if v else 0 for v in this_anno_pts_in_bounds])
            if num_pts_in_bounds == 0:
                logging.error(f"ERROR! (BB shrassr()) Annotation {anno_idx}. All annotation spatial indexing points reside outside tree cell bounds: {cell_bounds_low}, {cell_bounds_high}")
                raise RuntimeError(f"BB shrassr()")

            end_start_timeblocks("gather_in_bounds", "run_annotation_points_loop")

            this_anno_pts_propagated_to_child = [False] * len(pt_positions)
            tree_level_child_cell_local_ids = set()
            for pt_idx, [pt_pos_x, pt_pos_y, pt_pos_z] in enumerate(pt_positions):
                start_timeblock("annotation_points_loop")
                if this_anno_pts_in_bounds[pt_idx]:
                    pt_tree_level_child_cell_local_id = (
                        1 if pt_pos_x > cell_bounds_mid[0] else 0,
                        1 if pt_pos_y > cell_bounds_mid[1] else 0,
                        1 if pt_pos_z > cell_bounds_mid[2] else 0,
                    )

                    # logging.info(f"Annotation {anno_idx} point child cell: {pt_idx} {pt_tree_level_child_cell_local_id}")

                    if pt_tree_level_child_cell_local_id in tree_level_child_cell_local_ids:
                        # Don't propagate an annotation to the same child cell multiple times if its multiple points fall within the same child cell
                        num_cell_dups += 1
                    else:
                        start_start_timeblocks("annotation_non_dup", "determine_subtree_bounds")
                        tree_level_child_cell_local_ids.add(pt_tree_level_child_cell_local_id)

                        subtree_cell_bounds = determine_subtree_bounds(
                            cell_bounds_low, cell_bounds_mid, cell_bounds_high,
                            [pt_tree_level_child_cell_local_id[0], pt_tree_level_child_cell_local_id[1], pt_tree_level_child_cell_local_id[2]]
                        )
                        end_start_timeblocks("determine_subtree_bounds", "bounds_confirmation_DEBUG")
                        if pt_pos_x < subtree_cell_bounds[0][0] or pt_pos_x > subtree_cell_bounds[1][0] or \
                            pt_pos_y < subtree_cell_bounds[0][1] or pt_pos_y > subtree_cell_bounds[1][1] or \
                            pt_pos_z < subtree_cell_bounds[0][2] or pt_pos_z > subtree_cell_bounds[1][2]:
                            logging.error("ERROR!")
                            raise RuntimeError(f"shrassr(): Point about to be assigned to wrong child cell (or it's outside the parent bounds, which should have been caught earlier):\n  Lo: {cell_bounds_low}\n  Md: {cell_bounds_mid}\n  Hi: {cell_bounds_high}\n  Pt: [{pt_pos_x}, {pt_pos_y}, {pt_pos_z}]\n  ST: {subtree_cell_bounds}")

                        end_start_timeblocks("bounds_confirmation_DEBUG", "deepcopy_annotation")
                        # Making a deep copy of the annotation to pass to the next tree level is one of the most time-consuming operations in the entire pipeline.
                        # At the same time, most annotations won't be split into multiple child cells anyway,
                        # and deep copying an annotation that won't be split should be completely unnecessary.
                        # So, don't bother copying the annotation for the first propagation,
                        # only for any rare necessary split for subsequent points into differing child tree cells.
                        if pt_idx > 0:
                            # anno_copy = copy.deepcopy(annotation)  # See next line
                            anno_copy = annotation.copy()  # A custom deep copy function is MUCH faster than relying on the general purpose deepycopy module
                            num_deep_copies += 1
                        else:
                            anno_copy = annotation
                            num_ref_copies += 1
                        end_start_timeblocks("deepcopy_annotation", "append_annotation")
                        annos_children_levels[pt_tree_level_child_cell_local_id[0]][pt_tree_level_child_cell_local_id[1]][pt_tree_level_child_cell_local_id[2]].append(anno_copy)
                        end_timeblock("append_annotation")

                        this_anno_pts_propagated_to_child[pt_idx] = True
                        end_timeblock("annotation_non_dup")

                end_timeblock("annotation_points_loop")

            end_start_timeblocks("run_annotation_points_loop", "gather_num_pts_propagated_to_child")
            num_pts_propagated_to_child = sum([1 if v else 0 for v in this_anno_pts_propagated_to_child])
            if num_pts_propagated_to_child == len(pt_positions):
                num_cell_nondups += 1

            if num_pts_propagated_to_child == 0:
                raise RuntimeError(f"No annotation spatial indexing points were propagated to a child")

            end_end_timeblocks("gather_num_pts_propagated_to_child", "annotations_loop")

        if verbose:
            logging.info(f"Total duplicates, reference copies, deep copies: {num_cell_dups:>12} {num_ref_copies:>12} {num_deep_copies:>12}")

    if concat_method == "list":
        # Convert the lists to DataFrames
        if isinstance(annotations, pd.DataFrame):
            assert False
            start_timeblock("convert_lists_to_tables")
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        annos_children_levels[x][y][z] = pd.DataFrame(annos_children_levels[x][y][z], columns=annotations.columns)
            end_timeblock("convert_lists_to_tables")
        elif isinstance(annotations, RawTable):
            start_timeblock("convert_lists_to_tables_and_confirm_children_in_cells")
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        annos_children_levels[x][y][z] = RawTable(annos_children_levels[x][y][z], header=annotations.header)

                        start_timeblock("confirm_subtree_bounds_DEBUG")
                        # Currently disabled to expedite performance. Reenable if future development impacts this code in concerning ways.
                        # subtree_cell_bounds = determine_subtree_bounds(
                        #     cell_bounds_low,
                        #     cell_bounds_mid,
                        #     cell_bounds_high,
                        #     [x, y, z]
                        # )
                        # confirm_all_annotations_within_bounds(src_loc, annos_children_levels[x][y][z], subtree_cell_bounds[0], subtree_cell_bounds[1], "QQQ")
                        end_timeblock("confirm_subtree_bounds_DEBUG")
            end_timeblock("convert_lists_to_tables_and_confirm_children_in_cells")
        elif isinstance(annotations, list):  # A list of Annotation subclass objects
            start_timeblock("confirm_children_in_cells")
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        start_timeblock("confirm_subtree_bounds_DEBUG")
                        # Currently disabled to expedite performance. Reenable if future development impacts this code in concerning ways.
                        # subtree_cell_bounds = determine_subtree_bounds(
                        #     cell_bounds_low,
                        #     cell_bounds_mid,
                        #     cell_bounds_high,
                        #     [x, y, z]
                        # )
                        # confirm_all_annotations_within_bounds(src_loc, annos_children_levels[x][y][z], subtree_cell_bounds[0], subtree_cell_bounds[1], "QQQ")
                        end_timeblock("confirm_subtree_bounds_DEBUG")
            end_timeblock("confirm_children_in_cells")

    end_start_timeblocks("subdivide_subtree", "tally_subtree_DEBUG_1")

    if verbose:
        logging.info(f"Num rows stored in this cell:                                         {len(annos_this_level):>11,}")
        logging.info(f"Total num rows to be passed on:                                       {len(annotations_passed_on):>11,}")
        child_row_tally = 0
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    child_row_tally += len(annos_children_levels[x][y][z])
        logging.info(f"Tallied num rows passed to children (should be <=2X input data size): {child_row_tally:>11,}")
        nondup_percentage = f"{ num_cell_nondups / (num_cell_dups + num_cell_nondups) * 100:.2f}%" if num_cell_dups + num_cell_nondups > 0 else "         NA"
        logging.info(f"Total annotation point tree cell duplicates & non-duplicates & nondup-%:           {num_cell_dups:>11,}    {num_cell_nondups:>11,}    {nondup_percentage}")

    end_end_timeblocks("tally_subtree_DEBUG_1", "select_holdout_rows_and_subdivide_subtree_rows()")

    return annos_this_level, annos_children_levels, num_cell_nondups

def save_this_treecell_data_as_csv(annos_this_level, subdir, subsplit_id, num_splits, split_id, tree_level, tree_level_cell_id, shard_hex):
    start_timeblock("save_this_treecell_data_as_csv()")

    assert isinstance(annos_this_level, RawTable)

    filepath = f"{subdir}annotations_one_treecell__subsplit-{subsplit_id:02}__split-{split_id:03}@{num_splits}__treelevel-{tree_level:02}__treelevelcellid-{','.join([f'{v:0>3}' for v in tree_level_cell_id])}__shard-{shard_hex}.csv"
    # logging.info(f"Writing this cell's CSV file with {len(annos_this_level)} rows")
    # logging.info(f"Writing this cell's CSV file with {len(annos_this_level)} rows: {filepath}")
    annos_this_level.to_csv_to_disk_or_ram_data_pond(filepath, ram_data_pond, index=False, header=False)

    end_timeblock("save_this_treecell_data_as_csv()")

    return filepath

def save_this_treecell_data_as_object(annos_this_level, subdir, subsplit_id, num_splits, split_id, tree_level, tree_level_cell_id, shard_hex):
    '''
    This function does something rather tricky.
    The RAMDataPond was originally designed as a RAM disk, a place to store quasi-file-like-objects
    (it only supports char strings at the time of this writing, not even byte arrays) in RAM instead of on disk.
    However, in truth, the RAMDataPond is little more than a key/value store (a dictionary) where keys are virtual "absolute file paths"
    and values were the strings being stored (i.e., quasi-file-like-object contents).
    Therefore, so long as we are careful to exclusively use the RAMDataPond as a pure key/value store,
    we can store non-string objects in it, such as our list of Annotation objects.
    We can then retrieve them from the RAMDataPond later via the same key-retrieval (dictionary retrieval) without any trouble,
    so long as we are careful not to trigger any of the RAMDataPond's disk-IO or string-handling routines.
    TODO: Generalize the RAMDataPond class to more formally support this key/value RAM data store approach...or alternatively, alter this index-builder to not use the RAMDataPond to keep track of data as it moves through the index-building process.
    '''
    start_timeblock("save_this_treecell_data_as_object()")

    assert isinstance(annos_this_level, list)  # A list of Annotation subclass objects

    object_key = f"{subdir}annotations_one_treecell__subsplit-{subsplit_id:02}__split-{split_id:03}@{num_splits}__treelevel-{tree_level:02}__treelevelcellid-{','.join([f'{v:0>3}' for v in tree_level_cell_id])}__shard-{shard_hex}"
    # logging.info(f"Storing this cell's annotation list with {len(annos_this_level)} annotations")
    # logging.info(f"Storing this cell's annotation list with {len(annos_this_level)} annotations: {object_key}")
    ram_data_pond.write_to_disk_or_ram_data_pond(object_key, annos_this_level)

    end_timeblock("save_this_treecell_data_as_object()")

    return object_key

def save_this_treecell_data(src_loc, annos_this_level, subsplit_id, num_splits, split_id, tree_level, tree_level_cell_id, tree_level_shard_histograms, morton_code, cell_bounds_low, cell_bounds_high):
    if len(annos_this_level) <= 0:
        return

    start_timeblock("save_this_treecell_data()")

    shard_hex = get_shard_hex(tree_level, tree_level_cell_id)
    tree_level_shard_histograms[tree_level][tuple(tree_level_cell_id)] = shard_hex
    subdir = f"{results_loc}completed_treecells/annotations_one_treecell__subsplit-{subsplit_id:02}__split-{split_id:03}@{num_splits}__treelevel-{tree_level:02}__treelevelcellid-{','.join([f'{v:0>3}' for v in tree_level_cell_id])}__shard-{shard_hex}/"
    # os.makedirs(subdir, exist_ok=True)
    if isinstance(annos_this_level, pd.DataFrame):
        raise ValueError("DataFrame is no longer supported and not expected to be received at this point in the code")
    elif isinstance(annos_this_level, RawTable):
        ramdatapond_key = save_this_treecell_data_as_csv(annos_this_level, subdir, subsplit_id, num_splits, split_id, tree_level, tree_level_cell_id, shard_hex)
        confirm_all_annotations_within_bounds(src_loc, ramdatapond_key, cell_bounds_low, cell_bounds_high, "WWW", force_ram=True)
    elif isinstance(annos_this_level, list):  # A list of Annotation subclass objects
        ramdatapond_key = save_this_treecell_data_as_object(annos_this_level, subdir, subsplit_id, num_splits, split_id, tree_level, tree_level_cell_id, shard_hex)
        confirm_all_annotations_within_bounds(src_loc, annos_this_level, cell_bounds_low, cell_bounds_high, "WWW", force_ram=True)

    start_timeblock("write_cell_bounds_metadata_file")
    filepath = f"{subdir}oct_tree_cell_bounds.txt"
    # logging.info(f"Writing one cell bounds file:\n  For file: {ramdatapond_key}\n  File: {filepath}\n  Cell bounds: {cell_bounds_low} {cell_bounds_high}")
    ram_data_pond.write_to_disk_or_ram_data_pond(filepath, str([cell_bounds_low, cell_bounds_high]))

    end_start_timeblocks("write_cell_bounds_metadata_file", "write_subtree_metadata_file")

    filepath = f"{subdir}oct_tree__tree_cell__info.txt"
    # logging.info(f"Writing one cell info file (for debugging): {filepath}")
    s = \
f"""Num splits:         {num_splits}
Split id (1-based): {split_id}
Tree level:         {tree_level}
Tree level cell id: {tree_level_cell_id}
Morton code:        {morton_code}
Num rows:           {len(annos_this_level)}
Cell bounds:        {[cell_bounds_low, cell_bounds_high]}
"""
    ram_data_pond.write_to_disk_or_ram_data_pond(filepath, s)

    end_end_timeblocks("write_subtree_metadata_file", "save_this_treecell_data()")

    return shard_hex

def save_subtrees(src_loc, annos_children_levels, subsplit_id, num_splits, split_id, tree_level, tree_level_cell_id, cell_bounds_low, cell_bounds_mid, cell_bounds_high):
    '''
    Save the subtree files that will be passed to deeper levels of the tree for further processing
    '''
    start_timeblock("save_subtrees()")

    verbose = False
    if verbose:
        logging.info("\n\nBEWARE! verbose is True in save_subtrees()\n\n")

    # logging.info("Writing <= 8 subtree files")
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                annos_child_levels = annos_children_levels[x][y][z]
                # It's crucial to only write nonempty DataFrames. The eventual lack of any remaining tree child files will indicate that this stage of processing is complete.
                if len(annos_child_levels) <= 0:
                    continue

                start_timeblock("save_subtree_header_misc")

                tree_child_level_cell_id = [
                    tree_level_cell_id[0] * 2 + x,
                    tree_level_cell_id[1] * 2 + y,
                    tree_level_cell_id[2] * 2 + z,
                ]
                subtree_dir = f"{results_loc}subtrees/subsplit-{subsplit_id:02}__split-{split_id:03}@{num_splits}__treelevel-{tree_level+1:02}__subtree-{','.join([f'{v:0>3}' for v in tree_child_level_cell_id])}/"
                # os.makedirs(subtree_dir, exist_ok=True)

                shard_hex = get_shard_hex(tree_level+1, tree_child_level_cell_id)

                object_key = f"{subtree_dir}annotations_one_subtree__split-{split_id:03}@{num_splits}__treelevel-{tree_level+1:02}__treelevelcellid-{','.join([f'{v:0>3}' for v in tree_child_level_cell_id])}__shard-{shard_hex}"
                if verbose:
                    logging.info(f"  Writing subtree file of {len(annos_child_levels):>11,} annotations: {object_key}")

                end_start_timeblocks("save_subtree_header_misc", "save_subtree_data")

                if isinstance(annos_child_levels, pd.DataFrame):
                    raise ValueError("DataFrame is no longer supported and not expected to be received at this point in the code")
                elif isinstance(annos_child_levels, RawTable):
                    filepath = object_key + ".csv"
                    annos_child_levels.to_csv_to_disk_or_ram_data_pond(filepath, ram_data_pond, index=False)
                elif isinstance(annos_child_levels, list):  # A list of Annotation subclass objects
                    # See important note regarding RAMDataPond usage at the top of save_this_treecell_data_as_object()
                    ram_data_pond.write_to_disk_or_ram_data_pond(object_key, annos_child_levels)

                end_start_timeblocks("save_subtree_data", "write_cell_bounds_metadata_file")

                if annos_child_levels is not None and len(annos_child_levels) > 0:
                    subtree_cell_bounds = determine_subtree_bounds(
                        cell_bounds_low,
                        cell_bounds_mid,
                        cell_bounds_high,
                        [x, y, z]
                    )
                    # logging.info(f"  subtree_cell_bounds: {subtree_cell_bounds}")
                    # logging.info(f"  subtree_cell_bounds width: {subtree_cell_bounds[1][0] - subtree_cell_bounds[0][0]} {subtree_cell_bounds[1][1] - subtree_cell_bounds[0][1]} {subtree_cell_bounds[1][2] - subtree_cell_bounds[0][2]}")

                    if isinstance(annos_child_levels, RawTable):
                        confirm_all_annotations_within_bounds(src_loc, filepath, subtree_cell_bounds[0], subtree_cell_bounds[1], "SSS", force_ram=True)
                    elif isinstance(annos_child_levels, list):  # A list of Annotation subclass objects
                        confirm_all_annotations_within_bounds(src_loc, annos_child_levels, subtree_cell_bounds[0], subtree_cell_bounds[1], "SSS", force_ram=True)

                    filepath = f"{subtree_dir}oct_tree_cell_bounds.txt"
                    # logging.info(f"  Writing child cell bounds file:\n    For key: {subtree_dir}annotations_one_subtree__split-{split_id:03}@{num_splits}__treelevel-{tree_level+1:02}__treelevelcellid-{','.join([f'{v:0>3}' for v in tree_child_level_cell_id])}__shard-{shard_hex}\n    File: {filepath}\n    Cell bounds: {subtree_cell_bounds}")
                    # with open(filepath, 'w') as f:
                    #     f.write(str(subtree_cell_bounds))
                    ram_data_pond.write_to_disk_or_ram_data_pond(filepath, str(subtree_cell_bounds))

                end_start_timeblocks("write_cell_bounds_metadata_file", "write_subtree_metadata_file")

                filepath = f"{subtree_dir}oct_tree__subtree__info.txt"
                # logging.info(f"  Writing one cell info file (for debugging): {filepath}")
                s = \
f"""Num splits:         {num_splits}
Split id (1-based): {split_id}
Tree level:         {tree_level}
Tree child level:   {tree_level+1}
Tree level cell id: {tree_level_cell_id}
Tree child cell id: {tree_child_level_cell_id}
Child num rows:     {len(annos_child_levels)}
Child cell bounds:  {subtree_cell_bounds}
"""
                # with open(filepath, 'w') as f:
                #     f.write(s)
                ram_data_pond.write_to_disk_or_ram_data_pond(filepath, s)
                end_timeblock("write_subtree_metadata_file")

    end_timeblock("save_subtrees()")

def confirm_all_annotations_within_bounds(src_loc, annotations_file_or_rawtable, cell_bounds_low, cell_bounds_high, label, force_ram=False):

    # In the interests of expediting peformance, only run this function during periods of development when this issue may be impacted by code changes
    return

    start_timeblock("confirm_all_annotations_within_bounds() DEBUG")
    # logging.info(f"\nconfirm_all_annotations_within_bounds() ({label})    src_loc: {src_loc}    data_loc: {data_loc}")
    # logging.info(f"  Cell bounds: {cell_bounds_low} {cell_bounds_high}")

    columns = config['DATA_CONFIG']['columns']
    id_column = config['DATA_CONFIG']['id_column']
    # logging.info(f"id_column: {id_column}")
    if id_column is None:
        logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {columns[0]}.")

    if isinstance(annotations_file_or_rawtable, str):
        if annotations_file_or_rawtable.endswith(".csv"):
            # logging.info(f"  Confirming file: {annotations_file_or_rawtable}")
            lines = ram_data_pond.read_splitlines_from_disk_or_ram_data_pond(annotations_file_or_rawtable, None, None, src_loc==data_loc if not force_ram else False)
            # header_present = lines[0].startswith(columns[0 if id_column is not None else 1])
            cols = lines[0].split(',')
            header_present = cols[0] == columns[0] or cols[1] == columns[1]  # Sometimes pandas leaves the first column in the header row
            annotations = RawTable(lines, columns if not header_present else None)
        elif annotations_file_or_rawtable.endswith(".parquet"):
            logging.info("confirm_all_annotations_within_bounds() Parquet in-bounds confirmation not implemented yet")
            end_timeblock("confirm_all_annotations_within_bounds() DEBUG")
            return
    elif isinstance(annotations_file_or_rawtable, RawTable):
        # logging.info("  Confirming a RawTable, not a file")
        annotations = annotations_file_or_rawtable
    elif isinstance(annotations_file_or_rawtable, list):  # A list of Annotation subclass objects
        annotations = annotations_file_or_rawtable

    if isinstance(annotations_file_or_rawtable, str) or isinstance(annotations_file_or_rawtable, RawTable):
        spatial_pt_columns = config['DATA_CONFIG']['spatial_pt_columns']

        # logging.info(f"  Num rows: {len(annotations)}")
        n_all_pt_out_of_bounds = 0
        for row_idx, line in enumerate(annotations.iterlines()):
            # row = line.split(',')
            reader = csv.reader(io.StringIO(line))
            row = next(reader)

            all_pts_out_of_bounds = True
            for pt_desc, location in spatial_pt_columns.items():
                pt_pos_x = float(annotations.get_row_field_val(row, location['x']))
                pt_pos_y = float(annotations.get_row_field_val(row, location['y']))
                pt_pos_z = float(annotations.get_row_field_val(row, location['z']))

                out_of_bounds = False
                if pt_pos_x < cell_bounds_low[0] or pt_pos_x > cell_bounds_high[0]:
                    out_of_bounds = True
                elif pt_pos_y < cell_bounds_low[1] or pt_pos_y > cell_bounds_high[1]:
                    out_of_bounds = True
                elif pt_pos_z < cell_bounds_low[2] or pt_pos_z > cell_bounds_high[2]:
                    out_of_bounds = True
                if not out_of_bounds:
                    all_pts_out_of_bounds = False

            if all_pts_out_of_bounds:
                n_all_pt_out_of_bounds += 1
                logging.error(f"ERROR! (AA caswb {label}) Row {row_idx}, err {n_all_pt_out_of_bounds}. All spatial indexing points reside outside tree cell bounds:\n  Dimensions or axis errors: {cell_bounds_low}, {cell_bounds_high}")
                raise RuntimeError(f"AA caswb {label}")
    elif isinstance(annotations_file_or_rawtable, list):
        n_all_pt_out_of_bounds = 0
        for anno_idx, annotation in enumerate(annotations):
            all_points = annotation.get_all_points()
            all_pts_out_of_bounds = True
            for pt_pos_x, pt_pos_y, pt_pos_z in all_points:
                out_of_bounds = False
                if pt_pos_x < cell_bounds_low[0] or pt_pos_x > cell_bounds_high[0]:
                    out_of_bounds = True
                elif pt_pos_y < cell_bounds_low[1] or pt_pos_y > cell_bounds_high[1]:
                    out_of_bounds = True
                elif pt_pos_z < cell_bounds_low[2] or pt_pos_z > cell_bounds_high[2]:
                    out_of_bounds = True
                if not out_of_bounds:
                    all_pts_out_of_bounds = False

            if all_pts_out_of_bounds:
                n_all_pt_out_of_bounds += 1
                logging.error(f"ERROR! (AA caswb {label}) Annotation {anno_idx}, err {n_all_pt_out_of_bounds}. All spatial indexing points reside outside tree cell bounds:\n  Dimensions or axis errors: {cell_bounds_low}, {cell_bounds_high}")
                raise RuntimeError(f"AA caswb {label}")

    # logging.info("  confirm_all_annotations_within_bounds() Done\n")
    # logging.info("")
    end_timeblock("confirm_all_annotations_within_bounds() DEBUG")

def add_tree_level_cell_id_to_saved_annotations(annos_this_level, tree_level_cell_id):
    start_timeblock("add_tree_level_cell_id_to_saved_annotations()")

    treecell_index = '_'.join([str(v) for v in tree_level_cell_id])

    if isinstance(annos_this_level, pd.DataFrame):
        annos_this_level["treecell_index"] = treecell_index
    elif isinstance(annos_this_level, RawTable):
        annos_this_level.add_column("treecell_index", treecell_index)
    elif isinstance(annos_this_level, list):  # A list of Annotation subclass objects
        for anno in annos_this_level:
            anno.add_treecell_index(treecell_index, "csv")

    end_timeblock("add_tree_level_cell_id_to_saved_annotations()")

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

def process_one_treecell_input_file_or_dir(subsplit_id, subsplit_range_row_start, subsplit_range_row_end, file_idx, src_loc, input_src,
        cell_bounds_low, cell_bounds_high,
        num_splits, split_id, tree_level, tree_level_cell_id, tree_level_shard_histograms):
    # start_start_timeblocks("process_one_treecell_input_file_or_dir()", "preprocess")
    # start_timeblock("process_one_treecell_input_file_or_dir()")
    start_timeblock("preprocess")

    num_subsplits = config['DATA_CONFIG']['data_size'][6]

    # verbose = file_idx < 5 or (file_idx < 1000 and file_idx % 200 == 0) or (file_idx < 100000 and file_idx % 20000 == 0)
    verbose = file_idx == 0
    if verbose:
        logging.info(f"\n\nBEWARE! verbose is True in process_one_treecell_input_file_or_dir() for file index {file_idx}\n")

    if verbose:
        logging.info("")
        logging.info(f"Split id: {split_id}")
        logging.info(f"Tree level, tree level cell id: {tree_level} {tree_level_cell_id}")
        logging.info(f"Cell bounds: {cell_bounds_low} {cell_bounds_high}")
    cell_width_x, cell_width_y, cell_width_z = cell_bounds_high[0] - cell_bounds_low[0], cell_bounds_high[1] - cell_bounds_low[1], cell_bounds_high[2] - cell_bounds_low[2]
    dimensions = config['DATA_CONFIG']['dimensions']
    if verbose:
        logging.info(f"Dimensions: {dimensions}")
    cell_width_x_nm, cell_width_y_nm, cell_width_z_nm = cell_width_x * dimensions['x'][0], cell_width_y * dimensions['y'][0], cell_width_z * dimensions['z'][0]
    cell_width_x_um, cell_width_y_um, cell_width_z_um = cell_width_x_nm / 1000, cell_width_y_nm / 1000, cell_width_z_nm / 1000
    cell_volume_um = cell_width_x_um * cell_width_y_um * cell_width_z_um
    annotation_micron_limit = config['DATA_CONFIG']['spatial_limit']['max_annotations_per_cubic_micron']
    annotation_num_limit = (cell_volume_um * annotation_micron_limit) if annotation_micron_limit is not None else None
    if verbose:
        logging.info(f"Cell width (voxels):      {cell_width_x:13,.1f} {cell_width_y:13,.1f} {cell_width_z:13,.1f}")
        logging.info(f"Cell width (nm):          {cell_width_x_nm:13,.1f} {cell_width_y_nm:13,.1f} {cell_width_z_nm:13,.1f}")
        logging.info(f"Cell width (um):          {cell_width_x_um:13,.1f} {cell_width_y_um:13,.1f} {cell_width_z_um:13,.1f}")
        logging.info(f"Cell volume (um^3):       {cell_volume_um:13,.1f}")
        logging.info(f"Max annotations per cell: {annotation_num_limit if annotation_num_limit is not None else 'no indicated limit (see DATA_CONFIG.spatial_limit.max_annotations_per_cubic_micron)'}")

    grid_dim = 2 ** tree_level
    grid_shape = (grid_dim, grid_dim, grid_dim)
    if verbose:
        logging.info(f"Tree level, grid dimension, grid shape: {tree_level} {grid_dim} {grid_shape}")
    morton_code = utilities.compressed_morton_code(tree_level_cell_id, grid_shape)
    if verbose:
        logging.info(f"Tree level cell id Morton code: {morton_code}")
        logging.info("")

    cell_bounds_mid = [
        (cell_bounds_low[i] + cell_bounds_high[i]) / 2 for i in range(3)
    ]

    end_timeblock("preprocess")

    columns = config['DATA_CONFIG']['columns']
    id_column = config['DATA_CONFIG']['id_column']
    # logging.info(f"id_column: {id_column}")
    if id_column is None:
        logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {columns[0]}.")

    # logging.info(f"Data config indicates data structure: {config['DATA_CONFIG']['structure']}")
    if config['DATA_CONFIG']['structure'] == 'one_annotation_per_row__multiple_points_per_row' or \
        config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row_in_one_field":
        # At tree level 0, this is an on-disk input file (csv or parquet).
        # At deeper tree levels, this is a list of Annotation (subclass) objects.
        assert isinstance(input_src, list) or os.path.isfile(input_src)

        header_present = False  # Assume the header is absent, since it won't be included at deeper tree levels. We only need to check the original input file at level 0.
        if tree_level == 0:
            # Confirm that the header row is or is not present based on the script's circumstances.
            # We don't expect a header from the test file, but do expect it under all other circumstances.
            if verbose:
                try:
                    num_preview_lines = 2
                    logging.info(f"\nBeginning of input file (first {num_preview_lines} lines):")
                    # with open(input_src) as f:
                    #     for i in range(num_preview_lines):
                    #         logging.info(f"  Line {i+1:>2}: " + f.readline().strip())
                    lines = ram_data_pond.read_splitlines_from_disk_or_ram_data_pond(input_src, None, None, src_loc==data_loc)
                    if not lines:
                        # Subsplits are done
                        logging.info("No more data. Subsplits are done.")
                        # end_timeblock("process_one_treecell_input_file_or_dir()")
                        return None
                    for i in range(num_preview_lines):
                        logging.info(f"  Line {i+1:>2}: " + lines[i].strip())
                    logging.info("")
                except Exception as e:
                    logging.info(e)

            # Check for the presence of a header line
            lines = ram_data_pond.read_nlines_from_disk_or_ram_data_pond(input_src, 1, src_loc==data_loc)
            # header_present = lines[0].startswith(columns[0 if id_column is not None else 1])
            cols = lines[0].split(',')
            header_present = cols[0] == columns[0] or cols[1] == columns[1]  # Sometimes pandas leaves the first column in the header row
            logging.info(f"header_present: {header_present}")

        start_timeblock("read_input")

        # annotations = None  # Use Pandas Dataframes
        annotations = []  # Use Python Lists
        if annotations is None:
            assert False, "This code block has not been used for a long time and is becoming incompatible with current work. It requires attention before it can be turned back on."
            if input_src.endswith(".csv"):
                if not header_present:
                    # We shouldn't need a header in a pipeline because the previous capsule should have added it.
                    annotations = pd.read_csv(input_src, names=config['DATA_CONFIG']['columns'], index_col=False)  # Header is explicitly passed in
                else:
                    annotations = pd.read_csv(input_src, index_col=False)  # Header will be inferred from first line of file
                    # logging.info(f"Inferred header: {annotations.columns}")
            elif input_src.endswith(".parquet"):
                annotations = pd.read_parquet(input_src, engine=config['PARQUET_ENGINE'])
        else:
            if tree_level == 0:
                # with open(input_src) as f:
                #     annotations = RawTable(f.read().splitlines(), config['DATA_CONFIG']['columns'] if not header_present else None)
                lines = ram_data_pond.read_splitlines_from_disk_or_ram_data_pond(input_src, subsplit_range_row_start, subsplit_range_row_end, src_loc==data_loc)
                if not lines:
                    logging.info("No more data. Subsplits are done.")
                    end_timeblock("read_input")
                    # end_end_timeblocks("read_input", "process_one_treecell_input_file_or_dir()")
                    end_timeblock("read_input")
                    # end_timeblock("process_one_treecell_input_file_or_dir()")
                    return None
                # header_present = lines[0].startswith(columns[0 if id_column is not None else 1])
                cols = lines[0].split(',')
                header_present = cols[0] == columns[0] or cols[1] == columns[1]  # Sometimes pandas leaves the first column in the header row
                if header_present:
                    logging.info("Header line will be removed")
                    lines = lines[1:]
                # annotations = RawTable(lines, config['DATA_CONFIG']['columns'] if not header_present else None)

                subsplit_range_row_start_str = f"{subsplit_range_row_start:,}" if subsplit_range_row_start is not None else 'None'
                subsplit_range_row_end_str = f"{subsplit_range_row_end:,}" if subsplit_range_row_end is not None else 'None'
                logging.info(f"Subsplit rows for subsplit range {subsplit_range_row_start_str}-{subsplit_range_row_end_str} first 2 rows:\n  {'\n  '.join(lines[:2])}")
                logging.info(f"Subsplit rows for subsplit range {subsplit_range_row_start_str}-{subsplit_range_row_end_str} last 2 rows:\n  {'\n  '.join(lines[-2:])}")

                columns = config['DATA_CONFIG']['columns']
                id_column, id_column_idx = None, None
                if 'id_column' in config['DATA_CONFIG']:
                    id_column = config['DATA_CONFIG']['id_column']
                    # logging.info(f"id_column: {id_column}")
                    if id_column is not None:
                        id_column_idx = columns.index(id_column)
                    else:
                        logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {columns[0]}.")
                        id_column_idx = 0
                        split_size = config['DATA_CONFIG']['data_size'][3]
                        split_subsplit_id_start = (split_id - 1) * split_size + subsplit_range_row_start + 1
                        logging.info(f"split_id, split_size, subsplit_id, split_subsplit_id_start: {split_id} {split_size} {subsplit_id} {split_subsplit_id_start}")
                elif 'id_src' in config['DATA_CONFIG']:
                    id_src = config['DATA_CONFIG']['id_src']
                    logging.info(f"id_src: {id_src}")
                    raise RuntimeError("id_src support (Wan-Qing's swc data) is not implemented yet")

                spatial_pt_columns = config['DATA_CONFIG']['spatial_pt_columns']

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

                for line_i, line in enumerate(lines):
                    # row = line.split(',')
                    reader = csv.reader(io.StringIO(line))
                    row = next(reader)

                    if id_column is not None:
                        id_ = row[id_column_idx]
                    else:
                        id_ = split_subsplit_id_start + line_i

                    if line_i < 3 or line_i > len(lines) - 4:
                        logging.info(f"Annotation line {line_i} of {len(lines)}, id: {id_:>10}")

                    if id_column is None:
                        line = f"{id_}," + line
                        row = [id_] + row

                    if line_i < 3 or line_i > len(lines) - 4:
                        logging.info(f"Annotation line {line_i} of {len(lines)} with id added, id: {id_:>10}")

                    if isinstance(spatial_pt_columns, dict):
                        pt_positions = {
                            pt_desc: [float(row[pt_x_col_idx]), float(row[pt_y_col_idx]), float(row[pt_z_col_idx])] \
                                for pt_desc, [pt_x_col_idx, pt_y_col_idx, pt_z_col_idx] in spatial_pt_col_idxs.items()
                        }
                    elif spatial_pt_columns == "single_field_list":
                        pt_positions = read_single_field_point_list(row)

                    if 'point_annotation_config' in config['DATA_CONFIG']:
                        position = pt_positions[config['DATA_CONFIG']['point_annotation_config']['pt_column_label']]
                        annotation = PointAnnotation(id_, position, line.strip())
                    elif 'line_annotation_config' in config['DATA_CONFIG']:
                        start = pt_positions[config['DATA_CONFIG']['line_annotation_config']['start_pt_column_label']]
                        end = pt_positions[config['DATA_CONFIG']['line_annotation_config']['end_pt_column_label']]
                        annotation = LineAnnotation(id_, start, end, line.strip())
                    elif 'polyline_annotation_config' in config['DATA_CONFIG']:
                        annotation = PolyLineAnnotation(id_, list(pt_positions.values()), line.strip())

                    annotations.append(annotation)
            else:
                annotations = input_src

        num_annotations = len(annotations)
        if verbose:
            logging.info(f"Num annotations received: {num_annotations:,}")

        if annotation_num_limit is not None:
            if num_annotations > (annotation_num_limit // num_splits) * 1.2:
                logging.warning(f"WARNING! Number of rows in split ({num_annotations}) exceeds 120% of realistic limit ({annotation_num_limit // num_splits:.1f}) on number of annotations in a tree cell of this size for one split! (this might be okay if the other splits have very few annotations in this tree cell)")
            if num_annotations > annotation_num_limit:
                logging.error(f"ERROR! Number of rows in input file ({num_annotations}) exceeds realistic limit ({annotation_num_limit:.1f}) on number of annotations in a tree cell of this size across all splits! This is not physically possible and indicates either a data error or a bug.")

        # end_start_timeblocks("read_input", "get_unique_post_pt_root_ids")
        # if isinstance(annotations, pd.DataFrame):
        #     post_pt_root_ids = list(annotations['post_pt_root_id'].unique())
        # else:
        #     post_pt_root_ids = list(annotations.unique('post_pt_root_id'))
        # if verbose:
        #     logging.info(f"Unique post_pt_root_ids in input file ({len(post_pt_root_ids):,}) (first 5 shown): {post_pt_root_ids[:5]}")
        # end_start_timeblocks("get_unique_post_pt_root_ids", "process_input")

        end_start_timeblocks("read_input", "process_input")

        annos_this_level, annos_children_levels, num_cell_nondups = select_holdout_rows_and_subdivide_subtree_rows(src_loc, num_splits, num_subsplits, annotations, tree_level, cell_bounds_low, cell_bounds_mid, cell_bounds_high, verbose)

        end_start_timeblocks("process_input", "call_add_tree_level_cell_id_to_saved_annotations()")

        # This added field won't be accessed again until the last capsule, 'conglomerate spatial index by shard'. When the annotations were merely CSV rows, it was easy to append the tree level cell id here and pass it on to the end of the pipeline, but now that the annotations are Annotation objects, doing so is considerably slower. It is also unnecessary. The necessary information is available in the final capsule so we can defer that work until then, where it should be faster since the data will be CSV at that point (as of this writing).
        # Actually, the profiling results of run 3191269 suggest I may have completely misjudged this issue. Perhaps it wasn't slow to begin with.
        # add_tree_level_cell_id_to_saved_annotations(annos_this_level, tree_level_cell_id)

        end_start_timeblocks("call_add_tree_level_cell_id_to_saved_annotations()", "confirm_subtree_bounds_DEBUG")

        # Currently disabled to expedite performance. Reenable if future development impacts this code in concerning ways.
        # for x in [0, 1]:
        #     for y in [0, 1]:
        #         for z in [0, 1]:
        #             subtree_cell_bounds = determine_subtree_bounds(
        #                 cell_bounds_low,
        #                 cell_bounds_mid,
        #                 cell_bounds_high,
        #                 [x, y, z]
        #             )
        #             confirm_all_annotations_within_bounds(src_loc, annos_children_levels[x][y][z], subtree_cell_bounds[0], subtree_cell_bounds[1], "RRR")

        end_start_timeblocks("confirm_subtree_bounds_DEBUG", "tally_subtree_DEBUG_2")

        if verbose:
            logging.info(f"Num rows saved in this oct tree cell:                                 {len(annos_this_level):>11,}")
            logging.info("Oct tree child cell row counts:")
        subtree_row_tally = 0
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    if verbose:
                        logging.info(f"  [{x}, {y}, {z}]    {len(annos_children_levels[x][y][z]):>11,}")
                    subtree_row_tally += len(annos_children_levels[x][y][z])

        if verbose:
            logging.info(f"Total num rows saved in this treecell: {len(annos_this_level):>11,}")
            logging.info(f"Total num rows sent to oct subtrees:   {subtree_row_tally:>11,}")

        end_start_timeblocks("tally_subtree_DEBUG_2", "save_results")

        # Save the held rows for this tree cell.
        # It's important to remove the header from the final output file so that, when we union the file with other splits,
        # the union doesn't get littered with numerous copies of the header row. Alternatively, the unioning step could judiciously remove the header from each incoming piece before it combines them.
        shard_hex = save_this_treecell_data(src_loc, annos_this_level, subsplit_id, num_splits, split_id, tree_level, tree_level_cell_id, tree_level_shard_histograms, morton_code, cell_bounds_low, cell_bounds_high)

        save_subtrees(src_loc, annos_children_levels, subsplit_id, num_splits, split_id, tree_level, tree_level_cell_id, cell_bounds_low, cell_bounds_mid, cell_bounds_high)

        # end_end_timeblocks("save_results", "process_one_treecell_input_file_or_dir()")
        end_timeblock("save_results")
        # end_timeblock("process_one_treecell_input_file_or_dir()")

        return num_annotations, len(annos_this_level), subtree_row_tally, num_cell_nondups, shard_hex

    elif config['DATA_CONFIG']['structure'] == 'one_annotation_per_file__one_point_per_row':
        assert os.path.isdir(input_src)
        if input_src[-1] != '/':
            input_src += '/'

        input_files = ram_data_pond.fastglob_ram_data_pond(input_src)
        logging.info(f"input_files {input_files}")

        annotations = []
    else:
        raise ValueError(f"Unknown structure: {config['DATA_CONFIG']['structure']}")

    return True

def process_input_files_for_one_tree_level_0(subsplit_id, subsplit_range_row_start, subsplit_range_row_end, src_loc, tree_level, tree_level_shard_histograms, input_file):
    # There is a top-level split file, which means we are processing the top-level of the tree
    logging.info(f"input_file (or subtree key): {input_file}")
    logging.info(f"input_file type: {type(input_file)}")
    file_size_bytes = os.path.getsize(input_file)
    logging.info(f"Top-level split input file ({file_size_bytes/1000000:,}M): {input_file}")

    file_tree_level, tree_level_cell_id, tree_level_cell_id_str = 0, [0, 0, 0], "000,000,000"

    annotations_filename = os.path.basename(input_file)
    pcs = annotations_filename[:annotations_filename.rindex('.')].split("_")
    for pc in pcs:
        if "split-" in pc:
            splitnm = pc.split("-")[1]
            split_id, num_splits = (int(v) for v in splitnm.split('@'))
            enable_disable_profiling((split_id % 2 == 0) if config['PROFILING_ENABLED'] else False)
        elif "treelevel-" in pc:
            file_tree_level = int(pc.split("-")[1])
        elif "treelevelcellid-" in pc:
            tree_level_cell_id_str = pc.split("-")[1]
            tree_level_cell_id = [int(v) for v in tree_level_cell_id_str.split(",")]

    assert file_tree_level == tree_level

    if config['DATA_CONFIG']['structure'] == 'one_annotation_per_file__one_point_per_row':
        # Get the current directory listing so we can identify the new files after extraction
        data_loc_contents_before_extraction = sorted(list(glob.glob(f"{data_loc}*")))  # Use glob to preserve the full path for later use
        data_loc_contents_before_extraction = [v for v in data_loc_contents_before_extraction if "placeholder" not in v]

        logging.info(f"Extracting zip file: {input_file}")
        with zipfile.ZipFile(input_file, 'r') as zip_ref:
            zip_ref.extractall(f"{data_loc}")
        logging.info(f"{data_loc} contents after zip extraction (first 30 shown):")

        # Remove some common book-keeping directories that sometimes float along from one OS to another
        if os.path.exists(f"{data_loc}__MACOSX"):
            shutil.rmtree(f"{data_loc}__MACOSX")
        if os.path.exists(f"{data_loc}.DS_Store"):
            shutil.rmtree(f"{data_loc}.DS_Store")

        data_loc_contents_after_extraction = sorted(list(glob.glob(f"{data_loc}*")))
        data_loc_contents_after_extraction = [v for v in data_loc_contents_after_extraction if "placeholder" not in v]
        logging.info('  ' + '\n  '.join(data_loc_contents_after_extraction[:30]).strip() + '\n')
        new_files_or_dirs = list(set(data_loc_contents_after_extraction) - set(data_loc_contents_before_extraction))
        logging.info(f"Extracted files or dirs: {new_files_or_dirs}")
        assert len(new_files_or_dirs) == 1
        assert os.path.isdir(new_files_or_dirs[0])
        input_dir = new_files_or_dirs[0]
        if input_dir[-1] != '/':
            input_dir += '/'
        logging.info(f"Determined input dir: {input_dir}")

    if config['HIGHEST_SPLIT_ID'] is not None:
        if split_id > config['HIGHEST_SPLIT_ID']:
            end_timeblock("process_input_files_for_one_tree_level()")
            return None, None
        logging.info(f"A Debugging split {split_id} <= {config['HIGHEST_SPLIT_ID']}")

    cell_bounds_low, cell_bounds_high = config['DATA_CONFIG']['volume_bounds'][0], config['DATA_CONFIG']['volume_bounds'][1]
    # logging.info(f"Cell width read from config: {cell_bounds_high[0] - cell_bounds_low[0]} {cell_bounds_high[1] - cell_bounds_low[1]} {cell_bounds_high[2] - cell_bounds_low[2]}")

    if config['DATA_CONFIG']['structure'] == 'one_annotation_per_row__multiple_points_per_row':
        confirm_all_annotations_within_bounds(src_loc, input_file, cell_bounds_low, cell_bounds_high, "NNN")

    data_src = None
    if config['DATA_CONFIG']['structure'] == 'one_annotation_per_row__multiple_points_per_row':
        data_src = input_file
    elif config['DATA_CONFIG']['structure'] == 'one_annotation_per_file__one_point_per_row':
        data_src = input_dir
    elif config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row_in_one_field":
        data_src = input_file
    else:
        raise ValueError(f"Unknown structure: {config['DATA_CONFIG']['structure']}")

    result = process_one_treecell_input_file_or_dir(subsplit_id, subsplit_range_row_start, subsplit_range_row_end, 0, src_loc, data_src, cell_bounds_low, cell_bounds_high, num_splits, split_id, file_tree_level, tree_level_cell_id, tree_level_shard_histograms)

    if result is None:
        # Subsplits are done
        return None

    num_rows, num_saved, num_sent, num_cell_nondups, shard_hex = result

    return split_id, num_splits, num_rows, num_saved, num_sent, num_cell_nondups, [(tree_level_cell_id_str, shard_hex)]

def process_input_files_for_one_tree_level_1up(subsplit_id, src_loc, tree_level, tree_level_shard_histograms):
    # Since no split file was found, there must be subtree directories
    logging.info("No top-level split input file found. Looking for subtrees.")

    # subtree_dir_contents = list(glob.glob(f"{src_loc}subtrees/subsplit-*__split-*_*subtree*/*"))
    subtree_dir_contents = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}subtrees/subsplit-*__split-*_*subtree*/*", src_loc==data_loc)
    logging.info(f"Subtree directory contents (first 9 shown, expecting 3 files per subtree subdir):\n  {'\n  '.join(subtree_dir_contents[:9])}\n")

    # Look for a subtree file
    # subtree_csv_files = list(glob.glob(f"{src_loc}subtrees/subsplit-*__split-*_*subtree*/*subtree_*csv"))
    subtree_csv_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}subtrees/subsplit-*__split-*_*subtree*/annotations_one_subtree_*csv", src_loc==data_loc)
    logging.info(f"Subtree files ({len(subtree_csv_files)}) (first 3 shown):\n  {'\n  '.join(subtree_csv_files[:3])}\n")
    assert not subtree_csv_files, "The results should be lists of Annotations. CSV files are deprecated."

    subtree_keys = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}subtrees/subsplit-*__split-*_*subtree*/annotations_one_subtree_*", src_loc==data_loc)
    logging.info(f"Subtree objects ({len(subtree_keys)}) (first 3 shown):\n  {'\n  '.join(subtree_keys[:3])}\n")

    total_rows, total_saved, total_sent = 0, 0, 0
    total_num_cell_nondups = 0
    one_tree_level__all_tree_level_cell_ids = []

    if subtree_keys:
        logging.info(f"Num subtree objects found: {len(subtree_keys)}")
        for subtree_key_i, subtree_key in enumerate(subtree_keys):
            # logging.info("")
            # logging.info("." * 100)

            if subtree_key_i == 0:
                logging.info(f"Processing subtree key: {subtree_key}")

            subtree_object = ram_data_pond.read_from_disk_or_ram_data_pond(subtree_key)
            assert isinstance(subtree_object, list)  # A list of Annotation subclass objects

            # file_size_bytes = os.path.getsize(subtree_key)
            # file_size_bytes = ram_data_pond.getsize(subtree_key)
            annotations_subdir = '/'.join(subtree_key.split('/')[:-1])
            subtree_filename = os.path.basename(subtree_key)
            if subtree_key_i == 0:
                logging.info(f"Processing subtree subtree_filename: {subtree_filename}")
            # logging.info(f"Processing subtree input file of size {file_size_bytes/1000000:,}M: {subtree_filename}")

            # These lower levels of the tree receive their bounds from their parent instead of deriving them from the global bounds in the config
            cell_bounds_file = f"{annotations_subdir}/oct_tree_cell_bounds.txt"
            # with open(cell_bounds_file) as f:
            #     cell_bounds = ast.literal_eval(f.read())
            cell_bounds = ast.literal_eval(ram_data_pond.read_from_disk_or_ram_data_pond(cell_bounds_file))
            # logging.info(f"Cell bounds assigned from parent cell: {cell_bounds}")
            cell_bounds_low = cell_bounds[0]
            cell_bounds_high = cell_bounds[1]
            xw = cell_bounds_high[0] - cell_bounds_low[0]
            yw = cell_bounds_high[1] - cell_bounds_low[1]
            zw = cell_bounds_high[2] - cell_bounds_low[2]
            # logging.info(f"Cell bounds assigned from parent cell, and widths: {cell_bounds_low}    {cell_bounds_high}    {xw} {yw} {zw}")

            confirm_all_annotations_within_bounds(src_loc, subtree_object, cell_bounds_low, cell_bounds_high, "VVV")

            # pcs = subtree_filename[:subtree_filename.rindex('.')].split("_")
            pcs = subtree_filename.split("_")
            for pc in pcs:
                if "split-" in pc:
                    splitnm = pc.split("-")[1]
                    split_id, num_splits = (int(v) for v in splitnm.split('@'))
                    enable_disable_profiling((split_id % 2 == 0) if config['PROFILING_ENABLED'] else False)
                elif "treelevel-" in pc:
                    file_tree_level = int(pc.split("-")[1])
                elif "treelevelcellid-" in pc:
                    tree_level_cell_id_str = pc.split("-")[1]
                    tree_level_cell_id = [int(v) for v in tree_level_cell_id_str.split(",")]

            assert file_tree_level == tree_level

            if config['HIGHEST_SPLIT_ID'] is not None:
                if split_id > config['HIGHEST_SPLIT_ID']:
                    end_timeblock("process_input_files_for_one_tree_level()")
                    return None, None
                if len(subtree_csv_files) // 10 > 0 and subtree_key_i % (len(subtree_csv_files) // 10) == 0:
                    logging.info(f"B Debugging split {split_id} <= {config['HIGHEST_SPLIT_ID']}")

            # logging.info(f"Calling process_one_treecell_input_file_or_dir() with cell bounds: {cell_bounds_low} {cell_bounds_high}")
            result = process_one_treecell_input_file_or_dir(subsplit_id, None, None, subtree_key_i, src_loc, subtree_object, cell_bounds_low, cell_bounds_high, num_splits, split_id, file_tree_level, tree_level_cell_id, tree_level_shard_histograms)

            if result is None:
                # Subsplits are done
                return None

            num_rows, num_saved, num_sent, num_cell_nondups, shard_hex = result

            tr, tsv, tsn = total_rows, total_saved, total_sent
            total_rows += num_rows
            total_saved += num_saved
            total_sent += num_sent
            total_num_cell_nondups += num_cell_nondups
            one_tree_level__all_tree_level_cell_ids.append((tree_level_cell_id_str, shard_hex))
            # logging.info(f"Rows num/saved/sent for one subtree and accumulated:    {tr} + {num_rows} => {total_rows}        {tsv} + {num_saved} => {total_saved}    {tsn} + {num_sent} => {total_sent}")

            # if subtree_key_i % 1000 == 0:
            #     logging.error(f"Elapsed times at mid-processing tree level {file_tree_level}, after subtree {subtree_key_i}:")
            #     dump_profile(False)

        # logging.info("\n" + ". " * 50 + "\n")

        return split_id, num_splits, total_rows, total_saved, total_sent, total_num_cell_nondups, one_tree_level__all_tree_level_cell_ids
    else:
        logging.info("No subtree files found")

    return None, None, None, None, None, None, None

def process_input_files_for_one_tree_level(subsplit_id, subsplit_range_row_start, subsplit_range_row_end, src_loc, tree_level, tree_level_shard_histograms):
    '''
    In a daisy-chain configuration, this capsule must have received either the top-level tree (the split file)
    or a subtree that requires further processing.
    Alternatively, if the capsule is configured to build the entire tree at all levels,
    then this function will be called repeatedly, once per tree level.
    '''
    start_timeblock("process_input_files_for_one_tree_level()")

    logging.info(f"process_input_files_for_one_tree_level() Tree level: {tree_level}")

    if tree_level == 0:
        if config['DATA_CONFIG']['structure'] == 'one_annotation_per_row__multiple_points_per_row':
            # split_files = glob.glob(f"{src_loc}*split-*_rows-*.csv")
            split_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}*split-*_rows-*.csv", src_loc==data_loc)
            if len(split_files) == 0:
                split_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}*split-*_rows-*.parquet", src_loc==data_loc)
        elif config['DATA_CONFIG']['structure'] == 'one_annotation_per_file__one_point_per_row':
            split_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}*split-*.zip", src_loc==data_loc)
        elif config['DATA_CONFIG']['structure'] == "one_annotation_per_row__multiple_points_per_row_in_one_field":
            split_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}*split-*.csv", src_loc==data_loc)
        else:
            raise ValueError(f"Unknown structure: {config['DATA_CONFIG']['structure']}")

        if len(split_files) > 1:
            raise ValueError("Expected <=1 top-level split input files")
        split_file = split_files[0]

        result = process_input_files_for_one_tree_level_0(subsplit_id, subsplit_range_row_start, subsplit_range_row_end, src_loc, tree_level, tree_level_shard_histograms, split_file)
        if result is None:
            # Subsplits are done
            end_timeblock("process_input_files_for_one_tree_level()")
            return None
        split_id, num_splits, total_rows, total_saved, total_sent, total_num_cell_nondups, one_tree_level__all_tree_level_cell_ids = result
    else:
        result = process_input_files_for_one_tree_level_1up(subsplit_id, src_loc, tree_level, tree_level_shard_histograms)
        if result is None:
            # Subsplits are done
            end_timeblock("process_input_files_for_one_tree_level()")
            return None
        split_id, num_splits, total_rows, total_saved, total_sent, total_num_cell_nondups, one_tree_level__all_tree_level_cell_ids = result

    logging.info(f"Total rows, total saved, & total sent deeper across this split or set of subtrees:    {total_rows:,}    {total_saved:,}    {total_sent:,}")
    if total_saved + total_sent - total_num_cell_nondups != total_rows:
        logging.error(f"ERROR! (2) Num saved + sent rows - num_cell_dups != total rows: {total_saved:,} + {total_sent:,} - {total_num_cell_nondups:,} = {total_saved + total_sent + total_num_cell_nondups:,} != {total_rows:,}")

    # I think this is no longer used, while at the same time it clutters the output with numerous files that impede CO Duration time to the next capsule
    # shards_summary_filepath = f"{results_loc}subsplit-{subsplit_id:02}__split-{split_id:03}@{num_splits}__tree_level-{tree_level}__tree_cell_shards.txt"
    # with open(shards_summary_filepath, 'w') as f:
    #     for tree_level_cell_id, shard_hex in one_tree_level__all_tree_level_cell_ids:
    #         f.write(f"{tree_level_cell_id}\t{shard_hex}\n")
    # logging.info(f"All tree cells used ({len(one_tree_level__all_tree_level_cell_ids)}): {one_tree_level__all_tree_level_cell_ids[:10]}...See output file for full list:\n  {shards_summary_filepath}")

    end_timeblock("process_input_files_for_one_tree_level()")

    return split_id, num_splits

def move_upstream_completed_tree_outputs_to_results(src_loc):
    start_timeblock("move_upstream_completed_tree_outputs_to_results()")

    if src_loc == data_loc:
        # Tree level 0 can't have any higher tree level completed tree cells.
        # The for loop below could simply fall through, but the fastglob call will fail.
        # We could selectively run the slow glob function in this case,
        # but we can also just return before reaching that point.
        treecell_dirs = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}completed_treecells*", src_loc==data_loc)
        logging.info(f"Tree level 0: There should be no upstream completed tree cell outputs to move to the results: {len(treecell_dirs)}")
        assert not treecell_dirs
        end_timeblock("move_upstream_completed_tree_outputs_to_results()")
        return

    # treecell_dirs = list(glob.glob(f"{src_loc}completed_treecells*"))
    # treecell_dirs = ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}completed_treecells*", src_loc==data_loc)
    start_timeblock("call fastglob_ram_data_pond()")
    treecell_dirs = ram_data_pond.fastglob_ram_data_pond(f"{src_loc}completed_treecells", src_loc==data_loc)

    end_start_timeblocks("call fastglob_ram_data_pond()", "treecell_dir loop")

    logging.info(f"Moving {len(treecell_dirs)} completed tree cell dirs from {src_loc} to {results_loc}")
    for i, treecell_dir in enumerate(treecell_dirs):
        treecell_dirname = os.path.basename(treecell_dir)
        if True:  # len(treecell_dirs) // 10 > 0 and i % (len(treecell_dirs) // 10) == 0:
            logging.info(f"Moving idx-{i} of {len(treecell_dirs)} {treecell_dir}\n  to {results_loc}{treecell_dirname}")
        # shutil.move(treecell_dir, f"{results_loc}{treecell_dirname}")
        # ram_data_pond.move_file_on_disk_or_ram_data_pond(treecell_dir, f"{results_loc}{treecell_dirname}")
        ram_data_pond.move_files_via_prefix_replacement(treecell_dir, f"{results_loc}{treecell_dirname}")

    end_end_timeblocks("treecell_dir loop", "move_upstream_completed_tree_outputs_to_results()")

def process_one_tree_level(subsplit_id, subsplit_range_row_start, subsplit_range_row_end, treelevel_iter, src_loc, tree_level_shard_histograms):
    start_timeblock("process_one_tree_level()")
    logging.info(f"Iterated tree level: {treelevel_iter}    Source location: {src_loc}\n")

    if treelevel_iter > config['MAX_NUM_TREE_LEVELS']:
        raise RuntimeError("Tree level iteration exceeded max num tree levels")

    if treelevel_iter == 0:
        logging.info(f"{src_loc} contents ({len(ram_data_pond.glob_disk_or_ram_data_pond(f'{src_loc}*', src_loc==data_loc))}) (first 10 shown):\n  {'\n  '.join(ram_data_pond.glob_disk_or_ram_data_pond(f'{src_loc}*', src_loc==data_loc)[:10]).strip()}\n")
    else:
        logging.info(f"{src_loc} contents ({len(ram_data_pond.fastglob_ram_data_pond(f'{src_loc}', src_loc==data_loc))}) (first 10 shown):\n  {'\n  '.join(ram_data_pond.fastglob_ram_data_pond(f'{src_loc}', src_loc==data_loc)[:10]).strip()}\n")
    logging.info(f"{src_loc} subcontents ({len(ram_data_pond.glob_disk_or_ram_data_pond(f'{src_loc}*/*', src_loc==data_loc))}) (first 3 shown):\n  {'\n  '.join(ram_data_pond.glob_disk_or_ram_data_pond(f'{src_loc}*/*', src_loc==data_loc)[:3]).strip()}\n")
    logging.info(f"{src_loc} subtrees contents ({len(ram_data_pond.glob_disk_or_ram_data_pond(f'{src_loc}subtrees/*', src_loc==data_loc))}) (first 3 shown):\n  {'\n  '.join(ram_data_pond.glob_disk_or_ram_data_pond(f'{src_loc}subtrees/*', src_loc==data_loc)[:3]).strip()}\n")
    logging.info(f"{src_loc} completed_treecells contents ({len(ram_data_pond.glob_disk_or_ram_data_pond(f'{src_loc}completed_treecells/*', src_loc==data_loc))}) (first 3 shown):\n  {'\n  '.join(ram_data_pond.glob_disk_or_ram_data_pond(f'{src_loc}completed_treecells/*', src_loc==data_loc)[:3]).strip()}\n")
    logging.info(f"{src_loc} completed_treecells subcontents ({len(ram_data_pond.glob_disk_or_ram_data_pond(f'{src_loc}completed_treecells/*/*', src_loc==data_loc))}) (first 3 shown):\n  {'\n  '.join(ram_data_pond.glob_disk_or_ram_data_pond(f'{src_loc}completed_treecells/*/*', src_loc==data_loc)[:3]).strip()}\n")

    # logging.info(f"Input completed tree cells (packed):\n  {'\n  '.join(sorted(list(glob.glob(f'{src_loc}completed_treecells*')))).strip()}\n")

    # logging.info(f"Input completed tree cells (not packed):\n  {'\n  '.join(sorted(list(glob.glob(f'{src_loc}completed_treecells/annotations_one_treecell*')))).strip()}\n")

    tree_level = determine_capsule_tree_level(src_loc)
    logging.info(f"Tree level this iteration: {tree_level}")

    result = process_input_files_for_one_tree_level(subsplit_id, subsplit_range_row_start, subsplit_range_row_end, src_loc, tree_level, tree_level_shard_histograms)

    logging.info(f"Process one tree level received result: {result}")

    if result is None:
        # Subsplits are done
        end_timeblock("process_one_tree_level()")
        return None
    split_id, num_splits = result

    # For debugging, some workers return None to indicate they are being skipped
    if split_id is None:
        end_timeblock("process_one_tree_level()")
        return None

    move_upstream_completed_tree_outputs_to_results(src_loc)

    logging.info(f"\nProcessing of tree level {tree_level} is done")

    # results_loc_contents = sorted(os.listdir(results_loc))
    # logging.info(f"\nresults_loc contents ({len(results_loc_contents)}):\n  {'\n  '.join(results_loc_contents)}")
    # # results_loc_subcontents = sorted(list(glob.glob(f"{results_loc}*/*")))
    # results_loc_subcontents = ram_data_pond.glob_disk_or_ram_data_pond(f"{results_loc}*/*")
    # logging.info(f"\nresults_loc subcontents ({len(results_loc_subcontents)}) (first 5 shown):\n  {'\n  '.join(results_loc_subcontents[:5]).strip()}\n")

    # If there are no more subtrees, the tree level iteration is complete (we have reached the bottom of the tree)
    # subtree_input_dirs = list(glob.glob(f"{results_loc}split*subtree*")) + list(glob.glob(f"{results_loc}subtrees.tar*"))
    # subtree_input_dirs = ram_data_pond.glob_disk_or_ram_data_pond(f"{results_loc}split*subtree*") + ram_data_pond.glob_disk_or_ram_data_pond(f"{results_loc}subtrees.tar*")
    # logging.info("EEE")
    subtree_input_dirs = ram_data_pond.glob_disk_or_ram_data_pond(f"{results_loc}subtrees/subsplit-*__split*subtree*") + ram_data_pond.fastglob_ram_data_pond(f"{results_loc}subtrees.tar")
    logging.info(f"\nsubtree_input_dirs ({len(subtree_input_dirs)}) (first 5 shown):\n  {'\n  '.join(subtree_input_dirs[:5])}")
    if not subtree_input_dirs:
        logging.info("There are no subtrees. This tree level is the bottom of the tree.")
        end_timeblock("process_one_tree_level()")
        return split_id, num_splits, src_loc, False

    # Move the results to the source location so they can be picked up by the next tree level iteration (for the next tree level down)
    start_timeblock("move_tree_level_results_to_source_for_next_tree_level_iteration")
    next_src_dir = f"{data_loc}results_treelevel-{treelevel_iter}/"
    logging.info(f"\nMoving results to source location for next tree level iteration: {next_src_dir}")
    # os.makedirs(next_src_dir)

    if ram_data_pond.ram_data_pond is None:
        # for item in os.listdir(results_loc):
            # shutil.move(item, f"{next_src_dir}{item}")

        for item in glob.glob(f"{results_loc}*"):
            item_tail = item[len(results_loc):]
            logging.info(f"Moving\n  {item}\n  to\n  {next_src_dir}{item_tail}")
            shutil.move(item, f"{next_src_dir}{item_tail}")
    else:
        # for item in ram_data_pond.glob_disk_or_ram_data_pond(f"{results_loc}*"):
        # logging.info("FFF")
        # for item in ram_data_pond.fastglob_ram_data_pond(results_loc):
        #     dst = '/'.join(item.split('/')[2:])
        #     # logging.info(f"Moving\n  {item}\n  to\n  {next_src_dir}{dst}")
        #     ram_data_pond.move_file_on_disk_or_ram_data_pond(item, next_src_dir + dst)

        # See note inside move_files_via_prefix_replacement().
        ram_data_pond.move_files_via_prefix_replacement(results_loc, next_src_dir)

    src_loc = next_src_dir

    end_end_timeblocks("move_tree_level_results_to_source_for_next_tree_level_iteration", "process_one_tree_level()")

    return split_id, num_splits, src_loc, True

def archive_results(ARCHIVE_MEMORY_STORE, subsplit_id, split_id, num_splits):
    start_timeblock("write RAM data pond to disk")
    logging.info("Writing final results from RAM data pond to disk")

    # ram_files = ram_data_pond.glob_disk_or_ram_data_pond("*")
    # logging.info(f"ram_data_pond *:\n  {'\n  '.join(ram_files)}")
    # ram_files = ram_data_pond.glob_disk_or_ram_data_pond("*/*")
    # logging.info(f"ram_data_pond */*:\n  {'\n  '.join(ram_files)}")
    # ram_files = ram_data_pond.glob_disk_or_ram_data_pond("*/*/*")
    # logging.info(f"ram_data_pond */*/*:\n  {'\n  '.join(ram_files)}")
    # ram_files = ram_data_pond.glob_disk_or_ram_data_pond("*/*/*/*")
    # logging.info(f"ram_data_pond */*/*/*:\n  {'\n  '.join(ram_files)}")
    # logging.info("")

    # start_timeblock("glob ram data pond at level 3")
    # annotations_one_treecell_ram_dirs = ram_data_pond.glob_disk_or_ram_data_pond(f"{results_loc}completed_treecells/annotations_one_treecell*")
    # end_timeblock("glob ram data pond at level 3")
    # logging.info("annotations_one_treecell results:")
    # logging.info(f"annotations_one_treecell_ram_dirs ({len(annotations_one_treecell_ram_dirs)}) (first 5 shown):\n  {'\n  '.join(annotations_one_treecell_ram_dirs[:5])}")

    # logging.info("GGG")
    start_timeblock("glob ram data pond final results")
    # annotations_one_treecell_ram_objects = ram_data_pond.fastglob_ram_data_pond(f"{results_loc}completed_treecells/annotations_one_treecell*/annotations_one_treecell__subsplit-{subsplit_id:02}__*.csv")
    annotations_one_treecell_ram_objects = ram_data_pond.fastglob_ram_data_pond(f"{results_loc}completed_treecells/annotations_one_treecell*/annotations_one_treecell__subsplit-{subsplit_id:02}__*")
    end_timeblock("glob ram data pond final results")
    logging.info(f"annotations_one_treecell_ram_objects ({len(annotations_one_treecell_ram_objects)}) (first 5 shown):\n  {'\n  '.join(annotations_one_treecell_ram_objects[:5])}")
    num_ram_data_pond_files_a = len(annotations_one_treecell_ram_objects)

    # Convert lists of Annotations to an archival format
    start_timeblock("convert annotations to archival format")
    for annotations_one_treecell_ram_object_i, annotations_one_treecell_ram_object in enumerate(annotations_one_treecell_ram_objects):
        start_start_timeblocks("annotation file loop body", "annotation file loop body setup")
        if annotations_one_treecell_ram_object_i < 3:
            logging.info(f"\nConverting annotations_one_treecell_ram_object ({annotations_one_treecell_ram_object_i} of {len(annotations_one_treecell_ram_objects)}): {annotations_one_treecell_ram_object}")
        if not annotations_one_treecell_ram_object.endswith(".csv"):
            # By not have a .csv file extention, this object is probably a list of Annotations. We can confirm that by retrieving it and checking its type.
            annotations_one_treecell_ram_list = ram_data_pond.read_from_disk_or_ram_data_pond(annotations_one_treecell_ram_object)
            assert isinstance(annotations_one_treecell_ram_list, list)  # A list of Annotation subclass objects
            if annotations_one_treecell_ram_object_i < 3:
                logging.info(f"Retrieved a list of {len(annotations_one_treecell_ram_list)} annotations from the RAMDataPond")
            end_timeblock("annotation file loop body setup")

            csv_rows = []
            start_timeblock("gather annotations raw data")
            for annotation_i, annotation in enumerate(annotations_one_treecell_ram_list):
                # if annotation_i == 0:
                #     logging.info(f"First annotation retrieved is a {type(annotation)} from the RAMDataPond")
                csv_row = annotation.raw_data  # The raw_data was originally populated with the original CSV line from the input file
                csv_rows.append(csv_row.strip())
            end_timeblock("gather annotations raw data")
            csv_str = '\n'.join(csv_rows)
            start_timeblock("call write_to_disk_or_ram_data_pond()")
            if annotations_one_treecell_ram_object_i < 3:
                logging.info(f"Writing annotation list of len ({len(annotations_one_treecell_ram_list)}) to csv: {annotations_one_treecell_ram_object}")
            ram_data_pond.write_to_disk_or_ram_data_pond(annotations_one_treecell_ram_object + ".csv", csv_str)
            end_timeblock("call write_to_disk_or_ram_data_pond()")

        else:
            end_timeblock("annotation file loop body setup")
        end_timeblock("annotation file loop body")

    # Regrab the list of csv files from above now that they have been converted.
    # This isn't really necessary if none of the annotations were *not* csvs to begin with (i.e., were lists).
    start_timeblock("glob ram data pond final csvs")
    annotations_one_treecell_ram_files = ram_data_pond.fastglob_ram_data_pond(f"{results_loc}completed_treecells/annotations_one_treecell*/annotations_one_treecell__subsplit-{subsplit_id:02}__*.csv")
    end_timeblock("glob ram data pond final csvs")
    logging.info(f"annotations_one_treecell_ram_files ({len(annotations_one_treecell_ram_files)}) (first 5 shown):\n  {'\n  '.join(annotations_one_treecell_ram_files[:5])}")
    end_timeblock("convert annotations to archival format")

    total_data_size = 0
    if not ARCHIVE_MEMORY_STORE:
        assert False, "Development & implementation of this option has fallen behind"
        # Write the RAM data pond to disk, one file at a time.
        # This can result in thousands of files, which can impede disk I/O time
        # even if the total storage in bytes is not overly excessive.
        for annotations_one_treecell_ram_dir in annotations_one_treecell_ram_dirs:
            # logging.info(f"  {annotations_one_treecell_ram_dir}")
            # annotations_one_treecell_ram_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{annotations_one_treecell_ram_dir}/annotations_one_treecell__subsplit-{subsplit_id:02}__*")
            # logging.info("HHH")
            annotations_one_treecell_ram_files = ram_data_pond.fastglob_ram_data_pond(f"{annotations_one_treecell_ram_dir}/annotations_one_treecell__subsplit-{subsplit_id:02}__")
            for annotations_one_treecell_ram_file in annotations_one_treecell_ram_files:
                file_size_bytes = ram_data_pond.getsize(annotations_one_treecell_ram_file)
                total_data_size += file_size_bytes
                # logging.info(f"    {file_size_bytes/1000000:.1f} MB {annotations_one_treecell_ram_file}")
                file_contents = ram_data_pond.read_from_disk_or_ram_data_pond(annotations_one_treecell_ram_file)
                start_timeblock("call write_to_disk_or_ram_data_pond()")
                ram_data_pond.write_to_disk_or_ram_data_pond(annotations_one_treecell_ram_file, file_contents, force_disk=True)
                end_timeblock("call write_to_disk_or_ram_data_pond()")
    elif not config['ARCHIVE_MEMORY_STORE_VIA_CUSTOM_METHOD']:
        assert False, "Development & implementation of this option has fallen behind"
        # Tar the RAM data pond in memory into a single object.
        # Then write the in-memory tar to disk as a single file.
        start_timeblock("tar RAM data pond in memory")
        logging.info(f"Tarring {len(annotations_one_treecell_ram_dirs)} RAM data pond directories to in-memory tar buffer")
        # num_ram_data_pond_files_a, num_ram_data_pond_files_b = 0, 0
        num_ram_data_pond_files_b = 0
        total_data_size_bytes = 0
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
            for annotations_one_treecell_ram_dir in annotations_one_treecell_ram_dirs:
                # logging.info(f"  {annotations_one_treecell_ram_dir}")
                start_timeblock("glob ram data pond at level 4")
                # annotations_one_treecell_ram_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{annotations_one_treecell_ram_dir}/annotations_one_treecell__subsplit-{subsplit_id:02}__*")
                # logging.info("III")
                annotations_one_treecell_ram_files = ram_data_pond.fastglob_ram_data_pond(f"{annotations_one_treecell_ram_dir}/annotations_one_treecell__subsplit-{subsplit_id:02}__")
                end_timeblock("glob ram data pond at level 4")
                num_ram_data_pond_files_a += len(annotations_one_treecell_ram_files)
                for annotations_one_treecell_ram_file in annotations_one_treecell_ram_files:
                    start_timeblock("prep file to add to in-memory tar")
                    num_ram_data_pond_files_b += 1
                    file_size_bytes = ram_data_pond.getsize(annotations_one_treecell_ram_file)
                    total_data_size += file_size_bytes
                    annotations_one_treecell_ram_file_tail = annotations_one_treecell_ram_file[len(results_loc):]
                    # logging.info(f"    {file_size_bytes/1000000:.1f} MB {annotations_one_treecell_ram_file}\n  {annotations_one_treecell_ram_file_tail}")
                    file_contents = ram_data_pond.read_from_disk_or_ram_data_pond(annotations_one_treecell_ram_file)
                    file_contents_bytes = file_contents.encode("utf-8")
                    info = tarfile.TarInfo(name=annotations_one_treecell_ram_file_tail)
                    total_data_size_bytes += len(file_contents_bytes)
                    info.size = len(file_contents_bytes)
                    end_start_timeblocks("prep file to add to in-memory tar", "add file to in-memory tar")
                    tar.addfile(info, io.BytesIO(file_contents_bytes))
                    end_timeblock("add file to in-memory tar")
        logging.info(f"Wrote {total_data_size_bytes/1000000:.1f} MB from {num_ram_data_pond_files_a}|{num_ram_data_pond_files_b} RAM data pond files to in-memory tar, resulting in memory-tar of length {tar_buffer.getbuffer().nbytes/1000000:,}M")
        end_start_timeblocks("tar RAM data pond in memory", "write in-memory tar to disk")
        tar_filepath = f"{results_loc}split-{split_id:03}@{num_splits}__subsplit-{subsplit_id:02}.tar"
        with open(tar_filepath, "wb") as f:
            f.write(tar_buffer.getbuffer())
        logging.info(f"Tar written to disk ({os.path.getsize(tar_filepath)/1000000:,}M): {tar_filepath}")
        end_timeblock("write in-memory tar to disk")

        # os.makedirs(f"{results_loc}test")
        # with tarfile.open(tar_filepath, "r") as tar:
        #     tar.extractall(path=f"{results_loc}test")
        # list_directory(f"{results_loc}test/*")
        # list_directory(f"{results_loc}test/*/*")
    elif config['ARCHIVE_COMPLETED_TREECELLS_WITH_SHARD_GROUPING']:
        start_timeblock("archive RAM data pond in memory with shard grouping")
        # Archive all the completed tree cell directories with grouping them by shard.
        # This produces a larger number of files since it separates the shards,
        # which enables great horizontal distribution in the next stage,
        # but increasing the number of archive files can slow down Nextflow's file handling between capsules.

        # treecell_dirs = ram_data_pond.fastglob_ram_data_pond(f"{results_loc}completed_treecells/annotations_one_treecell") + ram_data_pond.fastglob_ram_data_pond(f"{src_loc}completed_treecells/annotations_one_treecell", src_loc==data_loc)
        # logging.info(f"treecell_dirs ({len(treecell_dirs)}) (first 5 shown):\n  {'\n  '.join(sorted(treecell_dirs[:5])).strip()}\n")

        treelevel_shards = defaultdict(set)  # Debug only
        completed_treecells_grouped_by_treelevel_and_shard = defaultdict(list)
        completed_treecells_grouped_by_treelevel_and_shardworker = defaultdict(list)
        for annotations_one_treecell_ram_file in annotations_one_treecell_ram_files:
            # logging.info(f"annotations_one_treecell_ram_file: {annotations_one_treecell_ram_file}")
            treecell_filename = os.path.basename(annotations_one_treecell_ram_file)
            if "treelevel-0_" in treecell_filename or "treelevel-1_" in treecell_filename \
                or "treelevel-00_" in treecell_filename or "treelevel-01_" in treecell_filename:
                logging.info(f"Upper tree level file treecell_filename: {os.path.basename(treecell_filename)}")
            assert treecell_filename.endswith(".csv")
            treecell_filename_no_ext = treecell_filename[:treecell_filename.rindex('.')]
            pcs = treecell_filename_no_ext.split('__')
            tree_level = int(pcs[3].split('-')[1])
            shard_hex = pcs[5].split('-')[1]
            shard_worker_desc_file_hash, shard_worker_desc = shard_worker_lookup[shard_hex]
            treelevel_shards[tree_level].add(shard_hex)  # Debug only
            completed_treecells_grouped_by_treelevel_and_shard[shard_worker_desc_file_hash].append(annotations_one_treecell_ram_file)
            completed_treecells_grouped_by_treelevel_and_shardworker[shard_worker_desc_file_hash].append(annotations_one_treecell_ram_file)

        # Debug
        # for tree_level, shard_hexes in treelevel_shards.items():
        #     logging.info(f"\nTree level {tree_level} shards:")
        #     sharding_spec = anno.ShardingSpec(
        #         hash=config['SPATIAL_SHARDING_HASH'],
        #         preshift_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['preshift_bits'],
        #         shard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits'],
        #         minishard_bits=config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['minishard_bits'])

        #     # Generate the full shard used/unused output as one text line
        #     s = f"  {tree_level:0>2}:"
        #     for shard_num in range(sharding_spec.num_shards):
        #         shard_hex = sharding.get_shard_hex(shard_num, config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['shard_bits'])
        #         s += "_" + (shard_hex if shard_hex in shard_hexes else "_" * len(shard_hex))
        #     logging.info(s)
        logging.info("")

        # GROUP_BY_SHARD_WORKER = false is the older implementation, in which all results are separated by shard. Setting this switch to true enables shard worker grouping to facilitate the next capsule in only reading the files it needs, in true MapReduce Reducer style, which is already implemented for the ID and relation indices. Once debuggedd and confirmed, the true setting should be used going forward.
        GROUP_BY_SHARD_WORKER = True
        completed_treecells_src = completed_treecells_grouped_by_treelevel_and_shard if not GROUP_BY_SHARD_WORKER else completed_treecells_grouped_by_treelevel_and_shardworker

        # for shard_hex, annotations_one_treecell_ram_files in completed_treecells_grouped_by_treelevel_and_shard.items():
        for shard_hex_key, annotations_one_treecell_ram_files in completed_treecells_src.items():
            shard_hex_key_lbl = "shard" if not GROUP_BY_SHARD_WORKER else "shard_worker"

            subdir = f"{results_loc}split-{split_id:03}/"
            os.makedirs(subdir, exist_ok=True)
            archive_filepath = f"{subdir}split-{split_id:03}@{num_splits}__subsplit-{subsplit_id:02}__{shard_hex_key_lbl}-{shard_hex_key}__archive.txt"
            logging.info(f"Archiving {len(annotations_one_treecell_ram_files)} tree cell files for {shard_hex_key_lbl} {shard_hex_key} to {archive_filepath}")
            num_ram_data_pond_files_this_shard = 0
            total_data_size_this_shard = 0
            with open(archive_filepath, "w") as f:
                for annotations_one_treecell_ram_file in annotations_one_treecell_ram_files:
                    start_timeblock("prep file to add to in-memory archive")
                    num_ram_data_pond_files_this_shard += 1
                    file_size_bytes = ram_data_pond.getsize(annotations_one_treecell_ram_file)
                    total_data_size_this_shard += file_size_bytes
                    total_data_size += file_size_bytes
                    annotations_one_treecell_ram_file_tail = annotations_one_treecell_ram_file[len(results_loc):]
                    if "treelevel-0_" in annotations_one_treecell_ram_file_tail or "treelevel-1_" in annotations_one_treecell_ram_file_tail \
                        or "treelevel-00_" in annotations_one_treecell_ram_file_tail or "treelevel-01_" in annotations_one_treecell_ram_file_tail:
                        logging.info(f"  Upper tree level file {file_size_bytes/1000000:.1f} MB {os.path.basename(annotations_one_treecell_ram_file)}\n    {os.path.basename(annotations_one_treecell_ram_file_tail)}")
                    file_contents = ram_data_pond.read_from_disk_or_ram_data_pond(annotations_one_treecell_ram_file)
                    end_start_timeblocks("prep file to add to in-memory archive", "add file to in-memory archive")
                    ram_data_pond.archive_str_data(annotations_one_treecell_ram_file, file_contents, f"{results_loc}completed_treecells/", f)
                    end_timeblock("add file to in-memory archive")
            # logging.info(f"Wrote {num_ram_data_pond_files_this_shard:>3} RAM data pond files for {shard_hex_key_lbl} {shard_hex_key} of {total_data_size_this_shard/1000000:,}M to disk, resulting in archive file of length {os.path.getsize(archive_filepath)/1000000:,}M")
        end_timeblock("archive RAM data pond in memory with shard grouping")
    else:
        # Archive without sharding, using the custom method instead of tarfile
        start_timeblock("archive RAM data pond in memory")
        # logging.info(f"Archiving {len(annotations_one_treecell_ram_files)} RAM data pond files to in-memory archive buffer")
        num_ram_data_pond_files_a, num_ram_data_pond_files_b = 0, 0
        archive_filepath = f"{results_loc}split-{split_id:03}@{num_splits}__subsplit-{subsplit_id:02}_archive.txt"
        with open(archive_filepath, "w") as f:
            # for annotations_one_treecell_ram_dir in annotations_one_treecell_ram_dirs:
            #     # logging.info(f"  {annotations_one_treecell_ram_dir}")
            #     start_timeblock("glob ram data pond at level 4")
            #     # annotations_one_treecell_ram_files = ram_data_pond.glob_disk_or_ram_data_pond(f"{annotations_one_treecell_ram_dir}/annotations_one_treecell__subsplit-{subsplit_id:02}__*")
            #     # logging.info("JJJ")
            #     annotations_one_treecell_ram_files = ram_data_pond.fastglob_ram_data_pond(f"{annotations_one_treecell_ram_dir}/annotations_one_treecell__subsplit-{subsplit_id:02}__")
            #     end_timeblock("glob ram data pond at level 4")
            #     num_ram_data_pond_files_a += len(annotations_one_treecell_ram_files)
            #     for annotations_one_treecell_ram_file in annotations_one_treecell_ram_files:
            #         start_timeblock("prep file to add to in-memory archive")
            #         num_ram_data_pond_files_b += 1
            #         file_size_bytes = ram_data_pond.getsize(annotations_one_treecell_ram_file)
            #         total_data_size += file_size_bytes
            #         annotations_one_treecell_ram_file_tail = annotations_one_treecell_ram_file[len(results_loc):]
            #         # logging.info(f"    {file_size_bytes/1000000:.1f} MB {annotations_one_treecell_ram_file}\n  {annotations_one_treecell_ram_file_tail}")
            #         file_contents = ram_data_pond.read_from_disk_or_ram_data_pond(annotations_one_treecell_ram_file)
            #         end_start_timeblocks("prep file to add to in-memory archive", "add file to in-memory archive")
            #         ram_data_pond.archive_str_data(file_contents, f)
            #         end_timeblock("add file to in-memory archive")
            for annotations_one_treecell_ram_file in annotations_one_treecell_ram_files:
                start_timeblock("prep file to add to in-memory archive")
                num_ram_data_pond_files_b += 1
                file_size_bytes = ram_data_pond.getsize(annotations_one_treecell_ram_file)
                total_data_size += file_size_bytes
                annotations_one_treecell_ram_file_tail = annotations_one_treecell_ram_file[len(results_loc):]
                # logging.info(f"    {file_size_bytes/1000000:.1f} MB {annotations_one_treecell_ram_file}\n  {annotations_one_treecell_ram_file_tail}")
                file_contents = ram_data_pond.read_from_disk_or_ram_data_pond(annotations_one_treecell_ram_file)
                end_start_timeblocks("prep file to add to in-memory archive", "add file to in-memory archive")
                ram_data_pond.archive_str_data(annotations_one_treecell_ram_file, file_contents, f"{results_loc}completed_treecells/", f)
                end_timeblock("add file to in-memory archive")
        # logging.info(f"Wrote {num_ram_data_pond_files_a}|{num_ram_data_pond_files_b} RAM data pond files of {total_data_size/1000000:,}M to disk, resulting in archive file of length {os.path.getsize(archive_filepath)/1000000:,}M")
        end_timeblock("archive RAM data pond in memory")

    logging.info(f"Wrote {total_data_size/1000000:,}M from RAM data pond to disk")
    end_timeblock("write RAM data pond to disk")

def upload_results_to_bucket():
    start_timeblock("upload results to external storage")
    if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] != "internal":
        st = default_timer()
        files_to_upload_to_scratch = sorted(list(glob.glob(f"{results_loc}split*/*")))
        logging.info(f"files_to_upload_to_scratch (first 30 shown):\n  {'\n  '.join(files_to_upload_to_scratch[:30])}")
        if config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "gcp":
            raise RuntimeError("GCP bucket no longer supported due to possible financial cost if done incorrectly!")
            logging.info("\nUploading files to Google storage")
            filenames_to_upload_to_scratch = [os.path.basename(filepath) for filepath in files_to_upload_to_scratch]
            upload_files_to_gcp(results_loc, filenames_to_upload_to_scratch, f"{config['TIMESTAMP']}/spatial_index", config['GCP_BUCKET'], config['GCP_SCRATCH_BLOB_PATH'])#, dryrun=True)
        elif config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] == "aws":
            logging.info("\nUploading files to AWS storage")
            upload_folder_relative_path = f"aws_upload/{config['TIMESTAMP']}/spatial_index/tree_builder/"
            os.makedirs(f"{results_loc}{upload_folder_relative_path}", exist_ok=True)
            for file in files_to_upload_to_scratch:
                # logging.info(f"Move {file} -> {results_loc}{upload_folder_relative_path}{os.path.basename(file)}")
                shutil.move(file, f"{results_loc}{upload_folder_relative_path}{os.path.basename(file)}")
            upload_folder_to_aws(f"{results_loc}aws_upload/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])#, dryrun=True)
            query_folder_on_aws(f"{config['TIMESTAMP']}/spatial_index/tree_builder/", config['AWS_BUCKET'], config['AWS_PROJECT_PATH'])
            for file in files_to_upload_to_scratch:
                file_dir = '/'.join(file.split('/')[:-1])
                # Move the files back so we can delete them from the original location below
                # logging.info(f"Move {results_loc}{upload_folder_relative_path}{os.path.basename(file)} -> {file}")
                shutil.move(f"{results_loc}{upload_folder_relative_path}{os.path.basename(file)}", file)
            shutil.rmtree(f"{results_loc}aws_upload/")

        t1 = default_timer()
        logging.info(f"External bucket upload elapsed time: {seconds_to_hms(t1 - st)}")

        logging.info("")
        for file in files_to_upload_to_scratch:
            logging.info(f"Deleting result file after uploading to external bucket: {file}")
            os.remove(file)

        t1 = default_timer()
        logging.info(f"Delete results elapsed time: {seconds_to_hms(t1 - st)}")
    else:
        logging.info(f"\n{data_loc}PASS_DATA_BETWEEN_CAPSULES_METHOD indicates Code Ocean. Results won't be uploaded externally.")

    end_timeblock("upload results to external storage")

# def pack_completed_treecells(split_id):
#     # Gather all the completed tree cell directories, optionally grouping them by shard.
#     # Completed tree cells from the final tree level will be in the results directory,
#     # while completed tree cells from all higher tree levels will be in the source directory.
#     # Actually, because of move_upstream_completed_tree_outputs_to_results(), src shouldn't have any, so just assert that to be sure.
#     start_timeblock("pack_completed_treecells()")

#     if config['ARCHIVE_OUTPUT']:
#         # Most of this module offered the option of working within the RAM data pond instead of disk,
#         # but at this final stage, we need to archive actual files on disk,
#         # so it doesn't make sense to rely on the RAM data pond here.
#         # Consequently, the references to ram_data_pond below need to be carefully scrutinized. I'm not sure they are correct.
#         # Perhaps they should "be replaced with the versions they replaced", commented out one line above them in each instance.

#         # assert not list(glob.glob(f"{src_loc}completed_treecells/annotations_one_treecell*"))
#         # treecell_dirs = list(glob.glob(f"{results_loc}completed_treecells/annotations_one_treecell*")) + list(glob.glob(f"{src_loc}completed_treecells/annotations_one_treecell*"))
#         # assert not ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}completed_treecells/annotations_one_treecell*", src_loc==data_loc)
#         assert not ram_data_pond.fastglob_ram_data_pond(f"{src_loc}completed_treecells/annotations_one_treecell", src_loc==data_loc)
#         # treecell_dirs = ram_data_pond.glob_disk_or_ram_data_pond(f"{results_loc}completed_treecells/annotations_one_treecell*") + ram_data_pond.glob_disk_or_ram_data_pond(f"{src_loc}completed_treecells/annotations_one_treecell*", src_loc==data_loc)
#         treecell_dirs = ram_data_pond.fastglob_ram_data_pond(f"{results_loc}completed_treecells/annotations_one_treecell") + ram_data_pond.fastglob_ram_data_pond(f"{src_loc}completed_treecells/annotations_one_treecell", src_loc==data_loc)
#         logging.info(f"treecell_dirs ({len(treecell_dirs)}) (first 5 shown):\n  {'\n  '.join(sorted(treecell_dirs[:5])).strip()}\n")

#         completed_treecells_grouped_by_treelevel_and_shard = defaultdict(list)
#         for treecell_dir in treecell_dirs:
#             # logging.info(f"treecell_dir: {treecell_dir}")
#             treecell_dirname = os.path.basename(treecell_dir)
#             pcs = treecell_dirname.split('__')
#             treecell_tree_level = int(pcs[2].split('-')[1])
#             cell_id = pcs[3]
#             treecell_tree_level_cell_id = [int(v) for v in cell_id.split('-')[1].split(',')]
#             shard_hex = get_shard_hex(treecell_tree_level, treecell_tree_level_cell_id)
#             completed_treecells_grouped_by_treelevel_and_shard[shard_hex].append(treecell_dir)

#         if config['ARCHIVE_COMPLETED_TREECELLS_WITH_SHARD_GROUPING']:
#             # Archive all the completed tree cell directories with grouping them by shard.
#             # This produces a larger number of files since it separates the shards,
#             # which enables great horizontal distribution in the next stage,
#             # but increasing the number of archive files can slow down Nextflow's file handling between capsules.
#             for shard_hex, treecell_dirs in completed_treecells_grouped_by_treelevel_and_shard.items():
#                 logging.info(f"Packing {len(treecell_dirs)} tree cell dirs for shard {shard_hex}")
#                 random.seed()
#                 subdir = f"{results_loc}split-{split_id:03}/"
#                 os.makedirs(subdir, exist_ok=True)
#                 ext, mode = (".tar.gz", "w:gz") if config['COMPRESS_ARCHIVE'] else (".tar", "w")
#                 with tarfile.open(f"{subdir}completed_treecells__shard-{shard_hex}__{hex(int(random.random()*1000000000000))[2:]}{ext}", mode) as tar:
#                     for treecell_dir in treecell_dirs:
#                         if treecell_dir[-1] == '/':
#                             treecell_dir = treecell_dir[:-1]
#                         # logging.info(f"  Adding treecell dir to tar: {treecell_dir}")
#                         tar.add(treecell_dir, arcname=os.path.basename(treecell_dir))
#                 for treecell_dir in treecell_dirs:
#                     shutil.rmtree(treecell_dir)
#         else:
#             # Archive all the completed tree cell directories without grouping them by shard
#             logging.info(f"Packing {len(treecell_dirs)} tree cell dirs for alls shards")
#             random.seed()
#             subdir = f"{results_loc}split-{split_id:03}/"
#             os.makedirs(subdir, exist_ok=True)
#             ext, mode = (".tar.gz", "w:gz") if config['COMPRESS_ARCHIVE'] else (".tar", "w")
#             with tarfile.open(f"{subdir}completed_treecells__{hex(int(random.random()*1000000000000))[2:]}{ext}", mode) as tar:
#                 for shard_hex, treecell_dirs in completed_treecells_grouped_by_treelevel_and_shard.items():
#                     logging.info(f"  Adding {len(treecell_dirs)} treecell dirs for shard {shard_hex} to the archive")
#                     for treecell_dir in treecell_dirs:
#                         if treecell_dir[-1] == '/':
#                             treecell_dir = treecell_dir[:-1]
#                         # logging.info(f"    Adding treecell dir to tar: {treecell_dir}")
#                         tar.add(treecell_dir, arcname=os.path.basename(treecell_dir))
#                     for treecell_dir in treecell_dirs:
#                         shutil.rmtree(treecell_dir)

#     end_timeblock("pack_completed_treecells()")

def process_subsplit(subsplit_id, subsplit_range_row_start, subsplit_range_row_end):
    start_timeblock("process_subsplit()")

    split_id = None
    src_loc = data_loc
    treelevel_iter = -1
    tree_level_shard_histograms = []
    process_next_tree_level = True
    while process_next_tree_level:
        start_timeblock("tree_level_loop")
        treelevel_iter += 1
        logging.info("\n" + "*" * 100)
        tree_level_shard_histograms.append(Counter())
        start_timeblock("call process_one_tree_level()")
        result = process_one_tree_level(subsplit_id, subsplit_range_row_start, subsplit_range_row_end, treelevel_iter, src_loc, tree_level_shard_histograms)
        if result is None:
            # Subsplits are done
            end_timeblock("call process_one_tree_level()")
            end_end_timeblocks("tree_level_loop", "process_subsplit()")
            return None, False
        split_id, num_splits, src_loc, process_next_tree_level = result
        end_timeblock("call process_one_tree_level()")
        if split_id is None:
            end_timeblock("tree_level_loop")
            break
        analyze_memory_usage()
        ram_data_pond_size = ram_data_pond.get_total_size()
        logging.info(f"RAM data pond size: {len(ram_data_pond.ram_data_pond)} items    {ram_data_pond_size} B    {ram_data_pond_size / 1000:,.1f} KB    {ram_data_pond_size / 1000000:,.1f} MB\n")
        logging.error(f"Elapsed times at mid-processing after tree level {treelevel_iter}:")
        logging.error(f"BEWARE!  BEWARE!  BEWARE!\n  Outer times will not show full accumulated time relative to their inner times at these mid-processing stages.\n  Only the final profile display will show all accumulated time.")
        dump_profile(False)
        end_timeblock("tree_level_loop")

    logging.info("\n" + "* " * 50 + "\n")

    # logging.info("Tree level, tree cell shard histograms:\n")
    # for histogram_tree_level, tree_level_shard_histogram in enumerate(tree_level_shard_histograms):
    #     logging.info(f"  Tree level: {histogram_tree_level}")
    #     logging.info(f"  {tree_level_shard_histogram.most_common()}\n")

    if split_id is not None:
        # ARCHIVE_MEMORY_STORE: Pack RAM data pond into a tar or custom buffer and then write a single tar or custom archive file to disk.
        # Else, write each RAM data pond file to a separate file on disk (which could be 1000s and impede CodeOcean performance between capsules).
        ARCHIVE_MEMORY_STORE = True

        if ram_data_pond.ram_data_pond is not None:
            archive_results(ARCHIVE_MEMORY_STORE, subsplit_id, split_id, num_splits)

        # if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
        #     upload_results_to_bucket()
        # else:
        #     logging.info(f"\n{data_loc}DEBUG_FLAG.txt file found. Results won't be uploaded externally.")

        analyze_memory_usage()
        ram_data_pond_size = ram_data_pond.get_total_size()
        logging.info(f"RAM data pond size: {len(ram_data_pond.ram_data_pond)} items    {ram_data_pond_size} B    {ram_data_pond_size / 1000:,.1f} KB    {ram_data_pond_size / 1000000:,.1f} MB\n")

        # if not ARCHIVE_MEMORY_STORE:
        #     pack_completed_treecells(split_id)
        #     analyze_memory_usage()
        #     ram_data_pond_size = ram_data_pond.get_total_size()
        #     logging.info(f"RAM data pond size: {len(ram_data_pond.ram_data_pond)} items    {ram_data_pond_size} B    {ram_data_pond_size / 1000:,.1f} KB    {ram_data_pond_size / 1000000:,.1f} MB\n")

    end_timeblock("process_subsplit()")

    return split_id, True

def collate_subsplit_archives():
    start_start_timeblocks("collate_subsplit_archives()", "collect_subsplit_archives")

    logging.info("\nCollating subsplit archives")

    list_directory(f"{results_loc}*")
    list_directory(f"{results_loc}*/*")
    list_directory(f"{results_loc}*/*/*")

    split_dirs = glob.glob(f"{results_loc}/split-*/")
    for split_dir in split_dirs:
        logging.info(f"\n  Collect split dir {split_dir}")
        multiarchive_sections = {}
        for shard_worker_desc_file_hash, shard_worker_desc in shard_worker_descs:
            assigned_shards = '_'.join(shard_worker_desc)
            logging.info(f"\n    Collecting shard worker {assigned_shards}")
            split_archives = glob.glob(f"{split_dir}/*shard_worker-{shard_worker_desc_file_hash}*")
            logging.info(f"    Split archives for shard worker {assigned_shards}:\n        {'\n        '.join(split_archives)}")
            for archive_i, archive in enumerate(split_archives):
                if archive_i < 1:
                    logging.info(f"\n      Loading archive {archive_i+1} of {len(split_archives)} {archive}")
                archive_sections = ram_data_pond.dearchive_file_and_return_sectioned(archive)
                if archive_i < 1:
                    logging.info(f"      Archive {archive_i+1} sections:")
                # logging.info(f"          {'\n          '.join([v.split('/')[0] for v in archive_sections.keys()])}")
                for archive_section_i, archive_section in enumerate(archive_sections):
                    pcs = archive_section.split('/')[0].split('__')
                    archive_section_nosubsplit = pcs[0] + '__' + '__'.join(pcs[2:])
                    if archive_section_nosubsplit not in multiarchive_sections:
                        multiarchive_sections[archive_section_nosubsplit] = ""
                    else:
                        multiarchive_sections[archive_section_nosubsplit] += "\n"
                    if archive_i < 1 and archive_section_i < 1:
                        logging.info(f"          Archive {archive_i+1} section: {archive_section_nosubsplit.split('/')[0]}")
                        logging.info(f"          Archive {archive_i+1} section length: {len(archive_sections[archive_section])}")
                    multiarchive_sections[archive_section_nosubsplit] += archive_sections[archive_section]

                if archive_i < 1:
                    logging.info(f"\n      Deleting subsplit archive {archive_i+1}: {archive}")
                os.remove(archive)

        # logging.info(f"multiarchive_sections keys: {multiarchive_sections.keys()}")

        end_start_timeblocks("collect_subsplit_archives", "merge_archives")

        logging.info("\n  Merging archives")
        for multiarchive_section_i, multiarchive_section in enumerate(multiarchive_sections):
            if multiarchive_section_i < 1:
                # logging.info(f"    Multiarchive:\n    {multiarchive_section}\n    {multiarchive_sections[multiarchive_section]}")
                logging.info(f"    Multiarchive:\n    {multiarchive_section}")
                # logging.info(f"    Multiarchive: {multiarchive_section.split('/')[0]}")
            filedir = multiarchive_section
            filename = filedir + ".csv"
            split_desc, shard_desc = filedir.split('__')[1], filedir.split('__')[4]
            shard = shard_desc.split('-')[1]
            shard_worker_desc_file_hash, shard_worker_desc = shard_worker_lookup[shard]
            if multiarchive_section_i < 1:
                logging.info(f"      split_desc, shard_desc, shard, shard_worker_hash, shard_worker:    {split_desc}    {shard_desc}    {shard}    {shard_worker_desc_file_hash}    {shard_worker_desc}")
            multiarchive_filename = f"{split_desc}__shard_worker-{shard_worker_desc_file_hash}__archive.txt"
            if multiarchive_section_i < 1:
                logging.info(f"      Multiarchive filepath: {split_dir}{multiarchive_filename}")
            with open(f"{split_dir}{multiarchive_filename}", 'a') as f:
                content = multiarchive_sections[multiarchive_section].strip()
                f.write(f"{multiarchive_section}/{multiarchive_section}.csv\n")
                f.write(f"{len(content)}\n")
                f.write(f"{content}\n")

                if multiarchive_section_i < 1:
                    # logging.info(f"      Archived filepath: {multiarchive_section}")
                    logging.info(f"      Archived filepath content length: {len(content)}\n")

    logging.info("\nResults after merging subsplits:")
    list_directory(f"{results_loc}*")
    list_directory(f"{results_loc}*/*")
    list_directory(f"{results_loc}*/*/*")

    end_end_timeblocks("merge_archives", "collate_subsplit_archives()")

if __name__ == "__main__":
    data_loc = "../data/"
    results_loc = "../results/"

    logging_uid = hex(int(random.random()*1000000000000))[2:]

    # There shouldn't be any upstream logs in this capsule
    # # Copy upstream logs from input to output
    # logs = sorted(list(glob.glob(f"{data_loc}log*.log")))
    # for log in logs:
    #     shutil.copy(log, f"{results_loc}{os.path.basename(log)}")

    # ____________________________________________________________________________________________________
    # logging.basicConfig(level=logging.INFO, handlers=[
        #     logging.StreamHandler(sys.stdout),
        #     logging.FileHandler(f"../results/log_{logging_uid}.log", mode="a")
        # ], format='%(message)s')
    # config = {}
    # test_nested_profiler()
    # sys.exit(0)
    # ____________________________________________________________________________________________________

    # ____________________________________________________________________________________________________
    # logging.basicConfig(level=logging.INFO, handlers=[
        #     logging.StreamHandler(sys.stdout),
        #     logging.FileHandler(f"../results/log_{logging_uid}.log", mode="a")
        # ], format='%(message)s')
    # config = {}
    # test_sharding()
    # sys.exit(0)
    # ____________________________________________________________________________________________________

    # ____________________________________________________________________________________________________
    # logging.basicConfig(level=logging.INFO, handlers=[
        #     logging.StreamHandler(sys.stdout),
        #     logging.FileHandler(f"../results/log_{logging_uid}.log", mode="a")
        # ], format='%(message)s')
    # config = {}
    # test_morton_code_and_shardhex()
    # sys.exit(0)
    # ____________________________________________________________________________________________________

    logging.basicConfig(level=logging.CRITICAL, handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{results_loc}log_build_spatial_index_oct_tree_{logging_uid}.log", mode="a")
        ], format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("BUILD SPATIAL INDEX OCT TREE")

    analyze_memory_usage()

    ram_data_pond = RAMDataPond(True)

    analyze_memory_usage()
    ram_data_pond_size = ram_data_pond.get_total_size()
    logging.info(f"RAM data pond size: {len(ram_data_pond.ram_data_pond)} items    {ram_data_pond_size} B    {ram_data_pond_size / 1000:,.1f} KB    {ram_data_pond_size / 1000000:,.1f} MB\n")

    # profile_labels = []
    # timestamps = []
    # elap_accum_times = Counter()

    start_start_timeblocks("run capsule", "init_stuff")

    # Make sure this subpipeline's config is loaded last so it can override any other config values
    config = read_config(["id", "relation", "spatial"])
    logging.basicConfig(level=get_logging_level_from_desc(config['LOGGING_LEVEL']), handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{results_loc}log_build_spatial_index_oct_tree_{logging_uid}.log", mode="a")
        ], format=config['LOGGING_FORMAT'], force=True)

    for module in ['simple_writer_no_spatial_indexing', 'sharding', 'annotations']:
        logging.getLogger(module).setLevel(get_logging_level_from_desc(config['PRECOMPUTED_FILE_WRITER_LOGGING_LEVEL']))
        logging.getLogger(module).addHandler(logging.StreamHandler(sys.stdout))
        logging.getLogger(module).addHandler(logging.FileHandler(f"{results_loc}log_build_spatial_index_oct_tree_{logging_uid}.log", mode="a"))

    if config['SPATIAL_INDEX_ENABLED']:
        data_loc_contents = sorted(os.listdir(data_loc))
        data_loc_contents = [v for v in data_loc_contents if "placeholder" not in v]
        logging.info(f"{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):")
        logging.info('  ' + '\n  '.join(data_loc_contents[:30]).strip() + '\n')

        logging.info(f"{data_loc} subcontents ({len(list(glob.glob(f'{data_loc}*/*')))}) (first 5 shown):\n  {'\n  '.join(sorted(list(glob.glob(f'{data_loc}*/*'))[:5])).strip()}\n")

        start_timeblock("read_shard_worker_descriptions")

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
            logging.info(f"One shard worker assigned shards: {assigned_shards}")
            shard_worker_descs.add((shard_worker_desc_file_hash, assigned_shards))
        logging.info("\n")

        end_timeblock("read_shard_worker_descriptions")

        # Detect an empty input directory or the presence of the no-op file, indicating that this capsule isn't being used in the current pipeline.
        data_loc_contents_set = set(data_loc_contents)
        if data_loc_contents_set == set(["job_config.py", "job_spatial_config.py"]) or "no_op.txt" in data_loc_contents_set:
            logging.info("Empty data input directory or no-op file found. Presumably, this capsule isn't being used in the current pipeline.")
            # with open(f"{results_loc}no_op.txt", 'w') as f:
            #     f.write("no_op\n""Empty input directory or no-op file found. Presumably, this capsule isn't being used in the current pipeline.")
            ram_data_pond.write_to_disk_or_ram_data_pond(f"{results_loc}no_op.txt", "no_op\n""Empty input directory or no-op file found. Presumably, this capsule isn't being used in the current pipeline.")
            end_timeblock("init_stuff")
        else:
            # If this is the first iteration, read from the upstream input directory (initialize src_loc to data_loc).
            # If this is any subsequent iteration, read from the previous iteration's results directory.

            analyze_memory_usage()
            ram_data_pond_size = ram_data_pond.get_total_size()
            logging.info(f"RAM data pond size: {len(ram_data_pond.ram_data_pond)} items    {ram_data_pond_size} B    {ram_data_pond_size / 1000:,.1f} KB    {ram_data_pond_size / 1000000:,.1f} MB\n")

            end_timeblock("init_stuff")

            src_loc = None  # Define here so it has global scope later
            subsplit_range_row_start, subsplit_range_row_end = 0, 0
            num_subsplits = config['DATA_CONFIG']['data_size'][6]
            for subsplit_idx in range(num_subsplits):
                logging.info("\n" + "#" * 100)
                logging.info("#" * 100)
                logging.info("#" * 100 + "\n")

                subsplit_range_row_end += config['DATA_CONFIG']['data_size'][5]
                logging.info(f"Subsplit {subsplit_idx+1} of {num_subsplits}: size, start, end: {config['DATA_CONFIG']['data_size'][5]} {subsplit_range_row_start} {subsplit_range_row_end}")
                analyze_memory_usage()

                if subsplit_idx == num_subsplits - 1:
                    # Force the last split to gather any lingering rows at the end of the file
                    logging.info("Last subsplit")
                    subsplit_range_row_end = None
                subsplit_split_id, success = process_subsplit(subsplit_idx + 1, subsplit_range_row_start, subsplit_range_row_end)
                if subsplit_split_id is not None:
                    split_id = subsplit_split_id
                if not success:
                    break
                subsplit_range_row_start += config['DATA_CONFIG']['data_size'][5]

                analyze_memory_usage()
                ram_data_pond.clear()

            logging.info("\n" + "#" * 100)
            logging.info("#" * 100)
            logging.info("#" * 100 + "\n")

            # The archives are divided by subsplit at this point, but the next capsule will expect them to be grouped by split,
            # so we need to collate them back together now.
            collate_subsplit_archives()

            # Tar the treecell shard summary files so we pass few total files through CO to the next capsules.
            # These files are relatively small, but they can be numerous, when summed across all splits.
            # tree_cell_shard_files = sorted(list(glob.glob(f"{results_loc}subsplit-*__split-*__tree_level-*__tree_cell_shards.txt")))
            # if tree_cell_shard_files:
            #     split_desc = os.path.basename(tree_cell_shard_files[0]).split('__')[1]
            #     ext, mode = (".tar.gz", "w:gz") if config['COMPRESS_ARCHIVE'] else (".tar", "w")
            #     with tarfile.open(f"{results_loc}tree_cell_shards__{split_desc}{ext}", mode) as tar:
            #         for tree_cell_shard_file in tree_cell_shard_files:
            #             tar.add(tree_cell_shard_file, arcname=os.path.basename(tree_cell_shard_file))
            #             os.remove(tree_cell_shard_file)

            if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
                upload_results_to_bucket()
            else:
                logging.info(f"\n{data_loc}DEBUG_FLAG.txt file found. Results won't be uploaded externally.")

        logging.info(f"ZZZ {split_id}")
        if split_id == 1:  # To avoid CodeOcean name collisions, only do this from one capsule
            logging.info("Copying config files to results for next capsule")
            for f in glob.glob(f"{data_loc}*config*.py"):
                shutil.copy(f, f"{results_loc}{os.path.basename(f)}")
    else:
        end_timeblock("init_stuff")

    start_timeblock("finalize_results()")
    finalize_results(results_loc)
    end_timeblock("finalize_results()")

    analyze_memory_usage()
    ram_data_pond_size = ram_data_pond.get_total_size()
    logging.info(f"RAM data pond size: {len(ram_data_pond.ram_data_pond)} items    {ram_data_pond_size} B    {ram_data_pond_size / 1000:,.1f} KB    {ram_data_pond_size / 1000000:,.1f} MB\n")

    end_timeblock("run capsule")

    dump_profile()

    logging.info("\nDone")
    process_running_time()
