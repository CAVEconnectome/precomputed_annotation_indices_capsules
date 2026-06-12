import logging
from typing import Literal
import math

def generate_sharding_spec(
    total_count,
    total_bytes,
    hashtype: Literal["murmurhash3_x86_128", "identity"] = "murmurhash3_x86_128",
    gzip_compress=True,
    MINISHARD_TARGET_COUNT=1000,
    SHARD_TARGET_SIZE=50000000,
):
    """
    This is Jeremy's code, as given to Keith by Forrest
    """

    if total_count == 1:
        return None
    # if ts is None:
    #     return None

    # test if hashtype is valid
    if hashtype not in ["murmurhash3_x86_128", "identity"]:
        raise ValueError(
            f"Invalid hashtype {hashtype}."
            "Must be one of 'murmurhash3_x86_128' "
            "or 'identity'"
        )

    total_minishard_bits = 0
    while (total_count >> total_minishard_bits) > MINISHARD_TARGET_COUNT:
        total_minishard_bits += 1

    shard_bits = 0
    while (total_bytes >> shard_bits) > SHARD_TARGET_SIZE:
        shard_bits += 1

    preshift_bits = 0
    while MINISHARD_TARGET_COUNT >> preshift_bits:
        preshift_bits += 1

    minishard_bits = total_minishard_bits - min(total_minishard_bits, shard_bits)
    data_encoding: Literal["raw", "gzip"] = "raw"
    minishard_index_encoding: Literal["raw", "gzip"] = "raw"

    if gzip_compress:
        data_encoding = "gzip"
        minishard_index_encoding = "gzip"

    # Nonzero preshift_bits generate indices that don't work in Neuroglancer
    preshift_bits = 0

    # return ShardSpec(
    #     type="neuroglancer_uint64_sharded_v1",
    #     hash=hashtype,
    #     preshift_bits=preshift_bits,
    #     shard_bits=shard_bits,
    #     minishard_bits=minishard_bits,
    #     data_encoding=data_encoding,
    #     minishard_index_encoding=minishard_index_encoding,
    # )
    return {
        "hashtype": hashtype,
        "preshift_bits": preshift_bits,
        "shard_bits": shard_bits,
        "minishard_bits": minishard_bits,
        "data_encoding": data_encoding,
        "minishard_index_encoding": minishard_index_encoding,
    }

def generate_spatial_index_sharding_spec_2(
    tree_level,
    num_annotations_per_tree_cell,
    data_size,
    hashtype: Literal["murmurhash3_x86_128", "identity"] = "murmurhash3_x86_128",
    gzip_compress=True,
    MINISHARD_TARGET_COUNT=1000,
    SHARD_TARGET_SIZE=50000000,
):
    grid_dim = 2 ** tree_level
    num_cells = grid_dim ** 3
    
    # The maximum number of annotations stored in this tree cell is the lesser of the theoretical storage limit of the tree level and the actual amount of data available.
    tree_level_num_annotations = min(num_cells * num_annotations_per_tree_cell, data_size[2])
    # round() would be more accurate in the next line, but ceil() is more robust by allocating some headroom
    annotation_size = math.ceil(data_size[1] / data_size[2])
    tree_level_total_size = tree_level_num_annotations * annotation_size
    
    sharding_spec = generate_sharding_spec(
        tree_level_num_annotations,
        tree_level_total_size,
        hashtype,
        gzip_compress,
        MINISHARD_TARGET_COUNT,
        SHARD_TARGET_SIZE,
    )
    return {
        "preshift_bits": 0,  # sharding_spec["preshift_bits"],
        "shard_bits": sharding_spec["shard_bits"],
        "minishard_bits": sharding_spec["minishard_bits"],
    }

def generate_spatial_index_sharding_spec(tree_level, num_annotations_per_tree_cell, data_size, SHARD_TARGET_SIZE=50000000,):
    grid_dim = 2 ** tree_level
    num_cells = grid_dim ** 3

    # The maximum number of annotations stored in this tree cell is the lesser of the theoretical storage limit of the tree level and the actual amount of data available.
    tree_level_num_annotations = min(num_cells * num_annotations_per_tree_cell, data_size[2])
    # round() would be more accurate in the next line, but ceil() is more robust by allocating some headroom
    annotation_size = math.ceil(data_size[1] / data_size[2])
    tree_level_total_size = tree_level_num_annotations * annotation_size
    num_shards = math.ceil(tree_level_total_size / SHARD_TARGET_SIZE)
    shard_bits = math.ceil(math.log2(num_shards))

    num_annotations_per_shard = math.ceil(SHARD_TARGET_SIZE / annotation_size)
    minishard_target_count = 6  # round(math.sqrt(num_annotations_per_shard))  # I can't remember why I hard-coded this. Did varying minishards break the index? TODO: investigate this

    preshift_bits = 0
    minishard_bits = math.ceil(math.log2(minishard_target_count))

    logging.critical("Inputs:")
    logging.critical(f"  Tree level:                    {tree_level:40}")
    logging.critical(f"  Num annotations per tree cell: {num_annotations_per_tree_cell:40,}")
    logging.critical(f"  Annotation size:               {annotation_size:40,} B")
    logging.critical(f"  Shard target size:             {SHARD_TARGET_SIZE:40,} B")

    logging.critical("Grid:")
    logging.critical(f"  Grid dim:                      {grid_dim:40}")
    logging.critical(f"  Num cells:                     {num_cells:40,}")

    logging.critical("Outputs:")
    logging.critical(f"  Tree level num annotations:    {tree_level_num_annotations:40,}")
    logging.critical(f"  Tree level total size:         {tree_level_total_size:40,} B")
    logging.critical(f"  Num shards:                    {num_shards:40,}")
    logging.critical(f"  Num annotations per shard:     {num_annotations_per_shard:40,}")
    logging.critical(f"  Minishard target count:        {minishard_target_count:40,}")
    logging.critical(f"  Shard bits:                    {shard_bits:40}")
    logging.critical(f"  Minishard bits:                {minishard_bits:40}")
    logging.critical(f"  Preshift bits:                 {preshift_bits:40}")

    return {
        "preshift_bits": preshift_bits,
        "shard_bits": shard_bits,
        "minishard_bits": minishard_bits,
    }
