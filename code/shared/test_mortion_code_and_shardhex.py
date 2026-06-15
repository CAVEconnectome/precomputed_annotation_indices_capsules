import logging
import pandas as pd

config = {}

import shared.annotations as anno
import shared.sharding as sharding
import shared.utilities as utilities

def get_test_shard_number(chunk_id: int, hash, preshift_bits, shard_bits, minishard_bits):
    """
    Taken from annotations.py:ShardingSpec.get_shard_number()
    """
    if hash == "identity":
        hashed_chunk_id = chunk_id >> preshift_bits
    elif hash == "murmurhash3_x86_128":  # pragma: no cover
        # Would need to implement MurmurHash3_x86_128 here
        raise NotImplementedError("MurmurHash3_x86_128 not implemented")
    else:  # pragma: no cover
        raise ValueError(f"Unknown hash function: {hash}")

    shard_number = (hashed_chunk_id >> minishard_bits) & ((1 << shard_bits) - 1)

    # desc_str = f"(hashed_chunk_id >> minishard_bits) & ((1 << shard_bits) - 1)    =    ({hashed_chunk_id} >> {minishard_bits}) & ((1 << {shard_bits}) - 1)    =    {(hashed_chunk_id >> minishard_bits) & ((1 << shard_bits) - 1)}"
    desc_str = f"({hashed_chunk_id} >> {minishard_bits}) & ((1 << {shard_bits}) - 1)"

    return hashed_chunk_id, desc_str, shard_number

def get_test_shard_hex(tree_level, tree_level_cell_id):
    grid_dim = 2 ** tree_level
    grid_shape = (grid_dim, grid_dim, grid_dim)
    morton_code = utilities.compressed_morton_code(tree_level_cell_id, grid_shape)
    sharding_spec = anno.ShardingSpec(preshift_bits=config['SPATIAL_PRESHIFT_BITS'], shard_bits=config['SPATIAL_SHARDING_BITS'], minishard_bits=config['SPATIAL_MINISHARDING_BITS'])
    shard_num = sharding_spec.get_shard_number(morton_code)
    hash_ = "identity"
    hashed_chunk_id, desc_str, shard_num2 = get_test_shard_number(morton_code, hash_, config['SPATIAL_PRESHIFT_BITS'], config['SPATIAL_SHARDING_BITS'], config['SPATIAL_MINISHARDING_BITS'])
    if hash_ == "identity" and hashed_chunk_id != morton_code:
        logging.info(f"ERROR! Hashed Morton code != Morton code for 'identity' hash: {hashed_chunk_id} != {morton_code}")
    if shard_num2 != shard_num:
        logging.info(f"ERROR! Shard num mismatch: {shard_num2} != {shard_num}")
    shard_hex = sharding.get_shard_hex(shard_num, config['SPATIAL_SHARDING_BITS'])
    
    return grid_dim, grid_shape, morton_code, hashed_chunk_id, shard_num, desc_str, shard_hex

def test_morton_code_and_shardhex():
    logging.info("Testing compressed Morton code and shard hex")

    config['SPATIAL_PRESHIFT_BITS'] = 0
    config['SPATIAL_SHARDING_BITS'] = 4
    config['SPATIAL_MINISHARDING_BITS'] = 3

    logging.info(f"Preshift bits: {config['SPATIAL_PRESHIFT_BITS']}    Sharding bits: {config['SPATIAL_SHARDING_BITS']}    Minisharding bits: {config['SPATIAL_MINISHARDING_BITS']}")

    rows = []
    for tree_level in range(3):
        grid_dim = 2**tree_level
        grid_shape = (grid_dim, grid_dim, grid_dim)
        logging.info(f"  Tree level: {tree_level:>2}    Dimension: {grid_dim:>3}    Grid shape: {grid_shape}")
        for x in range(grid_dim):
            for y in range(grid_dim):
                for z in range(grid_dim):
                    cell_index = [x, y, z]
                    mc = utilities.compressed_morton_code(cell_index, grid_shape)
                    grid_dim2, grid_shape2, mc2, hashed_chunk_id, shard_num, desc_str, shard_hex = get_test_shard_hex(tree_level, cell_index)
                    if grid_dim2 != grid_dim or grid_shape2 != grid_shape or mc2 != mc:
                        logging.info(f"    ERROR! {grid_dim2} {grid_shape2} {mc2} != {grid_dim} {grid_shape} {mc}")
                    mc_bits = bin(mc)
                    # logging.info(f"    Tree level: {tree_level:>2}    Cell: {x:>3} {y:>3} {z:>3}    MC: {mc:>8}    MC bits: {mc_bits[2:]:>8}    Hashed MC: {hashed_chunk_id:>8}    Math: {desc_str:30}   Shard num: {shard_num:>8}    Shard hex: {shard_hex:>3}")
                    # logging.info(f"        {desc_str}")
                    # logging.info("")
                    row = [tree_level, x, y, z, mc, mc_bits[2:], hashed_chunk_id,
                            config['SPATIAL_PRESHIFT_BITS'], config['SPATIAL_SHARDING_BITS'], config['SPATIAL_MINISHARDING_BITS'],
                            desc_str, shard_num, shard_hex]
                    rows.append(row)
    
    df = pd.DataFrame(rows, columns=["Level", "X", "Y", "Z", "MC", "MC_bits", "MC_hashed", "Preshift_bits", "Shard_bits", "Minishard_bits", "Math", "Shard_num", "Shard_hex"])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    logging.info(df.to_string(index=False))
