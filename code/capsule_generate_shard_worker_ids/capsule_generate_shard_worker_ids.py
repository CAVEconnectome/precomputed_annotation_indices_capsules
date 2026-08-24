import sys
import logging
import os
import math
from timeit import default_timer
import random

from shared.util import *

import shared.sharding as sharding

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("GENERATE SHARD WORKER IDS")

    data_loc = "../data/"
    results_loc = "../results/"

    config = read_config(["relation", "spatial", "id"])
    logging.basicConfig(stream=sys.stdout, level=get_logging_level_from_desc(config['LOGGING_LEVEL']), format=config['LOGGING_FORMAT'], force=True)

    logging.info(f"{data_loc} contents:")
    logging.info('  ' + '\n  '.join(sorted(os.listdir(data_loc))).strip() + '\n')

    shard_bits = None
    if "ID_SHARDING_BITS" in config:
        if shard_bits is not None:
            raise RuntimeError("Sharding bits config parameter found for multiple targets (id/relation/spatial).")
        shard_bits = config["ID_SHARDING_BITS"]
    if "RELATION_SHARDING_BITS" in config:
        if shard_bits is not None:
            raise RuntimeError("Sharding bits config parameter found for multiple targets (id/relation/spatial).")
        shard_bits = config["RELATION_SHARDING_BITS"]
    if "TREE_LEVEL_SHARDING_SPECS" in config:
        if shard_bits is not None:
            raise RuntimeError("Sharding bits config parameter found for multiple targets (id/relation/spatial).")
        shard_bits = 0
        for tree_level_shard_spec in config["TREE_LEVEL_SHARDING_SPECS"]:
            shard_bits = max(shard_bits, tree_level_shard_spec["shard_bits"])
    if shard_bits is None:
        raise RuntimeError("Sharding bits config parameter not found.")
    
    shard_range = 2**shard_bits
    logging.info(f"Shard bits (for spatial pipeline, this is the max shard bits across all tree levels), range: {shard_bits} {shard_range}\n")

    logging.info(f"Num shard workers: {config["NUM_SHARD_WORKERS"]}\n")

    shard_worker_assigned_shards = {i: [] for i in range(config["NUM_SHARD_WORKERS"])}
    next_shard_worker_id = 0
    for shard_number in range(shard_range):
        shard_hex = sharding.get_shard_hex(shard_number, shard_bits)
        # For the spatial index, add all shard hexes, including those for fewer bits, but truncating their leading 0s
        while shard_hex:
            # logging.info(f"Adding shard_hex {shard_hex} to worker {next_shard_worker_id:>3}")
            shard_worker_assigned_shards[next_shard_worker_id].append(shard_hex)
            if "TREE_LEVEL_SHARDING_SPECS" not in config or shard_hex[0] != '0':
                break
            shard_hex = shard_hex[1:]
        next_shard_worker_id = (next_shard_worker_id + 1) % config["NUM_SHARD_WORKERS"]
    
    for swi, (shard_worker, assigned_shards) in enumerate(shard_worker_assigned_shards.items()):
        if assigned_shards:
            assigned_shards_desc = '_'.join(assigned_shards)
            logging.info(f"Shard worker {swi:>3}, assigned shards:    {shard_worker:>3}    {assigned_shards_desc}")
            with open(f"{results_loc}shard_worker_{hex(int(random.random()*1000000000000))[2:]}.txt", 'w') as f:
            # with open(f"{results_loc}shard_worker_{assigned_shards_desc}.txt", 'w') as f:
                f.write(assigned_shards_desc)

    logging.info("\nDone")
    process_running_time()
