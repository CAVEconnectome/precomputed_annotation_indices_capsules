import sys
import logging
import os
import glob
from timeit import default_timer
from collections import Counter
import random
import shutil

from shared.util import *

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

if __name__ == "__main__":
    PRESERVE_ALL_FIlES = False

    data_loc = "../data/"
    results_loc = "../results/"

    logging_uid = hex(int(random.random()*1000000000000))[2:]
    # os.makedirs(f"{results_loc}logs/", exist_ok=True)

    logging.basicConfig(level=logging.CRITICAL, handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{results_loc}log_regroup_spatial_oct_tree_worker_outputs_{logging_uid}.log", mode="a")
        ], format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("STRIP AWAY SPATIAL TREE AND PASS ALONG MAX TREE DEPTH")

    # Make sure this subpipeline's config is loaded last so it can override any other config values
    config = read_config(["id", "relation", "spatial"])
    logging.basicConfig(level=get_logging_level_from_desc(config['LOGGING_LEVEL']), handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{results_loc}log_strip_away_spatial_tree_and_pass_along_max_tree_depth_{logging_uid}.log", mode="a")
        ], format=config['LOGGING_FORMAT'], force=True)

    timestamps = []
    timestamps.append(("start", default_timer()))

    if config['SPATIAL_INDEX_ENABLED']:
        data_loc_contents = sorted(os.listdir(data_loc))
        data_loc_contents = [v for v in data_loc_contents if "placeholder" not in v]
        logging.info(f"{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):")
        logging.info('  ' + '\n  '.join(data_loc_contents[:30]).strip() + '\n')

        logging.info(f"{data_loc} subcontents ({len(list(glob.glob(f"{data_loc}*/*")))}) (first 5 shown):\n  {'\n  '.join(sorted(list(glob.glob(f"{data_loc}*/*"))[:5])).strip()}")
        logging.info("\n")

        shard_worker_desc_file_hash, assigned_shards = read_shardworker_file()

        # Copy upstream logs from input to output
        if "0" in assigned_shards:  # To avoid CodeOcean name collisions, only do this from one capsule
            logs = sorted(list(glob.glob(f"{data_loc}log*.log")))
            for log in logs:
                # logging.info(f"Copying log from {data_loc} to {results_loc}logs/: {log}")
                # shutil.copy(log, f"{results_loc}logs/{os.path.basename(log)}")
                logging.info(f"Copying log from {data_loc} to {results_loc}: {log}")
                shutil.copy(log, f"{results_loc}{os.path.basename(log)}")

        timestamps.append(("read_shardworker_file", default_timer()))

        # Copy the max tree level files to the output.
        # Don't copy the spatial tree directories to the output (they will only be present if UPLOAD_RESULTS_TO_GCP is false)
        max_tree_level_files = list(glob.glob(f"{data_loc}max_tree_level*.txt"))
        for max_tree_level_file in max_tree_level_files:
            filename = os.path.basename(max_tree_level_file)
            logging.info(f"Moving {filename} to results")
            shutil.move(max_tree_level_file, f"{results_loc}{filename}")

    finalize_results(results_loc)
    timestamps.append(("finalize_results", default_timer()))

    logging.error("\nElapsed timestamps:")
    accum_elapsed_times = Counter()
    for ti, time in enumerate(timestamps):
        if ti > 0:
            elap_t = time[1] - timestamps[ti-1][1]
            accum_elapsed_times[time[0]] += elap_t
            logging.error(f"  {seconds_to_hms(elap_t)} {time[0]}")

    logging.error("Accumulated elapsed timestamps:")
    for label, elap_t in accum_elapsed_times.items():
        logging.error(f"  {seconds_to_hms(elap_t)} {label}")

    logging.info("\nDone")
    process_running_time()
