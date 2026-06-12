import sys
import logging
from timeit import default_timer
import json
import pprint

from shared.util import *
from sharding_spec_calculations import *

id_config = {
    # "LOGGING_LEVEL": "info",
    # "LOGGING_FORMAT": "%(message)s",
    # "LOGGING_FORMAT": "%(levelname)s: %(message)s",

    "PRECOMPUTED_FILE_WRITER_LOGGING_LEVEL": "info",
    
    "ID_INDEX_ENABLED": True,

    "ID_SHARDING": True,

    "ID_SHARDING_HASH": "murmurhash3_x86_128",  #  "murmurhash3_x86_128", "identity"
    "ID_PRESHIFT_BITS": 0,
    "ID_SHARDING_BITS": 0,
    "ID_MINISHARDING_BITS": 0,

    "ARCHIVE_FORMAT": "parquet_pyarrow",  # None, "", "tar", "parquet_pyarrow", "parquet_fastparquet", "custom" -- Custom method is provided & documented in ram_data_pond.py
    "COMPRESS_ARCHIVE": False,
 
    # Shard grouping separates the output into multiple archives by shard or shard worker.
    # This saves the next capsule the effort of dearchiving data for shards it won't process.
    # However, doing so multiplies the number of archive files, which impedes Code Ocean intra-capsule performance.
    # It's hard to say which approach is necessarily better.
    "ARCHIVE_WITH_SHARD_GROUPING": True,
 
    # This parameter instructs some sequential pairs of capsules to send/receive data
    # by uploading the data to a cloud location (GCP or S3) and download it from there,
    # instead of letting Code Ocean pass the data directly.
    # Note that this setting probably prefers that ARCHIVE_WITH_SHARD_GROUPING be True.
    # "PASS_DATA_BETWEEN_CAPSULES_OUTSIDE_CODE_OCEAN": False,
    "PASS_DATA_BETWEEN_CAPSULES_METHOD": "aws",  # "internal", "gcp", "aws"

    # Assign this to some arbitary random number to force Code Ocean to reprocess the file instead of using a previously cached run
    "FORCE_NO_CACHE": 345434,
}

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("GENERATE ID INDEX CONFIG")

    data_loc = "../data/"
    results_loc = "../results/"

    config = read_config(["relation", "spatial", "id"])

    # Enact any overrides
    if 'pipeline_id_config' in config['DATA_CONFIG']:
        for k, v in config['DATA_CONFIG']['pipeline_id_config'].items():
            if k == 'docstring':
                continue
            logging.info(f"Overriding id pipeline config {k} = {id_config[k]} with {v}")
            id_config[k] = v
    logging.info("")

    if id_config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] != "internal" and not id_config['ARCHIVE_WITH_SHARD_GROUPING']:
        raise ValueError("PASS_DATA_BETWEEN_CAPSULES_METHOD!=internal requires ARCHIVE_WITH_SHARD_GROUPING. Otherwise large amounts of GCP/AWS egress can occur, which is inefficient at best and can be very expensive at worst.")

    # if id_config['ID_SHARDING_HASH'] != "identity" or id_config['ID_SHARDING_HASH'] == "murmurhash3_x86_128":
    #     raise ValueError("The ID index doesn't support Murmur hash. The resulting index won't work in Neuroglancer. This only applies to the ID index. Murmur hash works for the relation and spatial indices. I have no idea what's wrong at the current time.")
    if id_config['ID_PRESHIFT_BITS'] != 0:
        raise ValueError("Indexing doesn't support sharding specs with preshift-bits other than 0. The resulting index won't work in Neuroglancer. I have no idea what's wrong at the current time.")

    sharding_spec = generate_sharding_spec(
        config['DATA_CONFIG']['data_size'][2],
        config['DATA_CONFIG']['data_size'][1],
        id_config['ID_SHARDING_HASH'],
        MINISHARD_TARGET_COUNT=config["MINISHARD_TARGET_COUNT"],
        SHARD_TARGET_SIZE=config["SHARD_TARGET_SIZE"],
    )
    logging.critical(f"Generated sharding_spec:\n{json.dumps(sharding_spec, indent=2)}")
    logging.critical(f"\nGenerated sharding spec will produce 2^{sharding_spec['shard_bits']} = {2**sharding_spec['shard_bits']} shard files")

    if "parquet" in id_config['ARCHIVE_FORMAT']:
        id_config['PARQUET_ENGINE'] = id_config['ARCHIVE_FORMAT'].split('_')[1]

    # DEBUG
    # I developed this pipeline against hard-coded sharding specs of 0 preshift,
    # 4 sharding, and 3 minisharding, which would yield 16 shards and a relatively
    # good spread of data across the shards. The upgrade to a more sophisticated
    # sharding spec, ala Jeremy's code in the sharding_spec_calculations.py script,
    # has resulted in a much larger number of shards (upwards of 1024), but producing
    # a spec that assigns all data to a single shard. So for the time being,
    # I'm commenting this out, leaving the development defaults place.
    debug = False
    if not debug:
        id_config['ID_PRESHIFT_BITS'] = sharding_spec["preshift_bits"]
        id_config['ID_SHARDING_BITS'] = sharding_spec["shard_bits"]
        id_config['ID_MINISHARDING_BITS'] = sharding_spec["minishard_bits"]
    else:
        logging.critical(f"DEBUG! Dynamically generated sharding spec is currently overridden. Hard-coded values:\n  Preshift:     {id_config['ID_PRESHIFT_BITS']}\n  Sharding:     {id_config['ID_SHARDING_BITS']}\n  Minisharding: {id_config['ID_MINISHARDING_BITS']}")

    with open(f"{results_loc}job_id_config.py", 'w') as f:
        # f.write("{\n")
        # for k, v in id_config.items():
        #     f.write(f'\t"{k}": {v},\n')
        # f.write("}\n")
        f.write(pprint.pformat(id_config, indent=2) + '\n')
        # f.write(json.dumps(id_config, indent=2) + "\n")  # Can't be read by ast.literal(), only json.load()/loads()

logging.info("\nDone")
process_running_time()
