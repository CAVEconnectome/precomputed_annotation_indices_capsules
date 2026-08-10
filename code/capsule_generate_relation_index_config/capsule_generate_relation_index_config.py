import sys
import logging
from timeit import default_timer
import json
import pprint

from shared.util import *
from shared.sharding_spec_calculations import *

relation_config = {
    # "LOGGING_LEVEL": "info",
    # "LOGGING_FORMAT": "%(message)s",
    # "LOGGING_FORMAT": "%(levelname)s: %(message)s",

    "PRECOMPUTED_FILE_WRITER_LOGGING_LEVEL": "info",

    "RELATION_INDEX_ENABLED": True,

    "RELATION_SHARDING": True,

    "RELATION_SHARDING_HASH": "murmurhash3_x86_128",  #  "murmurhash3_x86_128", "identity"
    "RELATION_PRESHIFT_BITS": 0,
    "RELATION_SHARDING_BITS": None,  # 1,  # Set to None to autogenerate the sharding spec
    "RELATION_MINISHARDING_BITS": 0,

    # Relationship index column names
    # "RELATIONSHIPS": {"Presynaptic Cell": "pre_pt_root_id", "Postsynaptic Cell": "post_pt_root_id"},

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
}

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("GENERATE RELATION INDEX CONFIG")

    data_loc = "../data/"
    results_loc = "../results/"

    # Make sure this subpipeline's config is loaded last so it can override any other config values
    config = read_config(["id", "spatial", "relation"])

    # Enact any overrides
    if 'pipeline_relation_config' in config['DATA_CONFIG']:
        for k, v in config['DATA_CONFIG']['pipeline_relation_config'].items():
            if k == 'docstring':
                continue
            logging.info(f"Overriding relation pipeline config {k} = {relation_config[k]} with {v}")
            relation_config[k] = v
    logging.info("")

    if relation_config['PASS_DATA_BETWEEN_CAPSULES_METHOD'] != "internal" and not relation_config['ARCHIVE_WITH_SHARD_GROUPING']:
        raise ValueError("PASS_DATA_BETWEEN_CAPSULES_METHOD!=internal requires ARCHIVE_WITH_SHARD_GROUPING. Otherwise large amounts of GCP/AWS egress can occur, which is inefficient at best and can be very expensive at worst.")

    if relation_config['RELATION_PRESHIFT_BITS'] != 0:
        raise ValueError("Indexing doesn't support sharding specs with preshift-bits other than 0. The resulting index won't work in Neuroglancer. I have no idea what's wrong at the current time.")

    sharding_spec = generate_sharding_spec(
        config['DATA_CONFIG']['data_size'][2],
        config['DATA_CONFIG']['data_size'][1],
        relation_config['RELATION_SHARDING_HASH'],
        MINISHARD_TARGET_COUNT=config["MINISHARD_TARGET_COUNT"],
        SHARD_TARGET_SIZE=config["SHARD_TARGET_SIZE"],
    )
    logging.critical(f"Generated sharding_spec:\n{json.dumps(sharding_spec, indent=2)}")
    logging.critical(f"\nGenerated sharding spec will produce 2^{sharding_spec['shard_bits']} = {2**sharding_spec['shard_bits']} shard files")

    if "parquet" in relation_config['ARCHIVE_FORMAT']:
        relation_config['PARQUET_ENGINE'] = relation_config['ARCHIVE_FORMAT'].split('_')[1]

    # DEBUG
    # id_config['RELATION_SHARDING_BITS'] == None indicates to autogenerate the sharding parameters. Otherwise, use the defaults.
    if relation_config['RELATION_SHARDING_BITS'] is None:
        relation_config['RELATION_PRESHIFT_BITS'] = sharding_spec["preshift_bits"]
        relation_config['RELATION_SHARDING_BITS'] = sharding_spec["shard_bits"]
        relation_config['RELATION_MINISHARDING_BITS'] = sharding_spec["minishard_bits"]
    else:
        logging.critical(f"DEBUG! Dynamically generated sharding spec is currently overridden. Hard-coded values:\n  Preshift:     {relation_config['RELATION_PRESHIFT_BITS']}\n  Sharding:     {relation_config['RELATION_SHARDING_BITS']}\n  Minisharding: {relation_config['RELATION_MINISHARDING_BITS']}")

    with open(f"{results_loc}job_relation_config.py", 'w') as f:
        # f.write("{\n")
        # for k, v in relation_config.items():
        #     f.write(f'\t"{k}": {v},\n')
        # f.write("}\n")
        f.write(pprint.pformat(relation_config, indent=2) + '\n')
        # f.write(json.dumps(relation_config, indent=2))  # Can't be read by ast.literal(), only json.load()/loads()

logging.info("\nDone")
process_running_time()
