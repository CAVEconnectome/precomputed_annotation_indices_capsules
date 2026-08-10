import sys
import logging
from timeit import default_timer
import json
import pprint

from shared.util import *
from shared.sharding_spec_calculations import *

spatial_config = {
    # "LOGGING_LEVEL": "info",
    # "LOGGING_FORMAT": "%(message)s",
    # "LOGGING_FORMAT": "%(levelname)s: %(message)s",

    "PRECOMPUTED_FILE_WRITER_LOGGING_LEVEL": "info",

    "PROFILING_ENABLED": False,

    "SPATIAL_INDEX_ENABLED": True,
    "SPATIAL_INDEX_UNSHARDED_ENABLED": False,  # The capsule that performs this has not been maintained. It will require development to turn this back on.

    "SPATIAL_SHARDING_HASH": "murmurhash3_x86_128",  #  "murmurhash3_x86_128", "identity"
    "DEFAULT_SPATIAL_PRESHIFT_BITS": 0,
    "DEFAULT_SPATIAL_SHARDING_BITS": None,  # 3,  # Set to None to autogenerate the sharding spec
    "DEFAULT_SPATIAL_MINISHARDING_BITS": 0,

    # Similar to NUM_ROOT_ID_WORKERS but for a different pipeline.
    # Setting this to the max value of 8 effectively disables its utility as a conglomerator of tree outputs into a smaller number of workers, since each subtree at every level will to a new Code Ocean capsule.
    # Setting this to a value 2 <= OCT_TREE_FAN_OUT_DEGREE <= 7 will collapse the 8X fan out of tree children into OCT_TREE_FAN_OUT_DEGREE worker capsules in the next link of the daisy-chain.
    # Setting this to the min value of 1 confines all process of a given split to a non-branching sequence of capsules.
    # "OCT_TREE_FAN_OUT_DEGREE": 1,

    # Set the maximum number of annotations permitted per tree cell id.
    # If this value is exceeded during the tree-building process, an exception will be thrown.
    "MAX_DATA_ROWS_PER_TREE_CELL": 10000,

    # The maximum possible number of tree levels. If the tree-level loop exceeds this value without self-exiting,
    # something has gone wrong somewhere in the process and an exception will be thrown.
    # This is calculated from a starting spatial bounds and a physiological limit on annotation density.
    # For example, synapses are assumed to have an upperbound on their density of .5 per cubic micron (no more than one synapse will fit within two cubic microns).
    # The initial bounds are repeatedly halved on all three axes, with the maximum annotation count computed at each level.
    # Once the maximum count falls below the prescribed density bounds, the loop exits,
    # with the iteration-count indicating the maximum conceivable number of tree levels, then stored in this parameter.
    # For the MiCRONS synapse dataset, the outer bounds is [52708, 64831, 14838], [427816, 311022, 27868] voxels with units of [4, 4, 40] nm.
    # Halving those bounds until we reach a density of .5 takes 10 steps, so the value assigned here is 10.
    "MAX_NUM_TREE_LEVELS": 10,

    # "DIMENSIONS": {"x": [4, "nm"], "y": [4, "nm"], "z": [40, "nm"]},

    # The volume bounds below were calculated as min/max from the 337 million row table/file.
    # Note that ctr column produced both the min and max bounds, obviating pre and post columns for this calculation.
    # The original 20GB CSV export from the DB represents the relevant columns (pre_pt_position, ctr_pt_position, post_pt_position) in voxels.
    # That they are voxels is easy to see because the Z axis values is much smaller than X and Y, because the DIMENSIONS (see above) are 4/4/40.
    # To convert the data from voxels to nm, multiply by the DIMENSIONS, above.
    # To convert from nm to microns (to apply sensible bounds on expected synapses/µm^3 (.5syn/µm^3 as offered by Forrest) or on how deep to expect the oct tree to subdivide (7 or 8)), divide 1000.
    # "VOLUME_BOUNDS": [[52708, 64831, 14838], [427816, 311022, 27868]],

    "PARQUET_ENGINE": "fastparquet",  # auto, fastparquet, pyarrow

    "ARCHIVE_OUTPUT": True,
    # ARCHIVE_MEMORY_STORE_VIA_CUSTOM_METHOD: Instead of using tarfile to create tar objects either in memory or on disk
    # (which can be very slow), use a custom archival that consists of concatenting CSVs files with a tiny metadata block
    # between files to enable splitting (dearchiving) them later. This is MUCH faster than tarfile.
    "ARCHIVE_MEMORY_STORE_VIA_CUSTOM_METHOD": True,
    "ARCHIVE_REGROUPED_OUTPUT": True,
    "ARCHIVE_COMBINED_OUTPUT": True,
    "COMPRESS_ARCHIVE": False,

    # Grouping the outputs by shard results in more individual files, which appears to harm overall performance in the
    # "build spatial index oct tree" capsule. However, this behavior enables the next capsule, "regroup spatial oct tree worker outputs",
    # to be selective about which tar files to decompress. Decompressing all the data in the regrouping capsule is a huge bottleneck.
    # Therefore, the detriment of increasing the file size by maintaining shard separation is compensated by the benefit of saving the
    # regrouping capsule that extraneous work.
    "ARCHIVE_COMPLETED_TREECELLS_WITH_SHARD_GROUPING": True,

    # This parameter instructs some sequential pairs of capsules to send/receive data
    # by uploading the data to a cloud location (GCP or S3) and download it from there,
    # instead of letting Code Ocean pass the data directly.
    # Note that this setting probably prefers that ARCHIVE_WITH_SHARD_GROUPING be True.
    # "PASS_DATA_BETWEEN_CAPSULES_OUTSIDE_CODE_OCEAN": False,
    "PASS_DATA_BETWEEN_CAPSULES_METHOD": "aws",  # "internal", "gcp", "aws"

    # After combining the results of processing the split trees into a global tree,
    # the results can then be grouped by shard or by treelevel-and-shard for final outputting.
    # Grouping by tree level as well as shard will increase the parallelism of the next step.
    "COMBINE_BY_TREELEVEL": True,

    # In addition to generating the binary precomputed shard files required by NG (the primary output of this capsule and pipeline),
    # the option is saving the same output as human-readable CSV is supported, at obvious costs of storage and some additional processing time.
    "SAVE_CSV": False,

    "HIGHEST_SPLIT_ID": None,
}

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("GENERATE SPATIAL INDEX CONFIG")

    data_loc = "../data/"
    results_loc = "../results/"

    # Make sure this subpipeline's config is loaded last so it can override any other config values
    config = read_config(["id", "relation", "spatial"])

    # Enact any overrides
    if 'pipeline_spatial_config' in config['DATA_CONFIG']:
        for k, v in config['DATA_CONFIG']['pipeline_spatial_config'].items():
            if k == 'docstring':
                continue
            logging.info(f"Overriding relation pipeline config {k} = {spatial_config[k]} with {v}")
            spatial_config[k] = v
    logging.info("")

    sharding_spec = generate_sharding_spec(
        config['DATA_CONFIG']['data_size'][2],
        config['DATA_CONFIG']['data_size'][1],
        spatial_config['SPATIAL_SHARDING_HASH'],
        MINISHARD_TARGET_COUNT=config["MINISHARD_TARGET_COUNT"],
        SHARD_TARGET_SIZE=config["SHARD_TARGET_SIZE"],
    )
    logging.critical(f"Generated sharding_spec:\n{json.dumps(sharding_spec, indent=2)}")
    logging.critical(f"\nGenerated sharding spec will produce 2^{sharding_spec['shard_bits']} = {2**sharding_spec['shard_bits']} shard files")

    # DEBUG
    # spatial_config['DEFAULT_SPATIAL_SHARDING_BITS'] == None indicates to autogenerate the sharding parameters. Otherwise, use the defaults.
    if spatial_config['DEFAULT_SPATIAL_SHARDING_BITS'] is None:
        spatial_config['TREE_LEVEL_SHARDING_SPECS'] = [None] * spatial_config['MAX_NUM_TREE_LEVELS']
        for tree_level in range(spatial_config['MAX_NUM_TREE_LEVELS']):
            logging.critical("\n" + "_" * 100)

            # tree_level_sharding_spec = generate_spatial_index_sharding_spec(
            #     tree_level,
            #     spatial_config['MAX_DATA_ROWS_PER_TREE_CELL'],
            #     config['DATA_CONFIG']['data_size'],
            # )

            tree_level_sharding_spec = generate_spatial_index_sharding_spec_2(
                tree_level,
                spatial_config['MAX_DATA_ROWS_PER_TREE_CELL'],
                config['DATA_CONFIG']['data_size'],
                spatial_config['SPATIAL_SHARDING_HASH'],
                MINISHARD_TARGET_COUNT=config["MINISHARD_TARGET_COUNT"],
                SHARD_TARGET_SIZE=config["SHARD_TARGET_SIZE"],
            )

            spatial_config['TREE_LEVEL_SHARDING_SPECS'][tree_level] = tree_level_sharding_spec

            # if spatial_config['TREE_LEVEL_SHARDING_SPECS'][tree_level]['preshift_bits'] != 0:
            #     raise ValueError("Indexing doesn't support sharding specs with preshift-bits other than 0. The resulting index won't work in Neuroglancer. I have no idea what's wrong at the current time.")
    else:
        logging.critical(f"DEBUG! Dynamically generated sharding spec is currently overridden. Hard-coded values:\n  Preshift:     {spatial_config['DEFAULT_SPATIAL_PRESHIFT_BITS']}\n  Sharding:     {spatial_config['DEFAULT_SPATIAL_SHARDING_BITS']}\n  Minisharding: {spatial_config['DEFAULT_SPATIAL_MINISHARDING_BITS']}")

        spatial_config['TREE_LEVEL_SHARDING_SPECS'] = [None] * spatial_config['MAX_NUM_TREE_LEVELS']
        for tree_level in range(spatial_config['MAX_NUM_TREE_LEVELS']):
            spatial_config['TREE_LEVEL_SHARDING_SPECS'][tree_level] = {
                "preshift_bits":  spatial_config["DEFAULT_SPATIAL_PRESHIFT_BITS"],
                "shard_bits":     spatial_config["DEFAULT_SPATIAL_SHARDING_BITS"],
                "minishard_bits": spatial_config["DEFAULT_SPATIAL_MINISHARDING_BITS"],
            }

        # Spread the small deveopment dataset out over the tree levels for development
        spatial_config['MAX_DATA_ROWS_PER_TREE_CELL'] = 10000

    spatial_config_no_debug = {}
    for k, v in spatial_config.items():
        if not k.startswith("DEBUG"):
            spatial_config_no_debug[k] = v
    spatial_config = spatial_config_no_debug

    with open(f"{results_loc}job_spatial_config.py", 'w') as f:
        # f.write("{\n")
        # for k, v in spatial_config.items():
        #     f.write(f'\t"{k}": {v},\n')
        # f.write("}\n")
        f.write(pprint.pformat(spatial_config, indent=2) + '\n')
        # f.write(json.dumps(spatial_config, indent=2) + "\n")  # Can't be read by ast.literal(), only json.load()/loads()

logging.info("\nDone")
process_running_time()
