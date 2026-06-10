import sys
import os
import logging
from timeit import default_timer
import math
import json
import pprint

from shared.util import *

config = {
    "LOGGING_LEVEL": "debug",  # debug, info, warning, error, critical
    "LOGGING_FORMAT": "%(message)s",
    # "LOGGING_FORMAT": "%(levelname)s: %(message)s",

    # This has been moved to an App Builder runtime parameter
    # "DATA_SOURCE_NAME": "synapses_pni_2_v1_filtered_view__test1",
    # "DATA_SOURCE_NAME": "synapses_pni_2_v1_filtered_view__test2",
    # "DATA_SOURCE_NAME": "synapses_pni_2_v1_filtered_view__test3",
    # "DATA_SOURCE_NAME": "synapses_pni_2_v1_filtered_view__test4",
    # "DATA_SOURCE_NAME": "synapses_pni_2_v1_filtered_view__test5",
    # "DATA_SOURCE_NAME": "synapses_pni_2_v1_filtered_view__test6",
    # "DATA_SOURCE_NAME": "synapses_pni_2_v1_filtered_view",
    
    # Describe how the input data is split, i.e., the number of lines/rows chunked together during splitting.
    # The resulting row-assignment indicates how many lines/rows will be processed by each split worker.
    # A positive value prescribes a split size in number of rows.
    # A negative value indicates the number of splits to generate (with an obvious sign flip).
    "SPLIT_DESC": 2000000 * 3,
    # "SPLIT_DESC": 100000,
    # "SPLIT_DESC": -10,  # Instead of prescribing a split size, prescribe a number of splits (with a sign flip)

    # The subsplit informs the workers to process their data split in pieces smaller than the entire split, so as to stay within memory limitations. Each split will be divided into subsplits exactly the same way the original data was divided into splits, processes iteratively (serially) by a given worker, until all the data has been processed. The same "pneumonic" as before is applied: positive values indicate number of data items per subsplit while negative values indicate number of subsplits to divide a split into.
    # "SUBSPLIT_DESC": 2000000,
    # "SUBSPLIT_DESC": 2500000,
    "SUBSPLIT_DESC": -3,

    # Set this to None to generate all splits and process the entire input file.
    # Otherwise, for development/debugging, use this to process a subset of the data, starting from the begining of the file.
    # This limit is indicated in the same units as the data size and split size, i.e., probably lines or rows.
    # Namely, this limit is *not* on the number of generated splits or split-config files or split workers, but rather is a limit on the number of data items processed).
    "LIMIT": None,

    "NUM_SHARD_WORKERS": 32,  # If the number of shards exceeds this value, multiple shards will be processed by a single shard worker
    "MINISHARD_TARGET_COUNT": 1000,  # Default 1000
    "SHARD_TARGET_SIZE": 50000000,  # Default 50000000
 
    # Assign this to some arbitary random number to force Code Ocean to reprocess the file instead of using a previously cached run
    "FORCE_NO_CACHE": 345434,

    # If there are billing or other concerns with data transfer outside of Code Ocean,
    # you have the option of not uploading the results to a Google bucket.
    # This will have the effect that Neuroglancer can't use the results of the associated run.
    # You would have to find some other way to move the results to a bucket for NG at a later time.
    # That could be some unrelated capsule or pipeline that is dedicated to that task,
    # although it would require first saving the results of the run to a Code Ocean data resource.
    # In the pipeline runs history on the right edge of the pipeline view,
    # the three-dot menu for any given run offers an option to save th results in this manner.
    # The three-dot menu isn't the top-level menu next to the run's date and running time.
    # It is the secondary menu next to the Run data folder.
    # The three dots only appear when you hover over the total size, which then transforms into a three-dot menu option.
    "UPLOAD_RESULTS_TO_GCP": True,

    "GCP_BUCKET": "keith-dev",
    "GCP_SCRATCH_BLOB_PATH": "ng_precomputed_annotations_pipeline_scratch",
    "GCP_RESULTS_BLOB_PATH": "ng_precomputed_annotations_unreleased",

    "AWS_BUCKET": "pga-connectomics-wg-802451596237-us-west-2",
    "AWS_PROJECT_PATH": "from_workgroups/ng_precomputed_annotations_pipeline_scratch",
}

def add_timestamp_and_uri_to_config(config):
    if not os.path.exists(f"{data_loc}DEBUG_FLAG.txt"):
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%Z")
    else:
        # All single-capsule runs use a hard-coded timestamp. All pipeline runs use a dynamically generate timestamp (even if they are development runs as opposed to production runs).
        timestamp = "9999-01-01_01-01-01_UTC"
    config['TIMESTAMP'] = timestamp
    config['NEUROGLANCER_URI'] = f"gs://{config['GCP_BUCKET']}/{config['GCP_RESULTS_BLOB_PATH']}/{timestamp}/"
    return config

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("GENERATE CONFIG")

    logging.basicConfig(stream=sys.stdout, level=get_logging_level_from_desc(config['LOGGING_LEVEL']), format=config['LOGGING_FORMAT'], force=True)

    analyze_memory_usage()

    args = parse_args()

    data_loc = f"../../data/{args.capsule}/"
    results_loc = "../../results/"

    logging.info(f"{data_loc} contents:")
    logging.info('  ' + '\n  '.join(sorted(os.listdir(data_loc))).strip() + '\n')

    config = add_timestamp_and_uri_to_config(config)
    
    # Read the description of the data for the run from a separate JSON file.
    # This description differs conceptually from configuring how the pipeline operates, which is indicated in the dictionary provided at the top of this source file.
    # Note that the DATA_SIZE configuration parameter has been left in the dictionary instead of the JSON file
    # because it can be modified more quickly and frequently (for development) from the code (in this file) than from the JSON file (which would require altering the Code Ocean data source, which is incredibly tedious).
    # This has been moved to an App Builder runtime parameter
    
    # data_config_filename = "data_config.json"
    # data_config_filename = "spine_info_sample_bigger__data-config.json"
    # data_config_filename = "synapses_pni_2_v1_filtered_view__v1412__data-config.json"
    # data_config_filename = "synapses_pni_2_v1_filtered_view__v1412_w_fake_enums_and_multirelations___data-config.json"
    # data_config_filename = "synapses_v1718_with_target_predictions_v2_limit100__data-config.json"
    # data_config_filename = "wan-qing_polyline_data_config.json"

    # if not os.path.exists(f"{data_loc}{data_config_filename}"):
    #     data_config_filename = args.data_config_filename
    
    # if os.path.exists(f"{data_loc}{data_config_filename}"):
    #     with open(f"{data_loc}{data_config_filename}") as f:
    #         data_config = json.load(f)
    #     config['DATA_CONFIG'] = data_config
    # else:
    #     raise FileNotFoundError(f"File not found: {data_loc}{data_config_filename}")

    # Read the data config file
    json_files = list(glob.glob(f"{data_loc}*.json"))
    assert len(json_files) == 1
    data_config_file = json_files[0]
    logging.info(f"Found config json file: {data_config_file}")
    with open(data_config_file) as f:
        data_config = json.load(f)
    config['DATA_CONFIG'] = data_config
    logging.info("")

    # Enact any overrides
    if 'pipeline_config' in config['DATA_CONFIG']:
        for k, v in config['DATA_CONFIG']['pipeline_config'].items():
            if k == 'docstring':
                continue
            logging.info(f"Overriding default pipeline config {k} = {config[k]} with {v}")
            config[k] = v
    logging.info("")

    # In order to ensure there is no way to accidentally upload results to GCP, utterly break the GCP bucket designation
    if not config['UPLOAD_RESULTS_TO_GCP']:
        config['GCP_BUCKET'] = f"UPLOAD_RESULTS_TO_GCP_IS_FALSE__BUCKET_IS_DISABLED__REMOVE_THIS_TEXT_TO_GENERATE_USABLE_BUCKET__{config['GCP_BUCKET']}"

    # Add an index column if needed
    if 'id_column' in config['DATA_CONFIG'] and config['DATA_CONFIG']['id_column'] is None:
        id_column_name = "id"
        id_column_name_w_suffix = id_column_name
        suffix = 0
        while id_column_name_w_suffix in config['DATA_CONFIG']['columns']:
            suffix += 1
            id_column_name_w_suffix = f"{id_column_name}_{suffix}"
        # config['DATA_CONFIG']['id_column_added'] = id_column_name_w_suffix
        config['DATA_CONFIG']['columns'] = [id_column_name_w_suffix] + config['DATA_CONFIG']['columns']
        logging.info(f"Added ID column since one wasn't configured: {id_column_name_w_suffix}\n")
    
    # Calculate the split size for the indicated data source using the configured SPLIT_DESC add the corresponding 'data_size' key.
    if args.data_source_name.lower() not in ["", " ", "na", "none", "unspecified"]:
        for data_source_name, one_data_size in config['DATA_CONFIG']['data_sizes'].items():
            if data_source_name == "docstring":
                continue
            if data_source_name == args.data_source_name:  # config['DATA_SOURCE_NAME']:
                if config['SPLIT_DESC'] > 0:
                    # Splits are configured by split size
                    split_size = config['SPLIT_DESC']
                    num_splits = math.ceil(one_data_size[1] / split_size)
                    logging.info(f"For the configured data size of {one_data_size[1]:,} and configured split_size of {config['SPLIT_DESC']:,}, num_splits has been calculated to be {num_splits}.")
                else:
                    # Splits are configured by number of splits
                    num_splits = -config['SPLIT_DESC']
                    split_size = math.ceil(one_data_size[1] / num_splits)
                    logging.info(f"For the configured data size of {one_data_size[1]:,} and configured num_splits of {-config['SPLIT_DESC']}, split_size has been calculated to be {split_size:,}.")
                config['DATA_CONFIG']['data_size'] = [data_source_name] + one_data_size + [split_size, num_splits]
                break
        if 'data_size' not in config['DATA_CONFIG']:
            raise ValueError("'data_size' not added to DATA_CONFIG. Are you sure the correct 'data_sizes' key was passed in?")
    
        # Add the data_source_name arg to the config so the splitters can confirm that it matches the input file
        config['DATA_CONFIG']['data_source_name'] = args.data_source_name
    else:
        logging.info(f"Note that no data_source_name argument was provided. The largest one in the config's 'data_sizes' will be used.")
        largest_dataset = [None, 0, 0]
        for data_source_name, one_data_size in config['DATA_CONFIG']['data_sizes'].items():
            if data_source_name == "docstring":
                continue
            if one_data_size[0] > largest_dataset[1]:
                largest_dataset = [data_source_name, one_data_size[0], one_data_size[1]]
        assert largest_dataset[0]
        if config['SPLIT_DESC'] > 0:
            # Splits are configured by split size
            split_size = config['SPLIT_DESC']
            num_splits = math.ceil(largest_dataset[2] / split_size)
            logging.info(f"For the configured data size of {largest_dataset[2]:,} and configured split_size of {config['SPLIT_DESC']:,}, num_splits has been calculated to be {num_splits}.")
        else:
            # Splits are configured by number of splits
            num_splits = -config['SPLIT_DESC']
            split_size = math.ceil(largest_dataset[2] / num_splits)
            logging.info(f"For the configured data size of {largest_dataset[2]:,} and configured num_splits of {-config['SPLIT_DESC']}, split_size has been calculated to be {split_size:,}.")
        config['DATA_CONFIG']['data_size'] = largest_dataset + [split_size, num_splits]
 
    # Process the subsplits
    if config['SUBSPLIT_DESC'] > 0:
        # Subsplits are configured by split size
        subsplit_size = config['SUBSPLIT_DESC']
        num_subsplits = math.ceil(split_size / subsplit_size)
        logging.info(f"For the configured split size of {split_size:,} and configured subsplit_size of {config['SUBSPLIT_DESC']:,}, num_subsplits has been calculated to be {num_subsplits}.")
    else:
        # Subsplits are configured by number of subsplits
        num_subsplits = -config['SUBSPLIT_DESC']
        subsplit_size = math.ceil(split_size / num_subsplits)
        logging.info(f"For the configured split size of {split_size:,} and configured num_subsplits of {-config['SUBSPLIT_DESC']}, subsplit_size has been calculated to be {subsplit_size:,}.")
    config['DATA_CONFIG']['data_size'].extend([subsplit_size, num_subsplits])

    # Clean up property names. They are subject to fairly strict restrictions since they must be legal javascript variable names.
    properties_clean = {}
    for prop_name, prop_info in config['DATA_CONFIG']['properties'].items():
        prop_name_clean = prop_name.lower().replace(' ', '_')
        properties_clean[prop_name_clean] = prop_info
    config['DATA_CONFIG']['properties'] = properties_clean

    # Clean up property enums
    PROPERTY_ENUMABLE_TYPES = set(["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32"])
    for prop_name, prop_info in config['DATA_CONFIG']['properties'].items():
        # logging.info(f"AAA {prop_name} {prop_info}")
        if 'enum_values' in prop_info:
            if 'enum_labels' not in prop_info:
                raise ValueError("Property description includes 'enum_values' without accompanying 'enum_labels'.")
            if prop_info['type'] not in PROPERTY_ENUMABLE_TYPES:
                raise ValueError("Property includes enum description with non-enumable type.")
            # Add an enum for missing data
            for i in range(-1, -1000, -1):
                if i not in prop_info['enum_values']:
                    prop_info['enum_values'].append(i)
                    prop_info['enum_labels'].append("NULL")
                    break
        elif 'enum_labels' in prop_info:
            raise ValueError("Property description includes 'enum_labels' without accompanying 'enum_values'.")
        else:
            # Create null values for the enum fields so code later in the pipeline doesn't have to branch on or otherwise be concerned with their presence/absence.
            prop_info['enum_values'] = None
            prop_info['enum_labels'] = None
    # logging.info(f"Properties: {config['DATA_CONFIG']['properties']}")
    
    # logging.info("\nConfig:")
    # for key, val in config.items():
    #     if key != 'DATA_CONFIG':
    #         logging.info(f"  {key}: {val}")
    #     else:
    #         logging.info(f"  {key}: {json.dumps(config['DATA_CONFIG'], indent=2)}")
    # logging.info("\n")
    
    with open(f"{results_loc}job_config.py", 'w') as f:
        # f.write("{\n")
        # for k, v in config.items():
        #     f.write(f'\t"{k}": {v},\n')
        # f.write("}\n")
        f.write(pprint.pformat(config) + '\n')#.replace('{', "{\n ").replace('}', "\n}"))
        # f.write(json.dumps(config, indent=2) + "\n")  # Can't be read by ast.literal(), only json.load()/loads()
    
    analyze_memory_usage()

logging.info("\nDone")
process_running_time()
