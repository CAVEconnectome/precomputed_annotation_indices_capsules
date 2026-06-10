import sys
import logging
import os
from timeit import default_timer
import math

from shared.util import *

def generate_one_data_source_split_range(split_size, num_splits):
    for split_idx in range(num_splits):
        split_start = split_idx * split_size
        split_end = split_start + split_size  # This value will likely overrun the end of the data for the final split. That is handled by the split-generation capsule.
        if config['LIMIT'] is not None and split_start >= config['LIMIT']:
            logging.info(f"Limit exceeded: {split_start} >= {config['LIMIT']}. Aborting early...")
            break
        filepath = f"{results_loc}input_splits/table_split_{split_idx+1:07}.txt"
        file_content = f"""
num_splits:\t{num_splits}
one_based_split_id:\t{split_idx+1}
split_start:\t{split_start}
split_end:\t{split_end}
""".strip() + '\n'
        logging.info(f"Output destination: {filepath}")
        logging.info(f"Output content:\n{file_content}")
        with open(filepath, 'w') as f:
            f.write(file_content)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("GENERATE INPUT SPLIT RANGES")

    args = parse_args()

    data_loc = f"../../data/{args.capsule}/"
    results_loc = "../../results/"

    config = read_config()
    logging.basicConfig(stream=sys.stdout, level=get_logging_level_from_desc(config['LOGGING_LEVEL']), format=config['LOGGING_FORMAT'], force=True)

    logging.info(f"{data_loc} contents:")
    if os.path.exists(data_loc):
        logging.info('  ' + '\n  '.join(sorted(os.listdir(data_loc))).strip() + '\n')

    os.makedirs(f"{results_loc}input_splits/", exist_ok=True)
    os.makedirs(f"{results_loc}conglomeration_worker_allocations/", exist_ok=True)
    
    

    
    split_size = config['DATA_CONFIG']['data_size'][3]
    num_splits = config['DATA_CONFIG']['data_size'][4]
    generate_one_data_source_split_range(split_size, num_splits)
    

    

    # DEPRECATED
    # if config['SPLIT_DESC'] > 0:
    #     # Splits are configured by split size
    #     split_size = config['SPLIT_DESC']
    #     num_splits = config['DATA_SIZE'][1] // split_size + 1
    #     logging.info(f"For the configured DATA_SIZE of {config['DATA_SIZE'][1]} and configured split_size of {config['SPLIT_DESC']}, num_splits has been calculated to be {num_splits}.")
    # else:
    #     # Splits are configured by number of splits
    #     num_splits = -config['SPLIT_DESC']
    #     split_size = math.ceil(config['DATA_SIZE'][1] / num_splits)
    #     logging.info(f"For the configured DATA_SIZE of {config['DATA_SIZE'][1]} and configured num_splits of {-config['SPLIT_DESC']}, split_size has been calculated to be {split_size}.")
    # logging.info("\n")
    
    # if num_splits > 200:
    #     logging.info("Num splits calculated to be >200. Are you sure this is correct? Aborting for now. You can override by modifying the code if needed...")
    #     sys.exit(1)
    
    # generate_one_data_source_split_range(split_size, num_splits)

logging.info("\nDone")
process_running_time()
