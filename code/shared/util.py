import sys
import logging
import os
import psutil
import glob
import datetime
import argparse
from timeit import default_timer
import random
import ast

timeit_start_timestamp = default_timer()

def get_logging_level_from_desc(desc):
    if desc == "debug":
        return logging.DEBUG
    if desc == "info":
        return logging.INFO
    if desc == "warning":
        return logging.WARNING
    if desc == "error":
        return logging.ERROR
    if desc == "critical":
        return logging.CRITICAL
    return None

def get_datetime_now_timestamp():
    return datetime.datetime.now(datetime.timezone.utc)
datetime_now_start_timestamp = get_datetime_now_timestamp()

def seconds_to_s(seconds, clean=False):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return '%05.2fs' % (secs)
 
def seconds_to_ms(seconds, clean=False):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if clean:
        if mins == 0:
            return '___ %04.1fs' % (secs)
    return '%02dm %04.1fs' % (mins, secs)
 
def seconds_to_hms(seconds, clean=False):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if clean:
        if mins == 0:
            return '___ ___ %04.1fs' % (secs)
        if hours == 0:
            return '___ %02dm %04.1fs' % (mins, secs)
    return '%02dh %02dm %02.0fs' % (hours, mins, secs)

def process_datetime_now_running_time():
    datetime_now_end_timestamp = get_datetime_now_timestamp()
    elapsed_time = datetime_now_end_timestamp - datetime_now_start_timestamp
    elapsed_secs = elapsed_time.total_seconds()
    logging.critical(f"start_time, end_time, elapsed_time:    {datetime_now_start_timestamp}    {datetime_now_end_timestamp}    {seconds_to_hms(elapsed_secs)}")

def process_timeit_running_time():
    timeit_end_timestamp = default_timer()
    elapsed_secs = timeit_end_timestamp - timeit_start_timestamp
    logging.critical(f"start_time, end_time, elapsed_time:    {timeit_start_timestamp:.0f}    {timeit_end_timestamp:.0f}    {seconds_to_hms(elapsed_secs)}")

def process_running_time():
    return process_datetime_now_running_time()

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        raise ValueError(f"Invalid Hex RGB string: {hex_str}")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def hex_to_rgba(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 8:
        raise ValueError(f"Invalid Hex RGB string: {hex_str}")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4, 6))

def parse_args(extra_args=[]):
    # create a parser object
    parser = argparse.ArgumentParser()
    
    # add the corresponding parameters
    # parser.add_argument('--data_config_filename', dest='data_config_filename', default='data_config.json')
    parser.add_argument('--capsule', dest='capsule')
    parser.add_argument('--data_source_name', dest='data_source_name')
    for extra_arg in extra_args:
        parser.add_argument(f'--{extra_arg}', dest=extra_arg)
    
    # return the data in the object and save in args
    args = parser.parse_args()

    logging.critical(f"Args:\n{args}\n")

    return args

def read_config(sub_configs=[]):
    data_loc = "../data/"
    
    # Ordinarily, this would be hard-coded to 'job_config.py', but there is a bug in CO when I switch filenames back and forth for testing. It is easier to search for all the filenames and take the first one found.
    config = {}
    for config_file_path in [f"{data_loc}job_config.py", f"{data_loc}job_config_spine.py", f"{data_loc}job_config_synapse.py", f"{data_loc}job_config_axon.py", f"{data_loc}job_config_merscope.py"]:
        if os.path.exists(config_file_path):
            logging.critical(f"Found pipeline config file at {config_file_path}")
            with open(config_file_path) as f:
                config = f.read()
            config = ast.literal_eval(config)
        if config:
            break
    
    for index in sub_configs:
        index_specific_config_file_path = f"{data_loc}job_{index}_config.py"
        if os.path.exists(index_specific_config_file_path):
            logging.info(f"Found index-specific config: {index}")  # This is called before basicConfig!
            with open(index_specific_config_file_path) as f:
                index_specific_config = f.read()
            index_specific_config = ast.literal_eval(index_specific_config)
            for k, v in index_specific_config.items():
                config[k] = v

    logging.basicConfig(stream=sys.stdout, level=get_logging_level_from_desc(config['LOGGING_LEVEL']), format=config['LOGGING_FORMAT'], force=True)
    
    logging.debug("Config:")
    max_len = 0
    for key, val in config.items():
        logging.debug(f"  {key:50}: {val}")
    logging.debug("\n")

    return config

def analyze_memory_usage():
    virtual_memory = psutil.virtual_memory()
    logging.info(f"\nTotal Memory:        {virtual_memory.total / (1024**3):.2f} GB")
    logging.info(f"Available Memory:    {virtual_memory.available / (1024**3):.2f} GB") # Recommended value
    logging.info(f"Used Memory:         {virtual_memory.used / (1024**3):.2f} GB")
    logging.info(f"Free Memory:         {virtual_memory.free / (1024**3):.2f} GB")       # Actual free (unused)
    logging.info(f"Memory Percent Used: {virtual_memory.percent}%")

    pid = os.getpid()
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    logging.info(f"\nProcess memory rss:    {memory_info.rss / (1024**3):.2f} GB")
    logging.info(f"Process memory vms:    {memory_info.vms / (1024**3):.2f} GB")
    logging.info(f"Process memory shared: {memory_info.shared / (1024**3):.2f} GB")
    logging.info(f"Process memory text:   {memory_info.text / (1024**3):.2f} GB")
    logging.info(f"Process memory data:   {memory_info.data / (1024**3):.2f} GB")

    logging.info("\n")

def list_directory(dir_path, limit=5):
    contents = sorted(list(glob.glob(dir_path)))
    if contents:
        logging.info(f"{dir_path} contents ({len(contents)}):")
        for fi, file in enumerate(contents):
            if fi >= limit:
                logging.info("  [...more]")
                break
            file_desc = None
            if os.path.isfile(file):
                file_size_bytes = os.path.getsize(file)
                if file_size_bytes >= 1000000000:
                    file_desc = f"  {file_size_bytes/1000000000:>4.1f}G {file}"
                elif file_size_bytes >= 1000000:
                    file_desc = f"  {file_size_bytes/1000000:>4.1f}M {file}"
                elif file_size_bytes >= 1000:
                    file_desc = f"  {file_size_bytes/1000:>4.1f}K {file}"
                else:
                    file_desc = f"  {file_size_bytes:>4}B {file}"
            elif os.path.isdir(file):
                file_desc = f"        {file}/"
            else:
                file_desc = f"        {file}"
            logging.info(f"{file_desc}")
    else:
        logging.info(f"{dir_path} is empty")

def finalize_results(results_loc):
    logging.info("\nResult contents:")

    list_directory(f"{results_loc}*")
    list_directory(f"{results_loc}*/*")
    list_directory(f"{results_loc}*/*/*")

    # results_loc_contents = sorted(list(glob.glob(f"{results_loc}*")))
    # if results_loc_contents:
    #     logging.info(f"  results_loc contents ({len(results_loc_contents)}):")
    #     for file in results_loc_contents:
    #         file_desc = f"  {os.path.getsize(file)/1000000:>4.1f}M {file}" if os.path.isfile(file) \
    #             else f"        {file}/" if os.path.isdir(file) \
    #             else f"        {file}"
    #         logging.info(f"{file_desc}")
    # else:
    #     logging.info(f"  {results_loc} is empty")
    
    # results_loc_subcontents = sorted(list(glob.glob(f"{results_loc}*/*")))
    # if results_loc_subcontents:
    #     logging.info(f"  results_loc subcontents ({len(results_loc_subcontents)}):")
    #     for file in results_loc_subcontents:
    #         file_desc = f"  {os.path.getsize(file)/1000000:>4.1f}M {file}" if os.path.isfile(file) \
    #             else f"        {file}/" if os.path.isdir(file) \
    #             else f"        {file}"
    #         logging.info(f"{file_desc}")
    
    # results_loc_subcontents = sorted(list(glob.glob(f"{results_loc}*/*/*")))
    # if results_loc_subcontents:
    #     logging.info(f"\n  results_loc subsubcontents ({len(results_loc_subcontents)}):")
    #     for file in results_loc_subcontents:
    #         file_desc = f"  {os.path.getsize(file)/1000000:>4.1f}M {file}" if os.path.isfile(file) \
    #             else f"        {file}/" if os.path.isdir(file) \
    #             else f"        {file}"
    #         logging.info(f"{file_desc}")
    
    # If any CodeOcean capsule fails to produce an output file, that capsule and the entire pipeline fail, so we have to generate a pointless placeholder file to keep things alive. WTF?!
    # Furthermore, add a random signifier to the filename to avoid any trouble from CodeOcean name collisions between capsules.
    if not glob.glob(f"{results_loc}*"):
        with open(f"{results_loc}placeholder_{hex(int(random.random()*1000000000000))[2:]}.txt", 'w') as f:
            f.write("Placeholder")
