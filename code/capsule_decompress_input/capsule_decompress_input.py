import sys
import logging
import os
import glob
from timeit import default_timer
import shutil
import zipfile
import gzip
import tarfile

from shared.util import *

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("DECOMPRESS INPUT")

    data_loc = "../data/"
    results_loc = "../results/"

    config = read_config()
    logging.basicConfig(stream=sys.stdout, level=get_logging_level_from_desc(config['LOGGING_LEVEL']), format=config['LOGGING_FORMAT'], force=True)
    
    data_loc_contents = sorted(glob.glob(f'{data_loc}*'))
    data_loc_contents = [v for v in data_loc_contents if "/disabled" not in v][:30]
    logging.info(f"{data_loc} contents ({len(data_loc_contents)}) (first 30 shown):\n  {'\n  '.join(data_loc_contents)}\n")
    data_loc_subcontents = sorted(glob.glob(f'{data_loc}*/*'))
    data_loc_subcontents = [v for v in data_loc_subcontents if "/disabled" not in v][:30]
    logging.info(f"{data_loc} subcontents ({len(data_loc_subcontents)}) (first 30 shown):\n  {'\n  '.join(data_loc_subcontents)}\n")
    
    # Make sure tar.gz occurs before .gz
    accepted_file_formats = ["zip", "tar.gz", "gz", "tar"]
    file_ext = None
    compressed_input_subpaths = None
    for file_format in accepted_file_formats:
        # Search the top directory for an input file
        compressed_input_subpaths = sorted(list(glob.glob(f"{data_loc}*.{file_format}")))
        compressed_input_subpaths = [v for v in compressed_input_subpaths if not v.startswith(f"{data_loc}disabled")]
        if compressed_input_subpaths:
            logging.info(f"Found top-level compressed input subpaths: {compressed_input_subpaths}")
            file_ext = file_format
            break
        if not compressed_input_subpaths:
            # Search one directory down for an input file
            compressed_input_subpaths = sorted(list(glob.glob(f"{data_loc}*/*.{file_format}")))
            compressed_input_subpaths = [v for v in compressed_input_subpaths if not v.startswith(f"{data_loc}disabled")]
            if compressed_input_subpaths:
                logging.info(f"Found one-deep compressed input subpaths:\n  {'\n  '.join(compressed_input_subpaths)}")
                file_ext = file_format
                break

    if compressed_input_subpaths:
        logging.info("Decompressing input files")
        for compressed_input_subpath in compressed_input_subpaths:
            logging.info(f"  Compressed input: {compressed_input_subpath}")
            logging.info(f"  File extension: {file_ext}")
            if file_ext == "zip":
                with zipfile.ZipFile(compressed_input_subpath, 'r') as zip_ref:
                    zip_ref.extractall(results_loc)
            elif file_ext == "tar.gz":
                with tarfile.open(compressed_input_subpath, 'r:gz') as tar:
                    tar.extractall(path=results_loc)
            elif file_ext == "gz":
                compressed_input_subpath_no_ext = compressed_input_subpath[:-len(file_ext)]
                with gzip.open(compressed_input_subpath, 'rb') as f_in:
                    with open(f"{results_loc}{os.path.basename(compressed_input_subpath_no_ext)}", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            elif file_ext == "tar":
                with tarfile.open(compressed_input_subpath, 'r') as tar:
                    tar.extractall(path=results_loc)
    else:
        logging.info(f"No compressed input files found. Looking for uncompressed input files indicated by config: '{config['DATA_CONFIG']['data_size'][0]}'")
        input_subpaths = sorted(list(glob.glob(f"{data_loc}{config['DATA_CONFIG']['data_size'][0]}")))
        if input_subpaths:
            logging.info(f"Found top-level input subpaths: {input_subpaths}")
            for input_subpath in input_subpaths:
                if os.path.isfile(input_subpath):
                    shutil.copy(input_subpath, f"{results_loc}{os.path.basename(input_subpath)}")
                elif os.path.isdir(input_subpath):
                    shutil.copytree(input_subpath, f"{results_loc}{os.path.basename(input_subpath)}")
        else:
            raise RuntimeError("No uncompressed input files found indicated by config.")

logging.info("\nDone")
process_running_time()
