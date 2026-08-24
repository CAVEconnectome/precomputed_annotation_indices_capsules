import sys
import logging
import os
import glob
from timeit import default_timer
import shutil
import zipfile
import gzip
import tarfile
import pprint
import math

from shared.util import *

def decompress_input():
    logging.info("\nDecompressing input (or passing the input through if it isn't compressed)\n")

    logging.info(f"Looking for compressed files in {data_loc}{data_path}")

    # Make sure tar.gz occurs before .gz
    accepted_compression_formats = ["zip", "tar.gz", "gz", "tar"]
    file_ext = None
    compressed_input_subpaths = None
    for compression_format in accepted_compression_formats:
        # Search the top directory for an input file
        logging.info(f"Looking for compressed files for format {compression_format}...")
        compressed_input_subpaths = sorted(list(glob.glob(f"{data_loc}{data_path}*.{compression_format}")))
        compressed_input_subpaths = [v for v in compressed_input_subpaths if not v.startswith(f"{data_loc}{data_path}disabled")]
        if compressed_input_subpaths:
            logging.info(f"  Found top-level compressed input subpaths for format {compression_format}: {compressed_input_subpaths}")
            file_ext = compression_format
            break
        if not compressed_input_subpaths:
            # Search one directory down for an input file.
            # This helps when processing from a capsule instead of a pipeline.
            # Searching one level deep for the data predates the 'data_path' parameter.
            # It doesn't really make sense anymore, since 'data_path' ostensibly indicates exactly where to find the data.
            # Consequently, this section should be removed at some point.
            logging.info(f"  No compressed files found for format {compression_format}. Looking one level deeper in {data_loc}{data_path}*/")
            compressed_input_subpaths = sorted(list(glob.glob(f"{data_loc}{data_path}*/*.{compression_format}")))
            compressed_input_subpaths = [v for v in compressed_input_subpaths if not v.startswith(f"{data_loc}{data_path}disabled")]
            if compressed_input_subpaths:
                logging.info(f"    Found one-deep compressed input subpaths for format {compression_format}:\n  {'\n  '.join(compressed_input_subpaths)}")
                file_ext = compression_format
                break
            logging.info(f"    No compressed files found for format {compression_format}.")
    logging.info("")

    if compressed_input_subpaths:
        logging.info("Found compressed files. Proceeding to decompress input files...")
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
        logging.info(f"No compressed input files found. Looking for input directory (presumably containing uncompressed input files) indicated by config 'data_size' subdir:\n  '{config['DATA_CONFIG']['data_size'][0]}'")
        input_subpaths = sorted(list(glob.glob(f"{data_loc}{data_path}{config['DATA_CONFIG']['data_size'][0]}")))
        if not input_subpaths and config['DATA_CONFIG']['data_path']:
            # Desperately attempt to find the input. Look directly in the explicitly configured 'data_path'. WTH is the data?!
            logging.info(f"No directory of uncompressed input files found. Looking for input directory (presumably containing uncompressed input files) indicated by config 'data_path':\n  '{config['DATA_CONFIG']['data_path']}'")
            input_subpaths = sorted(list(glob.glob(f"{data_loc}{data_path}")))
        if input_subpaths:
            logging.info(f"Found top-level input subpaths: {input_subpaths}")
            for input_subpath in input_subpaths:
                if os.path.isfile(input_subpath):
                    raise TypeError(f"Expected input path to be a directory: {input_subpath}")
                    logging.info(f"Copying input file to results: {input_subpath}")
                    shutil.copy(input_subpath, f"{results_loc}{os.path.basename(input_subpath)}")
                elif os.path.isdir(input_subpath):
                    if input_subpath[-1] == '/':
                        input_subpath = input_subpath[:-1]
                    logging.info(f"Copying input directory to results: {input_subpath}")
                    shutil.copytree(input_subpath, f"{results_loc}{os.path.basename(input_subpath)}")
                else:
                    raise TypeError(f"Unknown file type (not a file or a directory): {input_subpath}")
        else:
            raise RuntimeError(f"No input directory found for indicated dataset: {config['DATA_CONFIG']['data_size'][0]}. Make sure the Connection Type leading from the dataset to this split-generation capsule was reassigned from 'Default' to 'Collect'!")
        
        # logging.info(f"No compressed input files found. All subdirectories under {data_loc}{data_path} will be copied to results instead.")
        # input_subpaths = sorted(list(glob.glob(f"{data_loc}{data_path}*")))
        # if input_subpaths:
        #     logging.info(f"Found top-level input subpaths (possibly input data subpaths): {input_subpaths}")
        #     for input_subpath in input_subpaths:
        #         if os.path.isdir(input_subpath):
        #             logging.info(f"Copying input directory to results: {input_subpath}")
        #             shutil.copytree(input_subpath, f"{results_loc}{os.path.basename(input_subpath)}")

def find_input_subpaths(input_dir):
    # This function is repeated in the "decompress input, find volume bounds" capsule.
    # It feels too specialized for the overly broad util.py, but too idiosyncratic to create a separate file for inclusiong by just these two capsules. In effect, these two capsules are tightly coupled,
    # but for reasons of horizontal behavior, they can't be collapsed into a single capsule.
    # Some sort of refactorization should be considered in the future.

    # input_dir distinguishes between the data/ and results/ locations.
    # In the decompression capsule, since files were decompressed into results/ (or moved there if not compressed),
    # that capsule needs to look for the input in results/.
    # However, in the splitting capsule, the files will be found in the capsule's input location (data/),
    # so that capsule needs to look for the input in data/.

    # Search for an input file of various accepted types (file extensions) either in the top-level directory or one-deep in any subdirectory.
    # Prioritize CSV over Parquet.
    # Prioritize top-level files over one-deep files.
    # If more than one file is found at a given priority, fail. Otherwise, ignore lower priority input files and proceed.

    logging.info("\nLooking for input subpaths\n")

    input_subpath = None
    file_ext = None
    accepted_file_formats = ["csv", "parquet", "swc"]
    
    debug_input_filename = None
    # debug_input_filename = "synapses_pni_2_v1_filtered_view__1-root-ids_337312429-row-limit_2108-rows.csv"
    # debug_input_filename = "synapses_pni_2_v1_filtered_view__1-root-ids_337312429-row-limit_2108-rows.parquet"
    # debug_input_filename = "transcripts.csv"
    
    if not debug_input_filename:
        # Iterate over accepted file formats (and their associated file extensions)
        for file_format in accepted_file_formats:
            # Search the top directory for an input file
            input_subpaths = sorted(list(glob.glob(f"{input_dir}*.{file_format}")))
            input_subpaths = [v for v in input_subpaths if not v.startswith(f"{input_dir}disabled")]
            if input_subpaths:
                logging.info(f"  Found top-level input subpaths: {input_subpaths}")
                file_ext = file_format
            if not input_subpaths:
                # Search one directory down for an input file
                input_subpaths = sorted(list(glob.glob(f"{input_dir}*/*.{file_format}")))
                input_subpaths = [v for v in input_subpaths if not v.startswith(f"{input_dir}disabled")]
                if input_subpaths:
                    logging.info(f"  Found one-deep input subpaths:\n  {'\n  '.join(input_subpaths)}")
                    file_ext = file_format
            if input_subpaths:
                break
    else:
        input_subpaths = list(glob.glob(f"{input_dir}*/{debug_input_filename}"))[0]
        file_ext = input_subpaths[input_subpaths.rindex('.')+1:]

    assert file_ext in accepted_file_formats

    if not input_subpaths:
        raise RuntimeError("No valid input files found. Accepted formats: CSV, Parquet, SWC.\nNOTE! Make sure the Connection Type leading from the dataset to this split-generation capsule was reassigned from 'Default' to 'Collect'!")
    
    if file_ext in ["csv", "parquet"]:
        assert len(input_subpaths) == 1
    
    # This block is just data validation. It doesn't "do" anything.
    for input_subpath in input_subpaths:
        if "-rows" in input_subpath:
            logging.info("File name includes rows indication. Confirming rows against config...")
            a = input_subpath.find("-rows")
            b = input_subpath.rfind('_', 0, a)
            rows = int(input_subpath[b+1:input_subpath.find('-', b)])
            logging.info(f"Does configured data size match ({config['DATA_CONFIG']['data_size'][2]}) match test filename indicated num rows ({rows})?")
            if config['DATA_CONFIG']['data_size'][2] != rows:
                raise ValueError(f"Configured data size != discovered test dataset row size indicated in test filename: {config['DATA_CONFIG']['data_size'][2]} != {rows}")
    
    for input_file_path in input_subpaths:
        # The config file specifies not the name of the input file, but the name of the directory above it, which is the name of the Code Ocean Data Asset. These must match.
        logging.info(f"Parsing input file path: {input_file_path}")
        input_file_path_pcs = input_file_path.split('/')
        input_file_dir = input_file_path_pcs[-2]
        logging.info(f"  input_file_dir: {input_file_dir}")
        input_file_name = os.path.basename(input_file_path)
        assert input_file_name == input_file_path_pcs[-1]
        logging.info(f"  input_file_name: {input_file_name}")
    
    # This verification only applies if the input is connected directly to this capsule, as was the case prior to 20260603. With the addition of the decompression capsule between the input and this (the splitter), this verification can no longer be applied since there is no way to guarantee that decompressed directories or files will conform to any naming convention.
    # if 'data_source_name' in config['DATA_CONFIG'] and input_file_dir != config['DATA_CONFIG']['data_source_name']:
        # raise ValueError(f"Input file directory (i.e., the name of the connected Code Ocean Data Asset) doesn't match configured 'data_source_name': {input_file_dir} != {config['DATA_CONFIG']['data_source_name']}\nAre you certain you (1) indicated the correct directory in the 'data_source_name' parameter and (2) reassigned the connection's Connection Type from 'Default' to 'Collect'?")
    
    return file_ext, input_subpaths

def find_volume_bounds(input_path):
    # Find the volume bounds of each XYZ axis of the total data (before it gets split)

    logging.info("\nFinding volume bounds")

    volume_bounds = config['DATA_CONFIG']['volume_bounds']
    if volume_bounds:
        logging.info(f"\nVolume bounds is provided by the data config. There is no need to calculate it:\n{volume_bounds}")
        return
    
    logging.info("\nVolume bounds not provided by data config. Proceeding to calculate it...\n")

    columns = config['DATA_CONFIG']['columns']
    col_indices = {column: index for index, column in enumerate(columns)}
    logging.info(f"Column indices: {col_indices}")

    spatial_pt_columns = config['DATA_CONFIG']['spatial_pt_columns']
    x_col_indices, y_col_indices, z_col_indices = [], [], []
    for spatial_pt_column_lbl, spatial_pt_column_desc in spatial_pt_columns.items():
        x_col_indices.append(col_indices[spatial_pt_column_desc['x']])
        y_col_indices.append(col_indices[spatial_pt_column_desc['y']])
        z_col_indices.append(col_indices[spatial_pt_column_desc['z']])
    
    logging.info(f"x_col_indices: {x_col_indices}")
    logging.info(f"y_col_indices: {y_col_indices}")
    logging.info(f"z_col_indices: {z_col_indices}")
    
    x_min, y_min, z_min = math.inf, math.inf, math.inf
    x_max, y_max, z_max = -math.inf, -math.inf, -math.inf

    data_size_num_rows = config['DATA_CONFIG']['data_size'][2]
    
    t0 = default_timer()
    logging.info("Progress:")
    with open(input_path) as f:
        line_idx = 0
        while True:  # line_idx < 100000000:
            if line_idx % 1000000 == 0:
                s = f"{line_idx:,}L"

                if line_idx > 0:
                    t1 = default_timer()
                    et = t1 - t0
                    rows_per_sec = line_idx / et
                    total_data_estimate = data_size_num_rows / rows_per_sec
                    remaining_time = (data_size_num_rows - line_idx) / rows_per_sec
                    s += f";{rows_per_sec:,.0f}r/s"
                    s += f";{total_data_estimate:,.0f}Tt"
                    s += f";{remaining_time:,.0f}Rt"
                logging.info(s)
            # elif line_idx % 1000000 == 0:
            #     logging.info(f"|")
            # elif line_idx % 100000 == 0:
            #     logging.info(f".")
            
            try:
                line = f.readline().strip()
                if not line:
                    logging.info("EOF")
                    break
                cells = line.split(',')

                header_present = False
                if line_idx == 0:
                    header_present = cells[0] == columns[0] or cells[1] == columns[1]  # Sometimes pandas leaves the first column in the header row
                if not header_present:
                    # X
                    for x_col_idx in x_col_indices:
                        v = float(cells[x_col_idx])
                        if v < x_min:
                            x_min = v
                        if v > x_max:
                            x_max = v
        
                    # Y
                    for y_col_idx in y_col_indices:
                        v = float(cells[y_col_idx])
                        if v < y_min:
                            y_min = v
                        if v > y_max:
                            y_max = v
        
                    # Z
                    for z_col_idx in z_col_indices:
                        v = float(cells[z_col_idx])
                        if v < z_min:
                            z_min = v
                        if v > z_max:
                            z_max = v
            
            except Exception as e:
                logging.info(f"Exception at line {line_idx}: {e}")
                logging.info(f"Line: {line}")
                break
            
            line_idx += 1

    t1 = default_timer()
    et = t1 - t0
    logging.info(f"Total elapsed time to find volume bounds: {et:,.1f}s")

    logging.info(f"\n\nFinal line count: {line_idx}")

    logging.info("\n\nFinal volume bounds in format required by pipeline config json:")
    logging.info("(You can copy this and paste it directly into the config file or the App Panel's 'config_override' parameter to avoid recalculating the volume bounds in the future)")
    logging.info('    "volume_bounds": [')
    logging.info(f"        [{x_min}, {y_min}, {z_min}],")
    logging.info(f"        [{x_max}, {y_max}, {z_max}]")
    logging.info("    ]")

    volume_bounds = [
        [x_min, y_min, z_min],
        [x_max, y_max, z_max],
    ]
    config['DATA_CONFIG']['volume_bounds'] = volume_bounds

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
    
    data_loc_contents = sorted(glob.glob(f'{data_loc}*/*'))
    data_loc_contents = [v for v in data_loc_contents if "/disabled" not in v][:30]
    logging.info(f"{data_loc} subcontents ({len(data_loc_contents)}) (first 30 shown):\n  {'\n  '.join(data_loc_contents)}\n")

    data_path = config['DATA_CONFIG']['data_path']
    if data_path and data_path[-1] != '/':
        data_path += '/'
    data_loc_contents = sorted(glob.glob(f'{data_loc}{data_path}*'))
    data_loc_contents = [v for v in data_loc_contents if "/disabled" not in v][:30]
    logging.info(f"{data_loc}{data_path} contents ({len(data_loc_contents)}) (first 30 shown):\n  {'\n  '.join(data_loc_contents)}\n")

    # Searching one level deep for the data predates the 'data_path' parameter.
    # It doesn't really make sense anymore, since 'data_path' ostensibly indicates exactly where to find the data.
    # Consequently, this section should be removed at some point.
    # Note that these lines are just for logging. The actual data retrieval occurs farther below.
    data_loc_subcontents = sorted(glob.glob(f'{data_loc}{data_path}*/*'))
    data_loc_subcontents = [v for v in data_loc_subcontents if "/disabled" not in v][:30]
    logging.info(f"{data_loc}{data_path} subcontents ({len(data_loc_subcontents)}) (first 30 shown):\n  {'\n  '.join(data_loc_subcontents)}\n")

    decompress_input()

    file_ext, input_subpaths = find_input_subpaths(results_loc)

    find_volume_bounds(input_subpaths[0])

    # Since the volume bounds in the config might have been updated,
    # write the config to the results to propagate into the pipeline.
    with open(f"{results_loc}job_config.py", 'w') as f:
        f.write(pprint.pformat(config) + '\n')

    finalize_results(results_loc)

    logging.info("\nDone")
    process_running_time()
