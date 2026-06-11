import sys
import logging
import os
import glob
from timeit import default_timer
import pandas as pd
import pprint
import pyarrow.parquet as pq
import pyarrow as pa
import shutil

from shared.util import *

def split_csv_file(input_file_path, num_splits, split_id, row_start, row_end):
    logging.info("Splitting CSV file\n\n")
    
    input_file = os.path.basename(input_subpath)  # input_subpath.split('/')[-1]
    logging.info(f"input_file: {input_file}")

    row_last = row_end - 1
    input_split_file = f"{input_file[:input_file.rindex('.')]}__split-{split_id:03}@{num_splits}__rows-{row_start:011,}-{row_last:011,}.csv"
    input_split_file_path = f"{results_loc}{input_split_file}"
    logging.info(f"Results input_split_file_path: {input_split_file_path}")
    logging.info("\n")

    columns = config['DATA_CONFIG']['columns']
    id_column = config['DATA_CONFIG']['id_column']
    logging.info(f"id_column: {id_column}")
    if id_column is None:
        logging.info(f"id_column is NULL, so it will be inferred from the split id and row idx, and inserted into the corresponding id column: {columns[0]}.")

    with open(input_file_path) as fin:
        with open(input_split_file_path, 'w') as fout:
            logging.info("Input and output files opened. Skipping rows up to start...")

            # To efficiently skip a large number of lines of a large file,
            # don't call readline() over and over. Just enumerate the file object (or call next() N times.
            if row_start > 0:
                logging.info("Skipping lines of file up to starting location of the split")
                s = ""
                for i, line in enumerate(fin):
                    if i % 1000000 == 0:
                        if i > 0:
                            # logging.info("#", end="")
                            s += "#"
                    elif i % 100000 == 0:
                        # logging.info("|", end="")
                        s += "|"
                    elif i % 10000 == 0:
                        # logging.info(".", end="")
                        s += "."
                    if i >= row_start - 1:
                        break
                logging.info(f"{s}\n")
            
            # Read and write out the rows for this split
            logging.info("Reading and writing split rows...")
            num_lines_written = 0
            for i in range(row_end - row_start):
                line = fin.readline()
                if not fin or not line:
                    logging.info("EOF")
                    break
                if i <= 1:
                    logging.info(f"Beginning split line {i}: {line.strip()}")
                elif i >= row_end - row_start - 2:
                    logging.info(f"Ending split line {i}: {line.strip()}")
                
                if i == 0:
                    # Detect whether the split already includes the header.
                    # Only add the header if it isn't already present.
                    header_present = line.startswith(columns[0 if id_column is not None else 1])
                    if not header_present:
                        # Header is not present, so add it
                        logging.info("Header not present. Writing it first...")
                        # If we added an id column to the columns list, we can't include it in this header because we are still processing the original file here and the added id column doesn't exist yet
                        if id_column is not None:
                            fout.write(','.join(config["DATA_CONFIG"]["columns"]) + "\n")
                        else:
                            fout.write(','.join(config["DATA_CONFIG"]["columns"][1:]) + "\n")
                    else:
                        logging.info("Header is present. There is no need to write it at the top of the split.")

                fout.write(line)
                num_lines_written += 1
            
            logging.info(f"\nSplitting of assigned row range complete with num lines: {num_lines_written}")

            assert num_lines_written > 0
        
        logging.info(f"Split file size: {os.path.getsize(input_split_file_path)/1000000}M")
    
    # If we are producing the last split, update the filename to reflect to true final row, not just the offset calculated end row from the start row
    if row_start + num_lines_written < row_end:
        logging.info("\n")
        logging.info("row_start + num_lines_written < row_end. Presumably this is the final split. Renaming file to reflect true row count.")
        row_last = row_start + num_lines_written - 1
        input_split_file2 = f"{input_file[:input_file.rindex('.')]}__split-{split_id:03}@{num_splits}__rows-{row_start:011,}-{row_last:011,}.csv"
        input_split_file2_path = f"{results_loc}{input_split_file2}"
        logging.info(f"Results input_split_file_path: {input_split_file2_path}")
        os.rename(input_split_file_path, input_split_file2_path)

def split_parquet_file(input_file_path, num_splits, split_id, row_start, row_end):
    logging.info("Splitting parquet file\n\n")

    input_file = os.path.basename(input_subpath)  # input_subpath.split('/')[-1]
    logging.info(f"input_file: {input_file}")

    row_last = row_end - 1
    input_split_file = f"{input_file[:input_file.rindex('.')]}__split-{split_id:03}@{num_splits}__rows-{row_start:011,}-{row_last:011,}.parquet"
    input_split_file_path = f"{results_loc}{input_split_file}"
    logging.info(f"Results input_split_file_path: {input_split_file_path}")
    logging.info("\n")

    BATCH_SIZE = 50 # Process in manageable chunks

    # Open the Parquet file
    parquet_file = pq.ParquetFile(input_file_path)
    schema = parquet_file.schema.to_arrow_schema()

    # Keep track of the current total row index
    current_row_index = 0
    last_row_written = 0
    rows_to_write = []

    # Iterate through batches
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        batch_start = current_row_index
        batch_end = current_row_index + len(batch)
        
        # Check if this batch overlaps with the desired range
        if batch_end > row_start and batch_start < row_end:
            # Calculate the relevant slice within the current batch
            slice_start = max(0, row_start - batch_start)
            slice_end = min(len(batch), row_end - batch_start)
            last_row_written = current_row_index + slice_end
            
            # Slice the batch and add to list of tables to write
            rows_to_write.append(batch.slice(offset=slice_start, length=slice_end - slice_start))
            
        # Stop iterating if we have passed the end row
        if batch_end >= row_end:
            break
            
        current_row_index = batch_end
    
    num_lines_written = last_row_written - row_start
    logging.info(f"Final row index:  {last_row_written}")
    logging.info(f"Num rows written: {num_lines_written}")

    assert num_lines_written > 0

    # Concatenate the relevant batches into a single Table
    if rows_to_write:
        table_to_write = pa.Table.from_batches(rows_to_write, schema=schema)
        # Write the resulting Table to a new Parquet file
        pq.write_table(table_to_write, input_split_file_path)
        logging.info(f"Successfully wrote rows {row_start} to {row_end} to {input_split_file_path}")
    else:
        logging.info("No rows found in the specified range.")
    
    # If we are producing the last split, update the filename to reflect to true final row, not just the offset calculated end row from the start row
    if row_start + num_lines_written < row_end:
        logging.info("\n")
        logging.info("row_start + num_lines_written < row_end. Presumably this is the final split. Renaming file to reflect true row count.")
        row_last = row_start + num_lines_written - 1
        input_split_file2 = f"{input_file[:input_file.rindex('.')]}__split-{split_id:03}@{num_splits}__rows-{row_start:011,}-{row_last:011,}.parquet"
        input_split_file2_path = f"{results_loc}{input_split_file2}"
        logging.info(f"Results input_split_file_path: {input_split_file2_path}")
        os.rename(input_split_file_path, input_split_file2_path)

def split_swc_files(input_file_paths, num_splits, split_id, row_start, row_end):
    raise RuntimeError("Until further notice, this code leading into this function (split_swc_files()) should have landed in split_and_rowify_swc_files() instead. Getting here was an error.")
    
    input_file_paths = input_file_paths[row_start:row_end]
    assert input_file_paths and len(input_file_paths) > 0
    for input_file_path in input_file_paths:
        input_split_file_name = os.path.basename(input_file_path)
        subdir = input_file_path.split('/')[-2]
        subdir += f"__split-{split_id:03}@{num_splits}"
        os.makedirs(f"{results_loc}{subdir}", exist_ok=True)
        input_split_file2_path = f"{results_loc}{subdir}/{input_split_file_name}"
        logging.info(f"Moving {input_file_path} to {input_split_file2_path}")
        shutil.copy(input_file_path, input_split_file2_path)

def split_and_rowify_swc_files(input_file_paths, num_splits, split_id, row_start, row_end):
    """
    Take a set of SWC files where each file indicates a single polyline annotation and each row indicates a polyline point.
    Split the set of files for horizontal processing, but combine the files of a single split into a single CSV file
    such that an annotation is on a single row and all its points are in a single semicolon-delimited field.
    """
    x_col = config['DATA_CONFIG']['spatial_pt_columns']['point']['x']
    y_col = config['DATA_CONFIG']['spatial_pt_columns']['point']['y']
    z_col = config['DATA_CONFIG']['spatial_pt_columns']['point']['z']
    x_colidx, y_colidx, z_colidx = None, None, None
    
    input_file_paths = input_file_paths[row_start:row_end]
    header = None
    combined_rows = []
    for input_file_path in input_file_paths:
        filename = os.path.basename(input_file_path)
        id_ = filename[:filename.rindex('.')]
        if not header:
            with open(input_file_path) as f:
                header = f.readline()
                header = header.split()[1]  # Skip the leading pound sign
                header = header.split(',')
                x_colidx = header.index(x_col)
                y_colidx = header.index(y_col)
                z_colidx = header.index(z_col)
        df = pd.read_csv(input_file_path, sep=' ', skiprows=1, names=["id", "type", "x", "y", "z", "r", "pid"])
        pt_fields = []
        for row_idx, row in enumerate(df.itertuples(index=False)):
            x = str(row[x_colidx])
            y = str(row[y_colidx])
            z = str(row[z_colidx])
            pt = [x, y, z]
            pt_field = ','.join(pt)
            pt_fields.append(pt_field)
        pt_fields_field = ';'.join(pt_fields)
        row_out = [id_, pt_fields_field]
        combined_rows.append(row_out)
    assert combined_rows  # len(combined_rows) > 0
    header_out = ["ID", "Points"]
    combined_df = pd.DataFrame(combined_rows, columns=header_out)
    combined_df.to_csv(f"{results_loc}rows__split-{split_id}@{num_splits}.csv", index=False, header=False)

    config['DATA_CONFIG']['columns'] = header_out
    del config['DATA_CONFIG']['id_src']
    config['DATA_CONFIG']['id_column'] = "ID"
    config['DATA_CONFIG']['structure'] = "one_annotation_per_row__multiple_points_per_row_in_one_field"
    config['DATA_CONFIG']['spatial_pt_columns'] = "single_field_list"

    return config

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("GENERATE INPUT SPLIT")

    data_loc = "../data/"
    results_loc = "../results/"

    config = read_config()
    logging.basicConfig(stream=sys.stdout, level=get_logging_level_from_desc(config['LOGGING_LEVEL']), format=config['LOGGING_FORMAT'], force=True)

    logging.info(f"{data_loc} contents:")
    logging.info('  ' + '\n  '.join(sorted(os.listdir(data_loc))).strip() + '\n')

    # Search for an input file of various accepted types (file extensions) either in the top-level directory or one-deep in any subdirectory.
    # Prioritize CSV over Parquet.
    # Prioritize top-level files over one-deep files.
    # If more than one file is found at a given priority, fail. Otherwise, ignore lower priority input files and proceed.
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
            input_subpaths = sorted(list(glob.glob(f"{data_loc}*.{file_format}")))
            input_subpaths = [v for v in input_subpaths if not v.startswith(f"{data_loc}disabled")]
            if input_subpaths:
                logging.info(f"Found top-level input subpaths: {input_subpaths}")
                file_ext = file_format
            if not input_subpaths:
                # Search one directory down for an input file
                input_subpaths = sorted(list(glob.glob(f"{data_loc}*/*.{file_format}")))
                input_subpaths = [v for v in input_subpaths if not v.startswith(f"{data_loc}disabled")]
                if input_subpaths:
                    logging.info(f"Found one-deep input subpaths:\n  {'\n  '.join(input_subpaths)}")
                    file_ext = file_format
            if input_subpaths:
                break
    else:
        input_subpaths = list(glob.glob(f"{data_loc}*/{debug_input_filename}"))[0]
        file_ext = input_subpaths[input_subpaths.rindex('.')+1:]

    if not input_subpaths:
        raise RuntimeError("No valid input files found. Accepted formats: CSV, Parquet, SWC.\nNOTE! Make sure the Connection Type leading from the dataset to this split-generation capsule was reassigned from 'Default' to 'Collect'!")
    
    if file_ext in ["csv", "parquet"]:
        assert len(input_subpaths) == 1
    
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
        input_file_path_pcs = input_file_path.split('/')
        input_file_dir = input_file_path_pcs[-2]
        logging.info(f"input_file_dir: {input_file_dir}")
        input_file_name = os.path.basename(input_file_path)
        assert input_file_name == input_file_path_pcs[-1]
        logging.info(f"input_file_name: {input_file_name}")

    # This verification only applies if the input is connected directly to this capsule, as was the case prior to 20260603. With the addition of the decompression capsule between the input and this (the splitter), this verification can no longer be applied since there is no way to guarantee that decompressed directories or files will conform to any naming convention.
    # if 'data_source_name' in config['DATA_CONFIG'] and input_file_dir != config['DATA_CONFIG']['data_source_name']:
        # raise ValueError(f"Input file directory (i.e., the name of the connected Code Ocean Data Asset) doesn't match configured 'data_source_name': {input_file_dir} != {config['DATA_CONFIG']['data_source_name']}\nAre you certain you (1) indicated the correct directory in the 'data_source_name' parameter and (2) reassigned the connection's Connection Type from 'Default' to 'Collect'?")
    
    row_splits_file = "row_splits.txt"  # As configured by the associated pipeline's connection I/O remapping
    row_splits_file_path = f"{data_loc}{row_splits_file}"

    if not os.path.exists(row_splits_file_path):
        raise RuntimeError(f"Expected row splits file not found: {row_splits_file_path}")
    
    with open(row_splits_file_path) as f:
        row_splits = f.read()
    row_splits = row_splits.split('\n')
    num_splits = int(row_splits[0].split()[1])
    split_id = int(row_splits[1].split()[1])
    row_start = int(row_splits[2].split()[1])
    row_end = int(row_splits[3].split()[1])
    logging.info(f"Num splits, Split id, Row start-end: {num_splits}, {split_id}, {row_start}-{row_end}")
    logging.info("\n")

    assert file_ext in accepted_file_formats

    # At the time of this writing, SWC is handled completely differently from CSV and Parquet. The latter two are assumed to consist of a single file, with one annotation per row and all data for an annotation in various columns. SWC is assumed to consist of numerous files, one per annotation, with each data point represented as a row of the file, i.e., SWC is used for polyline annotations while CSV/Parquet are used for point & line annotations.

    if file_ext == "csv":
        split_csv_file(input_subpaths[0], num_splits, split_id, row_start, row_end)
    elif file_ext == "parquet":
        split_parquet_file(input_subpaths[0], num_splits, split_id, row_start, row_end)
    elif file_ext == "swc":
        config = split_and_rowify_swc_files(input_subpaths, num_splits, split_id, row_start, row_end)
    
    # The config might have been modified and needs to be propagated as such to later capsules
    with open(f"{results_loc}job_config.py", 'w') as f:
        f.write(pprint.pformat(config) + '\n')

logging.info("\nDone")
process_running_time()
