import logging
import os
import glob
from timeit import default_timer
from collections import defaultdict
import fnmatch
import re
import shutil
import ast
import pandas as pd

try:
    import pyarrow.parquet as pq
    # import pyarrow as pa
    PYARROW_INSTALLED = True
    print("pyarrow is installed")
except ImportError:
    PYARROW_INSTALLED = False
    print("pyarrow is not installed")

class RAMDataPond:
    """
    To save I/O time, this class offers the option of saving data in a RAM "pond",
    kind of like a data lake, but not quite as fancy as official data lakes.
    The intended full file path where a file would be saved to disk is insteaad used as a unique key into the pond,
    which then stores string representations of data as it would have otherwise been written to disk.
    The same file path keys can then be used to retrieve the data.

    The constructor also offers the option of disabling the RAM pond,
    in which case the class simply acts as a pass-through to traditional I/O.
    In this way, the calling code can use the same I/O interface (this class) for either scenario
    and the coder can merely toggle RAM pond usage on and off by a simple constructor argument.

    A force_disk option is provided on each API call.
    This can be used to force initial pipeline input files to be read from disk
    or to write final results to disk.

    A fancier version of this class would be built around actual byte strings instead of ASCII strings,
    but I didn't see a need for that nuance at the current time.
    """
    def __init__(self, use_ram_data_pond):
        self.ram_data_pond = {} if use_ram_data_pond else None
    
    def clear(self):
        self.ram_data_pond.clear()

    def glob_disk_or_ram_data_pond(self, pattern, force_disk=False):
        if self.ram_data_pond is None or force_disk:
            return sorted(list(glob.glob(pattern)))
        else:
            # results = [filepath for filepath in self.ram_data_pond if re.match(re.compile(pattern.replace('.', '\\.').replace('*', '.*')), filepath)]
            results = [filepath for filepath in self.ram_data_pond if fnmatch.fnmatch(filepath, pattern)]

            # Only leave as many slashes as are in the pattern, the way glob would perform
            pattern_num_slashes = sum([1 if c == '/' else 0 for c in pattern])
            for i in range(len(results)):
                pcs = results[i].split('/')
                results[i] = '/'.join(pcs[:pattern_num_slashes+1])
            
            return sorted(list(set(results)))
    
    def fastglob_ram_data_pond(self, dir_path, force_disk=False):
        """
        Regex matching with the re module is slow. We can perform simple prefix searches faster by more direct means.
        """
        return self.glob_disk_or_ram_data_pond(dir_path + "*", force_disk)

        # if self.ram_data_pond is None or force_disk:
        #     raise RuntimeError("fastglob_ram_data_pond not supported for disk, only memory")
        # else:
        #     # logging.info("\nfastglob_ram_data_pond()")
        #     # logging.info(f"{dir_path}")
        #     # logging.info("-----")
        #     # logging.info('\n'.join(sorted(self.ram_data_pond.keys())))
        #     # In accordance with the way glob performs, only accept hits that don't indiciate deeper subdirectories, only the current directory
        #     results = set()
        #     for filepath in self.ram_data_pond:
        #         if filepath.startswith(dir_path):
        #             tail = filepath[len(dir_path):]
        #             if '/' in tail:
        #                 tail = tail[:tail.index('/')]
        #             results.add(dir_path + tail)
        #     # logging.info(f"Num hits: {len(results)}")
        #     # logging.info('\n'.join(sorted(list(results))))
        #     # logging.info("~~~~")
        #     return sorted(list(results))
    
    def getsize(self, filepath, force_disk=False):
        if self.ram_data_pond is None or force_disk:
            return os.path.getsize(filepath)
        else:
            assert filepath in self.ram_data_pond
            return len(self.ram_data_pond[filepath])
    
    def get_total_size(self, force_disk=False):
        return sum([self.getsize(filepath, force_disk) for filepath in self.ram_data_pond])

    def write_to_disk_or_ram_data_pond(self, filepath, s, force_disk=False):
        if self.ram_data_pond is None or force_disk:
            with open(filepath, 'w') as f:
                f.write(s)
        else:
            self.ram_data_pond[filepath] = s

    def read_from_disk_or_ram_data_pond(self, filepath, force_disk=False):
        if self.ram_data_pond is None or force_disk:
            with open(filepath, 'r') as f:
                return f.read().strip()
        else:
            assert filepath in self.ram_data_pond
            # As of this writing, RAMDataPond is sometimes used as a key/value store for arbitrary types, e.g. lists of Annotation objects, not just as a RAM disk of file-like string objects. Until this hybrid usage is properly sorted out, it is necessary to avoid certain string-specific operations in certain casees.
            if isinstance(self.ram_data_pond[filepath], str):
                return self.ram_data_pond[filepath].strip()
            return self.ram_data_pond[filepath]
    
    @staticmethod
    def read_nlines_from_disk(filepath, nlines):
        with open(filepath, 'r') as f:
            lines = []
            while len(lines) < nlines:
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            return lines

    def read_nlines_from_disk_or_ram_data_pond(self, filepath, nlines, force_disk=False):
        """
        Only read N lines from a file (useful for grabbing a header row)
        """
        if self.ram_data_pond is None or force_disk:
            if filepath.endswith(".csv"):
                return RAMDataPond.read_nlines_from_disk(filepath, nlines)
            elif filepath.endswith(".parquet"):
                # TODO: Read the Parquet file more efficiently, only reading the necessary number of lines. See the Input Split generation capsule for relative code.
                df = pd.read_parquet(filepath)
                lines = []
                for row_idx, row in enumerate(df.itertuples(index=False)):
                    lines.append(','.join([str(v) for v in list(row)]))
                    if row_idx >= nlines - 1:
                        break
                return lines
        else:
            assert filepath in self.ram_data_pond
            file_contents = self.ram_data_pond[filepath]
            lines = []
            char_idx = 0
            while char_idx < len(file_contents) and len(lines) < nlines:
                line = ""
                while char_idx < len(file_contents):
                    c = file_contents[char_idx]
                    char_idx += 1
                    if c == '\n':
                        break
                    line += c
                lines.append(line)
            return lines

    def read_splitlines_from_disk_or_ram_data_pond(self, filepath, row_start=None, row_end=None, force_disk=False):
        if self.ram_data_pond is None or force_disk:
            if filepath.endswith(".csv"):
                if row_start is None:
                    with open(filepath, 'r') as f:
                        return f.read().splitlines()
                else:
                    # This code was copied and adapted from the 'generate input split' capsule code
                    
                    lines = []
                    with open(filepath) as fin:
                        logging.info(f"Input file opened. Skipping rows up to start={row_start:,} ...")

                        # To efficiently skip a large number of lines of a large file,
                        # don't call readline() over and over. Just enumerate the file object (or call next() N times.
                        if row_start > 0:
                            logging.info("Skipping lines of file up to starting location")
                            s = ""
                            last_line = None
                            for i, line in enumerate(fin):
                                last_line = line
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
                            logging.info(f"Last line read while skipping to start: {last_line.strip()}")
                        
                        # Read and write out the rows
                        row_end_str = f"{row_end:,}" if row_end is not None else 'None'
                        logging.info(f"Reading and writing rows with row_end={row_end_str} ...")
                        num_lines_written = 0
                        i = -1
                        while True:
                            i += 1
                            if row_end is not None and i >= row_end - row_start:
                                break
                            line = fin.readline()
                            if not fin or not line:
                                logging.info("EOF")
                                break
                            lines.append(line.strip())
                            num_lines_written += 1
                    logging.info(f"First line saved: {lines[0].strip() if lines else None}")
                    logging.info(f"Last line saved: {lines[-1].strip() if lines else None}")
                    return lines
            elif filepath.endswith(".parquet"):
                if not PYARROW_INSTALLED:
                    raise RuntimeError("Parquet unsupported because pyarrow not installed")

                if row_start is None:
                    df = pd.read_parquet(filepath)
                    lines = []
                    for row_idx, row in enumerate(df.itertuples(index=False)):
                        lines.append(','.join([str(v) for v in list(row)]))
                    return lines
                else:
                    # This code was copied and adapted from the 'generate input split' capsule code

                    BATCH_SIZE = 50 # Process in manageable chunks

                    # Open the Parquet file
                    parquet_file = pq.ParquetFile(filepath)
                    schema = parquet_file.schema.to_arrow_schema()

                    # Keep track of the current total row index
                    current_row_index = 0
                    last_row_written = 0
                    rows = []

                    # Iterate through batches
                    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
                        batch_start = current_row_index
                        batch_end = current_row_index + len(batch)
                        
                        # Check if this batch overlaps with the desired range
                        if batch_end > row_start and (row_end is None or batch_start < row_end):
                            # Calculate the relevant slice within the current batch
                            slice_start = max(0, row_start - batch_start)
                            slice_end = min(len(batch), row_end - batch_start) if row_end is not None else len(batch)
                            last_row_written = current_row_index + slice_end
                            
                            # Slice the batch and add to list of tables to write
                            batch_slice_list = batch.slice(offset=slice_start, length=slice_end - slice_start).to_pandas().values.tolist()
                            for row in batch_slice_list:
                                line = ','.join([str(v) for v in row])
                                rows.append(line)
                            
                        # Stop iterating if we have passed the end row
                        if row_end is not None and batch_end >= row_end:
                            break
                            
                        current_row_index = batch_end
                    
                    return rows
        else:
            assert filepath in self.ram_data_pond
            return self.ram_data_pond[filepath].strip().split('\n')
    
    def move_file_on_disk_or_ram_data_pond(self, src, dst, force_disk=False):
        if self.ram_data_pond is None or force_disk:
            shutil.move(src, dst)
        else:
            keys = list(self.ram_data_pond.keys())
            for item_src in keys:
                if item_src.startswith(src):
                    item_dst = item_src.replace(src, dst)
                    self.ram_data_pond[item_dst] = self.ram_data_pond[item_src]
                    del self.ram_data_pond[item_src]
    
    def move_files_via_prefix_replacement(self, prefix_pattern, dst):
        """
        NOTE! I only added this function after investigating the profiling results and believing
        # I was seeing a heavy slowdown in move_file_on_disk_or_ram_data_pond().
        # But after further scrutiny, I believe I may have been looking at the wrong line of the profiling metrics.
        # This may have never been a problem in the first place.

        Reproduce the fastglob (i.e. prefix) search.
        For any matches, replace (i.e. rename) the prefix with the destination,
        as if moving the tail to a new directory.
        """
        keys = list(self.ram_data_pond.keys())
        logging.info(f"move_files_via_prefix_replacement(): {len(keys)} items total")
        num_moved = 0
        for item_src in keys:
            if item_src.startswith(prefix_pattern):
                tail = item_src[len(prefix_pattern):]
                item_dst = dst + tail
                self.ram_data_pond[item_dst] = self.ram_data_pond[item_src]
                del self.ram_data_pond[item_src]
                num_moved += 1
        logging.info(f"move_files_via_prefix_replacement(): {num_moved} items moved")
    
    @staticmethod
    def archive_str_data(filepath, file_contents, filepath_root, file_out=None):
        """
        Archive a string to a file oject, either in ram or on disk.
        By preserving the external file object as open outside this function,
        multiple archives can be added to a single archive file.

        The filepath indicates where this file will be dearchived relative to some specified root.
        This enables multiple files to be archived together in a file hierarchy.

        If no file object is provided, the archive will be returned as a string.

        Since this class supports string arrays for pseudo-files not byte arrays,
        archival consists of generating string with a tiny header:
        <file path>
        <end line>
        <file length>
        <end line>
        <file content>
        <end line>
        """
        file_contents = file_contents.strip()
        file_len = len(file_contents)
        if file_len > 999999999:
            raise ValueError("RAMDataPond.archive_str_data() file_len ({file_len}) > 999,999,999!")
        file_len_str = f"{file_len}"
        # file_len_str_len = len(file_len_str)
        # file_len_str_len_str = f"{file_len_str_len}"
        # if len(file_len_str_len_str) > 1:
            # raise ValueError("RAMDataPond.archive_str_data() len(file_len_str_len_str) ({len(file_len_str_len_str)}) > 1!")

        if filepath.startswith(filepath_root):
            filepath = filepath[len(filepath_root):]

        if not file_out:
            archive = ""
            archive += filepath + '\n'
            # archive += file_len_str_len_str + '\n'
            archive += file_len_str + '\n'
            archive += file_contents
            archive += '\n'
            return archive
        else:
            file_out.write(filepath + '\n')
            # file_out.write(file_len_str_len_str + '\n')
            file_out.write(file_len_str + '\n')
            file_out.write(file_contents)
            file_out.write('\n')
    
    @staticmethod
    def archive_strbin_data(filepath, file_contents, filepath_root, file_out, file_index):
        """
        Identical to archive_str_data() except that the incoming data can be a string or a bytes object.
        If it is a string object, it is converted to a bytes object first.
        The outgoing file is written in binary, along with indexing information, so as to faciliate file seeking upon dearchival.

        file_index is a list of pairs, where each pair is a filepath of the to-be-added content and the length of the content as bytes.
        """
        if isinstance(file_contents, str):
            file_contents = file_contents.strip()
            file_contents_bytes = file_contents.encode('utf-8')
        else:
            file_contents_bytes = file_contents
        filebytes_len = len(file_contents_bytes)
        if filebytes_len > 999999999:
            raise ValueError("RAMDataPond.archive_strbin_data() filebytes_len ({filebytes_len}) > 999,999,999!")
        
        if filepath.startswith(filepath_root):
            filepath = filepath[len(filepath_root):]

        file_out.write(file_contents_bytes)
        file_index.append([filepath, None, filebytes_len])  # The second element will be filled in later
    
    # def archive_files(self, filepaths, file_out=None):
    #     """
    #     Archive the indicated filepaths in the following string (not byte) format:
    #       1 char: File length N in ASCII length in ASCII
    #       1-9 chars: File length N in ASCII
    #       N chars: File string of length N
        
    #     For example, a file of length 765,432 characters would be encoded as:
    #       6765432file_content__file_content__file_content__etc.

    #     If the file length exceeds nine ASCII characters (999,999,999) an exception is raised.

    #     An optional file_out object can be passed in to indicate that the archive contents should be written to disk instead of a new memory object.
    #     """
    #     print('\n'.join(self.ram_data_pond.keys()))
    #     archive = "" if not file_out else None
    #     for filepath in filepaths:
    #         archive += archive_str_data(filepath, self.ram_data_pond[filepath], file_out)
        
    #     return archive

    @staticmethod
    def dearchive_file(data_loc, archive_filepath, keep_filters=[""]):
        """
            As files are read from the archive, only keep those whose filename includes any strings
            in the keep_filters list. All other dearchived files are discarded.
            """
        with open(archive_filepath) as f:
            while True:
                filepath = ""
                while len(filepath) < 1000:
                    c = f.read(1)
                    if c == '\n' or c == '':
                        break
                    filepath += c
                if filepath:
                    if len(filepath) >= 1000:
                        raise ValueError(f"Invalid filepath: {filepath}")
                    # logging.info(f"\nNext archive filepath: {filepath}")
                    # file_len_str_len_str = f.readline()
                    # file_len_str_len = int(file_len_str_len_str)
                    file_len_str = f.readline()
                    file_len = int(file_len_str)
                    file_contents = f.read(file_len)
                    assert f.read(1) == '\n'
                    filename = os.path.basename(filepath)
                    for keep_filter in keep_filters:
                        if keep_filter in filename:
                            logging.info(f"In filter ({file_len} B): {filename}")
                            filedirpath = '/'.join(filepath.split('/')[:-1])
                            # logging.info(f"Next archive filedirpath: {filedirpath}")
                            # logging.info(f"Next archive filename: {filename}")
                            os.makedirs(f"{data_loc}{filedirpath}", exist_ok=True)
                            with open(f"{data_loc}{filepath}", 'w') as fout:
                                fout.write(file_contents)
                            break
                else:
                    break
    
    @staticmethod
    def dearchive_file_verbose(data_loc, archive_filepath, keep_filters=[""]):
        """
        As files are read from the archive, only keep those whose filename includes any strings
        in the keep_filters list. All other dearchived files are discarded.
        """
        logging.info(f"Dearchiving {archive_filepath}\n  with keep_filters: {keep_filters}")
        st = default_timer()
        num_in_filter = 0
        total_subarchives_len = 0
        with open(archive_filepath) as f:
            subarchive_idx = -1
            while True:
                subarchive_idx += 1
                filepath = ""
                while len(filepath) < 1000:
                    c = f.read(1)
                    if c == '\n' or c == '':
                        break
                    filepath += c
                if filepath:
                    if len(filepath) >= 1000:
                        raise ValueError(f"Invalid filepath: {filepath}")
                    # logging.info(f"\nNext archive filepath: {filepath}")
                    # file_len_str_len_str = f.readline()
                    # file_len_str_len = int(file_len_str_len_str)
                    file_len_str = f.readline()
                    file_len = int(file_len_str)
                    file_contents = f.read(file_len)
                    assert f.read(1) == '\n'
                    filename = os.path.basename(filepath)
                    for keep_filter in keep_filters:
                        if keep_filter in filename:
                            num_in_filter += 1
                            total_subarchives_len += file_len
                            if subarchive_idx < 5:
                                logging.info(f"In filter (first 5 shown) ({file_len:>10,} B): {filename}")
                            filedirpath = '/'.join(filepath.split('/')[:-1])
                            # logging.info(f"Next archive filedirpath: {filedirpath}")
                            # logging.info(f"Next archive filename: {filename}")
                            os.makedirs(f"{data_loc}{filedirpath}", exist_ok=True)
                            with open(f"{data_loc}{filepath}", 'w') as fout:
                                fout.write(file_contents)
                            break  # Why is this break here?!
                else:
                    break
            logging.info(f"Total subarchive count, num subarchives in filter: {subarchive_idx+1} {num_in_filter}")
        logging.info(f"RAMDataPond.dearchive_file() elapsed time: {default_timer() - st:.3f}s")
        
        return num_in_filter, total_subarchives_len
    
    @staticmethod
    def dearchive_file__group_by_treelevel_and_shard(archive_i, data_loc, archive_filepath, keep_filters=[""]):
        """
        As files are read from the archive, only keep those whose filename includes any strings
        in the keep_filters list. All other dearchived files are discarded.
        """
        logging.info(f"\nDearchiving {archive_filepath}\n  with keep_filters: {keep_filters}")
        st = default_timer()
        num_in_filter = 0
        total_subarchives_len = 0
        treelevelcellids = set()  # debug
        with open(archive_filepath) as f:
            subarchive_groups = defaultdict(str)
            subarchive_idx = -1
            while True:
                subarchive_idx += 1
                filepath = ""
                while len(filepath) < 1000:
                    c = f.read(1)
                    if c == '\n' or c == '':
                        break
                    filepath += c
                if filepath:
                    if len(filepath) >= 1000:
                        raise ValueError(f"Invalid filepath: {filepath}")
                    # logging.info(f"\nNext archive filepath: {filepath}")
                    # file_len_str_len_str = f.readline()
                    # file_len_str_len = int(file_len_str_len_str)
                    file_len_str = f.readline()
                    file_len = int(file_len_str)
                    file_contents = f.read(file_len)
                    assert f.read(1) == '\n'
                    filename = os.path.basename(filepath)
                    for keep_filter in keep_filters:
                        if keep_filter in filename:
                            num_in_filter += 1
                            total_subarchives_len += file_len
                            if archive_i < 5 and subarchive_idx < 5:
                                logging.info(f"In filter (first 5 shown) ({file_len:>10,} B): {filename}")
                            pcs = filename.split('__')
                            
                            treelevelcellid = pcs[3].split('-')[1].replace(',', '_')
                            treelevelcellids.add(treelevelcellid)

                            group_filename = f"{pcs[0]}__{pcs[1]}__{pcs[2]}__{pcs[4]}"
                            if archive_i < 5 and subarchive_idx < 5:
                                logging.info(f"Subarchive group key: {group_filename}")
                            
                            file_content_lines_ends_with_endline = file_contents == '\n'
                            file_content_lines = file_contents.split('\n')
                            for i, file_content_line in enumerate(file_content_lines):
                                file_content_lines[i] = file_content_line + f",{treelevelcellid}"
                            file_contents = '\n'.join(file_content_lines)
                            if True:  # file_content_lines_ends_with_endline:
                                file_contents += '\n'
                            
                            subarchive_groups[group_filename] += file_contents
                else:
                    break
            logging.info(f"Total subarchive count, num subarchives in filter: {subarchive_idx} {num_in_filter}")
            for sag_i, (group_filename, file_contents) in enumerate(subarchive_groups.items()):
                filedirpath = group_filename[:group_filename.rindex('.')]
                os.makedirs(f"{data_loc}{filedirpath}", exist_ok=True)
                if archive_i < 5 and sag_i < 5:
                    logging.info(f"Writing subarchive group file {sag_i+1} of {len(subarchive_groups)} (only first 5 shown) of len {len(file_contents):>10,} B: {data_loc}{group_filename}")
                with open(f"{data_loc}{group_filename}", 'w') as fout:
                    fout.write(file_contents)
        
        # logging.info(f"treelevelcellids: {treelevelcellids}")
        
        logging.info(f"RAMDataPond.dearchive_file() elapsed time: {default_timer() - st:.3f}s")
        
        return num_in_filter, len(subarchive_groups), total_subarchives_len
    
    @staticmethod
    def dearchive_file_and_return_sectioned(archive_filepath):
        """
        Dearchive and return as labeled sections
        """
        archive_sections = {}
        with open(archive_filepath) as f:
            while True:
                filepath = ""
                while len(filepath) < 1000:
                    c = f.read(1)
                    if c == '\n' or c == '':
                        break
                    filepath += c
                if filepath:
                    if len(filepath) >= 1000:
                        raise ValueError(f"Invalid filepath: {filepath}")
                    # logging.info(f"\nNext archive filepath: {filepath}")
                    # file_len_str_len_str = f.readline()
                    # file_len_str_len = int(file_len_str_len_str)
                    file_len_str = f.readline()
                    file_len = int(file_len_str)
                    file_contents = f.read(file_len)
                    assert f.read(1) == '\n'
                    filename = os.path.basename(filepath)
                    filedirpath = '/'.join(filepath.split('/')[:-1])
                    # logging.info(f"Next archive filedirpath: {filedirpath}")
                    # logging.info(f"Next archive filename: {filename}")
                    archive_sections[filepath] = file_contents
                else:
                    break
        return archive_sections

    @staticmethod
    def dearchive_bin_file(data_loc, archive_filepath, filepath, start, length):
        """
        Given a file and its index metafile created with archive_strbin_data(), read a specific section out. 
        """
        with open(archive_filepath, 'rb') as f:
            f.seek(start)
            file_contents_bytes = f.read(length)
        filedirpath = '/'.join(filepath.split('/')[:-1])
        os.makedirs(f"{data_loc}{filedirpath}", exist_ok=True)
        with open(f"{data_loc}{filepath}", 'wb') as fout:
            fout.write(file_contents_bytes)
