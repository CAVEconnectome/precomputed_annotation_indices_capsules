import numpy as np
import random

from shared.ram_data_pond import *
from shared.ram_file_tree import *

class RawTable():
    """
    Pandas Dataframes are very slow. The interface is convenient for development,
    but for production work at massive scale, especially when fancier functions aren't required,
    it is preferable to represent massive tabular data with native types like strings and lists.
    """

    def __init__(self, lines, header):
        """
        lines: a List of comma-delimited text lines (not split into lists of field values).
            This storage is faster to write out. It is, essentially premature optimization to split the text lines apart,
            only to have to put them back together again for export. Depending on how heavily the data is accessed in between loading/resaving,
            there could be a trade-off where splitting the rows apart becomes advantageous. Or, at a cost of doubling the storage,
            a lazy approach would only split the lines apart if/when they are accessed on a per-field basis the first time,
            but then store both versions, line and list, for the remainder of the RawTable's lifetime.
            [
                "abc,def,ghi",
                "jkl,mno,pqr",
                "stu,vwx,yz_",
            ]
        header: a List of column names (not a comma-delimited text line)
            [ "date", "type", "units" ]
        """
        if header:
            self.header = header
            self.lines = lines  # Assume there is no header row
        else:
            self.header = lines[0].split(',') if len(lines) >= 1 else []  # Infer the header from the first row
            self.lines = lines[1:] if len(lines) >= 2 else []  # Skip the header row
        self.header_col_idx = {column: idx for idx, column in enumerate(self.header)}
    
    def __len__(self):
        return len(self.lines)
    
    def unique(self, column_name):
        col_idx = self.header_col_idx[column_name]
        values = set()
        for line in self.lines:
            fields = line.split(',')
            field = fields[col_idx]
            values.add(field)
        return values
    
    def sample(self, n, random_state=None):
        # import time
        # random.seed(time.time())
        if random_state is not None:
            random.seed(random_state)
        row_indices = random.sample(list(np.arange(len(self.lines))), n)
        sampled_rows = [self.lines[row_idx] for row_idx in row_indices]
        sampled_table = RawTable(sampled_rows, self.header)
        return sampled_table, row_indices

    def drop(self, indices_to_drop):
        if not indices_to_drop:
            return RawTable([line for line in self.lines], self.header)

        indices_to_drop = sorted(indices_to_drop)
        new_lines = []

        new_lines.extend(self.lines[:indices_to_drop[0]])
        for i in range(len(indices_to_drop) - 1):
            assert indices_to_drop[i] < indices_to_drop[i+1]  # The list is sorted, but could conceivably contain duplicates
            new_lines.extend(self.lines[ indices_to_drop[i]+1 : indices_to_drop[i+1] ])
        new_lines.extend(self.lines[indices_to_drop[-1]+1:])

        assert len(new_lines) == len(self.lines) - len(indices_to_drop)
        new_table = RawTable(new_lines, self.header)
        
        return new_table
    
    def iterlines(self):
        for i in range(len(self.lines)):
            yield self.lines[i]
    
    def get_col_idx(self, column):
        return self.header_col_idx[column]
    
    def get_row_field_val(self, line, column):
        if isinstance(line, str):
            return line.split(',')[self.header_col_idx[column]]
        else: # elif isinstance(line, list):
            return line[self.header_col_idx[column]]
    
    def add_column(self, column_name, val):
        self.header.append(column_name)

        if isinstance(val, list):
            if len(list) != len(self.lines):
                raise ValueError("RawTable.add_column() received list of new values with different length than table")
            for i in range(len(self.lines)):
                self.lines[i] += f",{val[i]}"
        else:
            for i in range(len(self.lines)):
                self.lines[i] += f",{val}"
    
    def to_csv(self, filepath, index=False, header=False):
        with open(filepath, 'w') as f:
            if header:
                f.write(','.join(header) + '\n')
            if index:
                for i, line in enumerate(self.lines):
                    f.write(str(i) + ',' + line + '\n')
            else:
                for line in self.lines:
                    if line.startswith('id'):
                        logging.error("ERROR! Line contains header")
                        logging.info(line)
                        assert False
                    f.write(line + '\n')
    
    def to_csv_to_disk_or_ram_data_pond(self, filepath, ram_data_pond, index=False, header=False):
        s = ""
        if header:
            s += ','.join(header) + '\n'
        if index:
            for i, line in enumerate(self.lines):
                s += str(i) + ',' + line + '\n'
        else:
            for line in self.lines:
                if line.startswith('id'):
                    logging.error("ERROR! Line contains header")
                    logging.info(line)
                    assert False
                s += line + '\n'
        
        ram_data_pond.write_to_disk_or_ram_data_pond(filepath, s)
    
    def to_csv_to_disk_or_ram_file_tree(self, filepath, ram_file_tree, index=False, header=False):
        s = ""
        if header:
            s += ','.join(header) + '\n'
        if index:
            for i, line in enumerate(self.lines):
                s += str(i) + ',' + line + '\n'
        else:
            for line in self.lines:
                if line.startswith('id'):
                    logging.error("ERROR! Line contains header")
                    logging.info(line)
                    assert False
                s += line + '\n'
        
        ram_file_tree.put(filepath, s)
