import logging
import os
import struct
import io
from collections import defaultdict

import annotations as anno

class ShardFileReader:
    def __init__(self,
            filepath,
            preshift_bits, shard_bits, minishard_bits,
            property_specs=None, 
            relationships=None,
            verbose_level=0
        ):
        self.preshift_bits = preshift_bits
        self.shard_bits = shard_bits
        self.minishard_bits = minishard_bits
        
        self.num_minishards = 2**self.minishard_bits

        self.property_specs = property_specs
        self.relationships = relationships
        
        if verbose_level >= 1:
            logging.info("Shard file size:", os.path.getsize(filepath), "B")
        with open(filepath, 'rb') as f:
            self.content = f.read()
        if verbose_level >= 1:
            logging.info("Shard content len:", len(self.content), "B")
    
    def read_n(self, n):
        ret, self.content = self.content[:n], self.content[n:]
        return ret

    def read_lQ(self):
        return struct.unpack("<Q", self.read_n(8))[0]

    def read_minishard_start_end_offsets(self, verbose_level):
        """
        This function reads data written by sharding.write_shard_file()
        """
        if verbose_level >= 2:
            logging.info("\nMinishard start/end offsets")
        self.minishard_start_ends = {}
        total_n_chunks = 0
        for minishard_i in range(self.num_minishards):
            start_offset = self.read_lQ()
            end_offset = self.read_lQ()
            self.minishard_start_ends[minishard_i] = [start_offset, end_offset]
            
            minishard_len = end_offset - start_offset
            assert minishard_len % 24 == 0  # 3 sets of arrays of 8-byte values
            n_chunks_in_minishard = minishard_len // 24

            total_n_chunks += n_chunks_in_minishard
            
            if verbose_level >= 2:
                logging.info(f"  Minishard: {minishard_i:>4}    start/end offsets: {start_offset:>8}  to{end_offset:>8}    -> len: {minishard_len:>8}    // 24 = num chunks: {n_chunks_in_minishard}")

        if verbose_level >= 2:
            logging.info(f"Total chunks in shard across all minishards: {total_n_chunks}")
        if total_n_chunks == 0:
            raise ValueError(f"ERROR! Total chunks in shard across all minishards = 0")

    def read_minishard_indices(self, verbose_level):
        """
        This function reads data written by sharding.write_shard_file()
        """
        if verbose_level >= 3:
            logging.info("\nMinishard indices")
        self.minishard_idx_chunk_id_deltas = defaultdict(list)
        self.minishard_idx_offset_deltas = defaultdict(list)
        self.minishard_idx_sizes = defaultdict(list)
        for minishard_i in range(self.num_minishards):
            if verbose_level >= 3:
                logging.info(f"  Minishard {minishard_i}")
            start, end = self.minishard_start_ends[minishard_i]
            minishard_len = end - start
            assert minishard_len % 24 == 0  # 3 sets of arrays of 8-byte values
            n_chunks_in_minishard = minishard_len // 24
            if verbose_level >= 3:
                logging.info(f"    Length: from {start} to {end} -> {minishard_len:>8}        // 24 = num chunks: {n_chunks_in_minishard}")
            
            if n_chunks_in_minishard > 0:
                for chunk_id_delta_i in range(n_chunks_in_minishard):
                    self.minishard_idx_chunk_id_deltas[minishard_i].append(self.read_lQ())
                
                for offset_delta_i in range(n_chunks_in_minishard):
                    self.minishard_idx_offset_deltas[minishard_i].append(self.read_lQ())
                
                for minishard_idx_size_i in range(n_chunks_in_minishard):
                    self.minishard_idx_sizes[minishard_i].append(self.read_lQ())
                
                if verbose_level >= 3:
                    logging.info(f"    Chunk id deltas: {self.minishard_idx_chunk_id_deltas[minishard_i]}")
                    logging.info(f"    Offset deltas:   {self.minishard_idx_offset_deltas[minishard_i]}")
                    logging.info(f"    Sizes:           {self.minishard_idx_sizes[minishard_i]}")

    def read_minishard_chunks(self, verbose_level):
        if verbose_level >= 4:
            logging.info("\nData")
        for minishard_i in range(self.num_minishards):
            if verbose_level >= 4:
                logging.info(f"  Minishard {minishard_i}")
            
            chunk_id_deltas = self.minishard_idx_chunk_id_deltas[minishard_i]
            if verbose_level >= 4:
                logging.info(f"    Chunk id deltas: {chunk_id_deltas}")
            offset_deltas = self.minishard_idx_offset_deltas[minishard_i]
            if verbose_level >= 4:
                logging.info(f"    Offset deltas:   {offset_deltas}")
            sizes = self.minishard_idx_sizes[minishard_i]
            if verbose_level >= 4:
                logging.info(f"    Sizes:           {sizes}")
            
            n_chunks_in_minishard = len(chunk_id_deltas)

            prev_chunk_id = 0
            for chunk_i in range(n_chunks_in_minishard):
                if verbose_level >= 4:
                    logging.info(f"    Chunk i: {chunk_i}")
                chunk_id_delta = chunk_id_deltas[chunk_i]
                chunk_id = prev_chunk_id + chunk_id_delta
                prev_chunk_id = chunk_id
                offset_delta = offset_deltas[chunk_i]
                size = sizes[chunk_i]
                if verbose_level >= 4:
                    logging.info(f"      Id delta, id, offset delta, size: {chunk_id_delta} {chunk_id} {offset_delta} {size}")
                # The next line reads data written by writer.Writer.compile_multi_annotation_buffer()
                
                chunk_buffer = self.read_n(size)
                if verbose_level >= 4:
                    logging.info(f"      Chunk buffer: {type(chunk_buffer)} {len(chunk_buffer)}")
                chunk_bytes_io = io.BytesIO(chunk_buffer)
                
                chunk_n_annos = struct.unpack("<Q", chunk_bytes_io.read(8))[0]
                if verbose_level >= 4:
                    logging.info(f"      Num annotations: {chunk_n_annos}")
                for anno_i in range(chunk_n_annos):
                    # This section reads data written by annotations.Annotation.write() (via writer.Writer.compile_multi_annotation_buffer()).
                    annotation = anno.Annotation.read(chunk_bytes_io, "LINE", self.property_specs, self.relationships)
                    if verbose_level >= 5:
                        logging.info(f"        Annotation {anno_i}: {type(annotation)}")
                        logging.info(f"          start:             {annotation.start}")
                        logging.info(f"          end:               {annotation.end}")
                    if annotation.properties:
                        if verbose_level >= 5:
                            logging.info(f"          properties:        {annotation.properties}")
                    if annotation.relations:
                        if verbose_level >= 5:
                            logging.info(f"          relations:         {annotation.relations}")
                # The next line reads data written by writer.Writer.compile_multi_annotation_buffer()
                for anno_i in range(chunk_n_annos):
                    anno_id = struct.unpack("<Q", chunk_bytes_io.read(8))[0]
                    if verbose_level >= 5:
                        logging.info(f"        Annotation id: {anno_id:>12}")

    def read(self, verbose_level=0):
        self.read_minishard_start_end_offsets(verbose_level)
        # self.read_minishard_indices(verbose_level)
        # self.read_minishard_chunks(verbose_level)
