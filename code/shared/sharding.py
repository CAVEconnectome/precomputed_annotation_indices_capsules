"""
Code that supports writing precomputed annotation files in sharded format
(to greatly reduce the number of files on disk). Reference:
https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md

Adapted from zetta_utils.layer.volumetric.annotation.sharding
"""
# pylint: disable=too-many-branches,too-many-statements,too-many-locals

import logging
import gzip
import io
import os
import struct
from timeit import default_timer
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import BinaryIO, Dict, List

import shared.annotations as annotations
import shared.utilities as utilities

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """
    Represents a chunk of data to be stored in a shard.
    """

    chunk_id: int
    data: bytes


def get_shard_hex(shard_number: int, shard_bits: int) -> str:
    """Convert shard number to zero-padded lowercase hex string.

    :param shard_number: The shard number to convert
    :param shard_bits: Number of bits for the shard
    :return: Zero-padded lowercase hex string
    """
    padding = ceil(shard_bits / 4)
    return f"{shard_number:0{padding}x}"


def write_shard_file(
    output_file: BinaryIO, sharding_spec: annotations.ShardingSpec, shard_number: int, chunks: List[Chunk]
):# -> None:
    """
    Write a shard file according to the Neuroglancer sharded format.

    :param output_file: File-like object to write the shard data to
    :param sharding_spec: The sharding specification
    :param shard_number: The shard number being written
    :param chunks: List of chunks that belong to this shard
    """
    times = [("start", default_timer())]
    t0 = default_timer()

    # Group chunks by minishard
    minishard_chunks: Dict[int, List[Chunk]] = defaultdict(list)
    minishard_chunk_ids = defaultdict(set)
    t1 = default_timer()

    logger.debug(f"Num chunks: {len(chunks)}    shard_number: {shard_number}")
    chunk_elap_time_accums = [0, 0]
    for chunk in chunks:
        t1_0 = default_timer()
        if shard_number is not None:  # Can be none when writing via ram buffer for a single shard (but might be included for validation anyway)
            expected_shard, minishard_num = sharding_spec.get_shard_number(chunk.chunk_id, True)
            # logger.debug(f"sharding.py.write_shard_file()   chunk loop chunk_id:{chunk.chunk_id:>10}    expected_shard ({expected_shard:>3}) ==? shard_number ({shard_number:>3})")
            if expected_shard != shard_number:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} belongs to shard {expected_shard}, "
                    f"not shard {shard_number}"
                )
        t1_1 = default_timer()

        # logger.debug(f"minishard_num: {minishard_num}")
        # if shard_number == 0 and chunk.chunk_id % 100 == 0:
        #     logger.info(f"BBB {chunk.chunk_id:15} {shard_number:5} {get_shard_hex(shard_number, sharding_spec.shard_bits)} {minishard_num:5}")
        if chunk.chunk_id in minishard_chunk_ids[minishard_num]:
            raise ValueError(f"About to add chunk to minishard {minishard_num} with duplicate chunk id {chunk.chunk_id}")
        minishard_chunks[minishard_num].append(chunk)
        minishard_chunk_ids[minishard_num].add(chunk.chunk_id)
        t1_2 = default_timer()
        chunk_elap_time_accums[0] += t1_1 - t1_0
        chunk_elap_time_accums[1] += t1_2 - t1_1
    logger.debug(f"sharding.write_shard_file chunk loop times:    t1_1: {chunk_elap_time_accums[0]:.1f}s    t1_2: {chunk_elap_time_accums[1]:.1f}s")

    t2 = default_timer()
    logger.debug(f"sharding    t1: {t1 - t0:.1f}s    t2: {t2 - t1:.1f}s")
    times.append(("group chunks by minishard", default_timer()))

    # Sort chunks within each minishard by chunk_id
    for minishard_num in minishard_chunks:
        minishard_chunks[minishard_num].sort(key=lambda c: c.chunk_id)

    times.append(("sort minishard chunks", default_timer()))

    # Build minishard indices and collect data
    num_minishards = sharding_spec.num_minishards_per_shard
    minishard_indices = {}
    minishard_data_sections = {}
    current_data_offset = 0

    # Keep track of where the next chunk of data is going to appear,
    # relative to the start of the minishard indexes.  The total
    # minishard index length is 24 (3 uint64's) times the number of chunks.
    next_data_pos = 24 * len(chunks)
    for minishard_num in range(num_minishards):
        chunks_in_minishard = minishard_chunks.get(minishard_num, [])

        if not chunks_in_minishard:
            # Empty minishard
            minishard_indices[minishard_num] = b""
            minishard_data_sections[minishard_num] = b""
            continue

        # Build minishard index arrays
        chunk_ids = []
        data_offsets = []
        data_sizes = []

        times.append((f"minishard {minishard_num} - init", default_timer()))

        # Process chunks and apply data encoding
        # (ToDo: figure out if we're supposed to gzip each chunk separately like this,
        # which would pretty much never be a good idea for annotations, or gzip the
        # entire data section at once.  The spec is unclear on this.)
        encoded_chunk_data = []
        for chunk in chunks_in_minishard:
            if sharding_spec.data_encoding == "gzip":
                encoded_data = gzip.compress(chunk.data)
            else:  # "raw"
                encoded_data = chunk.data

            encoded_chunk_data.append(encoded_data)
            data_sizes.append(len(encoded_data))
        
        times.append((f"minishard {minishard_num} - process chunk data and get sizes", default_timer()))

        # Delta encode chunk IDs
        prev_id = 0
        for chunk in chunks_in_minishard:
            chunk_ids.append(chunk.chunk_id - prev_id)
            prev_id = chunk.chunk_id

        # Delta encode data offsets relative to end of previous chunk
        # (with the first one equal to the next data position).
        for i, data_size in enumerate(data_sizes):
            if i == 0:
                data_offsets.append(next_data_pos)
            else:
                # Subsequent chunks: 0 additional offset, as we're not
                # putting any extra space between chunks
                data_offsets.append(0)
            next_data_pos += data_size
        
        times.append((f"minishard {minishard_num} - process id deltas and position deltas", default_timer()))

        # Build the minishard index binary data
        # Format: [3, n] array of uint64le values
        # array[0, :] = delta-encoded chunk IDs
        # array[1, :] = delta-encoded data offsets (from end of prior chunk)
        # array[2, :] = data sizes
        index_data = io.BytesIO()

        # Write chunk IDs (delta encoded)
        for chunk_id_delta in chunk_ids:
            index_data.write(struct.pack("<Q", chunk_id_delta))
        
        times.append((f"minishard {minishard_num} - write chunk id deltas", default_timer()))

        # Write data offsets (delta encoded)
        for offset_delta in data_offsets:
            index_data.write(struct.pack("<Q", offset_delta))
        
        times.append((f"minishard {minishard_num} - write offset deltas", default_timer()))

        # Write data sizes
        for size in data_sizes:
            index_data.write(struct.pack("<Q", size))
        
        times.append((f"minishard {minishard_num} - write sizes", default_timer()))

        raw_index = index_data.getvalue()

        # Apply minishard index encoding
        # NOTE: doing this will make the initial data offset above wrong.
        # This is a circular dependency: we need to know the final size of
        # all minishard indexes, but that will change depending on what
        # it contains (including that offset).  I see no obvious way to
        # resolve that.  So, don't use gzip!
        if sharding_spec.minishard_index_encoding == "gzip":
            encoded_index = gzip.compress(raw_index)
        else:  # "raw"
            encoded_index = raw_index

        minishard_indices[minishard_num] = encoded_index

        # Concatenate all chunk data for this minishard
        minishard_data = b"".join(encoded_chunk_data)
        minishard_data_sections[minishard_num] = minishard_data

        current_data_offset += len(minishard_data)
        
        times.append((f"minishard {minishard_num}", default_timer()))

    times.append(("process minishards", default_timer()))

    # Calculate minishard index positions
    minishard_index_offsets = {}
    current_index_offset = 0

    for minishard_num in range(num_minishards):
        start_offset = current_index_offset
        end_offset = start_offset + len(minishard_indices[minishard_num])
        minishard_index_offsets[minishard_num] = (start_offset, end_offset)
        current_index_offset = end_offset

    times.append(("calculate offset index", default_timer()))

    # Write shard index (2**minishard_bits * 16 bytes)
    for minishard_num in range(num_minishards):
        start_offset, end_offset = minishard_index_offsets[minishard_num]
        output_file.write(struct.pack("<Q", start_offset))  # start_offset: uint64le
        output_file.write(struct.pack("<Q", end_offset))  # end_offset: uint64le

    times.append(("write offset index", default_timer()))

    # Write minishard indices
    for minishard_num in range(num_minishards):
        output_file.write(minishard_indices[minishard_num])

    times.append(("write minishard indices", default_timer()))

    # Write chunk data
    for minishard_num in range(num_minishards):
        output_file.write(minishard_data_sections[minishard_num])

    times.append(("write chunk data", default_timer()))

    # logger.debug(f"sharding.py.write_shard_file() for shard {shard_number} elapsed time: {times[-1][1] - times[0][1]:.1f}s")

    elapsed_times = [(times[i][0], times[i][1] - times[i-1][1]) for i in range(1, len(times))]
    return {
        label: f"{et:.1f}s" for i, (label, et) in enumerate(elapsed_times)
    }


def write_shard_to_file(
    filepath: str, sharding_spec: annotations.ShardingSpec, shard_number: int, chunks: List[Chunk]
) -> None:
    """
    Convenience function to write a shard file to disk.

    :param filepath: Path where the shard file should be written
    :param sharding_spec: The sharding specification
    :param shard_number: The shard number being written
    :param chunks: List of chunks that belong to this shard
    """
    with open(os.path.expanduser(filepath), "wb") as f:
        write_shard_file(f, sharding_spec, shard_number, chunks)
        # logger.debug(f'Wrote {len(chunks)} items to shard file: {filepath}')

def writef_shard_to_file(
    file: BinaryIO, sharding_spec: annotations.ShardingSpec, shard_number: int, chunks: List[Chunk]
) -> None:
    """
    Convenience function to write a shard file to disk.

    :param filepath: Path where the shard file should be written
    :param sharding_spec: The sharding specification
    :param chunks: List of chunks that belong to this shard
    """
    return write_shard_file(file, sharding_spec, shard_number, chunks)
    # logger.debug(f'Wrote {len(chunks)} items to shard file: {filepath}')


def write_shard_files(dir_path: str, sharding_spec: annotations.ShardingSpec, chunks: List[Chunk]) -> None:
    # Sort chunks into groups by shard number
    qty_shards = sharding_spec.num_shards
    shard_chunks: List[List[Chunk]] = list([] for _ in range(0, qty_shards))
    for chunk in chunks:
        shard_num = sharding_spec.get_shard_number(chunk.chunk_id)
        shard_chunks[shard_num].append(chunk)
    # Then, write 'em out!
    dir_path = os.path.expanduser(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    for i in range(0, qty_shards):
        shard_hex = get_shard_hex(i, sharding_spec.shard_bits)
        file_path = utilities.path_join(dir_path, f"{shard_hex}.shard")
        write_shard_to_file(file_path, sharding_spec, i, shard_chunks[i])
        logger.debug(f"Shard file size for shard {i}, hex {shard_hex}: {os.path.getsize(file_path)}")

def writef_shard_files(file: BinaryIO, sharding_spec: annotations.ShardingSpec, shard_number, chunks: List[Chunk]) -> None:
    st = default_timer()
    for chunk in chunks:
        chunk_shard_num = sharding_spec.get_shard_number(chunk.chunk_id)
        if chunk_shard_num != shard_number:
            raise ValueError(f"sharding.py.writef_shard_files() received chunk with shard number that doesn't match the shard number being written: {chunk_shard_num} != {shard_number}")
    logger.debug(f"sharding.writef_shard_files() Chunk loop time: {default_timer() - st:.1f}s")
    
    return writef_shard_to_file(file, sharding_spec, shard_number, chunks)

def writef_shard_files_wo_ioobj(sharding_spec: annotations.ShardingSpec, shard_number: int, chunks: List[Chunk]) -> None:
    logging.info(f"writef_shard_files() Shard {shard_number}")
    st = default_timer()
    # Select the chunks for the indicated shard, dicarding the rest
    shard_chunks = []
    for chunk in chunks:
        chunk_shard_num = sharding_spec.get_shard_number(chunk.chunk_id)
        # logging.info(f"Chunk id & shard num:    {chunk.chunk_id} {chunk_shard_num}")
        if chunk_shard_num == shard_number:
            shard_chunks.append(chunk)
    
    logging.info(f"Of {len(chunks)} chunks, {len(shard_chunks)} were saved for shard {shard_number} to write to the file buffer (with {len(chunks) - len(shard_chunks)} discarded)")
    
    # Then, write 'em out!
    file_buffer = io.BytesIO()
    with file_buffer as f_buf:
        profile_time = writef_shard_to_file(f_buf, sharding_spec, shard_number, shard_chunks)
        file_buffer_bytes = file_buffer.getvalue()
    logger.debug(f"sharding.writef_shard_files() Time: {default_timer() - st:.1f}s")

    return file_buffer_bytes, profile_time
