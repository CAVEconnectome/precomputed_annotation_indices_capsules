"""
This module provides the SimpleWriter class, which is able to write a precomputed
annotation file given a complete set of annotations (including properties and
relations), including sharding support.

Adapted from zetta_utils.layer.volumetric.annotation.simple_writer
"""
# pylint: disable=too-many-instance-attributes,too-many-branches,too-many-nested-blocks,too-many-locals

import logging
import io
import os
import struct
from timeit import default_timer
from collections import defaultdict
from dataclasses import dataclass
from random import random, shuffle
from typing import Any, Sequence, BinaryIO

import shared.geometry as geometry
import shared.annotations as annotations
import shared.sharding as sharding
import shared.utilities as utilities

logger = logging.getLogger(__name__)

class SimpleWriter:
    def __init__(self, anno_type, dimensions=None, lower_bound=None, upper_bound=None, tree_level=None, cell_index=None):
        """
        Initialize SimpleWriter with required parameters.

        :param anno_type: one of 'POINT', 'LINE' (and later others)
        :param dimensions: dimensions for the annotation space
        :param lower_bound: lower bound coordinates
        :param upper_bound: upper bound coordinates
        :param annotations: sequence of LineAnnotation objects (optional)
        """
        self.anno_type = anno_type
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.tree_level = tree_level
        self.cell_index = cell_index
        self.annotations = []  # list of Annotation objects
        self.cell_annotations = defaultdict(list)  # Annotations grouped by cell index
        self.spatial_specs = []  # SpatialEntry objects
        self.property_specs = []  # PropertySpec objects
        self.relationships = []  # Relationship objects
        self.by_id_sharding = None  # optional ShardingSpec
        self.spatial_sharding = None  # optional ShardingSpec

    def format_info(self):
        """Format the info JSON structure using instance properties."""
        spatial_json = "    " + ",\n        ".join([se.to_json() for se in self.spatial_specs])
        property_json = "    " + ",\n        ".join([ps.to_json() for ps in self.property_specs])
        relationship_json = "    " + ",\n        ".join([r.to_json() for r in self.relationships])
        if self.by_id_sharding is None:
            by_id_json = '{ "key" : "by_id" }'
        else:
            by_id_json = '{ "key" : "by_id", "sharding": ' + self.by_id_sharding.to_json() + " }"

        return f"""{{
    "@type" : "neuroglancer_annotations_v1",
    "annotation_type" : "{self.anno_type}",
    "by_id" : {by_id_json},
    "dimensions" : {str(self.dimensions).replace("'", '"')},
    "lower_bound" : {list(self.lower_bound)},
    "upper_bound" : {list(self.upper_bound)},
    "properties" : [
    {property_json}
    ],
    "relationships" : [
    {relationship_json}
    ],
    "spatial" : [
    {spatial_json}
    ]
}}
"""

    def write(self, dir_path: str):
        """
        Write all annotation data to the specified directory.

        :param dir_path: path to the directory where files will be written
        """
        assert RuntimeError("Due to code changes by which Writer.annotations are now grouped by cell index under Writer.cell_annotations, Writer.write() is currently deprecated until its downstream functions, e.g., Writer._write_by_id_index(), Writer._write_related_index() are properly updated.")

        # Write by-id index (including relationships)
        self._write_by_id_index(utilities.path_join(dir_path, "by_id"))

        # Write the spatial index
        self._write_spatial_index(dir_path)

        # Write the related-object-id indexes
        for rel in self.relationships:
            self._write_related_index(dir_path, rel)

        # Write info file (AFTER subdivision, so we have accurate limits)
        info_content = self.format_info()
        info_file_path = utilities.path_join(dir_path, "info")
        utilities.write_bytes(info_file_path, info_content.encode("utf-8"))

    def writef(self, file: BinaryIO, shard_number=None):
        """
        Write all annotation data to the specified directory.

        :param file: file object where files will be written
        """
        # Write by-id index (including relationships)
        # self._write_by_id_index(utilities.path_join(dir_path, "by_id"))

        # Write the spatial index
        self._writef_spatial_index(file, shard_number)

        # Write the related-object-id indexes
        # for rel in self.relationships:
        #     self._write_related_index(dir_path, rel)

        # Write info file (AFTER subdivision, so we have accurate limits)
        # info_content = self.format_info()
        # info_file_path = utilities.path_join(dir_path, "info")
        # utilities.write_bytes(info_file_path, info_content.encode("utf-8"))

    def compile_multi_annotation_buffer(
        self,
        annotations: Sequence[annotations.Annotation] | None = None,
        randomize: bool = False,
    ):
        """
        Compile a set of lines to a bytes, in 'multiple annotation encoding' format:
                1. Line count (uint64le)
                2. Data for each line (excluding ID), one after the other
                3. The line IDs (also as uint64le)

        :param file_or_gs_path: local file or GS path of file to write
        :param annotations: iterable of Annotation objects (uses self.annotations if None)
        :param randomize: if True, the annotations will be written in random
                order (without mutating the lines parameter)
        :return: bytes representing compiled annotations
        """
        if annotations is None:
            annotations = self.annotations

        annotations = list(annotations)
        if randomize:
            annotations = annotations[:]
            shuffle(annotations)

        buffer = io.BytesIO()
        # first write the count
        buffer.write(struct.pack("<Q", len(annotations)))

        logger.debug(f"Writer.compile_multi_annotation_buffer(): Wrote number of annotations: {len(annotations)}")
        
        logger.debug(f"Writer.compile_multi_annotation_buffer(): Buffer length so far: {buffer.getbuffer().nbytes} B")

        # then write the annotation data
        for anno in annotations:
            logger.debug(f"\nWriter.compile_multi_annotation_buffer(): Writing annotation:\n  {anno}\n  with property_specs {self.property_specs}")
            anno.write(buffer, self.property_specs)
        
            logger.debug(f"Writer.compile_multi_annotation_buffer(): Buffer length so far: {buffer.getbuffer().nbytes} B")
        
        logger.debug(f"Writer.compile_multi_annotation_buffer(): Buffer length so far: {buffer.getbuffer().nbytes} B")

        # finally write the ids at the end of the buffer
        for anno in annotations:
            logger.debug(f"\nWriter.compile_multi_annotation_buffer(): Writing annotation id: {anno.id}")
            buffer.write(struct.pack("<Q", anno.id))
        
            logger.debug(f"Writer.compile_multi_annotation_buffer(): Buffer length so far: {buffer.getbuffer().nbytes} B")
        
        logger.debug(f"Writer.compile_multi_annotation_buffer(): Wrote buffer of final length: {buffer.getbuffer().nbytes} B")

        buffer.seek(0)  # Rewind buffer to the beginning
        return buffer.getvalue()

    def write_annotations(
        self,
        file_or_gs_path: str,
        annotations: Sequence[annotations.Annotation] | None = None,
        randomize: bool = False,
    ):
        """
        Write a set of lines to the given file, in 'multiple annotation encoding' format:
                1. Line count (uint64le)
                2. Data for each line (excluding ID), one after the other
                3. The line IDs (also as uint64le)

        :param file_or_gs_path: local file or GS path of file to write
        :param annotations: iterable of Annotation objects (uses self.annotations if None)
        :param randomize: if True, the annotations will be written in random
                order (without mutating the lines parameter)
        """
        data = self.compile_multi_annotation_buffer(annotations, randomize)
        utilities.write_bytes(file_or_gs_path, data)

    def _write_by_id_index(self, by_id_path: str):
        """
        Write the Annotation id index for the given set of annotations.
        Currently, in unsharded uint64 index format.

        :param by_id_path: complete path to the by_id directory.
        """
        if self.by_id_sharding is None:
            # In unsharded format, the by_id directory simply contains a little
            # binary file for each annotation, named with its id.
            for anno in self.annotations:
                file_path = utilities.path_join(by_id_path, str(anno.id))
                buffer = io.BytesIO()
                anno.write(buffer, self.property_specs, self.relationships)
                utilities.write_bytes(file_path, buffer.getvalue())
        else:
            # Otherwise, it's a chunk per annotation, shoved into shard files.
            chunks = []

            # logger.debug("Writer._write_by_id_index(): Checking annotations for duplicates (remove this once validated for production)")
            # anno_ids = set()

            for anno in self.annotations:
                # if anno.id in anno_ids:
                #     raise ValueError(f"Duplicate annotation id: {anno.id}")
                # anno_ids.add(anno.id)
                buffer = io.BytesIO()
                anno.write(buffer, self.property_specs, self.relationships)
                # logger.debug(f"_write_by_id_index() Chunk size: {buffer.getbuffer().nbytes}")
                chunks.append(sharding.Chunk(anno.id, buffer.getvalue()))
            # logger.debug(f"_write_by_id_index() Num chunks: {len(chunks)}")
            sharding.write_shard_files(by_id_path, self.by_id_sharding, chunks)

    def _writef_by_id_index(self, file: BinaryIO, shard_num=None):
        """
        Write the Annotation id index for the given set of annotations.
        Currently, in unsharded uint64 index format.

        :param file: file object where files will be written
        """
        if self.by_id_sharding is None:
            raise RuntimeError("SimpleWriter._writef_by_id_index() with (by_id_sharding is None) not implemented yet")
            # In unsharded format, the by_id directory simply contains a little
            # binary file for each annotation, named with its id.
            # for anno in self.annotations:
            #     file_path = utilities.path_join(by_id_path, str(anno.id))
            #     buffer = io.BytesIO()
            #     anno.write(buffer, self.property_specs, self.relationships)
            #     utilities.write_bytes(file_path, buffer.getvalue())
        else:
            # Otherwise, it's a chunk per annotation, shoved into shard files.
            chunks = []

            # logger.debug("Writer._write_by_id_index(): Checking annotations for duplicates (remove this once validated for production)")
            # anno_ids = set()

            anno_write_accum_elap_t = 0
            for anno in self.annotations:
                # if anno.id in anno_ids:
                #     raise ValueError(f"Duplicate annotation id: {anno.id}")
                # anno_ids.add(anno.id)
                buffer = io.BytesIO()
                st = default_timer()
                anno.write(buffer, self.property_specs, self.relationships)
                anno_write_accum_elap_t += default_timer() - st
                # logger.debug(f"_writef_by_id_index() Chunk size: {buffer.getbuffer().nbytes}")
                chunks.append(sharding.Chunk(anno.id, buffer.getvalue()))
            # logger.debug(f"_writef_by_id_index() Num chunks: {len(chunks)}")
            st = default_timer()
            sharding_profile = sharding.writef_shard_files(file, self.by_id_sharding, shard_num, chunks)
            shard_write_elap_t = default_timer() - st

            # logger.debug(f"_writef_by_id_index() anno_write_accum_elap_t: {anno_write_accum_elap_t:.1f}s")
            # logger.debug(f"_writef_by_id_index() shard_write_elap_t:      {shard_write_elap_t:.1f}s")

            return {
                "Writer.writef_by_id_index() - anno_write_accum_elap_t": f"{anno_write_accum_elap_t:.1f}s",
                "Writer.writef_by_id_index() - shard_write_elap_t": f"{shard_write_elap_t:.1f}s",
                "sharding_profile": sharding_profile,
            }

    def _write_related_index(self, dir_path: str, relation: annotations.Relationship):
        """
        Write a related object ID index, where for each related object ID,
        we have a file of annotations that contain that ID for that relation.

        :param dir_path: path to the directory containing the info file
        :param relation: the Relationship object to process
        """
        # Gather up the annotations for each related value
        rel_id_to_anno: dict[int, list[annotations.Annotation]] = {}
        for anno in self.annotations:
            related_ids = anno.relations.get(relation.id, [])
            if isinstance(related_ids, int):
                related_ids = [related_ids]
            for rel_id in related_ids:
                anno_list = rel_id_to_anno.get(rel_id, None)
                if anno_list is None:
                    anno_list = []
                    rel_id_to_anno[rel_id] = anno_list
                anno_list.append(anno)

        # Then write to disk directly, or prepare as shard files
        assert relation.key is not None  # which it can't be, silly black
        rel_dir_path = utilities.path_join(dir_path, relation.key)
        if relation.sharding is None:
            for related_id, anno_list in rel_id_to_anno.items():
                file_path = utilities.path_join(rel_dir_path, str(related_id))
                self.write_annotations(file_path, anno_list, False)
        else:
            chunks = []
            for related_id, anno_list in rel_id_to_anno.items():
                data = self.compile_multi_annotation_buffer(anno_list, False)
                chunks.append(sharding.Chunk(related_id, data))
                # print(f"Related id {related_id} compiles to {len(data)} bytes")
            sharding.write_shard_files(rel_dir_path, relation.sharding, chunks)

    def _writef_related_index(self, relation: annotations.Relationship, shard_number: int):
        """
        Write a related object ID index, where for each related object ID,
        we have a file of annotations that contain that ID for that relation.

        :param relation: the Relationship object to process
        """
        # Gather up the annotations for each related value
        rel_id_to_anno: dict[int, list[annotations.Annotation]] = {}
        for anno in self.annotations:
            related_ids = anno.relations.get(relation.id, [])
            # logging.info(f"_writef_related_index() Annotation: {anno}    Relation ids: {related_ids}")
            if isinstance(related_ids, int):
                related_ids = [related_ids]
            for rel_id in related_ids:
                anno_list = rel_id_to_anno.get(rel_id, None)
                if anno_list is None:
                    anno_list = []
                    rel_id_to_anno[rel_id] = anno_list
                anno_list.append(anno)

        # Then write to disk directly, or prepare as shard files
        assert relation.key is not None  # which it can't be, silly black
        if relation.sharding is None:
            raise RuntimeError("SimpleWriter._writef_related_index() with (relation.sharding is None) not implemented yet")
            # for related_id, anno_list in rel_id_to_anno.items():
            #     file_path = utilities.path_join(rel_dir_path, str(related_id))
            #     self.write_annotations(file_path, anno_list, False)
        else:
            chunks = []
            for related_id, anno_list in rel_id_to_anno.items():
                # logging.info(f"_writef_related_index() Adding chunk: {related_id} {len(anno_list)}")
                data = self.compile_multi_annotation_buffer(anno_list, False)
                chunks.append(sharding.Chunk(related_id, data))
                # print(f"Related id {related_id} compiles to {len(data)} bytes")
            st = default_timer()
            file_buffer_bytes, writer_profile = sharding.writef_shard_files_wo_ioobj(relation.sharding, shard_number, chunks)
            shard_write_elap_t = default_timer() - st
            return file_buffer_bytes, writer_profile

    def _write_spatial_index(self, dir_path: str):
        """
        Write the spatial index for the given set of annotations.

        :param dir_path: Directory path for output files, or None to skip file writing
        """
        if dir_path is not None:
            task_spec = self.spatial_specs[self.tree_level]
            if self.spatial_sharding is None:
                raise RuntimeError("SimpleWriter._write_spatial_index() with (spatial_sharding is None) not implemented yet")
                # TODO: The various annotations can have various cell indexes!
                # file_name = "_".join(str(i) for i in self.cell_index)
                # file_path = utilities.path_join(dir_path, task_spec.key, file_name)
                # self.write_annotations(file_path, self.annotations, True)
            else:
                chunk_data = self.compile_multi_annotation_buffer(self.annotations, True)
                chunk_id = utilities.compressed_morton_code(self.cell_index, task_spec.grid_shape)
                chunk = sharding.Chunk(chunk_id, chunk_data)

                shard_dir = utilities.path_join(dir_path, task_spec.key)
                sharding.write_shard_files(shard_dir, task_spec.sharding, [chunk])

    def _writef_spatial_index(self, file: BinaryIO, shard_num=None):
        """
        Write the spatial index for the given set of annotations.

        :param dir_path: Directory path for output files, or None to skip file writing
        """
        if file is not None:
            task_spec = self.spatial_specs[self.tree_level]
            if self.spatial_sharding is None:
                raise RuntimeError("SimpleWriter._writef_spatial_index() with (spatial_sharding is None) not implemented yet")
                # TODO: The various annotations can have various cell indexes!
                # file_name = "_".join(str(i) for i in self.cell_index)
                # file_path = utilities.path_join(dir_path, task_spec.key, file_name)
                # self.write_annotations(file_path, self.annotations, True)
            else:
                chunks = []

                # logger.debug("Writer._writef_spatial_index(): Checking cells for duplicates (remove this once validated for production)")
                # cell_indices = set()
                
                for cell_index, cell_annotations in self.cell_annotations.items():
                    # if cell_index in cell_indices:
                    #     raise ValueError(f"Duplicate cell index: {cell_index}")
                    # cell_indices.add(cell_index)
                    cell_index = [int(v) for v in cell_index.split('_')]
                    chunk_data = self.compile_multi_annotation_buffer(cell_annotations, True)
                    chunk_id = utilities.compressed_morton_code(cell_index, task_spec.grid_shape)
                    logger.debug(f"Writer._writef_spatial_index() annotation loop {cell_index}    task_spec.grid_shape {task_spec.grid_shape}    Chunk id: {chunk_id:>10}    Chunk data size: {len(chunk_data)} B")
                    chunks.append(sharding.Chunk(chunk_id, chunk_data))

                sharding.writef_shard_files(file, task_spec.sharding, shard_num, chunks)
