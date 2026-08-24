# precomputed_annotation_indices_capsules
Code Ocean capsules

For documentation regarding this code's use within Code Ocean or Nextflow, please refer here: https://github.com/CAVEconnectome/precomputed_annotation_indices_pipeline

The following describes how to build an annotation spatial index in a standalone fashion, with no reliance on Code Ocean or Nextflow. The relevant module is simply imported as a straightforward Python module and builds a spatial index from an input CSV or Parquet file on local disk without reliance on a distributed cluster (Code Ocean) or even on a local pipelining framework (Nextflow).

