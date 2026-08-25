# precomputed_annotation_indices_capsules
Code Ocean capsules

For documentation regarding this code's use within Code Ocean or Nextflow, please refer here: https://github.com/CAVEconnectome/precomputed_annotation_indices_pipeline

The following describes how to build an annotation spatial index in a standalone fashion, with no reliance on Code Ocean or Nextflow. The relevant module is simply imported as a straightforward Python module and builds a spatial index from an input CSV or Parquet file on local disk without reliance on a distributed cluster (Code Ocean) or even on a local pipelining framework (Nextflow).

The basic steps are:
- Clone the spatial indexing git repo (this same repo is used by Code Ocean or by Nextflow, but it has a standalone usage pathway too).
- Construct a data config json file.
- Build and run the relevant Python to initiate the indexing process, which is shown below.

The repo resides here:
https://github.com/CAVEconnectome/precomputed_annotation_indices_capsules

The data config json file is documented elsewhere, namely here:
https://github.com/CAVEconnectome/precomputed_annotation_indices_pipeline
In the repo, you only need to study the section of the README that describes building a data config file. You can ignore the rest of the documentation on that page. Note that that repo also includes a data config example and a data config template that you can use to build your own data config file for your own data.

A basic Python script to build a spatial index then looks somelike like the following:
```
import capsule_generate_config.capsule_generate_config as gc
import capsule_build_spatial_index_standalone.capsule_build_spatial_index_standalone as bsis

if __name__ == "__main__":
    data_loc = "../data/"
    results_loc = "../results/"

    input_file = <full path to an input CSV or Parquet file>
    
    data_config = gc.read_data_config(data_loc)
    bsis.run(data_loc, results_loc, data_config, input_file)
```

If you are running this in a capsule in Code Ocean, then the `input_file` probably begins with `../data/`, just like the indicated `data_loc` variable.