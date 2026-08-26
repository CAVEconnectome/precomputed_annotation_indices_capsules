# precomputed_annotation_indices_capsules

This repo implements a set of Code Ocean capsules, one per subdirectory under `code/` that represent the pieces of a Code Ocean or Nextflow pipeline for generating sharding precomputed ID, relation, and spatial indices of annotations for use by Neuroglancer or other analysis and visualization tools.

For documentation regarding this code's use ***within*** Code Ocean or Nextflow, please refer here: https://github.com/CAVEconnectome/precomputed_annotation_indices_pipeline

The following describes how to build an annotation spatial index ***in a standalone fashion***, with no reliance on Code Ocean or Nextflow. The relevant module is simply imported as a straightforward Python module and builds a spatial index from an input CSV or Parquet file on local disk without reliance on a distributed cluster (Code Ocean) or even on a local pipelining framework (Nextflow).

The basic steps are:
- Clone this spatial indexing git repo (this same repo is used by Code Ocean or by Nextflow, but it has a standalone usage pathway too).
- Construct a data config json file.
- Build and run the relevant Python to initiate the indexing process, which is shown below.

The repo resides here:
https://github.com/CAVEconnectome/precomputed_annotation_indices_capsules

The data config json file is documented elsewhere, namely here:
https://github.com/CAVEconnectome/precomputed_annotation_indices_pipeline

You don't need to clone that repo. Just study the top-level README. Of that README, you only need to focus on the section near the top that describes building a data config file. You can ignore the rest of the documentation on that page. Note that that repo also includes a data config example and a data config template that you can use to build your own data config file for your own data.

After building an initial data config file as described above, you must make some small additions or alterations to it. The documentation above shows how to construct a data config file for a Code Ocean or Nextflow run. For a standalone run, you need to add `data_size` to the json. I like to put it between `data_version` and `data_sizes` (notice the extra 's'; don't confuse them). This `data_size` key/value indicates a list with the following items (in the following order, of course):
- [string] A *label* that for your purposes is unimportant and unused.
- [int] *Data size in bytes*.
- [int] *Data size in rows or annotations*.
- [int] *Split size*: This will almost certainly be ignored under normal usage (if your data config json includes a non-null `id_column` parameter, then this split size parameter will be ignored). Most scenarios will involve indicating an id column in the config, and therefore this split size parameter will usually be ignored. It relates to the number of splits into which to divide a huge data file, but for standalone usage, the subsplits parameter below serves that purpose instead.
- [int] Ignored - This parameter indicates the number of distributed data splits, which in a standalone scenario is obviously fixed at 1; this parameter isn't even accessed by the standalone process.
- [int] *Subsplit size in rows*. This parameter provides a memory-management technique whereby only this many rows are ingested and processed at a time. For small machines I usually set this to 2,000,000.
- [int] Ignored - This parameter indicates the number of subsplits to divide the input file into. It only applies in distributed scenarios, not standalone usage.

Note that you need to know your data size in both bytes and rows (or annotation-count) in order to build a spatial index. These values are crucial to the sharding calculations.

Another parameter that can be helpful to provide is the `volume_bounds` of your data. Notice that parameter documented for the data config json above. It can be optionally assigned a null or empty list value, but omitting it will significantly increase the processing time of the overall pipeline. Note that when it is calculated, the volume bounds is written to the logging output. So, one option is to copy it from the output and paste it into your data config after your initial run so that subsequent runs don't have to calculate it again. You might not need to generate an index or the same input data, but in the event of any other issues that force a subsequent attempt, copying and pasting the volume bounds from the initial run into the data config will expedite those later runs.

A basic Python script to build a spatial index then looks something like the following:
```
import capsule_generate_config.capsule_generate_config as gc
import capsule_build_spatial_index_standalone.capsule_build_spatial_index_standalone as bsis

if __name__ == "__main__":
    data_loc = "../data/"
    results_loc = "../results/"

    input_filepath = <full path to an input CSV or Parquet file of annotations>
    
    data_config = gc.read_data_config(data_loc)
    bsis.run(data_loc, results_loc, data_config, input_filepath)
```

If you are running this in a capsule in Code Ocean, then the `input_filepath` probably begins with `../data/`, just like the indicated `data_loc` variable.

Please contact Keith Wiley (keith.wiley@alleninstitute.org) for assistance.
