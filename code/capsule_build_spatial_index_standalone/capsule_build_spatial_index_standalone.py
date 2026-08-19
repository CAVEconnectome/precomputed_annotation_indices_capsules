"""
Standalone spatial index builder.

Combines the functionality of three distributed pipeline capsules into a single local module:
  - capsule_build_spatial_index.py        (mapper: builds the oct-tree)
  - capsule_generate_spatial_index_shards.py (reducer grouper: reorganizes by shard)
  - finalize_annotations.py              (reducer finalizer: writes precomputed shard files)

Input:
  A single CSV file placed in ../data/ whose filename contains 'split-001@1'
  (e.g. my_data_split-001@1.csv).  If --input_file is given on the command line,
  the file is copied to ../data/ with that naming convention automatically.

  A job_config.py (and optional job_spatial_config.py, job_id_config.py, etc.)
  must also be present in ../data/ as usual.

Output:
  Precomputed shard files written to:
    ../results/spatial{tree_level}/{shard_hex}.shard

The entire CSV is treated as one split and one subsplit, so no archiving,
no GCP/AWS transfers, and no shard-worker coordination are performed.
All intermediate data lives in RAM (via RAMDataPond).
"""

import sys
import logging
import os
import pprint
import random
import shutil
import json
from collections import defaultdict, Counter
from timeit import default_timer

from shared.sharding_spec_calculations import *
import shared.annotations as anno
import shared.sharding as sharding
import shared.utilities as utilities

from shared.util import *
from shared.nested_profiler import *
from shared.ram_data_pond import *

# Import existing capsule modules so we can reuse their functions without duplicating logic.
# Their module-level globals (data_loc, results_loc, config, ram_data_pond, …) are patched
# just before any of their functions are called.
import capsule_generate_config.capsule_generate_config as gc
import capsule_generate_spatial_index_config.capsule_generate_spatial_index_config as gsic
import capsule_build_spatial_index.capsule_build_spatial_index as bsi
import capsule_generate_spatial_index_shards.finalize_annotations as fa
import capsule_reorganize_directory_structure.capsule_reorganize_directory_structure as rds


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _deep_dictionary_override(default: dict, override: dict, parent_keys=[]) -> dict:
    result = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_dictionary_override(result[key], value, parent_keys+[key])
        else:
            if key in default:
                logging.info(f"Overriding default pipeline config '{".".join(parent_keys)}{"." if parent_keys else ""}{key}' = {default[key]} with {value}")
            else:
                logging.info(f"Adding pipeline config '{".".join(parent_keys)}{"." if parent_keys else ""}{key}' = {value}")
            result[key] = value
    return result

def _prepare_input_file(input_csv_path, data_loc):
    """
    If the caller supplied an explicit CSV path, ensure a copy with the required
    'split-001@1' naming convention exists inside data_loc.  Returns the path
    that will be found by the glob inside the tree builder.
    """
    filename = os.path.basename(input_csv_path)
    # Check whether the file already carries the required split tag.
    if 'split-' in filename and '@' in filename:
        target = os.path.join(data_loc, filename)
    else:
        target = os.path.join(data_loc, 'input_split-001@1.csv')

    if os.path.abspath(input_csv_path) != os.path.abspath(target):
        logging.info(f"Copying input CSV to data_loc:\n  {input_csv_path}\n  -> {target}")
        shutil.copy(input_csv_path, target)
    return target


def _run_tree_building(data_loc, results_loc, config, ram_data_pond):
    """
    Run the oct-tree building phase.

    This replicates the loop from bsi.process_subsplit() but skips
    archive_results() entirely — we read the results straight from the
    RAM data pond instead of serialising them to disk archives.

    Returns split_id (int) on success, or None if no data was found.
    """
    # Patch all module-level globals that bsi functions read.
    bsi.data_loc = data_loc
    bsi.results_loc = results_loc
    bsi.config = config
    bsi.ram_data_pond = ram_data_pond
    # These are only used by archive_results() / collate_subsplit_archives(),
    # which we deliberately skip, so empty collections are fine.
    bsi.shard_worker_descs = set()
    bsi.shard_worker_lookup = {}

    split_id = None
    src_loc = data_loc
    treelevel_iter = -1
    tree_level_shard_histograms = []
    process_next_tree_level = True

    while process_next_tree_level:
        treelevel_iter += 1
        logging.info(f"\n{'*' * 80}\nTree-level iteration {treelevel_iter}  src_loc={src_loc}\n")
        tree_level_shard_histograms.append(Counter())

        # subsplit_id=1, row_start=0, row_end=None → process the entire file
        result = bsi.process_one_tree_level(
            1, 0, None, treelevel_iter, src_loc, tree_level_shard_histograms)

        if result is None:
            logging.info("process_one_tree_level returned None — no data or subsplits exhausted.")
            break

        split_id, num_splits, src_loc, process_next_tree_level = result
        if split_id is None:
            break

    return split_id


def _extract_treecell_csv_with_index(results_loc, ram_data_pond):
    """
    Walk the completed tree cells stored in the RAM data pond, convert each to
    a CSV string, and append the treecell_index as the last field of every row.

    Returns a dict mapping (tree_level: int, shard_hex: str) -> merged CSV string.
    """
    logging.info("\nExtracting completed tree cell data from RAM data pond …")

    # Glob all objects whose key matches the completed-treecell pattern.
    # bsi stores them under keys like:
    #   {results_loc}completed_treecells/
    #     annotations_one_treecell__subsplit-01__split-001@1__treelevel-LL__treelevelcellid-CCC__shard-HHH/
    #       annotations_one_treecell__subsplit-01__split-001@1__treelevel-LL__treelevelcellid-CCC__shard-HHH
    all_keys = ram_data_pond.fastglob_ram_data_pond(
        f"{results_loc}completed_treecells/"
        f"annotations_one_treecell*/annotations_one_treecell__subsplit-01__*")

    logging.info(f"Found {len(all_keys)} completed-tree-cell objects in RAM data pond")

    treelevel_shard_csv = defaultdict(str)

    for key in all_keys:
        filename = os.path.basename(key)

        # Determine whether this entry is a raw-CSV string or an Annotation-object list.
        is_csv_entry = filename.endswith('.csv')
        name_base = filename[:-4] if is_csv_entry else filename

        # Parse components from the filename.
        # Format: annotations_one_treecell__subsplit-SS__split-NNN@M__treelevel-LL__treelevelcellid-A,B,C__shard-HHH
        pcs = name_base.split('__')
        if len(pcs) < 6:
            logging.warning(f"Unexpected filename format, skipping: {filename}")
            continue

        try:
            tree_level = int(pcs[3].split('-')[1])
            cell_id_str = pcs[4].split('-')[1]       # e.g. "000,000,000"
            treecell_index = '_'.join(            # e.g. "0_0_0"
                str(int(v)) for v in cell_id_str.split(','))
            shard_hex = pcs[5].split('-')[1]
        except (IndexError, ValueError) as exc:
            logging.warning(f"Could not parse key {key}: {exc}")
            continue

        # Retrieve the data.
        data = ram_data_pond.read_from_disk_or_ram_data_pond(key)

        if is_csv_entry:
            # Already a CSV string — split into lines.
            raw_lines = [ln for ln in data.strip().split('\n') if ln.strip()]
        elif isinstance(data, list):
            # List of Annotation subclass objects — use each object's original CSV row.
            raw_lines = [ann.raw_data.strip() for ann in data]
        else:
            logging.warning(f"Unexpected data type {type(data)} for key {key}, skipping")
            continue

        # Append the treecell_index as the final CSV field on every row.
        lines_with_index = [f"{ln},{treecell_index}" for ln in raw_lines if ln]
        treelevel_shard_csv[(tree_level, shard_hex)] += '\n'.join(lines_with_index) + '\n'

    logging.info(
        f"Grouped into {len(treelevel_shard_csv)} (tree_level, shard_hex) buckets")
    return treelevel_shard_csv


def _write_precomputed_shards(treelevel_shard_csv, data_loc, results_loc, config):
    """
    For each (tree_level, shard_hex) bucket, call the finalize_annotations logic
    to write a precomputed .shard file at:
        {results_loc}spatial{tree_level}/{shard_hex}.shard
    """
    logging.info("\nWriting precomputed shard files …")

    # Patch finalize_annotations globals.
    fa.data_loc = data_loc
    fa.results_loc = results_loc
    fa.config = config
    fa.timestamps = []
    fa.dimensions = config['DATA_CONFIG']['dimensions']
    fa.missing_enum_labels = set()

    columns = config['DATA_CONFIG']['columns'] + ['treecell_index']
    cell_bounds_low = config['DATA_CONFIG']['volume_bounds'][0]
    cell_bounds_high = config['DATA_CONFIG']['volume_bounds'][1]

    os.makedirs(results_loc, exist_ok=True)

    for (tree_level, shard_hex), merged_csv in sorted(treelevel_shard_csv.items()):
        logging.info(
            f"  Writing shard: tree_level={tree_level}  shard_hex={shard_hex}"
            f"  data_size={len(merged_csv):,} B")
        fa.save_shard_data_as_precomputed(
            merged_csv, columns,
            cell_bounds_low, cell_bounds_high,
            "",              # subdir — empty means write directly under results_loc
            tree_level, shard_hex)

    logging.info("All precomputed shard files written.")


def _write_finalization_files(treelevel_shard_csv, data_loc, results_loc, config):
    """
    Generate the Neuroglancer info file and pipeline_config.json using the logic
    from capsule_reorganize_directory_structure.

    ID and RELATION index processing is disabled because this standalone module
    only builds the spatial index.  Safe defaults are injected for any config keys
    that the id/relation config files would normally supply, so generate_info_files
    works correctly even when those optional configs were never loaded.
    """
    # Determine annotation type from DATA_CONFIG (same logic as reorganize capsule).
    data_config = config['DATA_CONFIG']
    if "point_annotation_config" in data_config:
        annotation_type = "POINT"
    elif "line_annotation_config" in data_config:
        annotation_type = "LINE"
    elif "polyline_annotation_config" in data_config:
        annotation_type = "POLYLINE"
    else:
        raise ValueError(
            "Cannot determine annotation type from DATA_CONFIG "
            "(expected point_annotation_config, line_annotation_config, "
            "or polyline_annotation_config)")

    # Derive the maximum tree level from the shards we actually wrote.
    max_tree_level = (
        max(tl for tl, _ in treelevel_shard_csv.keys())
        if treelevel_shard_csv else 0)
    logging.critical(f"Max tree level: {max_tree_level}")

    # Build a config view for the reorganize module with ID/RELATION indices off
    # and safe defaults for any keys those optional configs would have provided.
    cfg = dict(config)
    cfg['ID_INDEX_ENABLED'] = False
    cfg['RELATION_INDEX_ENABLED'] = False
    cfg.setdefault('ID_SHARDING', False)
    cfg.setdefault('RELATION_SHARDING', False)
    # Avoid mutating the caller's DATA_CONFIG when generate_info_files strips 'docstring'.
    cfg['DATA_CONFIG'] = dict(config['DATA_CONFIG'])
    cfg['DATA_CONFIG'].setdefault('relations', [])

    # Patch the reorganize module's globals before calling its functions.
    rds.config = cfg
    rds.data_loc = data_loc
    rds.results_loc = results_loc

    # Write the info file via the reorganize capsule's function.
    rds.generate_info_files(max_tree_level, annotation_type)

    # Write pipeline_config.json (json.dump may fail on non-serialisable values;
    # fall back to repr so we always produce something).
    pipeline_config_path = os.path.join(results_loc, "pipeline_config.json")
    try:
        with open(pipeline_config_path, 'w') as f:
            json.dump(cfg, f, indent=4)
    except (TypeError, ValueError) as exc:
        logging.warning(
            f"Config is not fully JSON-serialisable ({exc}); writing repr instead")
        with open(pipeline_config_path, 'w') as f:
            f.write(repr(cfg))
    logging.critical(f"Wrote pipeline_config.json to {pipeline_config_path}")


def generate_spatial_config(data_loc, config):
    """
    Replicates the logic from capsule_generate_spatial_index_config.__main__:
    applies any pipeline_spatial_config overrides from config, computes per-tree-level
    sharding specs, and writes the result to {data_loc}job_spatial_config.py so that
    a subsequent read_config(["id", "relation", "spatial"]) picks it up.

    In the distributed CodeOcean pipeline this capsule writes to its own results_loc,
    which becomes the data_loc of the next capsule.  Here we write directly to data_loc
    so the file is immediately available to the following read_config() call.
    """
    spatial_config = dict(gsic.spatial_config)  # work on an independent copy

    if 'pipeline_spatial_config' in config['DATA_CONFIG']:
        for k, v in config['DATA_CONFIG']['pipeline_spatial_config'].items():
            if k == 'docstring':
                continue
            logging.info(
                f"Overriding spatial pipeline config {k} = {spatial_config.get(k)} with {v}")
            spatial_config[k] = v
    logging.info("")

    sharding_spec = generate_sharding_spec(
        config['DATA_CONFIG']['data_size'][2],
        config['DATA_CONFIG']['data_size'][1],
        spatial_config['SPATIAL_SHARDING_HASH'],
        MINISHARD_TARGET_COUNT=config["MINISHARD_TARGET_COUNT"],
        SHARD_TARGET_SIZE=config["SHARD_TARGET_SIZE"],
    )
    logging.critical(
        f"Generated sharding_spec:\n{json.dumps(sharding_spec, indent=2)}\n"
        f"Will produce 2^{sharding_spec['shard_bits']} = "
        f"{2**sharding_spec['shard_bits']} shard files")

    if spatial_config['DEFAULT_SPATIAL_SHARDING_BITS'] is None:
        spatial_config['TREE_LEVEL_SHARDING_SPECS'] = [None] * spatial_config['MAX_NUM_TREE_LEVELS']
        for tree_level in range(spatial_config['MAX_NUM_TREE_LEVELS']):
            tree_level_sharding_spec = generate_spatial_index_sharding_spec_2(
                tree_level,
                spatial_config['MAX_DATA_ROWS_PER_TREE_CELL'],
                config['DATA_CONFIG']['data_size'],
                spatial_config['SPATIAL_SHARDING_HASH'],
                MINISHARD_TARGET_COUNT=config["MINISHARD_TARGET_COUNT"],
                SHARD_TARGET_SIZE=config["SHARD_TARGET_SIZE"],
            )
            spatial_config['TREE_LEVEL_SHARDING_SPECS'][tree_level] = tree_level_sharding_spec
    else:
        spatial_config['TREE_LEVEL_SHARDING_SPECS'] = [None] * spatial_config['MAX_NUM_TREE_LEVELS']
        for tree_level in range(spatial_config['MAX_NUM_TREE_LEVELS']):
            spatial_config['TREE_LEVEL_SHARDING_SPECS'][tree_level] = {
                "preshift_bits":  spatial_config["DEFAULT_SPATIAL_PRESHIFT_BITS"],
                "shard_bits":     spatial_config["DEFAULT_SPATIAL_SHARDING_BITS"],
                "minishard_bits": spatial_config["DEFAULT_SPATIAL_MINISHARDING_BITS"],
            }

    spatial_config = {k: v for k, v in spatial_config.items() if not k.startswith("DEBUG")}

    out_path = os.path.join(data_loc, "job_spatial_config.py")
    with open(out_path, 'w') as f:
        f.write(pprint.pformat(spatial_config, indent=2) + '\n')
    logging.critical(f"Wrote spatial config to {out_path}")

    return spatial_config


def build_spatial_index(data_loc, results_loc, config, input_csv_path=None):
    """
    End-to-end spatial index builder for a single CSV input.

    Parameters
    ----------
    data_loc : str
        Path to the data directory.  Must contain a job_config.py and a CSV
        file whose name includes 'split-001@1' (created automatically if
        input_csv_path is provided).
    results_loc : str
        Path to the results directory.  Precomputed shard files are written here.
    config : dict
        Pipeline configuration (typically loaded by read_config()).
    input_csv_path : str or None
        If given, this CSV file is copied into data_loc with the required
        naming convention before processing begins.
    """
    os.makedirs(data_loc, exist_ok=True)
    os.makedirs(results_loc, exist_ok=True)

    if input_csv_path is not None:
        _prepare_input_file(input_csv_path, data_loc)

    ram_data_pond = RAMDataPond(True)

    # Phase 1: build the oct-tree.
    split_id = _run_tree_building(data_loc, results_loc, config, ram_data_pond)
    if split_id is None:
        logging.warning("Tree building produced no output — is the input file empty or misconfigured?")
        return

    # Phase 2: extract and group completed tree cell data from the RAM data pond.
    treelevel_shard_csv = _extract_treecell_csv_with_index(results_loc, ram_data_pond)

    if not treelevel_shard_csv:
        logging.warning("No completed tree cell data found after tree building.")
        return

    # Phase 3: write precomputed .shard files.
    _write_precomputed_shards(treelevel_shard_csv, data_loc, results_loc, config)

    # Phase 4: write the Neuroglancer info file and pipeline_config.json.
    _write_finalization_files(treelevel_shard_csv, data_loc, results_loc, config)


def run(input_dir, output_dir, data_config, input_file):
    config = gc.init_config_w_data_config(data_config)

    logging.basicConfig(
        level=get_logging_level_from_desc(config['LOGGING_LEVEL']),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f"{output_dir}log_build_spatial_index_standalone_{logging_uid}.log",
                mode="a"),
        ],
        format=config['LOGGING_FORMAT'],
        force=True)
    
    os.makedirs(output_dir, exist_ok=True)

    # Generate job_spatial_config.py in input_dir so read_config can pick it up below.
    spatial_config = generate_spatial_config(input_dir, config)

    # read_config() reads the main config, then the sub-configs, then folds the sub-configs into the main config.
    # For this entrypoint however, we don't want to read, much less reread, config files, since the configuration is passed in.
    # So we will just fold the spatial sub-config in right here.
    # Note that this replicats that code, but it is a mere key/value copy-over, as proceeds below:
    for k, v in spatial_config.items():
        config[k] = v

    build_spatial_index(input_dir, output_dir, config, input_file)

    finalize_results(output_dir)
    process_running_time()
    dump_profile()

    logging.info("\nDone")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    data_loc = "../data/"
    results_loc = "../results/"

    logging_uid = hex(int(random.random() * 1000000000000))[2:]

    os.makedirs(results_loc, exist_ok=True)

    logging.basicConfig(
        level=logging.CRITICAL,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f"{results_loc}log_build_spatial_index_standalone_{logging_uid}.log",
                mode="a"),
        ],
        format='%(message)s')
    logging.critical("_" * 100)
    logging.critical("BUILD SPATIAL INDEX STANDALONE")

    parser = argparse.ArgumentParser(
        description="Standalone spatial index builder (single CSV -> precomputed shards)")
    parser.add_argument(
        '--input_file', dest='input_file', default=None,
        help="Path to the input CSV file.  If not given, a file matching "
             "*split-001@1*.csv must already be present in ../data/")
    parser.add_argument('--capsule', dest='capsule', default=None)
    parser.add_argument('--config_override', dest='config_override', default=None)
    args, _ = parser.parse_known_args()

    # Read base config (without spatial) to supply inputs to generate_spatial_config.
    config = gc.init_config_w_data_config_file(data_loc)
    data_config = gc.read_data_config()

    # Apply any App Panel overrides now so pipeline_spatial_config overrides are
    # available during sharding spec calculation.
    if args.config_override:
        config['DATA_CONFIG'] = _deep_dictionary_override(
            config['DATA_CONFIG'], json.loads(args.config_override))

    # Generate job_spatial_config.py in data_loc so read_config can pick it up below.
    spatial_config = generate_spatial_config(data_loc, config)

    # Now read the full config, which includes the spatial config we just generated.
    config = read_config(["id", "relation", "spatial"])

    # Re-apply App Panel overrides to the full config.
    if args.config_override:
        config['DATA_CONFIG'] = _deep_dictionary_override(
            config['DATA_CONFIG'], json.loads(args.config_override))

    logging.basicConfig(
        level=get_logging_level_from_desc(config['LOGGING_LEVEL']),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f"{results_loc}log_build_spatial_index_standalone_{logging_uid}.log",
                mode="a"),
        ],
        format=config['LOGGING_FORMAT'],
        force=True)

    build_spatial_index(data_loc, results_loc, config, args.input_file)

    finalize_results(results_loc)
    process_running_time()
    dump_profile()

    logging.info("\nDone")



    run(data_loc, results_loc, data_config, args.input_file)
