import argparse

import capsule_build_spatial_index_standalone as bsis

if __name__ == "__main__":
    data_loc = "../data/"
    results_loc = "../results/"

    os.makedirs(results_loc, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Standalone spatial index builder (single CSV -> precomputed shards)")
    parser.add_argument(
        '--input_file', dest='input_file', default=None,
        help="Path to the input CSV file.  If not given, a file matching "
             "*split-001@1*.csv must already be present in ../data/")
    parser.add_argument('--capsule', dest='capsule', default=None)
    parser.add_argument('--config_override', dest='config_override', default=None)
    args, _ = parser.parse_known_args()
    
    data_config = gc.read_data_config(data_loc)
    run(data_loc, results_loc, data_config, args.input_file)
