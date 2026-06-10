{'AWS_BUCKET': 'pga-connectomics-wg-802451596237-us-west-2',
 'AWS_PROJECT_PATH': 'from_workgroups/ng_precomputed_annotations_pipeline_scratch',
 'DATA_CONFIG': {'columns': ['id',
                             'pre_pt_position_x',
                             'pre_pt_position_y',
                             'pre_pt_position_z',
                             'ctr_pt_position_x',
                             'ctr_pt_position_y',
                             'ctr_pt_position_z',
                             'post_pt_position_x',
                             'post_pt_position_y',
                             'post_pt_position_z',
                             'size',
                             'pre_pt_supervoxel_id',
                             'pre_pt_root_id',
                             'post_pt_supervoxel_id',
                             'post_pt_root_id'],
                 'data_label': 'MICRoNS minnie65_phase3_v1 '
                               'synapses_pni_2_v1_filtered_view',
                 'data_size': ['synapses_pni_2_v1_filtered_view__v1412__test1',
                               313245,
                               2108,
                               6000000,
                               1,
                               2000000,
                               3],
                 'data_sizes': {'docstring': 'A tuple indicating: (1) Num '
                                             'bytes and (2) num records (i.e. '
                                             'table/CSV rows, etc.) in the '
                                             'data. Multiple data sizes are '
                                             'supported via Code Ocean data '
                                             'source names to facilitate '
                                             'development/debugging against '
                                             'smaller subsets of the data.',
                                'synapses_pni_2_v1_filtered_view__v1412': [50345534259,
                                                                           337312429],
                                'synapses_pni_2_v1_filtered_view__v1412__test1': [313245,
                                                                                  2108],
                                'synapses_pni_2_v1_filtered_view__v1412__test2': [3645906,
                                                                                  24426],
                                'synapses_pni_2_v1_filtered_view__v1412__test3': [37056517,
                                                                                  247392],
                                'synapses_pni_2_v1_filtered_view__v1412__test4': [382826433,
                                                                                  2558733],
                                'synapses_pni_2_v1_filtered_view__v1412__test5': [3833155147,
                                                                                  25632771],
                                'synapses_pni_2_v1_filtered_view__v1412__test6': [15550001077,
                                                                                  103980993]},
                 'data_source_name': 'synapses_pni_2_v1_filtered_view__v1412__test1',
                 'data_version': 'v1412',
                 'dimensions': {'x': [4, 'nm'],
                                'y': [4, 'nm'],
                                'z': [40, 'nm']},
                 'docstring': 'This is a 337 million synapse dataset for v1412 '
                              'of the minnie65_phase3_v1 dataset.\n'
                              '\n'
                              'The volume bounds were calculated as min/max '
                              'from the 337 million row table/file. Note that '
                              'ctr column produced both the min and max '
                              'bounds, obviating pre and post columns for this '
                              'calculation. The original 20GB CSV export from '
                              'the DB represents the relevant columns '
                              '(pre_pt_position, ctr_pt_position, '
                              'post_pt_position) in voxels. That they are '
                              'voxels is easy to see because the Z axis values '
                              'is much smaller than X and Y, because the '
                              'dimensions are 4/4/40. To convert the data from '
                              'voxels to nm, multiply by the dimensions. To '
                              'convert from nm to microns (to apply sensible '
                              'bounds on expected synapses/µm^3 (.5syn/µm^3 as '
                              'offered by Forrest) or on how deep to expect '
                              'the oct tree to subdivide (7 or 8)), divide by '
                              '1000.',
                 'id_column': 'id',
                 'line_annotation_config': {'docstring': 'The values specified '
                                                         'in this object for '
                                                         'pt_column_labels '
                                                         'must indicate names '
                                                         'in the '
                                                         "'spatial_pt_columns' "
                                                         'object.',
                                            'end_pt_column_label': 'post',
                                            'start_pt_column_label': 'pre'},
                 'pipeline_config': {'docstring': 'All settings in this '
                                                  'section are optional. This '
                                                  'section can be completely '
                                                  'empty if you wish. However, '
                                                  'if you need to override any '
                                                  'default pipeline '
                                                  'configuration settings, '
                                                  'they go here. See the '
                                                  'pipeline documentation for '
                                                  'a description of such '
                                                  'options.'},
                 'properties': {'synapse_size': {'enum_labels': None,
                                                 'enum_values': None,
                                                 'id': 'size',
                                                 'type': 'int32'}},
                 'relations': {'Postsynaptic Cell': {'id': 'post_pt_root_id'},
                               'Presynaptic Cell': {'id': 'pre_pt_root_id'}},
                 'spatial_limit': {'docstring': 'Assume a limit of .5 synapses '
                                                'per cubic micron',
                                   'max_annotations_per_cubic_micron': 0.5},
                 'spatial_pt_columns': {'post': {'x': 'post_pt_position_x',
                                                 'y': 'post_pt_position_y',
                                                 'z': 'post_pt_position_z'},
                                        'pre': {'x': 'pre_pt_position_x',
                                                'y': 'pre_pt_position_y',
                                                'z': 'pre_pt_position_z'}},
                 'structure': 'one_annotation_per_row__multiple_points_per_row',
                 'volume_bounds': [[52708, 64831, 14838],
                                   [427816, 311022, 27868]]},
 'FORCE_NO_CACHE': 345434,
 'GCP_BUCKET': 'keith-dev',
 'GCP_RESULTS_BLOB_PATH': 'ng_precomputed_annotations_unreleased',
 'GCP_SCRATCH_BLOB_PATH': 'ng_precomputed_annotations_pipeline_scratch',
 'LIMIT': None,
 'LOGGING_FORMAT': '%(message)s',
 'LOGGING_LEVEL': 'debug',
 'MINISHARD_TARGET_COUNT': 1000,
 'NEUROGLANCER_URI': 'gs://keith-dev/ng_precomputed_annotations_unreleased/9999-01-01_01-01-01_UTC/',
 'NUM_SHARD_WORKERS': 32,
 'SHARD_TARGET_SIZE': 50000000,
 'SPLIT_DESC': 6000000,
 'SUBSPLIT_DESC': -3,
 'TIMESTAMP': '9999-01-01_01-01-01_UTC',
 'UPLOAD_RESULTS_TO_GCP': True}
