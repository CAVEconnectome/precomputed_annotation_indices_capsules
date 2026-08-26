"""
This folder (precomputed_annotation_indices/) and this init file enable the standalone module (or any other modules) to be imported via a single overarching import, ala:
    from precomputed_annotation_indices import *
If this folder and init file were not included, then each capsule in the code/ folder would have to be imported as a separate module, such as this example for the standalone scenario:
    import capsule_generate_config.capsule_generate_config as gc
    import capsule_build_spatial_index_standalone.capsule_build_spatial_index_standalone as bsis
"""

import capsule_generate_config.capsule_generate_config as gc
import capsule_build_spatial_index_standalone.capsule_build_spatial_index_standalone as bsis
from shared import util, nested_profiler, annotations
