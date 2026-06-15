from shared.sharding_spec_calculations import *

import shared.annotations as anno

def test_sharding():
    sharding_spec = generate_sharding_spec(
        337312429,
        50345534259,
        "identity_hash",  # hashtype: Literal["murmurhash3_x86_128", "identity_hash"] = "identity_hash",
        MINISHARD_TARGET_COUNT=1000,
        SHARD_TARGET_SIZE=50000000,
    )
    print(f"sharding_spec: {sharding_spec}")

    # for tree_level in range(4):
    #     print("#" * 100)
    #     print("tree level   cell id  morton code  shard bits  shard range  shard num  shard hex")
    #     dim = 2 ** tree_level
    #     row = 0
    #     for x in range(dim):
    #         for y in range(dim):
    #             for z in range(dim):
    #                 if row % 10 == 9:
    #                     logging.info("tree level   cell id  morton code  shard bits  shard range  shard num  shard hex")
    #                 shard_hex = get_shard_hex(tree_level, [x, y, z])
    #                 row += 1
    
    sharding_spec = anno.ShardingSpec(preshift_bits=sharding_spec["preshift_bits"], shard_bits=sharding_spec["shard_bits"], minishard_bits=sharding_spec["minishard_bits"])
    for morton_code in range(100000):
        shard_num = sharding_spec.get_shard_number(morton_code)
        if shard_num != 0:
            print(f"{morton_code:>5} {shard_num}")
    
    print("Test sharding done")
