import logging
from timeit import default_timer
from collections import Counter

from util import *

profiling_enabled = True

# TODO: Is this list/set pair approach faster or slower than using an OrderedDict with empty values?
profile_labels = []  # Preserve insertion order
profile_labels_set = set()  # Enable constant query time
timestamps = []
timestamp_usage_counts = Counter()
elap_accum_times = Counter()

def enable_disable_profiling(enabled):
    global profiling_enabled
    profiling_enabled = enabled

def reset_nested_profiler():
    global profile_labels, profile_labels_set, timestamps, elap_accum_times

    if not profiling_enabled:
        return

    profile_labels = []  # Preserve insertion order
    profile_labels_set = set()  # Enable constant query time
    timestamps = []
    elap_accum_times = Counter()

def start_timeblock(label):
    global profile_labels, profile_labels_set, timestamps

    if not profiling_enabled:
        return
    
    # depth_label = '| ' * len(timestamps) + label
    depth_label = (timestamps[-1][0] if timestamps else "") + '|' + label
    if depth_label not in profile_labels_set:
        profile_labels.append(depth_label)
        profile_labels_set.add(depth_label)
    timestamps.append((depth_label, default_timer()))
    timestamp_usage_counts[depth_label] += 1

def end_timeblock(label_UNUSED):
    global timestamps, elap_accum_times

    if not profiling_enabled:
        return
    
    # assert timestamps[-1][0][(len(timestamps)-1)*2:] == label_UNUSED, f"{timestamps[-1][0][(len(timestamps)-1)*2:]} != {label_UNUSED}"
    assert timestamps[-1][0].split('|')[-1] == label_UNUSED, f"{timestamps[-1][0].split('|')[-1]} != {label_UNUSED}"
    elap_accum_times[timestamps[-1][0]] += default_timer() - timestamps[-1][1]
    timestamps = timestamps[:-1]

def start_start_timeblocks(label1, label2):
    """
        The top of a loop or a function may involve pushing two labels onto the timeblock stack.
        This can be done more efficiently than two separate calls to start_timeblock().
    """
    global profile_labels, profile_labels_set, timestamps

    if not profiling_enabled:
        return

    t = default_timer()

    # From start_timeblock()
    # depth_label = '| ' * len(timestamps) + label1
    depth_label = (timestamps[-1][0] if timestamps else "") + '|' + label1
    if depth_label not in profile_labels_set:
        profile_labels.append(depth_label)
        profile_labels_set.add(depth_label)
    timestamps.append((depth_label, t))
    timestamp_usage_counts[depth_label] += 1

    # From start_timeblock()
    # depth_label = '| ' * len(timestamps) + label2
    depth_label = (timestamps[-1][0] if timestamps else "") + '|' + label2
    if depth_label not in profile_labels_set:
        profile_labels.append(depth_label)
        profile_labels_set.add(depth_label)
    timestamps.append((depth_label, t))
    timestamp_usage_counts[depth_label] += 1

def end_end_timeblocks(label1_UNUSED, label2_UNUSED):
    """
        The bottom of a loop or a function may involve popping two labels off the timeblock stack.
        This can be done more efficiently than two separate calls to end_timeblock().
    """
    global timestamps, elap_accum_times

    if not profiling_enabled:
        return
    
    t = default_timer()

    # From end_timeblock()
    # assert timestamps[-1][0][(len(timestamps)-1)*2:] == label1_UNUSED, f"{timestamps[-1][0][(len(timestamps)-1)*2:]} != {label1_UNUSED}"
    assert timestamps[-1][0].split('|')[-1] == label1_UNUSED, f"{timestamps[-1][0].split('|')[-1]} != {label1_UNUSED}"
    elap_accum_times[timestamps[-1][0]] += t - timestamps[-1][1]
    # assert timestamps[-2][0][(len(timestamps)-2)*2:] == label2_UNUSED, f"{timestamps[-2][0][(len(timestamps)-2)*2:]} != {label2_UNUSED}"
    assert timestamps[-2][0].split('|')[-1] == label2_UNUSED, f"{timestamps[-2][0].split('|')[-1]} != {label2_UNUSED}"
    elap_accum_times[timestamps[-2][0]] += t - timestamps[-2][1]  # We have to look back two from the end since we didnt pop the stack yet
    timestamps = timestamps[:-2]

def end_start_timeblocks(label_prev_UNUSED, label_next):
    """
        The middle of a sequence of steps may involve popping one label off and immediately pushing another label onto the timeblock stack.
        This can be done more efficiently than two separate calls to end_timeblock() and start_timeblock().
    """
    global profile_labels, profile_labels_set, timestamps, elap_accum_times

    if not profiling_enabled:
        return
    
    t = default_timer()

    # From end_timeblock()
    # assert timestamps[-1][0][(len(timestamps)-1)*2:] == label_prev_UNUSED, f"{timestamps[-1][0][(len(timestamps)-1)*2:]} != {label_prev_UNUSED}"
    assert timestamps[-1][0].split('|')[-1] == label_prev_UNUSED, f"{timestamps[-1][0].split('|')[-1]} != {label_prev_UNUSED}"
    elap_accum_times[timestamps[-1][0]] += t - timestamps[-1][1]
    # timestamps = timestamps[:-1]

    # From start_timeblock()
    # depth_label = '| ' * (len(timestamps) - 1) + label_next  # We have to shorten the length by one since we didnt pop the stack yet
    depth_label = (timestamps[-2][0] if len(timestamps) > 1 else "") + '|' + label_next
    if depth_label not in profile_labels_set:
        profile_labels.append(depth_label)
        profile_labels_set.add(depth_label)
    # timestamps.append((depth_label, default_timer()))
    timestamps[-1] = (depth_label, t)
    timestamp_usage_counts[depth_label] += 1

def dump_profile(validate=True):
    global profile_labels, timestamps, elap_accum_times

    if not profiling_enabled:
        logging.error("Profiler disabled")
        return

    if validate and timestamps:
        logging.error(f"ERROR! Timestamps stack not fully popped! Final timestamps:")
        for ts in timestamps:
            logging.error(f"  {ts}")

    summed_subtimes = Counter()
    for label in profile_labels:
        elap_t = elap_accum_times[label]
        pcs = label.split('|')
        parent = '|'.join(pcs[:-1])
        summed_subtimes[parent] += elap_t
    
    max_elap_t = 0
    for label in profile_labels:
        elap_t = elap_accum_times[label]
        if elap_t > max_elap_t:
            max_elap_t = elap_t
    time_formatter_ftn = seconds_to_hms if max_elap_t >= 3600 else seconds_to_ms if max_elap_t >= 60 else seconds_to_s
    
    logging.error("\nElapsed accumulated times:")
    if max_elap_t >= 3600:
        time_formatter_ftn = seconds_to_hms
        logging.error("Captured      - Subcaptured   = Lost            Label")
    elif max_elap_t >= 60:
        time_formatter_ftn = seconds_to_ms
        logging.error("Captured  - Subcaptur = Lost        Label")
    else:
        time_formatter_ftn = seconds_to_s
        logging.error("Captur - Subcap = Lost     Label")
    
    for label in profile_labels:
        elap_t = elap_accum_times[label]
        summed_subtime = summed_subtimes[label] if label in summed_subtimes else None
        lost_time = max(elap_t - summed_subtime, 0) if summed_subtime is not None else None
        label_simple = ""
        pcs = label.split('|')
        for i in range(len(pcs)-2):
            label_simple += '| '
        label_simple += pcs[-1]
        label_simple += f"    < {timestamp_usage_counts[label]:,} >"
        if summed_subtime:
            logging.error(f"{time_formatter_ftn(elap_t, True)}   {time_formatter_ftn(summed_subtime, True)}   {time_formatter_ftn(lost_time, True)}   {label_simple}")
        elif max_elap_t >= 3600:
            logging.error(f"{time_formatter_ftn(elap_t, True)}                                   {label_simple}")
        elif max_elap_t >= 60:
            logging.error(f"{time_formatter_ftn(elap_t, True)}                           {label_simple}")
        else:
            logging.error(f"{time_formatter_ftn(elap_t, True)}                     {label_simple}")

def get_profile(include_times=True):
    global profile_labels, elap_accum_times

    if not profiling_enabled:
        logging.error("Profiler disabled")
        return ""

    s = ""
    for label in profile_labels:
        elap_t = elap_accum_times[label]
        label_simple = ""
        pcs = label.split('|')
        for i in range(len(pcs)-2):
            label_simple += '| '
        label_simple += pcs[-1]
        if include_times:
            s += f"{seconds_to_hms(elap_t)} {label_simple}\n"
        else:
            s += f"{label_simple}\n"
    return s

def test_nested_profiler():
    logging.info("Beginning nested profiler correctness tests...\n")

    # ====================================================================================================

    reset_nested_profiler()
    start_timeblock("aaa")

    for i in range(1000):
        for i in range(1000):
            _ = 0
            _ = 1

    end_timeblock("aaa")

    logging.info(get_profile())
    profile = get_profile(False)
    assert profile == """aaa
"""

    # ====================================================================================================

    reset_nested_profiler()
    start_timeblock("aaa")

    for i in range(1000):
        start_timeblock("bbb")
        for i in range(1000):
            _ = 0
            _ = 1
        end_timeblock("bbb")

    end_timeblock("aaa")

    logging.info(get_profile())
    profile = get_profile(False)
    assert profile == """aaa
| bbb
"""

    # ====================================================================================================

    reset_nested_profiler()
    start_timeblock("aaa")

    for i in range(1000):
        start_timeblock("bbb")
        for i in range(1000):
            start_timeblock("ccc")
            _ = 0
            _ = 1
            end_timeblock("ccc")
        end_timeblock("bbb")

    end_timeblock("aaa")

    logging.info(get_profile())
    profile = get_profile(False)
    assert profile == """aaa
| bbb
| | ccc
"""

    # ====================================================================================================

    reset_nested_profiler()
    start_timeblock("aaa")

    for i in range(1000):
        start_timeblock("bbb")
        for i in range(1000):
            start_timeblock("ccc")
            _ = 0
            end_timeblock("ccc")
            start_timeblock("ddd")
            _ = 1
            end_timeblock("ddd")
        end_timeblock("bbb")

    end_timeblock("aaa")

    logging.info(get_profile())
    profile = get_profile(False)
    assert profile == """aaa
| bbb
| | ccc
| | ddd
"""

    # ====================================================================================================

    reset_nested_profiler()
    start_timeblock("aaa")

    for i in range(1000):
        start_timeblock("bbb")
        for i in range(1000):
            start_timeblock("ccc")
            _ = 0
            end_start_timeblocks("ccc", "ddd")
            _ = 1
            end_timeblock("ddd")
        end_timeblock("bbb")

    end_timeblock("aaa")

    logging.info(get_profile())
    profile = get_profile(False)
    assert profile == """aaa
| bbb
| | ccc
| | ddd
"""

    # ====================================================================================================

    reset_nested_profiler()
    start_timeblock("aaa")

    for i in range(1000):
        start_timeblock("bbb")
        for i in range(1000):
            start_start_timeblocks("ccc", "ddd")
            _ = 0
            end_start_timeblocks("ddd", "eee")
            _ = 1
            end_end_timeblocks("eee", "ccc")
        end_timeblock("bbb")

    end_timeblock("aaa")

    logging.info(get_profile())
    profile = get_profile(False)
    assert profile == """aaa
| bbb
| | ccc
| | | ddd
| | | eee
"""

    # ====================================================================================================

    logging.info("All nested profiler tests passed. Beginning speed runs...")

    # ====================================================================================================

    logging.info("\n(A) Testing start and end speed...")

    reset_nested_profiler()

    t0 = default_timer()
    for i in range(1000000):
        start_timeblock("aaaa")
        end_timeblock("aaaa")
    a_et = default_timer() - t0
    logging.info(a_et)

    # ====================================================================================================

    logging.info("\n(B) Testing start and end speed with long, deep labels (should be slower than A)...")

    reset_nested_profiler()

    t0 = default_timer()
    start_timeblock("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    start_timeblock("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    start_timeblock("cccccccccccccccccccccccccccccccccccccccccccccccccc")
    start_timeblock("dddddddddddddddddddddddddddddddddddddddddddddddddd")
    start_timeblock("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    for i in range(1000000):
        start_timeblock("ffffffffffffffffffffffffffffffffffffffffffffffffff")
        end_timeblock("ffffffffffffffffffffffffffffffffffffffffffffffffff")
    end_timeblock("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    end_timeblock("dddddddddddddddddddddddddddddddddddddddddddddddddd")
    end_timeblock("cccccccccccccccccccccccccccccccccccccccccccccccccc")
    end_timeblock("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    end_timeblock("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    b_et = default_timer() - t0
    logging.info(b_et)
    assert b_et > a_et

    # ====================================================================================================

    logging.info("\n(C) Testing end_start speed (should be faster than A)...")

    reset_nested_profiler()

    t0 = default_timer()
    start_timeblock("aaaa")
    for i in range(1000000):
        end_start_timeblocks("aaaa", "aaaa")
    end_timeblock("aaaa")
    c_et = default_timer() - t0
    logging.info(c_et)
    assert c_et < a_et

    # ====================================================================================================

    # I theorize that double-start and double-end might be slower than start and end
    # because it 

    logging.info("\n(D) Testing double-start and double-end speed (should be comparable to A)...")

    reset_nested_profiler()

    t0 = default_timer()
    for i in range(500000):
        start_timeblock("aa")
        start_timeblock("bb")
        end_timeblock("bb")
        end_timeblock("aa")
    d_et = default_timer() - t0
    logging.info(d_et)

    # ====================================================================================================

    logging.info("\n(E) Testing start_start and end_end speed (should be faster than D)...")

    reset_nested_profiler()

    t0 = default_timer()
    for i in range(500000):
        start_start_timeblocks("aa", "bb")
        end_end_timeblocks("bb", "aa")
    e_et = default_timer() - t0
    logging.info(e_et)
    assert e_et < d_et

    # ====================================================================================================

    logging.info("\nAll nested profiler speed runs complete and relative speed tests passed.")
    