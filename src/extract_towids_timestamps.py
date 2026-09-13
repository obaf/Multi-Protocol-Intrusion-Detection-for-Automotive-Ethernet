"""Extract per-packet capture timestamps from the TOW-IDS pcaps.

analysis_latency.py converts detection latency from packets to milliseconds using
these timestamps. build_towids.py does not keep them, so run this once after the
raw data are in place:

    python src/extract_towids_timestamps.py

Output: data/towids_ts_train.npy and data/towids_ts_test.npy (float64 seconds,
one value per labelled packet, in capture order).
"""
import os
import sys
import numpy as np
import dpkt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_towids import PCAPS, LABELS, DATA, read_labels  # noqa: E402

OUT_NAMES = {0: 'towids_ts_train.npy', 1: 'towids_ts_test.npy'}


def main():
    for split_id in (0, 1):
        n_labels = len(read_labels(LABELS[split_id]))
        ts = []
        with open(PCAPS[split_id], 'rb') as f:
            for t, _buf in dpkt.pcap.Reader(f):
                if len(ts) >= n_labels:
                    print(f'WARNING: more packets than labels ({n_labels}); truncating')
                    break
                ts.append(t)
        assert len(ts) == n_labels, f'packet/label mismatch: {len(ts)} vs {n_labels}'
        out = os.path.join(DATA, OUT_NAMES[split_id])
        np.save(out, np.asarray(ts, dtype=np.float64))
        print(f'saved {out} ({len(ts)} timestamps, {ts[-1] - ts[0]:.1f} s of capture)')


if __name__ == '__main__':
    main()
