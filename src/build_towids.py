"""Build TOW-IDS (DCRL automotive Ethernet) feature arrays from pcaps + label CSVs.

Output: data/towids.npz with
  X        float32 [N, D]  static + context features
  y_bin    int8   [N]      0 normal / 1 attack
  y_multi  int8   [N]      0 Normal, 1 F_I, 2 P_I, 3 M_F, 4 C_D, 5 C_R
  ethertypes int16 [N]     inner ethertype
  split    int8   [N]      0 train / 1 test
"""
import sys, os, csv
import numpy as np
import dpkt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_common import extract_static_features, StreamContext, frame_ethertype

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PCAPS = {
    0: os.path.join(DATA, 'towids_extracted',
                    'Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap'),
    1: os.path.join(DATA, 'towids_extracted',
                    'Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap'),
}
LABELS = {
    0: os.path.join(DATA, 'towids_extracted', 'y_train.csv'),
    1: os.path.join(DATA, 'towids_extracted', 'y_test.csv'),
}
CLASS_TO_ID = {'Normal': 0, 'F_I': 1, 'P_I': 2, 'M_F': 3, 'C_D': 4, 'C_R': 5}


def read_labels(path):
    y = []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row or len(row) < 3:
                continue
            y.append(CLASS_TO_ID[row[2].strip()])
    return np.asarray(y, dtype=np.int8)


def build_split(split_id):
    labels = read_labels(LABELS[split_id])
    statics, contexts, ets = [], [], []
    ctx = StreamContext(window=100)
    n = 0
    with open(PCAPS[split_id], 'rb') as f:
        for ts, buf in dpkt.pcap.Reader(f):
            if n >= len(labels):
                print(f'WARNING: more packets ({n+1}) than labels ({len(labels)}); truncating')
                break
            et = frame_ethertype(buf)
            ctx.update(buf, et)
            statics.append(extract_static_features(buf))
            contexts.append(ctx.features())
            ets.append(et)
            n += 1
    assert n == len(labels), f'packet/label mismatch: {n} vs {len(labels)}'
    X = np.concatenate([np.stack(statics), np.stack(contexts)], axis=1)
    return X.astype(np.float32), labels, np.asarray(ets, dtype=np.int32)


def main():
    Xs, ys, ets, splits = [], [], [], []
    for split_id in (0, 1):
        X, y, et = build_split(split_id)
        print(f'split {split_id}: X={X.shape} labels={np.bincount(y.astype(int), minlength=6).tolist()}')
        Xs.append(X); ys.append(y); ets.append(et)
        splits.append(np.full(len(y), split_id, dtype=np.int8))
    X = np.concatenate(Xs); y = np.concatenate(ys); et = np.concatenate(ets); sp = np.concatenate(splits)
    np.savez_compressed(os.path.join(DATA, 'towids.npz'), X=X, y_multi=y,
                        y_bin=(y > 0).astype(np.int8), ethertypes=et, split=sp)
    print('saved data/towids.npz', X.shape)


if __name__ == '__main__':
    main()
