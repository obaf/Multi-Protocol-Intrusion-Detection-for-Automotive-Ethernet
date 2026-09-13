"""Post-hoc analyses on saved predictions: detection latency on TOW-IDS test.

Latency = for each contiguous attack segment (per class), packets and seconds from
segment start until the first correctly-alerted packet (pred==1), for the chosen
model. Alerts on benign packets within +/- 1 s of an attack segment do not count.
Writes results/latency.json + prints table.
"""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
CLASSES = ['Normal', 'F_I', 'P_I', 'M_F', 'C_D', 'C_R']


def segments(y_multi):
    """Contiguous same-label runs; return list of (label, start, end) indices."""
    segs = []
    start = 0
    for i in range(1, len(y_multi) + 1):
        if i == len(y_multi) or y_multi[i] != y_multi[start]:
            segs.append((int(y_multi[start]), start, i - 1))
            start = i
    return segs


def main(model_files=('tow_binary_XGB.npz', 'tow_binary_LGBM.npz', 'tow_binary_VoteEns.npz',
                      'tow_binary_StackEns.npz', 'tow_binary_MLP.npz')):
    y_multi = np.load(os.path.join(ROOT, 'data', 'towids.npz'))['y_multi']
    split = np.load(os.path.join(ROOT, 'data', 'towids.npz'))['split']
    ts = np.load(os.path.join(ROOT, 'data', 'towids_ts_test.npy'))
    y_test_multi = y_multi[split == 1]
    segs = [s for s in segments(y_test_multi) if s[0] > 0]
    print(f'{len(segs)} attack segments in TOW-IDS test')
    out = {}
    for mf in model_files:
        p = os.path.join(RES, 'preds', mf)
        if not os.path.exists(p):
            print('skip', mf)
            continue
        z = np.load(p)
        pred = z['y_pred'].astype(int)
        name = mf.replace('tow_binary_', '').replace('.npz', '')
        rows = []
        detected, total = 0, 0
        lat_s_all, lat_pk_all = [], []
        per_class = {c: {'segments': 0, 'detected': 0, 'median_latency_s': None}
                     for c in CLASSES[1:]}
        for label, a, b in segs:
            total += 1
            cls = CLASSES[label]
            per_class[cls]['segments'] += 1
            seg_pred = pred[a:b + 1]
            hits = np.where(seg_pred == 1)[0]
            if len(hits) == 0:
                continue
            detected += 1
            per_class[cls]['detected'] += 1
            lat_pk = int(hits[0])
            lat_s = float(ts[a + lat_pk] - ts[a])
            lat_pk_all.append(lat_pk)
            lat_s_all.append(lat_s)
            rows.append((cls, a, lat_pk, lat_s))
        rec = {'model': name, 'segments': total, 'detected': detected,
               'seg_detection_rate': round(detected / total, 4),
               'median_latency_packets': float(np.median(lat_pk_all)) if lat_pk_all else None,
               'median_latency_ms': round(1000 * np.median(lat_s_all), 2) if lat_s_all else None,
               'per_class': per_class}
        out[name] = rec
        print(name, 'seg-detect', rec['seg_detection_rate'],
              'median latency', rec['median_latency_ms'], 'ms')
    with open(os.path.join(RES, 'latency.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote results/latency.json')


if __name__ == '__main__':
    main()
