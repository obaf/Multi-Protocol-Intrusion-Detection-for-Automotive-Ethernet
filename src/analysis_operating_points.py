"""Operating-point analysis: FPR at fixed attack-recall targets (99%, 95%, 90%)
for binary TOW-IDS and SOME/IP predictions saved by the experiment runner.

Writes results/operating_points.json
"""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

JOBS = ['tow_binary_XGB.npz', 'tow_binary_LGBM.npz', 'tow_binary_VoteEns.npz',
        'tow_binary_StackEns.npz', 'tow_binary_RF.npz', 'tow_binary_MLP.npz',
        'someip_binary_XGB.npz', 'someip_binary_LGBM.npz', 'someip_binary_VoteEns.npz',
        'someip_binary_StackEns.npz', 'someip_binary_MLP.npz']


def main():
    out = {}
    for j in JOBS:
        p = os.path.join(RES, 'preds', j)
        if not os.path.exists(p):
            continue
        z = np.load(p)
        y, score = z['y_true'].astype(int), z['score'].astype(np.float32)
        if score.ndim != 1 or len(score) != len(y):
            print('skip (no score)', j)
            continue
        pos = score[y == 1]
        neg = score[y == 0]
        rec = {}
        for target in (0.99, 0.95, 0.90):
            thr = np.quantile(pos, 1 - target)      # threshold achieving >= target recall
            fpr = float((neg >= thr).mean())
            rec[f'recall_{int(target*100)}'] = round(fpr, 6)
        out[j.replace('.npz', '')] = rec
        print(j, rec)
    with open(os.path.join(RES, 'operating_points.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote results/operating_points.json')


if __name__ == '__main__':
    main()
