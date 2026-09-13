"""Metrics + sequence windowing utilities shared by all experiments."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


def classification_metrics(y_true, proba, class_names=None):
    """Binary (positive=attack) and multiclass metric bundle from proba [N, C]."""
    y_pred = proba.argmax(axis=1)
    out = {
        'acc': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    n_classes = proba.shape[1]
    if n_classes == 2:
        out['pr_auc'] = average_precision_score(y_true, proba[:, 1])
        out['roc_auc'] = roc_auc_score(y_true, proba[:, 1])
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        out['fpr'] = float(fp) / max(1, fp + tn)
    else:
        # one-vs-rest macro PR-AUC / ROC-AUC computed per present class (robust
        # to label sets that are a subset of the model's output classes)
        n_present = int(y_true.max()) + 1
        p = proba[:, :n_present] if proba.shape[1] > n_present else proba
        aps, aucs = [], []
        for c in np.unique(y_true):
            ybin = (y_true == c).astype(int)
            if ybin.sum() == 0 or ybin.sum() == len(ybin):
                continue
            aps.append(average_precision_score(ybin, p[:, c]))
            aucs.append(roc_auc_score(ybin, p[:, c]))
        out['pr_auc'] = float(np.mean(aps)) if aps else float('nan')
        out['roc_auc'] = float(np.mean(aucs)) if aucs else float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    out['confusion'] = cm.tolist()
    if class_names is not None:
        n_named = min(n_classes, len(class_names))
        recalls = {class_names[i]: (cm[i, i] / cm[i].sum() if cm[i].sum() else float('nan'))
                   for i in range(n_named)}
        out['per_class_recall'] = recalls
    return out


def make_windows(X, y, group, seq_len=16, stride=8, max_windows=None, rng=None):
    """Sliding windows over consecutive packets within a capture group.

    X: [N, F], y: [N], group: [N] capture/stream id.
    Window label = 1 if ANY packet in the window is an attack (as in ref protocol).
    Returns Xw [W, L, F], yw [W], last_idx [W] (index of the window's last packet,
    used to re-align window predictions to packets).
    """
    Xw, yw, last_idx = [], [], []
    for g in np.unique(group):
        m = np.where(group == g)[0]
        Xg, yg = X[m], y[m]
        for i in range(0, len(m) - seq_len + 1, stride):
            Xw.append(Xg[i:i + seq_len])
            yw.append(int(yg[i:i + seq_len].max() > 0))
            last_idx.append(m[i + seq_len - 1])
    Xw = np.stack(Xw) if Xw else np.empty((0, seq_len, X.shape[1]), dtype=X.dtype)
    yw = np.asarray(yw)
    last_idx = np.asarray(last_idx)
    if max_windows is not None and len(yw) > max_windows:
        rng = rng or np.random.default_rng(42)
        keep = rng.choice(len(yw), size=max_windows, replace=False)
        keep.sort()
        Xw, yw, last_idx = Xw[keep], yw[keep], last_idx[keep]
    return Xw, yw, last_idx


def aggregate_results(rows, path_csv):
    import csv, os
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path_csv} ({len(rows)} rows)')
