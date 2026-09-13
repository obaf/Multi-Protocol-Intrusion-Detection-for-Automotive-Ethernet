"""Experiment runner for multi-protocol automotive Ethernet IDS.

Suites:
  R1  per-dataset benchmark (TOW-IDS and SOME/IP; binary + multiclass)
  R2  unified multi-protocol IDS (train on both datasets, evaluate per protocol)
  R3  leave-one-protocol-family-out zero-day transfer
  R4  deployment cost (model size, throughput)

All results -> results/*.csv + results/summary.json
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (build_tabular_models, build_torch_mlp, build_torch_sequence,
                    model_size_mb, torch_size_mb, inference_throughput)
from exp_utils import classification_metrics, make_windows, aggregate_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

TOW_CLASSES = ['Normal', 'F_I', 'P_I', 'M_F', 'C_D', 'C_R']
SOME_CLASSES = ['Normal', 'EoE', 'EoV', 'MissReq', 'MissResp']
UNI_CLASSES = TOW_CLASSES + SOME_CLASSES[1:]           # 10 classes

# protocol family of each multiclass label (for R3)
TOW_FAMILY = {1: 'AVTP', 2: 'gPTP', 3: 'SWITCH', 4: 'UDP-CAN', 5: 'UDP-CAN'}
SOME_FAMILY = {1: 'SOMEIP', 2: 'SOMEIP', 3: 'SOMEIP', 4: 'SOMEIP'}

TRAIN_CAP = 500_000     # cap training rows (stratified) for CPU feasibility
if '--smoke' in sys.argv:
    TRAIN_CAP = 40_000
    SEQ_CAP_TRAIN, SEQ_CAP_TEST = 8_000, 4_000
    SMOKE = True
else:
    SEQ_CAP_TRAIN, SEQ_CAP_TEST = 150_000, 120_000
    SMOKE = False


def cap_train_idx(y, cap=TRAIN_CAP, seed=42):
    """Stratified subsample indices keeping all minorities, capping the majority."""
    if len(y) <= cap:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    keep_per = {c: min(n, max(1, int(round(cap * n / total)))) for c, n in zip(classes, counts)}
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        if len(ci) > keep_per[c]:
            ci = rng.choice(ci, size=keep_per[c], replace=False)
        idx.append(ci)
    return np.sort(np.concatenate(idx))


def cap_train(X, y, cap=TRAIN_CAP, seed=42):
    idx = cap_train_idx(y, cap, seed)
    return X[idx], y[idx]


def load():
    tow = np.load(os.path.join(ROOT, 'data', 'towids.npz'), allow_pickle=True)
    sip = np.load(os.path.join(ROOT, 'data', 'someip.npz'), allow_pickle=True)
    return tow, sip


def build_tabular_models_filtered(n_classes):
    """In smoke mode, skip the expensive stacking ensemble."""
    models = build_tabular_models(n_classes)
    if SMOKE:
        models.pop('StackEns')
    return models


def eval_model(name, model, Xtr, ytr, Xte, yte, class_names, is_torch=False,
               Xval=None, yval=None, save_prefix='', extra=None):
    t0 = time.perf_counter()
    if is_torch:
        model.fit(Xtr, ytr, X_val=Xval, y_val=yval)
    else:
        model.fit(Xtr, ytr)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = model.predict_proba(Xte)
    test_s = time.perf_counter() - t0
    m = classification_metrics(yte, proba, class_names)
    m.update({'model': name, 'train_s': round(train_s, 1), 'test_s': round(test_s, 2),
              'n_train': len(ytr), 'n_test': len(yte)})
    m['size_mb'] = round(torch_size_mb(model) if is_torch else model_size_mb(model), 2)
    m['thr_kmsg_s'] = round(inference_throughput(model, Xte, is_torch) / 1000, 1)
    if extra:
        m.update(extra)
    print(json.dumps({k: v for k, v in m.items()
                      if k not in ('confusion', 'per_class_recall')}))
    # persist test predictions for post-hoc analysis (confusion figs, latency)
    try:
        os.makedirs(os.path.join(RES, 'preds'), exist_ok=True)
        task = (extra or {}).get('task', save_prefix or 'unk')
        np.savez_compressed(
            os.path.join(RES, 'preds', f'{task}_{name}.npz'),
            y_pred=proba.argmax(axis=1).astype(np.int8),
            y_true=np.asarray(yte, dtype=np.int8),
            score=proba[:, -1].astype(np.float16) if proba.shape[1] == 2 else np.zeros(1, np.float16))
    except Exception as e:
        print('pred save failed:', e)
    return m, proba


def suite_r1(tow, sip):
    """R1: per-dataset benchmark, binary + multiclass."""
    rows = []
    # ---- TOW-IDS ----
    tr, te = tow['split'] == 0, tow['split'] == 1
    for task, ycol, names in [('tow_binary', 'y_bin', None),
                              ('tow_multi', 'y_multi', TOW_CLASSES)]:
        ytr, yte = tow[ycol][tr], tow[ycol][te]
        Xtr, ytr_c = cap_train(tow['X'][tr], ytr)
        Xte = tow['X'][te]
        n_classes = len(np.unique(ytr))
        names = names or ['Normal', 'Attack']
        models = build_tabular_models_filtered(n_classes)
        for name, mdl in models.items():
            m, _ = eval_model(name, mdl, Xtr, ytr_c, Xte, yte, names,
                              save_prefix=task, extra={'task': task})
            rows.append(m)
        mlp = build_torch_mlp(n_classes, Xtr.shape[1])
        m, _ = eval_model('MLP', mlp, Xtr, ytr_c, Xte, yte, names, is_torch=True,
                          extra={'task': task})
        rows.append(m)
    # sequence models on TOW-IDS (windowed) - binary only
    seq_r1(tow, 'tow', None, rows)

    # ---- SOME/IP ----
    tr, te = sip['split'] == 0, sip['split'] == 1
    for task, ycol, names in [('someip_binary', 'y_bin', None),
                              ('someip_multi', 'y_multi', SOME_CLASSES)]:
        ytr, yte = sip[ycol][tr], sip[ycol][te]
        Xtr, ytr_c = cap_train(sip['X'][tr], ytr)
        Xte = sip['X'][te]
        n_classes = len(np.unique(sip[ycol]))
        names = names or ['Normal', 'Attack']
        models = build_tabular_models_filtered(n_classes)
        for name, mdl in models.items():
            m, _ = eval_model(name, mdl, Xtr, ytr_c, Xte, yte, names,
                              extra={'task': task})
            rows.append(m)
        mlp = build_torch_mlp(n_classes, Xtr.shape[1])
        m, _ = eval_model('MLP', mlp, Xtr, ytr_c, Xte, yte, names, is_torch=True,
                          extra={'task': task})
        rows.append(m)
    seq_r1(sip, 'someip', None, rows)
    aggregate_results(rows, os.path.join(RES, 'r1_benchmark.csv'))


def seq_r1(ds, dsname, _, rows):
    """Windowed sequence models (CNN, GRU) for binary detection on one dataset."""
    tr, te = ds['split'] == 0, ds['split'] == 1
    X, y = ds['X'], ds['y_bin']
    if 'file_id' in ds:
        grp = ds['file_id']
    else:
        grp = np.zeros(len(y))          # TOW-IDS: one stream per split
    tr_grp = grp[tr] if 'file_id' in ds else np.where(tr, 0, 1)[tr]
    te_grp = grp[te] if 'file_id' in ds else np.where(tr, 0, 1)[te]
    Xw_tr, yw_tr, _ = make_windows(X[tr], y[tr], tr_grp, seq_len=16, stride=8,
                                   max_windows=SEQ_CAP_TRAIN)
    Xw_te, yw_te, _ = make_windows(X[te], y[te], te_grp, seq_len=16, stride=8,
                                   max_windows=SEQ_CAP_TEST)
    for kind in ('CNN', 'GRU'):
        mdl = build_torch_sequence(kind, 2, Xw_tr.shape[2], seq_len=16)
        m, _ = eval_model(kind, mdl, Xw_tr, yw_tr, Xw_te, yw_te, ['Normal', 'Attack'],
                          is_torch=True, extra={'task': f'{dsname}_seq_binary'})
        rows.append(m)


def suite_r2(tow, sip):
    """R2: unified multi-protocol detector trained on both datasets."""
    rows = []
    Xfull = np.concatenate([tow['X'][tow['split'] == 0], sip['X'][sip['split'] == 0]])
    ytr_bin = np.concatenate([tow['y_bin'][tow['split'] == 0], sip['y_bin'][sip['split'] == 0]])
    # unified multiclass labels (offset SOME/IP labels by 5), capped with the SAME index
    ytr_multi = np.concatenate([
        tow['y_multi'][tow['split'] == 0],
        np.where(sip['y_multi'][sip['split'] == 0] > 0,
                 sip['y_multi'][sip['split'] == 0] + 5, 0)])
    idx = cap_train_idx(ytr_bin)
    Xtr = Xfull[idx]
    ytr = ytr_bin[idx]
    ytr_m = ytr_multi[idx]
    test_sets = {
        'tow': (tow['X'][tow['split'] == 1], tow['y_bin'][tow['split'] == 1],
                tow['y_multi'][tow['split'] == 1]),
        'someip': (sip['X'][sip['split'] == 1], sip['y_bin'][sip['split'] == 1],
                   sip['y_multi'][sip['split'] == 1] + 5 * (sip['y_multi'][sip['split'] == 1] > 0)),
    }
    for task, ytr_t, names in [('unified_binary', ytr, None),
                               ('unified_multi', ytr_m, UNI_CLASSES)]:
        n_classes = int(max(ytr_t.max() + 1, 2))
        for name, mdl in build_tabular_models_filtered(n_classes).items():
            mrows = []
            t0 = time.perf_counter()
            mdl.fit(Xtr, ytr_t)
            train_s = time.perf_counter() - t0
            for tname, (Xte, yte_b, yte_m) in test_sets.items():
                yte = yte_b if task == 'unified_binary' else yte_m
                tnames = (['Normal', 'Attack'] if task == 'unified_binary'
                          else (TOW_CLASSES if tname == 'tow' else UNI_CLASSES))
                t0 = time.perf_counter()
                proba = mdl.predict_proba(Xte)
                # pad proba columns if test set lacks some classes
                if proba.shape[1] < n_classes:
                    pad = np.zeros((proba.shape[0], n_classes - proba.shape[1]))
                    proba = np.hstack([proba, pad])
                m = classification_metrics(yte, proba, tnames)
                m.update({'model': name, 'task': task, 'test_set': tname,
                          'train_s': round(train_s, 1), 'n_train': len(ytr_t)})
                mrows.append(m)
                print(json.dumps({k: v for k, v in m.items()
                                  if k not in ('confusion', 'per_class_recall')}))
            rows += mrows
        # MLP on unified
        mlp = build_torch_mlp(n_classes, Xtr.shape[1])
        mlp.fit(Xtr, ytr_t)
        for tname, (Xte, yte_b, yte_m) in test_sets.items():
            yte = yte_b if task == 'unified_binary' else yte_m
            proba = mlp.predict_proba(Xte)
            if proba.shape[1] < n_classes:
                pad = np.zeros((proba.shape[0], n_classes - proba.shape[1]))
                proba = np.hstack([proba, pad])
            tnames = (['Normal', 'Attack'] if task == 'unified_binary'
                      else (TOW_CLASSES if tname == 'tow' else UNI_CLASSES))
            m = classification_metrics(yte, proba, tnames)
            m.update({'model': 'MLP', 'task': task, 'test_set': tname, 'n_train': len(ytr_t)})
            print(json.dumps({k: v for k, v in m.items()
                              if k not in ('confusion', 'per_class_recall')}))
            rows.append(m)
    aggregate_results(rows, os.path.join(RES, 'r2_unified.csv'))


def suite_r3(tow, sip):
    """R3: leave-one-protocol-family-out (LOFO) zero-day transfer.

    Families: AVTP (F_I), gPTP (P_I), SWITCH (M_F), UDP-CAN (C_D+C_R), SOMEIP.
    Train binary on all other families (normal from both datasets + other attacks),
    test on held-out family's attack packets + its normal packets.
    """
    rows = []
    # build a combined frame: X, y_bin, family
    X = np.concatenate([tow['X'], sip['X']])
    y = np.concatenate([tow['y_bin'], sip['y_bin']])
    fam = np.empty(len(y), dtype=object)
    tow_fam = np.array([TOW_FAMILY.get(int(v), 'NORMAL') for v in tow['y_multi']])
    sip_fam = np.array([SOME_FAMILY.get(int(v), 'NORMAL') for v in sip['y_multi']])
    fam[:] = np.concatenate([tow_fam, sip_fam])
    split = np.concatenate([tow['split'], sip['split'] + 2])  # 0,1 tow tr/te; 2,3 sip tr/te
    tow_te = tow['split'] == 1
    sip_te = sip['split'] == 1
    normal_m = fam == 'NORMAL'
    families = ['AVTP', 'gPTP', 'SWITCH', 'UDP-CAN', 'SOMEIP']
    models = ['LR', 'RF', 'XGB', 'LGBM', 'VoteEns', 'MLP']
    for held in families:
        # train: train-split packets whose family != held (normal + other-family attacks)
        tr_mask = np.isin(split, [0, 2]) & (fam != held)
        te_mask = np.isin(split, [1, 3]) & ((fam == held) | normal_m)
        Xtr, ytr = X[tr_mask], y[tr_mask]
        Xtr, ytr = cap_train(Xtr, ytr)
        Xte, yte = X[te_mask], y[te_mask]
        for name in models:
            if name == 'MLP':
                mdl = build_torch_mlp(2, Xtr.shape[1])
                m, _ = eval_model('MLP', mdl, Xtr, ytr, Xte, yte, ['Normal', 'Attack'],
                                  is_torch=True,
                                  extra={'task': 'r3_lofo', 'held_out': held})
            else:
                mdl = build_tabular_models_filtered(2)[name]
                m, _ = eval_model(name, mdl, Xtr, ytr, Xte, yte, ['Normal', 'Attack'],
                                  extra={'task': 'r3_lofo', 'held_out': held})
            rows.append(m)
    aggregate_results(rows, os.path.join(RES, 'r3_lofo.csv'))


def suite_r4(tow, sip, r1_rows_path=None):
    """R4: deployment cost summary from stored predictions of the best models."""
    rows = []
    Xte = np.concatenate([tow['X'][tow['split'] == 1], sip['X'][sip['split'] == 1]])
    # retrain compact candidates on unified binary and measure cost precisely
    Xtr = np.concatenate([tow['X'][tow['split'] == 0], sip['X'][sip['split'] == 0]])
    ytr = np.concatenate([tow['y_bin'][tow['split'] == 0], sip['y_bin'][sip['split'] == 0]])
    Xtr, ytr = cap_train(Xtr, ytr)
    for name in ('LR', 'RF', 'ET', 'XGB', 'LGBM'):
        mdl = build_tabular_models_filtered(2)[name]
        t0 = time.perf_counter()
        mdl.fit(Xtr, ytr)
        train_s = time.perf_counter() - t0
        thr = inference_throughput(mdl, Xte)
        rows.append({'model': name, 'size_mb': round(model_size_mb(mdl), 2),
                     'train_s': round(train_s, 1), 'thr_kmsg_s': round(thr / 1000, 1)})
        print(rows[-1])
    aggregate_results(rows, os.path.join(RES, 'r4_cost.csv'))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    which = args[0] if args else 'all'
    tow, sip = load()
    print('TOW-IDS:', tow['X'].shape, 'SOME/IP:', sip['X'].shape)
    if which in ('all', 'r1'):
        suite_r1(tow, sip)
    if which in ('all', 'r2'):
        suite_r2(tow, sip)
    if which in ('all', 'r3'):
        suite_r3(tow, sip)
    if which in ('all', 'r4'):
        suite_r4(tow, sip)
