"""Model zoo for multi-protocol automotive Ethernet IDS experiments.

Uniform sklearn-like interface (fit / predict / predict_proba) for:
  - Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM
  - Torch MLP, 1D-CNN (sequence), GRU (sequence)
  - Soft-voting ensemble (RF+XGB+LGBM), stacking ensemble (LR meta-learner)
"""
import time, os
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = 'cpu'
N_THREADS = 8
torch.set_num_threads(N_THREADS)


# --------------------------------------------------------------------------
# Classical models
# --------------------------------------------------------------------------
def make_lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, random_state=SEED))


def make_rf(n_estimators=250):
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=N_THREADS,
                                  max_depth=20, min_samples_leaf=4,
                                  random_state=SEED, class_weight='balanced_subsample')


def make_et(n_estimators=250):
    return ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=N_THREADS,
                                max_depth=20, min_samples_leaf=4,
                                random_state=SEED, class_weight='balanced_subsample')


def make_xgb(n_estimators=300):
    return xgb.XGBClassifier(n_estimators=n_estimators, tree_method='hist', max_depth=8,
                             learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                             n_jobs=N_THREADS, random_state=SEED, eval_metric='mlogloss',
                             verbosity=0)


def make_lgbm(n_estimators=300):
    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.1, num_leaves=63,
                              subsample=0.9, colsample_bytree=0.9, n_jobs=N_THREADS,
                              random_state=SEED, verbose=-1)


def make_voting():
    return VotingClassifier(
        estimators=[('rf', make_rf(200)), ('xgb', make_xgb(250)), ('lgbm', make_lgbm(250))],
        voting='soft', n_jobs=1)


def make_stacking():
    return StackingClassifier(
        estimators=[('rf', make_rf(150)), ('xgb', make_xgb(200)), ('lgbm', make_lgbm(200))],
        final_estimator=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        cv=3, n_jobs=1, stack_method='predict_proba')


# --------------------------------------------------------------------------
# Torch models
# --------------------------------------------------------------------------
class TorchMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=(256, 128, 64), p_drop=0.2):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchCNN1D(nn.Module):
    """1D-CNN over a window of consecutive packets (input [B, L, F] -> permuted to [B, F, L])."""

    def __init__(self, d_in, n_classes, seq_len, ch=(128, 64)):
        super().__init__()
        c1, c2 = ch
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, c1, kernel_size=3, padding=1), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4))
        self.fc = nn.Sequential(nn.Linear(c2 * 4, 128), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(128, n_classes))

    def forward(self, x):          # x: [B, L, F]
        x = x.permute(0, 2, 1)
        return self.fc(self.conv(x).flatten(1))


class TorchGRU(nn.Module):
    def __init__(self, d_in, n_classes, hidden=96, layers=2, p_drop=0.2):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True,
                          dropout=p_drop if layers > 1 else 0.0, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(p_drop),
                                nn.Linear(128, n_classes))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TorchWrapper:
    """Sklearn-like wrapper for torch classifiers (tabular MLP or sequence nets)."""

    def __init__(self, kind, n_classes, d_in, seq_len=16, epochs=25, batch=512, lr=1e-3, name=None):
        self.kind, self.n_classes, self.d_in, self.seq_len = kind, n_classes, d_in, seq_len
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.name = name or kind
        if kind == 'MLP':
            self.net = TorchMLP(d_in, n_classes)
        elif kind == 'CNN':
            self.net = TorchCNN1D(d_in, n_classes, seq_len)
        elif kind == 'GRU':
            self.net = TorchGRU(d_in, n_classes)
        else:
            raise ValueError(kind)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X_val is None:
            idx = np.random.permutation(len(X))
            n_val = max(1, int(0.1 * len(X)))
            vi, ti = idx[:n_val], idx[n_val:]
            X_val, y_val = X[vi], y[vi]
            X_tr, y_tr = X[ti], y[ti]
        else:
            X_tr, y_tr = X, y
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)
        # feature standardisation statistics from training data
        self.mu = X_tr.mean(axis=0)
        self.sd = X_tr.std(axis=0) + 1e-8
        ds = TensorDataset(torch.tensor((X_tr - self.mu) / self.sd),
                           torch.tensor(y_tr))
        dl = DataLoader(ds, batch_size=self.batch, shuffle=True, drop_last=False)
        dv = TensorDataset(torch.tensor((X_val - self.mu) / self.sd), torch.tensor(y_val))
        dvl = DataLoader(dv, batch_size=2048, shuffle=False)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss(weight=self._class_weights(y_tr))
        best, best_state, patience = 1e9, None, 4
        for ep in range(self.epochs):
            self.net.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = crit(self.net(xb), yb)
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vloss = sum(crit(self.net(xb), yb).item() * len(xb) for xb, yb in dvl) / len(dv)
            if vloss + 1e-4 < best:
                best, best_state, patience = vloss, {k: v.clone() for k, v in self.net.state_dict().items()}, 4
            else:
                patience -= 1
                if patience == 0:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _class_weights(self, y):
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        w = counts.sum() / np.maximum(counts, 1) / self.n_classes
        return torch.tensor(w, dtype=torch.float32)

    def predict_proba(self, X):
        X = (np.asarray(X, dtype=np.float32) - self.mu) / self.sd
        self.net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                z = self.net(torch.tensor(X[i:i + 4096]))
                outs.append(torch.softmax(z, dim=1).numpy())
        return np.vstack(outs)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def build_tabular_models(n_classes):
    """Returns dict name -> unfitted sklearn-compatible model (tabular features)."""
    return {'LR': make_lr(), 'RF': make_rf(), 'ET': make_et(),
            'XGB': make_xgb(), 'LGBM': make_lgbm(),
            'VoteEns': make_voting(), 'StackEns': make_stacking()}


def build_torch_mlp(n_classes, d_in):
    return TorchWrapper('MLP', n_classes, d_in, epochs=25, batch=1024)


def build_torch_sequence(kind, n_classes, d_in, seq_len=16):
    return TorchWrapper(kind, n_classes, d_in, seq_len=seq_len, epochs=15, batch=512)


def model_size_mb(model):
    fd, path = None, None
    try:
        import tempfile
        fd, path = tempfile.mkstemp(suffix='.joblib')
        joblib.dump(model, path)
        return os.path.getsize(path) / 1e6
    finally:
        if fd is not None:
            os.close(fd)
        if path and os.path.exists(path):
            os.remove(path)


def torch_size_mb(wrapper):
    return sum(p.numel() for p in wrapper.net.parameters()) * 4 / 1e6


def inference_throughput(model, X, is_torch=False, n=20000):
    """Messages classified per second (single batch of n, averaged over 3 runs)."""
    Xs = X[:n]
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        if is_torch:
            model.predict_proba(Xs)
        else:
            model.predict(Xs)
        times.append(time.perf_counter() - t0)
    return len(Xs) / min(times)
