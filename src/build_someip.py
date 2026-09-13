"""Build SOME/IP dataset feature arrays from the 58-byte pickle rows.

Files: someip_data/<AttackType>/{name}_{i}_x.pickle,_y.pickle
Each row of x is 58 leading bytes of an Ethernet frame; y in {0,1} per message.

Split protocol: deterministic file-level split (70% train / 30% test) per attack
directory, files sorted and split without shuffling across the boundary, so no
capture file straddles train and test (avoids within-capture leakage).

Output: data/someip.npz with X, y_bin, y_multi (0 Normal, 1 EoE, 2 EoV, 3 MR, 4 MResp),
        ethertypes, split, file_id
"""
import sys, os, glob, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_common import extract_static_features, StreamContext, frame_ethertype

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
SRC = os.path.join(DATA, 'someip_data')
ATTACK_TO_ID = {'Error_on_error': 1, 'Error_on_event': 2, 'Missing_request': 3, 'Missing_response': 4}


def process_capture(x_path, y_path, attack_id):
    with open(x_path, 'rb') as f:
        x = pickle.load(f)
    with open(y_path, 'rb') as f:
        y = pickle.load(f)
    assert x.shape[0] == y.shape[0], f'{x_path}: {x.shape[0]} vs {y.shape[0]}'
    frames = [row.tobytes() for row in np.asarray(x, dtype=np.uint8)]
    ctx = StreamContext(window=100)
    statics, contexts, ets = [], [], []
    for buf in frames:
        et = frame_ethertype(buf)
        ctx.update(buf, et)
        statics.append(extract_static_features(buf))
        contexts.append(ctx.features())
        ets.append(et)
    X = np.concatenate([np.stack(statics), np.stack(contexts)], axis=1).astype(np.float32)
    y_multi = np.where(np.asarray(y) > 0, attack_id, 0).astype(np.int8)
    return X, y_multi, np.asarray(ets, dtype=np.int32)


def main():
    all_X, all_y, all_et, all_split, all_file = [], [], [], [], []
    file_counter = 0
    for atk, atk_id in sorted(ATTACK_TO_ID.items()):
        x_files = sorted(glob.glob(os.path.join(SRC, atk, '*_x.pickle')))
        n_train = int(round(len(x_files) * 0.7))
        print(f'{atk}: {len(x_files)} captures -> {n_train} train / {len(x_files) - n_train} test')
        for i, xp in enumerate(x_files):
            yp = xp.replace('_x.pickle', '_y.pickle')
            X, y, et = process_capture(xp, yp, atk_id)
            all_X.append(X); all_y.append(y); all_et.append(et)
            all_split.append(np.full(len(y), 0 if i < n_train else 1, dtype=np.int8))
            all_file.append(np.full(len(y), file_counter, dtype=np.int32))
            file_counter += 1
    X = np.concatenate(all_X); y = np.concatenate(all_y); et = np.concatenate(all_et)
    sp = np.concatenate(all_split); fid = np.concatenate(all_file)
    np.savez_compressed(os.path.join(DATA, 'someip.npz'), X=X, y_multi=y,
                        y_bin=(y > 0).astype(np.int8), ethertypes=et, split=sp, file_id=fid)
    print('saved data/someip.npz', X.shape)
    print('label counts:', np.bincount(y.astype(int), minlength=5).tolist())
    tr = sp == 0
    print('train counts:', np.bincount(y[tr].astype(int), minlength=5).tolist())
    print('test  counts:', np.bincount(y[~tr].astype(int), minlength=5).tolist())


if __name__ == '__main__':
    main()
