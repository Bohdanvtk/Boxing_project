# loaders.py
import os
import numpy as np
from sklearn.model_selection import train_test_split

def _iter_sample_dirs(root_dir: str):
    # stable order
    for d in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, d)
        if os.path.isdir(p) and d.startswith("img_"):
            yield p

def make_train_val(cfg: dict, Task):
    root_dir = cfg["paths"]["data_root"]
    val_split = float(cfg["train"]["val_split"])
    seed = int(cfg["train"]["seed"])

    X_list, y_list, extras_list = [], [], []

    for sdir in _iter_sample_dirs(root_dir):
        X_i, y_i, extra_i = Task.load_sample(sdir)
        X_list.append(X_i)
        y_list.append(y_i)
        extras_list.append(extra_i)

    # to numpy
    X = np.array(X_list)
    y = np.array(y_list)


    extras_full = {}
    if extras_list:
        keys = set().union(*[e.keys() for e in extras_list])
        for k in keys:
            vals = [e.get(k, None) for e in extras_list]
            if all(isinstance(v, np.ndarray) for v in vals):
                extras_full[k] = np.stack(vals, axis=0)

    # р split of indexes
    N = len(X)
    idx = np.arange(N)
    idx_tr, idx_val = train_test_split(
        idx, test_size=val_split, random_state=seed, shuffle=True
    )

    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_val], y[idx_val]

    # split exstras in the same way
    extras_tr, extras_val = {}, {}
    for k, arr in extras_full.items():
        extras_tr[k] = arr[idx_tr]
        extras_val[k] = arr[idx_val]

    return (X_tr, y_tr, extras_tr), (X_val, y_val, extras_val)
