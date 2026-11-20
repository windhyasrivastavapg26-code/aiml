# Improved ZEN-IQA on KonIQ-10k: learned fusion of zero-shot CLIP attributes.
#
# What changed compared to demo/zeniqa_koniq_demo.py
# --------------------------------------------------
# 1. More zero-shot attributes (configs/zeniqa/zeniqa_attribute_extended.py):
#    the first four are the original S/N/O/U, the rest add overall quality,
#    contrast, colour, detail, artifacts, distortion and focus prompts.
# 2. The final score is learned from ALL attributes (not just O,U -> MOS and
#    then a plain average with S and N). Attributes are mapped to logit space
#    first because they are softmax probabilities squashed into [0, 1].
# 3. Proper model selection: hyper-parameters are chosen on the KonIQ
#    *validation* split, the test split is only used once for reporting.
# 4. The old MLP was trained with 120 full-batch steps (heavily under-fit).
#    Here Ridge / MLP-ensemble / gradient-boosting are compared and the best
#    on validation is used, then refit on train+val.
# 5. A 4-parameter logistic mapping (standard in IQA, VQEG) is fitted on
#    out-of-fold predictions. It is monotonic so SRCC is untouched, but PLCC
#    goes up because the relation to MOS is no longer forced to be linear.
# 6. Attributes are cached to disk, so extraction (the slow part) runs once.
#
# Usage
#   python demo/zeniqa_koniq_improved.py \
#       --file_path dataset/1024x768 \
#       --csv_path dataset/koniq10k_distributions_sets.csv
# Re-run instantly (no CLIP forward passes) with the cache:
#   python demo/zeniqa_koniq_improved.py --cache attrs_cache.npz

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def srcc(a, b):
    return float(stats.spearmanr(a, b)[0])


def plcc(a, b):
    return float(stats.pearsonr(np.asarray(a, np.float64), np.asarray(b, np.float64))[0])


def report(name, pred, mos):
    s, p = srcc(pred, mos), plcc(pred, mos)
    print(f'  {name:<44s} SRCC {s:.4f} | PLCC {p:.4f}')
    return s, p


# ----------------------------------------------------------------------------
# Attribute extraction (needs torch + mmedit; skipped when a cache exists)
# ----------------------------------------------------------------------------
def extract(model, df, file_path, desc):
    from mmedit.apis import restoration_inference
    from tqdm import tqdm
    out = []
    for name in tqdm(df['image_name'], desc=desc):
        o, _ = restoration_inference(
            model, os.path.join(file_path, name), return_attributes=True)
        out.append(o.float().detach().cpu().numpy()[0])
    return np.asarray(out, dtype=np.float32)


# ----------------------------------------------------------------------------
# Attribute extraction with checkpoint / resume support
# ----------------------------------------------------------------------------
def extract(model, df, file_path, desc, checkpoint_path=None, save_every=50):
    """Extract attributes. If checkpoint_path is given, resume from it and
    periodically save progress so a crash does not lose work."""
    from mmedit.apis import restoration_inference
    from tqdm import tqdm

    n = len(df)
    attrs = None
    start_idx = 0

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        attrs = ckpt['attrs']
        start_idx = int(ckpt['done'])
        print(f'  Resuming {desc} from image {start_idx}/{n}')
        if start_idx >= n:
            return attrs.astype(np.float32)

    if attrs is None:
        # We don't know the attribute dimension yet → extract first image
        first = df.iloc[0]['image_name']
        o, _ = restoration_inference(
            model, os.path.join(file_path, first), return_attributes=True)
        dim = o.shape[-1]
        attrs = np.zeros((n, dim), dtype=np.float32)
        attrs[0] = o.float().detach().cpu().numpy()[0]
        start_idx = 1
        if checkpoint_path:
            np.savez(checkpoint_path, attrs=attrs, done=start_idx)

    for i in tqdm(range(start_idx, n), desc=desc, initial=start_idx, total=n):
        name = df.iloc[i]['image_name']
        o, _ = restoration_inference(
            model, os.path.join(file_path, name), return_attributes=True)
        attrs[i] = o.float().detach().cpu().numpy()[0]

        if checkpoint_path and (i + 1) % save_every == 0:
            np.savez(checkpoint_path, attrs=attrs, done=i + 1)

    # final save for this split
    if checkpoint_path:
        np.savez(checkpoint_path, attrs=attrs, done=n)

    return attrs.astype(np.float32)


def load_or_extract(args):
    if os.path.isfile(args.cache):
        print(f'Loading cached attributes from {args.cache}')
        d = np.load(args.cache)
        return {k: d[k] for k in d.files}

    import torch
    from mmedit.apis import init_model
    model = init_model(args.config, args.checkpoint,
                       device=torch.device(args.device))
    model.float().eval()

    df = pd.read_csv(args.csv_path, on_bad_lines='skip')
    data = {}
    os.makedirs('checkpoints', exist_ok=True)

    for split, key in [('training', 'train'), ('validation', 'val'),
                       ('test', 'test')]:
        sub = df[df.set == split].reset_index(drop=True)
        if args.max_per_split:
            sub = sub.iloc[:args.max_per_split]

        ckpt_path = os.path.join('checkpoints', f'{key}_attrs_ckpt.npz')
        attrs = extract(model, sub, args.file_path, f'{key} attrs',
                        checkpoint_path=ckpt_path, save_every=args.save_every)
        data[f'{key}_attrs'] = attrs
        data[f'{key}_mos'] = sub['MOS'].values.astype(np.float32)

        # also keep a per-split final file (useful if you only want one split)
        np.savez(os.path.join('checkpoints', f'{key}_attrs_final.npz'),
                 attrs=attrs, mos=data[f'{key}_mos'])

    np.savez(args.cache, **data)
    print(f'Saved full attribute cache to {args.cache}')
    return data

# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def make_features(A, n_attr=None):
    """A: (N, K) softmax probabilities in [0,1], first four = S, N, O, U."""
    A = np.clip(A.astype(np.float64), 1e-4, 1 - 1e-4)
    logit = np.log(A / (1 - A))
    S, N, O, U = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    expo_min = np.minimum(O, U)
    expo_prod = O * U
    # original ZEN-IQA zero-shot score as an extra feature (strong prior)
    zen = (S + N + expo_min) / 3.0
    extra = np.stack([expo_min, expo_prod, zen,
                      np.log(expo_min / (1 - expo_min + 1e-6) + 1e-6)], 1)
    return np.concatenate([A, logit, extra], axis=1)


# ----------------------------------------------------------------------------
# Regressors
# ----------------------------------------------------------------------------
class Ensemble:
    """Average of several sklearn regressors with different seeds."""

    def __init__(self, make, n=5):
        self.models = [make(s) for s in range(n)]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def candidates():
    c = {}
    c['ridge'] = lambda: RidgeCV(alphas=np.logspace(-3, 3, 13))
    for h in [(32, 32), (64, 32)]:
        for a in [1e-3, 1e-2]:
            c[f'mlp{h}_a{a}'] = (lambda h=h, a=a: Ensemble(
                lambda s: MLPRegressor(hidden_layer_sizes=h, alpha=a, max_iter=800,
                                       early_stopping=True,
                                       validation_fraction=0.1,
                                       n_iter_no_change=30, random_state=s)))
    c['hgb'] = lambda: Ensemble(
        lambda s: HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
            random_state=s), n=3)
    return c


class Pipe:
    def __init__(self, reg):
        self.sc, self.reg = StandardScaler(), reg

    def fit(self, X, y):
        self.reg.fit(self.sc.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.reg.predict(self.sc.transform(X))


# ----------------------------------------------------------------------------
# 4-parameter logistic mapping (monotonic, improves PLCC)
# ----------------------------------------------------------------------------
def _logistic(x, b1, b2, b3, b4):
    return (b1 - b2) / (1 + np.exp(-(x - b3) / (np.abs(b4) + 1e-6))) + b2


def fit_logistic(pred, mos):
    p0 = [mos.max(), mos.min(), np.mean(pred), np.std(pred) / 4 + 1e-3]
    try:
        popt, _ = curve_fit(_logistic, pred, mos, p0=p0, maxfev=20000)
        return popt
    except Exception:
        return None


def apply_logistic(popt, pred):
    return pred if popt is None else _logistic(pred, *popt)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/zeniqa/zeniqa_attribute_extended.py')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--file_path', default='dataset/1024x768')
    ap.add_argument('--csv_path', default='dataset/koniq10k_distributions_sets.csv')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--cache', default='attrs_cache.npz')
    ap.add_argument('--max_per_split', type=int, default=0,
                    help='debug: only use first N images per split')
    ap.add_argument('--out_csv', default='zeniqa_improved_predictions.csv')
    ap.add_argument('--save_every', type=int, default=50,
                    help='save extraction checkpoint every N images')
    args = ap.parse_args()

    d = load_or_extract(args)
    Atr, ytr = d['train_attrs'], d['train_mos']
    Ava, yva = d['val_attrs'], d['val_mos']
    Ate, yte = d['test_attrs'], d['test_mos']
    print(f'attrs: {Atr.shape[1]} | train {len(ytr)} val {len(yva)} test {len(yte)}')

    # MOS normalised for stable fitting
    lo, hi = ytr.min(), ytr.max()
    norm = lambda y: (y - lo) / (hi - lo + 1e-8)
    denorm = lambda y: y * (hi - lo) + lo

    print('\nBaselines on TEST (no learning):')
    zen = lambda A: (A[:, 0] + A[:, 1] + np.minimum(A[:, 2], A[:, 3])) / 3
    report('ZEN-IQA (S,N,min(O,U)) average', zen(Ate), yte)
    if Ate.shape[1] > 4:
        report('Plain CLIP-IQA good/bad prompt', Ate[:, 4], yte)

    Xtr, Xva, Xte = make_features(Atr), make_features(Ava), make_features(Ate)

    # ---- model selection on validation ----
    print('\nModel selection on VALIDATION:')
    best, best_name, best_score = None, None, -9
    for name, mk in candidates().items():
        m = Pipe(mk()).fit(Xtr, norm(ytr))
        pv = m.predict(Xva)
        s, p = srcc(pv, yva), plcc(pv, yva)
        print(f'  {name:<22s} SRCC {s:.4f} | PLCC {p:.4f}')
        if s + p > best_score:
            best, best_name, best_score = mk, name, s + p
    print(f'-> selected: {best_name}')

    # ---- refit on train+val, calibrate with out-of-fold predictions ----
    Xall = np.vstack([Xtr, Xva])
    yall = norm(np.concatenate([ytr, yva]))
    oof = np.zeros_like(yall)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xall):
        oof[te] = Pipe(best()).fit(Xall[tr], yall[tr]).predict(Xall[te])
    popt = fit_logistic(oof, yall)

    final = Pipe(best()).fit(Xall, yall)
    raw = final.predict(Xte)
    cal = apply_logistic(popt, raw)

    print('\nFINAL on TEST (used once):')
    report('learned fusion (raw)', denorm(raw), yte)
    s, p = report('learned fusion + logistic mapping', denorm(cal), yte)
    print('\n==============================')
    print(f'SRCC: {s:.4f} | PLCC: {p:.4f}')
    print('==============================')

    pd.DataFrame({'pred': denorm(cal), 'raw': denorm(raw), 'mos': yte}
                 ).to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    main()