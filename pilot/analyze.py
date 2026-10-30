"""Collect run results: full dev, clean dev (leakage-free), H3 axes, paired bootstrap. Dev only."""
import json, os, glob, collections
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
P = os.path.expanduser('~/.cache/plantseg/prep')
meta = json.load(open(f'{P}/meta.json'))
leak = set(json.load(open(f'{P}/leakage.json'))['val_leak_stems'])
NC = 116

# class-frequency terciles (by number of training images)
cnt = collections.Counter(m['cls'] for m in meta.values() if m['split'] == 'train')
order = sorted(range(1, NC), key=lambda c: -cnt[c])
third = len(order) // 3
groups = {'head': set(order[:third]), 'mid': set(order[third:2 * third]), 'tail': set(order[2 * third:])}


def miou(per, cls=None):
    I, U = per[:, 0].sum(0).astype(float), per[:, 1].sum(0).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        iou = I / U
    if cls is not None:
        iou = iou[sorted(cls)]
    return float(np.nanmean(iou) * 100)


def load(tag):
    z = np.load(f'{D}/{tag}/dev_per_image.npz')
    return z['per'], list(z['stems'])


runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f'{D}/*/result.json'))
out = {}
for t in runs:
    per, stems = load(t)
    r = json.load(open(f'{D}/{t}/result.json'))
    clean = np.array([s not in leak for s in stems])
    mr = np.array([meta[s]['mask_ratio'] for s in stems])
    med = np.median(mr)
    out[t] = {'mIoU_dev': round(r['mIoU'], 2), 'mAcc_dev': round(r['mAcc'], 2), 'mIoU_clean_dev': round(miou(per[clean]), 2),
              'mIoU_leaked_dev': round(miou(per[~clean]), 2),
              **{f'mIoU_{g}': round(miou(per, c), 2) for g, c in groups.items()},
              'mIoU_small_lesion': round(miou(per[mr < med]), 2), 'mIoU_large_lesion': round(miou(per[mr >= med]), 2),
              'train_min': round(r['train_min'], 1)}


def boot(ta, tb, B=2000, seed=0):
    pa, sa = load(ta); pb, sb = load(tb)
    assert sa == sb
    rng = np.random.default_rng(seed); n = len(sa); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(miou(pa[i]) - miou(pb[i]))
    return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]


def agg(prefix):
    v = [out[t] for t in out if t.startswith(prefix + '_s')]
    if not v:
        return None
    a = {}
    for k in v[0]:
        x = np.array([e[k] for e in v])
        a[k] = [round(float(x.mean()), 2), round(float(x.std(ddof=1)) if len(x) > 1 else 0.0, 2)]
    a['n_seeds'] = len(v)
    return a


summary = {p: agg(p) for p in ('B0', 'P1', 'P2', 'M2', 'H2', 'B0L')}
pairs = {}
for a_, b_ in [('P1', 'B0'), ('P2', 'B0'), ('M2', 'B0'), ('P1', 'M2'), ('H2', 'P2'), ('B0L', 'B0')]:
    for s in (0, 1):
        ta, tb = f'{a_}_s{s}', f'{b_}_s{s}'
        if ta in out and tb in out:
            pairs[f'{ta}-{tb}'] = {'diff': round(out[ta]['mIoU_dev'] - out[tb]['mIoU_dev'], 2), 'boot95': boot(ta, tb)}
res = {'runs': out, 'summary_mean_sd': summary, 'paired': pairs,
       'groups_train_image_count': {g: [min(cnt[c] for c in cs), max(cnt[c] for c in cs)] for g, cs in groups.items()}}
json.dump(res, open(f'{D}/analysis.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
