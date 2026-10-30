"""Image-level bootstrap: mIoU confidence intervals, paired differences, leakage interaction.

Input:  results/{dev,test}/<run>_{dev,test}_per_image.npz (per: [N, 3, 116] = intersection, union, target pixels)
        results/leakage.json (clean/leaked lists; produced by src/leakage.py)
Output: results/analysis/stats.json and a summary on stdout. No GPU needed.
Usage:  python3 analysis/stats.py [--B 5000] [--seed 0]

Class set: on the full set the mmseg rule (classes with union > 0, comparable with the dataset paper); the mIoU of a
clean/leaked subset uses that subset's GT classes, fixed across the bootstrap; differences that compare the two
subsets (leaked − clean, interaction) use the GT classes common to both.
Interval: BCa 95% (Efron 1987; acceleration from an image-level jackknife); p is the smallest α at which the BCa
interval excludes 0.
"""
import argparse
import json
import pathlib
from statistics import NormalDist

import numpy as np

_N = NormalDist()

ROOT = pathlib.Path(__file__).resolve().parent.parent
GROUPS = {
    "dinov3": ["a_v3L_lora_s0", "d_v3L_lora_s1", "e_v3L_lora_s2"],
    "cnx": ["b_cnxL_s0", "f_cnxL_s1", "g_cnxL_s2"],
    "cnx_paper": ["c_cnxL_mak_s0"],
    "h2": ["i_H2_s0"],
    "dinov3_s0": ["a_v3L_lora_s0"],
    "dinov2": ["h_v2L_lora_s0"],
}


def load(split):
    """Run -> (per [N,3,C], stems). Image order must be identical across runs."""
    out, order = {}, None
    for runs in GROUPS.values():
        for k in runs:
            if k in out:
                continue
            z = np.load(ROOT / f"results/{split}/{k}_{split}_per_image.npz")
            stems = list(z["stems"])
            if order is None:
                order = stems
            assert stems == order, f"{k}: image order differs"
            out[k] = z["per"].astype(np.float64)
    return out, np.array(order)


def miou(I, U, S=None):
    """I, U: [..., C] -> mIoU (%). S: fixed class mask [C]; classes inside it with union 0 are skipped.
    Without S, all classes with union > 0 (like mmseg IoUMetric)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(U > 0, I / U, np.nan)
    return np.nanmean(r if S is None else r[..., S], axis=-1) * 100


def mdice(I, U, S=None):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(U > 0, 2 * I / (U + I), np.nan)
    return np.nanmean(r if S is None else r[..., S], axis=-1) * 100


def gt_classes(per, w):
    """w: [N] weights (subset indicator) -> mask [C] of classes with GT pixels; GT is the same in every run."""
    return (w @ next(iter(per.values()))[:, 2]) > 0


def group_score(per, W, runs, S=None, metric=miou):
    """W: [B, N] bootstrap weights (one resample per row). Group = mean over seeds."""
    return np.mean([metric(W @ per[k][:, 0], W @ per[k][:, 1], S) for k in runs], axis=0)


def jackknife_weights(one):
    """one: [N] subset weights -> rows [m, N] that each leave one image of the subset out."""
    idx = np.nonzero(one)[0]
    Wj = np.repeat(one[None], len(idx), axis=0)
    Wj[np.arange(len(idx)), idx] = 0
    return Wj


def summary(stat, W, one):
    """stat: weight matrix [B, N] -> statistic [B]. Point estimate, BCa 95% interval and two-sided p."""
    point = float(stat(one[None])[0])
    boot, jack = stat(W), stat(jackknife_weights(one))
    B = len(boot)
    z0 = _N.inv_cdf(float(np.clip((boot < point).mean() + 0.5 * (boot == point).mean(), 1 / B, 1 - 1 / B)))
    d = jack.mean() - jack
    a = float((d ** 3).sum() / (6 * (d ** 2).sum() ** 1.5)) if (d ** 2).sum() > 0 else 0.0

    def percentile(z):  # bootstrap percentile that nominal z maps to under BCa
        return _N.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boot, [percentile(_N.inv_cdf(0.025)), percentile(_N.inv_cdf(0.975))])
    # p: invert the BCa transform at the bootstrap percentile of 0 -> nominal z at which 0 is an interval bound
    v = _N.inv_cdf(float(np.clip((boot <= 0).mean(), 1 / (2 * B), 1 - 1 / (2 * B)))) - z0
    z = v / (1 + a * v) - z0
    p = max(2 * (1 - _N.cdf(abs(z))), 1 / B)
    return {"point": round(point, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)], "p": round(p, 4)}


def weights(rng, n, B, mask=None):
    """Multinomial resampling; with a mask, resamples only from that subset."""
    idx = np.arange(n) if mask is None else np.nonzero(mask)[0]
    W = np.zeros((B, n))
    for b in range(B):
        np.add.at(W[b], rng.choice(idx, size=len(idx)), 1)
    return W


def analyse(split, B, rng, clean_set):
    per, stems = load(split)
    n = len(stems)
    clean = np.array([s in clean_set for s in stems])
    subsets = {"all": None, "clean": clean, "leaked": ~clean}
    W = {a: weights(rng, n, B, m) for a, m in subsets.items()}
    one = {a: (np.ones(n) if m is None else m.astype(float)) for a, m in subsets.items()}
    gt = {a: gt_classes(per, one[a]) for a in subsets}
    S = {**gt, "all": None}  # the full set keeps the mmseg rule (comparable with the dataset paper)
    S_common = gt["clean"] & gt["leaked"]

    def score(g, S_):
        return lambda w: group_score(per, w, GROUPS[g], S_)

    result = {"split": split, "n": {a: int(t.sum()) for a, t in one.items()}, "B": B, "interval": "BCa",
              "n_classes": {**{a: int(s.sum()) for a, s in gt.items()}, "clean∩leaked": int(S_common.sum())},
              "groups": {}, "diffs": {}}
    for g, runs in GROUPS.items():
        result["groups"][g] = {
            a: {**{k: v for k, v in summary(score(g, S[a]), W[a], one[a]).items() if k != "p"},
                "mDice": round(float(group_score(per, one[a][None], runs, S[a], mdice)[0]), 2)}
            for a in subsets}

    comparisons = {
        "H1: DINOv3 − ConvNeXt (ours)": ("dinov3", "cnx"),
        "H1: DINOv3 − ConvNeXt (paper recipe)": ("dinov3", "cnx_paper"),
        "H2: H2 − DINOv3 (same seed s0)": ("h2", "dinov3_s0"),
        "DINOv3 − DINOv2 (same seed s0)": ("dinov3_s0", "dinov2"),
        "DINOv3 (3 seeds) − DINOv2 (s0)": ("dinov3", "dinov2"),
        "DINOv2 − ConvNeXt (ours)": ("dinov2", "cnx"),
    }
    for name, (g1, g2) in comparisons.items():
        result["diffs"][name] = {
            a: summary(lambda w, a=a: score(g1, S[a])(w) - score(g2, S[a])(w), W[a], one[a]) for a in subsets}

    # Differences comparing clean with leaked: the two strata are resampled separately (W clean + W leaked),
    # with their common GT classes
    W_strat = W["clean"] + W["leaked"]
    one_strat = np.ones(n)

    def diff(g1, g2, m):
        return lambda w: score(g1, S_common)(w * m) - score(g2, S_common)(w * m)
    inter = summary(lambda w: diff("dinov3", "cnx", clean)(w) - diff("dinov3", "cnx", ~clean)(w), W_strat, one_strat)
    result["diffs"]["Leakage interaction: (DINOv3−CNN) clean − leaked"] = inter
    for g in ("cnx", "dinov3"):
        result["diffs"][f"{g}: leaked − clean"] = summary(
            lambda w, g=g: score(g, S_common)(w * ~clean) - score(g, S_common)(w * clean), W_strat, one_strat)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lk = json.loads((ROOT / "results/leakage.json").read_text())
    sets = {"test": set(lk["clean_test"]),
            "dev": set(lk["n"] and [s for s in load("dev")[1] if s not in set(lk["val_leak_stems"])])}
    rng = np.random.default_rng(a.seed)
    out = {sp: analyse(sp, a.B, rng, sets[sp]) for sp in ("test", "dev")}
    path = ROOT / "results/analysis/stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for sp, r in out.items():
        print(f"\n== {sp} (n: {r['n']}, B={r['B']})")
        for g, v in r["groups"].items():
            print(f"  {g:10s}", "  ".join(f"{a}: {x['point']} {x['ci95']}" for a, x in v.items()))
        for name, v in r["diffs"].items():
            if "all" in v:
                print(f"  {name}:", "  ".join(f"{a}: {x['point']} {x['ci95']} p={x['p']}" for a, x in v.items()))
            else:
                print(f"  {name}: {v['point']} {v['ci95']} p={v['p']}")
    print(f"\n-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
