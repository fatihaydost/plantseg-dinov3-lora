"""Extra analyses (no GPU): H3 subgroup bootstrap, image-level class accuracy, class-agnostic mask IoU (test), leakage
table, label inconsistency.

Input:  results/{dev,test}/*_per_image.npz, results/leakage.json, the local prepared folder (--prep: meta.json,
        phash.json, {train,val}/lab_orig/*.png). Predicted pixel count is derived from I/U/target: pred = U − target + I.
Output: results/analysis/extra_analysis.json and a summary on stdout.
Usage:  uv run --with numpy --with pillow python3 analysis/extra_analysis.py [--prep ~/.cache/plantseg/prep] [--B 5000]
"""
import argparse
import collections
import json
import pathlib

import numpy as np

from stats import GROUPS, ROOT, gt_classes, load, summary, weights

CONTRASTS = {"DINOv3 − ConvNeXt": ("dinov3", "cnx"), "DINOv2 − ConvNeXt": ("dinov2", "cnx")}


def miou_cls(W, per, cls):
    """mIoU (%) over the `cls` columns only, from I/U summed with weights W [B,N]."""
    I, U = W @ per[:, 0][:, cls], W @ per[:, 1][:, cls]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.where(U > 0, I / U, np.nan), axis=-1) * 100


def group(per, W, g, cls):
    return np.mean([miou_cls(W, per[k], cls) for k in GROUPS[g]], axis=0)


def h3(split, per, stems, meta, B, rng):
    """Model differences for head/mid/tail (thirds by training frequency) and small/large lesion (mask ratio median)."""
    cnt = collections.Counter(m["cls"] for m in meta.values() if m["split"] == "train")
    order = sorted(range(1, 116), key=lambda c: -cnt[c])
    t = len(order) // 3
    n = len(stems)
    gt = gt_classes(per, np.ones(n))  # classes absent from the split's GT (test 42, dev 69) are left out of the mean
    classes = {g: [c for c in v if gt[c]] for g, v in
               {"head": order[:t], "mid": order[t:2 * t], "tail": order[2 * t:]}.items()}
    W, one = weights(rng, n, B), np.ones(n)
    # lesion size: small and large resampled separately (stratified); class set = each subset's GT, common GT in the diff
    mr = np.array([meta[s]["mask_ratio"] for s in stems])
    small = mr < np.median(mr)
    Ws, Wl = weights(rng, n, B, small), weights(rng, n, B, ~small)
    Ss, Sl = gt_classes(per, small.astype(float)), gt_classes(per, (~small).astype(float))
    cs, cl, cc = (np.nonzero(x)[0] for x in (Ss, Sl, Ss & Sl))
    out = {}
    for name, (g1, g2) in CONTRASTS.items():
        def diff(cls, m=None):
            return lambda w: (group(per, w if m is None else w * m, g1, cls)
                              - group(per, w if m is None else w * m, g2, cls))
        d = {gr: summary(diff(cls), W, one) for gr, cls in classes.items()}
        # tail − head: from the same resamples, is the difference larger on the tail
        d["tail − head"] = summary(lambda w: diff(classes["tail"])(w) - diff(classes["head"])(w), W, one)
        d["small lesion"] = summary(diff(cs), Ws, small.astype(float))
        d["large lesion"] = summary(diff(cl), Wl, (~small).astype(float))
        d["small − large"] = summary(lambda w: diff(cc, small)(w) - diff(cc, ~small)(w), Ws + Wl, one)
        out[name] = d
    return {"n_classes": {**{k: len(v) for k, v in classes.items()},
                          "small": len(cs), "large": len(cl), "small∩large": len(cc)}, "diffs": out}


def class_accuracy(split, per, stems, meta, B, rng):
    """Image class = class with the most predicted foreground pixels; correct = the meta class."""
    true = np.array([meta[s]["cls"] for s in stems])
    correct = {}
    for g in ("dinov3", "cnx", "cnx_paper", "dinov2"):
        d = []
        for k in GROUPS[g]:
            I, U, T = per[k][:, 0], per[k][:, 1], per[k][:, 2]
            pred = (U - T + I)[:, 1:]  # background excluded
            choice = np.where(pred.sum(1) > 0, pred.argmax(1) + 1, 0)  # no foreground -> 0 = wrong
            d.append((choice == true).astype(float))
        correct[g] = np.mean(d, axis=0)  # mean over seeds, per image
    n = len(stems)
    W, one = weights(rng, n, B), np.ones(n)

    def rate(v):
        return lambda w: (w @ v) / w.sum(1) * 100
    out = {"accuracy": {g: {k: x for k, x in summary(rate(v), W, one).items() if k != "p"} for g, v in correct.items()}}
    for name, (g1, g2) in CONTRASTS.items():
        out[name] = summary(rate(correct[g1] - correct[g2]), W, one)
    return out


def class_agnostic(stems, clean_set, B, rng):
    """Foreground / background binary mask IoU (results/extra/<run>_test_fg.npz; src/extra_inference.py, test only)."""
    fg = {}
    for g in ("dinov3", "cnx", "cnx_paper", "dinov2"):
        zs = [np.load(ROOT / f"results/extra/{k}_test_fg.npz") for k in GROUPS[g]]
        assert all(list(z["stems"]) == list(stems) for z in zs)
        fg[g] = [z["fg"].astype(np.float64) for z in zs]
    clean = np.array([s in clean_set for s in stems])
    subsets = {"all": None, "clean": clean, "leaked": ~clean}
    n = len(stems)

    def iou(W, g):  # mean over seeds
        return np.mean([(W @ f[:, 0]) / (W @ f[:, 1]) * 100 for f in fg[g]], axis=0)

    out = {"iou": {}, "diffs": {}}
    for a, m in subsets.items():
        W = weights(rng, n, B, m)
        one = np.ones(n) if m is None else m.astype(float)
        out["iou"][a] = {g: {k: x for k, x in summary(lambda w, g=g: iou(w, g), W, one).items() if k != "p"} for g in fg}
        for name, (g1, g2) in {**CONTRASTS, "DINOv3 − ConvNeXt (paper recipe)": ("dinov3", "cnx_paper")}.items():
            out["diffs"].setdefault(name, {})[a] = summary(lambda w, g1=g1, g2=g2: iou(w, g1) - iou(w, g2), W, one)
    return out


def leakage_table(lk):
    n = lk["n"]
    return {
        "hamming_threshold": lk["hamming_threshold"],
        "dev": {"n": n["val"], "leaked": lk["val_imgs_with_dup_in_train"],
                "ratio": round(100 * lk["val_imgs_with_dup_in_train"] / n["val"], 1)},
        "test": {"n": n["test"], "dup_in_train": lk["test_imgs_with_dup_in_train"],
                 "dup_in_dev": lk["test_imgs_with_dup_in_val"], "leaked_total": n["test"] - lk["clean_test_n"],
                 "ratio": round(100 * (n["test"] - lk["clean_test_n"]) / n["test"], 1)},
        "duplicate_pairs": {"same_class": lk["pairs_same_class"], "diff_class": lk["pairs_diff_class"]},
    }


def label_inconsistency(prep, meta):
    """(a) near-duplicates with different class labels, (b) mask ratio outliers, (c) mask value ≠ meta class (train+val)."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from leakage import THRESHOLD, _bits, _hamming
    ph = json.loads((prep / "phash.json").read_text())
    B = _bits([h for _, _, h in ph])
    pairs = []
    for i in range(0, len(B), 512):
        for r, j in zip(*np.nonzero(_hamming(B[i:i + 512], B) <= THRESHOLD)):
            a, b = ph[i + r], ph[j]
            if j > i + r and meta[a[1]]["cls"] != meta[b[1]]["cls"]:
                pairs.append((a[1], b[1]))
    same_plant = sum(meta[a]["plant"] == meta[b]["plant"] for a, b in pairs)
    class_pair = collections.Counter(tuple(sorted((meta[a]["disease"], meta[b]["disease"]))) for a, b in pairs)

    # (b) robust z (median/MAD) of log(mask ratio) within each class; empty and near-full masks counted separately
    by = collections.defaultdict(list)
    for s, m in meta.items():
        by[m["cls"]].append((s, m["mask_ratio"]))
    empty = [s for s, m in meta.items() if m["mask_ratio"] < 1e-3]
    full = [s for s, m in meta.items() if m["mask_ratio"] > 0.95]
    outlier = []
    for c, v in by.items():
        x = np.log(np.clip([r for _, r in v], 1e-4, 1))
        mad = np.median(np.abs(x - np.median(x))) * 1.4826
        if mad > 0:
            outlier += [s for (s, _), z in zip(v, (x - np.median(x)) / mad) if abs(z) > 3.5]

    # (c) do the foreground values in the label PNG match the meta class (test labels are not available locally)
    from PIL import Image
    mismatch = []
    for sp in ("train", "val"):
        # lab/ is a nearest-neighbour downscale: small lesion values can drop out; full resolution is lab_orig/
        paths = sorted((prep / sp / "lab_orig").glob("*.png")) or sorted((prep / sp / "lab").glob("*.png"))
        n_meta = sum(m["split"] == sp for m in meta.values())
        assert len(paths) == n_meta, f"{prep / sp}: {len(paths)} label PNGs, {n_meta} images in meta"
        for p in paths:
            u = set(np.unique(np.array(Image.open(p))).tolist()) - {0, 255}
            if u != {meta[p.stem]["cls"]}:
                mismatch.append({"stem": p.stem, "meta": meta[p.stem]["cls"], "mask": sorted(u)})
    return {
        "diff_class_duplicates": {"pairs": len(pairs), "same_plant": same_plant, "diff_plant": len(pairs) - same_plant,
                                  "images": len({s for c in pairs for s in c}),
                                  "most_common": [[a, b, k] for (a, b), k in class_pair.most_common(10)],
                                  "example": pairs[:20]},
        "mask_ratio": {"empty(<0.1%)": len(empty), "near_full(>95%)": len(full),
                       "within_class_outlier(|z|>3.5)": len(outlier),
                       "example_empty": empty[:10], "example_full": full[:10], "example_outlier": outlier[:20]},
        "mask_meta_mismatch": {"n": len(mismatch), "list": mismatch},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=str(pathlib.Path.home() / ".cache/plantseg/prep"))
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    prep = pathlib.Path(a.prep)
    meta = json.loads((prep / "meta.json").read_text())
    rng = np.random.default_rng(a.seed)
    out = {}
    for sp in ("test", "dev"):
        per, stems = load(sp)
        out[sp] = {"h3": h3(sp, per, stems, meta, a.B, rng),
                   "class_accuracy": class_accuracy(sp, per, stems, meta, a.B, rng)}
    lk = json.loads((ROOT / "results/leakage.json").read_text())
    out["test"]["class_agnostic"] = class_agnostic(load("test")[1], set(lk["clean_test"]), a.B, rng)
    out["leakage"] = leakage_table(lk)
    out["label"] = label_inconsistency(prep, meta)
    path = ROOT / "results/analysis/extra_analysis.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for sp in ("test", "dev"):
        print(f"\n== {sp}")
        for name, d in out[sp]["h3"]["diffs"].items():
            print(f"  H3 {name}:", "  ".join(f"{k}: {v['point']} {v['ci95']} p={v['p']}" for k, v in d.items()))
        ca = out[sp]["class_accuracy"]
        print("  class accuracy:", "  ".join(f"{g}: {v['point']} {v['ci95']}" for g, v in ca["accuracy"].items()))
        for name in CONTRASTS:
            print(f"  class accuracy {name}: {ca[name]}")
        if "class_agnostic" in out[sp]:
            cg = out[sp]["class_agnostic"]
            print("  class-agnostic IoU:", {a: {g: v["point"] for g, v in d.items()} for a, d in cg["iou"].items()})
            for name, d in cg["diffs"].items():
                print(f"  class-agnostic {name}:", "  ".join(f"{k}: {v['point']} {v['ci95']} p={v['p']}" for k, v in d.items()))
    print("\nleakage:", json.dumps(out["leakage"], ensure_ascii=False))
    e = out["label"]
    print("label:", {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in e.items()})
    print("most common different-class duplicate pairs:", e["diff_class_duplicates"]["most_common"][:5])
    print(f"\n-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
