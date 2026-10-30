"""Regenerates the run tables from results/{dev,test}/*.json.

mIoU, mAcc and training time come from the JSON; tail, small lesion, clean and leaked come from
<run>_<split>_per_image.npz, using the subset's GT classes (same rule as src/common.py miou_subset; older JSONs used
the mmseg rule). Needs the local prepared meta (--prep).
Usage: python3 analysis/tables.py [--prep ~/.cache/plantseg/prep] [--print]
Rewrites README.md between the RUN-TABLE / TEST-TABLE markers if they exist; --print writes the tables to stdout.
"""
import argparse
import collections
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
MARKERS = {"dev": ("<!-- RUN-TABLE:BEGIN -->", "<!-- RUN-TABLE:END -->"),
           "test": ("<!-- TEST-TABLE:BEGIN -->", "<!-- TEST-TABLE:END -->")}


def num(v):
    return "—" if v is None else f"{v:.2f}"


def subsets(per, stems, meta, clean):
    """per [N,3,C] -> subset mIoUs; each subset uses only classes with GT pixels in it."""
    def m(mask, cls=None):
        I, U, T = per[mask].sum(0)
        S = T > 0
        if cls is not None:
            S &= np.isin(np.arange(len(S)), cls)
        return float((I[S] / U[S]).mean() * 100) if S.any() else None
    cnt = collections.Counter(v["cls"] for v in meta.values() if v["split"] == "train")
    order = sorted(range(1, per.shape[2]), key=lambda c: -cnt[c])
    t = len(order) // 3
    every = np.ones(len(stems), bool)
    mr = np.array([meta[s]["mask_ratio"] for s in stems])
    small = mr < np.median(mr)
    cl = np.array([s in clean for s in stems])
    return {"mIoU_head": m(every, order[:t]), "mIoU_mid": m(every, order[t:2 * t]), "mIoU_tail": m(every, order[2 * t:]),
            "mIoU_small_lesion": m(small), "mIoU_large_lesion": m(~small), "mIoU_clean": m(cl), "mIoU_leaked": m(~cl)}


def npz_subsets(split, run_id, meta, clean):
    z = np.load(ROOT / f"results/{split}/{run_id}_{split}_per_image.npz")
    return subsets(z["per"].astype(np.float64), list(z["stems"]), meta, clean[split])


def table(split, meta, clean):
    rows = [
        "| Run | Model | mIoU | mAcc | Tail | Small lesion | Clean | Leaked | Train (h) | GPU |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in sorted((ROOT / "results" / split).glob("*.json")):
        d = json.loads(p.read_text())
        dev = ROOT / "results/dev" / p.name   # test JSONs lack label and training info; take them from dev
        if split != "dev" and dev.exists():
            d = {**json.loads(dev.read_text()), **d}
        d.update(npz_subsets(split, d["id"], meta, clean))
        gpu = d.get("gpu", "").replace("NVIDIA ", "").replace("-SXM4", "")
        rows.append(
            f"| `{d['id']}` | {d.get('label', '')} | **{num(d.get('mIoU'))}** | {num(d.get('mAcc'))} | "
            f"{num(d.get('mIoU_tail'))} | {num(d.get('mIoU_small_lesion'))} | {num(d.get('mIoU_clean'))} | "
            f"{num(d.get('mIoU_leaked'))} | {num(d.get('train_h'))} | {gpu} |"
        )
    return rows


def clean_sets():
    lk = json.loads((ROOT / "results/leakage.json").read_text())
    leaked_dev = set(lk["val_leak_stems"])
    dev = np.load(next((ROOT / "results/dev").glob("*_dev_per_image.npz")))["stems"]
    return {"test": set(lk["clean_test"]), "dev": {s for s in dev if s not in leaked_dev}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=str(pathlib.Path.home() / ".cache/plantseg/prep"))
    ap.add_argument("--print", action="store_true", help="print the tables to stdout")
    a = ap.parse_args()
    meta = json.loads((pathlib.Path(a.prep) / "meta.json").read_text())
    clean = clean_sets()
    text = README.read_text() if README.exists() else None
    for split, (begin_m, end_m) in MARKERS.items():
        rows = table(split, meta, clean)
        if a.print:
            print(f"\n{split}:\n" + "\n".join(rows))
        if text is not None and begin_m in text and end_m in text:
            begin, end = text.index(begin_m) + len(begin_m), text.index(end_m)
            text = text[:begin] + "\n" + "\n".join(rows) + "\n" + text[end:]
            print(f"{split}: {len(rows) - 2} runs -> README.md")
    if text is not None:
        README.write_text(text)


if __name__ == "__main__":
    main()
