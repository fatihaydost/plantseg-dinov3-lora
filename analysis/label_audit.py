"""Sample, contact sheets and blank form for the manual audit of near-duplicate pairs with different class labels.

Sample (seed 2): 32 random Hamming-0 pairs + all Hamming 2-4 pairs (28) = 60; order shuffled, Hamming distance not
shown on the sheets (so it does not influence the rater).
Output: results/label_audit/{sheet_1..6.png, form.csv (blank), key.csv (pair no -> Hamming; opened after rating)}.
form.csv and key.csv are only written if absent, so re-running never overwrites collected answers.
Usage: uv run --with numpy --with pillow --with matplotlib python3 analysis/label_audit.py
"""
import csv
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from leakage import THRESHOLD, _bits, _hamming  # noqa: E402

PREP = pathlib.Path.home() / ".cache/plantseg/prep"
RAW = pathlib.Path.home() / ".cache/plantseg/raw/plantseg"
OUT = ROOT / "results/label_audit"


def pairs(ph, meta):
    B = _bits([h for _, _, h in ph])
    out = []
    for i in range(0, len(B), 512):
        H = _hamming(B[i:i + 512], B)
        for r, j in zip(*np.nonzero(H <= THRESHOLD)):
            if j > i + r and meta[ph[i + r][1]]["cls"] != meta[ph[j][1]]["cls"]:
                out.append((ph[i + r], ph[j], int(H[r, j])))
    return out


def load(sp, s):
    im = np.array(ImageOps.exif_transpose(Image.open(RAW / "images" / sp / f"{s}.jpg")).convert("RGB"))
    lab = np.array(Image.open(RAW / "annotations" / sp / f"{s}.png"))
    lab = lab[..., 0] if lab.ndim == 3 else lab
    o = im.astype(np.float32)
    m = (lab > 0) & (lab != 255)
    o[m] = 0.55 * o[m] + 0.45 * np.array([40, 110, 255])
    return im, o.astype(np.uint8)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ph = json.loads((PREP / "phash.json").read_text())
    meta = json.loads((PREP / "meta.json").read_text())
    every = pairs(ph, meta)
    rng = np.random.default_rng(2)
    h0 = [c for c in every if c[2] == 0]
    sel = [h0[k] for k in rng.choice(len(h0), 32, replace=False)] + [c for c in every if c[2] > 0]
    assert len(sel) == 60, len(sel)
    sel = [sel[k] for k in rng.permutation(len(sel))]
    OUT.mkdir(parents=True, exist_ok=True)

    for sheet in range(6):
        fig, axs = plt.subplots(10, 4, figsize=(13, 30), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
        for r in range(10):
            no = sheet * 10 + r + 1
            a, b, _ = sel[no - 1]
            for k, (sp, s, _) in enumerate((a, b)):
                im, ov = load(sp, s)
                m = meta[s]
                axs[r, 2 * k].imshow(im)
                axs[r, 2 * k].set_title(f"#{no}{'AB'[k]}  {m['plant']} | {m['disease']}", fontsize=9, loc="left")
                axs[r, 2 * k + 1].imshow(ov)
                axs[r, 2 * k + 1].set_title("mask", fontsize=8)
            for x in axs[r]:
                x.axis("off")
        fig.suptitle(f"Label audit, sheet {sheet + 1}/6 (pairs {sheet * 10 + 1}-{sheet * 10 + 10})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.savefig(OUT / f"sheet_{sheet + 1}.png", dpi=80)
        plt.close(fig)

    if not (OUT / "form.csv").exists():
        with open(OUT / "form.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["no", "A", "A_label", "B", "B_label", "same_photo (Y/N)", "note"])
            for no, (a, b, _) in enumerate(sel, 1):
                w.writerow([no, a[1], f"{meta[a[1]]['plant']} | {meta[a[1]]['disease']}",
                            b[1], f"{meta[b[1]]['plant']} | {meta[b[1]]['disease']}", "", ""])
    if not (OUT / "key.csv").exists():
        with open(OUT / "key.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["no", "A", "A_split", "B", "B_split", "hamming"])
            for no, (a, b, h) in enumerate(sel, 1):
                w.writerow([no, a[1], a[0], b[1], b[0], h])
    print("->", OUT.relative_to(ROOT), "| all different-class pairs:", len(every), "| H0:", len(h0))


if __name__ == "__main__":
    main()
