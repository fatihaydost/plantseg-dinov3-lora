"""README figures from the committed results (no GPU, no dataset needed).

Usage: uv run --with numpy --with matplotlib python3 analysis/figures.py
Writes results/figures/fig*.png.
"""
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CNX, V3, V2, PAPER = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"  # categorical slots 1-3 + neutral
NEG, POS = "#2a78d6", "#eb6834"  # diverging poles (ConvNeXt better / DINOv3 better)

MODELS = [  # (label, run ids, colour)
    ("ConvNeXt-L (our protocol)", ["b_cnxL_s0", "f_cnxL_s1", "g_cnxL_s2"], CNX),
    ("ConvNeXt-L (paper recipe)", ["c_cnxL_mak_s0"], PAPER),
    ("DINOv3 ViT-L + LoRA", ["a_v3L_lora_s0", "d_v3L_lora_s1", "e_v3L_lora_s2"], V3),
    ("DINOv3 + LoRA + logit adj. (H2, τ tuned on dev)", ["i_H2_s0"], V3),
    ("DINOv2 ViT-L + LoRA", ["h_v2L_lora_s0"], V2),
]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def run(split, rid):
    return json.loads((R / split / f"{rid}.json").read_text())


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def fig_main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, split, title in zip(axes, ["dev", "test"], ["Dev (846 images)", "Test (1,561 images)"]):
        for y, (label, ids, col) in enumerate(MODELS[::-1]):
            v = [run(split, i)["mIoU"] for i in ids]
            ax.scatter(v, [y] * len(v), s=36, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            m = np.mean(v)
            ax.plot([m, m], [y - 0.28, y + 0.28], color=INK, lw=2, zorder=4)
            ax.text(m, y + 0.34, f"{m:.2f}" + (f" ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else ""),
                    ha="center", va="bottom", fontsize=8.5, color=INK)
        if split == "test":
            for x, name in [(46.24, "PlantSeg paper"), (48.48, "TaxoSeg")]:
                ax.axvline(x, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
                ax.text(x, -0.55, name, rotation=90, ha="right", va="bottom", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("mIoU (%)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(41, 51.5)
    axes[0].set_yticks(range(len(MODELS)), [m[0] for m in MODELS[::-1]])
    axes[0].set_ylim(-0.6, len(MODELS) - 0.2)
    fig.suptitle("Dots: individual seeds · black tick: mean (± sd over seeds)", x=0.99, ha="right", y=0.99,
                 fontsize=8.5, color=INK2)
    save(fig, "fig1_main_results.png")


def fig_curves():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    seen = set()
    for label, ids, col in [MODELS[0], MODELS[2], MODELS[4]]:
        for i in ids:
            rows = list(csv.DictReader(open(R / "train_curves" / f"{i}.csv")))
            it = {int(r["iter"]): float(r["loss"]) for r in rows}  # resumed runs repeat iterations; keep last
            x = np.array(sorted(it))
            y = np.array([it[k] for k in x])
            k = 9  # moving average over ~1.2k iterations
            ys = np.convolve(y, np.ones(k) / k, mode="valid")
            ax.plot(x[k - 1:], ys, color=col, lw=1.6, alpha=0.9, label=None if label in seen else label)
            seen.add(label)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration (batch 16)")
    ax.set_ylabel("Training loss (log, smoothed)")
    ax.set_title("Training loss, 40k iterations; one line per seed", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "fig4_training_curves.png")


def fig_per_class():
    def mean_iou(ids):
        d = [run("test", i)["class_IoU"] for i in ids]
        return {c: np.mean([np.nan if x[c] is None else x[c] for x in d]) for c in d[0] if c != "background"}  # None: class absent from test GT
    cnx, v3 = mean_iou(MODELS[0][1]), mean_iou(MODELS[2][1])
    diff = sorted(((v3[c] - cnx[c], c) for c in cnx if not (np.isnan(v3[c]) or np.isnan(cnx[c]))), reverse=True)
    d = np.array([x for x, _ in diff])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(range(len(d)), d, width=0.8, color=[POS if x > 0 else NEG for x in d], edgecolor=SURFACE, linewidth=0.5)
    ax.axhline(0, color=INK2, lw=0.8)
    for j in [0, 1, 2, len(d) - 1]:
        name = diff[j][1].split("|")[1]
        ax.annotate(name, (j, d[j]), xytext=(4 if j < 3 else -4, 0), textcoords="offset points",
                    ha="left" if j < 3 else "right", va="center", fontsize=7.5, color=INK2)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(d)} disease classes present in test, sorted by difference ({(d > 0).sum()} favour DINOv3, {(d < 0).sum()} favour ConvNeXt)")
    ax.set_ylabel("Δ IoU (points)")
    ax.set_title("Per-class test IoU: DINOv3 + LoRA − ConvNeXt-L (mean of 3 seeds each)", loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    save(fig, "fig5_per_class_delta.png")


def fig_leakage_counts():
    lk = json.loads((R / "leakage.json").read_text())
    n = lk["n"]
    rows = [("Dev", n["val"], lk["val_imgs_with_dup_in_train"], 0),
            ("Test", n["test"], lk["test_imgs_with_dup_in_train"], n["test"] - lk["clean_test_n"] - lk["test_imgs_with_dup_in_train"])]
    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    for y, (name, tot, tr, va) in enumerate(rows[::-1]):
        ax.barh(y, tr / tot * 100, color=V3, height=0.55, edgecolor=SURFACE, linewidth=2, label="duplicate in train" if y == 0 else None)
        ax.barh(y, va / tot * 100, left=tr / tot * 100, color=PAPER, height=0.55, edgecolor=SURFACE, linewidth=2,
                label="duplicate only in dev" if y == 0 else None)
        ax.text((tr + va) / tot * 100 + 0.3, y, f"{tr + va} / {tot} ({(tr + va) / tot * 100:.1f}%)", va="center", fontsize=9)
    ax.set_yticks([0, 1], [r[0] for r in rows[::-1]])
    ax.set_xlim(0, 20)
    ax.set_xlabel("Evaluation images with a near-duplicate (pHash Hamming ≤ 4) in an earlier split (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2)
    save(fig, "fig2_leakage_rate.png")



def fig_subgroups():
    ex = json.loads((R / "analysis/extra_analysis.json").read_text())
    st = json.loads((R / "analysis/stats.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    groups = ["head", "mid", "tail", "small lesion", "large lesion"]
    for split, col, off in [("dev", PAPER, 0.15), ("test", V3, -0.15)]:
        d = ex[split]["h3"]["diffs"]["DINOv3 − ConvNeXt"]
        for y, g in enumerate(groups[::-1]):
            lo, hi = d[g]["ci95"]
            a.plot([lo, hi], [y + off] * 2, color=col, lw=2, solid_capstyle="round")
            a.scatter(d[g]["point"], y + off, s=40, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                      label=split.capitalize() if y == 0 else None)
    a.set_yticks(range(len(groups)), [g.capitalize() + (" classes" if "lesion" not in g else "s") for g in groups[::-1]])
    a.axvline(0, color=INK2, lw=0.8)
    a.set_xlabel("Δ mIoU, DINOv3 + LoRA − ConvNeXt-L (BCa 95% CI)")
    a.set_title("(a) Gain by class frequency and lesion size", loc="left", fontsize=10.5)
    a.grid(axis="y", visible=False)
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    t = st["test"]["diffs"]
    rows = [("DINOv3 − ConvNeXt, clean test", t["H1: DINOv3 − ConvNeXt (ours)"]["clean"], V3),
            ("DINOv3 − ConvNeXt, leaked test", t["H1: DINOv3 − ConvNeXt (ours)"]["leaked"], V3),
            ("ConvNeXt: leaked − clean", t["cnx: leaked − clean"], CNX),
            ("DINOv3: leaked − clean", t["dinov3: leaked − clean"], V3)]
    for y, (label, d, col) in enumerate(rows[::-1]):
        b.plot(d["ci95"], [y, y], color=col, lw=2, solid_capstyle="round")
        b.scatter(d["point"], y, s=40, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        b.text(d["ci95"][1] + 0.4, y, f"{d['point']:+.2f}", va="center", fontsize=8.5)
    b.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    b.axvline(0, color=INK2, lw=0.8)
    b.set_xlabel("Δ mIoU on test (BCa 95% CI)")
    b.set_title("(b) Leakage: the CNN profits from seen photos", loc="left", fontsize=10.5)
    b.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=3)
    save(fig, "fig3_subgroups.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_main()
    fig_leakage_counts()
    fig_curves()
    fig_per_class()
    fig_subgroups()
