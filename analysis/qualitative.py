"""Example prediction figures: image | ground truth | ConvNeXt-L | DINOv3 + LoRA | DINOv2 + LoRA (all s0).

Input:  results/extra/predictions/<run>/<stem>.png (src/extra_inference.py; class map at the original label
        resolution), raw image and label (--raw: PlantSeg images/test, annotations/test), prep/meta.json (class names).
Colour: blue = ground truth; green = foreground labelled with the correct class, orange = another disease class.
Panel title: image-level mIoU (over the classes present in that image, results/test/*_per_image.npz).
Output: results/figures/examples_random.png (first 12: clean test, head/mid/tail × 4, seed 0) and
        examples_extremes.png (last 6).
Usage:  uv run --with numpy --with pillow --with matplotlib python3 analysis/qualitative.py
"""
import argparse
import json
import pathlib

import numpy as np
from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = [("b_cnxL_s0", "ConvNeXt-L"), ("a_v3L_lora_s0", "DINOv3 + LoRA"), ("h_v2L_lora_s0", "DINOv2 + LoRA")]
# same order as the figure stem list used by the extra-inference notebook
FIGURE_STEMS = ("coffee_leaf_rust_google_0135,grape_downy_mildew_251,wheat_loose_smut_Bing_0178,soybean_mosaic_Bing_0010,"
                "apple_black_rot_7,tomato_yellow_leaf_curl_virus_44,bell_pepper_bacterial_spot_Bing_0008,rice_sheath_blight_148,"
                "lettuce_mosaic_virus_8,strawberry_leaf_scorch_google_0186,cherry_powdery_mildew_76,"
                "eggplant_phytophthora_blight_Google_0028,soybean_mosaic_Bing_0076,coffee_berry_blotch_Bing_0027,"
                "celery_early_blight_9,cucumber_powdery_mildew_59,cucumber_powdery_mildew_google_0000,"
                "cucumber_angular_leaf_spot_9").split(",")
BLUE, GREEN, ORANGE = np.array([40, 110, 255]), np.array([30, 200, 70]), np.array([255, 140, 0])


def overlay(img, mask, colour, a=0.55):
    out = img.astype(np.float32).copy()
    out[mask] = (1 - a) * out[mask] + a * colour
    return out.astype(np.uint8)


def image_miou(run):
    z = np.load(ROOT / f"results/test/{run}_test_per_image.npz")
    I, U = z["per"][:, 0].astype(float), z["per"][:, 1].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.nanmean(np.where(U > 0, I / U, np.nan), axis=1) * 100
    return dict(zip(z["stems"], m))


def draw(stems, meta, raw, mious, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 2 + len(RUNS)
    fig, axs = plt.subplots(len(stems), cols, figsize=(2.6 * cols, 2.3 * len(stems)), squeeze=False)
    for i, s in enumerate(stems):
        img = np.array(ImageOps.exif_transpose(Image.open(raw / "images/test" / f"{s}.jpg")).convert("RGB"))
        gt = np.array(Image.open(raw / "annotations/test" / f"{s}.png"))
        gt = gt[..., 0] if gt.ndim == 3 else gt
        assert gt.shape == img.shape[:2], (s, gt.shape, img.shape)
        cls = meta[s]["cls"]
        panels = [(img, meta[s]["disease"]), (overlay(img, (gt > 0) & (gt != 255), BLUE), "ground truth")]
        for k, name in RUNS:
            pr = np.array(Image.open(ROOT / "results/extra/predictions" / k / f"{s}.png"))
            assert pr.shape == gt.shape, (k, s)
            p = overlay(img, pr == cls, GREEN)
            p = overlay(p, (pr > 0) & (pr != cls), ORANGE)
            panels.append((p, f"{name}  {mious[k][s]:.0f}"))
        for j, (p, t) in enumerate(panels):
            axs[i, j].imshow(p)
            axs[i, j].set_title(t, fontsize=8)
            axs[i, j].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("->", path.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(pathlib.Path.home() / ".cache/plantseg/raw/plantseg"))
    ap.add_argument("--prep", default=str(pathlib.Path.home() / ".cache/plantseg/prep"))
    a = ap.parse_args()
    meta = json.loads((pathlib.Path(a.prep) / "meta.json").read_text())
    mious = {k: image_miou(k) for k, _ in RUNS}
    raw = pathlib.Path(a.raw)
    draw(FIGURE_STEMS[:12], meta, raw, mious, ROOT / "results/figures/examples_random.png",
         "12 random clean-test images (head / mid / tail × 4). Green: correct class, orange: wrong class, blue: ground truth")
    draw(FIGURE_STEMS[12:], meta, raw, mious, ROOT / "results/figures/examples_extremes.png",
         "Extremes: images where DINOv3 leads ConvNeXt the most (first 3) and the least (last 3) (s0)")


if __name__ == "__main__":
    main()
