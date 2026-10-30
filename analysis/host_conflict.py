"""Classifies near-duplicate pairs with different class labels by host plant species (the part needing no expert).

One photo cannot show two different botanical species: if the host species differ, at least one label is wrong.
- host_conflict: different botanical species (e.g. plum / peach, apple / grape)
- same_species_diff_name: same species, different cultivar name (broccoli / cabbage / cauliflower = B. oleracea)
- same_plant_diff_disease: same plant name, different disease; one leaf can carry two diseases, the right label
  needs an expert
Species names: accepted names in POWO (powo.science.kew.org). PlantSeg's "Squash" is an unspecified Cucurbita;
it may be the same species as zucchini (C. pepo), so it counts as "same species" (the conservative side).
Output: results/label_audit/host_classification.{csv,json}
Usage: uv run --with numpy --with pillow python3 analysis/host_conflict.py
"""
import collections
import csv
import json
import pathlib

from label_audit import PREP, ROOT, pairs

SPECIES = {
    "Apple": "Malus domestica", "Banana": "Musa spp.", "Basil": "Ocimum basilicum", "Bean": "Phaseolus vulgaris",
    "Bell pepper": "Capsicum annuum", "Blueberry": "Vaccinium spp.", "Broccoli": "Brassica oleracea",
    "Cabbage": "Brassica oleracea", "Carrot": "Daucus carota", "Cauliflower": "Brassica oleracea",
    "Celery": "Apium graveolens", "Cherry": "Prunus spp. (cherry)", "Citrus": "Citrus spp.", "Coffee": "Coffea spp.",
    "Corn": "Zea mays", "Cucumber": "Cucumis sativus", "Eggplant": "Solanum melongena", "Garlic": "Allium sativum",
    "Ginger": "Zingiber officinale", "Grape": "Vitis vinifera", "Lettuce": "Lactuca sativa", "Maple": "Acer spp.",
    "Peach": "Prunus persica", "Plum": "Prunus domestica", "Potato": "Solanum tuberosum", "Raspberry": "Rubus idaeus",
    "Rice": "Oryza sativa", "Soybean": "Glycine max", "Squash": "Cucurbita pepo", "Strawberry": "Fragaria × ananassa",
    "Tobacco": "Nicotiana tabacum", "Tomato": "Solanum lycopersicum", "Wheat": "Triticum aestivum",
    "Zucchini": "Cucurbita pepo",
}


def category(ma, mb):
    if ma["plant"] == mb["plant"]:
        return "same_plant_diff_disease"
    return "same_species_diff_name" if SPECIES[ma["plant"]] == SPECIES[mb["plant"]] else "host_conflict"


def main():
    ph = json.loads((PREP / "phash.json").read_text())
    meta = json.loads((PREP / "meta.json").read_text())
    assert set(SPECIES) >= {m["plant"] for m in meta.values()}, "a plant is missing from the species table"
    rows = []
    for (sa, a, _), (sb, b, _), h in pairs(ph, meta):
        ma, mb = meta[a], meta[b]
        rows.append({"A": a, "A_split": sa, "A_label": f"{ma['plant']} | {ma['disease']}", "A_species": SPECIES[ma["plant"]],
                     "B": b, "B_split": sb, "B_label": f"{mb['plant']} | {mb['disease']}", "B_species": SPECIES[mb["plant"]],
                     "hamming": h, "category": category(ma, mb)})
    out = ROOT / "results/label_audit"
    with open(out / "host_classification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    count = collections.Counter(s["category"] for s in rows)
    splits = collections.Counter(tuple(sorted((s["A_split"], s["B_split"])))
                                 for s in rows if s["category"] == "host_conflict")
    summary = {"pairs": len(rows), "category": dict(count),
               "category_hamming0": dict(collections.Counter(s["category"] for s in rows if s["hamming"] == 0)),
               "host_conflict_split_pairs": {"↔".join(k): v for k, v in splits.items()},
               "host_conflict_images": len({x for s in rows if s["category"] == "host_conflict" for x in (s["A"], s["B"])}),
               "most_common_species_pairs": [[*k, v] for k, v in collections.Counter(
                   tuple(sorted((s["A_species"], s["B_species"]))) for s in rows
                   if s["category"] == "host_conflict").most_common(8)]}
    (out / "host_classification.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    assert sum(count.values()) == len(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
