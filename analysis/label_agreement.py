"""Label audit: agreement between two raters on "same photo?" (pair level and photo-cluster level).

Raters: rater_human (a human, results/label_audit/form.csv) and rater_model (an LLM agent, rater_model.csv), each
blind to the other's answers and to the Hamming distances (key.csv).
Output: results/label_audit/agreement.json. When almost every decision falls in one category, Cohen's kappa is
undefined or misleading (prevalence paradox); percent agreement is primary, kappa is written only when defined.
Usage: python3 analysis/label_agreement.py
"""
import collections
import csv
import json
import pathlib

D = pathlib.Path(__file__).resolve().parent.parent / "results/label_audit"


def wilson_lower(k, n, z=1.96):
    """95% Wilson lower bound of a proportion (meaningful also when k/n = 1)."""
    p = k / n
    return round((p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / (1 + z * z / n), 3)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = collections.Counter(a), collections.Counter(b)
    pe = sum(pa[k] * pb[k] for k in set(a) | set(b)) / n ** 2
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def main():
    form = list(csv.DictReader(open(D / "form.csv", newline="")))
    model = {r["no"]: r["same_photo"] for r in csv.DictReader(open(D / "rater_model.csv", newline=""))}
    hamming = {r["no"]: int(r["hamming"]) for r in csv.DictReader(open(D / "key.csv", newline=""))}
    human = {r["no"]: r["same_photo (Y/N)"] for r in form}
    assert set(human) == set(model) and all(v in ("Y", "N") for v in (*human.values(), *model.values())), \
        "missing or invalid answer"

    root = {}  # pairs sharing an image form one photo cluster

    def find(x):
        while root.setdefault(x, x) != x:
            x = root[x]
        return x
    for r in form:
        root[find(r["A"])] = find(r["B"])
    cluster = collections.defaultdict(list)
    for r in form:
        cluster[find(r["A"])].append(r["no"])

    nos = sorted(human, key=int)
    a, b = [human[k] for k in nos], [model[k] for k in nos]
    ka = ([all(human[k] == "Y" for k in v) for v in cluster.values()],
          [all(model[k] == "Y" for k in v) for v in cluster.values()])
    out = {
        "pairs": len(nos), "agreement": sum(x == y for x, y in zip(a, b)),
        "agreement_pct": round(100 * sum(x == y for x, y in zip(a, b)) / len(nos), 1),
        "kappa": kappa(a, b), "rater_human": dict(collections.Counter(a)), "rater_model": dict(collections.Counter(b)),
        "mismatched": [k for k in nos if human[k] != model[k]],
        "cluster": len(cluster), "cluster_both_yes": sum(x and y for x, y in zip(*ka)),
        "same_photo_rate_wilson95_lower": {"pairs": wilson_lower(sum(x == y == "Y" for x, y in zip(a, b)), len(nos)),
                                           "cluster": wilson_lower(sum(x and y for x, y in zip(*ka)), len(cluster))},
        "hamming": {h: {"pairs": sum(hamming[k] == h for k in nos),
                        "both_yes": sum(hamming[k] == h and human[k] == model[k] == "Y" for k in nos)}
                    for h in sorted(set(hamming.values()))},
    }
    (D / "agreement.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
