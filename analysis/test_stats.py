"""Self-check for stats.py. Usage: python3 analysis/test_stats.py"""
import numpy as np

from stats import gt_classes, miou, summary


def test_non_gt_class_excluded_from_mean():
    # 2 images, 3 classes: class 2 is absent from GT but wrongly predicted in image 1 (I=0, U>0)
    per = {"k": np.array([[[5, 5, 0], [10, 10, 4], [10, 10, 0]],
                          [[5, 5, 0], [10, 10, 0], [10, 10, 0]]], float)}  # [N, (I,U,T), C]
    w = np.ones(2)
    S = gt_classes(per, w)
    assert S.tolist() == [True, True, False]
    I, U = w @ per["k"][:, 0], w @ per["k"][:, 1]
    assert np.isclose(miou(I, U), 100 / 3)  # old rule: the wrongly predicted class enters as 0
    assert np.isclose(miou(I, U, S), 50)  # new rule: GT classes only


def test_bca_p_consistent_with_interval():
    rng = np.random.default_rng(1)
    for shift in (0.0, 0.1, 0.25, 0.5):
        x = rng.normal(shift, 1, 200)
        W = np.zeros((4000, 200))
        for b in range(4000):
            np.add.at(W[b], rng.integers(0, 200, 200), 1)
        o = summary(lambda w: (w @ x) / w.sum(1), W, np.ones(200))
        lo, hi = o["ci95"]
        assert abs(o["point"] - x.mean()) < 0.01
        # 0 outside the interval <=> p < 0.05 (with a rounding margin)
        if lo > 0.02 or hi < -0.02:
            assert o["p"] < 0.05, o
        if lo < -0.02 and hi > 0.02:
            assert o["p"] > 0.05, o
        # symmetric case: BCa ≈ mean ± 1.96·SE
        se = x.std(ddof=1) / np.sqrt(200)
        assert abs((hi - lo) - 2 * 1.96 * se) < 0.06, (hi - lo, 2 * 1.96 * se)


if __name__ == "__main__":
    test_non_gt_class_excluded_from_mean()
    test_bca_p_consistent_with_interval()
    print("ok")
