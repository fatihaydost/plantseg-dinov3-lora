"""Near-duplicate (64-bit pHash, Hamming <= 4) analysis and the clean list. Same definition as the pilot's leakage.json:
a dev/test image is 'leaked' if it has an image within Hamming <= 4 in train. Image hashes only; no labels, no model.
Usage: python leakage.py --phash prep/phash.json --meta prep/meta.json --out prep/leakage.json"""
import argparse, json
import numpy as np

THRESHOLD = 4


def phash_one(p):
    import imagehash
    from PIL import Image
    return str(imagehash.phash(Image.open(p).convert('RGB')))


def _bits(hexes):
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def _hamming(a, b):
    x = a[:, None] ^ b[None, :]
    try:
        return np.bitwise_count(x)
    except AttributeError:  # numpy < 2
        return np.unpackbits(x.view(np.uint8).reshape(len(a), len(b), 8), axis=2).sum(2)


def analyze(ph, meta, threshold=THRESHOLD):
    """ph: [[split, stem, hex], ...] (train/val/test). Output: the pilot leakage.json fields + test lists."""
    by = {sp: [(s, h) for p, s, h in ph if p == sp] for sp in ('train', 'val', 'test')}
    B = {sp: _bits([h for _, h in v]) for sp, v in by.items()}
    S = {sp: [s for s, _ in v] for sp, v in by.items()}
    def match(a, b):
        out = np.zeros((len(S[a]), len(S[b])), bool)
        for i in range(0, len(S[a]), 512):
            out[i:i + 512] = _hamming(B[a][i:i + 512], B[b]) <= threshold
        return out
    vt, tt, tv = match('val', 'train'), match('test', 'train'), match('test', 'val')
    # duplicate pairs: across all images (within-split included), same definition as the pilot's count
    all_bits = _bits([h for _, _, h in ph]); same = diff = 0
    for i in range(0, len(all_bits), 512):
        for r, j in zip(*np.nonzero(_hamming(all_bits[i:i + 512], all_bits) <= threshold)):
            if j > i + r:
                if meta[ph[i + r][1]]['cls'] == meta[ph[j][1]]['cls']:
                    same += 1
                else:
                    diff += 1
    val_leak = [s for s, m in zip(S['val'], vt.any(1)) if m]
    test_leak_tr = [s for s, m in zip(S['test'], tt.any(1)) if m]
    test_leak_val = [s for s, m in zip(S['test'], tv.any(1)) if m]
    dirty = set(test_leak_tr) | set(test_leak_val)
    return {'hamming_threshold': threshold,
            'val_imgs_with_dup_in_train': len(val_leak), 'test_imgs_with_dup_in_train': len(test_leak_tr),
            'test_imgs_with_dup_in_val': len(test_leak_val), 'n': {k: len(v) for k, v in S.items()},
            'pairs_same_class': same, 'pairs_diff_class': diff,
            'val_clean_n': len(S['val']) - len(val_leak), 'val_leak_stems': val_leak,
            'test_leak_train_stems': test_leak_tr, 'test_leak_val_stems': test_leak_val,
            'clean_test': [s for s in S['test'] if s not in dirty],
            'clean_test_n': len(S['test']) - len(dirty),
            'clean_test_definition': 'test images without a Hamming<=4 near-duplicate in train or dev',
            'clean_test_train_only': [s for s in S['test'] if s not in set(test_leak_tr)]}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phash', required=True); ap.add_argument('--meta', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    r = analyze(json.load(open(a.phash)), json.load(open(a.meta)))
    json.dump(r, open(a.out, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}, ensure_ascii=False))
