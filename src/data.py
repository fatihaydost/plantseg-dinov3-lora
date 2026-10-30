"""Download PlantSeg v7 from Zenodo (parallel Range requests, resumable), verify size + md5, extract, apply the same
preprocessing as the pilot's prepare.py (EXIF orientation, short side 512 / long <= 2048), pHash + leakage analysis.
The test split is not prepared by default (its images are read only for pHash); --test is only for evaluate_test.py.
Usage: python data.py --target /content/plantseg [--zip existing.zip] [--subset 40] [--test]"""
import argparse, collections, csv, hashlib, json, os, sys, threading, time, urllib.request, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leakage

URL = 'https://zenodo.org/api/records/17719108/files/plantseg.zip/content'
SIZE = 1057281724
MD5 = '9358a66dff88cdd15c4fe009763c40a3'  # Zenodo API checksum, also verified in the pilot


def download(z, n=8):
    parts = z + '.parts'
    os.makedirs(parts, exist_ok=True)
    R = (SIZE + n - 1) // n
    rngs = [(i * R, min(SIZE - 1, (i + 1) * R - 1)) for i in range(n)]

    def work(i, a, b):
        p = f'{parts}/p{i}'
        for _ in range(200):
            have = os.path.getsize(p) if os.path.exists(p) else 0
            st = a + have
            if st > b:
                return
            try:
                req = urllib.request.Request(URL, headers={'Range': f'bytes={st}-{b}'})
                with urllib.request.urlopen(req, timeout=60) as r, open(p, 'ab') as f:
                    while True:
                        c = r.read(1 << 20)
                        if not c:
                            break
                        f.write(c)
            except Exception as e:
                print(i, 'error', e, flush=True); time.sleep(5)
    th = [threading.Thread(target=work, args=(i, a, b)) for i, (a, b) in enumerate(rngs)]
    [t.start() for t in th]
    while any(t.is_alive() for t in th):
        time.sleep(15)
        got = sum(os.path.getsize(f'{parts}/p{i}') for i in range(n) if os.path.exists(f'{parts}/p{i}'))
        print(f'downloading {got/2**20:.0f}/{SIZE/2**20:.0f} MB', flush=True)
    with open(z, 'wb') as f:
        for i in range(n):
            with open(f'{parts}/p{i}', 'rb') as g:
                while True:
                    c = g.read(1 << 24)
                    if not c:
                        break
                    f.write(c)
    for i in range(n):
        os.remove(f'{parts}/p{i}')


def verify(z):
    if os.path.getsize(z) != SIZE:
        raise SystemExit(f'WRONG SIZE: {os.path.getsize(z)} != {SIZE}')
    h = hashlib.md5()
    with open(z, 'rb') as f:
        while True:
            c = f.read(1 << 24)
            if not c:
                break
            h.update(c)
    if h.hexdigest() != MD5:
        raise SystemExit(f'WRONG MD5: {h.hexdigest()}')
    print('zip verified (size + md5)', flush=True)


def rescale_size(w, h, long_max=2048, short_max=512):
    s = min(long_max / max(w, h), short_max / min(w, h))
    return max(1, int(w * s + 0.5)), max(1, int(h * s + 0.5))


def prep_one(args):
    """Same as prep_one in the pilot's prepare.py."""
    img_p, ann_p, out, split, stem = args
    im = ImageOps.exif_transpose(Image.open(img_p)).convert('RGB')
    w, h = im.size
    nw, nh = rescale_size(w, h)
    im.resize((nw, nh), Image.BILINEAR).save(f'{out}/{split}/img/{stem}.jpg', quality=95)
    lab_np = np.array(Image.open(ann_p))
    if lab_np.ndim == 3:
        lab_np = lab_np[..., 0]
    mism = lab_np.shape != (h, w)
    if split == 'train':
        Image.fromarray(lab_np).resize((nw, nh), Image.NEAREST).save(f'{out}/{split}/lab/{stem}.png')
    else:
        Image.fromarray(lab_np).save(f'{out}/{split}/lab_orig/{stem}.png')
    vals, cnts = np.unique(lab_np, return_counts=True)
    return stem, split, mism, {int(v): int(c) for v, c in zip(vals, cnts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True); ap.add_argument('--zip', default=None)
    ap.add_argument('--subset', type=int, default=0, help='local testing: train N, val N/2, test N/2')
    ap.add_argument('--test', action='store_true', help='also prepare the test images (evaluate_test.py only)')
    ap.add_argument('--proc', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    os.makedirs(a.target, exist_ok=True)
    OUT = f'{a.target}/prep'; RAW = f'{a.target}/raw'
    marker = f'{OUT}/READY' + ('_test' if a.test else '')
    if os.path.exists(marker):
        print('data already prepared:', OUT); return
    z = a.zip or f'{a.target}/plantseg.zip'
    if not os.path.isdir(f'{RAW}/plantseg'):
        if not os.path.exists(z):
            t = time.time(); download(z); print(f'download {time.time()-t:.0f} s', flush=True)
        verify(z)
        t = time.time()
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)
        print(f'extracted {time.time()-t:.0f} s', flush=True)
        if not a.zip:
            os.remove(z)  # free space on /content
    P = f'{RAW}/plantseg'
    rows = list(csv.DictReader(open(f'{P}/Metadata.csv', encoding='utf-8-sig')))
    smap = {'Training': 'train', 'Validation': 'val', 'Test': 'test'}
    split = collections.defaultdict(list); meta = {}
    for r in rows:
        sp = smap[r['Split']]; stem = r['Name'].rsplit('.', 1)[0]
        split[sp].append(stem)
        meta[stem] = {'cls': int(r['Index']) + 1, 'plant': r['Plant'], 'disease': r['Disease'],
                      'mask_ratio': float(r['Mask ratio']), 'split': sp}
    if a.subset:
        n = a.subset
        split = {'train': split['train'][:n], 'val': split['val'][:max(2, n // 2)], 'test': split['test'][:max(2, n // 2)]}
    splits = ('train', 'val', 'test') if a.test else ('train', 'val')
    if a.test and os.path.exists(f'{OUT}/READY'):
        splits = ('test',)  # train/val already prepared
    for sp in splits:
        for d in (('img', 'lab') if sp == 'train' else ('img', 'lab_orig')):
            os.makedirs(f'{OUT}/{sp}/{d}', exist_ok=True)
    jobs = [(f'{P}/images/{sp}/{s}.jpg', f'{P}/annotations/{sp}/{s}.png', OUT, sp, s) for sp in splits for s in split[sp]]
    t = time.time()
    with ProcessPoolExecutor(a.proc) as ex:
        res = list(ex.map(prep_one, jobs, chunksize=16))
    print(f'preprocessed {len(jobs)} images {time.time()-t:.0f} s', flush=True)
    mismatched = [f'{sp}/{stem}' for stem, sp, mism, _ in res if mism]
    assert not mismatched, f'image-label size mismatch ({len(mismatched)}): {mismatched[:10]}'
    pix = np.zeros(116, np.int64)
    for stem, sp, _, vc in res:
        if sp == 'train':
            for v, c in vc.items():
                if v < 116:
                    pix[v] += c
    if not os.path.exists(f'{OUT}/split.json'):
        json.dump({k: v for k, v in split.items()}, open(f'{OUT}/split.json', 'w'))
        json.dump(meta, open(f'{OUT}/meta.json', 'w'))
        json.dump(pix.tolist(), open(f'{OUT}/train_pixfreq.json', 'w'))
        print('train background pixel ratio', round(float(pix[0] / pix.sum()), 4), '| n', {k: len(v) for k, v in split.items()})
    if not os.path.exists(f'{OUT}/leakage.json'):
        t = time.time()
        allp = [(sp, s, f'{P}/images/{sp}/{s}.jpg') for sp in ('train', 'val', 'test') for s in split[sp]]
        with ProcessPoolExecutor(a.proc) as ex:
            hs = list(ex.map(leakage.phash_one, [x[2] for x in allp], chunksize=32))
        ph = [[x[0], x[1], h] for x, h in zip(allp, hs)]
        json.dump(ph, open(f'{OUT}/phash.json', 'w'))
        r = leakage.analyze(ph, meta)
        json.dump(r, open(f'{OUT}/leakage.json', 'w'), indent=1, ensure_ascii=False)
        print(f'pHash {time.time()-t:.0f} s:', {k: v for k, v in r.items() if not isinstance(v, list)}, flush=True)
    open(marker, 'w').write(time.strftime('%F %T'))
    print('DATA READY', OUT, flush=True)


if __name__ == '__main__':
    main()
