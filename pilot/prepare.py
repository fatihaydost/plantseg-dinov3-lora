"""PlantSeg preparation: extract the zip, derive the split lists from Metadata.csv, write the counts, and pre-resize
the train/val images to the dataset paper's test scale (short side 512, long side <= 2048).
Test images are read only for pHash (their labels are never opened and no model sees them)."""
import csv, json, os, sys, zipfile, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.expanduser('~/.cache/plantseg')
RAW = f'{ROOT}/raw'
OUT = f'{ROOT}/prep'


def find(name):
    for dp, dn, fn in os.walk(RAW):
        if name in fn:
            return os.path.join(dp, name)
    return None


def rescale_size(w, h, long_max=2048, short_max=512):
    s = min(long_max / max(w, h), short_max / min(w, h))
    return max(1, int(w * s + 0.5)), max(1, int(h * s + 0.5))


def prep_one(args):
    img_p, ann_p, split, stem = args
    im = ImageOps.exif_transpose(Image.open(img_p)).convert('RGB')  # mmcv/cv2 apply the EXIF orientation
    w, h = im.size
    nw, nh = rescale_size(w, h)
    im.resize((nw, nh), Image.BILINEAR).save(f'{OUT}/{split}/img/{stem}.jpg', quality=95)
    lab = Image.open(ann_p)
    lab_np = np.array(lab)
    if lab_np.ndim == 3:
        lab_np = lab_np[..., 0]
    mism = lab_np.shape != (h, w)
    if mism:  # label size differs from the image: resize the image to the label
        im = im.resize((lab_np.shape[1], lab_np.shape[0]), Image.BILINEAR); w, h = im.size
    Image.fromarray(lab_np).resize((nw, nh), Image.NEAREST).save(f'{OUT}/{split}/lab/{stem}.png')
    # original label: evaluation runs at the original resolution
    Image.fromarray(lab_np).save(f'{OUT}/{split}/lab_orig/{stem}.png')
    vals, cnts = np.unique(lab_np, return_counts=True)
    return stem, split, (w, h, mism), {int(v): int(c) for v, c in zip(vals, cnts)}


def phash_one(p):
    import imagehash
    return str(imagehash.phash(Image.open(p).convert('RGB')))


if __name__ == '__main__':
    if not os.path.isdir(RAW):
        with zipfile.ZipFile(f'{ROOT}/plantseg.zip') as z:
            z.extractall(RAW)
    files = []
    for dp, dn, fn in os.walk(RAW):
        for f in fn:
            files.append(os.path.join(dp, f))
    print('total files', len(files))
    print(collections.Counter(os.path.relpath(os.path.dirname(f), RAW) for f in files).most_common(20))
    metas = [f for f in files if f.lower().endswith('.csv')]
    print('csv', metas)
    sys.stdout.flush()
    P = f'{RAW}/plantseg'
    rows = list(csv.DictReader(open(f'{P}/Metadata.csv', encoding='utf-8-sig')))
    smap = {'Training': 'train', 'Validation': 'val', 'Test': 'test'}
    split = collections.defaultdict(list); meta = {}
    for r in rows:
        sp = smap[r['Split']]; stem = r['Name'].rsplit('.', 1)[0]
        assert os.path.exists(f'{P}/images/{sp}/{r["Name"]}'), r['Name']
        split[sp].append(stem)
        meta[stem] = {'cls': int(r['Index']) + 1, 'plant': r['Plant'], 'disease': r['Disease'],
                      'mask_ratio': float(r['Mask ratio']), 'split': sp}
    for sp in ('train', 'val'):
        for d in ('img', 'lab', 'lab_orig'):
            os.makedirs(f'{OUT}/{sp}/{d}', exist_ok=True)
    jobs = [(f'{P}/images/{sp}/{s}.jpg', f'{P}/annotations/{sp}/{s}.png', sp, s) for sp in ('train', 'val') for s in split[sp]]
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(prep_one, jobs, chunksize=16))
    pix = np.zeros(116, np.int64); img_cnt = collections.Counter(); bad = 0; vals_all = collections.Counter()
    for stem, sp, wh, vc in res:
        for v, c in vc.items():
            vals_all[v] += 1
        fg = [v for v in vc if v not in (0, 255)]
        if fg != [meta[stem]['cls']]:
            bad += 1
        if sp == 'train':
            for v, c in vc.items():
                if v < 116:
                    pix[v] += c
            img_cnt[meta[stem]['cls']] += 1
    json.dump({k: v for k, v in split.items()}, open(f'{OUT}/split.json', 'w'))
    json.dump(meta, open(f'{OUT}/meta.json', 'w'))
    json.dump(pix.tolist(), open(f'{OUT}/train_pixfreq.json', 'w'))
    stats = {'n': {k: len(v) for k, v in split.items()}, 'n_class_train': len(img_cnt),
             'label_values_seen(train+val)': sorted(vals_all), 'images_label_not_matching_meta_class': bad,
             'train_imgs_per_class_min_med_max': [min(img_cnt.values()), int(np.median(list(img_cnt.values()))), max(img_cnt.values())],
             'train_bg_pixel_frac': float(pix[0] / pix.sum())}
    stats['label_size_mismatch_after_exif'] = sum(r[2][2] for r in res)
    print(json.dumps(stats, indent=1))
    # pHash: all three splits, image only
    allp = [(sp, s, f'{P}/images/{sp}/{s}.jpg') for sp in ('train', 'val', 'test') for s in split[sp]]
    with ProcessPoolExecutor(10) as ex:
        hs = list(ex.map(phash_one, [a[2] for a in allp], chunksize=32))
    json.dump([[a[0], a[1], h] for a, h in zip(allp, hs)], open(f'{OUT}/phash.json', 'w'))
    json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
