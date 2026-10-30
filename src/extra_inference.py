"""Extra inference (test, with the models already opened once on test; no new selection): class-agnostic mask IoU and
example prediction maps. Per model, per-image foreground (class > 0) intersection/union -> results/extra/<id>_test_fg.npz;
the predicted class map of the images in --visual -> results/extra/predictions/<id>/<stem>.png (uint8, original label resolution).
Usage: python extra_inference.py --root <drive> --target /content/plantseg --runs a,b --visual s1,s2 --yes_test"""
import argparse, json, os, subprocess, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


@torch.no_grad()
def predict(model, root, s, dev, dtype):
    """Same argmax as common.evaluate (channel chunks of 8, original resolution)."""
    im = Image.open(f'{root}/test/img/{s}.jpg').convert('RGB')
    gt = torch.from_numpy(np.array(Image.open(f'{root}/test/lab_orig/{s}.png'))).long().to(dev)
    x = ((torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255 - common.MEAN) / common.STD).unsqueeze(0).to(dev)
    with torch.autocast('cuda', dtype=dtype):
        lg = model(x).float()
    best = None
    for c0 in range(0, common.NC, 8):
        up = F.interpolate(lg[:, c0:c0 + 8], gt.shape, mode='bilinear', align_corners=False)[0]
        mv, mi = up.max(0)
        if best is None:
            best, pr = mv, mi + c0
        else:
            upd = mv > best
            best = torch.where(upd, mv, best); pr = torch.where(upd, mi + c0, pr)
    return pr, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--target', required=True)
    ap.add_argument('--runs', required=True); ap.add_argument('--visual', default='')
    ap.add_argument('--yes_test', action='store_true')
    a = ap.parse_args()
    if not a.yes_test:
        raise SystemExit('Test is closed. Pass --yes_test only by explicit decision.')
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.py'),
                    '--target', a.target, '--test'], check=True)
    data = f'{a.target}/prep'
    stems = json.load(open(f'{data}/split.json'))['test']
    visual = {s for s in a.visual.split(',') if s}
    assert visual <= set(stems), visual - set(stems)
    out = os.path.join(a.root, 'results', 'extra'); os.makedirs(out, exist_ok=True)
    dev, dtype = 'cuda', common.amp_dtype()
    for rid in [x for x in a.runs.split(',') if x]:
        target = os.path.join(out, f'{rid}_test_fg.npz')
        if os.path.exists(target):
            print('exists, skipping:', rid); continue  # resume where it left off if the session drops
        ck = torch.load(os.path.join(a.root, 'model', f'{rid}.pt'), map_location='cpu', weights_only=False)
        k, r = ck['run'], ck['recipe']
        model = common.Seg(k['bb'], k['mode'], head_ch=r['head_ch'], aux=r['aux'], drop_path=r['drop_path'])
        missing, unexpected = model.load_state_dict(ck['model'], strict=False)
        P = dict(model.named_parameters())
        assert not unexpected and all(not P[m].requires_grad for m in missing if m in P)
        model = model.to(dev).eval()
        fg = np.zeros((len(stems), 3), np.int64)  # intersection, union, target foreground pixels
        for i, s in enumerate(stems):
            pr, gt = predict(model, data, s, dev, dtype)
            m = gt != common.IGN
            p, g = (pr > 0) & m, (gt > 0) & m
            fg[i] = [(p & g).sum().item(), (p | g).sum().item(), g.sum().item()]
            if s in visual:
                d = os.path.join(out, 'predictions', rid); os.makedirs(d, exist_ok=True)
                Image.fromarray(pr.to(torch.uint8).cpu().numpy()).save(f'{d}/{s}.png')
        np.savez_compressed(target, fg=fg, stems=np.array(stems))
        print(f'{rid}: class-agnostic IoU {100 * fg[:, 0].sum() / fg[:, 1].sum():.2f}', flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
