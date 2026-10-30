"""PlantSeg pilot (smoke test): backbone (ConvNeXt-T IN22k | DINOv3 ViT-B frozen/LoRA | DINOv2 ViT-B frozen) + the same UperNet-lite head.
Training uses the official train split, evaluation the official validation split (dev). The test split is never loaded."""
import argparse, json, math, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import timm

ROOT = os.path.expanduser('~/.cache/plantseg/prep')
NC = 116
IGN = 255
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------- data ----------------
class TrainSet(torch.utils.data.Dataset):
    def __init__(self, stems, crop=512, seed=0):
        self.stems, self.crop = stems, crop

    def __len__(self):
        return len(self.stems)

    def photometric(self, im):
        # close to mmseg PhotoMetricDistortion: brightness, contrast, saturation, hue (each with p=0.5)
        im = im.astype(np.float32)
        if random.random() < 0.5:
            im += random.uniform(-32, 32)
        if random.random() < 0.5:
            im *= random.uniform(0.5, 1.5)
        im = np.clip(im, 0, 255).astype(np.uint8)
        if random.random() < 0.5 or random.random() < 0.5:
            hsv = np.array(Image.fromarray(im).convert('HSV')).astype(np.int16)
            if random.random() < 0.5:
                hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.5, 1.5), 0, 255)
            if random.random() < 0.5:
                hsv[..., 0] = (hsv[..., 0] + random.randint(-18, 18) * 255 // 180) % 256
            im = np.array(Image.fromarray(hsv.astype(np.uint8), 'HSV').convert('RGB'))
        return im

    def __getitem__(self, i):
        s = self.stems[i]
        im = Image.open(f'{ROOT}/train/img/{s}.jpg').convert('RGB')
        lab = Image.open(f'{ROOT}/train/lab/{s}.png')
        r = random.uniform(0.5, 2.0)
        w, h = im.size
        nw, nh = max(1, int(w * r + .5)), max(1, int(h * r + .5))
        im = np.array(im.resize((nw, nh), Image.BILINEAR))
        lab = np.array(lab.resize((nw, nh), Image.NEAREST))
        c = self.crop
        # RandomCrop cat_max_ratio=0.75 (mmseg: 10 attempts)
        for _ in range(10):
            y0 = random.randint(0, max(0, nh - c)); x0 = random.randint(0, max(0, nw - c))
            lc = lab[y0:y0 + c, x0:x0 + c]
            u, cnt = np.unique(lc[lc != IGN], return_counts=True)
            if len(cnt) > 1 and cnt.max() / cnt.sum() < 0.75:
                break
        im = im[y0:y0 + c, x0:x0 + c]; lab = lab[y0:y0 + c, x0:x0 + c]
        if random.random() < 0.5:
            im = im[:, ::-1]; lab = lab[:, ::-1]
        im = self.photometric(np.ascontiguousarray(im))
        ph, pw = c - im.shape[0], c - im.shape[1]
        if ph > 0 or pw > 0:
            im = np.pad(im, ((0, ph), (0, pw), (0, 0)))
            lab = np.pad(lab, ((0, ph), (0, pw)), constant_values=IGN)
        x = (torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255 - MEAN) / STD
        return x, torch.from_numpy(np.ascontiguousarray(lab)).long()


# ---------------- model ----------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, i, o, k=1):
        super().__init__(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UperLite(nn.Module):
    """PPM + FPN, 256 channels; inputs at stride 4/8/16/32 (approximately)."""
    def __init__(self, in_chs, ch=256, nc=NC):
        super().__init__()
        self.lat = nn.ModuleList([ConvBNReLU(c, ch) for c in in_chs[:-1]])
        self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), ConvBNReLU(in_chs[-1], ch)) for s in (1, 2, 3, 6)])
        self.ppm_out = ConvBNReLU(in_chs[-1] + 4 * ch, ch, 3)
        self.fpn = nn.ModuleList([ConvBNReLU(ch, ch, 3) for _ in in_chs[:-1]])
        self.fuse = ConvBNReLU(4 * ch, ch, 3)
        self.drop = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(ch, nc, 1)

    def forward(self, feats):
        top = feats[-1]
        p = [top] + [F.interpolate(m(top), top.shape[2:], mode='bilinear', align_corners=False) for m in self.ppm]
        lat = [l(f) for l, f in zip(self.lat, feats[:-1])] + [self.ppm_out(torch.cat(p, 1))]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], lat[i - 1].shape[2:], mode='bilinear', align_corners=False)
        outs = [self.fpn[i](lat[i]) for i in range(len(lat) - 1)] + [lat[-1]]
        outs = [outs[0]] + [F.interpolate(o, outs[0].shape[2:], mode='bilinear', align_corners=False) for o in outs[1:]]
        return self.cls(self.drop(self.fuse(torch.cat(outs, 1))))


class ViTPyramid(nn.Module):
    """ViTDet-style simple pyramid: 4 intermediate layers -> 4x, 2x, 1x, 0.5x."""
    def __init__(self, d, out=256):
        super().__init__()
        self.up4 = nn.Sequential(nn.ConvTranspose2d(d, d // 2, 2, 2), nn.BatchNorm2d(d // 2), nn.GELU(), nn.ConvTranspose2d(d // 2, out, 2, 2))
        self.up2 = nn.ConvTranspose2d(d, out, 2, 2)
        self.id = nn.Conv2d(d, out, 1)
        self.down = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(d, out, 1))

    def forward(self, fs):
        return [self.up4(fs[0]), self.up2(fs[1]), self.id(fs[2]), self.down(fs[3])]


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=16):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, r, bias=False)
        self.b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5)); nn.init.zeros_(self.b.weight)
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.s


class Seg(nn.Module):
    def __init__(self, bb, mode):
        super().__init__()
        self.kind, self.mode = bb, mode
        if bb == 'convnext_t':
            self.bb = timm.create_model('convnext_tiny.fb_in22k', pretrained=True, features_only=True, drop_path_rate=0.1)
            chs = self.bb.feature_info.channels()
            self.pyr = None
        else:
            name = {'dinov3_b': 'vit_base_patch16_dinov3.lvd1689m', 'dinov3_s': 'vit_small_patch16_dinov3.lvd1689m',
                    'dinov2_b': 'vit_base_patch14_dinov2.lvd142m'}[bb]
            kw = dict(dynamic_img_size=True, dynamic_img_pad=('dinov2' in bb))
            self.bb = timm.create_model(name, pretrained=True, num_classes=0, **kw)
            d = self.bb.embed_dim
            n = len(self.bb.blocks)
            self.idx = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
            self.pyr = ViTPyramid(d)
            chs = [256] * 4
            if mode == 'lora':
                for p in self.bb.parameters():
                    p.requires_grad = False
                for blk in self.bb.blocks:
                    blk.attn.qkv = LoRA(blk.attn.qkv); blk.attn.proj = LoRA(blk.attn.proj)
        if mode == 'frozen':
            for p in self.bb.parameters():
                p.requires_grad = False
        self.head = UperLite(chs)

    def feats(self, x):
        if self.pyr is None:
            return self.bb(x)
        fs = self.bb.forward_intermediates(x, indices=self.idx, output_fmt='NCHW', intermediates_only=True)
        return self.pyr(fs)

    def forward(self, x):
        H, W = x.shape[2:]
        m = 14 if self.kind == 'dinov2_b' else 32
        ph, pw = (-H) % m, (-W) % m
        x = F.pad(x, (0, pw, 0, ph))
        lg = self._fwd(x)
        return F.interpolate(lg, (H + ph, W + pw), mode='bilinear', align_corners=False)[:, :, :H, :W]

    def _fwd(self, x):
        if self.mode == 'frozen':
            self.bb.eval()
            with torch.no_grad():
                if self.pyr is None:
                    fs = self.bb(x)
                else:
                    fs = self.bb.forward_intermediates(x, indices=self.idx, output_fmt='NCHW', intermediates_only=True)
            fs = [f.detach() for f in fs]
            if self.pyr is not None:
                fs = self.pyr(fs)
        else:
            fs = self.feats(x)
        return self.head(fs)


# ---------------- evaluation ----------------
@torch.no_grad()
def evaluate(model, stems, dev, out_npz=None, max_n=None):
    model.eval()
    I = torch.zeros(NC, dtype=torch.float64, device=dev); U = torch.zeros_like(I); L = torch.zeros_like(I)
    per = []
    for s in stems[:max_n]:
        im = Image.open(f'{ROOT}/val/img/{s}.jpg').convert('RGB')
        gt = torch.from_numpy(np.array(Image.open(f'{ROOT}/val/lab_orig/{s}.png'))).long().to(dev)
        x = (torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255 - MEAN) / STD
        x = x.unsqueeze(0).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(x).float()  # logits at input resolution
        # bilinear upsampling to the original resolution; argmax over channel chunks to save memory (equivalent to a full argmax)
        best = None
        for c0 in range(0, NC, 8):
            up = F.interpolate(lg[:, c0:c0 + 8], gt.shape, mode='bilinear', align_corners=False)[0]
            mv, mi = up.max(0)
            if best is None:
                best, pr = mv, mi + c0
            else:
                upd = mv > best
                best = torch.where(upd, mv, best); pr = torch.where(upd, mi + c0, pr)
            del up
        m = gt != IGN
        g, p = gt[m], pr[m]
        ai = torch.bincount(g[g == p], minlength=NC).double()
        ap = torch.bincount(p, minlength=NC).double(); al = torch.bincount(g, minlength=NC).double()
        I += ai; U += ap + al - ai; L += al
        per.append(torch.stack([ai, ap + al - ai, al]).cpu().numpy().astype(np.int64))
    iou = (I / U).cpu().numpy(); acc = (I / L).cpu().numpy()
    res = {'mIoU': float(np.nanmean(iou) * 100), 'mAcc': float(np.nanmean(acc) * 100),
           'aAcc': float((I.sum() / L.sum()).item() * 100), 'n_img': len(per)}
    if out_npz:
        np.savez_compressed(out_npz, per=np.stack(per), stems=np.array(stems[:max_n]), iou=iou, acc=acc)
    return res


# ---------------- main ----------------
def wif(w):
    s = torch.initial_seed() % 2**31
    random.seed(s); np.random.seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bb', required=True); ap.add_argument('--mode', required=True, choices=['full', 'frozen', 'lora'])
    ap.add_argument('--iters', type=int, default=8000); ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--accum', type=int, default=2); ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--bb_lr_mult', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--logit_adj', type=float, default=0.0)
    ap.add_argument('--tag', required=True); ap.add_argument('--eval_max', type=int, default=None)
    ap.add_argument('--bench', type=int, default=0)
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(__file__), 'results', a.tag)
    os.makedirs(out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = 'cuda'
    split = json.load(open(f'{ROOT}/split.json'))
    model = Seg(a.bb, a.mode).to(dev)
    bb_params = [p for n, p in model.named_parameters() if n.startswith('bb.') and p.requires_grad]
    lora_ids = {id(p) for n, p in model.named_parameters() if ('.a.' in n or '.b.' in n) and n.startswith('bb.') and p.requires_grad}
    other = [p for n, p in model.named_parameters() if not n.startswith('bb.') and p.requires_grad]
    groups = [{'params': other, 'lr': a.lr}]
    if a.mode == 'full':
        groups.append({'params': bb_params, 'lr': a.lr * a.bb_lr_mult})
    elif a.mode == 'lora':
        groups.append({'params': bb_params, 'lr': a.lr})
    for g in groups:
        g['base_lr'] = g['lr']
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    ntr = sum(p.numel() for g in groups for p in g['params'])
    prior = None
    if a.logit_adj > 0:
        freq = torch.tensor(json.load(open(f'{ROOT}/train_pixfreq.json')), dtype=torch.float32)
        prior = (a.logit_adj * torch.log(freq / freq.sum() + 1e-12)).view(1, NC, 1, 1).to(dev)
    ds = TrainSet(split['train'], a.crop)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=6, drop_last=True,
                                     persistent_workers=True, generator=g,
                                     worker_init_fn=wif)
    it, t0 = 0, time.time()
    log = open(f'{out}/train.log', 'a')
    print(json.dumps(vars(a)), f'trainable={ntr/1e6:.2f}M', file=log, flush=True)
    model.train(); torch.cuda.reset_peak_memory_stats()
    total = a.bench if a.bench else a.iters
    done = False
    while not done:
        for x, y in dl:
            step = it // a.accum
            nsteps = total // a.accum
            warm = 300 // a.accum
            f = (step + 1) / warm if step < warm else (1 - (step - warm) / max(1, nsteps - warm)) ** 1.0
            for gr in opt.param_groups:
                gr['lr'] = gr['base_lr'] * max(f, 0.0)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(x)
            lg = F.interpolate(lg.float(), y.shape[1:], mode='bilinear', align_corners=False)
            if prior is not None:
                lg = lg + prior
            loss = F.cross_entropy(lg, y, ignore_index=IGN) / a.accum
            loss.backward()
            if (it + 1) % a.accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
            it += 1
            if it % 50 == 0:
                print(f'it {it} loss {loss.item()*a.accum:.4f} lr {opt.param_groups[0]["lr"]:.2e} '
                      f'{(time.time()-t0)/it:.3f}s/it mem {torch.cuda.max_memory_allocated()/2**30:.2f}G', file=log, flush=True)
            if it >= total:
                done = True; break
    ttrain = time.time() - t0
    if a.bench:
        print(json.dumps({'bench_s_per_it': ttrain / it, 'mem_G': torch.cuda.max_memory_allocated() / 2**30}))
        t1 = time.time(); r = evaluate(model, split['val'], dev, max_n=50)
        print(json.dumps({'eval_s_per_img': (time.time() - t1) / 50}))
        return
    t1 = time.time()
    res = evaluate(model, split['val'], dev, out_npz=f'{out}/dev_per_image.npz', max_n=a.eval_max)
    res.update({'tag': a.tag, 'bb': a.bb, 'mode': a.mode, 'seed': a.seed, 'iters': a.iters, 'logit_adj': a.logit_adj,
                'train_min': ttrain / 60, 'eval_min': (time.time() - t1) / 60, 'trainable_M': ntr / 1e6,
                'peak_mem_G': torch.cuda.max_memory_allocated() / 2**30, 'split': 'dev (official validation)'})
    json.dump(res, open(f'{out}/result.json', 'w'), indent=1)
    print(json.dumps(res), file=log, flush=True)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
