"""PlantSeg full experiment: data, model, evaluation. Same as the pilot's train script (pilot/train_pilot.py);
added for scale: ViT-L / ConvNeXt-L backbones, full UperHead (512 channels) + auxiliary FCN head (paper recipe),
foreground-only logit adjustment (H2), bf16/fp16 selection, path parameters."""
import json, math, os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import timm

NC = 116
IGN = 255
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TIMM_NAME = {
    'convnext_t': 'convnext_tiny.fb_in22k',
    'convnext_l': 'convnext_large.fb_in22k',
    'dinov3_s': 'vit_small_patch16_dinov3.lvd1689m',
    'dinov3_b': 'vit_base_patch16_dinov3.lvd1689m',
    'dinov3_l': 'vit_large_patch16_dinov3.lvd1689m',
    'dinov2_b': 'vit_base_patch14_dinov2.lvd142m',
    'dinov2_l': 'vit_large_patch14_dinov2.lvd142m',
}


def amp_dtype():
    """A100/L4: bf16. T4/P100 (sm < 8): fp16 + GradScaler. Can be forced with AMP_DTYPE=fp16|bf16 (testing)."""
    z = os.environ.get('AMP_DTYPE')
    if z:
        return torch.float16 if z == 'fp16' else torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


# ---------------- data (same pipeline as the pilot) ----------------
class TrainSet(torch.utils.data.Dataset):
    def __init__(self, root, stems, crop=512):
        self.root, self.stems, self.crop = root, stems, crop

    def __len__(self):
        return len(self.stems)

    def photometric(self, im):
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
        im = Image.open(f'{self.root}/train/img/{s}.jpg').convert('RGB')
        lab = Image.open(f'{self.root}/train/lab/{s}.png')
        r = random.uniform(0.5, 2.0)
        w, h = im.size
        nw, nh = max(1, int(w * r + .5)), max(1, int(h * r + .5))
        im = np.array(im.resize((nw, nh), Image.BILINEAR))
        lab = np.array(lab.resize((nw, nh), Image.NEAREST))
        c = self.crop
        for _ in range(10):  # RandomCrop cat_max_ratio=0.75
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


class InfiniteBatchSampler(torch.utils.data.Sampler):
    """Resumable order: an infinite stream of (seed, epoch) permutations, split into micro-batches.
    `start` = number of global micro-steps already consumed; under DDP each step's rank slice."""
    def __init__(self, n, micro, seed, start=0, rank=0, world=1):
        self.n, self.micro, self.seed, self.start, self.rank, self.world = n, micro, seed, start, rank, world

    def __iter__(self):
        g_step = self.micro * self.world
        pos = self.start * g_step
        ep, ofs = divmod(pos, self.n)
        while True:
            perm = np.random.default_rng([self.seed, ep]).permutation(self.n)
            while ofs + g_step <= self.n:
                block = perm[ofs:ofs + g_step]
                yield block[self.rank * self.micro:(self.rank + 1) * self.micro].tolist()
                ofs += g_step
            ep += 1; ofs = 0  # remainder at epoch end is dropped (drop_last)

    def __len__(self):
        return 10 ** 12


# ---------------- model ----------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, i, o, k=1):
        super().__init__(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UperLite(nn.Module):
    """PPM + FPN. ch=256: the pilot's UperNet-lite; ch=512: same structure as mmseg UPerHead (paper recipe)."""
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


class FCNAux(nn.Module):
    """mmseg FCNHead (num_convs=1, 256 channels), on top of stage 3, loss weight 0.4."""
    def __init__(self, cin, ch=256, nc=NC):
        super().__init__()
        self.conv = ConvBNReLU(cin, ch, 3); self.drop = nn.Dropout2d(0.1); self.cls = nn.Conv2d(ch, nc, 1)

    def forward(self, f):
        return self.cls(self.drop(self.conv(f)))


class ViTPyramid(nn.Module):
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
    def __init__(self, bb, mode, head_ch=256, aux=False, drop_path=0.1, pretrained=True):
        super().__init__()
        self.kind, self.mode = bb, mode
        self.aux = None
        if bb.startswith('convnext'):
            self.bb = timm.create_model(TIMM_NAME[bb], pretrained=pretrained, features_only=True, drop_path_rate=drop_path)
            chs = self.bb.feature_info.channels()
            self.pyr = None
            if aux:
                self.aux = FCNAux(chs[2])
        else:
            kw = dict(dynamic_img_size=True, dynamic_img_pad=bb.startswith('dinov2'))
            self.bb = timm.create_model(TIMM_NAME[bb], pretrained=pretrained, num_classes=0, **kw)
            d = self.bb.embed_dim
            n = len(self.bb.blocks)
            self.idx = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
            self.pyr = ViTPyramid(d, head_ch)
            chs = [head_ch] * 4
            if mode == 'lora':
                for p in self.bb.parameters():
                    p.requires_grad = False
                for blk in self.bb.blocks:
                    blk.attn.qkv = LoRA(blk.attn.qkv); blk.attn.proj = LoRA(blk.attn.proj)
        if mode == 'frozen':
            for p in self.bb.parameters():
                p.requires_grad = False
        self.head = UperLite(chs, head_ch)
        self.patch_mult = 14 if bb.startswith('dinov2') else 32

    def _bbfeats(self, x):
        if self.pyr is None:
            return self.bb(x)
        return self.bb.forward_intermediates(x, indices=self.idx, output_fmt='NCHW', intermediates_only=True)

    def forward(self, x):
        H, W = x.shape[2:]
        m = self.patch_mult
        ph, pw = (-H) % m, (-W) % m
        x = F.pad(x, (0, pw, 0, ph))
        if self.mode == 'frozen':
            self.bb.eval()
            with torch.no_grad():
                fs = [f.detach() for f in self._bbfeats(x)]
        else:
            fs = self._bbfeats(x)
        raw = fs
        if self.pyr is not None:
            fs = self.pyr(fs)
        lg = F.interpolate(self.head(fs), (H + ph, W + pw), mode='bilinear', align_corners=False)[:, :, :H, :W]
        if self.aux is not None and self.training:
            la = F.interpolate(self.aux(raw[2]), (H + ph, W + pw), mode='bilinear', align_corners=False)[:, :, :H, :W]
            return lg, la
        return lg


def trainable_state(model):
    """For checkpoints: trained parameters + all buffers (excluding the frozen backbone's weights)."""
    P = dict(model.named_parameters())
    return {k: v for k, v in model.state_dict().items() if k not in P or P[k].requires_grad or not k.startswith('bb.')}


def la_prior(freq, tau, mod='fg'):
    """Logit adjustment prior (added to the logits during training).
    mod='all' : pilot H2 (tau * log pi_c, background included; it hurt the background).
    mod='fg'  : among foreground classes only: adj_0 = 0, adj_c = tau * log(K * q_c), q = within-foreground frequency,
                K = 115. Zero for a uniform foreground distribution; leaves the background/foreground balance untouched."""
    f = torch.tensor(freq, dtype=torch.float64).clamp(min=1)
    if mod == 'all':
        adj = tau * torch.log(f / f.sum())
    else:
        q = f[1:] / f[1:].sum()
        adj = torch.cat([torch.zeros(1, dtype=torch.float64), tau * torch.log(q * (len(q)))])
    return adj.float().view(1, -1, 1, 1)


# ---------------- evaluation (same as the pilot) ----------------
@torch.no_grad()
def evaluate(model, stems, dev, root, split='val', out_npz=None, max_n=None, dtype=torch.bfloat16):
    model.eval()
    I = torch.zeros(NC, dtype=torch.float64, device=dev); U = torch.zeros_like(I); L = torch.zeros_like(I)
    per = []
    stems = stems[:max_n]
    for s in stems:
        im = Image.open(f'{root}/{split}/img/{s}.jpg').convert('RGB')
        gt = torch.from_numpy(np.array(Image.open(f'{root}/{split}/lab_orig/{s}.png'))).long().to(dev)
        x = (torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255 - MEAN) / STD
        x = x.unsqueeze(0).to(dev)
        with torch.autocast('cuda', dtype=dtype):
            lg = model(x).float()
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
    with np.errstate(invalid='ignore', divide='ignore'):
        iou = (I / U).cpu().numpy(); acc = (I / L).cpu().numpy()
    res = {'mIoU': float(np.nanmean(iou) * 100), 'mAcc': float(np.nanmean(acc) * 100),
           'aAcc': float((I.sum() / L.sum()).item() * 100), 'n_img': len(per)}
    per = np.stack(per)
    if out_npz:
        np.savez_compressed(out_npz, per=per, stems=np.array(stems), iou=iou, acc=acc)
    return res, iou, acc, per


def miou_subset(per, cls=None):
    """Subset mIoU over only the classes that have GT pixels in that subset (restricted to cls if given);
    the mmseg rule counts a class predicted but absent from the subset's GT as 0 and lowers the score."""
    I, U = per[:, 0].sum(0).astype(float), per[:, 1].sum(0).astype(float)
    S = per[:, 2].sum(0) > 0
    if cls is not None:
        S &= np.isin(np.arange(len(S)), list(cls))
    return float((I[S] / U[S]).mean() * 100) if S.any() else float('nan')


def subset_report(per, stems, root, clean_set=None):
    """Same axes as the pilot's analyze.py: clean/leaked, head/mid/tail, small/large lesion."""
    import collections
    meta = json.load(open(f'{root}/meta.json'))
    cnt = collections.Counter(m['cls'] for m in meta.values() if m['split'] == 'train')
    order = sorted(range(1, NC), key=lambda c: -cnt[c])
    t = len(order) // 3
    groups = {'head': set(order[:t]), 'mid': set(order[t:2 * t]), 'tail': set(order[2 * t:])}
    out = {f'mIoU_{g}': round(miou_subset(per, c), 2) for g, c in groups.items()}
    mr = np.array([meta[s]['mask_ratio'] for s in stems]); med = np.median(mr)
    out['mIoU_small_lesion'] = round(miou_subset(per[mr < med]), 2)
    out['mIoU_large_lesion'] = round(miou_subset(per[mr >= med]), 2)
    out = {kk: (None if isinstance(v, float) and np.isnan(v) else v) for kk, v in out.items()}
    if clean_set is not None:
        cl = np.array([s in clean_set for s in stems])
        out['mIoU_clean'] = round(miou_subset(per[cl]), 2) if cl.any() else None  # can be nan if the subset has no classes
        out['n_clean'] = int(cl.sum())
        out['mIoU_leaked'] = round(miou_subset(per[~cl]), 2) if (~cl).any() else None
    return out


def class_names(root):
    meta = json.load(open(f'{root}/meta.json'))
    name = ['background'] + [None] * (NC - 1)
    for m in meta.values():
        name[m['cls']] = f"{m['plant']}|{m['disease']}"
    return name
