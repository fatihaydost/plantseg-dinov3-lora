"""Single run: train with the config from queue.json, write checkpoint/log/status to Drive, evaluate on dev at the end.
If interrupted, resumes from the last checkpoint (the runner calls it again). The test set is never loaded here.
Usage: python train.py --root <drive folder> --data <prep folder> --run <id> [--bench N]
DDP (Kaggle T4x2): torchrun --nproc_per_node 2 train.py ...
Environment: CKPT_MIN (default 30), LOG_S (60), SESSION_END (unix time) or TIME_LIMIT_H (unlimited if unset)."""
import argparse, contextlib, glob, json, os, random, shutil, sys, time
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import NC, IGN
from run_queue import Queue, family, now


def recipe(k):
    """Training settings from a queued run. 'ours' = the pilot protocol (docs/preregistration.md), scaled up.
    'paper' = the mmseg ConvNeXt-UperNet recipe (the PlantSeg repo has no ConvNeXt-L config; the mmseg ADE recipe
    adapted to 40k x 16): AdamW 1e-4, wd 0.05, layer decay 0.9 (stage_wise), 1500 warmup, poly 1.0,
    drop_path 0.4, UperHead 512 + FCN auxiliary (0.4)."""
    if k.get('recipe', 'ours') == 'paper':
        r = dict(lr=1e-4, wd=0.05, warmup=1500, warm_start=1e-6, power=1.0, head_ch=512, aux=True, drop_path=0.4,
                 bb_lr_mult=1.0, layer_decay=0.9, no_wd_1d=True)
    else:
        r = dict(lr=2e-4, wd=0.01, warmup=300, warm_start=0.0, power=1.0, head_ch=256, aux=False, drop_path=0.1,
                 bb_lr_mult=0.1, layer_decay=None, no_wd_1d=False)
    for key in ('lr', 'wd', 'warmup', 'head_ch', 'drop_path'):
        if k.get(key) is not None:
            r[key] = k[key]
    return r


def convnext_layer(n):
    # mmseg stage_wise: stem/downsample 0, stage i -> i+1, head 13; scale 0.9 ** (14 - id - 1)
    if 'stem' in n or 'downsample' in n:
        return 0
    for i in range(4):
        if f'stages_{i}' in n or f'stages.{i}' in n:
            return i + 1
    return 13


def param_groups(model, k, r):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        bb = n.startswith('bb.')
        lr = r['lr']
        if bb and k['mode'] == 'full':
            lr *= r['bb_lr_mult']
            if r['layer_decay']:
                lr *= r['layer_decay'] ** (14 - convnext_layer(n) - 1)
        wd = 0.0 if (r['no_wd_1d'] and (p.ndim == 1 or n.endswith('.bias'))) else r['wd']
        groups.setdefault((lr, wd), []).append(p)
    return [{'params': ps, 'lr': lr, 'base_lr': lr, 'weight_decay': wd} for (lr, wd), ps in groups.items()]


def lr_factor(step, total, r):
    w = r['warmup']
    if step < w:
        return r['warm_start'] + (1 - r['warm_start']) * (step + 1) / w
    return max(0.0, 1 - (step - w) / max(1, total - w)) ** r['power']


def ddp_setup():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0')); lrank = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        import datetime
        # DDP_BACKEND=gloo is only for local testing on a single GPU
        torch.distributed.init_process_group(os.environ.get('DDP_BACKEND', 'nccl'), timeout=datetime.timedelta(minutes=90))
    lrank = lrank % max(1, torch.cuda.device_count())
    torch.cuda.set_device(lrank)
    return world, rank, f'cuda:{lrank}'


def broadcast(flag, world, dev):
    if world == 1:
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=dev)
    torch.distributed.broadcast(t, 0)
    return bool(t.item())


def find_micro(model, micro, crop, dev, dtype, aux, world):
    """Largest micro-batch that fits (halve on OOM). Also halves if peak memory exceeds 92%."""
    tot = torch.cuda.get_device_properties(dev).total_memory
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}  # keep BN statistics unaffected by the probe
    try:
        return _find_micro(model, micro, crop, dev, dtype, world, tot)
    finally:
        with torch.no_grad():
            for n, b in model.named_buffers():
                b.copy_(buffers[n])


def _find_micro(model, micro, crop, dev, dtype, world, tot):
    lo = 1 if world > 1 else 2  # BN in the PPM's 1x1 pool fails with a single sample (DDP has SyncBN)
    while micro >= lo:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
            x = torch.randn(micro, 3, crop, crop, device=dev); y = torch.randint(0, NC, (micro, crop, crop), device=dev)
            model.train()
            with torch.autocast('cuda', dtype=dtype):
                out = model(x)
            lg = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(lg.float(), y)
            if isinstance(out, tuple):
                loss = loss + 0.4 * F.cross_entropy(out[1].float(), y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(dev)
            del x, y, out, lg, loss
            if peak < 0.92 * tot or micro == lo:
                ok = micro
            else:
                ok = None
        except torch.cuda.OutOfMemoryError:
            ok = None
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        if world > 1:
            t = torch.tensor([ok or 0], device=dev); torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
            ok = int(t.item()) or None
        if ok:
            return ok
        micro //= 2
    raise RuntimeError(f'not even micro-batch {lo} fits')


def save_ckpt(Q, k, model, opt, scaler, it, micro, accum):
    local = f"/tmp/ckpt_{k['id']}.pt"
    torch.save({'model': common.trainable_state(model), 'opt': opt.state_dict(),
                'scaler': scaler.state_dict() if scaler else None, 'iter': it, 'micro': micro, 'accum': accum}, local)
    target = os.path.join(Q.root, 'ckpt', f"{k['id']}_it{it:06d}.pt")
    shutil.copy(local, target + '.tmp'); os.replace(target + '.tmp', target); os.remove(local)
    Q.update(k['id'], ckpt=os.path.relpath(target, Q.root), iter=it)
    for old in glob.glob(os.path.join(Q.root, 'ckpt', f"{k['id']}_it*.pt")):  # Drive quota: keep only the latest
        if old != target:
            os.remove(old)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--data', required=True); ap.add_argument('--run', required=True)
    ap.add_argument('--bench', type=int, default=0); ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    world, rank, dev = ddp_setup()
    main_rank = rank == 0
    Q = Queue(a.root)
    k = Q.run(a.run)
    r = recipe(k)
    dtype = common.amp_dtype()
    gpu = torch.cuda.get_device_name(dev)
    ckpt_min = float(os.environ.get('CKPT_MIN', '30')); log_s = float(os.environ.get('LOG_S', '60'))
    # session end (unix time); the runner sets it from TIME_LIMIT_H. For Kaggle's 12 h limit.
    end_time = float(os.environ.get('SESSION_END', '0')) or None
    if end_time is None and os.environ.get('TIME_LIMIT_H'):
        end_time = time.time() + float(os.environ['TIME_LIMIT_H']) * 3600
    t_start = time.time()
    seed = k['seed']
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    split = json.load(open(f'{a.data}/split.json'))
    tr_stems = split['train'][:k['n_train']] if k.get('n_train') else split['train']
    dev_stems = split['val'][:k['n_dev']] if k.get('n_dev') else split['val']

    model = common.Seg(k['bb'], k['mode'], head_ch=r['head_ch'], aux=r['aux'], drop_path=r['drop_path']).to(dev)
    if world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    groups = param_groups(model, k, r)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda') if dtype == torch.float16 else None
    ntr = sum(p.numel() for g in groups for p in g['params'])
    total = k['total_iter']
    bs = k['bs']
    prior = None
    if k.get('la_tau'):
        freq = json.load(open(f'{a.data}/train_pixfreq.json'))
        prior = common.la_prior(freq, k['la_tau'], k.get('la_mod', 'fg')).to(dev)

    # micro-batch: use the one stored in the queue if present (keeps the order consistent on resume)
    micro, accum = k.get('micro'), k.get('accum')
    if not micro or a.bench:
        micro = find_micro(model, bs // world, k.get('crop', 512), dev, dtype, r['aux'], world)
        accum = (bs // world) // micro
        micro = (bs // world) // accum
        if main_rank and not a.bench:
            Q.update(k['id'], micro=micro, accum=accum)
    it = 0
    if k.get('ckpt') and not a.bench:
        path = os.path.join(a.root, k['ckpt'])
        if os.path.exists(path):
            ck = torch.load(path, map_location=dev, weights_only=False)
            missing, unexpected = model.load_state_dict(ck['model'], strict=False)
            P = dict(model.named_parameters())
            assert not unexpected and all(not P[m].requires_grad for m in missing if m in P), (missing[:5], unexpected[:5])
            opt.load_state_dict(ck['opt'])
            if scaler and ck.get('scaler'):
                scaler.load_state_dict(ck['scaler'])
            it = ck['iter']; micro, accum = ck['micro'], ck['accum']
            if main_rank:
                print(f'resuming: {path} iter {it}', flush=True)
        elif main_rank:
            print(f'WARNING: recorded checkpoint missing ({path}); starting from scratch', flush=True)
    dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev]) if world > 1 else model

    ds = common.TrainSet(a.data, tr_stems, k.get('crop', 512))
    samp = common.InfiniteBatchSampler(len(ds), micro, seed, start=it * accum, rank=rank, world=world)
    if a.workers is None and os.environ.get('WORKERS'):
        a.workers = int(os.environ['WORKERS'])
    nw = a.workers if a.workers is not None else max(2, min(12, (os.cpu_count() or 4) // world - 1))
    base_seed = seed * 100003 + it

    def wif(w):
        s = (base_seed * 1000 + w * 7 + rank * 131) % 2**31
        random.seed(s); np.random.seed(s)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=samp, num_workers=nw, pin_memory=True, worker_init_fn=wif,
                                     persistent_workers=True, prefetch_factor=4)
    dit = iter(dl)

    logp = os.path.join(a.root, 'log', f"{k['id']}.log")
    def log(s):
        if main_rank:
            with open(logp, 'a') as f:
                f.write(s + '\n')
            print(s, flush=True)
    if main_rank and not a.bench:
        Q.update(k['id'], status='running', gpu=gpu, micro=micro, accum=accum,
                 started=k.get('started') or now().strftime('%F %H:%M'))
        d = Q.read(); d['gpu'] = gpu; Q.write(d)
        log(f"== {now().strftime('%F %H:%M:%S')} start/resume {k['id']} iter {it}/{total} | gpu {gpu} world {world} | "
            f"micro {micro} x accum {accum} x world {world} = batch {micro*accum*world} | {dtype} | trained {ntr/1e6:.2f}M | "
            f"recipe {json.dumps(r)} | torch {torch.__version__} timm {common.timm.__version__} | workers {nw}")
    train_s0 = k.get('train_s', 0)
    t_seg = time.time(); t_log = t_seg; t_ckpt = t_seg; it_log = it
    losses = []
    meas = {}
    nan_count = 0
    bench_target = it + a.bench if a.bench else None
    dmodel.train()
    while it < total and (bench_target is None or it < bench_target):
        f = lr_factor(it, total, r)
        for g in opt.param_groups:
            g['lr'] = g['base_lr'] * f
        tl = 0.0
        for m in range(accum):
            x, y = next(dit)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            ctx = dmodel.no_sync() if (world > 1 and m < accum - 1) else contextlib.nullcontext()
            with ctx:
                with torch.autocast('cuda', dtype=dtype):
                    out = dmodel(x)
                lg = out[0] if isinstance(out, tuple) else out
                lg = lg.float()
                if prior is not None:
                    lg = lg + prior
                loss = F.cross_entropy(lg, y, ignore_index=IGN)
                if isinstance(out, tuple):
                    loss = loss + 0.4 * F.cross_entropy(out[1].float(), y, ignore_index=IGN)
                loss = loss / accum
                (scaler.scale(loss) if scaler else loss).backward()
            tl += loss.item()
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        it += 1
        losses.append(tl)
        if not np.isfinite(tl):
            nan_count += 1
            if nan_count > 50:
                raise RuntimeError(f'loss not finite for 50 steps (iter {it}); dtype {dtype}')
        else:
            nan_count = 0
        t_now = time.time()
        # speed measurement: first 20 iterations are warmup, measured from 20 -> 200
        if 'start' not in meas and len(losses) == 20:
            meas['start'] = (t_now, it)
        target_n = (a.bench if a.bench else 200)
        last_step = it >= total or (bench_target is not None and it >= bench_target)
        if main_rank and 'start' in meas and 'done' not in meas and (len(losses) >= target_n or (last_step and it > meas['start'][1])):
            t0, i0 = meas['start']
            spi = (t_now - t0) / max(1, it - i0)
            meas['done'] = spi
            Q.write_speed(family(k), spi, micro, accum, gpu)
            if not a.bench:
                Q.update(k['id'], s_per_it=round(spi, 4))
                Q.write_status({'id': k['id'], 'iter': it, 'total': total, 'it_s': 1 / spi,
                                'remaining_s': spi * (total - it), 'loss': float(np.mean(losses[-50:]))})
            log(f"SPEED MEASUREMENT {k['id']}: {spi:.3f} s/iter ({1/spi:.2f} it/s, {bs/spi:.1f} samples/s) -> "
                f"run estimate {spi*total/3600:.2f} h")
        if a.bench:
            continue
        if len(losses) == 1:  # keep the first step (data loader warmup) out of the speed/ETA calculation
            t_log, it_log = t_now, it
        elif main_rank and (t_now - t_log >= log_s or it == total):
            ips = (it - it_log) / (t_now - t_log)
            loss_avg = float(np.mean(losses[-max(1, it - it_log):]))
            remaining = (total - it) / max(ips, 1e-6)
            mem = torch.cuda.max_memory_allocated(dev) / 2**30
            log(f"{now().strftime('%F %H:%M:%S')} it {it}/{total} ({100*it/total:.1f}%) loss {loss_avg:.4f} "
                f"lr {opt.param_groups[0]['lr']:.2e} {ips:.2f} it/s {ips*bs:.1f} samples/s mem {mem:.1f}G "
                f"remaining {remaining/3600:.2f} h")
            Q.update(k['id'], iter=it, train_s=train_s0 + (t_now - t_seg))
            Q.write_status({'id': k['id'], 'iter': it, 'total': total, 'it_s': ips, 'remaining_s': remaining, 'loss': loss_avg})
            t_log, it_log = t_now, it
        time_up = end_time is not None and t_now >= end_time
        do_save = broadcast(main_rank and (t_now - t_ckpt >= ckpt_min * 60 or time_up), world, dev)
        if do_save and it < total:
            if main_rank:
                h = save_ckpt(Q, k, model, opt, scaler, it, micro, accum)
                Q.update(k['id'], train_s=train_s0 + (time.time() - t_seg))
                log(f"checkpoint: {os.path.basename(h)}")
            t_ckpt = time.time()
            if broadcast(main_rank and time_up, world, dev):
                if main_rank:
                    Q.update(k['id'], status='paused')
                    Q.write_status()
                    log('TIME LIMIT: checkpoint saved, paused (resumes when run again)')
                if world > 1:
                    torch.distributed.barrier(); torch.distributed.destroy_process_group()
                sys.exit(3)
    if a.bench:
        if main_rank:
            d = Q.read(); d['measure_s'] = d.get('measure_s', 0) + (time.time() - t_start); d['gpu'] = gpu; Q.write(d)
        if world > 1:
            torch.distributed.destroy_process_group()
        return

    train_s = train_s0 + (time.time() - t_seg)
    if main_rank:
        if not os.path.exists(os.path.join(a.root, 'ckpt', f"{k['id']}_it{total:06d}.pt")):
            save_ckpt(Q, k, model, opt, scaler, it, micro, accum)  # resume from here if evaluation is interrupted
        Q.update(k['id'], train_s=train_s, iter=it)
        log(f"training done ({train_s/3600:.2f} h); dev evaluation ({len(dev_stems)} images)")
        t1 = time.time()
        npz = os.path.join(a.root, 'results', f"{k['id']}_dev_per_image.npz")
        res, iou, acc, per = common.evaluate(model, dev_stems, dev, a.data, 'val', out_npz=npz, dtype=dtype)
        lkp = f'{a.data}/leakage.json'
        clean = None
        if os.path.exists(lkp):
            leak = set(json.load(open(lkp))['val_leak_stems'])
            clean = {s for s in dev_stems if s not in leak}
        sub = common.subset_report(per, dev_stems, a.data, clean)
        names = common.class_names(a.data)
        eval_s = time.time() - t1
        res.update({'id': k['id'], 'label': k['label'], 'split': 'dev (official validation)', **sub,
                    'class_IoU': {names[c]: (None if np.isnan(iou[c]) else round(float(iou[c]) * 100, 3)) for c in range(NC)},
                    'class_Acc': {names[c]: (None if np.isnan(acc[c]) else round(float(acc[c]) * 100, 3)) for c in range(NC)},
                    'config': {kk: k.get(kk) for kk in ('bb', 'mode', 'recipe', 'seed', 'bs', 'total_iter', 'la_tau', 'la_mod', 'crop')},
                    'recipe': r, 'micro': micro, 'accum': accum, 'world': world, 'gpu': gpu, 'dtype': str(dtype),
                    'train_h': round(train_s / 3600, 3), 'eval_min': round(eval_s / 60, 2),
                    's_per_it': meas.get('done') or k.get('s_per_it'), 'trained_params_M': round(ntr / 1e6, 2),
                    'torch': torch.__version__, 'timm': common.timm.__version__, 'finished': now().strftime('%F %H:%M')})
        json.dump(res, open(os.path.join(a.root, 'results', f"{k['id']}.json"), 'w'), indent=1, ensure_ascii=False)
        mp = os.path.join(a.root, 'model', f"{k['id']}.pt")
        tmp = f"/tmp/_final_{k['id']}.pt"  # per run: on Kaggle two runners finish in parallel on the same machine
        torch.save({'model': common.trainable_state(model), 'run': k, 'recipe': r}, tmp)
        shutil.copy(tmp, mp + '.tmp'); os.replace(mp + '.tmp', mp); os.remove(tmp)
        for old in glob.glob(os.path.join(a.root, 'ckpt', f"{k['id']}_it*.pt")):
            os.remove(old)
        Q.update(k['id'], status='done', ckpt=None, dev_miou=round(res['mIoU'], 2), eval_s=eval_s,
                 finished=now().strftime('%F %H:%M'), error=None)
        log(f"RESULT {k['id']}: dev mIoU {res['mIoU']:.2f} mAcc {res['mAcc']:.2f} clean {sub.get('mIoU_clean')} "
            f"tail {sub['mIoU_tail']} ({eval_s/60:.1f} min)")
        Q.write_status()
    if world > 1:
        torch.distributed.barrier(); torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
