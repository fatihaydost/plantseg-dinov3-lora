"""Run queue (queue.json) and human-readable status.md. Single-writer assumption: only one training process (or the
runner that launched it) writes at a time; writes are atomic via tmp + os.replace (also works on Drive FUSE)."""
import datetime, json, os, shutil, time

BUDGET_H = 61.0          # Colab Pro: 326 units / 5.3 units/h
UNITS_PER_H = 5.3
EVAL_H = 0.15            # dev evaluation + saving, per run (estimate)
SESSION_SETUP_H = 0.25   # data download + preparation in every Colab session (estimate)
# Rough prior for unmeasured families (A100, s/iter, global batch 16); overwritten by the first measurement. Source: on an
# RTX 3050, ViT-B LoRA 7.7 samples/s, ConvNeXt-T 10 samples/s; A100 ~10x, ViT-L ~2.9x ViT-B, ConvNeXt-L ~5.5x ConvNeXt-T (FLOP ratio).
# ±40% uncertain; the queue writes the real measurement in the first 200 iterations and with --measure_speed.
PRIOR_S_PER_IT = {'dinov3_l|lora|ours': 0.5, 'dinov2_l|lora|ours': 0.65, 'convnext_l|full|ours': 0.9,
                  'convnext_l|full|paper': 1.0, 'dinov3_l|frozen|ours': 0.3, 'dinov3_b|lora|ours': 0.2}
STATUS_NAME = {'pending': 'pending', 'running': 'running', 'done': 'done', 'error': 'ERROR',
               'awaiting_tau': 'awaiting τ', 'paused': 'paused (time limit)', 'deferred': 'deferred (budget)'}


def now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo('Europe/Istanbul'))
    except Exception:
        return datetime.datetime.now()


def family(k):
    return f"{k['bb']}|{k['mode']}|{k.get('recipe', 'ours')}"


class Queue:
    def __init__(self, root):
        self.root = root
        self.path = os.path.join(root, 'queue.json')
        for d in ('log', 'results', 'ckpt', 'model'):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        if not os.path.exists(self.path):
            init = os.path.join(root, 'queue_initial.json')
            if not os.path.exists(init):
                init = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue_initial.json')
            shutil.copy(init, self.path)

    def read(self):
        for _ in range(5):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                time.sleep(2)
        raise RuntimeError('could not read queue.json')

    def write(self, d):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    def run(self, rid):
        d = self.read()
        for k in d['runs']:
            if k['id'] == rid:
                return k
        raise KeyError(rid)

    def update(self, rid, **kw):
        d = self.read()
        for k in d['runs']:
            if k['id'] == rid:
                k.update(kw)
        self.write(d)
        return d

    def write_speed(self, fam, s_per_it, micro, accum, gpu):
        d = self.read()
        d.setdefault('speed', {})[fam] = {'s_per_it': round(s_per_it, 4), 'micro': micro, 'accum': accum, 'gpu': gpu,
                                          'timestamp': now().strftime('%F %H:%M')}
        self.write(d)

    # ---------------- status.md ----------------
    def estimate(self, d=None):
        d = d or self.read()
        speed = d.get('speed', {})
        rows, spent, remaining = [], 0.0, 0.0
        for k in sorted(d['runs'], key=lambda k: k['priority']):
            fam = family(k)
            meas = speed.get(fam)
            spi = k.get('s_per_it') or (meas or {}).get('s_per_it')
            source = 'measured' if (k.get('s_per_it') or meas) else 'prior (not measured)'
            if spi is None:
                spi = PRIOR_S_PER_IT.get(fam, 0.6)
            full_h = spi * k['total_iter'] / 3600 + EVAL_H
            done_h = k.get('train_s', 0) / 3600
            if k['status'] == 'done':
                rem = 0.0
                full_h = done_h + k.get('eval_s', 0) / 3600
            else:
                rem = spi * max(0, k['total_iter'] - k.get('iter', 0)) / 3600 + EVAL_H
            spent += done_h; remaining += rem
            rows.append((k, spi, source, full_h, rem))
        bench = d.get('measure_s', 0) / 3600
        return rows, spent + bench, remaining

    def write_status(self, active=None):
        """active: {'id', 'iter', 'total', 'it_s', 'remaining_s', 'loss'} (from the training process)."""
        d = self.read()
        rows, spent, remaining = self.estimate(d)
        sessions = d.get('session_count', 1)
        setup = SESSION_SETUP_H * max(1, sessions)
        total = spent + remaining + setup
        budget = float(os.environ.get('BUDGET_H', BUDGET_H)); unit = os.environ.get('BUDGET_UNIT', 'A100-hours')
        t = now()
        L = [f"# PlantSeg full experiment: status", '',
             f"Updated: **{t.strftime('%d.%m %H:%M')}** (Europe/Istanbul) · GPU: {d.get('gpu', '?')} · session: {sessions}", '']
        if active:
            pct = 100 * active['iter'] / max(1, active['total'])
            end = t + datetime.timedelta(seconds=active['remaining_s'])
            L += ['## Now', '',
                  f"**{active['id']}**: {active['iter']:,}/{active['total']:,} iter (**{pct:.1f}%**), "
                  f"{active['it_s']:.2f} it/s, loss {active['loss']:.3f}, ~{active['remaining_s']/3600:.1f} h left, "
                  f"estimated finish **{end.strftime('%d.%m %H:%M')}**", '']
        else:
            L += ['## Now', '', 'No training process (queue idle, finished, or session closed).', '']
        L += ['## Queue', '', '| # | run | status | progress | s/iter | run time (h) | remaining (h) | dev mIoU | speed source |',
              '|---|---|---|---|---|---|---|---|---|']
        for k, spi, source, full_h, rem in rows:
            prog = f"{k.get('iter', 0):,}/{k['total_iter']:,}"
            result = f"{k['dev_miou']:.2f}" if k.get('dev_miou') is not None else ''
            L.append(f"| {k['priority']} | {k['id']} ({k['label']}) | {STATUS_NAME.get(k['status'], k['status'])} | {prog} | "
                     f"{spi:.3f} | {full_h:.1f} | {rem:.1f} | {result} | {source} |")
        errors = [k for k, *_ in rows if k['status'] == 'error']
        L += ['', f'## Budget ({unit})', '',
              f"- Training spent (+speed measurement): **{spent:.1f}**",
              f"- Remaining runs (estimate): **{remaining:.1f}**",
              f"- Session setup ({sessions} sessions × {SESSION_SETUP_H}): {setup:.1f}",
              f"- **Total estimate: {total:.1f} / budget {budget:.0f}**" + (f" (≈ {total*UNITS_PER_H:.0f} / 326 Colab units)" if unit == 'A100-hours' else ''), '']
        if total > budget:
            L += [f"## !!! TOTAL ESTIMATE EXCEEDS BUDGET: {total:.1f} {unit.upper()} > {budget:.0f} !!!", '',
                  'RUNS AT THE END OF THE QUEUE MAY NOT FIT THE BUDGET. PRIORITY ORDER IS KEPT; A DECISION IS NEEDED.', '']
        if errors:
            L += ['## Errors', ''] + [f"- {k['id']} (attempt {k.get('attempt', 0)}): `{(k.get('error') or '')[-300:]}`" for k in errors] + ['']
        L += ['Note: s/iter is per global batch; "prior" rows are rough unmeasured estimates. Details in `log/<run>.log`, '
              'results in `results/<run>.json`.']
        tmp = os.path.join(self.root, 'status.md.tmp')
        with open(tmp, 'w') as f:
            f.write('\n'.join(L) + '\n')
        os.replace(tmp, os.path.join(self.root, 'status.md'))
        return total
