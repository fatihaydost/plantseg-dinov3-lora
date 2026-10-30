"""Queue runner: runs the runs in queue.json in priority order. Skips finished ones, resumes an interrupted one ('running',
'paused') from its last checkpoint, retries a failed one at most 3 times, skips 'awaiting_tau'.
Usage: python runner.py --root <drive folder> --data <prep> [--measure_speed] [--nproc 2] [--only id1,id2]"""
import argparse, collections, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_queue import Queue, family, now

RUNNABLE = ('pending', 'running', 'paused', 'error')
HERE = os.path.dirname(os.path.abspath(__file__))


def command(a, rid, extra=()):
    if a.nproc > 1:
        return [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={a.nproc}', f'{HERE}/train.py',
                '--root', a.root, '--data', a.data, '--run', rid, *extra]
    return [sys.executable, f'{HERE}/train.py', '--root', a.root, '--data', a.data, '--run', rid, *extra]


def execute(cmd):
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
    last = collections.deque(maxlen=40)
    for line in p.stdout:
        last.append(line.rstrip())
        print(line, end='', flush=True)
    return p.wait(), '\n'.join(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--measure_speed', action='store_true', help='short speed measurement for every unmeasured model family (budget estimate)')
    ap.add_argument('--bench_iter', type=int, default=40)
    ap.add_argument('--nproc', type=int, default=1); ap.add_argument('--only', default='')
    ap.add_argument('--max_attempts', type=int, default=3)
    a = ap.parse_args()
    if os.environ.get('TIME_LIMIT_H') and not os.environ.get('SESSION_END'):  # session-wide time limit
        os.environ['SESSION_END'] = str(time.time() + float(os.environ['TIME_LIMIT_H']) * 3600)
    Q = Queue(a.root)
    d = Q.read(); d['session_count'] = d.get('session_count', 0) + 1; Q.write(d)
    only = set(filter(None, a.only.split(',')))
    sel = lambda k: (not only or k['id'] in only)

    if a.measure_speed:
        d = Q.read(); seen = set(d.get('speed', {}))
        for k in sorted(d['runs'], key=lambda k: k['priority']):
            if family(k) in seen or k['status'] == 'done' or not sel(k):
                continue
            seen.add(family(k))
            print(f"\n=== speed measurement: {family(k)} ({k['id']}, {a.bench_iter} iter) ===", flush=True)
            rc, last = execute(command(a, k['id'], ['--bench', str(a.bench_iter)]))
            if rc != 0:
                print(f'speed measurement failed ({rc}); continuing\n{last[-1500:]}', flush=True)
        Q.write_status()

    while True:
        d = Q.read()
        cand = [k for k in d['runs'] if k['status'] in RUNNABLE and sel(k)
                and not (k['status'] == 'error' and k.get('attempt', 0) >= a.max_attempts)]
        if not cand:
            break
        k = min(cand, key=lambda k: k['priority'])
        if os.environ.get('SESSION_END') and time.time() > float(os.environ['SESSION_END']) - 600:
            print('session time almost up; no new run started', flush=True)
            break
        print(f"\n=== {now().strftime('%F %H:%M')} run {k['id']} ({k['label']}), status {k['status']}, "
              f"iter {k.get('iter', 0)}/{k['total_iter']} ===", flush=True)
        rc, last = execute(command(a, k['id']))
        k2 = Q.run(k['id'])
        if rc == 3 or k2['status'] == 'paused':  # torchrun turns the child's 3 into 1; read the status instead
            print('time limit: runner stopped (checkpoint saved)', flush=True)
            break
        if rc != 0 or k2['status'] != 'done':
            Q.update(k['id'], status='error', attempt=k2.get('attempt', 0) + 1, error=f'rc={rc}\n' + last[-2000:])
            with open(os.path.join(a.root, 'log', f"{k['id']}.log"), 'a') as f:
                f.write(f"== {now().strftime('%F %H:%M:%S')} ERROR rc={rc}\n{last[-3000:]}\n")
            print(f"ERROR {k['id']} rc={rc}", flush=True)
            time.sleep(5)
        Q.write_status()
    Q.write_status()
    d = Q.read()
    print('\nQUEUE SUMMARY:', flush=True)
    for k in sorted(d['runs'], key=lambda k: k['priority']):
        print(f"  {k['priority']:>2} {k['id']:<14} {k['status']:<15} {k.get('iter', 0)}/{k['total_iter']} "
              f"dev mIoU {k.get('dev_miou')}", flush=True)


if __name__ == '__main__':
    main()
