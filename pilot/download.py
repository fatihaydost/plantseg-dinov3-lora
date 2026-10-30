"""Download the remainder of plantseg.zip with parallel HTTP Range requests, then concatenate the parts."""
import os, sys, time, threading, urllib.request

URL = 'https://zenodo.org/api/records/17719108/files/plantseg.zip/content'
T = 1057281724
D = os.path.expanduser('~/.cache/plantseg')
Z = f'{D}/plantseg.zip'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
os.makedirs(f'{D}/parts', exist_ok=True)
S = os.path.getsize(Z)
R = (T - S + N - 1) // N
rngs = [(S + i * R, min(T - 1, S + (i + 1) * R - 1)) for i in range(N)]


def work(i, a, b):
    p = f'{D}/parts/p{i}'
    for k in range(200):
        have = os.path.getsize(p) if os.path.exists(p) else 0
        st = a + have
        if st > b:
            return
        try:
            req = urllib.request.Request(URL, headers={'Range': f'bytes={st}-{b}'})
            with urllib.request.urlopen(req, timeout=60) as r, open(p, 'ab') as f:
                while True:
                    c = r.read(1 << 16)
                    if not c:
                        break
                    f.write(c)
        except Exception as e:
            print(i, 'error', e, flush=True); time.sleep(5)


th = [threading.Thread(target=work, args=(i, a, b)) for i, (a, b) in enumerate(rngs)]
[t.start() for t in th]
while any(t.is_alive() for t in th):
    time.sleep(30)
    got = sum(os.path.getsize(f'{D}/parts/p{i}') for i in range(N) if os.path.exists(f'{D}/parts/p{i}'))
    print(time.strftime('%H:%M:%S'), got, T - S, flush=True)
with open(Z, 'ab') as f:
    for i in range(N):
        f.write(open(f'{D}/parts/p{i}', 'rb').read())
print('done', os.path.getsize(Z), T, flush=True)
