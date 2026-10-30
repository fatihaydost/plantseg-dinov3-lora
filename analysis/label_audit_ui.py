"""Local rating page for the label audit: shows pairs side by side, writes each Y/N answer to form.csv immediately.

Usage: python3 analysis/label_audit_ui.py [--port 8791]  -> open http://localhost:8791
Shortcuts: Y = same photo, N = different, ← → = previous / next. Writes only form.csv; never reads rater_model.csv
or key.csv (independent rating).
"""
import argparse
import csv
import json
import os
import pathlib
import tempfile
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent
FORM = ROOT / "results/label_audit/form.csv"
RAW = pathlib.Path.home() / ".cache/plantseg/raw/plantseg"
META = pathlib.Path.home() / ".cache/plantseg/prep/meta.json"
ANSWER, NOTE = "same_photo (Y/N)", "note"


def read():
    with open(FORM, newline="") as f:
        return list(csv.DictReader(f))


def write(rows):
    """Atomic write: an interrupted save must not corrupt the form."""
    fd, tmp = tempfile.mkstemp(dir=FORM.parent, suffix=".csv")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, FORM)


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label audit</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--text:#1d1d1b;--muted:#6b6b66;--y:#1f7a4d;--n:#b3261e;--border:#dcdad4}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#202020;--text:#ecebe7;--muted:#a09f99;--border:#3a3a37}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--border);flex-wrap:wrap}
header b{font-size:17px}#progress{color:var(--muted)}#bar{flex:1;min-width:120px;height:6px;background:var(--border);border-radius:3px}
#bar i{display:block;height:100%;background:var(--y);border-radius:3px;width:0}
main{padding:16px;max-width:1400px;margin:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:760px){.pair{grid-template-columns:1fr}}
figure{margin:0;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px}
figure img{width:100%;max-height:62vh;object-fit:contain;display:block;background:#0001;border-radius:6px}
figcaption{margin-top:8px;color:var(--muted);font-size:13px}
.controls{display:flex;gap:12px;align-items:center;margin-top:16px;flex-wrap:wrap}
button{font:inherit;padding:12px 22px;border-radius:8px;border:2px solid var(--border);background:var(--card);color:var(--text);cursor:pointer}
button.y.sel{background:var(--y);border-color:var(--y);color:#fff}button.n.sel{background:var(--n);border-color:var(--n);color:#fff}
button:focus-visible{outline:3px solid #4a7dff;outline-offset:2px}
input{flex:1;min-width:200px;font:inherit;padding:11px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text)}
.hint{color:var(--muted);font-size:13px;margin-top:10px}#status{color:var(--muted);font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:4px;margin-top:18px}nav a{width:30px;text-align:center;padding:3px 0;border-radius:4px;
border:1px solid var(--border);color:var(--text);text-decoration:none;font-size:12px}nav a.Y{background:var(--y);color:#fff;border-color:var(--y)}
nav a.N{background:var(--n);color:#fff;border-color:var(--n)}nav a.cur{outline:2px solid #4a7dff}
</style>
<header><b>Same photo?</b><span id="progress"></span><div id="bar"><i></i></div><span id="status"></span></header>
<main>
<div class="pair"><figure><img id="ia" alt="image A"><figcaption id="ca"></figcaption></figure>
<figure><img id="ib" alt="image B"><figcaption id="cb"></figcaption></figure></div>
<div class="controls"><button class="y" id="by">Y · same photo</button><button class="n" id="bn">N · different</button>
<input id="note" placeholder="Note (optional)"><button id="prev">←</button><button id="next">→</button></div>
<div class="hint">Cropping, resizing, compression or a slight colour shift count as the same. Which label is correct is not asked.
Shortcuts: Y / N, ← →. Every answer is saved to form.csv immediately.</div>
<nav id="nav"></nav></main>
<script>
let C=[],i=0;const $=x=>document.getElementById(x);
async function load(){C=await (await fetch('/api/pairs')).json();const e=C.findIndex(c=>!c.answer);i=e<0?0:e;show()}
function show(){const c=C[i];$('ia').src='/img/'+encodeURIComponent(c.A);$('ib').src='/img/'+encodeURIComponent(c.B);$('ca').textContent='#'+c.no+'A · '+c.A_label;
$('cb').textContent='#'+c.no+'B · '+c.B_label;$('note').value=c.note||'';$('by').classList.toggle('sel',c.answer==='Y');
$('bn').classList.toggle('sel',c.answer==='N');const n=C.filter(c=>c.answer).length;$('progress').textContent=`Pair ${c.no}/${C.length} · ${n} answered`;
$('bar').firstChild.style.width=(100*n/C.length)+'%';$('nav').innerHTML=C.map((x,k)=>`<a href="#" data-k="${k}" class="${x.answer||''} ${k===i?'cur':''}">${x.no}</a>`).join('')}
async function save(answer){const c=C[i];c.answer=answer??c.answer;c.note=$('note').value;$('status').textContent='saving…';
const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({no:c.no,answer:c.answer,note:c.note})});
$('status').textContent=r.ok?'saved ✓':'NOT SAVED';if(!r.ok)return false;return true}
async function answer(x){if(await save(x)&&i<C.length-1)i++;show()}
$('by').onclick=()=>answer('Y');$('bn').onclick=()=>answer('N');
$('next').onclick=async()=>{await save();if(i<C.length-1)i++;show()};$('prev').onclick=async()=>{await save();if(i>0)i--;show()};
$('nav').onclick=async e=>{const k=e.target.dataset.k;if(k===undefined)return;e.preventDefault();await save();i=+k;show()};
document.onkeydown=e=>{if(e.target.id==='note')return;const k=e.key.toLowerCase();if(k==='y')answer('Y');else if(k==='n')answer('N');
else if(e.key==='ArrowRight')$('next').click();else if(e.key==='ArrowLeft')$('prev').click()};
load();
</script></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/api/pairs":
            v = [{"no": s["no"], "A": s["A"], "B": s["B"], "A_label": s["A_label"],
                  "B_label": s["B_label"], "answer": s[ANSWER], "note": s[NOTE]} for s in read()]
            return self._send(200, json.dumps(v, ensure_ascii=False).encode(), "application/json")
        if self.path.startswith("/img/"):
            stem = urllib.parse.unquote(self.path[5:])
            if stem not in ALLOWED:  # only images in the form; no path traversal
                return self._send(404, b"not found", "text/plain")
            return self._send(200, (RAW / "images" / ALLOWED[stem] / f"{stem}.jpg").read_bytes(), "image/jpeg")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            return self._send(404, b"not found", "text/plain")
        g = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if g.get("answer") not in ("Y", "N", "", None):
            return self._send(400, b"answer must be Y or N", "text/plain")
        rows = read()
        target = [s for s in rows if s["no"] == str(g.get("no"))]
        if not target:
            return self._send(400, b"unknown no", "text/plain")
        target[0][ANSWER] = g.get("answer") or ""
        target[0][NOTE] = str(g.get("note") or "").replace("\n", " ")[:300]
        write(rows)
        self._send(200, b"ok", "text/plain")

    def log_message(self, *a):
        pass


def main():
    global ALLOWED
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    a = ap.parse_args()
    meta = json.loads(META.read_text())
    ALLOWED = {s[k]: meta[s[k]]["split"] for s in read() for k in ("A", "B")}
    assert all((RAW / "images" / sp / f"{st}.jpg").exists() for st, sp in ALLOWED.items()), "raw image missing"
    print(f"http://localhost:{a.port}  ({len(read())} pairs, form: {FORM.relative_to(ROOT)})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
