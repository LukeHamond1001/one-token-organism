"""THE DIARY protocol for the second body: the same page and endpoints the caregivers know.

  python3 -m body.serve --birth data/body2.pt --tok data/tok_char.json --port 8018 --period 0.5
  python3 -m body.serve --load  data/body2.pt --tok data/tok_char.json --port 8018

POST /type {"text"}   POST /face {"expr"}   GET /state?since=N   POST /save {}   (no /sleep: the day ends by the body alone; the review of 2026-09-08)
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from tokenizers import Tokenizer

from .life import Life, PHYSIOLOGY

PAGE = """<!doctype html><meta charset=utf-8><title>the diary, second body</title>
<style>body{margin:0;background:#f5f1e6;color:#222;font:16px/1.6 Georgia,serif}
#pg{white-space:pre-wrap;padding:32px 40px 120px;max-width:820px;margin:0 auto}.u{color:#1a1a1a}.n{color:#b08a5a}
#bar{position:fixed;left:0;right:0;bottom:0;background:#eae4d3;border-top:1px solid #cbbfa3;padding:10px 40px;font:13px ui-monospace,monospace;display:flex;gap:18px;flex-wrap:wrap}</style>
<div id=pg></div><div id=bar><span>you <b id=you>0</b></span><span>its face <b id=face>-</b></span><span id=night></span></div>
<script>let face=0,seen=0;const pg=document.getElementById('pg');
function post(p,b){return fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(r=>r.json())}
document.addEventListener('keydown',e=>{if(e.metaKey||e.ctrlKey||e.altKey)return;
 if(e.key==='ArrowUp'){face=Math.min(6,face+0.5);post('/face',{expr:face});e.preventDefault();return}
 if(e.key==='ArrowDown'){face=Math.max(-6,face-0.5);post('/face',{expr:face});e.preventDefault();return}
 if(e.key.length===1){post('/type',{text:e.key});e.preventDefault()}});
async function poll(){const d=await fetch('/state?since='+seen).then(r=>r.json());
 for(const [t,who] of d.page){const s=document.createElement('span');s.className=who?'n':'u';s.textContent=t;pg.appendChild(s)}
 seen=d.n;const last=d.page.length?d.page[d.page.length-1]:null;if(last){document.getElementById('you').textContent=last[2];document.getElementById('face').textContent=last[3]}
 document.getElementById('night').textContent=d.asleep?'asleep':('nights '+d.nights);
 window.scrollTo(0,document.body.scrollHeight)}
setInterval(poll,500);</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--birth", default=None, help="a new body, saved here")
    ap.add_argument("--load", default=None, help="a living body")
    ap.add_argument("--tok", default="data/tok_char.json")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--port", type=int, default=8018)
    ap.add_argument("--period", type=float, default=0.5, help="seconds per tick")
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    for k, v in PHYSIOLOGY.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None, help=f"physiology (default {v})")
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in PHYSIOLOGY if getattr(a, k) is not None}
    tok = Tokenizer.from_file(a.tok)
    if a.load:
        life = Life.load(a.load, tok, device=a.dev, cfg=cfg, seed=a.seed, save_path=a.load)
        print(f"[body] loaded {a.load}: ticks {life.ticks} nights {life.nights} store {life.store.n()}", flush=True)
    else:
        life = Life.birth(tok, device=a.dev, d=a.d, layers=a.layers, heads=a.heads, window=a.window, cfg=cfg, seed=a.seed, save_path=a.birth)
        life.save()
        print(f"[body] born: {sum(p.numel() for p in life.m.parameters())/1e6:.1f}M parameters, saved {a.birth}", flush=True)
    lock = threading.RLock()

    def loop():
        while True:
            t0 = time.time()
            try:
                with lock:
                    if not life.asleep:
                        life.tick()
            except Exception as e:
                life.last = {"error": str(e)[:200], "tick": life.ticks}
                print(f"[body] tick error: {e}", flush=True)
            time.sleep(max(0.0, a.period - (time.time() - t0)))
    threading.Thread(target=loop, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/state"):
                since = 0
                if "since=" in self.path:
                    try: since = int(self.path.split("since=")[1].split("&")[0])
                    except Exception: since = 0
                self._json(life.state(since))
            elif self.path.startswith("/insides"):                # the supervisor's instrument, never the caregiver's
                self._json(life.insides())
            else:
                b = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            body = json.loads(self.rfile.read(n).decode() or "{}") if n else {}
            try:
                if self.path == "/type":
                    self._json(life.type_text(str(body.get("text", ""))))
                elif self.path == "/face":
                    self._json(life.set_face(body.get("expr", 0)))
                elif self.path == "/save":
                    with lock:
                        self._json(life.save())
                else:
                    self._json({"error": "unknown path"}, 404)
            except Exception as e:
                self._json({"error": str(e)[:300]}, 500)

    print(f"[body] the page is open on http://localhost:{a.port} (tick {a.period}s, {a.dev})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
