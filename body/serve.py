"""THE DIARY protocol for the second body: the same page and endpoints the caregivers know.

  python3 -m body.serve --birth data/body2.pt --tok data/tok_char.json --port 8018 --period 0.5
  python3 -m body.serve --load  data/body2.pt --tok data/tok_char.json --port 8018

POST /type {"text", "who"}   POST /face {"expr"}   GET /state?since=N   POST /save {}   (no /sleep: the day ends by the body alone; the review of 2026-09-08)
GET /talk   the visitor's page (2026-09-11; redrawn 2026-09-17 as a conversation: each line a bubble, the body's speech its own, a box that types a line at the tick rate); a smile and a frown button; the typist yields for a minute after a visitor types
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

TALK = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>talk to the body</title>
<style>body{margin:0;background:#f5f1e6;color:#222;font:17px/1.55 Georgia,serif}
#pg{padding:96px 18px 190px;max-width:820px;margin:0 auto}
#top{position:fixed;left:0;right:0;top:0;background:#eae4d3;border-bottom:1px solid #cbbfa3;padding:6px 18px 4px}
#cv{display:block;width:100%;max-width:820px;height:64px;margin:0 auto;background:#f5f1e6;border:1px solid #d9d0b8;border-radius:6px}
#cl{display:block;max-width:820px;margin:3px auto 0;font:11px ui-monospace,monospace;color:#6b6252}
.b{margin:7px 0;max-width:78%;clear:both}.b .lab{display:block;font:11px ui-monospace,monospace;color:#8b8474;margin-bottom:1px}
.b .t{display:inline-block;padding:7px 12px;border-radius:12px;white-space:pre-wrap;word-break:break-word}
.w{float:left}.w.parent .t{background:#efe7d4;color:#5e4f33}.w.other .t{background:#e2eedb;color:#2f5f2f}.w.you .t{background:#dde6f3;color:#1f4e8c}
.o{float:right;text-align:right}.o .t{background:#111;color:#fff9ea;font-weight:bold;letter-spacing:.02em}.gap{display:inline-block;width:.45em}
#bar{position:fixed;left:0;right:0;bottom:0;background:#eae4d3;border-top:1px solid #cbbfa3;padding:10px 18px 12px}
#row{display:flex;gap:10px;align-items:center;max-width:820px;margin:0 auto}
#box{flex:1;font:18px Georgia,serif;padding:9px 12px;border:1px solid #b9ac8c;border-radius:8px;background:#fff}
button{font:16px Georgia,serif;padding:8px 16px;border:1px solid #b9ac8c;border-radius:8px;background:#fff9ea;cursor:pointer}
button:active{background:#e6dcc0}#st{max-width:820px;margin:6px auto 0;font:12px ui-monospace,monospace;color:#6b6252;display:flex;gap:14px;flex-wrap:wrap}
#help{max-width:820px;margin:4px auto 0;font:12px ui-monospace,monospace;color:#8b8474}</style>
<div id=top><canvas id=cv width=820 height=64></canvas><span id=cl>the faces over the last three minutes, tick by tick: blue the room's face (its parents' smiles and frowns, and yours), black its own face. -6 to +6; the middle line is neutral.</span></div>
<div id=pg></div>
<div id=bar><div id=row><input id=box autocomplete=off autocapitalize=off autocorrect=off spellcheck=false placeholder="type a line and press Enter"><button id=sm title="smile +2 (arrow up)">smile</button><button id=bg title="big smile +4 (shift + arrow up)">big smile</button><button id=fr title="frown -2 (arrow down)">frown</button></div>
<div id=st><span id=s1></span><span id=s2></span></div>
<div id=help>your line goes in one letter a tick (five a second), like its parents' lines; it answers the same way, in black. smile when it says something sensible (arrow up, +2); big smile when it answers you (shift + arrow up, +4); frown when it talks over you (arrow down, -2); the arrow keys work while the box is empty. the body learns from the surprise in your face, not its size: a smile it did not expect counts most. brown is its parent, green the other voice it overhears, blue is you. the parent steps back for a minute after you type.</div></div>
<script>const pg=document.getElementById('pg'),box=document.getElementById('box');let seen=0,faceT=null;
const GAP=8;let wOpen=null,wWho=null,wGap=0,oOpen=null,oGap=0,oSil=0;
function post(p,b){return fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(r=>r.json())}
function face(v){post('/face',{expr:v});if(faceT)clearTimeout(faceT);faceT=setTimeout(()=>post('/face',{expr:0}),1250);flash(v>0?'smile':'frown')}
const F=[];const cv=document.getElementById('cv'),cx=cv.getContext('2d');
function draw(){const W=cv.width,H=cv.height,N=900;cx.clearRect(0,0,W,H);cx.strokeStyle='#cbbfa3';cx.beginPath();cx.moveTo(0,H/2);cx.lineTo(W,H/2);cx.stroke();const y=v=>H/2-Math.max(-6,Math.min(6,v))*(H/2-4)/6;for(const[k,col]of[[1,'#111'],[0,'#1f4e8c']]){cx.strokeStyle=col;cx.lineWidth=k?1.6:1.2;cx.beginPath();F.forEach((f,i)=>{const x=(i+N-F.length)*W/N;if(i===0)cx.moveTo(x,y(f[k]));else cx.lineTo(x,y(f[k]))});cx.stroke()}}
function flash(w){const b=document.getElementById(w==='smile'?'sm':'fr');b.style.background='#e6dcc0';setTimeout(()=>b.style.background='',250)}
function send(t){t=[...t].filter(c=>/[a-zA-Z ?.!]/.test(c)).map(c=>c==='I'?c:c.toLowerCase()).join('').trim();if(t)post('/type',{text:t,who:'you'})}
function block(kind,who){const b=document.createElement('div');b.className='b '+(kind==='w'?'w '+who:'o');const l=document.createElement('span');l.className='lab';l.textContent=kind==='w'?(who==='parent'?'its parent':who==='other'?'the other voice':'you'):'the body';const t=document.createElement('span');t.className='t';b.appendChild(l);b.appendChild(t);pg.appendChild(b);while(pg.children.length>90){const f=pg.firstChild;if(wOpen&&f.contains(wOpen))wOpen=null;if(oOpen&&f.contains(oOpen))oOpen=null;pg.removeChild(f)}return t}
function feed(es){for(const e of es){const sym=e[0];if(e[1]===0){F.push([e[2]||0,e[3]||0]);if(F.length>900)F.shift();if(sym){const who=e[4]||'you';if(!wOpen||who!==wWho){wOpen=block('w',who);wWho=who}wOpen.appendChild(document.createTextNode(sym));wGap=0}else{wGap++;if(wOpen&&wGap>=GAP)wOpen=null}}
 else{if(sym){if(!oOpen){oOpen=block('o');oSil=0}else if(oSil>=3){const g=document.createElement('span');g.className='gap';oOpen.appendChild(g)}oOpen.appendChild(document.createTextNode(sym));oGap=0;oSil=0}else{oGap++;oSil++;if(oOpen&&oGap>=GAP)oOpen=null}}}}
document.getElementById('sm').onclick=()=>face(2);document.getElementById('bg').onclick=()=>face(4);document.getElementById('fr').onclick=()=>face(-2);
box.addEventListener('keydown',e=>{if(e.key==='Enter'){send(box.value);box.value='';e.preventDefault()}});
document.addEventListener('keydown',e=>{if(e.metaKey||e.ctrlKey||e.altKey)return;if(box.value!=='')return;if(e.key==='ArrowUp'){face(e.shiftKey?4:2);e.preventDefault()}else if(e.key==='ArrowDown'){face(-2);e.preventDefault()}});
document.addEventListener('click',e=>{if(e.target.tagName!=='BUTTON'&&!(window.getSelection&&window.getSelection().toString()))box.focus({preventScroll:true})});box.focus();
let busy=false;
async function poll(){if(busy)return;busy=true;let d;try{d=await fetch('/state?since='+seen).then(r=>r.json())}catch(err){d=null}finally{busy=false}
 if(!d||!d.page){document.getElementById('s1').textContent='no answer from the body (is it being restarted?)';return}
 if(d.n<seen||(d.base!==undefined&&d.base>seen&&seen>0)){seen=0;pg.textContent='';wOpen=oOpen=null}
 const es=(seen===0)?d.page.slice(-4000):d.page;const atBottom=window.innerHeight+window.scrollY>=document.body.scrollHeight-80;feed(es);draw();seen=d.n;
 document.getElementById('s1').textContent=(d.asleep?'asleep (a night is ten to seventeen minutes)':'awake')+' · nights '+d.nights;
 document.getElementById('s2').textContent=d.queued?('typing in ('+d.queued+' letters to go)'):'';if(atBottom)window.scrollTo(0,document.body.scrollHeight)}
setInterval(poll,400);poll();</script>"""


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
            elif self.path.startswith("/talk"):
                b = TALK.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            else:
                b = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            body = json.loads(self.rfile.read(n).decode() or "{}") if n else {}
            try:
                if self.path == "/type":
                    self._json(life.type_text(str(body.get("text", "")), who=str(body.get("who", "you"))))   # the diary page's keys are a visitor's too
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
