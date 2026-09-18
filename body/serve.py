"""THE DIARY protocol for the second body: the same page and endpoints the caregivers know.

  python3 -m body.serve --birth data/body2.pt --tok data/tok_char.json --port 8018 --period 0.5
  python3 -m body.serve --load  data/body2.pt --tok data/tok_char.json --port 8018

POST /type {"text", "who"}   POST /face {"expr"}   GET /state?since=N   POST /save {}   (no /sleep: the day ends by the body alone; the review of 2026-09-08)
GET /talk   the visitor's page (2026-09-11; redrawn 2026-09-17): a conversation, each line a bubble, the body's speech its own; the letters flow in as typed, no box; the number keys are the face (5 neutral, held until the next); the typist yields for a minute after a visitor types
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
<style>:root{--bg:#fafafa;--ink:#1c1c1e;--mute:#8e8e93;--line:#e5e5ea;--parent:#f0f0f2;--parent-ink:#3a3a3c;--other:#e9eef6;--other-ink:#2c3e5a;--you:#e7f1eb;--you-ink:#1f4a34;--body:#1c1c1e;--body-ink:#fafafa;--blue:#3b6ea5}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI","Helvetica Neue",Arial,sans-serif}
#top{position:fixed;left:0;right:0;top:0;background:rgba(250,250,250,.97);border-bottom:1px solid var(--line)}
#topin{max-width:760px;margin:0 auto;padding:10px 16px 8px;display:flex;align-items:center;gap:14px}
#st{width:8px;height:8px;border-radius:50%;background:#34c759;flex:none}#st.off{background:#c7c7cc}#st.away{background:#f0a030}
#cv{flex:1;height:48px;display:block;background:#fff;border:1px solid var(--line);border-radius:8px;min-width:0}
#mood{display:flex;gap:4px;flex:none}
#mood span{width:26px;height:26px;border-radius:6px;border:1px solid var(--line);display:flex;align-items:center;justify-content:center;font-size:12px;color:var(--mute);cursor:pointer;user-select:none;background:#fff}
#mood span.on{background:var(--ink);color:#fff;border-color:var(--ink)}
#pg{max-width:760px;margin:0 auto;padding:84px 16px 48px}
.b{margin:6px 0;max-width:80%;clear:both}.b .lab{display:block;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--mute);margin:0 0 2px 4px}
.b .t{display:inline-block;padding:8px 12px;border-radius:12px;white-space:pre-wrap;word-break:break-word}
.w{float:left}.w.parent .t{background:var(--parent);color:var(--parent-ink)}.w.other .t{background:var(--other);color:var(--other-ink)}.w.you .t{background:var(--you);color:var(--you-ink)}
.o{float:right;text-align:right}.o .lab{margin:0 4px 2px 0}.o .t{background:var(--body);color:var(--body-ink);font-weight:500}.gap{display:inline-block;width:.4em}
#k{position:fixed;left:0;top:0;width:1px;height:1px;opacity:0;border:0;padding:0}</style>
<div id=top><div id=topin><span id=st title="awake"></span><canvas id=cv width=760 height=48></canvas><div id=mood></div></div></div>
<div id=pg></div><input id=k autocomplete=off autocapitalize=off autocorrect=off spellcheck=false>
<script>const pg=document.getElementById('pg'),k=document.getElementById('k'),st=document.getElementById('st'),moodEl=document.getElementById('mood');let seen=0,held=5,busy=false;
const GAP=8;let wOpen=null,wWho=null,wGap=0,oOpen=null,oGap=0,oSil=0;
function post(p,b){return fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(r=>r.json()).catch(()=>null)}
/* THE FACE IS A NUMBER KEY, held until the next: 5 neutral, 6 to 9 warmer, 4 to 1 colder (the levels -4 to +4 of the face) */
for(let d=1;d<=9;d++){const s=document.createElement('span');s.textContent=d;s.dataset.d=d;s.onclick=()=>setMood(d);moodEl.appendChild(s)}
function setMood(d){held=d;post('/face',{expr:d-5});for(const s of moodEl.children)s.classList.toggle('on',+s.dataset.d===d)}
setMood(5);setInterval(()=>{if(held!==5)post('/face',{expr:held-5})},3000);
/* THE LETTERS FLOW IN AS TYPED: no box, no editing; lowercase words, space, ? . !; Enter is a space; nothing else goes in */
function send(t){t=[...t].filter(c=>/[a-zA-Z ?.!]/.test(c)).map(c=>c==='I'?c:c.toLowerCase()).join('');if(t)post('/type',{text:t,who:'you'})}
let away=false;function leaveRoom(){away=!away;held=5;post('/face',{expr:0});for(const s of moodEl.children)s.classList.toggle('on',!away&&+s.dataset.d===5);st.classList.toggle('away',away);st.title=away?'you have left the room (0 to return)':'awake'}
function takeChar(c){if(c==='0')leaveRoom();else if(away)return;else if(/^[1-9]$/.test(c))setMood(+c);else send(c)}
k.addEventListener('input',()=>{const v=k.value;k.value='';for(const c of v)takeChar(c)});
document.addEventListener('keydown',e=>{if(e.metaKey||e.ctrlKey||e.altKey)return;if(document.activeElement===k&&e.key.length===1&&!/^[1-9]$/.test(e.key))return;
 if(e.key==='Enter'){if(!away)send(' ');e.preventDefault();return}if(e.key===' '){if(!away)send(' ');e.preventDefault();return}
 if(e.key.length===1){takeChar(e.key);e.preventDefault()}});
function refocus(){if(document.activeElement!==k)k.focus({preventScroll:true})}
document.addEventListener('click',e=>{if(e.target.parentElement!==moodEl&&!(window.getSelection&&window.getSelection().toString()))refocus()});refocus();
const F=[];const cv=document.getElementById('cv'),cx=cv.getContext('2d');
function draw(){const W=cv.width,H=cv.height,N=900;cx.clearRect(0,0,W,H);cx.strokeStyle='#e5e5ea';cx.beginPath();cx.moveTo(0,H/2);cx.lineTo(W,H/2);cx.stroke();const y=v=>H/2-Math.max(-6,Math.min(6,v))*(H/2-3)/6;for(const[i,col,lw]of[[0,'#3b6ea5',1.2],[1,'#1c1c1e',1.6]]){cx.strokeStyle=col;cx.lineWidth=lw;cx.beginPath();F.forEach((f,j)=>{const x=(j+N-F.length)*W/N;if(j===0)cx.moveTo(x,y(f[i]));else cx.lineTo(x,y(f[i]))});cx.stroke()}}
function block(kind,who){const b=document.createElement('div');b.className='b '+(kind==='w'?'w '+who:'o');const l=document.createElement('span');l.className='lab';l.textContent=kind==='w'?(who==='parent'?'parent':who==='other'?'other voice':'you'):'body';const t=document.createElement('span');t.className='t';b.appendChild(l);b.appendChild(t);pg.appendChild(b);while(pg.children.length>90){const f=pg.firstChild;if(wOpen&&f.contains(wOpen))wOpen=null;if(oOpen&&f.contains(oOpen))oOpen=null;pg.removeChild(f)}return t}
function feed(es){for(const e of es){const sym=e[0];if(e[1]===0){F.push([e[2]||0,e[3]||0]);if(F.length>900)F.shift();if(sym){const who=e[4]||'you';if(!wOpen||who!==wWho){wOpen=block('w',who);wWho=who}wOpen.appendChild(document.createTextNode(sym));wGap=0}else{wGap++;if(wOpen&&wGap>=GAP)wOpen=null}}
 else{if(sym){if(!oOpen&&sym===' '){oGap=0;continue}if(!oOpen){oOpen=block('o');oSil=0}else if(oSil>=3){const g=document.createElement('span');g.className='gap';oOpen.appendChild(g)}oOpen.appendChild(document.createTextNode(sym));oGap=0;oSil=0}else{oGap++;oSil++;if(oOpen&&oGap>=GAP)oOpen=null}}}}
async function poll(){if(busy)return;busy=true;let d;try{d=await fetch('/state?since='+seen).then(r=>r.json())}catch(err){d=null}finally{busy=false}
 if(!d||!d.page){st.className='off';st.title='no answer from the body';return}
 if(d.n<seen||(d.base!==undefined&&d.base>seen&&seen>0)){seen=0;pg.textContent='';wOpen=oOpen=null}
 const es=(seen===0)?d.page.slice(-4000):d.page;const atBottom=window.innerHeight+window.scrollY>=document.body.scrollHeight-80;feed(es);draw();seen=d.n;
 st.className=d.asleep?'off':'';st.title=d.asleep?'asleep':'awake';if(atBottom)window.scrollTo(0,document.body.scrollHeight)}
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
