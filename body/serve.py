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

TALK = """<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name=color-scheme content="light dark"><title>talk to the body</title>
<script>(function(){var m=/[?&]theme=(dark|light)/.exec(location.search);if(m)document.documentElement.setAttribute('data-theme',m[1])})()</script>
<style>/* UI v2 (2026-09-24): two themes, light by default and dark for the film; ?theme=dark|light, else the system's. Display only: the requests, the polling, the keys and the bubbles' logic are the page of 2026-09-17 */
:root{color-scheme:light;--sans:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display","Inter","Segoe UI","Helvetica Neue",Arial,sans-serif;
--bg:#F5F5F7;--bg-img:none;--ink:#1D1D1F;--ink2:#3A3A3C;--mute:#6E6E73;--hair:rgba(0,0,0,.08);
--bar:rgba(245,245,247,.985);--bar-line:rgba(0,0,0,.08);--panel:#FFFFFF;--panel-line:rgba(0,0,0,.08);--panel-shadow:0 1px 2px rgba(0,0,0,.04);
--grid:rgba(0,0,0,.06);--zero:rgba(0,0,0,.14);--face:#3B7BC4;--its:#1D1D1F;--face-fill:rgba(59,123,196,.22);
--key:#FFFFFF;--key-line:rgba(0,0,0,.10);--key-ink:#6E6E73;--key-cool-ink:#3D6896;--key-warm-ink:#935A33;--key-on:#1D1D1F;--key-on-ink:#FFFFFF;--key-shadow:0 1px 1px rgba(0,0,0,.03);
--warm:#E39B6B;--warm-ink:#2B1609;--warm-halo:rgba(227,155,107,.24);--cool:#7FB2E5;--cool-ink:#0C2036;--cool-halo:rgba(127,178,229,.30);
--live:#3B7BC4;--live-halo:rgba(59,123,196,.20);--off:#C7C7CC;
--you:#D0E1F6;--you-ink:#0F2E52;--you-line:rgba(15,46,82,.07);--parent:#FFFFFF;--parent-ink:#1D1D1F;--parent-line:rgba(0,0,0,.07);--parent-shadow:0 1px 2px rgba(0,0,0,.06);
--other:rgba(255,255,255,0);--other-ink:#48484A;--other-line:rgba(0,0,0,.22);--body:#1D1D1F;--body-ink:#F5F5F7;
--bub-shadow:0 1px 1px rgba(0,0,0,.02);--body-shadow:0 1px 1px rgba(0,0,0,.10),0 6px 16px -4px rgba(0,0,0,.18);--sel:rgba(59,123,196,.2)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
--bg:#141518;--bg-img:radial-gradient(120% 80% at 50% -20%,#24262C 0%,#17181C 46%,#111215 100%);--ink:#ECECEE;--ink2:#C4C5CB;--mute:#8A8C94;--hair:rgba(255,255,255,.09);
--bar:rgba(22,23,27,.985);--bar-line:rgba(255,255,255,.07);--panel:rgba(255,255,255,.03);--panel-line:rgba(255,255,255,.08);--panel-shadow:0 0 0 transparent;
--grid:rgba(255,255,255,.05);--zero:rgba(255,255,255,.14);--face:#7FB2E5;--its:#F2F2F4;--face-fill:rgba(127,178,229,.26);
--key:rgba(255,255,255,.035);--key-line:rgba(255,255,255,.09);--key-ink:#8E9097;--key-cool-ink:#86A5C6;--key-warm-ink:#C4906F;--key-on:#ECECEE;--key-on-ink:#141518;--key-shadow:0 0 0 transparent;
--live:#7FB2E5;--live-halo:rgba(127,178,229,.20);--off:#55575E;
--you:#22364D;--you-ink:#D8E7F8;--you-line:rgba(127,178,229,.16);--parent:#2B2C31;--parent-ink:#D4D5DA;--parent-line:rgba(255,255,255,.05);--parent-shadow:0 0 0 transparent;
--other:rgba(255,255,255,0);--other-ink:#C3C5CB;--other-line:rgba(255,255,255,.2);--body:#ECECEE;--body-ink:#131417;
--bub-shadow:0 0 0 transparent;--body-shadow:inset 0 1px 0 rgba(255,255,255,.6),0 8px 24px -6px rgba(0,0,0,.55);--sel:rgba(127,178,229,.3)}}
:root[data-theme=dark]{color-scheme:dark;
--bg:#141518;--bg-img:radial-gradient(120% 80% at 50% -20%,#24262C 0%,#17181C 46%,#111215 100%);--ink:#ECECEE;--ink2:#C4C5CB;--mute:#8A8C94;--hair:rgba(255,255,255,.09);
--bar:rgba(22,23,27,.985);--bar-line:rgba(255,255,255,.07);--panel:rgba(255,255,255,.03);--panel-line:rgba(255,255,255,.08);--panel-shadow:0 0 0 transparent;
--grid:rgba(255,255,255,.05);--zero:rgba(255,255,255,.14);--face:#7FB2E5;--its:#F2F2F4;--face-fill:rgba(127,178,229,.26);
--key:rgba(255,255,255,.035);--key-line:rgba(255,255,255,.09);--key-ink:#8E9097;--key-cool-ink:#86A5C6;--key-warm-ink:#C4906F;--key-on:#ECECEE;--key-on-ink:#141518;--key-shadow:0 0 0 transparent;
--live:#7FB2E5;--live-halo:rgba(127,178,229,.20);--off:#55575E;
--you:#22364D;--you-ink:#D8E7F8;--you-line:rgba(127,178,229,.16);--parent:#2B2C31;--parent-ink:#D4D5DA;--parent-line:rgba(255,255,255,.05);--parent-shadow:0 0 0 transparent;
--other:rgba(255,255,255,0);--other-ink:#C3C5CB;--other-line:rgba(255,255,255,.2);--body:#ECECEE;--body-ink:#131417;
--bub-shadow:0 0 0 transparent;--body-shadow:inset 0 1px 0 rgba(255,255,255,.6),0 8px 24px -6px rgba(0,0,0,.55);--sel:rgba(127,178,229,.3)}
html{background:var(--bg);scrollbar-width:thin;scrollbar-color:var(--hair) transparent}
body{margin:0;min-height:100vh;background:var(--bg) var(--bg-img) no-repeat fixed;color:var(--ink);font:400 17px/1.4 var(--sans);-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;text-rendering:optimizeLegibility}
::selection{background:var(--sel)}
#top{position:fixed;left:0;right:0;top:0;z-index:5;background:var(--bar);-webkit-backdrop-filter:saturate(180%) blur(20px);backdrop-filter:saturate(180%) blur(20px);border-bottom:1px solid var(--bar-line);padding-top:env(safe-area-inset-top)}
#top::before{content:'';position:absolute;left:0;right:0;top:100%;height:24px;background:linear-gradient(var(--bg),transparent);pointer-events:none} #topin{max-width:840px;margin:0 auto;padding:12px 16px;display:flex;align-items:center;gap:14px}
#st{width:8px;height:8px;box-sizing:border-box;border-radius:50%;background:var(--live);flex:none;box-shadow:0 0 0 2px var(--live-halo)}
#st.off{background:var(--off);box-shadow:none;animation:none}#st.away{background:transparent;border:1.5px solid var(--mute);box-shadow:none;animation:none}
@keyframes br{0%,100%{box-shadow:0 0 0 2px var(--live-halo)}50%{box-shadow:0 0 0 5px rgba(0,0,0,0)}}
.gw{position:relative;flex:1;min-width:0;height:64px;border-radius:11px;background:var(--panel);box-shadow:inset 0 0 0 1px var(--panel-line),var(--panel-shadow)}
#cv{position:absolute;left:0;top:0;width:100%;height:100%;display:block}
.lg{position:absolute;left:11px;top:6px;display:flex;gap:14px;font:500 10px/12px var(--sans);letter-spacing:.01em;color:var(--ink2);white-space:nowrap;user-select:none;-webkit-user-select:none;pointer-events:none}
.lg span{display:flex;align-items:center;gap:6px}.lg i{display:block;width:14px;border-radius:2px}.lg .f i{height:2.6px;background:linear-gradient(90deg,var(--warm) 50%,var(--cool) 50%)}.lg .s i{height:1.5px;background:var(--its)}
#mood{display:flex;gap:5px;flex:none}
#mood span{width:30px;height:30px;box-sizing:border-box;border-radius:9px;border:1px solid var(--key-line);background:var(--key);box-shadow:var(--key-shadow);display:flex;align-items:center;justify-content:center;font:600 13px/1 var(--sans);font-variant-numeric:tabular-nums;color:var(--key-ink);cursor:pointer;user-select:none;-webkit-user-select:none;-webkit-tap-highlight-color:transparent;transition:background-color .18s ease,color .18s ease,border-color .18s ease,box-shadow .18s ease}
#mood span[data-d="1"],#mood span[data-d="2"],#mood span[data-d="3"],#mood span[data-d="4"]{color:var(--key-cool-ink)}
#mood span[data-d="6"],#mood span[data-d="7"],#mood span[data-d="8"],#mood span[data-d="9"]{color:var(--key-warm-ink)}
#mood span[data-d="5"]{margin:0 5px}#mood span:hover{border-color:var(--mute)}
#mood span.on{background:var(--key-on);color:var(--key-on-ink);border-color:transparent;box-shadow:0 1px 3px rgba(0,0,0,.2)}
#mood span.on[data-d="6"],#mood span.on[data-d="7"],#mood span.on[data-d="8"],#mood span.on[data-d="9"]{background:var(--warm);color:var(--warm-ink);box-shadow:0 0 0 var(--hw,4px) var(--warm-halo)}
#mood span.on[data-d="1"],#mood span.on[data-d="2"],#mood span.on[data-d="3"],#mood span.on[data-d="4"]{background:var(--cool);color:var(--cool-ink);box-shadow:0 0 0 var(--hw,4px) var(--cool-halo)}
#pg{max-width:840px;margin:0 auto;padding:110px 16px 64px}#pg::after{content:"";display:block;clear:both}
.b{margin:14px 0 0;max-width:80%;clear:both}
.b .lab{display:block;font:600 11px/13px var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--mute);margin:0 0 5px 15px}
.b .t{display:inline-block;padding:10px 15px;border-radius:21px;white-space:pre-wrap;word-break:break-word;text-align:left;font-size:18px;line-height:1.36;letter-spacing:-.011em}
.w{float:left}.o{float:right;text-align:right}.o .lab{margin:0 15px 5px 0}
.w.you .t{background:var(--you);color:var(--you-ink);box-shadow:inset 0 0 0 1px var(--you-line),var(--bub-shadow)}
.w.parent .t{background:var(--parent);color:var(--parent-ink);box-shadow:inset 0 0 0 1px var(--parent-line),var(--parent-shadow)}
.w.other .t{background:var(--other);color:var(--other-ink);box-shadow:inset 0 0 0 1px var(--other-line)}
.o .t{background:var(--body);color:var(--body-ink);font-weight:500;box-shadow:var(--body-shadow)}.gap{display:inline-block;width:.4em}
.b.nw{animation:bi .34s cubic-bezier(.2,.7,.3,1) backwards}@keyframes bi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.l{animation:lf .36s ease-out backwards}@keyframes lf{from{opacity:0}to{opacity:1}}
@media (prefers-reduced-motion:reduce){.b.nw,.l,#st{animation:none}}
#k{position:fixed;left:0;top:0;width:1px;height:1px;opacity:0;border:0;padding:0;font-size:16px}
@media (max-width:640px){#topin{flex-wrap:wrap;gap:10px 12px;padding:10px 14px 12px}.gw{flex:1 1 0;height:58px}
 #mood{flex:1 1 100%;gap:4px}#mood span{flex:1;width:auto;height:44px;font-size:15px;border-radius:11px}#mood span[data-d="5"]{margin:0 3px}#mood{--hw:2px}
 #pg{padding:150px 14px 48px}.b{max-width:86%}.b .t{font-size:17px;padding:9px 14px;border-radius:20px}}</style>
<div id=top><div id=topin><span id=st title="awake"></span><div class=gw><canvas id=cv width=760 height=48></canvas><div class=lg aria-hidden=true><span class=f><i></i>face shown</span><span class=s><i></i>its face</span></div></div><div id=mood></div></div></div>
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
/* THE STRIP: the face shown to it and its own face, the last 900 ticks, drawn at the screen's own pixel ratio (display only).
   The scale is the keys' own, -4 to +4 (its face clipped there); the face shown is a filled step, the wide line, so a smile of a few
   ticks still shows as a block; its face is the thin line drawn over it, so where the two agree both are seen */
const F=[];const cv=document.getElementById('cv'),cx=cv.getContext('2d');
function draw(){const N=300,dpr=window.devicePixelRatio||1,r=cv.getBoundingClientRect(),W=Math.max(1,Math.round(r.width*dpr)),H=Math.max(1,Math.round(r.height*dpr));if(cv.width!==W||cv.height!==H){cv.width=W;cv.height=H}
 const cs=getComputedStyle(document.documentElement),col=n=>cs.getPropertyValue(n).trim(),L=4.3,top=21*dpr,bot=H-7*dpr,rp=8*dpr,dx=(W-rp)/N,
  y=v=>top+(L-Math.max(-L,Math.min(L,v)))*(bot-top)/(2*L),X=j=>(j+N-F.length)*dx;
 cx.clearRect(0,0,W,H);const hl=Math.max(1,Math.round(dpr)),o=hl%2?.5:0;cx.lineWidth=hl;cx.strokeStyle=col('--grid');cx.beginPath();
 for(let q=1;q<6;q++){const x=Math.round(q*(W-rp)/6)+o;cx.moveTo(x,top);cx.lineTo(x,bot)}for(const v of[-4,4]){const yy=Math.round(y(v))+o;cx.moveTo(0,yy);cx.lineTo(W,yy)}cx.stroke();
 const y0=Math.round(y(0))+o;cx.strokeStyle=col('--zero');cx.beginPath();cx.moveTo(0,y0);cx.lineTo(W,y0);cx.stroke();if(!F.length)return;
 cx.lineJoin='miter';cx.lineCap='butt';const xl=X(F.length-1);
 const step=()=>{cx.beginPath();cx.moveTo(X(0),y(F[0][0]));for(let j=1;j<F.length;j++){const xm=X(j)-dx/2;cx.lineTo(xm,y(F[j-1][0]));cx.lineTo(xm,y(F[j][0]))}cx.lineTo(xl,y(F[F.length-1][0]))};
 step();cx.strokeStyle=col('--zero');cx.lineWidth=2.6*dpr;cx.stroke();const band=(y1,y2,c)=>{cx.save();cx.beginPath();cx.rect(0,y1,W,y2-y1);cx.clip();step();cx.lineTo(xl,y(0));cx.lineTo(X(0),y(0));cx.closePath();cx.globalAlpha=.28;cx.fillStyle=c;cx.fill();cx.globalAlpha=1;step();cx.strokeStyle=c;cx.lineWidth=2.6*dpr;cx.stroke();cx.restore()};band(0,y0-hl,col('--warm'));band(y0+hl,H,col('--cool'));cx.lineJoin='round';cx.lineCap='round';
 cx.beginPath();F.forEach((f,j)=>{if(j===0)cx.moveTo(X(j),y(f[1]));else cx.lineTo(X(j),y(f[1]))});cx.strokeStyle=col('--its');cx.lineWidth=1.3*dpr;cx.stroke();
 const f=F[F.length-1],xe=X(F.length-1);cx.beginPath();cx.arc(xe,y(f[0]),3.4*dpr,0,7);cx.fillStyle=f[0]>0?col('--warm'):f[0]<0?col('--cool'):col('--mute');cx.fill();cx.beginPath();cx.arc(xe,y(f[1]),2*dpr,0,7);cx.fillStyle=col('--its');cx.fill()}
const topEl=document.getElementById('top');function fitTop(){pg.style.paddingTop=(topEl.offsetHeight+18)+'px'}
try{if(window.ResizeObserver)new ResizeObserver(()=>{fitTop();draw()}).observe(topEl);fitTop();window.addEventListener('resize',()=>{fitTop();draw()});
 const mq=window.matchMedia&&matchMedia('(prefers-color-scheme: dark)');if(mq){if(mq.addEventListener)mq.addEventListener('change',draw);else if(mq.addListener)mq.addListener(draw)}draw()}catch(err){}
/* a letter that arrives while the page is open fades in, each at its own tick: the ticks of one poll spread over the time since the
   last, so the letters flow as they were made; the history drawn on opening does not fade (display only) */
let live=false,tq=-1,tT=1,tS=0,tAt=0;function pace(es){const now=Date.now();tS=tAt?Math.min(800,now-tAt):0;tAt=now;tT=0;for(const e of es)if(e[1]===0)tT++;tT=Math.max(1,tT);tq=-1}
function lag(e){const d=Math.round(tS*Math.max(0,tq)/tT);if(d>0)e.style.animationDelay=d+'ms'}
function glyph(s){if(!live)return document.createTextNode(s);const e=document.createElement('span');e.className='l';e.textContent=s;lag(e);return e}
function block(kind,who){const b=document.createElement('div');b.className='b '+(kind==='w'?'w '+who:'o')+(live?' nw':'');if(live)lag(b);const l=document.createElement('span');l.className='lab';l.textContent=kind==='w'?(who==='parent'?'parent':who==='other'?'other voice':'you'):'body';const t=document.createElement('span');t.className='t';b.appendChild(l);b.appendChild(t);pg.appendChild(b);while(pg.children.length>90){const f=pg.firstChild;if(wOpen&&f.contains(wOpen))wOpen=null;if(oOpen&&f.contains(oOpen))oOpen=null;pg.removeChild(f)}return t}
function feed(es){for(const e of es){const sym=e[0];if(e[1]===0){tq++;F.push([e[2]||0,e[3]||0]);if(F.length>900)F.shift();if(sym){const who=e[4]||'you';if(!wOpen||who!==wWho){wOpen=block('w',who);wWho=who}wOpen.appendChild(glyph(sym));wGap=0}else{wGap++;if(wOpen&&wGap>=GAP)wOpen=null}}
 else{if(sym){if(!oOpen&&sym===' '){oGap=0;continue}if(!oOpen){oOpen=block('o');oSil=0}else if(oSil>=3){const g=document.createElement('span');g.className='gap';oOpen.appendChild(g)}oOpen.appendChild(glyph(sym));oGap=0;oSil=0}else{oGap++;oSil++;if(oOpen&&oGap>=GAP)oOpen=null}}}}
async function poll(){if(busy)return;busy=true;let d;try{d=await fetch('/state?since='+seen).then(r=>r.json())}catch(err){d=null}finally{busy=false}
 if(!d||!d.page){st.className='off';st.title='no answer from the body';return}
 if(d.n<seen||(d.base!==undefined&&d.base>seen&&seen>0)){seen=0;pg.textContent='';wOpen=oOpen=null}
 const es=(seen===0)?d.page.slice(-4000):d.page;live=seen>0;pace(es);const atBottom=window.innerHeight+window.scrollY>=document.body.scrollHeight-80;feed(es);draw();seen=d.n;
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
