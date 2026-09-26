"""THE /sim PAGE (docs/SIM_DESIGN.md 11's W6; the lead's first build at S5a): watch the G1 live, on http://127.0.0.1:8030/ (never
8020, 8021 or 9333). The runner (tools/sim_life.py --page) takes a snapshot each tick in its own thread, between ticks, and this
module's server hands the latest snapshot out from another; the page never touches the physics, the body or the parent, and
nothing it does changes a life (the room camera renders from a copy of the tick's state in its own context).

What it shows: the room from a corner camera; the child's three views (its two grey eyes with their fovea windows, its colour
camera), as the renders come, not as its code reads them; her face line and the child's stress, its pain and cry, the born reading
of her face; the ten gates (each effector's probability of acting this tick); a two-voice transcript (her lines, and what the
child's token output said and what her ear heard); her episode; the tick, the day, day or night; the room's sounds and the visual
onsets. An instrument: the experimenter's view, never the body's.
"""
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np

PORT = 8030
ROOM_W, ROOM_H = 640, 400
KEEP_LINES = 60

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>G1 · live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f5f2;--card:#ffffff;--ink:#1d1d1f;--mute:#6e6e73;--line:#e3e1dc;--warm:#d97a2b;--cool:#2f6fdb;--pain:#c43d3d}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif}
header{display:flex;gap:18px;align-items:baseline;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--card)}
header h1{font-size:17px;margin:0;font-weight:600}header span{color:var(--mute)}
main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:16px;padding:16px 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:0 0 8px}
img{width:100%;display:block;border-radius:6px;background:#111}
.meters{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.m b{display:block;font-size:20px;font-weight:600}.m span{color:var(--mute);font-size:12px}
.gates{display:grid;grid-template-columns:repeat(10,1fr);gap:6px;align-items:end;height:90px}
.g{display:flex;flex-direction:column;justify-content:flex-end;height:100%}
.g i{display:block;background:var(--cool);border-radius:3px 3px 0 0;min-height:2px}.g small{font-size:10px;color:var(--mute);text-align:center}
#tx{height:300px;overflow:auto;font-size:13px}.her{color:var(--warm)}.it{color:var(--cool)}.ear{color:var(--mute)}
.tx div{padding:2px 0;border-bottom:1px dashed #efede8}
@media (max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>G1 · living</h1><span id="clock">…</span><span id="ep"></span></header>
<main>
<section style="display:grid;gap:16px">
 <div class="card"><h2>The room</h2><img id="room" alt="the room"></div>
 <div class="card"><h2>What it sees (two grey eyes, fovea windows; the colour camera)</h2><img id="eyes" alt="its views"></div>
</section>
<section style="display:grid;gap:16px;align-content:start">
 <div class="card"><h2>Its body</h2><div class="meters">
  <div class="m"><b id="h">–</b><span>stress · mood</span></div><div class="m"><b id="face">–</b><span>her face, as it reads it</span></div>
  <div class="m"><b id="pain">–</b><span>pain · cry</span></div><div class="m"><b id="onset">–</b><span>onsets · sounds</span></div></div></div>
 <div class="card"><h2>The ten gates</h2><div class="gates" id="gates"></div></div>
 <div class="card"><h2>Two voices</h2><div id="tx" class="tx"></div></div>
</section></main>
<script>
const $=id=>document.getElementById(id);let n=0;
async function tick(){try{const s=await (await fetch('state.json?'+Date.now())).json();
$('clock').textContent=`tick ${s.tick} · day ${s.day} · ${s.night?'night':'day'}`;$('ep').textContent=s.episode?('her episode: '+s.episode):'';
$('h').textContent=`${s.stress.toFixed(2)} · ${s.mood.toFixed(2)}`;$('face').textContent=s.reading.toFixed(2);$('pain').textContent=`${s.pain?'●':'○'} · ${s.cry?'crying':'quiet'}`;
$('onset').textContent=`${s.onset?'●':'○'} · ${s.sounds}`;
$('gates').innerHTML=s.gates.map(g=>`<div class="g"><i style="height:${Math.round(100*g[1])}%"></i><small>${g[0]}</small></div>`).join('');
$('tx').innerHTML=s.transcript.map(x=>`<div class="${x[1]}">${x[0]} · ${x[2]}</div>`).join('');$('tx').scrollTop=1e9;
n++;$('room').src='room.jpg?'+n;$('eyes').src='eyes.jpg?'+n;}catch(e){}setTimeout(tick,500)}
tick();
</script></body></html>"""


class Snapshot:
    """the runner's view of the life, taken between ticks in the runner's thread"""

    def __init__(self, world, lane, life, eyes=None, room_every=3):
        self.w, self.lane, self.L, self.eyes = world, lane, life, eyes
        self.lock = threading.Lock()
        self.state = {}
        self.room_jpg = self.eyes_jpg = b""
        self.transcript = []
        self.room_every = int(room_every)
        self.renderer = mujoco.Renderer(world.m, ROOM_H, ROOM_W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.0, -0.6, 0.2]; self.cam.distance = 2.6; self.cam.azimuth = 135.0; self.cam.elevation = -30.0
        self.data = mujoco.MjData(world.m)

    def take(self, inv):
        w, lane, L = self.w, self.lane, self.L
        t = int(w.tick)
        ls = lane.last if not w.night else {}
        if ls.get("line"):
            self.transcript.append((t, "her", ls["line"]))
        tok = None if w.words_out is None else inv.get(int(w.words_out))
        if tok and not w.night:
            self.transcript.append((t, "it", f"(its token) {tok}"))
        for h in (ls.get("heard") or []):
            if h:
                self.transcript.append((t, "ear", f"(her ear heard) {h}"))
        self.transcript = self.transcript[-KEEP_LINES:]
        names = [e.name for e in L.anatomy.motors]
        gates = [(n[:6], float(st.get("now", {}).get("p_act", 0.0))) for n, st in zip(names, L.motor)]
        gates.append(("words", float(getattr(L, "_last_choice", {}).get("p_act", 0.0))))
        f = getattr(w, "now", None)
        pain = f is not None and bool(np.any(f.obs.get("pain", 0)))
        st = dict(tick=t, day=int(lane.day), night=bool(w.night), stress=float(L.stress), mood=float(L.mood), reading=float(lane.reading), pain=pain,
                  cry=bool(w.crying), onset=bool(f is not None and f.obs.get("onset_periph", [0])[0]),
                  sounds=len(getattr(w.sounds, "last_events", [])), episode=None if lane.plan is None else lane.plan.kind,
                  gates=gates[:10], transcript=[list(x) for x in self.transcript])
        room = self.room_jpg
        if t % self.room_every == 0:
            self.data.qpos[:] = w.d.qpos; self.data.mocap_pos[:] = w.d.mocap_pos; self.data.mocap_quat[:] = w.d.mocap_quat
            mujoco.mj_forward(w.m, self.data)
            self.renderer.update_scene(self.data, camera=self.cam)
            room = _jpg(self.renderer.render())
        eyes = self.eyes_jpg
        if self.eyes is not None and self.eyes._cache is not None:
            tr = self.eyes._cache[-1]
            eyes = _jpg(_views(tr))
        with self.lock:
            self.state, self.room_jpg, self.eyes_jpg = st, room, eyes


def _views(tr):
    """the three views side by side, each grey eye's fovea window drawn as a box"""
    imgs = tr["images"]
    L_, R_ = imgs["L"].copy(), imgs["R"].copy()
    for im, sd in ((L_, "L"), (R_, "R")):
        x0, y0 = tr["windows"][sd]
        n = 64
        im[y0, x0:x0 + n] = im[y0 + n - 1, x0:x0 + n] = (255, 200, 0)
        im[y0:y0 + n, x0] = im[y0:y0 + n, x0 + n - 1] = (255, 200, 0)
    C = np.zeros((L_.shape[0], imgs["C"].shape[1], 3), np.uint8)
    C[:imgs["C"].shape[0]] = imgs["C"]
    return np.concatenate([L_, R_, C], axis=1)


def _jpg(arr):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="JPEG", quality=82)
    return b.getvalue()


def serve(snap, port=PORT):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            with snap.lock:
                if path in ("/", "/sim"):
                    body, ct = PAGE.encode(), "text/html; charset=utf-8"
                elif path.endswith("state.json"):
                    body, ct = json.dumps(snap.state).encode(), "application/json"
                elif path.endswith("room.jpg"):
                    body, ct = snap.room_jpg, "image/jpeg"
                elif path.endswith("eyes.jpg"):
                    body, ct = snap.eyes_jpg, "image/jpeg"
                else:
                    self.send_response(404); self.end_headers(); return
            self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)
    httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), H)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd
