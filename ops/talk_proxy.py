"""THE TALK PAGE BESIDE THE BODY (2026-09-17): serves the visitor's page (body/serve.py's TALK, the same page the body serves at /talk)
on its own port and forwards its calls (/state, /type, /face) to the served body. THE PARENT PAUSES WHILE THE PAGE IS OPEN (the
user's word, 2026-09-17): while a browser is polling this server the typist is stopped (SIGSTOP, as tools/rehearse.py stops it),
and when no browser has polled for pause_after seconds it is resumed (SIGCONT); it is resumed on any exit of this server. The body's
own /talk page does not pause the parent; this server's does.
usage: python3 ops/talk_proxy.py [--port 8021] [--body 8020] [--pause-parent 1] [--pause-after 6]"""
import sys, os, json, time, signal, atexit, threading, subprocess, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, __file__.rsplit("/ops/", 1)[0])
from body.serve import TALK

def arg(name, default):
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--" + name and i + 1 < len(sys.argv): return type(default)(sys.argv[i + 1])
    return default
PORT, BODY = arg("port", 8021), arg("body", 8020)
PAUSE, PAUSE_AFTER = arg("pause_parent", 1) if "--pause_parent" in sys.argv else arg("pause-parent", 1), float(arg("pause-after", 6.0))
BASE = f"http://localhost:{BODY}"

# --- the parent's pause: the typist stopped while a browser polls, resumed after the last poll ---
_last_poll = 0.0; _paused = []; _lock = threading.Lock()
def _typist_pids():
    out = subprocess.run(["pgrep", "-f", f"body.teacher --port {BODY}"], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.isdigit() and int(x) != os.getpid()]
def _stop_typist():
    with _lock:
        if _paused: return
        for pid in _typist_pids():
            try: os.kill(pid, signal.SIGSTOP); _paused.append(pid)
            except ProcessLookupError: pass
        if _paused: print(f"[talk] a visitor is here: the typist paused ({', '.join(map(str, _paused))})", flush=True)
def _resume_typist():
    with _lock:
        if not _paused: return
        for pid in list(_paused):
            try: os.kill(pid, signal.SIGCONT)
            except ProcessLookupError: pass
        print(f"[talk] the visitor left: the typist resumed ({', '.join(map(str, _paused))})", flush=True); _paused.clear()
def _watch():
    while True:
        time.sleep(1.0)
        if _paused and time.time() - _last_poll > PAUSE_AFTER: _resume_typist()
def _note_poll():
    global _last_poll
    _last_poll = time.time()
    if PAUSE: _stop_typist()
atexit.register(_resume_typist)
for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, lambda *_: (_resume_typist(), sys.exit(0)))
threading.Thread(target=_watch, daemon=True).start()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, ctype, b):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/state"):
            _note_poll()
            try:
                with urllib.request.urlopen(BASE + self.path, timeout=10) as r: self._send(200, "application/json", r.read())
            except Exception as e:
                print(f"[talk] {self.path[:40]} -> {type(e).__name__}: {str(e)[:120]}", flush=True)
                self._send(502, "application/json", json.dumps({"error": str(e)[:200]}).encode())
        else:
            self._send(200, "text/html; charset=utf-8", TALK.encode())
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0") or 0); body = self.rfile.read(n) if n else b"{}"
        if self.path in ("/type", "/face"):
            try:
                req = urllib.request.Request(BASE + self.path, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r: self._send(200, "application/json", r.read())
            except Exception as e:
                print(f"[talk] {self.path[:40]} -> {type(e).__name__}: {str(e)[:120]}", flush=True)
                self._send(502, "application/json", json.dumps({"error": str(e)[:200]}).encode())
        else:
            self._send(404, "application/json", b'{"error": "unknown path"}')

print(f"[talk] the page is open on http://localhost:{PORT} (the body on {BODY}); the parent {'pauses while the page is open' if PAUSE else 'keeps typing'}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
