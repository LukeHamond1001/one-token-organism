"""THE TALK PAGE BESIDE THE BODY (2026-09-17): serves the visitor's page (body/serve.py's TALK, the same page the body serves at /talk)
on its own port and forwards its calls (/state, /type, /face) to the served body, so a redrawn page can be used at once without
restarting the body; after the body's next reload /talk on the body's own port serves the same page.
usage: python3 ops/talk_proxy.py [--port 8021] [--body 8020]"""
import sys, json, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, __file__.rsplit("/ops/", 1)[0])
from body.serve import TALK

def arg(name, default):
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--" + name and i + 1 < len(sys.argv): return type(default)(sys.argv[i + 1])
    return default
PORT, BODY = arg("port", 8021), arg("body", 8020)
BASE = f"http://localhost:{BODY}"

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, ctype, b):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/state"):
            try:
                with urllib.request.urlopen(BASE + self.path, timeout=10) as r: self._send(200, "application/json", r.read())
            except Exception as e: self._send(502, "application/json", json.dumps({"error": str(e)[:200]}).encode())
        else:
            self._send(200, "text/html; charset=utf-8", TALK.encode())
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0") or 0); body = self.rfile.read(n) if n else b"{}"
        if self.path in ("/type", "/face"):
            try:
                req = urllib.request.Request(BASE + self.path, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r: self._send(200, "application/json", r.read())
            except Exception as e: self._send(502, "application/json", json.dumps({"error": str(e)[:200]}).encode())
        else:
            self._send(404, "application/json", b'{"error": "unknown path"}')

print(f"[talk] the page is open on http://localhost:{PORT} (the body on {BODY})", flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
