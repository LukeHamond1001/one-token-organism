"""THE LANGUAGE MODEL INSIDE (a supervisor's instrument, 2026-09-12): load a saved body on a copy, feed a prompt as the world's
symbols and the world's pause, then read what the MOUTH would say, greedily, with its own symbols fed back: the cortex's forecast
plus the hippocampal recall (the efference copy in the recall's query), as the tick reads them. Two columns: the mouth, and the
cortex alone (the hearing model, which is not the speaker).
usage: python3 tools/probe_lm.py data/watch2.pt "do you want milk?" ...   [--n=24]"""
import sys
sys.path.insert(0, "/Users/lukehamond/Projects/project")
import torch
from tokenizers import Tokenizer
from body.life import Life

args = [a for a in sys.argv[1:] if not a.startswith("--")]
n = 24
for a in sys.argv[1:]:
    if a.startswith("--n="):
        n = int(a[4:])
path, prompts = args[0], args[1:] or ["do you want milk?", "what do you have?", "are you here?", "are you sad?", "what do you want?", "hi", "I want ", "I see "]
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu"); m = life.m; m.eval()
print(f"body: nights {life.nights}, own_gain {m.own_gain}, store {life.store.n()} (own {int((life.store.W == 1).sum())})")
zero = torch.zeros(m.d)
for prompt in prompts:
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.n_own = 0
    with torch.no_grad():
        for ch in prompt:
            i = TOK.token_to_id(ch)
            life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.bag_w = life.cfg["bag_decay"] * m.shift(life.bag_w) + m.E.weight[i]
        for _ in range(int(life.cfg.get("offset_ticks", 8))):
            life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.bag_w = life.cfg["bag_decay"] * life.bag_w
        mouth, cortex = [], []
        for step in range(n):
            xs, whos, faces, bundles, reads = life._window_tensors(list(life.win)[-m.window:])
            C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
            rd, conf, _ = life.store.read(life.bag)
            lc = m.readout(m.forecast(C, zero)); lc[life.bans] = float("-inf"); lc[life.sil] = float("-inf")
            lm = m.readout(m.forecast(C, rd)); lm[life.bans] = float("-inf"); lm[life.sil] = float("-inf")
            sym = int(lm.argmax()); cortex.append(TOK.decode([int(lc.argmax())])); mouth.append(TOK.decode([sym]))
            life.win.append({"x": life.sil, "xo": sym, "face": torch.zeros(2), "bundle": life.bands, "read": rd, "r": 0.0})
            life.bag_o = life.cfg["bag_decay"] * m.shift(life.bag_o) + m.E.weight[sym]; life.n_own += 1
    print(f"{prompt!r:22} mouth: {''.join(mouth)!r:28} cortex alone: {''.join(cortex)!r}")

# THE LANGUAGE MODEL'S ACCURACY (2026-09-12, 23:55): the night's own gauge (teacher-forced argmax share, the cortex alone, the
# ladder run along each line from rest) on the parent's last sixty lines as heard: the number a language model is judged by,
# on material the night may not have replayed. Run with --lmloss.
if "--lmloss" in sys.argv:
    import json
    R = [json.loads(l) for l in open("/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl") if l.strip()]
    def ids_of(t): return [TOK.token_to_id(ch) for ch in t if TOK.token_to_id(ch) is not None]
    recent = [ids_of(r["text"].strip()) for r in R if r["action"] in ("line", "cue") and r.get("voice") != "b"][-60:]
    dreams = [ids_of(t) for t in ((life.last_night or {}).get("examples") or []) if len(t) >= 6]
    with torch.no_grad():
        g2 = life.gauge(recent); g1 = life.gauge(dreams) if dreams else (float("nan"), 0)
    print(f"LM accuracy (the night's gauge): the parent's last 60 lines {g2[0]:.2f} over {g2[1]} symbols | the last night's dreams {g1[0]:.2f} over {g1[1]}")
