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

# THE LANGUAGE MODEL'S LOSS (2026-09-12, 23:45): the cortex's mean next-symbol loss, teacher-forced, on the parent's last lines
# as heard (the world's symbols with a pause after each): the number a language model is judged by, on material the night may
# not have replayed. Run with --lmloss; the lines come from the page log.
if "--lmloss" in sys.argv:
    import json, math
    R = [json.loads(l) for l in open("/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl") if l.strip()]
    lines = [r["text"].strip() for r in R if r["action"] in ("line", "cue") and r.get("voice") != "b"][-60:]
    tot, cnt, hits = 0.0, 0, 0
    with torch.no_grad():
        for line in lines:
            life.win.clear()
            ids = [TOK.token_to_id(ch) for ch in line]
            for k, i in enumerate(ids):
                if k >= 2:
                    xs, whos, faces, bundles, reads = life._window_tensors(list(life.win)[-m.window:])
                    C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
                    lg = m.readout(m.forecast(C, zero)); p = torch.log_softmax(lg, -1)
                    tot += -float(p[i]); cnt += 1; hits = hits + (1 if int(lg.argmax()) == i else 0)
                life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
    print(f"LM on the parent's last {len(lines)} lines, teacher-forced, cortex alone: next-symbol argmax accuracy {hits/max(1,cnt):.2f}; loss at the mouth's sharpness {tot/max(1,cnt):.2f} nats")
