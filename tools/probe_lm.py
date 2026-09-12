"""THE LANGUAGE MODEL INSIDE (a supervisor's instrument, 2026-09-12): load a saved body on a copy, feed a prompt as the world's
symbols (then the world's pause), and read the mouth's greedy continuation, own symbols fed back as its own sound. What the
cortex would say next, without the gate, the sampling or the day's noise: whether the exchanges are forming an answerer.
usage: python3 tools/probe_lm.py data/watch2.pt "do you want milk?" "what do you have?" ...  [--n 30]"""
import sys
sys.path.insert(0, "/Users/lukehamond/Projects/project")
import torch
from tokenizers import Tokenizer
from body.life import Life

args = [a for a in sys.argv[1:] if not a.startswith("--")]
n = 30
for a in sys.argv[1:]:
    if a.startswith("--n="):
        n = int(a[4:])
path, prompts = args[0], args[1:] or ["do you want milk?", "what do you have?", "are you here?", "where is the dog?", "hi", "give me the ball"]
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu")
m = life.m; life.m.eval()
print(f"body: nights {life.nights}, own_gain {m.own_gain}, store {life.store.n()}")
for prompt in prompts:
    # the prompt as the world's symbols, then the world's pause (offset_ticks of quiet), read from the same window the mouth reads
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_()
    with torch.no_grad():
        for ch in prompt:
            i = TOK.token_to_id(ch)
            life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": torch.zeros(m.d), "r": 0.0})
        for _ in range(int(life.cfg.get("offset_ticks", 8))):
            life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": torch.zeros(m.d), "r": 0.0})
        out = []
        for _ in range(n):
            win = list(life.win)[-m.window:]
            xs, whos, faces, bundles, reads = life._window_tensors(win)
            C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
            lg = m.readout(m.forecast(C, torch.zeros(m.d))); lg[life.bans] = float("-inf")
            if "--speak" in sys.argv:
                lg[life.sil] = float("-inf")                   # as when the gate has decided to speak: the rest is not a choice
            sym = int(lg.argmax())
            if "--speak" in sys.argv and out and len(out) >= 3 and out[-1] == out[-2] == out[-3]:
                break
            if sym == life.sil:
                out.append("·")
                life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": torch.zeros(m.d), "r": 0.0})
                if len(out) > 3 and all(o == "·" for o in out[-4:]):
                    break
                continue
            out.append(TOK.decode([sym]))
            life.win.append({"x": life.sil, "xo": sym, "face": torch.zeros(2), "bundle": life.bands, "read": torch.zeros(m.d), "r": 0.0})
    print(f"{prompt!r:28} -> {''.join(out)!r}")
