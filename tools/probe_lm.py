"""THE LANGUAGE MODEL INSIDE (a supervisor's instrument, 2026-09-12): load a saved body on a copy, feed a prompt as the world's
symbols and the world's pause, then read what the MOUTH would say, greedily, with its own symbols fed back: the cortex's forecast
plus the hippocampal recall (the efference copy in the recall's query), as the tick reads them. Two columns: the mouth, and the
cortex alone (the hearing model, which is not the speaker).
usage: python3 tools/probe_lm.py data/watch2.pt "do you want milk?" ...   [--n=24]"""
import sys, json
sys.path.insert(0, "/Users/lukehamond/Projects/project")
import torch
from tokenizers import Tokenizer
from body.life import Life

args = [a for a in sys.argv[1:] if not a.startswith("--")]
n = 24; follow = 0.0
for a in sys.argv[1:]:
    if a.startswith("--n="):
        n = int(a[4:])
    if a.startswith("--follow="):
        follow = float(a[9:])                                   # the waking recall carries the episode (read_follow) at this gain
path, prompts = args[0], args[1:] or ["do you want milk?", "what do you have?", "are you here?", "are you sad?", "what do you want?", "hi", "I want ", "I see "]
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu", cfg=({"read_follow": follow} if follow > 1.0 else None)); m = life.m; m.eval()
_R = [json.loads(l) for l in open("/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl") if l.strip()] if True else []
_typed = " | ".join(r["text"].strip() for r in _R if r["action"] in ("line", "cue"))
def lived_share(text):
    """the longest prefix of the mouth's text that is a substring of a line ever typed, as a share of its length"""
    t = text.strip()
    for L in range(len(t), 2, -1):
        if t[:L] in _typed:
            return L / max(1, len(t))
    return 0.0
print(f"body: nights {life.nights}, own_gain {m.own_gain}, store {life.store.n()} (own {int((life.store.W == 1).sum())})")
zero = torch.zeros(m.d)
for prompt in prompts:
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.n_own = 0; life._follow = None
    with torch.no_grad():
        for ch in prompt:
            i = TOK.token_to_id(ch)
            life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.bag_w = life.cfg["bag_decay"] * m.shift(life.bag_w) + m.E.weight[i]
        for _ in range(int(life.cfg.get("offset_ticks", 8))):
            life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.bag_w = life.cfg["bag_decay"] * life.bag_w
        mouth, cortex = [], []; own_win = 0; n_win = 0
        for step in range(n):
            xs, whos, faces, bundles, reads = life._window_tensors(list(life.win)[-m.window:])
            C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
            rd, conf, win_ = life._recall(life.bag)
            if win_ >= 0:
                n_win += 1; own_win += int(life.store.W[win_] == 1)     # the recall's winner: its own song, or the world's
            lc = m.readout(m.forecast(C, zero)); lc[life.bans] = float("-inf"); lc[life.sil] = float("-inf")
            lm = m.readout(m.forecast(C, rd)); lm[life.bans] = float("-inf"); lm[life.sil] = float("-inf")
            sym = int(lm.argmax()); cortex.append(TOK.decode([int(lc.argmax())])); mouth.append(TOK.decode([sym]))
            life.win.append({"x": life.sil, "xo": sym, "face": torch.zeros(2), "bundle": life.bands, "read": rd, "r": 0.0})
            life.bag_o = life.cfg["bag_decay"] * m.shift(life.bag_o) + m.E.weight[sym]; life.n_own += 1
    print(f"{prompt!r:22} mouth: {''.join(mouth)!r:28} cortex alone: {''.join(cortex)!r}   lived prefix {lived_share(''.join(mouth)):.2f}   own winners {own_win}/{n_win}")

# THE LANGUAGE MODEL'S ACCURACY (2026-09-12, 23:55): the night's own gauge (teacher-forced argmax share, the cortex alone, the
# ladder run along each line from rest) on the parent's last sixty lines as heard: the number a language model is judged by,
# on material the night may not have replayed. Run with --lmloss.
if "--lmloss" in sys.argv:
    import json
    R = [json.loads(l) for l in open("/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl") if l.strip()]
    def ids_of(t): return [TOK.token_to_id(ch) for ch in t if TOK.token_to_id(ch) is not None]
    recent = [ids_of(r["text"].strip()) for r in R if r["action"] in ("line", "cue") and r.get("voice") != "b"][-60:]
    dreams = [ids_of(t) for t in ((life.last_night or {}).get("examples") or []) if len(t) >= 6]
    # THE OLD LINES (2026-09-13): the parent's distinct lines from days 110-125, long faded from the store (it holds about five hundred
    # utterances; the corpus near four thousand): material no night can have replayed, the language-model reading proper
    old = [ids_of(t) for t in dict.fromkeys(r["text"].strip() for r in R if r["action"] in ("line", "cue") and r.get("voice") != "b" and 110 <= int(r.get("day", -1)) <= 125)][:150]
    old = [x for x in old if len(x) >= 2]
    # THE HELD-OUT SET (2026-09-13): lines in the current stage's style and vocabulary, written by the supervisor and never typed
    # (any that a parent later happens to type are dropped at probe time): the generalisation reading proper
    typed = set(r["text"].strip() for r in R if r["action"] in ("line", "cue"))
    held = [ids_of(t.strip()) for t in open("/Users/lukehamond/Projects/project/tools/heldout_stage4.txt") if t.strip() and t.strip() not in typed]
    life.cfg["night_batch"] = max(1, int(life.cfg.get("night_batch", 0)))    # the gauge on the lockstep path (exact; fast)
    with torch.no_grad():
        g2 = life.gauge(recent); c2 = life._gauge_cos; g3 = life.gauge(old); c3 = life._gauge_cos
        g4 = life.gauge(held); c4 = life._gauge_cos; g1 = life.gauge(dreams) if dreams else (float("nan"), 0)
    print(f"LM accuracy (the night's gauge): the parent's last 60 lines {g2[0]:.3f} (cos {c2}) over {g2[1]} symbols | {len(old)} old lines (days 110-125) {g3[0]:.3f} (cos {c3}) over {g3[1]} | HELD-OUT {len(held)} lines {g4[0]:.3f} (cos {c4}) over {g4[1]} | the last night's dreams {g1[0]:.2f} over {g1[1]}")
    # THE FACTS (stage five, 2026-09-14): each fact of tools/facts_stage5.txt as a prefix the parents never type (tools/heldout_facts.txt);
    # a fact counts as learned when the mouth completes the prefix with the fact's own remainder; and the cortex alone on the fact
    # sentences, teacher-forced, as the finer number
    import os
    fp, hp = "/Users/lukehamond/Projects/project/tools/facts_stage5.txt", "/Users/lukehamond/Projects/project/tools/heldout_facts.txt"
    if os.path.exists(fp) and os.path.exists(hp):
        facts = [l.split("|")[1].strip() for l in open(fp) if "|" in l]
        prefixes = [l.rstrip("\n") for l in open(hp) if l.strip()]
        learned = 0; shown = []
        for pre in prefixes:
            rems = [f[len(pre):].strip() for f in facts if f.startswith(pre.strip() + " ") or f.startswith(pre)]
            rems = [r for r in rems if r]
            life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.n_own = 0; life._follow = None
            got = []
            with torch.no_grad():
                for ch in pre:
                    i = TOK.token_to_id(ch); life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
                    life.bag_w = life.cfg["bag_decay"] * m.shift(life.bag_w) + m.E.weight[i]
                for _ in range(12):
                    xs, whos, faces, bundles, reads = life._window_tensors(list(life.win)[-m.window:])
                    C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
                    rd, conf, _w = life._recall(life.bag)
                    lm = m.readout(m.forecast(C, rd)); lm[life.bans] = float("-inf"); lm[life.sil] = float("-inf")
                    sym = int(lm.argmax()); got.append(TOK.decode([sym]))
                    life.win.append({"x": life.sil, "xo": sym, "face": torch.zeros(2), "bundle": life.bands, "read": rd, "r": 0.0})
                    life.bag_o = life.cfg["bag_decay"] * m.shift(life.bag_o) + m.E.weight[sym]; life.n_own += 1
            text = "".join(got); ok = any(text.startswith(r[:max(3, len(r.split()[0]))]) for r in rems)
            learned += int(ok)
            if len(shown) < 6: shown.append(f"{pre!r}->{text[:14]!r}{'*' if ok else ''}")
        with torch.no_grad():
            gf = life.gauge([ids_of(f) for f in facts]); cf = life._gauge_cos
        print(f"FACTS: the mouth completes {learned} of {len(prefixes)} held-out prefixes | the cortex alone on the {len(facts)} fact sentences {gf[0]:.3f} (cos {cf}) | {' '.join(shown)}")

# THE QUESTION ANSWERED (stage five's demo ruler, 2026-09-14): each fact's own question (tools/facts_stage5.txt, left of the bar) as the
# world's line, then the pause; the mouth's twelve symbols read with the recall; the question counts as answered when the mouth says a
# content word of the fact that the question itself does not contain ("what is cold?" -> "ice"). The conversation's form of the fact,
# against the prefix completion above (the prefix says the fact's first words; the question does not). Run with --qa.
if "--qa" in sys.argv:
    import os, re
    fp = "/Users/lukehamond/Projects/project/tools/facts_stage5.txt"
    STOP = set("is are the a an in on and to do we i you it of my your they can what where who how does".split())
    if os.path.exists(fp):
        pairs = [tuple(s.strip() for s in l.split("|")[:2]) for l in open(fp) if "|" in l]
        # THE PAUSE (2026-09-14, qa_by_gap read): the served body after night 174 answered 8, 11, 11, 6 of 30 at pauses of 1, 2, 4, 8 rests,
        # all of it the recall's (the cortex alone 0 at every pause); the ruler reads at two rests (the child's early answer) and at the
        # offset (eight), the second the harder
        for pause in (2, int(life.cfg.get("offset_ticks", 8))):
            n_ans = 0; shown = []
            for q, fact in pairs:
                keys = [w for w in re.findall(r"[a-z]+", fact.lower()) if w not in STOP and w not in q.lower()]
                life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.n_own = 0; life._follow = None
                got = []
                with torch.no_grad():
                    for ch in q:
                        i = TOK.token_to_id(ch); life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
                        life.bag_w = life.cfg["bag_decay"] * m.shift(life.bag_w) + m.E.weight[i]
                    for _ in range(pause):
                        life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
                        life.bag_w = life.cfg["bag_decay"] * life.bag_w
                    for _ in range(20):                                   # twenty symbols (13:40: twelve cut 'bees make ho' before its word)
                        xs, whos, faces, bundles, reads = life._window_tensors(list(life.win)[-m.window:])
                        C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
                        rd, conf, _w = life._recall(life.bag)
                        lm = m.readout(m.forecast(C, rd)); lm[life.bans] = float("-inf"); lm[life.sil] = float("-inf")
                        sym = int(lm.argmax()); got.append(TOK.decode([sym]))
                        life.win.append({"x": life.sil, "xo": sym, "face": torch.zeros(2), "bundle": life.bands, "read": rd, "r": 0.0})
                        life.bag_o = life.cfg["bag_decay"] * m.shift(life.bag_o) + m.E.weight[sym]; life.n_own += 1
                text = "".join(got); ok = any(k in text.lower() for k in keys)
                n_ans += int(ok)
                if len(shown) < 8: shown.append(f"{q!r}->{text[:14]!r}{'*' if ok else ''}")
            print(f"QA (pause {pause}): the mouth answers {n_ans} of {len(pairs)} fact questions | {' '.join(shown)}")
