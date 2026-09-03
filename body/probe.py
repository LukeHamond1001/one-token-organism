"""the supervisor's instruments on a saved second body (BODY_SPEC.md §7). Read-only.

  python3 -m body.probe data/body2.pt --cues "dog will |give |scared " --n 8 --ticks 24
"""
import argparse
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, "/Users/lukehamond/Projects/project")
from body.life import Life  # noqa: E402

ANSWERS = {"dog will ": ["go"], "scared ": ["dog", "ball"], "give ": ["milk", "ball", "book"], "where ball? ": ["ball"],
           "I had ": ["milk"], "first milk then ": ["ball"], "big dog bigger ": ["dog"], "why dog up? ": ["because"]}


def fresh(path, tok, mem_on):
    life = Life.load(path, tok, device="cpu", save_path=None)
    life.save_path = None
    if not mem_on:
        # THE CORTEX ALONE: no hippocampus at all. Emptied only at load, the store took the cue's own
        # symbols as memories while the cue was typed and then echoed the cue's last word: the
        # 'alone' column's loops ("will will will", runs 17 to 22) were that, not the cortex.
        life.store.K = life.store.K[:0]; life.store.V = life.store.V[:0]; life.store.S = life.store.S[:0]; life.store.W = life.store.W[:0]
        life.cfg["store_off"] = True
    life.cfg["wake_ticks"] = 10 ** 9
    life.cfg["wake_every"] = 10 ** 9; life.cfg["gate_every"] = 10 ** 9   # no lessons during a probe
    return life


def run(path, tok, cue, ticks, mem_on, greedy, seed=0):
    life = fresh(path, tok, mem_on)
    life.gen.manual_seed(seed)
    # the answer begins in the tick the cue's last symbol enters (the body reads the forecast made as
    # it enters and may act in that same tick), so the record starts there
    for ch in cue[:-1]:
        life.type_text(ch)
    while life.queue:
        life.tick()
    life.type_text(cue[-1])
    out = []
    while life.queue:
        life.tick()
        s = life.last.get("said", "")
        out.append(tok.token_to_id(s) if s else life.sil)
    for _ in range(ticks):
        if greedy:
            # the mouth alone, memory as set, always acting, taking its best guess
            with torch.no_grad():
                life._decay_feelings()
                C, pred, _, _ = life._step(life.sil, 0, learn_store=False)
                lg = life.m.readout(pred).clone(); lg[life.bans] = float("-inf"); lg[life.sil] = float("-inf")
                nxt = int(lg.argmax()); out.append(nxt)
                life._step(nxt, 1, learn_store=False)
        else:
            life.tick()
            s = life.last.get("said", "")
            out.append(tok.token_to_id(s) if s else life.sil)
    return "".join("·" if i == life.sil else tok.decode([i]) for i in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("body"); ap.add_argument("--tok", default="/Users/lukehamond/Projects/project/data/tok_char.json")
    ap.add_argument("--cues", default="dog will |scared |give |where ball? |I had |first milk then |big dog bigger |why dog up? ")
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--ticks", type=int, default=24)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tok)
    tot_start = tot_full = 0; cues = [c for c in a.cues.split("|") if c]
    for cue in cues:
        alone = run(a.body, tok, cue, 14, mem_on=False, greedy=True)
        withm = run(a.body, tok, cue, 14, mem_on=True, greedy=True)
        samples = [run(a.body, tok, cue, a.ticks, mem_on=True, greedy=False, seed=s) for s in range(a.n)]
        ans = ANSWERS.get(cue, [])
        started = fulls = 0
        for s in samples:
            letters = s.replace("·", ""); first = letters.split(" ")[0] if letters.strip() else ""
            started += int(any(len(first) >= 2 and first[:2] == x[:2] for x in ans)); fulls += int(any(first == x for x in ans))
        quiet = sum(s.count("·") for s in samples) / max(1, sum(len(s) for s in samples))
        tot_start += started; tot_full += fulls
        print(f"{cue!r:>18} alone {alone!r:>16} memory {withm!r:>16} | started {started}/{a.n} full {fulls}/{a.n} quiet {quiet:.2f} | {[s.replace('·','')[:10] for s in samples[:4]]}")
    print(f"TOTAL started {tot_start}/{a.n*len(cues)} full {tot_full}/{a.n*len(cues)}")


if __name__ == "__main__":
    main()
