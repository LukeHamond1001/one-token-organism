"""THE GAUGE BY POSITION (a supervisor's instrument, 2026-09-13): the cortex-alone accuracy on the held-out lines, split by the
position in the line (the first symbols after rest against the rest of the line), for two saved bodies: where a night's change lands.
usage: python3 tools/gauge_by_position.py BEFORE.pt AFTER.pt"""
import sys, json
sys.path.insert(0, "/Users/lukehamond/Projects/project")
import torch
from tokenizers import Tokenizer
from body.life import Life
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
R = [json.loads(l) for l in open("/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl") if l.strip()]
typed = set(r["text"].strip() for r in R if r["action"] in ("line", "cue"))
def ids_of(t): return [TOK.token_to_id(ch) for ch in t if TOK.token_to_id(ch) is not None]
held = [ids_of(t.strip()) for t in open("/Users/lukehamond/Projects/project/tools/heldout_stage4.txt") if t.strip() and t.strip() not in typed]
BANDS = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 30)]
def by_position(path):
    life = Life.load(path, TOK, device="cpu", cfg={"night_batch": 32}); life.save_path = None; m = life.m
    hits = {b: [0, 0] for b in BANDS}
    with torch.no_grad():
        for i in range(0, len(held), 32):
            xs, xos, faces, bundles, reads, y, w = life._dream_batch(held[i:i + 32])
            lg = m.readout(m.latent_pred(m.stream(m.inputs(xs, xos, faces, bundles, reads))))
            lg[..., [b for b in life.bans if b != life.eot]] = float("-inf")
            if life.end_id != life.sil: lg[..., life.sil] = float("-inf")
            ok = (lg.argmax(-1) == y).float() * w
            for t in range(y.shape[1]):
                for b in BANDS:
                    if b[0] <= t < b[1]:
                        hits[b][0] += float(ok[:, t].sum()); hits[b][1] += float(w[:, t].sum())
    return {b: (hits[b][0] / max(1, hits[b][1]), int(hits[b][1])) for b in BANDS}
a = by_position(sys.argv[1]); b = by_position(sys.argv[2])
print(f"held-out {len(held)} lines; accuracy by position (before -> after, n symbols):")
for band in BANDS:
    print(f"  positions {band[0]:2d}-{band[1]-1:2d}: {a[band][0]:.3f} -> {b[band][0]:.3f}  ({b[band][0]-a[band][0]:+.3f}, n {a[band][1]})")
