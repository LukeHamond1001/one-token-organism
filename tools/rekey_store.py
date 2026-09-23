"""THE STORE REKEYED (a supervisor's instrument, 2026-09-18; item 40 of ITERATIONS.md): a copy's hippocampal store rebuilt from its
utterance memory (the world's last utterances, whole and in order, up to utt_cap of them) under whatever key the copy's constants
and the code now make: the reconsolidation a change of key needs, since a memory written under one form of the key is not found by
a query under another. Each utterance is replayed as the day heard it: the cortex, teacher-forced over the utterance alone (the
night's lockstep batch), gives each symbol's surprise, the write strength (no dopamine: the smiles of those days are not kept); the
fast bag and the slow context run as body/core/memory.py runs them (key, rest_tick, take_world, note_offset); the utterance's marks (the
seam on its first symbol, the start on its second, the boundary on its last) and its chain links are set as the day sets them.
The pause between utterances is not in the utterance memory: --gap ticks of the world's quiet stand for every pause (48, the
parent's line after the child's turn; the offset's own eight ticks come first). Memories older than the utterance memory's reach
are gone. Never saves back; --save-as writes the rekeyed copy.
usage: nice -n 19 python3 tools/rekey_store.py COPY.pt --flags ops/BASE_FLAGS.txt [--ctx-form shifted] --save-as OUT.pt [--gap 48] [--max-utts 0]"""
import sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch, torch.nn.functional as F
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY
from tools.faststore import FastStore as Store        # the body's store with room kept ahead (a replay of many symbols)

def arg(name, default):
    names = {f"--{name}", f"--{name.replace('_', '-')}"}
    for i, a in enumerate(sys.argv[1:], 1):
        if "=" in a and a.split("=", 1)[0] in names:
            return type(default)(a.split("=", 1)[1])
        if a in names and i + 1 < len(sys.argv):
            return type(default)(sys.argv[i + 1])
    return default

def parse_flags(s):
    toks = s.split(); cfg = {}; i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:].replace("-", "_")
            if k in PHYSIOLOGY:
                cfg[k] = type(PHYSIOLOGY[k])(toks[i + 1])
            i += 2
        else:
            i += 1
    return cfg

path = sys.argv[1]
cfg = parse_flags(open(arg("flags", "")).read()) if arg("flags", "") else {}
for a in sys.argv[1:]:
    if a.startswith("--") and "=" not in a:
        k = a[2:].replace("-", "_")
        if k in PHYSIOLOGY:
            cfg[k] = type(PHYSIOLOGY[k])(sys.argv[sys.argv.index(a) + 1])
gap = arg("gap", 48); max_utts = arg("max_utts", 0); save_as = arg("save_as", "")
TOK = Tokenizer.from_file(os.path.join(ROOT, "data/tok_char.json"))
life = Life.load(path, TOK, device="cpu", cfg=cfg); life.save_path = None; m = life.m; m.eval()
E = m.E.weight.detach(); sil = life.sil
OFFSET = int(life.cfg.get("offset_ticks", 8)); FLOOR = float(life.cfg.get("write_floor", 1e-6))
old = life.store
utts = [list(u) for u in life.utts if len(u) >= 1]
if max_utts > 0:
    utts = utts[-max_utts:]
print(f"body {os.path.basename(path)} nights {life.nights}: the store {old.n()} slots (mean strength {float(old.S.mean()):.3f}, seams {int(old.Bq.sum())} starts {int(old.Bs.sum())} boundaries {int(old.B.sum())}) | the utterance memory {len(utts)} utterances, {sum(len(u) for u in utts)} symbols | key_ctx {life.cfg.get('key_ctx')} ctx_form {life.cfg.get('ctx_form')} key_form {life.cfg.get('key_form')} gap {gap}", flush=True)
if str(life.cfg.get("key_form", "bag")) == "cortex":
    print("key_form cortex keys by the stream's state of the day, which a replay cannot rebuild; nothing done"); sys.exit(1)
# THE SURPRISES: the cortex teacher-forced over each utterance alone, in lockstep batches; the write strength of each symbol is one
# minus the cosine of the forecast made at the position before to the symbol's embedding (the day's rule, without dopamine)
t0 = time.time(); surps = []
with torch.no_grad():
    for i in range(0, len(utts), 32):
        chunk = utts[i:i + 32]
        xs, xos, faces, bundles, reads, y, w = life._dream_batch(chunk)
        pred = m.latent_pred(m.stream(m.inputs(xs, xos, faces, bundles, reads)))
        cos = F.cosine_similarity(pred, E[y], dim=-1)
        for j, ids in enumerate(chunk):
            surps.append([float(1.0 - cos[j, t]) for t in range(len(ids))])
        if (i // 32) % 20 == 0:
            print(f"  surprises for {min(i + 32, len(utts))} of {len(utts)} utterances, {time.time() - t0:.0f}s", flush=True)
flat = [s for u in surps for s in u]
print(f"surprises done in {time.time() - t0:.0f}s: mean {sum(flat) / len(flat):.3f}, share under 1e-4 (never written) {sum(1 for s in flat if s <= 1e-4) / len(flat):.3f}", flush=True)
# THE STORE REBUILT
st = Store(m.d, cap=int(life.cfg.get("store_cap", 8192)), temp=float(life.cfg["store_temp"]), device="cpu", links=int(life.cfg.get("store_links", 4)))
st.saturate = bool(getattr(old, "saturate", False)); st.temp = old.temp
life.store = st
life.bag_w.zero_(); life.bag_o.zero_(); life.n_own = 0; life.ctx_cur.zero_(); life.ctx_prev.zero_(); life._utt_open = False
life._prev_slot = -1; life._last_write = None; life._follow = None
t0 = time.time(); written = 0; merged = 0
with torch.no_grad():
    for i, ids in enumerate(utts):
        if i > 0:
            for _ in range(OFFSET):
                life.rest_tick()
            if life._last_write is not None:
                st.mark_boundary(*life._last_write)                        # the memory of the last symbol carries the boundary
            life._prev_slot = -1; life.note_offset(); life._follow = None
            for _ in range(max(0, gap - OFFSET)):
                life.rest_tick()
        st.episode += 1
        for j, x in enumerate(ids):
            key_ = life.key; ex = E[x]
            if key_.norm() > FLOOR:
                n_before = st.n()
                if st.write(key_, ex, surps[i][j], 0):
                    written += 1; merged += int(st.n() == n_before)
                    rm_ = getattr(st, "last_remap", None)
                    if rm_ is not None:                                      # the eviction moved slots: the link's index follows (the thirty-second defect)
                        life._prev_slot = int(rm_[life._prev_slot]) if 0 <= life._prev_slot < rm_.numel() else -1
                    if int(life.cfg.get("store_chain", 0)) and st.last_idx >= 0:
                        if life._prev_slot >= 0:
                            st.link(life._prev_slot, st.last_idx, tag=st.episode)
                        life._prev_slot = st.last_idx
                    if int(life.cfg.get("episode_chain", 0)) and st.last_idx >= 0:
                        st.note(st.episode, st.last_idx)
                    life._last_write = (key_.clone(), ex.clone())
                    if j == 1:
                        st.mark_start(key_, ex)                              # the utterance's first kept memory: the second symbol's (the first is the seam)
                    if j == 0:
                        st.mark_seam(key_, ex)                               # the first symbol under the last line's faded context
            life.rest_tick(world=True); life.take_world(int(x))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} utterances replayed, {st.n()} slots, {time.time() - t0:.0f}s", flush=True)
    for _ in range(OFFSET):
        life.rest_tick()
    if life._last_write is not None:
        st.mark_boundary(*life._last_write)
    life._prev_slot = -1; life.note_offset()
print(f"REKEYED in {time.time() - t0:.0f}s: {st.n()} slots from {written} writes ({merged} merged into a slot that said the same), mean strength {float(st.S.mean()):.3f}, seams {int(st.Bq.sum())} starts {int(st.Bs.sum())} boundaries {int(st.B.sum())}, links {int((st.N >= 0).sum())}", flush=True)
if save_as:
    life._store_fresh = True                                          # a rebuilt store lives a day before its first fade (the body's guard, 2026-09-19)
    life.save_path = save_as; life.save(); life.save_path = None; print(f"the rekeyed copy saved as {save_as} (marked fresh: no fade on its first night)", flush=True)
else:
    print("nothing saved")
