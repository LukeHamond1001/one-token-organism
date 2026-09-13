"""A NIGHT ON A COPY (a supervisor's instrument, 2026-09-13): load a saved body on a copy (never saved back), run its night as the
served body would with the given constants, and read the night's own report, the language-model accuracy on the parent's last
sixty lines (unreplayed material) before and after, the time it took, and the mouth.
usage: python3 tools/night_copy.py COPY.pt --flags FLAGS.txt --batch 8 --rounds 6 --starts 512 [--lr 1e-4]"""
import sys, json, time
sys.path.insert(0, "/Users/lukehamond/Projects/project")
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY

def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
        if a == f"--{name}":
            return type(default)(sys.argv[sys.argv.index(a) + 1])
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
cfg.update(dict(night_batch=arg("batch", 8), night_rounds=arg("rounds", 6), night_starts=arg("starts", 512), night_load=0.0, night_lr=arg("lr", 1e-4), night_warm=arg("warm", 0), dream_who=arg("who", 0), dream_tag=arg("tag", 0), dream_draw=arg("draw", "strength"), dream_skip=arg("skip", 0)))
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu", cfg=cfg, seed=arg("seed", 0)); m = life.m; m.eval()
life.save_path = None                                                   # a copy: the night must not save it
R = [json.loads(l) for l in open("/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl") if l.strip()]
def ids_of(t): return [TOK.token_to_id(ch) for ch in t if TOK.token_to_id(ch) is not None]
recent = [ids_of(r["text"].strip()) for r in R if r["action"] in ("line", "cue") and r.get("voice") != "b"][-60:]
recent = [x for x in recent if len(x) >= 2]
typed_ = set(r["text"].strip() for r in R if r["action"] in ("line", "cue"))
held = [ids_of(t.strip()) for t in open("/Users/lukehamond/Projects/project/tools/heldout_stage4.txt") if t.strip() and t.strip() not in typed_]
# THE OLD LINES: the parent's distinct lines from days long faded from the store (the store holds about five hundred utterances;
# the corpus near four thousand): material the night cannot have replayed, the language-model reading proper
d0, d1 = [int(x) for x in arg("old_days", "110-125").split("-")]
old_lines = list(dict.fromkeys(r["text"].strip() for r in R if r["action"] in ("line", "cue") and r.get("voice") != "b" and d0 <= int(r.get("day", -1)) <= d1))
old = [ids_of(t) for t in old_lines][:150]; old = [x for x in old if len(x) >= 2]

def mouth(prompt, n=24):
    zero = torch.zeros(m.d)
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.n_own = 0
    out, cortex = [], []
    with torch.no_grad():
        for ch in prompt:
            i = TOK.token_to_id(ch)
            life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.bag_w = life.cfg["bag_decay"] * m.shift(life.bag_w) + m.E.weight[i]
        for _ in range(int(life.cfg.get("offset_ticks", 8))):
            life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.bag_w = life.cfg["bag_decay"] * life.bag_w
        for _ in range(n):
            xs, whos, faces, bundles, reads = life._window_tensors(list(life.win)[-m.window:])
            C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
            rd, conf, _ = life.store.read(life.bag)
            lc = m.readout(m.forecast(C, zero)); lc[life.bans] = float("-inf"); lc[life.sil] = float("-inf")
            lm = m.readout(m.forecast(C, rd)); lm[life.bans] = float("-inf"); lm[life.sil] = float("-inf")
            sym = int(lm.argmax()); cortex.append(TOK.decode([int(lc.argmax())])); out.append(TOK.decode([sym]))
            life.win.append({"x": life.sil, "xo": sym, "face": torch.zeros(2), "bundle": life.bands, "read": rd, "r": 0.0})
            life.bag_o = life.cfg["bag_decay"] * m.shift(life.bag_o) + m.E.weight[sym]; life.n_own += 1
    return "".join(out), "".join(cortex)

PROMPTS = ["do you want milk?", "what do you have?", "are you here?", "can you play?", "I want ", "I see "]
print(f"body: nights {life.nights} store {life.store.n()} | night_batch {cfg['night_batch']} rounds {cfg['night_rounds']} starts {cfg['night_starts']} lr {cfg['night_lr']} warm {cfg['night_warm']} dream_who {cfg['dream_who']} tag {cfg['dream_tag']} draw {cfg['dream_draw']} seed {arg('seed', 0)}", flush=True)
life.cfg["night_batch"] = max(1, int(life.cfg.get("night_batch", 0)))
with torch.no_grad():
    g0 = life.gauge(recent); c0 = life._gauge_cos; o0 = life.gauge(old); oc0 = life._gauge_cos; h0 = life.gauge(held); hc0 = life._gauge_cos
print(f"before: the parent's last {len(recent)} lines {g0[0]} (cos {c0}) over {g0[1]} symbols | {len(old)} old lines (days {d0}-{d1}) {o0[0]} (cos {oc0}) over {o0[1]} | HELD-OUT {len(held)} lines {h0[0]} (cos {hc0}) over {h0[1]}", flush=True)
# A DIAGNOSTIC (2026-09-13): --lines-as-dreams N replaces the night's dreams with the parent's last N lines from the log, whole and
# clean, to tell whether the night's lesson form hurts the held-out or only what it dreams
n_lines = arg("lines_as_dreams", 0)
if n_lines > 0:
    lines_ = [ids_of(r["text"].strip()) for r in R if r["action"] in ("line", "cue")][-n_lines:]
    lines_ = [x for x in lines_ if len(x) >= 4]
    _orig_dreams = life.dreams
    life.dreams = lambda n=None, with_who=False: (lines_, [[False] * len(d) for d in lines_]) if with_who else lines_
    print(f"diagnostic: the night dreams the parent's last {len(lines_)} lines, whole", flush=True)
t0 = time.time(); rep = life.night(); dt = time.time() - t0
keep = {k: rep.get(k) for k in ("dreams", "mean_len", "own_share", "examples", "nrem_steps", "nrem_curve", "gauge", "rem_steps", "error", "discarded")}
print(f"the night took {dt:.0f}s: {json.dumps(keep)}", flush=True)
with torch.no_grad():
    g1 = life.gauge(recent); c1 = life._gauge_cos; o1 = life.gauge(old); oc1 = life._gauge_cos; h1 = life.gauge(held); hc1 = life._gauge_cos
print(f"after: the parent's last {len(recent)} lines {g1[0]} (cos {c1}) | old lines {o1[0]} (cos {oc1}) | HELD-OUT {h1[0]} (cos {hc1}) | dreams {rep.get('gauge', {}).get('before')} -> {rep.get('gauge', {}).get('after')}", flush=True)
for p in PROMPTS:
    mo, co = mouth(p); print(f"   {p!r:20} mouth {mo!r:26} cortex {co!r}", flush=True)
print("nothing saved", flush=True)
