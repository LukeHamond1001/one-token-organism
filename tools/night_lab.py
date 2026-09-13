"""THE NIGHT LAB (a supervisor's instrument, 2026-09-13): load a saved body on a copy (never saved back), dream the store as the
night does, and consolidate the dreams with a synaptic update every `batch` replays instead of one per round, at the night's rate.
Reads after each round: the NREM loss, the cortex-alone gauge on the dream set and on the parent's last sixty lines (material the
night may not have replayed: the language-model number), then the mouth on a few prompts.
usage: python3 tools/night_lab.py data/watch2.pt --flags FLAGS.txt --starts 256 --rounds 4 --batch 8 [--lr 1e-4] [--warm 0] [--seed 0]   (the lockstep path)"""
import sys, json, time, random
sys.path.insert(0, "/Users/lukehamond/Projects/project")
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY

def arg(name, default):
    names = {f"--{name}", f"--{name.replace('_', '-')}"}                   # --night_lr and --night-lr alike
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
flags = arg("flags", ""); starts = arg("starts", 256); rounds = arg("rounds", 4); batch = arg("batch", 8)
lr = arg("lr", 1e-4); seed = arg("seed", 0); n_lines = arg("lines", 60)
cfg = parse_flags(open(flags).read()) if flags else {}
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu", cfg=cfg, seed=seed); m = life.m; m.eval()
print(f"body: nights {life.nights} store {life.store.n()} | starts {starts} rounds {rounds} batch {batch} lr {lr} | night_lr {life.cfg['night_lr']} rounds served {life.cfg['night_rounds']}", flush=True)
R = [json.loads(l) for l in open("/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl") if l.strip()]
def ids_of(t): return [TOK.token_to_id(ch) for ch in t if TOK.token_to_id(ch) is not None]
recent = [ids_of(r["text"].strip()) for r in R if r["action"] in ("line", "cue") and r.get("voice") != "b"][-n_lines:]
recent = [x for x in recent if len(x) >= 2]

def mouth(prompt, n=24):
    zero = torch.zeros(m.d)
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.n_own = 0
    out = []
    with torch.no_grad():
        for ch in prompt:
            i = TOK.token_to_id(ch)
            life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.bag_w = life.cfg["bag_decay"] * m.shift(life.bag_w) + m.E.weight[i]
        for _ in range(int(life.cfg.get("offset_ticks", 8))):
            life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.bag_w = life.cfg["bag_decay"] * life.bag_w
        cortex = []
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
t0 = time.time()
dreams = life.dreams(starts)
print(f"dreams {len(dreams)} mean_len {sum(len(d) for d in dreams)/max(1,len(dreams)):.1f} drawn in {time.time()-t0:.0f}s; examples {[TOK.decode(d)[:24] for d in dreams[:4]]}", flush=True)
with torch.no_grad():
    g_d = life.gauge(dreams[:128]); g_r = life.gauge(recent)
print(f"before: gauge dreams {g_d[0]} (cos {life._gauge_cos}) | parent's last {len(recent)} lines {g_r[0]} over {g_r[1]} symbols", flush=True)
for p in PROMPTS[:3]:
    mo, co = mouth(p); print(f"   {p!r:20} mouth {mo!r:26} cortex {co!r}", flush=True)
opt = torch.optim.Adam(m.parameters(), lr=lr)
warm = arg("warm", 0); life.cfg["night_batch"] = batch                      # the gauge and the lesson on the lockstep path
rng = random.Random(seed)
steps = 0
for r in range(rounds):
    t1 = time.time(); m.train(); order = list(range(len(dreams))); rng.shuffle(order)
    tot = 0.0; nb = 0
    for i0 in range(0, len(order), batch):
        opt.zero_grad(set_to_none=True)
        xs, xos, faces, bundles, reads, y, w = life._dream_batch([dreams[j] for j in order[i0:i0 + batch]])
        C = m.stream(m.inputs(xs, xos, faces, bundles, reads))
        ll, _ = m.latent_loss(m.latent_pred(C), y, w=w)
        if not bool(torch.isfinite(ll.detach())):
            continue
        ll.backward(); tot += float(ll.detach()); nb += 1; steps += 1
        if warm:
            for g_ in opt.param_groups:
                g_["lr"] = lr * min(1.0, steps / warm)
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if bool(torch.isfinite(gn)):
            opt.step()
        opt.zero_grad(set_to_none=True)
    m.eval()
    with torch.no_grad():
        g_d = life.gauge(dreams[:128]); cd = life._gauge_cos; g_r = life.gauge(recent); cr = life._gauge_cos
    print(f"round {r+1}: steps {steps} | train loss {tot/max(1,nb):.4f} | gauge dreams {g_d[0]} (cos {cd}) | parent's lines {g_r[0]} (cos {cr}) | {time.time()-t1:.0f}s", flush=True)
    for p in PROMPTS:
        mo, co = mouth(p); print(f"   {p!r:20} mouth {mo!r:26} cortex {co!r}", flush=True)
print(f"done in {time.time()-t0:.0f}s; nothing saved", flush=True)
