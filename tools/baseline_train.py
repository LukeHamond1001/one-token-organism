"""THE BASELINE (a supervisor's instrument, 2026-09-14): the same cortex, born fresh, trained the ordinary way on the same lines the
served body heard (every parent and partner line in the page log, shuffled, many passes, batches of whole lines, the body's own
loss), read by the same held-out gauge every so many steps. What the corpus supports, against what a life of it reached.
usage: nice -n 19 python3 tools/baseline_train.py --steps 6000 --batch 64 --lr 1e-4 --every 500 [--d 1024 --layers 12 --heads 4]"""
import sys, json, time, random
sys.path.insert(0, "/Users/lukehamond/Projects/project")
import torch
from tokenizers import Tokenizer
from body.life import Life

def arg(name, default):
    names = {f"--{name}", f"--{name.replace('_', '-')}"}
    for i, a in enumerate(sys.argv[1:], 1):
        if "=" in a and a.split("=", 1)[0] in names:
            return type(default)(a.split("=", 1)[1])
        if a in names and i + 1 < len(sys.argv):
            return type(default)(sys.argv[i + 1])
    return default

steps, batch, lr, every = arg("steps", 6000), arg("batch", 64), arg("lr", 1e-4), arg("every", 500)
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.birth(TOK, device="cpu", d=arg("d", 1024), layers=arg("layers", 12), heads=arg("heads", 4), window=64, cfg={"night_batch": batch}, seed=0)
m = life.m
R = [json.loads(l) for l in open("/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl") if l.strip()]
def ids_of(t): return [TOK.token_to_id(ch) for ch in t if TOK.token_to_id(ch) is not None]
typed = [r["text"].strip() for r in R if r["action"] in ("line", "cue")]
lines = [ids_of(t) for t in typed if len(t) >= 2]
held = [ids_of(t.strip()) for t in open("/Users/lukehamond/Projects/project/tools/heldout_stage4.txt") if t.strip() and t.strip() not in set(typed)]
end_ = [life.end_id] if int(life.cfg.get("offset_ticks", 0)) > 0 else []
data = [l + end_ for l in lines]
print(f"baseline: {len(data)} lines heard ({sum(len(l) for l in data)} symbols, {len(set(typed))} distinct), held-out {len(held)}; cortex {sum(p.numel() for p in m.blocks.parameters())/1e6:.0f}M; steps {steps} batch {batch} lr {lr}", flush=True)
opt = torch.optim.Adam(m.parameters(), lr=lr); rng = random.Random(0); t0 = time.time()
def gauge():
    m.eval()
    with torch.no_grad():
        g, n = life.gauge(held); c = life._gauge_cos
    m.train(); return g, c
g0, c0 = gauge(); print(f"step 0: held-out {g0} (cos {c0})", flush=True)
m.train(); tot = 0.0; nb = 0
for step in range(1, steps + 1):
    chunk = rng.sample(data, batch)
    xs, xos, faces, bundles, reads, y, w = life._dream_batch(chunk)
    C = m.stream(m.inputs(xs, xos, faces, bundles, reads))
    ll, _ = m.latent_loss(m.latent_pred(C), y, w=w)
    opt.zero_grad(set_to_none=True); ll.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    for g_ in opt.param_groups: g_["lr"] = lr * min(1.0, step / 100)
    opt.step(); tot += float(ll.detach()); nb += 1
    if step % every == 0:
        g, c = gauge(); print(f"step {step}: train loss {tot/nb:.4f} | held-out {g} (cos {c}) | {(time.time()-t0)/60:.0f} min", flush=True); tot = 0.0; nb = 0
print("done", flush=True)
