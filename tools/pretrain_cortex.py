"""THE CORTEX OVERHEARS A CORPUS (2026-09-18, the user's word: its only sense is text, so a corpus is the world it can overhear, as
a child overhears millions of words aimed at no one). The night's own lesson (NREM: the cortex's forecast of the next symbol over
sequences run in lockstep, Adam at the night's rate, the store off) over sentences of a corpus instead of the utterance memory,
on a COPY of a saved body; the held-out (lines no parent typed) and the fact sentences read by the cortex alone before and after,
the copy saved as asked. The rest of the body (the store, the gate, the face, the mood) is untouched; the store's keys are the
embeddings' and must be rebuilt after (tools/rekey_store.py) before the copy serves.
THE CORPUS AS THE BODY HEARS IT: lower case; commas, quotes, colons and semicolons dropped; split at . ? and !; sentences of
8 to 63 symbols made only of the tokenizer's characters; shuffled at a fixed seed; taken until --chars symbols.
usage: nice -n 19 python3 tools/pretrain_cortex.py COPY.pt --flags ops/BASE_FLAGS.txt --corpus FILE --chars 1000000 [--batch 16] [--lr 1e-5] [--rounds 1] --save-as OUT.pt"""
import sys, os, re, time, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY

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
corpus = arg("corpus", ""); n_chars = arg("chars", 1000000); batch = arg("batch", 16); lr = arg("lr", 1e-5); rounds = arg("rounds", 1)
save_as = arg("save_as", ""); seed = arg("seed", 0); report_every = arg("report_every", 200)
TOK = Tokenizer.from_file(os.path.join(ROOT, "data/tok_char.json"))
life = Life.load(path, TOK, device="cpu", cfg=cfg); life.save_path = None; m = life.m
life.cfg["night_batch"] = max(1, batch)                     # the gauge runs batched
end_ = [life.end_id] if int(life.cfg.get("offset_ticks", 0)) > 0 else []
W = int(m.window) - len(end_)
def ids_of(t): return [TOK.token_to_id(ch) for ch in t if TOK.token_to_id(ch) is not None]
# the rulers: the held-out lines and the fact sentences, the cortex alone
R_ = [l.strip() for l in open(os.path.join(ROOT, "data/watch2_caregiver.jsonl"))]
import json
typed_ = set(json.loads(l)["text"].strip() for l in R_ if l and '"action": "line"' in l or '"action": "cue"' in l)
held = [ids_of(t.strip()) for t in open(os.path.join(ROOT, "tools/heldout_stage4.txt")) if t.strip() and t.strip() not in typed_]
facts = [ids_of(l.split("|")[1].strip()) for l in open(os.path.join(ROOT, "tools/facts_stage5.txt")) if "|" in l]
# THE CORPUS
ok_chars = set(ch for ch in "abcdefghijklmnopqrstuvwxyz .?!'")
raw = open(corpus, encoding="utf-8", errors="ignore").read().lower()
raw = raw.replace("<|endoftext|>", " ").replace('"', "").replace(",", "").replace(";", "").replace(":", "").replace("\n", " ")
sents = []
for s in re.split(r"(?<=[.?!])\s+", raw):
    s = re.sub(r"\s+", " ", s).strip()
    if 8 <= len(s) <= W and all(ch in ok_chars for ch in s):
        sents.append(s)
random.Random(seed).shuffle(sents)
take = []; total = 0
for s in sents:
    if total >= n_chars:
        break
    take.append(ids_of(s) + end_); total += len(s)
print(f"body {os.path.basename(path)} nights {life.nights} | corpus {os.path.basename(corpus)}: {len(sents)} usable sentences, {len(take)} taken ({total} symbols) | batch {batch} lr {lr} beta2 {life.cfg.get('night_beta2', 0.999)} rounds {rounds} window {m.window}", flush=True)
with torch.no_grad():
    m.eval(); h0 = life.gauge(held); hc0 = life._gauge_cos; f0 = life.gauge(facts); fc0 = life._gauge_cos
print(f"before: HELD-OUT {h0[0]} (cos {hc0}) over {h0[1]} | the facts by the cortex {f0[0]} (cos {fc0})", flush=True)
# THE NIGHT'S LESSON OVER THE CORPUS (life.night, the NREM branch with night_batch > 0)
opt = torch.optim.Adam(m.parameters(), lr=float(lr), betas=(0.9, float(life.cfg.get("night_beta2", 0.999))))
m.train(); t0 = time.time(); step = 0; losses = []
for r in range(rounds):
    order = torch.randperm(len(take), generator=torch.Generator().manual_seed(seed + r)).tolist()
    for i0 in range(0, len(order), batch):
        opt.zero_grad(set_to_none=True)
        xs, xos, faces, bundles, reads, y, w = life._dream_batch([take[j] for j in order[i0:i0 + batch]])
        C = m.stream(m.inputs(xs, xos, faces, bundles, reads))
        ll, _ = m.latent_loss(m.latent_pred(C), y, w=w)
        if not bool(torch.isfinite(ll.detach())):
            continue
        ll.backward(); losses.append(float(ll.detach())); step += 1
        life._night_step(opt)
        if step % report_every == 0:
            print(f"  step {step}: loss {sum(losses[-report_every:]) / report_every:.4f} | {time.time() - t0:.0f}s | {step * batch * (total / max(1, len(take))):.0f} symbols seen", flush=True)
m.eval()
with torch.no_grad():
    h1 = life.gauge(held); hc1 = life._gauge_cos; f1 = life.gauge(facts); fc1 = life._gauge_cos
print(f"after {step} steps in {time.time() - t0:.0f}s (loss {sum(losses[:50]) / max(1, len(losses[:50])):.4f} -> {sum(losses[-50:]) / max(1, len(losses[-50:])):.4f}): HELD-OUT {h1[0]} (cos {hc1}) | the facts by the cortex {f1[0]} (cos {fc1})", flush=True)
if save_as:
    life.save_path = save_as; life.save(); life.save_path = None; print(f"saved {save_as}", flush=True)
else:
    print("nothing saved")
