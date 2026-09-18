"""THE KEY'S SEPARATION (a supervisor's instrument, 2026-09-18; item 40 of ITERATIONS.md): the store's geometry alone, without the
cortex. An empty store is written as the day writes it, at strength one, from the page log's last --days days of typed lines (both
voices are the world's; the child's own symbols never enter a key) under a chosen form of the key; the thirty facts of
tools/facts_stage5.txt are heard once among them, after the first third of the lines (the talk that follows is what collides); then
each fact is asked in the taught wording and in the rephrased wording (tools/heldout_rephrased.txt): the question as the world's
line, the offset, and the read at the answer's onset and along the answer with its own symbols fed back (teacher-forced: the store's
knowledge at each position, not the mouth's). Per form: the onset hits, the mean accuracy along the answer, and the onset margin
(the fact's own slot's logit minus the best slot's that says another symbol, in nats: positive means the fact's memory wins).
FORMS, each "ctx:lam:swap" in --forms: ctx = bag (the previous utterance order-free) | shifted (with its order) | cortex (the
cortex's code of the utterance just ended: its stream state after the last symbol with the running mean out, unit length; the
form (c) of item 40, only with swap offset, since the code exists at the offset); lam = key_ctx;
swap = first (the previous utterance becomes the key's context at the next utterance's first symbol, after that symbol's own write:
the served rule, which keys an answer's onset by the utterance before the question) | offset (at the offset, so the onset is keyed by
the utterance just ended). --expand N --k K: a dentate-like expansion, a fixed random projection to N units with the top K kept, on
keys and queries alike (the query's norm kept).
usage: nice -n 19 python3 tools/key_separation.py BODY.pt --flags ops/BASE_FLAGS.txt --days 6 --forms bag:0.5:first,bag:0.5:offset,shifted:0.5:offset
"""
import sys, os, json, math
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
days = arg("days", 6); forms = arg("forms", "bag:0.5:first,bag:0.5:offset,shifted:0.5:offset").split(",")
expand = arg("expand", 0); kwta = arg("k", 64); seed = arg("seed", 0); gap_parent = arg("gap", 48); gap_other = arg("gap_other", 10)
TOK = Tokenizer.from_file(os.path.join(ROOT, "data/tok_char.json"))
life = Life.load(path, TOK, device="cpu", cfg=cfg); m = life.m; m.eval()
E = m.E.weight.detach().float(); d = m.d; sil = life.sil
En = F.normalize(E, dim=1)
def shift(v): return m.shift(v)
def ids_of(t): return [TOK.token_to_id(ch) for ch in t if TOK.token_to_id(ch) is not None]
DEC = float(life.cfg["bag_decay"]); REST = float(life.cfg.get("bag_rest_decay", 0.0)) or DEC; RHO = float(life.cfg.get("ctx_decay", 0.95))
OWNW = float(life.cfg["bag_own_weight"]); OWN_FADE = int(life.cfg.get("bag_own_fade", 0)); OFFSET = int(life.cfg.get("offset_ticks", 8))
TEMP = float(life.cfg["store_temp"]); LINKS = int(life.cfg.get("store_links", 4)); FOLLOW = float(life.cfg.get("read_follow", 0.0))
CHAIN = int(life.cfg.get("store_chain", 0)); CAP = int(life.cfg.get("store_cap", 8192)); SAT = bool(int(life.cfg.get("store_sat", 0)))

# THE MATERIAL: the last days' typed lines in order, the facts heard once after the first third
R = [json.loads(l) for l in open(os.path.join(ROOT, "data/watch2_caregiver.jsonl")) if l.strip()]
L = [r for r in R if r["action"] in ("line", "cue") and r.get("day")]
last_day = L[-1]["day"]
lines = [(r["text"].strip(), "other" if r.get("voice") == "b" else "parent") for r in L if r["day"] >= last_day - days + 1]
lines = [(t, w) for t, w in lines if len(ids_of(t)) >= 2]
facts = [tuple(x.strip() for x in l.split("|")) for l in open(os.path.join(ROOT, "tools/facts_stage5.txt")) if "|" in l]
reph = [tuple(x.strip() for x in l.split("|")) for l in open(os.path.join(ROOT, "tools/heldout_rephrased.txt")) if "|" in l]
reph_of = {a: q for q, a in reph}
third = len(lines) // 3
material = lines[:third] + [x for q, a in facts for x in ((q, "parent"), (a, "other"))] + lines[third:]
print(f"body {os.path.basename(path)} nights {life.nights} | {len(lines)} lines of days {last_day - days + 1}-{last_day}, {sum(len(ids_of(t)) for t, _ in lines)} symbols; {len(facts)} facts heard once after line {third} | temp {TEMP} links {LINKS} follow {FOLLOW} chain {CHAIN} bag {DEC}/{REST} ctx_decay {RHO} own_fade {OWN_FADE} offset {OFFSET} gaps {gap_parent}/{gap_other} | expand {expand} k {kwta}", flush=True)

class Ctx:
    """the two fast bags and the slow context, as body/life.py keeps them (take_world, take_own, rest_tick, key, bag), with the
    swap of the slow context placed by --swap"""
    def __init__(self, form, lam, swap):
        self.form, self.lam, self.swap = form, lam, swap
        self.bag_w = torch.zeros(d); self.bag_o = torch.zeros(d); self.n_own = 0
        self.ctx_cur = torch.zeros(d); self.ctx_prev = torch.zeros(d); self.open = False
    def scaled(self, ctx, bag):
        n = float(ctx.norm()); return ctx * (float(bag.norm()) / n) if n > 1e-6 else torch.zeros_like(ctx)
    def key(self):
        return self.bag_w if self.lam <= 0 else self.bag_w + self.lam * self.scaled(self.ctx_prev, self.bag_w)
    def query(self):
        w = self.bag_w
        for _ in range(min(self.n_own, 64)):
            w = shift(w)
        if OWN_FADE == 2 and self.n_own > 0:
            w = w * (DEC / REST) ** min(self.n_own, 64)
        q = w + OWNW * self.bag_o
        return q if self.lam <= 0 else q + self.lam * self.scaled(self.ctx_cur, q)
    def take_world(self, i):
        ex = E[i]; self.bag_w = shift(self.bag_w) + ex
        if not self.open:
            if self.swap == "first":
                self.ctx_prev = self.ctx_cur.clone()
            if self.form != "cortex":
                self.ctx_cur = torch.zeros(d)
            self.open = True
        if self.form == "shifted":
            self.ctx_cur = RHO * shift(self.ctx_cur) + ex
        elif self.form == "bag":
            self.ctx_cur = RHO * self.ctx_cur + ex
        self.bag_o = torch.zeros(d); self.n_own = 0
    def take_own(self, i):
        ex = E[i]; self.bag_o = (DEC / REST) * self.bag_o
        if OWN_FADE == 1:
            self.bag_w = (DEC / REST) * self.bag_w
        self.bag_o = shift(self.bag_o) + ex; self.n_own += 1
    def rest(self, world=False):
        self.bag_w = (DEC if world else REST) * self.bag_w; self.bag_o = REST * self.bag_o
    def offset(self, code=None):
        self.open = False
        if self.form == "cortex" and code is not None:
            self.ctx_cur = code.clone()                          # the code of the utterance just ended: the query's context from here
        if self.swap == "offset":
            self.ctx_prev = self.ctx_cur.clone()

g = torch.Generator().manual_seed(seed)
P = torch.randn(d, expand, generator=g) / math.sqrt(d) if expand > 0 else None
def ex_(v):
    """the dentate expansion of a key or a query: the projection, the top k kept, the norm kept"""
    if P is None:
        return v
    h = v @ P; thr = torch.topk(h, kwta).values[-1]; h = torch.where(h >= thr, h, torch.zeros_like(h))
    return F.normalize(h, dim=0) * float(v.norm())

CODE = {}
if any(f.split(":")[0] == "cortex" for f in forms):
    # THE CORTEX'S CODE OF AN UTTERANCE: the stream run over the utterance alone (the night's lockstep batch, no own sound, no reads),
    # the state after its last symbol (a rest appended so that position exists), the running mean of the day's states (c_mu, kept in
    # the save) taken out, unit length
    texts = list(dict.fromkeys([t for t, _ in material] + [q for q, _ in facts] + [q for q, _ in reph]))
    mu = life._c_mu.float()
    t0 = __import__("time").time()
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            chunk = texts[i:i + 32]; seqs = [ids_of(t) + [sil] for t in chunk]
            xs, xos, faces, bundles, reads, y, w = life._dream_batch(seqs)
            C = m.stream(m.inputs(xs, xos, faces, bundles, reads))
            for j, t in enumerate(chunk):
                c = C[j, len(seqs[j]) - 1].float()
                CODE[t] = F.normalize(c - mu, dim=0)
    codes = torch.stack([CODE[t] for t in texts])
    G = codes @ codes.T; off = G[~torch.eye(len(texts), dtype=torch.bool)]
    same = [float(CODE[q] @ CODE[reph_of[a]]) for q, a in facts if a in reph_of]
    print(f"cortex codes for {len(texts)} utterances in {__import__('time').time() - t0:.0f}s: cosine between different utterances mean {float(off.mean()):.3f} (90th pct {float(off.quantile(0.9)):.3f}); a question and its rephrasing mean {sum(same) / len(same):.3f}", flush=True)

class World:
    """the day as the store hears it: each line's symbols written under the key of the context before them"""
    def __init__(self, ctx):
        self.ctx = ctx; self.st = Store(expand if expand > 0 else d, cap=CAP, temp=TEMP, links=LINKS); self.st.saturate = SAT
        self.prev_slot = -1; self.follow = None; self.last_who = None; self.last_text = None
    def hear(self, text, who, record=None):
        ids = ids_of(text)
        gap = gap_other if (who == "other" and self.last_who == "parent") else gap_parent
        if self.last_who is not None:
            for t in range(OFFSET):
                self.ctx.rest()
            self.ctx.offset(code=CODE.get(self.last_text)); self.prev_slot = -1
            for t in range(max(0, gap - OFFSET)):
                self.ctx.rest()
        self.st.episode += 1
        for j, i in enumerate(ids):
            k = self.ctx.key()
            if k.norm() > 1e-30 and self.st.write(ex_(k), E[i], 1.0, 0):
                if record is not None and j == 0:
                    record.append(self.st.last_idx)
                if CHAIN and self.prev_slot >= 0:
                    self.st.link(self.prev_slot, self.st.last_idx, tag=self.st.episode)
                self.prev_slot = self.st.last_idx
            self.ctx.rest(world=True); self.ctx.take_world(i)
        self.last_who = who; self.last_text = text
    def read(self):
        st = self.st; q = ex_(self.ctx.query())
        fo = self.follow if (FOLLOW > 1.0 and self.follow is not None) else None
        pred, conf, win = st.read(q, follow=fo, follow_gain=FOLLOW)
        if FOLLOW > 1.0:
            if win >= 0 and st.N.shape[0] == st.n():
                tag = -1
                if fo is not None:
                    slot, t_ = fo; row = st.N[slot]
                    if bool(((row >= 0) & (st.NE[slot] == int(t_)) & (row == int(win))).any()):
                        tag = int(t_)
                if tag < 0 and int(st.N[win][0]) >= 0:
                    tag = int(st.NE[win][0])
                self.follow = (int(win), tag) if tag >= 0 else None
            else:
                self.follow = None
        sym = int((En @ F.normalize(pred, dim=0)).argmax()) if float(pred.norm()) > 1e-9 else -1
        return sym, conf, win, q

def ask(world, question, answer, onset_slot):
    """the question heard (and written, as live), the offset, two ticks of the gate's wait, then the read at the onset and along the
    answer teacher-forced; returns (onset hit, chain accuracy, onset margin in nats, onset winner share of the right symbol)"""
    ids = ids_of(answer)
    world.hear(question, "parent")
    for t in range(OFFSET):
        world.ctx.rest()
    world.ctx.offset(code=CODE.get(question)); world.prev_slot = -1; world.follow = None
    for t in range(2):
        world.ctx.rest()
    sym, conf, win, q = world.read()
    st = world.st
    sims = (st.K @ q) / TEMP
    says = (En @ st.V.T).argmax(0)                       # each slot's symbol
    right = says == ids[0]
    share = float(torch.softmax(sims, 0)[right].sum()) if bool(right.any()) else 0.0
    wrong_best = float(sims[~right].max()) if bool((~right).any()) else -1e9
    own = float(sims[onset_slot]) if 0 <= onset_slot < st.n() else -1e9
    margin = own - wrong_best
    hit = int(sym == ids[0]); chain = [hit]
    for j, i in enumerate(ids[:-1]):
        world.ctx.rest(); world.ctx.take_own(i)
        sym, conf, win, q = world.read()
        chain.append(int(sym == ids[j + 1]))
    return hit, sum(chain) / len(chain), margin, share

for form in forms:
    cf, lam, swap = form.split(":"); lam = float(lam)
    world = World(Ctx(cf, lam, swap)); onset = {}
    for text, who in material:
        rec = []
        world.hear(text, who, record=rec)
        if who == "other" and text in {a for _, a in facts} and text not in onset and rec:
            onset[text] = rec[0]
    n0 = world.st.n()
    res_t, res_r = [], []
    for q, a in facts:
        res_t.append(ask(world, q, a, onset.get(a, -1)))
        if a in reph_of:
            res_r.append(ask(world, reph_of[a], a, onset.get(a, -1)))
    def summ(res):
        if not res: return "-"
        hits = sum(r[0] for r in res); chain = sum(r[1] for r in res) / len(res); marg = sorted(r[2] for r in res)[len(res) // 2]
        wins = sum(1 for r in res if r[2] > 0); share = sum(r[3] for r in res) / len(res)
        return f"onset {hits}/{len(res)} chain {chain:.2f} margin(med) {marg:+.1f} nats, own slot wins {wins}/{len(res)}, right-symbol share {share:.2f}"
    print(f"FORM ctx {cf} key_ctx {lam} swap {swap} | store {n0} slots | TAUGHT: {summ(res_t)} | REPHRASED: {summ(res_r)}", flush=True)
    shown = " ".join(f"{a.split()[0]}:{'*' if r[0] else '.'}{'+' if r[2] > 0 else '-'}" for (q, a), r in zip(facts, res_t))
    print(f"   per fact (onset hit * / own slot wins +): {shown}", flush=True)
