"""WATCHING ONE LIFE, EVERY MOVE: a body lives day after day under the fast parent; every tick is written to a jsonl (what it
heard and said, the reward, the dopamine, the fast value, the gate, the working-memory slot, the planner's choices) and at
each day's end a digest is printed and appended to a digest file: smiles, cue completions, sequences, turnings-away, frowns,
the gate's ears (speaking while the parent types vs quiet), the rise of the value before a smile, the working memory's
latches, the planner's agreements with the cortex. The body is saved after every day.
    python3 tools/watch_life.py BODY.pt --days N [--birth] [--seed S] [--cfg-from X.pt] [--set k=v ...] [--d D --layers L --heads H]
The parent's environment comes from the usual variables (PARENT, REPLY, WAIT, TALKOVER_FROWN, WORD_SMILES, ROOM, ...).
"""
import os, sys, json, time, random, argparse, statistics as st, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from body.life import Life
from body.fastlife import FastCaregiver

ap = argparse.ArgumentParser(); ap.add_argument("body"); ap.add_argument("--days", type=int, default=1); ap.add_argument("--birth", action="store_true")
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--cfg-from", default=None); ap.add_argument("--set", action="append", default=[])
ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=4); ap.add_argument("--window", type=int, default=64)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--out", default=None)
a = ap.parse_args(); torch.set_num_threads(a.threads)
tok = Tokenizer.from_file("data/tok_char.json")
cfg = dict(torch.load(a.cfg_from, map_location="cpu", weights_only=False).get("cfg") or {}) if a.cfg_from else {}
for kv in a.set:
    k, v = kv.split("=", 1)
    try: cfg[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    except ValueError: cfg[k] = v
if a.birth:
    L = Life.birth(tok, device="cpu", d=a.d, layers=a.layers, heads=a.heads, window=a.window, cfg=cfg or None, seed=a.seed, save_path=a.body); L.save()
else:
    L = Life.load(a.body, tok, device="cpu", cfg=cfg or None, save_path=a.body)
name = os.path.basename(a.body).replace(".pt", ""); out = a.out or f"data/watch/{name}"; os.makedirs(out, exist_ok=True)
m = L.m; rng = random.Random(a.seed)
ENV = {k: os.environ.get(k, "") for k in ("PARENT", "REPLY", "WAIT", "TALKOVER_FROWN", "WORD_SMILES", "ROOM", "FIRST_LETTER_BIAS", "ANSWER_LEVELS", "REPLY_QUIET")}
with open(f"{out}/digest.txt", "a") as f: f.write(f"# {time.strftime('%Y-%m-%d %H:%M')} start at day {L.day_n + 1}: env {ENV} | sets {a.set} | arch d {L.m.d} layers {len(L.m.blocks)}\n")
print(f"env {ENV}")
print(f"watching {name}: day {L.day_n + 1} onward, {a.days} days | cfg {{{', '.join(f'{k}={cfg[k]}' for k in sorted(cfg) if k in [x.split('=')[0] for x in a.set])}}}", flush=True)
for _ in range(a.days):
    day = L.day_n + 1; ticks = []; rows = []; cg_ref = []
    orig_tick = L.tick
    def tick():
        typing = len(L.queue) > 0; wm0 = float(getattr(m, "wm_on", torch.zeros(())))
        L._plan_last = None
        orig_tick()
        la = L.last; wm1 = float(getattr(m, "wm_on", torch.zeros(()))); pl = getattr(L, "_plan_last", None)
        ticks.append({"t": L.ticks, "typing": typing, "said": la.get("said", ""), "felt": la.get("felt", 0), "gate": la.get("gate"), "dopa": la.get("dopamine"),
                      "vf": L.fast_value(), "r": la.get("r"), "vlong": la.get("vlong"), "e": cg_ref[0].e if cg_ref else None, "away": (cg_ref[0].away_until > L.ticks) if cg_ref else False,
                      "wm": wm1, "wm_latch": wm1 > wm0, "fatigue": la.get("fatigue"),
                      "plan": ({"cands": [tok.decode([c]) for c in pl["cands"]], "vals": [round(pl["vals"][c], 3) for c in pl["cands"]], "cortex": [round(pl["cortex"][c], 2) for c in pl["cands"]]} if pl else None)})
    L.tick = tick
    cg = FastCaregiver(L, day, [], rng); cg_ref.append(cg); orig_row = cg.row
    def row(obj):
        obj = dict(obj); obj["t"] = L.ticks; rows.append(obj); return orig_row(obj)
    cg.row = row
    t0 = time.time(); night = cg.run_day(); L.tick = orig_tick; L.save()
    with open(f"{out}/day{day:03d}_ticks.jsonl", "w") as f:
        for x in ticks: f.write(json.dumps(x) + "\n")
    with open(f"{out}/day{day:03d}_rows.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x, default=str) + "\n")
    # the digest
    byt = {x["t"]: x for x in ticks}; ts = sorted(byt)
    def spoke(sel):
        xs = [byt[t] for t in sel if t in byt]; return (st.mean(1.0 if (x.get("said") or "") != "" else 0.0 for x in xs) if len(xs) >= 20 else float("nan")), len(xs)
    typ = [t for t in ts if byt[t]["typing"]]; quiet = [t for t in ts if not byt[t]["typing"]]
    sm = [r["t"] for r in rows if r.get("action") == "smile"]; comp = sum(1 for r in rows if r.get("action") == "smile" and "completion" in str(r.get("why")))
    vf = [x["vf"] for x in ticks if x.get("vf") is not None]; mv = st.mean(vf) if vf else 0.0
    pre = [byt[t - 1]["vf"] - mv for t in sm if t - 1 in byt]; far = [byt[t - 8]["vf"] - mv for t in sm if t - 8 in byt]
    rise = (st.mean(pre) - st.mean(far)) / 2 * 100 if pre and far else float("nan")
    err = [byt[t + 1]["dopa"] for t in sm if t + 1 in byt and byt[t + 1].get("dopa") is not None]
    latches = sum(1 for x in ticks if x["wm_latch"]); plans = [x for x in ticks if x["plan"]]
    # the planner's flips: decisions where the symbol said was not the cortex's own favorite among the candidates (review: "agreed" was a coin)
    other = sum(1 for x in plans if x.get("said") and x["said"] in x["plan"]["cands"] and x["plan"]["cands"].index(x["said"]) != x["plan"]["cortex"].index(max(x["plan"]["cortex"])))
    beta_ = float(L.cfg.get("plan_beta", 4.0))                                 # the value's own flips: where it moved the favorite itself
    flips = sum(1 for x in plans if max(range(len(x["plan"]["cands"])), key=lambda i: x["plan"]["cortex"][i] + beta_ * x["plan"]["vals"][i]) != x["plan"]["cortex"].index(max(x["plan"]["cortex"])))
    aways = sum(1 for r in rows if r.get("action") == "away"); frowns = sum(1 for r in rows if r.get("action") == "frown"); seqs = sum(1 for r in rows if r.get("action") == "sequence")
    st_typ, n_typ = spoke(typ); st_q, _ = spoke(quiet)
    held = [r for r in rows if r.get("action") == "smile" and r.get("run_on") is not None]
    run_on = st.mean(r["run_on"] for r in held) if held else float("nan"); waited = st.mean(r.get("waited") or 0 for r in held) if held else float("nan")
    # THE PREFRONTAL BANDS: the long critic's value against the return it later realized (1024 ticks, discounted), its measured
    # reliability, and its effective weight in the gate's credit (the ceiling times the reliability): doing, and whether guiding
    gl = 1.0 - 1.0 / 1024.0; rr = [float(x.get("felt") or 0) for x in ticks]; vl = [x.get("vlong") for x in ticks]
    G = [0.0] * len(rr); acc = 0.0
    for i in range(len(rr) - 1, -1, -1): acc = rr[i] + gl * acc; G[i] = acc
    pairs = [(v, g) for v, g in zip(vl[: max(0, len(vl) - 1024)], G[: max(0, len(G) - 1024)]) if v is not None]
    if len(pairs) > 100:
        mv_, mg_ = st.mean(p[0] for p in pairs), st.mean(p[1] for p in pairs); sv_, sg_ = st.pstdev(p[0] for p in pairs), st.pstdev(p[1] for p in pairs)
        corr_long = (sum((p[0] - mv_) * (p[1] - mg_) for p in pairs) / len(pairs)) / (sv_ * sg_) if sv_ > 1e-9 and sg_ > 1e-9 else float("nan")
    else: corr_long = float("nan")
    vrel = float(getattr(L, "_vrel_corr", 0.0)); vslope = float(getattr(L, "_vrel_gain", 0.0))
    vw_eff = float(getattr(L, "_vw_now", float(L.cfg.get("vcrit_w", 0.0)) * (vslope if int(L.cfg.get("vcrit_auto", 0)) else 1.0)))   # the body's own weight (the slope, not the correlation: the review of 2026-09-08)
    digest = (f"day {day:3d} ({time.time()-t0:.0f}s, {len(ticks)} ticks): smiles {len(sm)} (completions {comp}, sequences {seqs}) aways {aways} frowns {frowns} | "
              f"ear: spoke {st_typ:.2f} while the parent typed (n {n_typ}) vs {st_q:.2f} quiet | fast value mean {mv:+.2f}, rise before a smile {rise:+.0f}% of a smile, "
              f"error at the reward {st.mean(err) if err else float('nan'):+.2f} | wm latches {latches} | planner: {len(plans)} choices, the value flipped the cortex's favorite {flips}, the mouth said another {other} | "
              f"prefrontal: long value vs its realized return {corr_long:+.2f}, reliability (correlation) {vrel:+.2f}, slope {vslope:.2f}, weight in the credit {vw_eff:.3f} | sharpness {float(getattr(L, "sharp_cal", L.cfg.get("sharp_base", 0))):.1f} ({L.cfg.get("sharp_form", "fixed")}) | run-on after the answer {run_on:.1f} symbols, the smile waited {waited:.1f} ticks (n {len(held)}) | gauge {night.get('gauge') if isinstance(night, dict) else ''}")
    print(digest, flush=True)
    with open(f"{out}/digest.txt", "a") as f: f.write(digest + "\n")
