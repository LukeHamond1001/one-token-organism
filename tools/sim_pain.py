"""PAIN UNDER BABBLE AND THE NEWBORN'S WITHDRAWAL (docs/SIM_DESIGN.md 6, A12, C5, C18 and C22; the build plan's W1, then W4): an
instrument, never the body. The G1 babbles (tools/sim_babble.py) with the withdrawal on (body/sim/reflexes.py: the newborn's
generalized flexion, each limb's act replaced by its flexion step for 2 ticks when the frame shows pain on any of its zones, as the
core's hook would; the grasp at the spinal cord, the world's), for a few seeds. It writes down:
  the pain ticks (any zone over F_pain), and which contact pair carried the force on each pained zone (the pair with the largest
  mean force on that zone over the tick);
  C22 AS THE NEWBORN'S (the owner's decision, made for him on 2026-09-24: C22 is written down at birth, not a bar the born reflex
  must pass; it is expected to fall only if a learned mechanism tunes the reflex, an open design item): at each withdrawal onset,
  from the same saved world, the reflex's two ticks lived three ways: the withdrawal (the reflex live on the twin, as the body
  does), the limb at rest (tone) and the babble's own acts (the chance comparisons), the other limbs taking the babble's acts in
  all three; the largest pain force (10 ms mean) on the zones it answered in each tick. A withdrawal RAISES THE PAIN IT ANSWERS
  (C22's question, counted as the W1 verifier counted it) when either of its two ticks presses those zones harder than the force
  that set it off; the first tick and the second are also counted apart (in the first the step starts: against a contact close to
  the joint, as the hip's housings against the pelvis, any change of torque can press it for a few ms, and a pressed housing that
  slides is pressed harder by the elliptic cone's friction coupling: SIM_DESIGN.md C5, 5.1; the second shows where the move took
  the limb), and the onsets still in pain at the second tick. Each onset is labelled with what its zones were pressed by (another
  zone of the G1: self-contact; else the world);
  C5 (phantom pain): the ticks in pain by the 10 ms filter and those with a single 2 ms step over F_pain, the still ticks (every
  effector at rest) and the pain on them; and the G1 as born left at rest for as many ticks, its pain and its largest force.
The skin's blind spots are A12's (the pairs pressing at rest: none for the G1 as born, body/sim/world.py rest_blind), so every
self-contact is felt. Run: nice -n 19 python3 tools/sim_pain.py [--ticks 400] [--seeds 29:1234,1:5,2:7,3:11,4:13,5:17,6:19,7:23]
[--p-rest 0.45] [--impratio 10]  (JSON on stdout; nothing written; --impratio sets the solver's for an instrument's comparison, the
world's own is the room's)"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from body.sim import reflexes as R  # noqa: E402
from body.sim import world as W  # noqa: E402
from sim_babble import Babbler  # noqa: E402


class PairWorld(W.G1World):
    """the world, recording each step's normal force per (zone, what pressed it) while `pairs_on`"""
    pairs_on = False
    steps = None

    def _zone_forces(self):
        out = super()._zone_forces()
        if self.pairs_on:
            d, m = self.d, self.m
            rec = collections.defaultdict(float)
            for i in range(d.ncon):
                c = d.contact[i]
                if c.efc_address < 0:
                    continue
                g1, g2 = int(c.geom[0]), int(c.geom[1])
                if self.blind[m.geom_bodyid[g1], m.geom_bodyid[g2]]:
                    continue
                for ga, gb in ((g1, g2), (g2, g1)):
                    z = self.zone_of_geom[ga]
                    if z >= 0:
                        zb = self.zone_of_geom[gb]
                        other = self.zones[zb] if zb >= 0 else (m.geom(gb).name or m.body(int(m.geom_bodyid[gb])).name or "world")
                        rec[(self.zones[z], other)] += float(d.efc_force[c.efc_address])
            self.steps.append(rec)
        return out


def run(wseed, bseed, ticks, p_rest, impratio=None):
    w, twin = PairWorld(seed=wseed), PairWorld(seed=wseed)
    if impratio is not None:
        w.m.opt.impratio = twin.m.opt.impratio = float(impratio)
    b = Babbler(seed=bseed, p_rest=p_rest)
    rf, rf2 = R.Reflexes(w), R.Reflexes(twin)
    st = {limb: {} for limb in R.LIMBS}
    pain_ticks, pairs, onsets = 0, collections.Counter(), []
    ph = collections.Counter()                                          # C5: pain by the filter and by a single step; the still ticks

    def two(blob, seq, zs, limb=None, state=None):
        twin.load_state(blob); out = []
        for a in seq:
            a = dict(a)
            if limb is not None:                                        # the withdrawal live on the twin
                g = rf2.withdrawal(twin.frame(), limb, state)
                a[limb] = g if g is not None else W.EFFECTOR_REST[limb]
            twin.apply(a); out.append(float(max(twin.frame().truth["pain_N"][z] for z in zs)))
        return out
    last_pairs = {}
    for _ in range(ticks):
        f = w.frame()
        pz = set(np.nonzero(f.obs["pain"])[0].tolist())
        pain_ticks += bool(pz)
        bst = b.state()
        acts = b.acts()
        new = []
        for limb in R.LIMBS:
            was = st[limb].get("withdraw", 0)
            a = rf.withdrawal(f, limb, st[limb])
            if a is not None:
                if was == 0:
                    new.append((limb, sorted(z for z in pz if z in rf.limb_zones[limb]), a))
                acts[limb] = a
        if new:
            blob = w.save_state()
            b2 = Babbler(seed=bseed, p_rest=p_rest); b2.load(bst)
            nxt = [b2.acts(), b2.acts()]
            for limb, zs, a in new:
                rest = W.EFFECTOR_REST[limb]
                by = sorted({last_pairs.get(z, "?") for z in zs})
                onsets.append({"limb": limb, "zones": [w.zones[z] for z in zs], "by": by,
                               "self": all(x in w.zones for x in by),
                               "at": float(max(f.truth["pain_N"][z] for z in zs)),
                               "withdraw": two(blob, nxt, zs, limb, {}),
                               "rest": two(blob, [dict(x, **{limb: rest}) for x in nxt], zs),
                               "babble": two(blob, nxt, zs)})
        w.pairs_on = True; w.steps = []
        w.apply(acts)
        w.pairs_on = False
        g = w.frame()
        still = all(int(acts.get(n, W.EFFECTOR_REST[n])) == W.EFFECTOR_REST[n] for n in W.EFFECTOR_REST)
        ph["ticks"] += 1; ph["still_ticks"] += still
        ph["pain (the 10 ms filter)"] += bool(g.obs["pain"].any())
        ph["a single 2 ms step over F_pain"] += bool((g.truth["peak_N"] > w.f_pain).any())
        ph["pain on still ticks"] += still and bool(g.obs["pain"].any())
        last_pairs = {}
        for z in np.nonzero(g.obs["pain"])[0]:
            tot = collections.defaultdict(float)
            for rec in w.steps:
                for (za, other), fn in rec.items():
                    if za == w.zones[z]:
                        tot[other] += fn / len(w.steps)
            top = max(tot, key=tot.get) if tot else "?"
            last_pairs[int(z)] = top
            pairs[(w.zones[z], top)] += 1
    return pain_ticks, pairs, onsets, ph


def still(ticks, impratio=None, wseed=1):
    """C5 at rest: the G1 as born, every effector at rest for `ticks` ticks: the ticks in pain (the 10 ms filter) and those with a
    single 2 ms step over F_pain"""
    w = W.G1World(seed=wseed)
    if impratio is not None:
        w.m.opt.impratio = float(impratio)
    out = collections.Counter()
    for _ in range(ticks):
        w.apply({})
        g = w.frame()
        out["ticks"] += 1
        out["pain (the 10 ms filter)"] += bool(g.obs["pain"].any())
        out["a single 2 ms step over F_pain"] += bool((g.truth["peak_N"] > w.f_pain).any())
        out["largest 10 ms force N"] = max(out["largest 10 ms force N"], round(float(g.truth["pain_N"].max())))
    return dict(out)


def c22_counts(oo, f_pain):
    """C22 over withdrawal onsets (each {"at": the onset's force, "withdraw" / "rest" / "babble": the answered zones' force in the
    reflex's two ticks}): per form, how many raise the pain they answer (either tick above the onset: C22's question), each tick
    apart, the onsets still in pain at the second tick, and the second tick's median force; and how often the withdrawal presses
    harder than rest (either tick)"""
    out = {"onsets": len(oo)}
    for k in ("withdraw", "rest", "babble"):
        out[k] = {"raises_the_pain_it_answers (either tick above the onset)": sum(max(o[k]) > o["at"] for o in oo),
                  "first_tick_above_onset": sum(o[k][0] > o["at"] for o in oo),
                  "second_tick_above_onset": sum(o[k][1] > o["at"] for o in oo),
                  "still_in_pain_at_the_second_tick": sum(o[k][1] > f_pain for o in oo),
                  "second_tick_median_N": round(float(np.median([o[k][1] for o in oo]))) if oo else None}
    out["withdraw_above_rest (either tick)"] = sum(max(o["withdraw"]) > max(o["rest"]) for o in oo)
    return out


def main(ticks=400, seeds=((29, 1234), (1, 5), (2, 7), (3, 11), (4, 13), (5, 17), (6, 19), (7, 23)), p_rest=0.45, impratio=None):
    pain, pairs, onsets, ph = 0, collections.Counter(), [], collections.Counter()
    f_pain = W.G1World(seed=1).f_pain
    for ws, bs in seeds:
        p, pr, on, c = run(ws, bs, ticks, p_rest, impratio)
        pain += p; pairs.update(pr); onsets += on; ph.update(c)
    rnd = lambda v: [rnd(x) for x in v] if isinstance(v, list) else (round(v) if isinstance(v, float) else v)
    return {"ticks": ticks * len(seeds), "seeds": [list(s) for s in seeds], "p_rest": p_rest, "blind_spots": "A12's rest list (none as born)",
            "impratio": impratio if impratio is not None else "the room's", "withdrawal": "the newborn's generalized flexion",
            "pain_ticks": pain, "pain_share": round(pain / (ticks * len(seeds)), 4), "phantom (C5)": dict(ph), "at rest (C5)": still(ticks, impratio),
            "pain_pairs": {f"{a} <- {b}": k for (a, b), k in pairs.most_common(16)},
            "C22": dict(c22_counts(onsets, f_pain), by_limb=dict(collections.Counter(o["limb"] for o in onsets)),
                        self_contact=c22_counts([o for o in onsets if o["self"]], f_pain),
                        world_contact=c22_counts([o for o in onsets if not o["self"]], f_pain),
                        raised=[{k: rnd(v) for k, v in o.items()} for o in onsets if max(o["withdraw"]) > o["at"]])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--seeds", default="29:1234,1:5,2:7,3:11,4:13,5:17,6:19,7:23")
    ap.add_argument("--p-rest", type=float, default=0.45)
    ap.add_argument("--impratio", type=float, default=None)
    a = ap.parse_args()
    seeds = tuple(tuple(int(x) for x in s.split(":")) for s in a.seeds.split(","))
    print(json.dumps(main(a.ticks, seeds, a.p_rest, a.impratio), indent=1))
