"""PAIN UNDER BABBLE AND THE WITHDRAWAL'S EFFECT (docs/SIM_DESIGN.md 6, A12, C5, C18 and C22; the build plan's W1, then W4): an
instrument, never the body. The G1 babbles (tools/sim_babble.py) with the withdrawal on (body/sim/reflexes.py: each limb's act
replaced by the withdrawal's for 2 ticks when the frame shows pain on it, as the core's hook would; the grasp at the spinal cord,
the world's), for a few seeds. It writes down:
  the pain ticks (any zone over F_pain), and which contact pair carried the force on each pained zone (the pair with the largest
  mean force on that zone over the tick);
  C22 at each withdrawal onset, from the same saved world, the two ticks lived three ways: the withdrawal (the reflex live on the
  twin, as the body does), the limb at rest (tone) and the babble's own acts, the other limbs taking the babble's acts in all
  three; the largest pain force (10 ms mean) on the zones it answered in each tick. Split by tick, since they are different
  things: in the reflex's FIRST tick its own step starts (the servo's torque steps within a physics step; against a contact close
  to the joint, as the hip's housings against the pelvis, any change of torque, a rest's included, can press it for a few ms
  first); its SECOND tick shows where the move took the limb: a withdrawal "drives the limb into a worse contact" (C22's question)
  when its second tick's force passes the force that set it off. Also counted: the onsets still in pain at the second tick, and
  those whose force passed the rest's there. Each onset is also labelled with what its zones were pressed by (another zone of the
  G1: self-contact; else the world) and whether the reflex found a direction (its act not the rest).
The skin's blind spots are A12's (the pairs pressing at rest: none for the G1 as born, body/sim/world.py rest_blind), so every
self-contact is felt. Run: nice -n 19 python3 tools/sim_pain.py [--ticks 400] [--seeds 29:1234,1:5,2:7,3:11,4:13,5:17,6:19,7:23]
[--p-rest 0.45]  (JSON on stdout; nothing written)"""
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

    def _zone_forces(self, sites=None):
        out = super()._zone_forces(sites)
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


def run(wseed, bseed, ticks, p_rest):
    w, twin = PairWorld(seed=wseed), PairWorld(seed=wseed)
    b = Babbler(seed=bseed, p_rest=p_rest)
    rf, rf2 = R.Reflexes(w), R.Reflexes(twin)
    st = {limb: {} for limb in R.LIMBS}
    pain_ticks, pairs, onsets = 0, collections.Counter(), []

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
                               "self": all(x in w.zones for x in by), "direction": a != rest,
                               "at": float(max(f.truth["pain_N"][z] for z in zs)),
                               "withdraw": two(blob, nxt, zs, limb, {}),
                               "rest": two(blob, [dict(x, **{limb: rest}) for x in nxt], zs),
                               "babble": two(blob, nxt, zs)})
        w.pairs_on = True; w.steps = []
        w.apply(acts)
        w.pairs_on = False
        g = w.frame()
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
    return pain_ticks, pairs, onsets


def main(ticks=400, seeds=((29, 1234), (1, 5), (2, 7), (3, 11), (4, 13), (5, 17), (6, 19), (7, 23)), p_rest=0.45):
    pain, pairs, onsets = 0, collections.Counter(), []
    f_pain = W.G1World(seed=1).f_pain
    for ws, bs in seeds:
        p, pr, on = run(ws, bs, ticks, p_rest)
        pain += p; pairs.update(pr); onsets += on

    def c22(oo):
        out = {"onsets": len(oo)}
        for k in ("withdraw", "rest", "babble"):
            out[k] = {"first_tick_above_onset": sum(o[k][0] > o["at"] for o in oo),
                      "second_tick_above_onset (into a worse contact)": sum(o[k][1] > o["at"] for o in oo),
                      "still_in_pain_at_the_second_tick": sum(o[k][1] > f_pain for o in oo),
                      "second_tick_median_N": round(float(np.median([o[k][1] for o in oo]))) if oo else None}
        out["withdraw_second_tick_above_rest"] = sum(o["withdraw"][1] > o["rest"][1] for o in oo)
        return out
    rnd = lambda v: [rnd(x) for x in v] if isinstance(v, list) else (round(v) if isinstance(v, float) else v)
    return {"ticks": ticks * len(seeds), "seeds": [list(s) for s in seeds], "p_rest": p_rest, "blind_spots": "A12's rest list (none as born)",
            "pain_ticks": pain, "pain_share": round(pain / (ticks * len(seeds)), 4),
            "pain_pairs": {f"{a} <- {b}": k for (a, b), k in pairs.most_common(16)},
            "C22": dict(c22(onsets), by_limb=dict(collections.Counter(o["limb"] for o in onsets)),
                        self_contact=c22([o for o in onsets if o["self"]]), world_contact=c22([o for o in onsets if not o["self"]]),
                        no_direction_rest=c22([o for o in onsets if not o["direction"]]),
                        into_a_worse_contact_or_still_in_pain=[{k: rnd(v) for k, v in o.items()} for o in onsets
                                                              if o["withdraw"][1] > o["at"] or o["withdraw"][1] > f_pain])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--seeds", default="29:1234,1:5,2:7,3:11,4:13,5:17,6:19,7:23")
    ap.add_argument("--p-rest", type=float, default=0.45)
    a = ap.parse_args()
    seeds = tuple(tuple(int(x) for x in s.split(":")) for s in a.seeds.split(","))
    print(json.dumps(main(a.ticks, seeds, a.p_rest), indent=1))
