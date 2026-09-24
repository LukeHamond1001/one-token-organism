"""PAIN UNDER BABBLE AND THE WITHDRAWAL'S EFFECT (docs/SIM_DESIGN.md 6, A12, C5, C18 and C22; the build plan's W1, then W4): an
instrument, never the body. The G1 babbles (tools/sim_babble.py) with the withdrawal on (each limb's act replaced by its flexion
for 2 ticks when the frame shows pain on it, as the core's hook would; the grasp at the spinal cord, the world's), for a few seeds.
It writes down:
  the pain ticks (any zone over F_pain), and which contact pair carried the force on each pained zone (the pair with the largest
  mean force on that zone over the tick);
  C22 at each withdrawal onset, from the same saved world, the two ticks lived three ways: the withdrawal (as the body does), the
  limb at rest (tone) and the babble's own acts; the largest pain force on the zones it answered in each. A withdrawal "raises the
  pain it answers" when its force passes the force that set it off; "worse than rest" when it passes the rest's.
--no-blind switches off the skin's blind spots inside the compound joints (body/sim/world.py joint_blind; A12), to show what they
remove. Run: nice -n 19 python3 tools/sim_pain.py [--ticks 400] [--seeds 29:1234,1:5,2:7,3:11,4:13,5:17,6:19,7:23] [--p-rest 0.45]
[--no-blind]  (JSON on stdout; nothing written)"""
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


def run(wseed, bseed, ticks, p_rest, blind=True):
    w, twin = PairWorld(seed=wseed), PairWorld(seed=wseed)
    if not blind:
        w.blind[:] = False; twin.blind[:] = False
    b = Babbler(seed=bseed, p_rest=p_rest)
    rf = R.Reflexes(w.zones); st = {limb: {} for limb in R.FLEXION}
    pain_ticks, pairs, onsets = 0, collections.Counter(), []

    def two(blob, seq, zs):
        twin.load_state(blob); out = []
        for a in seq:
            twin.apply(a); out.append(max(twin.frame().truth["pain_N"][z] for z in zs))
        return max(out)
    for _ in range(ticks):
        f = w.frame()
        pz = set(np.nonzero(f.obs["pain"])[0].tolist())
        pain_ticks += bool(pz)
        bst = b.state()
        acts = b.acts()
        new = []
        for limb in R.FLEXION:
            was = st[limb].get("withdraw", 0)
            a = rf.withdrawal(f, limb, st[limb])
            if a is not None:
                acts[limb] = a
                if was == 0:
                    new.append((limb, sorted(z for z in pz if z in rf.limb_zones[limb])))
        if new:
            blob = w.save_state()
            b2 = Babbler(seed=bseed, p_rest=p_rest); b2.load(bst)
            nxt = [b2.acts(), b2.acts()]
            for limb, zs in new:
                fx, rest = R.flexion_act(limb), W.EFFECTOR_REST[limb]
                onsets.append({"limb": limb, "zones": [w.zones[z] for z in zs], "at": float(max(f.truth["pain_N"][z] for z in zs)),
                               "withdraw": float(two(blob, [dict(x, **{limb: fx}) for x in nxt], zs)),
                               "rest": float(two(blob, [dict(x, **{limb: rest}) for x in nxt], zs)),
                               "babble": float(two(blob, nxt, zs))})
        w.pairs_on = True; w.steps = []
        w.apply(acts)
        w.pairs_on = False
        g = w.frame()
        for z in np.nonzero(g.obs["pain"])[0]:
            tot = collections.defaultdict(float)
            for rec in w.steps:
                for (za, other), fn in rec.items():
                    if za == w.zones[z]:
                        tot[other] += fn / len(w.steps)
            pairs[(w.zones[z], max(tot, key=tot.get) if tot else "?")] += 1
    return pain_ticks, pairs, onsets


def main(ticks=400, seeds=((29, 1234), (1, 5), (2, 7), (3, 11), (4, 13), (5, 17), (6, 19), (7, 23)), p_rest=0.45, blind=True):
    pain, pairs, onsets = 0, collections.Counter(), []
    for ws, bs in seeds:
        p, pr, on = run(ws, bs, ticks, p_rest, blind)
        pain += p; pairs.update(pr); onsets += on
    n = len(onsets)
    return {"ticks": ticks * len(seeds), "seeds": [list(s) for s in seeds], "p_rest": p_rest, "blind_spots": blind,
            "pain_ticks": pain, "pain_share": round(pain / (ticks * len(seeds)), 4),
            "pain_pairs": {f"{a} <- {b}": k for (a, b), k in pairs.most_common(16)},
            "C22": {"onsets": n, "by_limb": dict(collections.Counter(o["limb"] for o in onsets)),
                    "withdrawal_raised_the_pain": sum(o["withdraw"] > o["at"] for o in onsets),
                    "rest_raised_it": sum(o["rest"] > o["at"] for o in onsets), "babble_raised_it": sum(o["babble"] > o["at"] for o in onsets),
                    "withdrawal_above_rest": sum(o["withdraw"] > o["rest"] for o in onsets),
                    "withdrawal_raised_it_and_above_rest": sum(o["withdraw"] > o["at"] and o["withdraw"] > o["rest"] for o in onsets),
                    "those": [{k: (round(v) if isinstance(v, float) else v) for k, v in o.items()} for o in onsets
                              if o["withdraw"] > o["at"] and o["withdraw"] > o["rest"]]}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--seeds", default="29:1234,1:5,2:7,3:11,4:13,5:17,6:19,7:23")
    ap.add_argument("--p-rest", type=float, default=0.45)
    ap.add_argument("--no-blind", action="store_true")
    a = ap.parse_args()
    seeds = tuple(tuple(int(x) for x in s.split(":")) for s in a.seeds.split(","))
    print(json.dumps(main(a.ticks, seeds, a.p_rest, not a.no_blind), indent=1))
