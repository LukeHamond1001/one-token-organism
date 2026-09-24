"""THE BABBLER: smooth random acts for the simulated world's tests and measurements (docs/SIM_DESIGN.md 1, 3.6 and 11: W1, W4).
An instrument of the world, never the body: no weight of the body ever learns from it, and it is never given to the body. It
stands in for a body's acts: each joint effector (body/sim/world.py EFFECTOR_FACTORS) runs in movement units, as infants' moves
come (von Hofsten; SIM_DESIGN.md 3.6), so the servo targets move smoothly rather than as white noise: at a unit's start the
effector rests with probability p_rest (the design's sparse babble: 60% of the units rests) or draws each joint's setting
uniformly, and holds that act for a unit of 1-8 ticks (uniform); an act re-anchors each target at the measured angle plus its step
every tick, so a held step is a steady ramp. Its own stream is a numpy PCG64 from SeedSequence(seed, spawn_key=(2,)), saved
with `state()` and put back with `load(state)`, so a test that saves the world saves the babbler beside it.

Run (the speed of the world at the served tick, on this Mac, under babble; JSON on stdout):
  nice -n 19 python3 tools/sim_babble.py --ticks 400 [--seed 1] [--p-rest 0.6] [--rest-ticks 40] [--eyes [--shadows sun|none|all]]
It reports the wall ms of a tick (the world's apply: the physics' mj_step alone, and the world's own Python: the servo law, the
touch zones, pain's filter, the IMUs, the charge), of frame() (the senses read out; with --eyes the eyes' render, its split into
periphery and fovea, the retina's code and the face test, W3),
with the real-time factor (150 ms of sim time a tick), MuJoCo's contacts, the tick's largest 10 ms force against F_pain, the pain
ticks and the charge's drain."""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # this tree's body
from body.sim.world import EFFECTOR_FACTORS, EFFECTOR_REST  # noqa: E402

BABBLE_STREAM = 2               # the babbler's own stream: SeedSequence(seed, spawn_key=(2,)) (an instrument's; ours)
UNIT_TICKS = (1, 8)             # a movement unit's length in ticks, uniform (SIM_DESIGN.md 3.6: "each joint holding its step for 1-8 ticks")


class Babbler:
    """smooth random acts: per effector, movement units of 1-8 ticks, each a rest (p_rest) or one uniformly drawn act held"""

    def __init__(self, seed=0, p_rest=0.6, factors=None, units=UNIT_TICKS):
        self.factors = dict(EFFECTOR_FACTORS if factors is None else factors)
        self.p_rest = float(p_rest); self.units = tuple(units)
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(BABBLE_STREAM,))))
        self.now = {n: None for n in self.factors}; self.left = {n: 0 for n in self.factors}

    def acts(self):
        """this tick's acts, {effector: flat act} (a resting effector gives its rest)"""
        out = {}
        for n, fs in self.factors.items():
            if self.left[n] <= 0:
                self.left[n] = int(self.rng.integers(self.units[0], self.units[1] + 1))
                if self.rng.random() < self.p_rest:
                    self.now[n] = None
                else:
                    a = 0
                    for k in fs:
                        a = a * int(k) + int(self.rng.integers(0, int(k)))
                    self.now[n] = a
            self.left[n] -= 1
            out[n] = EFFECTOR_REST[n] if self.now[n] is None else self.now[n]
        return out

    def state(self):
        return {"rng": self.rng.bit_generator.state, "now": dict(self.now), "left": dict(self.left)}

    def load(self, st):
        self.rng.bit_generator.state = st["rng"]; self.now = dict(st["now"]); self.left = dict(st["left"])


def measure(ticks, seed=1, p_rest=0.6, rest_ticks=40, eyes=False, shadows="sun"):
    """the world's speed on this Mac: rest_ticks at rest, then ticks under babble (with the eyes: frame() renders them)"""
    from body.sim.world import G1World
    t0 = time.perf_counter(); w = G1World(seed=seed); born_s = time.perf_counter() - t0
    ey = None
    if eyes:
        from body.sim.eyes import Eyes
        ey = Eyes(w, shadows=shadows)
    b = Babbler(seed=seed, p_rest=p_rest)
    rows = []
    for phase, n in (("rest", rest_ticks), ("babble", ticks)):
        ap, ph, fr, rd = [], [], [], []
        pain_ticks = 0; peak = 0.0; ncon = []; h0 = w.h
        for _ in range(n):
            acts = {} if phase == "rest" else b.acts()
            p0 = w.timing["physics_s"]; r0 = ey.timing["render_s"] if ey is not None else 0.0; t1 = time.perf_counter()
            w.apply(acts)
            t2 = time.perf_counter()
            f = w.frame()
            t3 = time.perf_counter()
            if ey is not None:
                rd.append(ey.timing["render_s"] - r0)
            ap.append(t2 - t1); ph.append(w.timing["physics_s"] - p0); fr.append(t3 - t2)
            pain_ticks += int(f.obs["pain"].sum() > 0); peak = max(peak, float(f.truth["pain_N"].max())); ncon.append(f.truth["ncon"])
        ms = lambda xs: round(1e3 * statistics.fmean(xs), 2) if xs else None
        tick_ms = [1e3 * (a + c) for a, c in zip(ap, fr)]
        row = {"phase": phase, "ticks": n, "tick_ms_mean": round(statistics.fmean(tick_ms), 2),
               "tick_ms_median": round(statistics.median(tick_ms), 2),
               "tick_ms_p95": round(sorted(tick_ms)[int(0.95 * (len(tick_ms) - 1))], 2),
               "apply_ms": ms(ap), "physics_ms": ms(ph), "world_python_ms": round(ms(ap) - ms(ph), 2), "frame_ms": ms(fr),
               "eyes_render_ms": ms(rd), "eyes_split_ms": round(ms(fr) - ms(rd), 2) if rd else None, "realtime_x_physics": round(150.0 / ms(ph), 1), "realtime_x_world": round(150.0 / statistics.fmean(tick_ms), 1),
               "contacts_mean": round(statistics.fmean(ncon), 1), "largest_10ms_force_N": round(peak, 1), "f_pain_N": round(w.f_pain, 1),
               "pain_ticks": pain_ticks, "charge_drain_per_tick": round((h0 - w.h) / n, 7)}
        rows.append(row)
    load = os.getloadavg()[0]
    return {"born_s": round(born_s, 2), "seed": seed, "p_rest": p_rest, "eyes": bool(eyes), "shadows": shadows if eyes else None,
            "load_avg_1min": round(load, 2), "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--p-rest", type=float, default=0.6)
    ap.add_argument("--rest-ticks", type=int, default=40)
    ap.add_argument("--eyes", action="store_true")
    ap.add_argument("--shadows", default="sun", choices=("sun", "all", "none"))
    a = ap.parse_args()
    print(json.dumps(measure(a.ticks, a.seed, a.p_rest, a.rest_ticks, a.eyes, a.shadows), indent=1))
