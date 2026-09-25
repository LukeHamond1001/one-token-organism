"""THE BORN MOVEMENT UNITS OF THE G1, AT THE CORE (docs/SIM_DESIGN.md 3.6, C38; the core refactor's step R6h), and the cost of R6h's parts a
tick. An instrument: the G1's anatomy (body/sim/anatomy.py SimAnatomy, SIM_CFG) born at the design's core (d 512, 6 blocks, 8 heads,
window 64) with EVERY LEARNING RATE AT 0 (nothing of the body learns; the run is discarded), in a quiet stub of its world (the frame's
senses random and unit-scaled, no cue, no pain, the charge full: no physics, which W4 adds on a replica of the world and S5a on the real
core), its motor loop as born: each effector's gate at birth, the continuation draw, the persistence margin (log 4), chunk_max 8, the
spinal pattern generators on. It writes down, per effector, the movement units (a unit: an act and the continuations that held it, until
its gate's draw said no, chunk_max ended it or its proposal passed the margin to its rest), their lengths in ticks and seconds, the share
of one tick, the longest; the gate's p_act; the pattern generator's amplitude; and the tick's cost with its R6h parts (the born codes over
the window, act_inv's lessons batched every 8 ticks and, on a second body, unbatched, the cord's patterns, the orienting bias, the units'
hold). Rolls, travel and reaches need the physics: W4's.
usage: nice -n 19 python3 tools/motor_units.py [--ticks 3000] [--d 512] [--seed 1]"""
import collections
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import torch  # noqa: E402

from body.life import Life  # noqa: E402
from body.core.world import Frame, SimWorld, WorldLoop  # noqa: E402
from body.sim.anatomy import SIM_CFG, SIZES, SimAnatomy, born_table  # noqa: E402


def arg(name, default):
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--" + name and i + 1 < len(sys.argv):
            return type(default)(sys.argv[i + 1])
    return default


LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
           act_inv_lr=0.0, wake_ticks=10 ** 9)


class Quiet(SimWorld):
    """the G1's frame with every key's senses random and unit-scaled (uniform in -1..1, the breath left and the charge full), no cue, no
    pain, no word: the born loop's draws alone decide what it does"""

    def __init__(self, seed):
        self.t = 0; self.rng = random.Random(seed)

    def frame(self):
        R = self.rng
        obs = {n: [R.uniform(-1, 1) for _ in range(k)] for n, k in SIZES.items() if n not in ("face", "charge")}
        obs["body"][241] = 1.0; obs["face"] = [0.0, 0.0]; obs["charge"] = [1.0, 0.0]; obs["pain"] = [0.0] * 44
        return Frame(self.t, obs, 0.0, {})

    def apply(self, acts):
        self.t += 1

    def pause(self):
        pass

    def resume(self):
        pass

    def save_state(self):
        return b""

    def load_state(self, blob):
        pass


def born(cfg, d, seed):
    torch.manual_seed(seed)
    return Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=d, layers=6 if d >= 256 else 2, heads=8 if d >= 256 else 2,
                      window=64 if d >= 256 else 16, cfg=cfg, seed=seed, world=Quiet(seed))


def timed(L, name, box):
    f = getattr(L, name)

    def w(*a, **k):
        t0 = time.perf_counter(); out = f(*a, **k); box[name] += time.perf_counter() - t0; return out
    setattr(L, name, w)


def main():
    T = arg("ticks", 3000); d = arg("d", 512); seed = arg("seed", 1)
    torch.set_num_threads(1)
    cfg = dict(SIM_CFG, **LR0)
    L = born(cfg, d, seed); run = WorldLoop(L)
    box = collections.Counter()
    for n in ("_inverse_batch", "_cord", "_orient_bias", "_unit_hold", "_choose_effector", "_wake_lesson"):
        timed(L, n, box)
    calls = collections.Counter(); m = L.m; f_in = m.inputs

    def inputs(anatomy, obs, xos, bundles):
        calls["n"] += 1; calls["pos"] += int(xos.numel())                 # the positions encoded (a batch counts every window)
        return f_in(anatomy, obs, xos, bundles)
    m.inputs = inputs
    names = [e.name for e in L.anatomy.motors]
    units = {n: [] for n in names}; cur = {n: 0 for n in names}; pacts = {n: [] for n in names}; held = {n: 0 for n in names}
    cont_n = {n: 0 for n in names}; spg = collections.defaultdict(list); tick_s = []
    for t in range(T):
        t0 = time.perf_counter(); run.step(); tick_s.append(time.perf_counter() - t0)
        for e, st in zip(L.anatomy.motors, L.motor):
            now = st["now"]; n = e.name; pacts[n].append(now["p_act"])
            if now["acted"]:
                if now["cont"]:
                    cur[n] += 1; cont_n[n] += 1; held[n] += int(now["digits"] == prev[n])
                else:
                    if cur[n]:
                        units[n].append(cur[n])
                    cur[n] = 1
            elif cur[n]:
                units[n].append(cur[n]); cur[n] = 0
            if now.get("cord") is not None and e.spg:
                spg[n].append(max(abs(x) for x in now["cord"]))
        prev = {e.name: list(st["now"]["digits"]) for e, st in zip(L.anatomy.motors, L.motor)}
    print(f"THE BORN MOVEMENT UNITS OF THE G1 (C38), at the core: d {d}, {T} ticks, seed {seed}, every learning rate 0, a quiet world")
    print(f"{'effector':8s} {'p_act':>7s} {'units':>6s} {'mean ticks':>10s} {'mean s':>7s} {'share 1':>8s} {'max':>4s} {'held':>6s} {'spg rad':>8s}")
    allu = []
    for n in names:
        u = units[n]; allu += u
        print(f"{n:8s} {statistics.mean(pacts[n]):7.4f} {len(u):6d} {statistics.mean(u):10.3f} {0.15 * statistics.mean(u):7.3f} "
              f"{sum(1 for x in u if x == 1) / len(u):8.3f} {max(u):4d} {held[n]}/{cont_n[n]:<5d} "
              f"{(statistics.mean(spg[n]) if spg[n] else 0.0):8.4f}")
    p = statistics.mean(pacts[names[0]])
    print(f"all {len(allu)} units: mean {statistics.mean(allu):.3f} ticks ({0.15 * statistics.mean(allu):.3f} s), share of one tick "
          f"{sum(1 for x in allu if x == 1) / len(allu):.3f}, longest {max(allu)}; the law at p_act {p:.4f}: mean "
          f"{(1 - p ** 8) / (1 - p):.3f}, share of one {1 - p:.3f}")
    # the born codes' cost: every channel's code of one position, timed over a window's worth, times the positions the tick encoded
    x = {c.name: torch.randn(64, int(c.size)) for c in L.anatomy.channels[1:]}
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(50):
            for c in L.anatomy.channels[1:]:
                c.encode(m, x[c.name])
        per_pos = (time.perf_counter() - t0) / (50 * 64)
    enc_ms = 1000.0 * per_pos * calls["pos"] / T
    ts = sorted(tick_s[50:])
    print(f"THE TICK (one thread, nice 19): mean {1000 * statistics.mean(ts):.1f} ms, median {1000 * ts[len(ts) // 2]:.1f} ms over {len(ts)} ticks")
    per = lambda k: 1000.0 * box[k] / T  # noqa: E731
    print(f"R6h's parts a tick: the born codes over the window {enc_ms:.2f} ms ({calls['pos'] / T:.0f} positions encoded a tick, "
          f"{1e6 * per_pos:.1f} us each for the eight channels); act_inv batched every 8 {per('_inverse_batch'):.2f} ms; the cord's patterns "
          f"{per('_cord'):.3f} ms; the orienting bias {per('_orient_bias'):.3f} ms; the units' hold {per('_unit_hold'):.3f} ms; "
          f"the nine motor effectors' choices in all {per('_choose_effector'):.2f} ms")
    # act_inv unbatched, on a body born the same, for the design's figure (4-6 ms a tick unbatched)
    L2 = born(dict(cfg, act_inv_every=1, act_inv_chance=0), d, seed); run2 = WorldLoop(L2); box2 = collections.Counter()
    timed(L2, "_inverse_lesson", box2); timed(L2, "_inverse_batch", box2)
    T2 = min(T, 600)
    for _ in range(T2):
        run2.step()
    print(f"act_inv unbatched (a step each tick an effector acted, R6's): {1000 * (box2['_inverse_lesson'] + box2['_inverse_batch']) / T2:.2f} ms "
          f"a tick over {T2} ticks, against {per('_inverse_batch'):.2f} batched")


if __name__ == "__main__":
    main()
