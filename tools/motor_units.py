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
C54 (the generator's measured newborn rhythm, a movement of 2 + 3 ticks and a pause drawn from the seed): per rhythm (the legs' one,
each arm's own) the cycles drawn over the run (every cycle whose start fell inside it: their number, mean and SD in seconds, shortest,
longest, the share held at 1.0 s and at 8.5 s), per limb the share of ticks in flexion, extension and the pause, and its own
movement-to-movement intervals (the right leg's are half of one cycle and half of the next).
THE CEREBELLUM (on in SIM_CFG, the G1's mossy list, 219 numbers): the stub calls it below each tick as SimWorld's contract says (every
10 ms, 15 a tick): the mossy numbers in the anatomy's order (the efference copy from the tick's acts of every motor effector, the words'
token output none; every other number drawn about its declared middle within its half-range from the stub's own stream), a teacher at
each readout's joint (drawn within +-5 N m) under the model's torque limits, and at sub-step 0 a slip and a turn; so every sub-step runs
the whole law (the leak, the lesson, the bound; the flocculus once a tick). It writes down the law's wall time a tick inside the hook
(the organ's cost; the stub's building of the numbers is the world's and is timed apart) and the organ's lessons.
usage: nice -n 19 python3 tools/motor_units.py [--ticks 3000] [--d 512] [--seed 1] [--unbatched 1]"""
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
from body.core.world import Frame, SimWorld, SubFrame, WorldLoop  # noqa: E402
from body.sim.anatomy import BODY_JOINTS, CEREB_JOINTS, SIM_CFG, SIZES, SimAnatomy, born_table  # noqa: E402

# the model's own torque limits at the cerebellum's readouts (Menagerie's g1_with_hands.xml, each joint's actuatorfrcrange; 3.2): the
# stub's limit this tick, with no weakness (the charge full)
LIMIT = dict(zip(BODY_JOINTS, (88, 50, 50) + (25,) * 5 + (5, 5) + (25,) * 5 + (5, 5) + (2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4) * 2
                 + (88, 139, 88, 139, 50, 50) * 2))
assert len(LIMIT) == len(BODY_JOINTS) == 43


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
        self.t = 0; self.rng = random.Random(seed); self.rng_cb = random.Random(seed + 1); self.world_s = 0.0

    def frame(self):
        R = self.rng
        obs = {n: [R.uniform(-1, 1) for _ in range(k)] for n, k in SIZES.items() if n not in ("face", "charge")}
        obs["body"][241] = 1.0; obs["face"] = [0.0, 0.0]; obs["charge"] = [1.0, 0.0]; obs["pain"] = [0.0] * 44
        return Frame(self.t, obs, 0.0, {})

    def apply(self, acts):
        b = self.below
        if b is not None:                                                 # the loop below the tick (the tool's doc)
            an = b._life().anatomy; cb = an.cerebellar; R = self.rng_cb
            lim = [float(LIMIT[j]) for j in CEREB_JOINTS]; dig = {}
            for e in an.motors:
                a = int(acts.get(e.name, e.rest_id)); J = len(e.factors)
                dig[e.name] = [(a // 5 ** (J - 1 - k)) % 5 for k in range(J)]
            for s in range(15):
                t0 = time.perf_counter(); seen = collections.Counter(); mossy = []
                for name, mid, hr in zip(an.mossy, cb.mossy_offset, cb.mossy_scale):
                    kind, what = name.split(" ", 1)
                    if kind == "act":
                        eff = what.split(".", 1)[0]
                        mossy.append(float(dig[eff][seen[eff]])); seen[eff] += 1
                    else:
                        mossy.append(mid + hr * R.uniform(-1.0, 1.0))
                teach = [max(-l_, min(l_, R.uniform(-5.0, 5.0))) for l_ in lim]
                sf = SubFrame(self.t, s, mossy, teach, [R.uniform(-0.01, 0.01), R.uniform(-0.01, 0.01)] if s == 0 else None,
                              [R.uniform(-0.1, 0.1), R.uniform(-0.1, 0.1)] if s == 0 else None, limit=lim)
                self.world_s += time.perf_counter() - t0
                self.sub_tick(sf)
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


def spg_report(L, place, T):
    """C54: the drawn cycles of each rhythm over the run and each limb's places (the tool's doc)"""
    tick = 0.15
    lo, hi = L._reflex_const("spg_cycle_min"), L._reflex_const("spg_cycle_max")
    rhythms = {}
    for i, e in enumerate(L.anatomy.motors, 1):
        if e.spg:
            r, lead, lag = L._spg_rhythm(i, e)
            rhythms.setdefault(r, []).append(e.name)
    print(f"C54, THE SPINAL PATTERN GENERATOR'S CYCLES over {T} ticks ({T * tick:.0f} s): a movement of {L._reflex_const('spg_flex')} ticks' flexion "
          f"and {L._reflex_const('spg_ext')} ticks' extension, then a pause; each cycle drawn from the seed (the law: 3.56 +- 1.93 s held to "
          f"1.0-8.5 s, the held law's moments 3.514 +- 1.742 s)")
    pooled = []
    for r, limbs in rhythms.items():
        e0 = L.anatomy.motors[r]; lead = L._spg_phase0(r + 1, e0)
        s_, n, cyc = None, 0, []
        while True:
            Ln = L._spg_cycle(r, n); s_ = -lead * Ln if s_ is None else s_
            if s_ >= T:
                break
            if s_ + Ln > 0:
                cyc.append(Ln)                                             # a cycle under way at some tick of the run
            s_ += Ln; n += 1
        pooled += cyc
        sec = [tick * x for x in cyc]
        print(f"  the rhythm of {'+'.join(limbs):12s}: {len(cyc):4d} cycles, mean {statistics.mean(sec):.3f} s, sd {statistics.pstdev(sec):.3f} s, "
              f"shortest {min(sec):.2f} s, longest {max(sec):.2f} s; held at 1.0 s {sum(1 for x in cyc if x == lo)}, at 8.5 s {sum(1 for x in cyc if x == hi)}")
    sec = [tick * x for x in pooled]
    print(f"  all {len(pooled)} cycles drawn: mean {statistics.mean(sec):.3f} s, sd {statistics.pstdev(sec):.3f} s ({statistics.mean(pooled):.2f} +- "
          f"{statistics.pstdev(pooled):.2f} ticks); held at 1.0 s {sum(1 for x in pooled if x == lo) / len(pooled):.3f}, at 8.5 s "
          f"{sum(1 for x in pooled if x == hi) / len(pooled):.3f}")
    tot = collections.Counter()
    for n, pl in place.items():
        c = collections.Counter(pl); tot.update(c)
        starts = [t for t in range(1, len(pl)) if pl[t] == 1 and pl[t - 1] != 1] + ([0] if pl and pl[0] == 1 else [])
        starts.sort(); iv = [tick * (b - a) for a, b in zip(starts, starts[1:])]
        print(f"  {n:6s} flexion {c[1] / len(pl):.4f}  extension {c[-1] / len(pl):.4f}  pause {c[0] / len(pl):.4f}  "
              f"({len(starts)} movements; movement to movement {statistics.mean(iv):.3f} +- {statistics.pstdev(iv):.3f} s)")
    k = sum(tot.values())
    print(f"  all limbs: flexion {tot[1] / k:.4f}, extension {tot[-1] / k:.4f}, pause {tot[0] / k:.4f} (the law's: 2 / 23.425 = 0.0854, "
          f"3 / 23.425 = 0.1281, 0.7866)")


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
    place = collections.defaultdict(list)                                  # C54: each limb's place in its rhythm, tick by tick
    for t in range(T):
        t0 = time.perf_counter(); run.step(); tick_s.append(time.perf_counter() - t0)
        for e, st in zip(L.anatomy.motors, L.motor):
            now = st["now"]; n = e.name; pacts[n].append(now["p_act"])
            if e.spg:
                place[n].append(now.get("spg"))
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
    spg_report(L, place, T)
    b = L.world.below
    if b is not None:
        org = L.m.cereb
        print(f"THE CEREBELLUM (on; the G1's mossy list: {b.n_mossy} fibres, {len(b.joints)} readouts, the flocculus on {list(b.vor)}): "
              f"{b.calls / T:.0f} calls a tick, the law {1000 * b.seconds / T:.2f} ms a tick ({1e6 * b.seconds / max(1, b.calls):.0f} us a "
              f"sub-step) inside the hook; its lessons: limbs {int(org.n_limb)} of {int(org.n_sub)} sub-steps, flocculus {int(org.n_vor)}; "
              f"the stub's building of the numbers (the world's, apart) {1000 * L.world.world_s / T:.2f} ms a tick")
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
    if not arg("unbatched", 1):
        return
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
