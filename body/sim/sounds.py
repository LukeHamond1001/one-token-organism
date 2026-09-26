"""THE ROOM'S SOUNDS FROM ITS PHYSICS (docs/SIM_DESIGN.md 5.2, 3.4; W5, the lead's first build at S5a): "every sound comes from an
event in the physics: a toy's own clip on its event, scaled by the impulse or speed; contact clicks; the parent's voice and footsteps
(from its gait); the child's tract. Each source is placed in space for the two ears." The voices are the lane's and the tract's; this
module makes the rest, so the child hears its room as a child hears one: a ball landing, a block knocked over, its own hand slapping
the mat, her steps crossing the floor, a rattle shaken.

EVENTS, read from the physics after every step (never from anything the body knows):
  impacts    a contact pair new this step (not in contact the step before) whose relative velocity along its normal closes at
             more than V_MIN: a toy against anything, the G1's own link against the room or a toy, her foot against the floor. Its
             speed is read from the bodies' spatial velocities at the contact point in the step's own state (MuJoCo's cvel, the
             state the contact was found in), its place is the contact's, its loudness grows with the speed (below).
  motion     the toys whose sound is their motion (the rattle's beads, the car's wheels, the bear's bell, the ring's crinkle, the
             bottle's slosh), each tick from its travel over the tick: a sound for the tick at its speed, while it moves faster than
             V_MOVE.
Each toy's sound is made once, deterministically, from 5.2's description of it, as a few damped modes and noise (modal synthesis:
a struck body rings at its modes and decays; van den Doel, Kry and Pai 2001), its frequencies and decays chosen to match the
description's material (rubber, wood, plastic, a bell, a drumhead): ours, disclosed, never tuned to the child. The G1's own knock is
its plastic housing on foam; her footfall a soft heel on a wooden floor.

LEVEL: a sound's peak pressure at 1 m is REF_PA (an object dropped onto a table at about 1 m/s, recalled as about 65-70 dB SPL at a
metre: ours) times (speed / REF_SPEED) ** 1 (the struck body's velocity sets its modes' amplitude: linear), clipped at MAX_GAIN, then
the toy's own loudness (LOUD) and the body's size do the rest. Motion sounds scale the same way with the toy's speed.

MIXING: an event's clip starts at its step's sample in the tick (32 samples a 2 ms step at 16 kHz) and plays on across the ticks that
follow; the tick's sounds of one source (a toy, the G1's own knocks, her steps) are summed into that source's samples and placed at
its last event's point: {name: (Pa at 1 m, position)} for body/sim/ears.py. state() and load_state() carry the clips under way, so a
replay is exact. Nothing draws a random number but each clip's own seeded noise, fixed at its making.
"""
import math

import numpy as np

SR = 16000
TICK = 2400
STEP_SAMPLES = 32                 # 2 ms at 16 kHz
V_MIN = 0.05                      # m/s: a new contact closing slower than this makes no sound (ours: a toy laid down, not dropped)
V_MOVE = 0.08                     # m/s: a motion sound's toy moving faster than this over a tick (ours)
REF_SPEED = 1.0                   # m/s (ours)
REF_PA = 0.06                     # Pa peak at 1 m at REF_SPEED (about 67 dB SPL as a peak; ours, recalled)
MAX_GAIN = 4.0                    # a very hard hit is at most 4 x the reference (ours)
CLIP_S = 0.6                      # the longest clip, s

# each sounding thing's modes [(Hz, decay s, relative amplitude)], its noise (a click's length s, share), its loudness: ours, from 5.2
KINDS = {
    "ball": dict(modes=[(180, 0.035, 1.0), (430, 0.02, 0.4)], click=(0.002, 0.3), loud=0.8),                # a soft rubber bounce
    "block": dict(modes=[(640, 0.05, 1.0), (1460, 0.03, 0.6), (2900, 0.015, 0.3)], click=(0.002, 0.2), loud=1.0),   # a hollow knock
    "duck": dict(modes=[(1900, 0.03, 1.0)], click=(0.001, 0.1), loud=0.6, squeak=(1800, 2500, 0.22)),       # a squeak when squeezed
    "cup": dict(modes=[(2200, 0.06, 1.0), (3500, 0.04, 0.6), (5200, 0.02, 0.3)], click=(0.001, 0.2), loud=0.9),     # a plastic clink
    "rattle": dict(modes=[(2600, 0.012, 0.6)], click=(0.003, 1.0), loud=0.9, motion="beads"),                # beads shaking
    "car": dict(modes=[(900, 0.02, 0.8), (2100, 0.01, 0.4)], click=(0.002, 0.5), loud=0.8, motion="wheels"),  # a wheel rattle
    "bear": dict(modes=[(250, 0.03, 0.4)], click=(0.003, 0.3), loud=0.6, motion="bell"),                     # a soft bell inside
    "stacker": dict(modes=[(1800, 0.03, 1.0), (3200, 0.02, 0.5)], click=(0.002, 0.3), loud=0.9),             # the rings' clack
    "drum": dict(modes=[(110, 0.25, 1.0), (240, 0.12, 0.5), (420, 0.06, 0.2)], click=(0.003, 0.2), loud=1.2),        # a boom
    "ring": dict(modes=[(3000, 0.01, 0.3)], click=(0.004, 1.0), loud=0.5, motion="crinkle"),                 # a crinkle handled
    "bottle": dict(modes=[(700, 0.03, 0.6)], click=(0.002, 0.3), loud=0.7, motion="slosh"),                  # a soft slosh
    "g1": dict(modes=[(95, 0.04, 1.0), (310, 0.02, 0.5)], click=(0.002, 0.4), loud=1.0),                     # its housing on foam
    "step": dict(modes=[(80, 0.05, 1.0), (260, 0.03, 0.4)], click=(0.003, 0.5), loud=0.5),                   # her heel on the floor
}
MOTION = {"beads": dict(rate=40.0, grain=(2600, 0.008)), "wheels": dict(rate=25.0, grain=(1500, 0.006)),
          "bell": dict(partials=[(1250, 0.4, 1.0), (3100, 0.25, 0.5), (4600, 0.15, 0.3)]),
          "crinkle": dict(rate=60.0, grain=(4200, 0.004)), "slosh": dict(band=(150, 600))}


def _clip(kind):
    """a struck thing's clip at unit peak: its damped modes and its click of noise (a fixed seed per kind)"""
    k = KINDS[kind]
    n = int(CLIP_S * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f, tau, a in k["modes"]:
        x += a * np.exp(-t / tau) * np.sin(2 * math.pi * f * t)
    rng = np.random.default_rng(abs(hash_kind(kind)))
    cl, share = k["click"]
    m = int(cl * SR)
    x[:m] += share * rng.standard_normal(m) * np.exp(-np.arange(m) / max(1, m / 3))
    if "squeak" in k:
        f0, f1, dur = k["squeak"]
        ts = t[:int(dur * SR)]
        ph = 2 * math.pi * (f0 * ts + (f1 - f0) * ts ** 2 / (2 * dur))
        env = np.sin(math.pi * ts / dur) ** 2
        x[:len(ts)] += 0.8 * env * (np.sin(ph) + 0.3 * np.sin(2 * ph))
    end = np.nonzero(np.abs(x) > 1e-4 * np.abs(x).max())[0]
    x = x[:end[-1] + 1] if len(end) else x[:1]
    return x / np.abs(x).max()


def hash_kind(kind):
    return int.from_bytes(kind.encode()[:8].ljust(8, b"\0"), "little") % (2 ** 31)


CLIPS = {k: _clip(k) for k in KINDS}


def _motion_tick(kind, speed, seed):
    """a moving toy's sound over one tick at its speed (unit peak at REF_SPEED): grains at a rate that grows with speed (beads,
    wheels, crinkle), a bell's partials, or a band of noise (a slosh); `seed` the tick's own (the world's tick and the toy)"""
    mo = MOTION[KINDS[kind]["motion"]]
    rng = np.random.default_rng(seed)
    t = np.arange(TICK) / SR
    x = np.zeros(TICK)
    s = min(speed / REF_SPEED, MAX_GAIN)
    if "rate" in mo:
        n = rng.poisson(mo["rate"] * s * TICK / SR)
        f, tau = mo["grain"]
        g = np.exp(-np.arange(int(4 * tau * SR)) / (tau * SR)) * np.sin(2 * math.pi * f * np.arange(int(4 * tau * SR)) / SR)
        for at in rng.integers(0, TICK, n):
            e = min(TICK, at + len(g)); x[at:e] += g[:e - at] * rng.uniform(0.5, 1.0)
    elif "partials" in mo:
        for f, tau, a in mo["partials"]:
            x += a * np.exp(-t / tau) * np.sin(2 * math.pi * f * t)
    else:
        lo, hi = mo["band"]
        w = rng.standard_normal(TICK)
        X = np.fft.rfft(w); fr = np.fft.rfftfreq(TICK, 1 / SR); X[(fr < lo) | (fr > hi)] = 0
        x = np.fft.irfft(X, TICK)
    pk = np.abs(x).max()
    return (x / pk if pk > 0 else x) * s


class Sounds:
    """the room's sounds over a G1World (the module's doc): `step(world)` after each physics step (its impacts), `tick_end(world)`
    at the tick's end -> {source: (Pa at 1 m [2400], position)} for the ears"""

    def __init__(self, world):
        m = world.m
        self.m = m
        root = m.body_rootid
        self.kind_of_body = {}
        for b in range(m.nbody):
            n = m.body(b).name or ""
            if n.startswith("toy_") and n[4:] in KINDS:
                self.kind_of_body[b] = n[4:]
        g1r = int(root[m.body("pelvis").id])
        for b in range(m.nbody):
            if int(root[b]) == g1r:
                self.kind_of_body[b] = "g1"
        self.feet = frozenset(m.body(n).id for n in ("parent_foot_L", "parent_foot_R") if _has_body(m, n))
        self.floorish = frozenset(g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == 0)
        self.body_of_toy = {k: b for b, k in self.kind_of_body.items() if k != "g1"}
        self.prev_pairs = set()
        self.voices = []                 # [source, kind, start sample (from the tick's start), gain, position]
        self.toy_pos = {}                # toy -> its position at the last tick's end (the motion sounds)
        self.events = []                 # the tick's events (instruments): (step, source, kind, speed, position)
        self.step_k = 0

    # ------------------------------------------------------------------ every step
    def step(self, world, k):
        """step k of the tick: the new contact pairs closing faster than V_MIN, each a struck clip at its point"""
        m, d = self.m, world.d
        n = int(d.ncon)
        pairs = set()
        if n:
            gg = np.asarray(d.contact.geom[:n])
            b1, b2 = m.geom_bodyid[gg[:, 0]], m.geom_bodyid[gg[:, 1]]
            for i in range(n):
                a, b = int(b1[i]), int(b2[i])
                ka, kb = self.kind_of_body.get(a), self.kind_of_body.get(b)
                foot = a in self.feet or b in self.feet
                if ka is None and kb is None and not foot:
                    continue
                if ka == "g1" and kb == "g1":
                    continue                                     # its own links pressing each other: no sound here
                key = (min(a, b), max(a, b))
                pairs.add(key)
                if key in self.prev_pairs:
                    continue
                c = d.contact[i]
                p, nrm = np.asarray(c.pos, float), np.asarray(c.frame[:3], float)
                v = _vel_at(m, d, b, p) - _vel_at(m, d, a, p)    # the second body's velocity relative to the first, at the point
                speed = abs(float(v @ nrm))
                if speed < V_MIN:
                    continue
                if ka is not None and ka != "g1":
                    src, kind = "toy_" + ka, ka
                elif kb is not None and kb != "g1":
                    src, kind = "toy_" + kb, kb
                elif foot:
                    src, kind = "parent_steps", "step"
                else:
                    src, kind = "g1_knock", "g1"
                g = min(speed / REF_SPEED, MAX_GAIN) * KINDS[kind]["loud"] * REF_PA
                self.voices.append([src, kind, k * STEP_SAMPLES, g, p.copy()])
                self.events.append((k, src, kind, round(speed, 3), p.round(3).tolist()))
        self.prev_pairs = pairs

    # ------------------------------------------------------------------ the tick's end
    def tick_end(self, world):
        """the tick's sources: each voice's clip samples in this tick, summed per source; the motion sounds of the moving toys"""
        d = world.d
        out = {}
        keep = []
        for v in self.voices:
            src, kind, start, g, p = v
            clip = CLIPS[kind]
            a = max(0, -start)
            seg = clip[a:a + TICK - max(0, start)] * g
            buf, pos = out.setdefault(src, [np.zeros(TICK), p])
            s0 = max(0, start)
            buf[s0:s0 + len(seg)] += seg
            out[src][1] = p
            v[2] = start - TICK                                  # next tick it plays on from here
            if -v[2] < len(clip):
                keep.append(v)
        self.voices = keep
        for b, toy in sorted(self.kind_of_body.items()):
            if toy == "g1" or "motion" not in KINDS[toy]:
                continue
            p = d.xpos[b].copy()
            p0 = self.toy_pos.get(toy)
            self.toy_pos[toy] = p
            if p0 is None:
                continue
            speed = float(np.linalg.norm(p - p0)) / (TICK / SR)
            if speed < V_MOVE:
                continue
            x = _motion_tick(toy, speed, seed=(int(world.tick) * 131 + hash_kind(toy)) % (2 ** 31))
            buf, _ = out.setdefault("toy_" + toy, [np.zeros(TICK), p])
            buf += x * KINDS[toy]["loud"] * REF_PA
            out["toy_" + toy][1] = p
            self.events.append((-1, "toy_" + toy, "motion", round(speed, 3), p.round(3).tolist()))
        ev, self.events = self.events, []
        self.last_events = ev
        return {k: (v[0], v[1]) for k, v in out.items()}

    # ------------------------------------------------------------------ the save
    def state(self):
        return dict(prev_pairs=sorted([list(k) for k in self.prev_pairs]),
                    voices=[[s, k, int(st), float(g), np.asarray(p, float).tolist()] for s, k, st, g, p in self.voices],
                    toy_pos={k: np.asarray(v, float).tolist() for k, v in self.toy_pos.items()})

    def load_state(self, s):
        self.prev_pairs = {tuple(int(x) for x in k) for k in s["prev_pairs"]}
        self.voices = [[str(a), str(k), int(st), float(g), np.array(p, dtype=np.float64)] for a, k, st, g, p in s["voices"]]
        self.toy_pos = {k: np.array(v, dtype=np.float64) for k, v in s["toy_pos"].items()}
        self.events = []


def _has_body(m, name):
    try:
        m.body(name)
        return True
    except KeyError:
        return False


def _vel_at(m, d, b, p):
    """a body's linear velocity at a world point, from its spatial velocity (MuJoCo's cvel: angular then linear, at its tree's
    centre of mass, in world axes); the world's body (0) is still"""
    if b == 0:
        return np.zeros(3)
    cv = d.cvel[b]
    c = d.subtree_com[m.body_rootid[b]]
    return cv[3:6] + np.cross(cv[0:3], p - c)
