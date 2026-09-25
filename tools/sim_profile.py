"""THE SIM PROFILE OF THE BEHAVIOUR GUARD (the core refactor's step R8d; SIM_DESIGN.md 8: "after the plumbing run, a sim profile joins the
check"; tools/determinism_check.py --profile sim runs it): a tiny G1 born at a fixed seed lives a fixed stub of its world through a day
and R8's whole night, and its whole state is hashed. Run before and after an edit of body/: an equal digest means the edit changed nothing
the sim body does on that script. Imports the body from its own tree (the check's), so it can guard a worktree. Pinned threaded and
one-thread in tools/pins/digests.txt beside the language digests.

THE BODY: body/sim/anatomy.py's SimAnatomy (the born table's placeholder words) under SIM_CFG (every switch the sim is born with: the
frames, the amygdala, recall into action, the cerebellum, the born patterns and biases, the night over frames and the twitches), at d 32,
1 block, 2 heads, window 8, seed 0; SIM_CFG's constants but the sizes a tiny life can afford: wake_ticks 225 (the sleep switch at tick 225,
its night at the tick's end), night_ticks 120, 64 dreams (night_starts and night_starts_max) in 2 rounds of batches of 8, REM's 4 dreams of
4 steps, the waking lessons and the gates' every 8 ticks, the write floor at 1e-30 and the gates' floor at 0.3 (so a newborn acts), the
night on the CPU.
THE WORLD (a stub of the G1's world, never the sim's; a function of its tick and its two seeded streams, as body/tests/test_motor.py's
motor 12's): every channel's numbers uniform in -1..1 from its first stream (the frame's keys at body/sim/anatomy.py's sizes), loud (x 3)
6 ticks in every 40 and soft (x 0.3) between, so the frames' events end; the charge falling from 0.9; the breath left drawn; a smile every
40 ticks; pain on two ticks; a face in the periphery, a sound's onset and a sudden change now and then; a word every 17 ticks; a face in
the fovea every 23; the torso's unit turning and a little tilted; its truth the torso's true yaw (C51's instrument). Each apply calls the
cerebellum's hook 15 times (the mossy numbers from the tick's acts and its second stream, a teacher and a limit of 25 N m at each readout's
joint, a slip and a turn at sub-step 0 by day). It runs through the night (live_night): at dusk its frames carry the body's own senses
alone and the hook no slip; at dawn the day's senses again.
THE HASH: the organs (their state, their gradients, every buffer and every plain attribute of every module), the store, the optimizers'
states, the random streams (the life's and the process's), every working attribute of the life but the interface declarations (anatomy,
world) and the page's wall clock, the constants; a digest per section (printed on the "full:" line) and the digest over them."""
import collections
import dataclasses
import hashlib
import math
import os
import pickle
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import torch  # noqa: E402

from body.life import Life  # noqa: E402
from body.core.world import Frame, SimWorld, SubFrame, WorldLoop  # noqa: E402
from body.sim.anatomy import SIM_CFG, SIZES, SimAnatomy, born_table  # noqa: E402

TICKS = 300
SIZES_CFG = dict(wake_ticks=225, night_ticks=120, night_starts=64, night_starts_max=64, night_rounds=2, night_batch=8, rem_dreams=4, rem_steps=4,
                 wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3, night_dev="")


def mossy(anatomy, acts, R):
    """the G1's mossy numbers for one sub-step as a world hands them (motor 11's rule): each motor effector's joints' settings from the
    tick's acts (base-5 digits, joint 0 the most significant), every other number drawn about its declared middle within 1.1 of its
    half-range"""
    cb = anatomy.cerebellar; digits = {}
    for e in anatomy.motors:
        a = int(acts.get(e.name, e.rest_id)); J = len(e.factors)
        digits[e.name] = [(a // 5 ** (J - 1 - k)) % 5 for k in range(J)]
    out, seen = [], collections.Counter()
    for name, mid, hr in zip(anatomy.mossy, cb.mossy_offset, cb.mossy_scale):
        kind, what = name.split(" ", 1)
        if kind == "act":
            eff = what.split(".", 1)[0]
            out.append(float(digits[eff][seen[eff]])); seen[eff] += 1
        else:
            out.append(mid + hr * R.uniform(-1.1, 1.1))
    return out


class Script(SimWorld):
    """the profile's world (the module's doc)"""
    live_night = True

    def __init__(self, seed=0):
        self.t = 0; self.rng = random.Random(seed); self.rng_cb = random.Random(seed + 1); self.dark = False

    def frame(self):
        t = self.t; R = self.rng; a_ = 3.0 if t % 40 < 6 else 0.3
        obs = {n: [a_ * R.uniform(-1, 1) for _ in range(k)] for n, k in SIZES.items() if n not in ("face", "charge")}
        obs["face"] = [2.0 if t % 40 == 20 else 0.0, 0.0]; obs["charge"] = [max(0.05, 0.9 - 0.002 * t), -0.002]
        obs["body"][241] = R.uniform(0.2, 1.0)
        obs["pain"] = [1.0 if (t in (60, 147) and k == 5) else 0.0 for k in range(44)]
        if t % 30 == 3:
            obs["face_periph"] = [1.0, 0.4, -0.2]
        if t % 45 == 7:
            obs["sound_side"] = [1.0, 0.5]
        if t % 37 == 11:
            obs["onset_periph"] = [1.0, -0.3, 0.1]
        if t % 17 == 0:
            obs["words"] = R.randrange(3, 79)
        if t % 23 == 5:
            obs["face_fovea"] = [1.0]
        yaw_rate = 0.1 * math.sin(t / 7.0)
        obs["imu_torso"] = [0.3 * math.sin(t / 11.0), 0.2 * math.cos(t / 13.0), 9.81, 0.05 * math.cos(t / 5.0), 0.04 * math.sin(t / 9.0),
                            yaw_rate + 0.02]
        if self.dark:
            for k in ("ears", "eye_p", "eye_f", "face", "words", "face_periph", "sound_side", "onset_periph", "face_fovea"):
                obs.pop(k, None)
        true_yaw = sum(0.15 * 0.1 * math.sin(u / 7.0) for u in range(t))   # the torso's true yaw since birth (the gyro's 0.02 its bias)
        return Frame(t, obs, 0.0, {"who": "parent", "yaw": true_yaw})

    def apply(self, acts):
        if self.below is not None:
            life = self.below._life(); J = len(self.below.joints); R = self.rng_cb
            for s_ in range(15):
                self.sub_tick(SubFrame(self.t, s_, mossy(life.anatomy, acts, R), [R.uniform(-2.0, 2.0) for _ in range(J)],
                                       [R.uniform(-0.01, 0.01) for _ in range(2)] if (s_ == 0 and not self.dark) else None,
                                       [R.uniform(-0.1, 0.1) for _ in range(2)] if (s_ == 0 and not self.dark) else None, limit=[25.0] * J))
        self.t += 1

    def pause(self):
        pass

    def resume(self):
        pass

    def dusk(self):
        self.dark = True

    def dawn(self):
        self.dark = False

    def save_state(self):
        return pickle.dumps((self.t, self.rng.getstate(), self.rng_cb.getstate(), self.dark))

    def load_state(self, blob):
        self.t, a_, b_, self.dark = pickle.loads(blob); self.rng.setstate(a_); self.rng_cb.setstate(b_)


def canon(g, x, path):
    """every value fed to the hash with its type and shape (the check's --full and body/tests/test_anatomy.py's `_canon`)"""
    if torch.is_tensor(x):
        t = x.detach().cpu().contiguous(); g.update(f"T{t.dtype}{tuple(t.shape)}".encode()); g.update(t.numpy().tobytes())
    elif isinstance(x, dict):
        g.update(b"{")
        for k in sorted(x, key=repr):
            g.update(repr(k).encode() + b":"); canon(g, x[k], f"{path}.{k}")
        g.update(b"}")
    elif isinstance(x, collections.deque):
        g.update(f"Q{x.maxlen}[".encode())
        for i, v in enumerate(x):
            canon(g, v, f"{path}[{i}]")
        g.update(b"]")
    elif isinstance(x, (list, tuple)):
        g.update(b"[" if isinstance(x, list) else b"(")
        for i, v in enumerate(x):
            canon(g, v, f"{path}[{i}]")
        g.update(b"]")
    elif isinstance(x, (set, frozenset)):
        g.update(b"S"); canon(g, sorted(x, key=repr), path)
    elif x is None or isinstance(x, (bool, int, float, str, bytes)):
        g.update((type(x).__name__ + repr(x) + ";").encode())
    elif isinstance(x, (torch.dtype, torch.device)):
        g.update(repr(x).encode())
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        g.update(f"D{type(x).__name__}(".encode()); canon(g, [(f_.name, getattr(x, f_.name)) for f_ in dataclasses.fields(x)], path); g.update(b")")
    else:
        raise TypeError(f"sim profile: no hash for {type(x).__name__} at {path}")


def sections(L):
    """the whole life, section by section (the module's doc)"""
    m = L.m; OPTS = sorted(k for k, v in vars(L).items() if isinstance(v, torch.optim.Optimizer))
    org = [("sd", dict(m.state_dict())), ("grad", {n: p.grad for n, p in m.named_parameters()}), ("buf", dict(m.named_buffers()))]
    for mn, mod in m.named_modules():
        org.append(("attrs:" + (mn or "."), {k: v for k, v in vars(mod).items() if not k.startswith("_") and not isinstance(v, torch.nn.Module)}))
    SKIP = {"m", "store", "tok", "gen", "cfg", "save_path", "_t_feel", "anatomy", "effectors", "world"} | set(OPTS)
    out = []
    for name, items in (("organs", org), ("store", sorted(vars(L.store).items())), ("optim", [(k, getattr(L, k).state_dict()) for k in OPTS]),
                        ("rng", [("gen", L.gen.get_state()), ("torch", torch.get_rng_state())]),
                        ("work", [(k, v) for k, v in sorted(vars(L).items()) if k not in SKIP]), ("cfg", [("cfg", L.cfg)])):
        g = hashlib.sha256()
        for k, v in items:
            g.update(k.encode() + b"="); canon(g, v, name + "." + k)
        out.append((name, g.hexdigest()))
    return out


def main():
    cfg = dict(SIM_CFG, **SIZES_CFG)
    consts = hashlib.sha256(repr(sorted((k, type(v).__name__, repr(v)) for k, v in cfg.items())).encode()).hexdigest()[:12]
    torch.manual_seed(0)
    L = Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=32, layers=1, heads=2, window=8, cfg=cfg, seed=0, world=Script(0))
    run = WorldLoop(L)
    for _ in range(TICKS):
        run.step()
    rep = L.last_night or {}
    assert L.nights == 1 and not rep.get("error"), f"the sim profile's night: {rep.get('error')} (nights {L.nights})"
    secs = sections(L)
    h = hashlib.sha256()
    for name, d in secs:
        h.update(name.encode()); h.update(d.encode())
    live = rep.get("live") or {}
    print(f"digest {h.hexdigest()[:24]} | ticks {L.ticks} nights {L.nights} store {L.store.n()} episodes {len(getattr(L, '_episodes', []) or [])}"
          f" | dreams {rep.get('dreams')} tagged first {rep.get('tagged_first')} twitches {live.get('twitches')} | profile sim consts {consts}")
    print("full: " + " ".join(f"{n} {d[:12]}" for n, d in secs))


if __name__ == "__main__":
    main()
