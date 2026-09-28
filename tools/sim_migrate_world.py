"""THE WORLD CARRIED ACROSS A CHANGE OF ROOM (A115, 2026-09-27; docs/SIM_DESIGN.md C99): a life's saved world (DIR/world.pt) was made
under one model, and the world refuses a save of another (G1World.load_state: nstate). When something is added to the room through
g1scene.load_model's `extra` hook (body/sim/extras.py: the book), the old model is a prefix of the new (its bodies, joints, geoms
and actuators first, the new body's last), so the state carries across by position: the old world is built beside the new, the
save is loaded into the old, every joint's position and velocity, the actuators' activations and controls, the time and the warm
start are copied into the new by their old indices, the new body keeps its compiled place (qpos0), the model's mutable fields
(the day's light, the servos' bias) likewise, and everything else of the world (the senses' carry, the parent, the lane, the eyes)
is the same dictionary. The new pair is written beside a copy of the old (world_before_<extra>.pt). The life's own save (life.pt)
is untouched: the body is the same body.

    python3 tools/sim_migrate_world.py --pair data/g1_seed1 --extra book [--xy 0.0 -0.6] [--yaw 0] [--seed 1] [--voice real|fake]

Run at a dawn, before the runner is started with --extra book. Never on the running life's pair."""
import argparse
import os
import pickle
import shutil
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from body.sim import world as W  # noqa: E402
from body.sim import extras as X  # noqa: E402
from body.sim import g1scene as G  # noqa: E402


def build(seed, extra, voice, lane=True, eyes=True, room="a"):
    """the world with its eyes and lane as tools/sim_life.py builds them (so the lane's and the eyes' states carry)"""
    from body.sim import lane as L
    from body.sim import eyes as E
    w = W.G1World(seed=seed, extra=extra, xml=G.ROOMS[room])                 # A133: the room's layout
    ey = E.Eyes(w) if eyes else None
    ln = None
    if lane:
        if voice == "real":
            from body.sim.voice import synth as V
            vc = V.VoiceCache(os.path.join(os.environ.get("IGA_VOICE_DIR", "/tmp"), "voice_migrate"))
        else:
            from sim_life import FakeVoice
            vc = FakeVoice()
        ln = L.ParentLane(w, seed=seed, voice=vc, day_ticks=int(W.SIM_CFG.get("wake_ticks", 24000)) if hasattr(W, "SIM_CFG") else 24000)
    return w, ey, ln


def migrate_state(st, w_old, w_new):
    """the saved state dict `st` (of w_old's model) as w_new's: the physics and the model fields carried by position"""
    m0, d0 = w_old.m, w_old.d
    m1, d1 = w_new.m, w_new.d
    if st.get("nstate") != mujoco.mj_stateSize(m0, W.STATE_SPEC):
        raise ValueError("the save is not of the old model")
    for f, v in st["model"].items():
        getattr(m0, f)[...] = v
    mujoco.mj_setState(m0, d0, st["physics"], W.STATE_SPEC)
    mujoco.mj_forward(m0, d0)
    for i in range(m0.nbody):                                            # the old model a prefix of the new: every name in its place
        if m1.body(i).name != m0.body(i).name:
            raise ValueError(f"body {i} differs: {m0.body(i).name} vs {m1.body(i).name}")
    for i in range(m0.njnt):
        if m1.joint(i).name != m0.joint(i).name:
            raise ValueError(f"joint {i} differs")
    if m1.nu != m0.nu or m1.ngeom < m0.ngeom or m1.nq < m0.nq or m1.nv < m0.nv:
        raise ValueError("the new model is not the old with more")
    mujoco.mj_resetData(m1, d1)
    d1.time = d0.time
    d1.qpos[:m0.nq] = d0.qpos; d1.qvel[:m0.nv] = d0.qvel
    d1.act[:] = d0.act; d1.ctrl[:] = d0.ctrl
    d1.qacc_warmstart[:m0.nv] = d0.qacc_warmstart
    if m0.nmocap:
        d1.mocap_pos[:m0.nmocap] = d0.mocap_pos; d1.mocap_quat[:m0.nmocap] = d0.mocap_quat
    mujoco.mj_forward(m1, d1)
    model = {}
    for f, v in st["model"].items():
        cur = getattr(m1, f).copy()
        v = np.asarray(v)
        cur[tuple(slice(0, n) for n in v.shape)] = v                     # the old arrays into the prefix; the new body's rows its own
        model[f] = cur
    phys = np.zeros(mujoco.mj_stateSize(m1, W.STATE_SPEC))
    mujoco.mj_getState(m1, d1, phys, W.STATE_SPEC)
    out = dict(st)
    out["physics"] = phys; out["nstate"] = int(phys.size); out["model"] = model
    return out


def migrate_blob(blob, extra_name, xy=(0.0, -0.6), yaw=0.0, seed=1, voice="fake", old_extra=None):
    """the saved world's bytes (world.pt) -> the bytes of the same world with the extra added (old_extra: the extra the saved world
    already lives with, so its model is built as the save was made: A121, a world with the book carried to the book and the box)"""
    extra = X.EXTRAS[extra_name](xy=xy, yaw_deg=yaw)
    w_old, _, _ = build(seed, X.EXTRAS[old_extra]() if old_extra else None, voice, lane=False, eyes=False)
    w_new, ey, ln = build(seed, extra, voice)
    st = pickle.loads(bytes(blob))
    st2 = migrate_state(st, w_old, w_new)
    w_new._restore(st2)                                                  # the whole world as the save had it, in the new room
    if ln is not None:
        ln.conduct.set_world(ln._inventory(w_new.m))                     # her inventory names the new thing (the book, red)
    return w_new.save_state(), w_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--extra", required=True, choices=sorted(X.EXTRAS))
    ap.add_argument("--xy", type=float, nargs=2, default=(0.0, -0.6))
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--voice", choices=("real", "fake"), default="fake")
    ap.add_argument("--old-extra", default=None, choices=sorted(X.EXTRAS), help="the extra the saved world already lives with (A121)")
    a = ap.parse_args()
    src = os.path.join(a.pair, "world.pt")
    with open(src, "rb") as f:
        blob = pickle.load(f)
    blob2, w = migrate_blob(blob, a.extra, tuple(a.xy), a.yaw, a.seed, a.voice, old_extra=a.old_extra)
    bak = os.path.join(a.pair, f"world_before_{a.extra}.pt")
    shutil.copy2(src, bak)
    tmp = src + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(blob2, f, protocol=4)
    os.replace(tmp, src)
    b = w.m.body(f"toy_{X.NEW_TOY[a.extra]}").id
    print(f"migrated {src} to the room with the {X.NEW_TOY[a.extra]} at {np.round(w.d.xpos[b], 3).tolist()} (tick {w.tick}); the old world kept as {bak}", flush=True)


if __name__ == "__main__":
    main()
