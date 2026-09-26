"""THE EYE CHECK (docs/SIM_DESIGN.md 3.4, 13 risk 9, C2 and C3; the build plan's W3): does the software fovea tell the ten toys and
the parent's face apart at 3 px a degree (W3r: the grey foveae through the born bank and the colour window, eye_f 1,536; A42, A78)
in the furnished room, under morning, midday and dusk light, with and without the sun's
shadow; and does the face test's ray agree with a segmentation render? An instrument, never the body: no weight of any body learns
here, and the readouts are the instrument's own.

The G1 lies as born. Each view puts one thing (a toy, or the parent's head facing the eyes within 45 deg) at a random point 0.3-1.2
m in front of the eyes, within the fovea's reach, turned at random; the gaze is aimed at it with a 1.5 deg error (gaze_at, the
instrument's aim), and the two eyes render. The label is what fills the left fovea window in a segmentation render at the same eye
(the most pixels of one thing, at least 20 of 1024; else none): so an occluded or clipped thing is labelled as what is seen. The
same views are rendered under each light and shadow setting (paired). The code read is the eyes' fovea code as the body gets it
(eye_f: 2 x 640 of the born bank and the colour window's 256, body/sim/eyes.py), standardized on the training views; readouts: the nearest class mean and a ridge one-hot, trained on 2/3
of the views and tested on the rest, balanced over the classes present; and a small nonlinear readout (one hidden layer of 256), as
test 11 read born codes, since the body's reader is a cortex, not a linear map. The design's bar is 0.75 (C3). WHICH READOUT DECIDES
(the W1 verifier's third round): the nonlinear one. The core reads the eye channel as it reads every vector channel: the retina's
code through the channel's born code (a fixed projection into the cortex's d, SIM_DESIGN.md 3.4) into the cortex's learned blocks
(attention and a GELU MLP in each), so what it can tell apart is what a learned nonlinear map of the code can; the one-hidden-layer
readout is the instrument's stand-in for it, and the nearest mean and the ridge are reported beside as linear lower bounds, never
the verdict. Beside it (the W1 verifier's fourth round): the same readout on the code as the core receives it (proj256_mlp: a fixed
random projection into d = 256 and the input's LayerNorm), and both readouts trained on a third, two thirds and all of the training
views, so the verdict's dependence on the views a class shows. The lights are stand-ins
until the day's light is built (W5): the sun as built (midday), low from the window's side and warm (morning), low and orange
(dusk). C2: for the face views (a third of them with a toy put on the line between the eye and the mouth, at a random place
along it), the face test's verdict in the left eye against a segmentation render's: the same test with the ray replaced by the
render (8 x the eye's resolution, so the mouth resolves: the pixel at the mouth point is the parent's head) and the face's pixels
counted in the window; their agreement, and the ray's verdict against the render's visibility alone.
THE C2 AND C3 CHECK ON BABBLED FRAMES (--c2 N): N frames of the babbling G1 (a tick of the babbler between frames; the design's
2,000), each with the parent's head put at a random distance 0.3-3.5 m from the left eye (uniform in distance) in a random direction
within the fovea's reach, KEPT INSIDE THE ROOM (the W1 verifier's seventh finding: 231 of 300 placements had put her head beyond the
ceiling or a wall, where the ray and the render agree trivially): a placement whose head is not at least HEAD_CLEAR_M inside the
walls, the floor and the ceiling is drawn again (the distances reached are reported by bin), turned at random up to 90 deg about
her up axis and nodded up to 20 deg, a toy on the line between the eye and her mouth in a third of the frames; the gaze aimed at
her mouth with a 3 deg error per axis (so the mouth is sometimes outside the window). C2: the face test's verdict (the left eye) against the segmentation render's (as above), by distance. C3: the
born face template (body/sim/eyes.py) on the frame's own pixels against the face test: its hits (the template fires on a frame the
test passes), and its false alarms on frames the test fails and on frames with no face at all (her head moved out of the room),
under each light. The face test decides nothing here; both are only compared.
With --codes, the same views' fovea is also read (under the first light, no shadow) as its raw pixels (both eyes, 2 x 3,072) and
as a finer retina (2 px cells: 16 x 16 x 6 per eye), to place what the retina's 4 px cells keep of the pixels' identity.
Run: nice -n 19 python3 tools/sim_eye_check.py [--views 60] [--seed 1] [--lights midday,morning,dusk] [--codes] [--c2 2000]
(JSON on stdout; nothing written)"""
import argparse
import json
import math
import os
import sys
import time

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from body.sim import eyes as E  # noqa: E402
from body.sim import world as W  # noqa: E402

G = W.G
kin = G.kin
TOYS = ["ball", "block", "duck", "cup", "rattle", "car", "bear", "stacker", "drum", "ring"]
CLASSES = TOYS + ["face"]
LIGHTS = {"midday": None,
          "morning": dict(dir=(0.85, 0.35, -0.40), diffuse=(0.32, 0.28, 0.22)),
          "dusk": dict(dir=(0.85, -0.35, -0.40), diffuse=(0.30, 0.18, 0.10))}
AIM_ERR_DEG = 1.5
HEAD_CLEAR_M = 0.15             # her head's centre at least this far inside the room's walls, floor and ceiling (C2's sampler)
sys.path.insert(0, os.path.join(ROOT, "body", "sim"))
from make_g1room import ROOM_H, ROOM_X, ROOM_Y  # noqa: E402


def inside_room(p, clear=HEAD_CLEAR_M):
    return abs(p[0]) <= ROOM_X - clear and abs(p[1]) <= ROOM_Y - clear and clear <= p[2] <= ROOM_H - clear


def rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _bank_fine(Lg):
    """the born bank's maps pooled over 4 x 4 px cells instead of 8 x 8 (what the pooling keeps of the pixels' identity)"""
    keep = E.CELL_F
    try:
        E.CELL_F = 4
        return E.bank(Lg)
    finally:
        E.CELL_F = keep


def ridge_acc(Xtr, ytr, Xte, yte, k, lam=1.0):
    Y = np.eye(k)[ytr]
    A = np.c_[Xtr, np.ones(len(Xtr))]
    Wt = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    return (np.c_[Xte, np.ones(len(Xte))] @ Wt).argmax(1)


def mlp_acc(Xtr, ytr, Xte, k, seed=0, hidden=256, epochs=300):
    """a small nonlinear readout (one hidden layer, the instrument's own; full-batch Adam, weight decay), as t11 read born codes"""
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, k))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    X, y = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad(); torch.nn.functional.cross_entropy(net(X), y).backward(); opt.step()
    with torch.no_grad():
        return net(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()


CURVE_SHARES = (1 / 3, 2 / 3, 1.0)
PROJ_D, PROJ_SEED = 512, 0      # the core's d at birth (7.1: 512)
MIN_PX = 20 * (W.FOVEA_PX / 32) ** 2   # a thing labels the window with at least 20 px of the first build's 32 x 32: 80 of the 64 x 64


def proj_ln(X, d=PROJ_D, seed=PROJ_SEED):
    """the codes through a fixed Gaussian projection (unit variance a column over the input's width, drawn from a seed) and a
    LayerNorm without its affine (the born cortex's input: SIM_DESIGN.md 3.4's born code, the core's in_ln at birth)"""
    P = np.random.default_rng(seed).normal(size=(X.shape[1], d)) / math.sqrt(X.shape[1])
    Z = X @ P
    return (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-5)


def balanced(pred, y):
    cs = sorted(set(y.tolist()))
    return float(np.mean([np.mean(pred[y == c] == c) for c in cs]))


def main(views=60, seed=1, lights=tuple(LIGHTS), codes=False, dump=None):
    rng = np.random.default_rng(seed)
    w = W.G1World(seed=seed)
    m, d = w.m, w.d
    ey = E.Eyes(w, shadows="sun")
    seg = mujoco.Renderer(m, G.EYE_H, G.EYE_W)
    seg.enable_segmentation_rendering()
    UP = 4                                                             # 4 x the 336 px eye: the first build's 8 x its 168 (8 x would pass the 1,920 px framebuffer)
    seg8 = mujoco.Renderer(m, G.EYE_H * UP, G.EYE_W * UP)
    seg8.enable_segmentation_rendering()
    opt = G.eye_option()
    sun = [i for i in range(m.nlight) if m.light(i).name == "sun"][0]
    sun0 = (m.light_dir[sun].copy(), m.light_diffuse[sun].copy())
    toy_body = {t: m.body(f"toy_{t}").id for t in TOYS}
    head = m.body("parent_head").id
    mouth_geoms = {g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("parent_mouth")}
    cls_of_body = {b: i for i, b in enumerate(toy_body.values())}
    cls_of_body[head] = len(TOYS)
    base = w.save_state()
    # the views
    plan = []
    camL = m.camera("eye_L").id
    for v in range(views * len(CLASSES)):
        c = v % len(CLASSES)
        dist = rng.uniform(0.3, 1.2)
        yaw, pitch = math.radians(rng.uniform(-30, 30)), math.radians(rng.uniform(-15, 15))
        turn = (rng.uniform(-45, 45), rng.uniform(-20, 20)) if CLASSES[c] == "face" else None
        block = (TOYS[int(rng.integers(len(TOYS)))], rng.uniform(0.3, 0.9)) if CLASSES[c] == "face" and rng.random() < 1 / 3 else None
        plan.append((c, dist, yaw, pitch, turn, rng.normal(0, 1, 4), rng.normal(0, math.radians(AIM_ERR_DEG), 3), block))

    def place(c, dist, yaw, pitch, turn, q, block):
        w.load_state(base)
        R = d.cam_xmat[camL].reshape(3, 3)
        dirc = np.array([math.tan(yaw), math.tan(pitch), -1.0]); dirc /= np.linalg.norm(dirc)
        p = d.cam_xpos[camL] + R @ dirc * dist
        if CLASSES[c] == "face":
            x = d.cam_xpos[camL] - p; x /= np.linalg.norm(x)          # facing the eye, her crown toward the child's head (leaning
            z = np.array([-1.0, 0, 0]) - x * -x[0]; z /= np.linalg.norm(z)   # over it from its feet's side: upright in its eyes)
            x = rot(z, math.radians(turn[0])) @ x                       # turned about her up axis
            x = rot(np.cross(z, x), math.radians(turn[1])) @ x          # and nodded
            z = z - x * float(x @ z); z /= np.linalg.norm(z)
            Rh = np.column_stack([x, np.cross(z, x), z])
            centre = np.array([kin.head_surface_x(0, 0.15), 0.0, 0.15])
            w.scene.place_head(p - Rh @ centre, Rh)                     # her head there, her body straight under it (a still)
            if block is not None:                                       # a toy on the line between the eye and the mouth
                mujoco.mj_forward(m, d)
                mouth = E.mouth_point(m, d)[0]
                a = m.jnt_qposadr[m.body_jntadr[toy_body[block[0]]]]
                d.qpos[a:a + 3] = d.cam_xpos[camL] + (mouth - d.cam_xpos[camL]) * block[1]
        else:
            j = m.body_jntadr[toy_body[CLASSES[c]]]
            a = m.jnt_qposadr[j]
            d.qpos[a:a + 3] = p; d.qpos[a + 3:a + 7] = q / np.linalg.norm(q)
        mujoco.mj_forward(m, d)
        return p

    lights = {L: LIGHTS[L] for L in lights}
    labels, feats, facecmp = [], {k: [] for k in [(L, s) for L in lights for s in ("sun", "none")]}, []
    alt = {"raw grey pixels": [], "the bank's maps unpooled 4 x 4": []}
    raw_views = []
    t0 = time.perf_counter()
    for c, dist, yaw, pitch, turn, q, aim, block in plan:
        p = place(c, dist, yaw, pitch, turn, q, block)
        w.gaze = W.clamp_gaze(E.gaze_at(m, d, p) + aim)
        seg.update_scene(d, camera="eye_L", scene_option=opt)
        s = seg.render()
        x0, y0 = E.window_corner("L", w.gaze)
        win = s[y0:y0 + W.FOVEA_PX, x0:x0 + W.FOVEA_PX]
        counts = np.zeros(len(CLASSES), int); mouth_px = 0
        for gid, typ in win.reshape(-1, 2):
            if typ == int(mujoco.mjtObj.mjOBJ_GEOM) and gid >= 0:
                b = int(m.geom_bodyid[gid])
                if b in cls_of_body:
                    counts[cls_of_body[b]] += 1
                mouth_px += int(gid in mouth_geoms)
        lab = int(counts.argmax()) if counts.max() >= MIN_PX else -1
        labels.append(lab)
        if CLASSES[c] == "face":
            ft = E.face_test(m, d, w.gaze)["L"]
            mouth = E.mouth_point(m, d)[0]
            pr = E.project(m, d, "L", mouth)
            seg8.update_scene(d, camera="eye_L", scene_option=opt)
            s8 = seg8.render()
            r8, c8 = int(pr[1] * UP), int(pr[0] * UP)
            gid8, typ8 = s8[min(max(r8, 0), s8.shape[0] - 1), min(max(c8, 0), s8.shape[1] - 1)]
            visible = bool(typ8 == int(mujoco.mjtObj.mjOBJ_GEOM) and gid8 >= 0 and m.geom_bodyid[gid8] == head)
            in_win = x0 <= pr[0] < x0 + W.FOVEA_PX and y0 <= pr[1] < y0 + W.FOVEA_PX
            geo_ok = ft[1] not in ("not in the fovea", "turned away", "too small", "behind the eye")
            seg_verdict = bool(in_win and visible and counts[len(TOYS)] >= MIN_PX and ft[1] not in ("turned away",))
            facecmp.append((bool(ft[0]), seg_verdict, ft[1] != "blocked", visible, block is not None, geo_ok))
        for L, spec in lights.items():
            if spec is None:
                m.light_dir[sun], m.light_diffuse[sun] = sun0
            else:
                m.light_dir[sun] = np.asarray(spec["dir"]) / np.linalg.norm(spec["dir"]); m.light_diffuse[sun] = spec["diffuse"]
            for sh in ("sun", "none"):
                ey.set_shadows(sh)
                seen = ey.see()
                feats[(L, sh)].append(seen["eye_f"])
                if dump and sh == "sun" and L == next(iter(lights)):     # the raw foveae and colour window, for an offline look
                    raw_views.append(np.concatenate([seen["truth"]["fovea"][x].reshape(-1) for x in "LR"]
                                                    + [seen["truth"]["colour_window"].reshape(-1) / 255.0]))
                if codes and sh == "none" and L == next(iter(lights)):
                    fv = seen["truth"]["fovea"]
                    alt["raw grey pixels"].append(np.concatenate([fv[x].reshape(-1) for x in "LR"]))   # the grey foveae, 0-1
                    alt["the bank's maps unpooled 4 x 4"].append(np.concatenate([_bank_fine(fv[x]).reshape(-1) for x in "LR"]))
        m.light_dir[sun], m.light_diffuse[sun] = sun0
    secs = time.perf_counter() - t0
    y = np.array(labels)
    if dump:                                                            # the views' codes and labels, for an offline look
        np.savez_compressed(dump, y=y, raw=np.array(raw_views, dtype=np.float32), **{f"{L_}|{sh_}": np.array(v) for (L_, sh_), v in feats.items()})
    keep = y >= 0
    n = int(keep.sum())
    idx = np.where(keep)[0]
    perm = rng.permutation(idx)
    tr, te = perm[: int(len(perm) * 2 / 3)], perm[int(len(perm) * 2 / 3):]
    out = {"views": len(plan), "labelled": n, "label_counts": {CLASSES[k]: int((y == k).sum()) for k in range(len(CLASSES))},
           "none": int((y < 0).sum()), "seconds": round(secs, 1), "results": {}}
    todo = [(f"{k[0]}, shadows {k[1]}", F) for k, F in feats.items()]
    if codes:
        todo += [(f"{next(iter(lights))}, shadows none, {nm}", F) for nm, F in alt.items()]
    for key, F in todo:
        X = np.array(F)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - mu) / sd
        cs = sorted(set(y[tr].tolist()))
        C = np.stack([Xn[tr][y[tr] == c].mean(0) for c in cs])
        nm = np.array(cs)[((Xn[te][:, None, :] - C[None]) ** 2).sum(-1).argmin(1)]
        rd = ridge_acc(Xn[tr], y[tr], Xn[te], y[te], len(CLASSES), lam=float(len(tr)) * 0.1)
        mp = mlp_acc(Xn[tr], y[tr], Xn[te], len(CLASSES))
        # THE CORE'S VIEW (the W1 verifier's fourth round): the code through a fixed random projection into the cortex's d (256,
        # the channel's born code, from a seed) and the input's LayerNorm, then the same nonlinear readout
        Z = proj_ln(X)
        mz = mlp_acc(Z[tr], y[tr], Z[te], len(CLASSES))
        # HOW THE VERDICT DEPENDS ON THE VIEWS (the verifier's: 60 a class and 100 a class gave different verdicts): the readouts
        # trained on the first share of the training views (the same test views), by the training views a class
        curve = {}
        for share in CURVE_SHARES:
            sub = tr[: max(len(CLASSES), int(round(len(tr) * share)))]
            per = round(len(sub) / len(set(y[sub].tolist())), 1)
            curve[str(per)] = {"mlp": round(balanced(mlp_acc(Xn[sub], y[sub], Xn[te], len(CLASSES)), y[te]), 3),
                               "proj256_mlp": round(balanced(mlp_acc(Z[sub], y[sub], Z[te], len(CLASSES)), y[te]), 3)}
        out["results"][key] = {"nearest_mean": round(balanced(nm, y[te]), 3), "ridge": round(balanced(rd, y[te]), 3),
                               "mlp": round(balanced(mp, y[te]), 3), "proj256_mlp": round(balanced(mz, y[te]), 3),
                               "by_training_views_a_class": curve,
                               "face_mlp": round(float(np.mean(mp[y[te] == len(TOYS)] == len(TOYS))), 3) if (y[te] == len(TOYS)).any() else None,
                                                        "face_ridge": round(float(np.mean(rd[y[te] == len(TOYS)] == len(TOYS))), 3) if (y[te] == len(TOYS)).any() else None}
    if facecmp:
        a = np.array(facecmp)
        g = a[:, 5]                                                     # the views whose geometry passes (in the window, facing, big enough)
        out["face_test_vs_segmentation"] = {"views": len(a), "with_a_toy_on_the_line": int(a[:, 4].sum()),
                                            "test_agrees": round(float(np.mean(a[:, 0] == a[:, 1])), 3),
                                            "test_yes_render_no": int(np.sum(a[:, 0] & ~a[:, 1])), "test_no_render_yes": int(np.sum(~a[:, 0] & a[:, 1])),
                                            "geometry_passing": int(g.sum()),
                                            "ray_agrees_with_render_visibility": round(float(np.mean(a[g, 2] == a[g, 3])), 3) if g.any() else None,
                                            "blocked_by_ray": int(np.sum(g & ~a[:, 2])), "hidden_in_render": int(np.sum(g & ~a[:, 3]))}
    out["render_ms"] = round(1e3 * ey.timing["render_s"] / max(1, ey.timing["renders"]), 1)
    seg.close(); seg8.close(); ey.close()
    return out


def seg_face(m, d, seg, seg8, opt, gaze, head, mouth_geoms, up):
    """the segmentation render's verdict for the left eye: (the mouth point in the window, her head at the mouth point's pixel at
    8x, the face's pixels in the window at 1x, her MOUTH at that pixel: one of the drawn mouth's geoms, her lips, their opening or
    her teeth)"""
    mouth = E.mouth_point(m, d)[0]
    pr = E.project(m, d, "L", mouth)
    if pr is None:
        return False, False, 0, False
    x0, y0 = E.window_corner("L", gaze)
    seg.update_scene(d, camera="eye_L", scene_option=opt)
    s1 = seg.render()
    win = s1[y0:y0 + W.FOVEA_PX, x0:x0 + W.FOVEA_PX].reshape(-1, 2)
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    face_px = int(sum(1 for gid, typ in win if typ == geom and gid >= 0 and m.geom_bodyid[gid] == head))
    seg8.update_scene(d, camera="eye_L", scene_option=opt)
    s8 = seg8.render()
    r8, c8 = int(pr[1] * up), int(pr[0] * up)
    inside = 0 <= r8 < s8.shape[0] and 0 <= c8 < s8.shape[1]
    visible = mouth_seen = False
    if inside:
        gid8, typ8 = s8[r8, c8]
        visible = bool(typ8 == geom and gid8 >= 0 and m.geom_bodyid[gid8] == head)
        mouth_seen = bool(visible and mouth_geoms is not None and int(gid8) in mouth_geoms)
    in_win = x0 <= pr[0] < x0 + W.FOVEA_PX and y0 <= pr[1] < y0 + W.FOVEA_PX
    return bool(in_win), visible, face_px, mouth_seen


def c2c3(frames=2000, seed=1, lights=tuple(LIGHTS)):
    """C2 and C3 on babbled frames (see the module's doc)"""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from sim_babble import Babbler
    rng = np.random.default_rng(seed)
    w = W.G1World(seed=seed)
    m, d = w.m, w.d
    ey = E.Eyes(w, shadows="sun")
    UP = 4                                                             # 4 x the 336 px eye: the first build's 8 x its 168 (8 x would pass the 1,920 px framebuffer)
    seg = mujoco.Renderer(m, G.EYE_H, G.EYE_W); seg.enable_segmentation_rendering()
    seg8 = mujoco.Renderer(m, G.EYE_H * UP, G.EYE_W * UP); seg8.enable_segmentation_rendering()
    opt = G.eye_option()
    sun = [i for i in range(m.nlight) if m.light(i).name == "sun"][0]
    sun0 = (m.light_dir[sun].copy(), m.light_diffuse[sun].copy())
    head = m.body("parent_head").id
    toys = [m.body(f"toy_{t}").id for t in TOYS]
    mouth_geoms = {g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(("parent_mouth", "parent_lip_lo", "parent_teeth"))}
    camL = m.camera("eye_L").id
    b = Babbler(seed=seed, p_rest=0.6)
    rows, skipped = [], 0
    t0 = time.perf_counter()
    for k in range(frames):
        w.apply(b.acts())
        live = w.save_state()                                           # the babbling life goes on from here, untouched by the rig
        noface = rng.random() < 0.2                                     # a fifth of the frames: no face anywhere (C3's false alarms)
        R = d.cam_xmat[camL].reshape(3, 3)
        for _ in range(1000):                                           # her head inside the room (drawn again until it is)
            dist = rng.uniform(0.3, 3.5)
            yaw, pitch = math.radians(rng.uniform(-38, 38)), math.radians(rng.uniform(-20, 20))
            dirc = np.array([math.tan(yaw), math.tan(pitch), -1.0]); dirc /= np.linalg.norm(dirc)
            p = d.cam_xpos[camL] + R @ dirc * dist
            if noface or inside_room(p):
                break
        else:                                                           # its eyes face the floor or a wall too near: no room for a face
            skipped += 1
            w.load_state(live)
            continue
        turn, nod = rng.uniform(-90, 90), rng.uniform(-20, 20)
        block = rng.random() < 1 / 3
        aim = rng.normal(0, math.radians(3.0), 2)
        if noface:
            w.scene.place_head([0.0, 0.0, -5.0], np.eye(3))             # under the floor: no face (nor body) in the room
        else:
            x = d.cam_xpos[camL] - p; x /= np.linalg.norm(x)
            z = R @ np.array([0.0, 1.0, 0.0]); z = z - x * float(x @ z); z /= np.linalg.norm(z)   # upright in the eye's image
            x = rot(z, math.radians(turn)) @ x
            x = rot(np.cross(z, x), math.radians(nod)) @ x
            z = z - x * float(x @ z); z /= np.linalg.norm(z)
            Rh = np.column_stack([x, np.cross(z, x), z])
            local = np.array([kin.head_surface_x(0, kin.MOUTH_Z) + .0015, 0.0, kin.MOUTH_Z])
            w.scene.place_head(p - Rh @ local, Rh)                      # her head there, her body straight under it (a still)
        mujoco.mj_forward(m, d)
        if block and not noface:
            mouth = E.mouth_point(m, d)[0]
            tb = toys[int(rng.integers(len(toys)))]
            a = m.jnt_qposadr[m.body_jntadr[tb]]
            d.qpos[a:a + 3] = d.cam_xpos[camL] + (mouth - d.cam_xpos[camL]) * rng.uniform(0.3, 0.9)
            mujoco.mj_forward(m, d)
        mouth = E.mouth_point(m, d)[0]
        g = E.gaze_at(m, d, mouth)
        w.gaze = W.clamp_gaze([g[0] + aim[0], g[1] + aim[1], g[2]])
        ft = E.face_test(m, d, w.gaze)["L"]
        blocked_by = ""
        if ft[1] == "blocked":                                          # what the ray met first
            eye = d.cam_xpos[camL].copy(); to = mouth - eye
            gid = np.array([-1], dtype=np.int32)
            mujoco.mj_ray(m, d, eye, to / np.linalg.norm(to), E.EYE_GROUPS, 1, -1, gid)
            bb = int(m.geom_bodyid[gid[0]]) if gid[0] >= 0 else -1
            blocked_by = ("her own head" if bb == head else "a toy" if bb in toys else "the G1" if bb in w.scene.g1_set
                          else "her hands or arms" if bb >= 0 and m.body(bb).name.startswith("parent") else "the room")
            if bb == head:
                blocked_by += f" ({m.geom(int(gid[0])).name or 'unnamed'})"
        in_win, visible, face_px, mouth_seen = seg_face(m, d, seg, seg8, opt, w.gaze, head, mouth_geoms, UP)
        facing = ft[1] not in ("turned away", "behind the eye")
        seg_ok = bool(in_win and visible and facing and face_px >= E.FACE_MIN_PX) if not noface else False
        tmpl = {}
        for L, spec in {L_: LIGHTS[L_] for L_ in lights}.items():
            if spec is None:
                m.light_dir[sun], m.light_diffuse[sun] = sun0
            else:
                m.light_dir[sun] = np.asarray(spec["dir"]) / np.linalg.norm(spec["dir"]); m.light_diffuse[sun] = spec["diffuse"]
            seen = ey.see()
            tf = {x: E.face_template(seen["truth"]["fovea"][x]) for x in "LR"}          # the template, an instrument (C39 a)
            tmpl[L] = int(seen["face_fovea"][0]), int(E.template_match(tf["L"])), max(tf["L"][0], tf["R"][0])
        m.light_dir[sun], m.light_diffuse[sun] = sun0
        rows.append(dict(dist=dist, noface=noface, block=block, test=bool(ft[0]) and not noface, why=ft[1], by=blocked_by, seg=seg_ok, in_win=in_win,
                         mouth_seen=mouth_seen,
                         visible=visible, face_px=face_px, tmpl=tmpl))
        w.load_state(live)
    secs = time.perf_counter() - t0
    face = [r for r in rows if not r["noface"]]
    bins = [(0.3, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 3.5)]
    geo = [r for r in face if r["why"] in ("", "blocked")]             # the mouth in the window, the face turned toward the eye and big
                                                                        # enough by the test's own geometry: what the ray decides
    ray_ok = lambda r: r["why"] != "blocked"
    by = {}
    for lo, hi in bins:
        rr = [r for r in geo if lo <= r["dist"] < hi]
        if rr:
            by[f"{lo}-{hi} m"] = {"frames": len(rr), "ray_clear": sum(ray_ok(r) for r in rr), "render_visible": sum(r["visible"] for r in rr),
                                 "agree": round(float(np.mean([ray_ok(r) == r["visible"] for r in rr])), 3)}
    why = {}
    for r in face:
        why[r["why"] or "passes"] = why.get(r["why"] or "passes", 0) + 1
    out = {"frames": frames, "with_a_face": len(face), "no_face": len(rows) - len(face), "seconds": round(secs, 1),
           "head_inside_room": "every placement (drawn again until its centre is 0.15 m inside the walls, floor and ceiling)",
           "frames_with_no_room_for_a_face": skipped,
           "face_distances": {f"{lo}-{hi} m": sum(lo <= r["dist"] < hi for r in face) for lo, hi in bins},
           "C2": {"what": "the ray's verdict against the 8x segmentation render's (the mouth point visible), on the frames the test's "
                          "geometry passes (in the window, turned within 75 deg, big enough)",
                  "frames": len(geo), "ray_clear": sum(ray_ok(r) for r in geo), "ray_blocked": sum(not ray_ok(r) for r in geo),
                  "with_a_toy_on_the_line": sum(r["block"] for r in geo),
                  "agree": round(float(np.mean([ray_ok(r) == r["visible"] for r in geo])), 4) if geo else None,
                  "ray_clear_render_hidden": sum(ray_ok(r) and not r["visible"] for r in geo),
                  "ray_blocked_render_visible": sum(not ray_ok(r) and r["visible"] for r in geo),
                  "ray_blocked_render_visible_by": {k: sum(1 for r in geo if not ray_ok(r) and r["visible"] and r["by"] == k)
                                                    for k in sorted({r["by"] for r in geo if not ray_ok(r)})},
                  "ray_blocked_by": {k: sum(1 for r in geo if not ray_ok(r) and r["by"] == k) for k in sorted({r["by"] for r in geo if not ray_ok(r)})},
                  "note": "the render's verdict is the head's pixel at the mouth point: any part of her head, so a mouth hidden behind "
                          "her own nose or chin reads visible to it and blocked to the ray",
                  "agree_with_her_mouth_at_the_pixel": round(float(np.mean([ray_ok(r) == r["mouth_seen"] for r in geo])), 4) if geo else None,
                  "ray_clear_mouth_not_at_the_pixel": sum(ray_ok(r) and not r["mouth_seen"] for r in geo),
                  "ray_blocked_mouth_at_the_pixel": sum(not ray_ok(r) and r["mouth_seen"] for r in geo),
                  "by_distance": by,
                  "whole_test_vs_render": {"agree": round(float(np.mean([r["test"] == r["seg"] for r in face])), 4),
                                           "note": "the render's size criterion counts the whole head's pixels (hair included), the test "
                                                   "the face front's ellipse, so they part at the far end: informative only"},
                  "test_verdicts": why},
           "C3_template": {}}
    for L in lights:
        pos = [r for r in face if r["test"]]
        neg = [r for r in face if not r["test"]]
        nof = [r for r in rows if r["noface"]]
        hit = lambda rr: sum(r["tmpl"][L][0] for r in rr)
        out["C3_template"][L] = {"hits": f"{hit(pos)} of {len(pos)} frames the face test passes",
                                 "false_alarms_face_not_seen": f"{hit(neg)} of {len(neg)}",
                                 "false_alarms_no_face": f"{hit(nof)} of {len(nof)}",
                                 "best_r_where_the_test_passes": {q: round(float(np.percentile([r["tmpl"][L][2] for r in pos], k)), 3)
                                                                  for q, k in (("median", 50), ("p90", 90), ("max", 100))} if pos else None,
                                 "hits_by_distance": {f"{lo}-{hi} m": f"{hit([r for r in pos if lo <= r['dist'] < hi])} of "
                                                      f"{len([r for r in pos if lo <= r['dist'] < hi])}" for lo, hi in bins}}
    out["render_ms"] = round(1e3 * ey.timing["render_s"] / max(1, ey.timing["renders"]), 1)
    seg.close(); seg8.close(); ey.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--views", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lights", default=",".join(LIGHTS))
    ap.add_argument("--codes", action="store_true")
    ap.add_argument("--dump", default=None, help="write the views' codes and labels to this .npz")
    ap.add_argument("--c2", type=int, default=0, help="C2 and C3 on this many babbled frames instead of the identity check")
    a = ap.parse_args()
    if a.c2:
        print(json.dumps(c2c3(a.c2, a.seed, tuple(a.lights.split(","))), indent=1))
    else:
        print(json.dumps(main(a.views, a.seed, tuple(a.lights.split(",")), a.codes, a.dump), indent=1))
