"""THE G1'S LIFE RUNNER (S5b and birth; docs/SIM_DESIGN.md 11): the core (body/sim/anatomy.SimAnatomy under SIM_CFG, the parent's 50
birth words) living in the G1's world with every part attached: the eyes (the D435's three views, A78, A84), the room's sounds (A83),
the observer (A81), the parent's lane with her conduct, voice, feelings, face and day plan (A75, A77). One tick at a time through
the core's world loop; its nights are the core's (the live, dark night: A46, R8).

THE SAVE: the life saves itself after each night, between ticks (C74); this runner writes the world beside it in the same call
(world.pt: the physics, the parent, the lane, the eyes' and the sounds' memories, every stream), so the pair is one moment and a
resume continues exactly. `--resume DIR` loads the pair and lives on.

THE LOG (DIR/ticks.jsonl, one line a tick): the tick, the day, day or night, the wall time (the tick, the world's apply, the eyes'
render, the lane), its stress and mood, pain (joints, base), the cry, her line and her episode, the words channel, the
child's token and what her ear heard, the visual onset, the room's sound events, the face reading. DIR/report.json at each night and
at the end: rates, means and the night's report. Nothing here is read by the body; it is the experimenter's.

Run (S5b, every learning rate 0):  nice -n 10 python3 tools/sim_life.py --out DIR --days 2 --lr0 [--d 512] [--voice fake]
Birth (seed 1, the born learning rates): the same without --lr0, --out the life's own folder, never from a plumbing run's state.
"""
import argparse
import collections
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from body.life import Life  # noqa: E402
from body.core.world import WorldLoop  # noqa: E402
from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table  # noqa: E402
from body.sim import g1scene as G  # noqa: E402
from body.sim import world as W  # noqa: E402
from body.sim import eyes as E  # noqa: E402
from body.sim import lane as LN  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402

LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
           act_inv_lr=0.0)
INV = {int(LN.LX_TO_AN[i]): s for i, s in enumerate(LX.TABLE)}


class FakeVoice:
    """a stand-in voice (tests, plumbing without the speech engine): each word 2 ticks of a tone, its marks as the playback reads them"""

    def clip(self, text, register="plain", emphasis=None, heard=True, shape=None, **kw):
        import hashlib
        from body.sim.voice import synth as V
        ws = [w.strip(".,!?") for w in text.split() if w.strip(".,!?")]
        n = 2 * len(ws) * 2400
        pcm = (6000 * np.sin(2 * np.pi * 200 * np.arange(n) / 16000)).astype(np.int16)
        words = [(w, 2 * i * 2400, (2 * i + 2) * 2400) for i, w in enumerate(ws)]
        key = hashlib.sha256((text + "|" + register).encode()).hexdigest()
        return V.Clip(key, text, register, pcm, words, key)


_ARM_IDX = None
_NOV_LAST = 0.0


def _stops(world, margin=0.02):
    """C160 (2026-09-30): THE ARM JOINTS AT THEIR RANGE STOPS, a ruler: for each arm the count of its joints within `margin` of a range end
    (world.lo, world.hi in JOINTS order). Why: at day 42's tick 20,000 six of the fourteen arm joints sat at a stop with the child prone; a
    joint at a stop shows the inverse model nothing of its acts (kappa near chance), its readout stays flat (C148) and it keeps drawing big
    steps, and the pinned hand's wrist bears the swings. Read against pain and posture by the day"""
    global _ARM_IDX
    if _ARM_IDX is None:
        js = list(W.JOINTS)
        _ARM_IDX = [np.array([js.index(j) for j in dict(G.EFFECTORS)[e]]) for e in ("arm_l", "arm_r")]
    out = []; masks = []
    for idx in _ARM_IDX:
        q = world.d.qpos[world.qadr[idx]]
        s = (q - world.lo[idx]) / np.maximum(world.hi[idx] - world.lo[idx], 1e-9)
        at = (s < margin) | (s > 1.0 - margin)
        out.append(int(np.sum(at)))
        masks.append(int(sum((1 << j) for j, a_ in enumerate(at) if a_)))   # C161: which joints, a 7-bit mask in the effector's joint order
    return out, masks


def build(args):
    extra = None
    if args.extra:                                                          # A115: things added to the room (body/sim/extras.py); a pair
        from body.sim import extras as X                                    # saved before them is carried across by tools/sim_migrate_world.py
        extra = X.EXTRAS[args.extra]()
    world = W.G1World(seed=args.seed, extra=extra, xml=G.scene_path(args.room, args.body))   # A133/A134: the room's layout and the body
    eyes = E.Eyes(world) if not args.no_eyes else None
    if args.voice == "real":
        from body.sim.voice import synth as V
        voice = V.VoiceCache(os.path.join(args.out, "voice"))
    else:
        voice = FakeVoice()
    lane = LN.ParentLane(world, seed=args.seed, voice=voice, day_ticks=int(SIM_CFG.get("wake_ticks", 24000)))
    cfg = dict(SIM_CFG, **(LR0 if args.lr0 else {}))
    for kv in args.set:                                                      # a copy's switch (never the life's: its cfg is the tree's)
        k, v = kv.split("=", 1)
        cur = cfg.get(k, 0)
        cfg[k] = type(cur)(float(v)) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else v
        print(f"cfg {k} = {cfg[k]!r} (--set)", flush=True)
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=[float(x) for x in world.tau_max])
    life_path = os.path.join(args.out, "life.pt")
    if args.resume:
        with open(os.path.join(args.out, "world.pt"), "rb") as f:
            world.load_state(pickle.load(f))
        if world.dawn_left == W.DAWN_TICKS and world.carry_to_mat():       # A110: a pair saved at a dawn before A110 (the child asleep
            print(f"the child carried to the mat at this dawn (A110): {world.carried[-1]}", flush=True)   # off the mat): carried now
        if world.dawn_left == W.DAWN_TICKS and world.tidy_toys():          # A117 (B8): a pair saved at a dawn before A117: the lost toys
            print(f"lost toys put back at this dawn (B8, A117): {world.tidied[-3:]}", flush=True)   # put back now
        L = Life.load(life_path, anat, cfg=cfg, save_path=life_path, world=world)   # A101: a resumed life lives under the TREE's constants
                                                                                    # (SIM_CFG, and --lr0): until A101 the save's cfg ruled
                                                                                    # and no switch turned after birth could reach a life
    else:
        big = args.d >= 256
        L = Life.birth(anat, device="cpu", d=args.d, layers=6 if big else 2, heads=8 if big else 2, window=64 if big else 16, cfg=cfg,
                       seed=args.seed, save_path=life_path, world=world)
    save = L.save

    def save_pair(path=None, save=save, world=world):
        out = save(path)                                              # the life's own, between ticks (C74)
        wp = os.path.join(args.out, "world.pt")
        with open(wp + ".tmp", "wb") as f:
            pickle.dump(world.save_state(), f)
        os.replace(wp + ".tmp", wp)                                   # the world beside it, the same moment
        return out
    L.save = save_pair
    return world, eyes, lane, L


def _ended(lane, world, open_seen):
    """A102: the acts that ended this tick, [kind, target, status, why]; open_seen keeps each act's kind and target by its motion id
    while it is open (the conduct drops an ended act from acts_open the tick it ends)"""
    for a_ in lane.conduct.acts_open:
        open_seen[a_[0]] = (a_[1], a_[2])
    out = []
    for mid, st in (getattr(lane.conduct, "ended", None) or {}).items():
        kind, target = open_seen.pop(mid, ("?", None))
        try:
            why = str(world.parent.why(mid) or "")[:120]
        except Exception:
            why = ""
        out.append([kind, target, str(st), why])
    return out


def _last_count(path, key):
    """the record's last row's running count `key` (the runner's own counters, carried over a resume: C125), 0 when none"""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - (1 << 20)))
            rows = [r for r in f.read().split(b"\n") if r.strip()]
        return int(json.loads(rows[-1]).get(key, 0)) if rows else 0
    except Exception:
        return 0


def _cut_record_tail(path, tick):
    """the record (ticks.jsonl) cut back to the rows before `tick`: a resume from a mid-day checkpoint lives the ticks after it again, and
    the rows written the first time would be counted twice by every reader. The file is append-only; the last rows are scanned from
    the end and the file truncated at the first row whose t > tick (the pair's own tick was lived once; nothing to cut when none)"""
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    if size == 0:
        return
    with open(path, "rb+") as f:
        back = min(size, 64 << 20)                                     # the last 64 MB: more than a day of rows
        f.seek(size - back)
        buf = f.read(back)
        start = 0 if back == size else buf.index(b"\n") + 1            # the first whole row in the window
        cut = None
        pos = start
        while pos < len(buf):
            end = buf.find(b"\n", pos)
            if end < 0:
                end = len(buf)
            row = buf[pos:end]
            try:
                t = int(json.loads(row)["t"])
            except Exception:
                t = None
            if t is not None and t > tick and cut is None:
                cut = pos
            if t is not None and t <= tick:
                cut = None                                               # an older row after a newer one: only the final run of rows past the tick is cut
            pos = end + 1
        if cut is not None:
            f.truncate(size - back + cut)
            print(f"the record cut back to the pair's tick {tick} ({size - (size - back + cut)} bytes of rows lived again dropped)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--ticks", type=int, default=0, help="stop after this many ticks (0: the days)")
    ap.add_argument("--probe-joint", type=str, default=None, metavar="J[,J...]",
                    help="an instrument (C113): the record row gets pj for BODY_JOINTS[J]: [angle, low stop, high stop, the servo's target, its torque, the gear's sensed 10 ms peak load, the pain line]")
    ap.add_argument("--lr0", action="store_true", help="every learning rate 0 (S5b's plumbing)")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--voice", choices=("real", "fake"), default="real")
    ap.add_argument("--no-eyes", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=4000, metavar="N",
                    help="the pair saved every N waking ticks besides the night's own save (0: the night alone): a fix lands at the next checkpoint, not the next dawn (the owner's word 2026-09-29)")
    ap.add_argument("--page", action="store_true", help="serve the /sim page on http://127.0.0.1:8030/ (tools/sim_page.py)")
    ap.add_argument("--extra", default=None, help="a thing added to the room (body/sim/extras.py: book); the pair must have been migrated to it")
    ap.add_argument("--room", default="a", choices=("a", "b"), help="the room's layout (A133): a, the room of birth; b, its furniture moved")
    ap.add_argument("--body", default="a", choices=("a", "b"), help="the body (A134): a, the stock G1; b, longer forearms and shanks, heavier limbs")
    ap.add_argument("--film", default=None, metavar="DIR", help="A123: the room's view saved as JPEG frames in DIR, one every --film-every ticks (the page's render; the captions come from ticks.jsonl by tick)")
    ap.add_argument("--film-every", type=int, default=3, help="ticks between frames (a multiple of the page's 3)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="a cfg key set for this run (a measurement on a copy: e.g. --set actor=1); typed as the tree's own value")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)
    t_build = time.time()
    world, eyes, lane, L = build(args)
    run = WorldLoop(L)
    snap = None
    if args.page or args.film:
        import sim_page
        snap = sim_page.Snapshot(world, lane, L, eyes)
        if args.page:
            sim_page.serve(snap)
            print(f"the page: http://127.0.0.1:{sim_page.PORT}/", flush=True)
    if args.film:
        os.makedirs(args.film, exist_ok=True)                                  # A123: the film's frames (never in git: data/)
        print(f"the film: frames every {args.film_every} ticks in {args.film}", flush=True)
    day_ticks = int(SIM_CFG.get("wake_ticks", 24000)) + int(SIM_CFG.get("night_ticks", 24000))
    total = args.ticks or int(args.days * day_ticks)
    if args.resume:
        _cut_record_tail(os.path.join(args.out, "ticks.jsonl"), int(world.tick))   # the rows lived past the pair's tick are lived again: dropped
        world.stats_tendon = _last_count(os.path.join(args.out, "ticks.jsonl"), "tendon_n")   # C125: the tendon reflex's count so far, kept over a resume (an instrument's; the reflex is the world's)
    log = open(os.path.join(args.out, "ticks.jsonl"), "a")
    agg = collections.Counter(); walls = collections.defaultdict(list); prev_reading = 0.0
    nights_seen = [int(getattr(L, "nights", 0))]                       # C147: the nights whose reports are kept in nights.jsonl
    open_seen = {}                                                        # A102: each act's kind and target by its motion id, for its end
    print(f"built in {time.time() - t_build:.1f} s; living {total} ticks", flush=True)
    t0 = time.time()
    for k in range(total):
        tm = dict(world.timing); rs = 0.0 if eyes is None else eyes.timing["render_s"]
        a = time.perf_counter()
        run.step()
        wall = time.perf_counter() - a
        if args.save_every and not bool(world.night) and k > 0 and int(world.tick) % int(args.save_every) == 0:
            t_sv = time.perf_counter(); L.save(); print(f"checkpoint: the pair saved at tick {world.tick} ({time.perf_counter() - t_sv:.1f} s)", flush=True)
            log.flush()                                                # C135: the record flushed at the save (a runner stopped after a night's save lost the day's last 66 rows: the night writes none, the buffer held them)
        if snap is not None:
            snap.take(INV)
            if args.film and int(world.tick) % args.film_every == 0 and snap.room_jpg:
                with open(os.path.join(args.film, f"{int(world.tick):08d}.jpg"), "wb") as fj:
                    fj.write(snap.room_jpg)
        night = bool(world.night)
        f = world.now if getattr(world, "now", None) is not None else None
        pain = f.obs.get("pain") if f is not None else None
        rec = dict(t=int(world.tick), day=int(lane.day), night=night, ms=round(1000 * wall, 1),
                   apply_ms=round(1000 * (world.timing["apply_s"] - tm.get("apply_s", 0.0)), 1),
                   render_ms=round(1000 * ((0.0 if eyes is None else eyes.timing["render_s"]) - rs), 1),
                   stress=round(float(L.stress), 3), mood=round(float(L.mood), 3), cry=bool(world.crying),
                   pain=None if pain is None else [int(i) for i in np.nonzero(pain)[0]])
        rec["acts"] = {k: int(v) for k, v in (getattr(world, "_last_acts", None) or {}).items()}   # C117's instrument: the effectors' flat acts this tick
        rec["ptop"] = [[round(float(p_.max()), 2) for p_ in ((st.get("now") or {}).get("probs") or [])] for st in L.motor]   # C117: each joint's top probability
        rec["pbig"] = [[round(float(p_[0] + p_[-1]), 3) for p_ in ((st.get("now") or {}).get("probs") or [])][5:7] for st in L.motor[3:5]]   # C146: each arm's
                                                                                    # actor's probability of a big step (either way) at its wrist pitch and yaw: the
                                                                                    # wrist pain's ruler (big wrist steps precede 40% of it; days 29 to 39: 10.6% to 0.9%)
        rec["kj"] = [[round(float(k_), 2) for k_ in (st.get("inv_kappa") or [])] for st in L.motor[3:5]]   # C146: each arm's inverse model's
        rec["habit_w"] = getattr(L, "_habit_w", None)                        # A141 (C149): the last habit lesson's mean weight and its share of
        rec["actor_upd"] = [[round(float(st.get("a_upd", [0, 0])[0]), 2), round(float(st.get("a_upd", [0, 0])[1]), 2)] for st in L.motor[3:5]]   # A142:
                                                                            # each arm's actor's summed update norms, the fast lesson's and the tag's
                                                                            # positions under 1 (acts followed by net harm, not cloned)
                                                                                    # kappa per joint (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw): which
                                                                                    # joints it labels; the arms' mean sat at 0.2 to 0.35 for eleven days, under A104's "fair"
        rec["spinal"] = dict(getattr(world, "_spinal", {}) or {})          # A139's instrument (C117): the cord's events this tick (grasp,
        ferr = getattr(L, "_ferr", None) or {}                              # C137: the body's raw forecasting error per channel, its running
        rec["ferr"] = {k: round(float(v[1]), 4) for k, v in ferr.items()}    # mean (frames._frame_surprise's mu): the learning curve of its
        rec["ferr_fast"] = {k: round(float(v), 4) for k, v in (getattr(L, "_ferr_fast", None) or {}).items()}   # C147: the same error over the
                                                                            # last 512 ticks (ferr's horizon is 36,000): a dusk against the next dawn reads the night
        rec["tendon_n"] = int(getattr(world, "stats_tendon", 0))           # prone, tendon per effector) and the tendon reflex's joint count so far
        if args.probe_joint is not None:                                   # C113's instrument: joints against their stops and their gears
            d_ = world.d; pjs = [int(x) for x in str(args.probe_joint).split(",")]
            rows_ = [[round(float(d_.qpos[world.qadr[J_]]), 3), round(float(world.lo[J_]), 3), round(float(world.hi[J_]), 3),
                      round(float(d_.ctrl[world.aid[J_]]), 3), round(float(d_.qfrc_actuator[world.dof[J_]]), 2),
                      round(float(world._sensed["bd_peak"][J_]), 2), round(float(world.tau_hold[J_]), 2)] for J_ in pjs]
            rec["pj"] = rows_[0] if len(rows_) == 1 else rows_
        if not night:
            ls = lane.last
            rec.update(line=ls.get("line"), ep=None if lane.plan is None else lane.plan.kind, word=INV.get(int(ls.get("word", 0))),
                       heard=ls.get("heard"), reading=round(float(lane.reading), 2), judged=ls.get("judged"),
                       reward=(round(float(L._rec[int(L._rec_n) - 1][3]), 3) if getattr(L, "_rec", None) is not None and int(getattr(L, "_rec_n", 0)) > 0 else 0.0),   # the tick's net reward
                       nov=next((int(getattr(s_, "n_paid", 0)) for s_ in L.anatomy.rewards if s_.name == "novelty"), 0),   # the novelty drive's payments so far (A127)
                       nov_paid=next((round(float(getattr(s_, "paid", 0.0)), 2) for s_ in L.anatomy.rewards if s_.name == "novelty"), 0.0),   # and their sum (A127b)
                       imag=int(getattr(L, "_imag_n", 0)),                          # A138: the waking imaginings so far (an event's end or a pause)
                       ends=len(getattr(L, "_rec_ends", None) or []),                # the frames' event ends so far this day (R7f's WM latch, A138's trigger)
                       imag_N=(round(float(L._imag_N), 3) if getattr(L, "_imag_N", None) is not None else None),   # the imagined future's valence, fading
                       events=[e[0] for e in ls.get("events", ())], face_test=bool(ls.get("face_test")),
                       seen_by_child=bool(ls.get("seen_by_child")),
                       gates=[round(float((st.get("now") or {}).get("p_act", 0.0)), 3) for st in L.motor],   # each effector's p_act (her rulers'
                                                                                                             # partner: how much it acts)
                       pmax=[round(float(np.mean([float(p_.max()) for p_ in ((st.get("now") or {}).get("probs") or [])] or [0.0])), 3)
                             for st in L.motor],                                                             # A97: each effector's decisiveness
                       sharp=[round(float((st.get("now") or {}).get("sharp", 0.0)), 2) for st in L.motor],  # (its joints' mean top probability)
                       kappa=[round(float(st.get("inv_gain", 0.0) or 0.0), 3) for st in L.motor],           # and its inverse model's reliability
                       her_at=[round(float(x), 2) for x in world.parent.base["at"]] + [str(world.parent.base.get("mode"))],   # where she is
                       acts_open=[[a_[1], a_[2], a_[5]] for a_ in lane.conduct.acts_open][:6],               # her acts under way (kind, target, status)
                       acts_ended=_ended(lane, world, open_seen),                                              # and those that ended this tick, with why
                       refused=(list(lane.conduct.fast.refused[-1]) if lane.conduct.fast.refused and lane.conduct.fast.refused[-1][0] >= world.tick - 1 else None),
                       present=bool(ls.get("present")), holds=list(ls.get("holds") or ()),
                       child=[round(float(x), 2) for x in world.parent.child.torso[:2]] + [str(lane.posture)],   # A119 (C103): where it lies, its posture as the lane reads it
                       token=None if world.words_out is None else INV.get(int(world.words_out)),
                       covert=(INV.get(int(L._last_choice["top"])) if (getattr(L, "_last_choice", None) and not L._last_choice.get("acted")
                               and float(L._last_choice.get("p_top", 0.0)) >= 0.5) else None),   # A137: the inner word (sure, unsounded)
                       onset=None if f is None else int(f.obs.get("onset_periph", [0])[0]),
                       sounds=len(getattr(world.sounds, "last_events", [])))
            rec["stage"] = int(lane.conduct.stage)
            rec["stops"], rec["stopj"] = _stops(world)                   # C160/C161: the arm joints at their range stops, per arm: the count and the mask
            global _NOV_LAST
            nov_now = float(rec.get("nov_paid") or 0.0)
            if nov_now > _NOV_LAST + 1e-9 and getattr(L, "_ferr_now", None):   # C171: a novelty payment this tick: each channel's surprise then
                rec["nov_ch"] = {k: round(float(v), 4) for k, v in L._ferr_now.items()}
            _NOV_LAST = nov_now
            if world.tick % 16 == 0:                                        # C165: THE LADDER READ, every 16 ticks: each band's gate (sigmoid of its
                with torch.no_grad():                                       # Go/NoGo on its own state), its value, and its TD error this tick
                    rec["bands"] = dict(g=[round(float(torch.sigmoid(L.m.band_gate[b](L.bands[b]))), 3) for b in range(len(L.m.clocks))],
                                        v=[round(float(x), 3) for x in L.m.values(L.bands).tolist()],
                                        td=[round(float(x), 4) for x in (getattr(L, "_td_last", None) or [])])
            if ls.get("found"):
                rec["found"] = ls["found"]                                  # C158: a find this tick, [toy, whether the child saw the hide]                      # C142: her stage (1: the vocal turn smiled; 2: the words her ear accepts, the frowns)
            wh_ = getattr(L, "_whit", None)                             # C143: the words' forecast on her symbols heard: its top-1 hit rate and the
            rec["whit"] = None if not wh_ else [round(float(wh_[1]), 4), round(float(wh_[2]), 4)]   # probability it gave the symbol that came
        log.write(json.dumps(rec) + "\n")
        agg["ticks"] += 1; agg["night"] += night; agg["cry"] += rec["cry"]
        agg["pain"] += bool(rec["pain"]); agg["lines"] += bool(rec.get("line")); agg["onset"] += bool(rec.get("onset"))
        for w_, kind_, _o in (rec.get("judged") or ()):                    # her judgments by kind (A89: her rulers), and her smiles
            agg[f"j_{kind_}"] += 1                                          # the child saw (its reading rising above 0)
        if rec.get("reading", 0.0) > 0.0 and prev_reading <= 0.0:
            agg["smiles_seen"] += 1
        prev_reading = float(rec.get("reading", 0.0))
        walls["night" if night else "day"].append(wall)
        if k % 500 == 0 or k == total - 1:
            d_ = np.mean(walls["day"][-500:]) if walls["day"] else 0.0
            n_ = np.mean(walls["night"][-500:]) if walls["night"] else 0.0
            print(f"tick {world.tick} day {lane.day} {'night' if night else 'day'}: {1000 * d_:.0f} ms a day tick, {1000 * n_:.0f} a night"
                  f" tick; stress {L.stress:.2f} mood {L.mood:.2f}; {dict(agg)}", flush=True)
            log.flush()
            if int(getattr(L, "nights", 0)) != nights_seen[0] and getattr(L, "last_night", None):   # C147: every night's report kept
                nights_seen[0] = int(getattr(L, "nights", 0))                                        # (report.json holds only the last):
                with open(os.path.join(args.out, "nights.jsonl"), "a") as fn:                        # REM's and the store's history
                    fn.write(json.dumps(dict(L.last_night, saved_at=time.strftime("%Y-%m-%d %H:%M"), tick=int(world.tick)), default=str) + "\n")
            rep = dict(ticks=agg["ticks"], wall_s=round(time.time() - t0, 1), counts=dict(agg),
                       day_tick_ms=round(1000 * float(np.mean(walls["day"])), 1) if walls["day"] else None,
                       night_tick_ms=round(1000 * float(np.mean(walls["night"])), 1) if walls["night"] else None,
                       nights=int(getattr(L, "nights", 0)), last_night={k_: v_ for k_, v_ in (getattr(L, "last_night", None) or {}).items()
                                                                         if isinstance(v_, (int, float, str, bool))},
                       last_night_sweep=(getattr(L, "last_night", None) or {}).get("sweep"),          # A93's reverse sweep: chunks, ticks, windows
                       last_night_live={k_: v_ for k_, v_ in ((getattr(L, "last_night", None) or {}).get("live") or {}).items()
                                        if isinstance(v_, (int, float, str, bool))},
                       plan=[list(x)[:4] for x in (lane.plan.log[-20:] if lane.plan else [])])
            with open(os.path.join(args.out, "report.json"), "w") as fr:
                json.dump(rep, fr, indent=1, default=str)
    log.close()
    print("done", json.dumps(dict(agg)), flush=True)


if __name__ == "__main__":
    main()
