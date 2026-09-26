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


def build(args):
    world = W.G1World(seed=args.seed)
    eyes = E.Eyes(world) if not args.no_eyes else None
    if args.voice == "real":
        from body.sim.voice import synth as V
        voice = V.VoiceCache(os.path.join(args.out, "voice"))
    else:
        voice = FakeVoice()
    lane = LN.ParentLane(world, seed=args.seed, voice=voice, day_ticks=int(SIM_CFG.get("wake_ticks", 24000)))
    cfg = dict(SIM_CFG, **(LR0 if args.lr0 else {}))
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=[float(x) for x in world.tau_max])
    life_path = os.path.join(args.out, "life.pt")
    if args.resume:
        with open(os.path.join(args.out, "world.pt"), "rb") as f:
            world.load_state(pickle.load(f))
        L = Life.load(life_path, anat, save_path=life_path, world=world)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--ticks", type=int, default=0, help="stop after this many ticks (0: the days)")
    ap.add_argument("--lr0", action="store_true", help="every learning rate 0 (S5b's plumbing)")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--voice", choices=("real", "fake"), default="real")
    ap.add_argument("--no-eyes", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--page", action="store_true", help="serve the /sim page on http://127.0.0.1:8030/ (tools/sim_page.py)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)
    t_build = time.time()
    world, eyes, lane, L = build(args)
    run = WorldLoop(L)
    snap = None
    if args.page:
        import sim_page
        snap = sim_page.Snapshot(world, lane, L, eyes)
        sim_page.serve(snap)
        print(f"the page: http://127.0.0.1:{sim_page.PORT}/", flush=True)
    day_ticks = int(SIM_CFG.get("wake_ticks", 24000)) + int(SIM_CFG.get("night_ticks", 24000))
    total = args.ticks or int(args.days * day_ticks)
    log = open(os.path.join(args.out, "ticks.jsonl"), "a")
    agg = collections.Counter(); walls = collections.defaultdict(list); prev_reading = 0.0
    print(f"built in {time.time() - t_build:.1f} s; living {total} ticks", flush=True)
    t0 = time.time()
    for k in range(total):
        tm = dict(world.timing); rs = 0.0 if eyes is None else eyes.timing["render_s"]
        a = time.perf_counter()
        run.step()
        wall = time.perf_counter() - a
        if snap is not None:
            snap.take(INV)
        night = bool(world.night)
        f = world.now if getattr(world, "now", None) is not None else None
        pain = f.obs.get("pain") if f is not None else None
        rec = dict(t=int(world.tick), day=int(lane.day), night=night, ms=round(1000 * wall, 1),
                   apply_ms=round(1000 * (world.timing["apply_s"] - tm.get("apply_s", 0.0)), 1),
                   render_ms=round(1000 * ((0.0 if eyes is None else eyes.timing["render_s"]) - rs), 1),
                   stress=round(float(L.stress), 3), mood=round(float(L.mood), 3), cry=bool(world.crying),
                   pain=None if pain is None else [int(i) for i in np.nonzero(pain)[0]])
        if not night:
            ls = lane.last
            rec.update(line=ls.get("line"), ep=None if lane.plan is None else lane.plan.kind, word=INV.get(int(ls.get("word", 0))),
                       heard=ls.get("heard"), reading=round(float(lane.reading), 2), judged=ls.get("judged"),
                       events=[e[0] for e in ls.get("events", ())], face_test=bool(ls.get("face_test")),
                       seen_by_child=bool(ls.get("seen_by_child")),
                       gates=[round(float((st.get("now") or {}).get("p_act", 0.0)), 3) for st in L.motor],   # each effector's p_act (her rulers'
                                                                                                             # partner: how much it acts)
                       her_at=[round(float(x), 2) for x in world.parent.base["at"]] + [str(world.parent.base.get("mode"))],   # where she is
                       acts_open=[[a_[1], a_[2], a_[5]] for a_ in lane.conduct.acts_open][:6],               # her acts under way (kind, target, status)
                       refused=(list(lane.conduct.fast.refused[-1]) if lane.conduct.fast.refused and lane.conduct.fast.refused[-1][0] >= world.tick - 1 else None),
                       present=bool(ls.get("present")), holds=list(ls.get("holds") or ()),
                       token=None if world.words_out is None else INV.get(int(world.words_out)),
                       onset=None if f is None else int(f.obs.get("onset_periph", [0])[0]),
                       sounds=len(getattr(world.sounds, "last_events", [])))
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
