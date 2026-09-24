"""WHAT THE PARENT PERCEIVES (docs/SIM_DESIGN.md 4.5: "filled from what the parent can see: what the child looks at, holds, or just
said"; 4.10's joint attention; A14's "what cannot be shown"; A40; package P3). The fast layer reads the moment only through a
Percept, and the line check holds every line to it: she names only what a person in her place could see or hear.

The world (W2-W3) fills one each tick from her senses, never from anything a person could not know:
  present         she is with the child (not away in the hall)
  child_in_view   she can see the child (its body, for "your foot")
  seen_by_child   she is where the child's eyes can reach her (a person knows when a baby cannot see her: behind its head)
  child_target    where she reads the child looking (A40): the nameable object nearest the line its head's camera faces,
                  within the fovea's reach of that line (+-38 x +-20 degrees), the line read with a person's error (Reader.look);
                  "mama" when that is her own face; or None. A real G1 shows no eyes, so she reads its head (on the G1, its
                  trunk: no neck, A22), never its software fovea's window, which stays the child's own and the instruments'
  child_holds     the object ids in its hands, as she sees them
  child_reaches   the object ids its hands reach toward, as she sees them (Reader.reaches: a hand's path over the last 3 ticks
                  closing on the object), none it already holds
  seen            the objects she can see now (the world's ray test from her eyes), each with its word, its colour and where it
                  rests as she sees it; child_sees marks those in the child's view as she reads it: before its head's camera about
                  the line she reads (Reader.look), never the render (the ledger's "heard"; a person can tell whether a toy is
                  before a baby's face)
  fixtures        the room's words she can see ("mat", "sofa", "window", and the growth queue's "table", "shelf", ...)
  events          what she saw or heard happen this tick: (kind, object id or None), kind one of EVENT_KINDS
  child_sounding  she hears the child's voice this tick (the transcriber's own reading of it)
None of these is the child's inside: no gate, value, forecast or feeling reaches her (A14's rule for Claude holds for her too),
and none is its fovea's window.

Reader is her reading of the child's head and hands (A40): the world gives it only what a person in her place sees (the head's
pose, the hands' places, the things' places), and it gives the world child_target, child_reaches and each thing's child_sees. Its
error is drawn from her own seeded stream (the body's seed, spawn key READ_STREAM), two draws a reading whatever is in view, so a
replay is exact; state() and load_state() carry the stream and the hands' paths.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import consts as K

EVENT_KINDS = (
    "fell",          # a toy dropped or fell (the object)
    "rolled",        # the child rolled (a whole roll, A8)
    "sat",           # the child came to sit
    "got",           # the child took hold of a toy itself (the object)
    "gave",          # the child put a toy in her hand (the object)
    "hit_her",       # the child's act struck her above her own pain threshold (4.10)
    "reflex_hit",    # a reflex she triggered struck her (logged as her defect, never frowned at)
    "pain",          # the child's pain (a thump or a pain event she saw)
    "distress",      # A13's outward signs: face down over 100 ticks, thumps, the charge light low
    "charge_low",    # the charge light is low (h < 0.35: the meal, 4.7)
    "talk_over",     # the child started sounding during her line
    "lost_toy",      # a toy just fell from its hand
    "arm_raise",     # its arm raised, as she sees it (the side, "left" or "right", as object): she may copy it (A52)
    "wave",          # its hand waved (the side)
    "shake",         # its hand shook (the side; a toy in it or not)
    "open_hand",     # its hand opened (the side)
)


@dataclass(frozen=True)
class Seen:
    id: str                       # the world's object id ("ball", "ball_blue")
    name: str                     # its word ("ball")
    colour: str = ""              # as she sees it ("red")
    on: str = ""                  # what it rests on as she sees it: a fixture word, "hand" (the child's) or "mama" (hers)
    child_sees: bool = False      # in the child's view as she reads it (before its head's camera, A40)


@dataclass(frozen=True)
class Percept:
    tick: int
    present: bool = True
    child_in_view: bool = True
    seen_by_child: bool = True
    child_target: str = None      # where she reads it looking (A40): an object id, "mama", or None
    child_holds: tuple = ()
    seen: tuple = ()              # Seen, one per object she sees
    fixtures: frozenset = frozenset()
    events: tuple = ()            # (kind, object id or None)
    child_sounding: bool = False
    extra: dict = field(default_factory=dict)   # instruments only; the fast layer never reads it
    child_reaches: tuple = ()     # the object ids its hands reach toward, as she sees them (A40)

    def obj(self, oid):
        for s in self.seen:
            if s.id == oid:
                return s
        return None

    def names(self):
        return {s.name for s in self.seen}

    def target_obj(self):
        """where she reads it looking, as an object she sees, or None ("mama" and unseen objects are not nameable things)."""
        return self.obj(self.child_target) if self.child_target not in (None, "mama") else None

    def attended(self):
        """what she reads the child attending (A40; 4.10's "the child's target, as she reads it"): the object its head's line
        is on, then those its hands hold, then those they reach toward, each once, those she sees."""
        out, ids = [], set()
        for oid in (self.child_target,) + tuple(self.child_holds) + tuple(self.child_reaches):
            o = self.obj(oid) if oid not in (None, "mama") else None
            if o is not None and o.id not in ids:
                out.append(o)
                ids.add(o.id)
        return out

    def attended_names(self):
        """the words of what she reads it attending, and "mama" when she reads it looking at her face."""
        return {o.name for o in self.attended()} | ({"mama"} if self.child_target == "mama" else set())


def check_events(p):
    bad = [k for k, _ in p.events if k not in EVENT_KINDS]
    assert not bad, f"unknown event kinds {bad}"
    return p


# ------------------------------------------------------------------------------------------------------ her reading (A40)
def _angles(v, axes):
    """a direction in the head camera's frame (axes: its forward, left and up unit vectors as rows) -> (azimuth, elevation) in
    degrees, or None behind its head."""
    x, y, z = (float(np.dot(a, v)) for a in axes)
    if x <= 0.0:
        return None
    return math.degrees(math.atan2(y, x)), math.degrees(math.atan2(z, math.hypot(x, y)))


class Reader:
    """her reading of where the child looks and what its hands do, as a person reads a robot that shows no eyes (A40).

    look(head_pos, head_axes, things) -> (target, before): head_pos, the head camera's place; head_axes, its forward, left and up
      unit vectors (a 3 x 3 array, rows), as a person sees the head's pose; things: [(id, position)] of what she can see, her face
      as "mama" among them. The line she reads is the camera's forward line turned by her error (yaw and pitch, each normal with
      sd READ_ERR_DEG, two draws every reading). target: the thing nearest that line (the least angle) within FOVEA_REACH_DEG of
      it, or None; before: the ids before the camera's field (CAMERA_FIELD_DEG) about that line, the Seen's child_sees.
    reaches(hands, things, holds=()) -> the ids its hands reach toward: hands, {hand: its position}; a thing is reached toward
      when the hand's distance to it fell on each of the last REACH_TICKS ticks, by at least REACH_CLOSE_M in all; per hand the
      one it closed on most; never one it holds.
    The world calls both once a tick, in that order, and fills the Percept from them; the conduct keeps the Reader and saves
    it."""

    def __init__(self, seed=1):
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(K.READ_STREAM,))))
        self.paths = {}                               # hand -> its last REACH_TICKS + 1 positions
        self.n = 0

    def look(self, head_pos, head_axes, things):
        ea, ee = (float(e) for e in self.rng.normal(0.0, K.READ_ERR_DEG, 2))
        self.n += 1
        axes = np.asarray(head_axes, np.float64)
        line = np.array([math.cos(math.radians(ee)) * math.cos(math.radians(ea)),
                         math.cos(math.radians(ee)) * math.sin(math.radians(ea)), math.sin(math.radians(ee))])
        best, before = None, []
        for tid, pos in things:
            v = np.asarray(pos, np.float64) - np.asarray(head_pos, np.float64)
            n = float(np.linalg.norm(v))
            ang = _angles(v, axes) if n > 0 else None
            if ang is None:
                continue
            da, de = abs(ang[0] - ea), abs(ang[1] - ee)
            if da <= K.CAMERA_FIELD_DEG[0] / 2 and de <= K.CAMERA_FIELD_DEG[1] / 2:
                before.append(tid)
            if da <= K.FOVEA_REACH_DEG[0] and de <= K.FOVEA_REACH_DEG[1]:
                u = np.array([float(np.dot(a, v)) for a in axes]) / n
                off = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, line))))))
                if best is None or off < best[0]:
                    best = (off, tid)
        return (None if best is None else best[1]), before

    def reaches(self, hands, things, holds=()):
        out = []
        for hand in sorted(hands):
            path = self.paths.setdefault(hand, [])
            path.append(np.asarray(hands[hand], np.float64).tolist())
            del path[:-(K.REACH_TICKS + 1)]
            if len(path) < K.REACH_TICKS + 1:
                continue
            best = None
            for tid, pos in things:
                if tid in holds or tid == "mama":
                    continue
                d = [float(np.linalg.norm(np.asarray(q) - np.asarray(pos, np.float64))) for q in path]
                if all(b < a for a, b in zip(d, d[1:])) and d[0] - d[-1] >= K.REACH_CLOSE_M:
                    if best is None or d[0] - d[-1] > best[0]:
                        best = (d[0] - d[-1], tid)
            if best is not None and best[1] not in out:
                out.append(best[1])
        return tuple(out)

    def state(self):
        return dict(rng=self.rng.bit_generator.state, paths={h: [list(q) for q in v] for h, v in self.paths.items()}, n=self.n)

    def load_state(self, s):
        self.rng.bit_generator.state = s["rng"]
        self.paths = {h: [list(q) for q in v] for h, v in s["paths"].items()}
        self.n = s["n"]
