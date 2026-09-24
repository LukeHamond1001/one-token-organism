"""WHAT THE PARENT PERCEIVES (docs/SIM_DESIGN.md 4.5: "filled from what the parent can see: what the child looks at, holds, or just
said"; 4.10's joint attention; A14's "what cannot be shown"; package P3). The fast layer reads the moment only through a Percept,
and the line check holds every line to it: she names only what a person in her place could see or hear.

The world (W2-W3) fills one each tick from her senses, never from anything a person could not know:
  present         she is with the child (not away in the hall)
  child_in_view   she can see the child (its body, for "your foot")
  seen_by_child   she is where the child's eyes can reach her (a person knows when a baby cannot see her: behind its head)
  child_target    what the child is looking at, read from its software fovea's window as a person reads a baby's eyes (B14): the
                  object id its central ray hits 3 ticks running, "mama" for her own face, or None
  child_holds     the object ids in its hands, as she sees them
  seen            the objects she can see now (the world's ray test from her eyes), each with its word, its colour and where it
                  rests as she sees it; child_sees marks those in the child's view (the ledger's "heard": a person can tell
                  whether a toy is before a baby's eyes)
  fixtures        the room's words she can see ("mat", "sofa", "window", and the growth queue's "table", "shelf", ...)
  events          what she saw or heard happen this tick: (kind, object id or None), kind one of EVENT_KINDS
  child_sounding  she hears the child's voice this tick (the transcriber's own reading of it)
None of these is the child's inside: no gate, value, forecast or feeling reaches her (A14's rule for Claude holds for her too).
"""
from dataclasses import dataclass, field

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
)


@dataclass(frozen=True)
class Seen:
    id: str                       # the world's object id ("ball", "ball_blue")
    name: str                     # its word ("ball")
    colour: str = ""              # as she sees it ("red")
    on: str = ""                  # what it rests on as she sees it: a fixture word, "hand" (the child's) or "mama" (hers)
    child_sees: bool = False      # in the child's view


@dataclass(frozen=True)
class Percept:
    tick: int
    present: bool = True
    child_in_view: bool = True
    seen_by_child: bool = True
    child_target: str = None
    child_holds: tuple = ()
    seen: tuple = ()              # Seen, one per object she sees
    fixtures: frozenset = frozenset()
    events: tuple = ()            # (kind, object id or None)
    child_sounding: bool = False
    extra: dict = field(default_factory=dict)   # instruments only; the fast layer never reads it

    def obj(self, oid):
        for s in self.seen:
            if s.id == oid:
                return s
        return None

    def names(self):
        return {s.name for s in self.seen}

    def target_obj(self):
        """the child's target as an object she sees, or None ("mama" and unseen objects are not nameable things)."""
        return self.obj(self.child_target) if self.child_target not in (None, "mama") else None


def check_events(p):
    bad = [k for k, _ in p.events if k not in EVENT_KINDS]
    assert not bad, f"unknown event kinds {bad}"
    return p
