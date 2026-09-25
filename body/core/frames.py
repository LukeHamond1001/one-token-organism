"""the body in frames (a mixin of `Life`, body/life.py; the core refactor's step R7, docs/SIM_DESIGN.md 7.2, 7.4, 7.6, 8's R7 row, 10): a
body that lives in the world's frames (the sim) rather than on the page alone.

STEP R7a, THE EVENT LINES (7.2, 7.4's low road; A37, A43): the anatomy's born event lines (`Anatomy.events`, body/core/anatomy.py
`EventLine`), read once a tick from the frame the tick is lived on (`_event_lines`): a line fires (1) when any of its numbers is above 0,
on its side where it declares one (both sides where the direction has none), and not where a line further along its limb's chain fires
(isolation: the furthest group that feels a contact names it); else 0. One declaration, read by the striatal expansion (the critics:
`_events_push` puts the tick's fired lines into the striatum's event line, whose born rows body/model.py appends after the effectors'
blocks) and by the amygdala (R7d). The diary declares none, so nothing here runs for it."""


def read_event_lines(events, obs):
    """THE BORN RULE (step R7a; the module's doc) on one frame's observations `obs` ({name: numbers}): per declared line, in order, 1.0 where
    it fires and 0.0 where not: any of its numbers above 0; on its side where it declares one (the direction's sense times the line's
    above 0), both lines of a pair where the direction is exactly 0 (no side); and none of the lines further along its limb's chain
    firing (isolation: those are judged by the first two rules alone)"""
    raw = {}
    for ln in events:
        o_ = obs.get(ln.obs)
        f_ = o_ is not None and any(float(o_[int(i)]) > 0.0 for i in ln.fired)
        if f_ and ln.side is not None:
            v_ = float(o_[int(ln.side[0])])
            f_ = v_ == 0.0 or float(ln.side[1]) * v_ > 0.0              # its side, or no side at all (both lines of the pair)
        raw[ln.name] = f_
    return tuple(1.0 if raw[ln.name] and not any(raw[d_] for d_ in ln.distal) else 0.0 for ln in events)


class FramesMixin:
    # ---------------- step R7a: the event lines ----------------
    def _event_lines(self, frame=None):
        """THE TICK'S EVENT LINES (the module's doc; `read_event_lines`): one number per declared line, 1.0 where it fires and 0.0 where
        not, in the declared order; read once a tick from the frame the tick is lived on (`frame`, the tick's own readers pass it) and
        kept for the tick; with no frame (an instrument's read, between ticks) the lines of the tick last read, never read afresh (the
        tick's count has moved on then while the world's frame has not); () for a body that declares none"""
        ev = self.anatomy.events
        if not ev:
            return ()
        now_ = getattr(self, "_events_now", None)
        if frame is None:
            return now_[1] if now_ is not None else tuple(0.0 for _ in ev)
        if now_ is not None and now_[0] == self.ticks:
            return now_[1]
        out = read_event_lines(ev, frame.obs)
        self._events_now = (self.ticks, out)
        return out

    def _events_push(self, u):
        """the tick's fired event lines enter the striatum's event line (step R7a; called in `_hear` with the heard events, before the
        striatal input is read): one event, their bits, when any fired; under stri_quiet an empty one when none did (the line carries
        time, as the language line's quiet does)"""
        ev = self._event_lines(self._tick_frame(u))
        mask = 0
        for i_, v_ in enumerate(ev):
            if v_:
                mask |= 1 << i_
        if mask:
            self.m.striatum_push_events(mask)
        elif int(self.cfg.get("stri_quiet", 0)):
            self.m.striatum_push_events(0)
