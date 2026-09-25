"""the body in frames (a mixin of `Life`, body/life.py; the core refactor's step R7, docs/SIM_DESIGN.md 7.2, 7.4, 7.6, 8's R7 row, 10): a
body that lives in the world's frames (the sim) rather than on the page alone.

STEP R7a, THE EVENT LINES (7.2, 7.4's low road; A37, A43): the anatomy's born event lines (`Anatomy.events`, body/core/anatomy.py
`EventLine`), read once a tick from the frame the tick is lived on (`_event_lines`): a line fires (1) when any of its numbers is above 0,
on its side where it declares one (both sides where the direction has none), and not where a line further along its limb's chain fires
(isolation: the furthest group that feels a contact names it); else 0. One declaration, read by the striatal expansion (the critics:
`_events_push` puts the tick's fired lines into the striatum's event line, whose born rows body/model.py appends after the effectors'
blocks) and by the amygdala (R7d). The diary declares none, so nothing here runs for it.

STEP R7b, THE FRAMES (7.4's "the fast memory's writes", 9's store, 10's "event end"; physiology.py FRAMES, the switch `frames`, which the
diary's cfg does not hold), at each tick's end (`_frame_tick`, after the tick's lessons and before its bookkeeping):
- THE FRAME'S SURPRISE (`_frame_surprise`): each forecasting channel's error on the frame that came: the words' the tick's surprise (the
  forecast's cosine distance to the symbol heard, the rest included: the language body's own, `_surp_tick`), every later channel's its
  head's squared error to the code that came (0.5 |head(C_t-1) - code_t|^2: the waking lesson's own loss, on the forecast made at the
  last tick's end); each divided by its own running mean (1 / min(n, err_tau): the critics' statistics' horizon), and their mean: every
  channel weighs alike, and a frame of ordinary surprise reads 1 (THE BUILDER'S READING, for the lead: 7.4 and 10 say "the summed
  forecast error", the first design "all channels weighted equally"; the mean is that sum weighted equally, and it keeps a frame's
  write strength in the words' units, a typical error reading 1, so the frames and the words share the store's floors and its
  eviction on one scale).
- THE EVENT'S END FOR FRAMES (`_frame_settle`; 10's "event end"): today's settle law on it, a fast and a slow running average (offset_fast
  4, offset_slow 64 ticks), settled while the fast is at most offset_settle 0.5 of the slow; an event ends on the first settled tick
  after one that was not (its surprise has risen and fallen back). There the store's last frame written is marked as an event's end
  (its boundary, `Store.mark_boundary`), the next frame written as an event's start (`mark_start`), and the tick is kept in the day's
  list of ends (R8 cuts the day's tape into episodes there). The night ends every event: an event still open at nightfall ends there.
- THE WRITE (`_frame_gate`, `_frame_write`): a frame is written when its surprise x (1 + tag) passes its own running write_q (0.9)
  quantile (the tag the amygdala's, R7d; 0 until then), so about a tenth of the ticks are written whatever the scale; the key is the
  stream's state at the tick before (the context before the frame, pattern-separated as the store's cortex key is: `query_from`), the
  value the frame's codes (every channel's code, summed in the anatomy's order: the frame as the cortex's input receives it), the
  strength surprise x (1 + |dopamine|) x (1 + tag), the store's own law for a heard word with the tag; the slot's `who` is 2 (a frame:
  the words' dreams never start at one; the night over frames is R8's).
- THE TICK'S RECORD (`_record_tick`): (the frame's surprise, dopamine, the tag, the net reward received), float32, 16 B a tick, in the
  day's record (`_rec`, its first `_rec_n` rows), R8's tape beside the codes; the day's ends beside it (`_rec_ends`). The night lets
  both go (R8 cuts and keeps them first).
Every state here is a working attribute of the life, saved with a motor body's day (A70)."""
import math

import torch

from .physiology import FRAMES


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

    # ---------------- step R7b: the frames ----------------
    def _frame_const(self, k):
        """a frame constant: the body's cfg when it was given, else FRAMES'"""
        return self.cfg.get(k, FRAMES[k])

    def _frames_on(self):
        """the switch `frames` (step R7b): the body lives in frames (0 for the diary, whose cfg holds no such key)"""
        return bool(int(self.cfg.get("frames", FRAMES["frames"])))

    def _frame_codes(self, u):
        """THE FRAME AS THE CORTEX RECEIVES IT: each channel's code of this tick's frame ({name: [d]}, detached; the words' the row of the
        symbol heard, the rest's in the quiet) and their sum in the anatomy's order (the cortex's input before its LayerNorm, without the
        ladder's bundle and the efference copies)"""
        m = self.m; codes = {}; tot = None
        with torch.no_grad():
            for c_ in self.anatomy.channels:
                o_ = c_.observe(self, u, 0)
                x_ = torch.tensor(int(o_), device=self.dev) if c_.kind == "symbol" else o_
                code = c_.encode(m, x_).detach()
                codes[c_.name] = code
                tot = code if tot is None else tot + code
        return codes, tot

    def _frame_surprise(self, codes):
        """THE FRAME'S SURPRISE (the module's doc): the mean over the forecasting channels of each one's error on this frame over its own
        running mean (each mean taking this tick's error first); None when no channel foresaw it (after birth or a night)"""
        st = getattr(self, "_ferr", None)
        if st is None:
            st = {}; self._ferr = st
        fc = getattr(self, "_ffc", None) or {}
        fw = getattr(self, "_fw_err", None)
        tau = float(self._frame_const("err_tau")); vals = []
        for i_, c_ in enumerate(self.anatomy.channels):
            if i_ == 0:
                e = fw
            elif c_.name in fc:
                with torch.no_grad():
                    e = float(0.5 * ((fc[c_.name].float() - codes[c_.name].float()) ** 2).sum())
            else:
                e = None
            if e is None:
                continue
            n_, mu = st.get(c_.name, [0, 0.0])
            n_ += 1; mu = mu + (float(e) - mu) / min(float(n_), tau)
            st[c_.name] = [n_, mu]
            if mu > 0.0:
                vals.append(float(e) / mu)
        return (sum(vals) / len(vals)) if vals else None

    def _frame_settle(self, s):
        """THE EVENT'S END FOR FRAMES (the module's doc): today's settle law on the frame's surprise `s`; True on the first settled tick
        after one that was not"""
        af = 1.0 / max(1.0, float(self.cfg.get("offset_fast", 4))); as_ = 1.0 / max(1.0, float(self.cfg.get("offset_slow", 64)))
        fast = getattr(self, "_fs_fast", None); slow = getattr(self, "_fs_slow", None)
        fast = float(s) if fast is None else (1.0 - af) * fast + af * float(s)
        slow = float(s) if slow is None else (1.0 - as_) * slow + as_ * float(s)
        self._fs_fast, self._fs_slow = fast, slow
        settled = fast <= float(self.cfg.get("offset_settle", 0.5)) * max(1e-6, slow)
        ended = settled and not getattr(self, "_fs_settled", True)
        self._fs_settled = settled
        return ended

    def _frame_end(self):
        """an event ends (the module's doc): the last frame written marked as its end, the next to be written as the next one's start,
        the tick kept in the day's ends"""
        lw = getattr(self, "_flast_write", None)
        if lw is not None:
            self.store.mark_boundary(*lw)
        self._flast_write = None; self._fstart_armed = True
        ends = getattr(self, "_rec_ends", None)
        if ends is None:
            ends = []; self._rec_ends = ends
        ends.append(int(getattr(self, "_rec_n", 0)))

    def _frame_gate(self, g):
        """THE WRITE GATE (the module's doc): whether `g` (the frame's surprise x (1 + tag)) passes its own running write_q quantile, in
        log units, stepped by write_eta after the test; its first 1/write_eta samples settle it as their sample quantile, and nothing
        passes while it settles"""
        if not g > 0.0:
            return False
        lx = math.log(float(g)); eta = float(self._frame_const("write_eta")); p = float(self._frame_const("write_q"))
        q = getattr(self, "_fq", None)
        if q is None:
            w_ = getattr(self, "_fq_warm", None)
            if w_ is None:
                w_ = []; self._fq_warm = w_
            w_.append(lx)
            if len(w_) >= max(1, int(round(1.0 / max(eta, 1e-9)))):
                self._fq = float(self._pace_q(w_, p)); self._fq_warm = None
            return False
        passed = lx > float(q)
        self._fq = float(q) + eta * (p - (1.0 if lx <= float(q) else 0.0))
        return passed

    def _frame_write(self, key, value, strength):
        """the frame written (the module's doc): its codes under the key of the stream before it, at `strength`, who 2; its start mark
        when an event ended since the last write; kept as the event's last frame for its end mark. True when the store kept it"""
        st = self.store
        if not st.write(key, value, float(strength), 2):
            return False
        self._flast_write = (key.clone(), value.clone())
        if getattr(self, "_fstart_armed", False):
            st.mark_start(key, value); self._fstart_armed = False
        self._fwrites = int(getattr(self, "_fwrites", 0)) + 1
        return True

    def _record_tick(self, s, delta, tag, r):
        """THE TICK'S RECORD (the module's doc): (surprise, dopamine, tag, the net reward received) as float32, a row of the day's record"""
        rec = getattr(self, "_rec", None); n_ = int(getattr(self, "_rec_n", 0))
        if rec is None or n_ >= int(rec.shape[0]):
            new = torch.zeros(max(4096, 2 * (int(rec.shape[0]) if rec is not None else 0)), 4, dtype=torch.float32)
            if rec is not None:
                new[:n_] = rec[:n_]
            rec = new; self._rec = rec
        rec[n_] = torch.tensor([float(s), float(delta), float(tag), float(r)], dtype=torch.float32)
        self._rec_n = n_ + 1

    def _frame_key(self, C):
        """the key a frame is written under: the stream's state `C` pattern-separated (the store's cortex key, `query_from`); under
        key_form "cortex" the very key the tick's own step made (its running mean moved there), else made here, the running mean moving
        once a tick"""
        if str(self.cfg.get("key_form", "bag")) == "cortex" and getattr(self, "_q_prev", None) is not None:
            return self._q_prev
        return self.query_from(C, learn=True)

    def _frame_foresee(self):
        """at the tick's end, from the stream after its own step: each later forecasting channel's forecast of the next frame's code, and
        the key the next frame will be written under (none after a night, before the first step)"""
        C = getattr(self, "_C_last", None)
        if C is None:
            self._ffc = None; self._fkey_prev = None
            return
        with torch.no_grad():
            self._ffc = {c_.name: self.m.head(self.anatomy, i_)(C).detach() for i_, c_ in enumerate(self.anatomy.channels) if i_ and c_.forecast}
        self._fkey_prev = self._frame_key(C)

    def _tag_now(self):
        """the tag of this tick and the tag at a write (this tick's received part plus the forecast made the tick before): the amygdala's
        (R7d); 0 while it is off"""
        return 0.0, 0.0

    def _frame_tick(self, u, delta, r):
        """THE FRAME'S TICK (the module's doc), at the tick's end: the frame's codes and surprise, the event's end, the gated write, the
        tick's record, the forecast and the key for the next frame"""
        codes, total = self._frame_codes(u)
        s = self._frame_surprise(codes)
        tag, tag_w = self._tag_now()
        if s is not None:
            if self._frame_settle(s):
                self._frame_end()
            key = getattr(self, "_fkey_prev", None)
            if key is not None and self._frame_gate(float(s) * (1.0 + float(tag_w))):
                self._frame_write(key, total, float(s) * (1.0 + abs(float(delta))) * (1.0 + float(tag_w)))
        self._record_tick(s if s is not None else 0.0, delta, tag, r)
        self._frame_foresee()

    def _frames_nightfall(self):
        """THE NIGHT ENDS EVERY EVENT (the module's doc; called as the night begins): an event still open ends at nightfall"""
        if not getattr(self, "_fs_settled", True):
            self._frame_end()

    def _words_store(self):
        """THE STORE'S WORDS ALONE (step R7b), for the words' dreams of a body in frames: a copy of the store without its frames (who 2),
        its links, marks and episodes remapped as the store's own compaction remaps them (`Store._keep`); the store itself untouched. The
        frames are the night over frames' (R8's), never a words' dream: read as words they ran the pattern completion on through them
        (THE BUILDER'S READING, for the lead: until R8, a body in frames dreams its words as a body without frames would)"""
        from ..model import Store
        st = self.store
        ws = Store(st.d, cap=st.cap, temp=st.temp, device=st.dev, read_strength=st.read_strength, links=st.NK)
        ws.load_state_dict(st.state_dict()); ws.saturate = st.saturate
        ws._keep(torch.nonzero(st.W != 2).flatten())
        ws.last_remap = None
        return ws

    def _frames_night(self):
        """the frames' working state wakes fresh after the night (called with the rest of it): no forecast, no key and no last frame
        carried over, the next frame written the morning's start; the day's record and its ends let go (R8 cuts them into episodes
        first); the running means, the settle law's averages and the write gate's quantile kept"""
        self._ffc = None; self._fkey_prev = None; self._fw_err = None; self._flast_write = None; self._fstart_armed = True
        self._rec = None; self._rec_n = 0; self._rec_ends = []
