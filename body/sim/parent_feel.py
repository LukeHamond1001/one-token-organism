"""The parent's feelings and its face (from the 2026-09-24 prototype; the world's side, nothing here is the child's body;
docs/SIM_DESIGN.md section 4.3).

The parent has a small, scripted, disclosed state of feeling, driven only by outward events the world logs:
  joy          pulses, one per judgment of a completed, visible act (the fast layer's table; never anything else)
  displeasure  the child hitting it (above the parent's own pain threshold), and in stage 2 talk-over
  surprise     sudden events (a loud sound onset, a toy flying, a roll completed, the peekaboo reveal), habituating
  concern      the child's pain or distress; it cancels any smile (no smile ever answers distress)
  attention    how engaged it is (interplay, its own tasks, away), plus where it looks (the behaviour layer's choice)
  mood         a slow leaky sum of the pleasures and displeasures (tens of minutes)
Each tick, display() maps the state to graded face parameters (kin.FACE_NEUTRAL: FACS action units) that
kin.face_geoms draws. The one law that keeps the reward honest: ONLY joy pulses and frowns move the mouth corners,
which are all the born expression reading reads (kin.face_reading). Surprise, interest, concern, the question face,
the greeting flash, speech, drowsiness and mood live on the brows, lids and jaw, which read 0. So looking at the
parent, being near it, or making it surprised or concerned can never be felt as reward.

Also here, for testing the parent only: BornReading, the child's born reading with its 30-tick hold and the sim
body's face reward (rises of the reading's positive part felt as positive, rises of its negative part as negative,
falls never felt), and selftest(), which checks the no-farming invariants under random gaze.
Run: nice -n 19 python3 body/sim/parent_feel.py  (the self-test)."""
import math

import numpy as np

import parent_kin as kin

# the parent's feeling constants (teacher method, ours; disclosed; set before birth and never tuned to a rate)
FEEL = dict(
    smile_onset=2,            # ticks for a smile to reach its apex (a spontaneous smile's onset, about 0.3 s)
    smile_wait=20,            # an unseen smile is held at most 20 ticks, then eases off (A3); A96: one she is bringing to the child's
                              # line of sight (her lean_in child_line under way) is held until she is there, then 20 more, at most
                              # queue_expiry ticks in all
    smile_after_seen=10,      # once seen, held 10 ticks more (A3)
    offset=5,                 # every smile and frown eases back to neutral over 5 ticks (A3)
    queue_expiry=40,          # a smile that must wait (another on the face, or one still held in the child's reading)
                              # waits at most 40 ticks from its judgment, then is dropped (logged)
    neutral_gap=2,            # a waiting smile starts after the mouth has been neutral 2 ticks
    reading_hold=30,          # the child's born reading's disclosed hold (A2), which the parent's display respects
    frown_hold=10,            # a frown is held 10 ticks and never waits for a look (A2)
    frown_hit=.5,             # displeasure at a hit: frown intensity .5 (reads -1)
    frown_talkover=0.0,       # stage 2's talk-over: was .25 (reads -0.5). C231 (2026-10-02): RETIRED FOR THE BABBLING CHILD. Life day 65 to its
                              # tick 17,300: 504 ticks of negative reward summing -257, 473 of them this frown at -0.5 (about 47 talk-overs,
                              # each held 10 ticks), against +39 of positive reward in all; the child's overlap with her speech sat at
                              # chance (473 against 514 expected from its 27% of ticks sounding and her 11% speaking), so 65 days of the
                              # frown taught no turn-taking and it stood as the largest term in the dopamine's scale (A172 divides every
                              # smile by it). A parent of a babbling infant does not scold the overlap: she stops and listens (4.6's cut
                              # and listen stay); turn-taking comes of her contingent pauses (Gratier et al. 2015; Goldstein and Schwade
                              # 2008). Zero: Feelings.harm at level 0 is nothing (no frown, no smile cancelled, no mood harm). Ours.
    surprise_tau=3.0,         # ticks; surprise is the briefest emotion (under a second)
    habituation=.5,           # a sudden event of a kind seen in the last 200 ticks surprises half as much, again half...
    habit_window=200,
    concern_tau=40.0,         # ticks (6 s), renewed while pain or distress lasts
    concern_pain=1.0, concern_distress=.6,
    concern_blocks=.5,        # no judged smile while concern is above .5 (about 4 s after a pain; all through distress)
    attention_tau=5.0,
    flash_ticks=3,            # the greeting eyebrow flash (Eibl-Eibesfeldt): 0.45 s at the first mutual gaze...
    flash_refractory=200,     # ...after at least 200 ticks without one
    mood_tau=4000.0,          # ticks (10 min)
    mood_joy=.08, mood_harm=.15, mood_pain=.04,
    blink_every=(20, 34),     # ticks between blinks (3-5 s), from the parent's own seeded stream
    wind_down_lids=.35,
)


class Pulse:
    """One judgment's smile: amplitude w/2 (so the reading's peak is the worth w), held until seen."""

    serial = 0

    def __init__(self, t, w, kind):
        Pulse.serial += 1
        self.n = Pulse.serial
        self.t0, self.a, self.kind = t, float(np.clip(w, 0, 2)) / 2, kind
        self.seen_at = None
        self.started = None       # the tick its display began (it may have waited in the queue)
        self.bringing = False     # A96: she is bringing it to the child's line of sight (her lean_in child_line under way)
        self.arrived = None       # the tick that bringing ended (there, or the act refused); None while under way or never brought

    def end_hold(self, C=FEEL):
        """the last tick it is held at its apex: 10 after it was seen; unseen, 20 after its start, or, brought to the child's line
        of sight, 20 after she is there (A96), never past queue_expiry ticks from its start"""
        if self.seen_at is not None:
            return self.seen_at + C["smile_after_seen"]
        cap = self.started + C["queue_expiry"]
        if self.bringing and self.arrived is None:
            return cap
        base = self.started if self.arrived is None else self.arrived
        return min(base + C["smile_wait"], cap)

    def value(self, t, C=FEEL):
        s = self.started
        rise = min(1.0, (t - s + 1) / C["smile_onset"])
        end_hold = self.end_hold(C)
        if t <= end_hold:
            return self.a * rise
        k = (t - end_hold) / C["offset"]
        return self.a * rise * max(0.0, 1 - k)

    def done(self, t, C=FEEL):
        return t >= self.end_hold(C) + C["offset"]


class Feelings:
    def __init__(self, seed=1, C=FEEL):
        self.C = C
        self.rng = np.random.default_rng([seed, 7031])       # the parent's own stream (blinks), logged for replay
        self.t = -1
        self.pulse, self.queued = None, None
        self.frown_t0, self.frown_amp = None, 0.0
        self.U, self.Cn, self.A, self.M = 0.0, 0.0, 0.0, 0.0
        self.A_target = 1.0
        self.q, self.loud, self.wind = 0.0, 0.0, 0.0
        self.sudden_log = []                                 # (t, kind)
        self.flash_t0, self.last_seen = None, -10 ** 9
        self.was_seen = False
        self.neutral_run = 10 ** 6                           # ticks the mouth has been neutral
        self.last_seen_L, self.last_seen_t = 0.0, -10 ** 9   # its own face's reading when the child last saw it
        self.next_blink = int(self.rng.integers(*C["blink_every"]))
        self.log = []                                        # (t, event, value)

    # ---- events (outward only: what the world logs about the child and the room)
    def judge(self, w, kind="act"):
        """The fast layer judged a completed, visible act worth w in (0, 2] (its fixed table). The only source of a smile."""
        if self.Cn > self.C["concern_blocks"]:               # no smile answers distress or pain
            self.log.append((self.t, "judge_suppressed_by_concern", w)); return
        p = Pulse(self.t, w, kind)
        self.M = min(1.0, self.M + self.C["mood_joy"] * p.a * 2)
        if self.queued is None or p.a > self.queued.a:     # a queue of one: the larger waits (see step)
            if self.queued is not None:
                self.log.append((self.t, "smile_dropped_for_larger", self.queued.a * 2))
            self.queued = p
        self.log.append((self.t, "judge", w))

    def harm(self, level=None):
        """The child's own act hit the parent above its pain threshold (stage 2), or talked over it (level given)."""
        amp = self.C["frown_hit"] if level is None else level
        if amp <= 0.0:                                           # C231: a frown of nothing is no frown (the talk-over's, retired)
            return
        self.frown_t0, self.frown_amp = self.t, max(amp, self.frown_amp if self.frown_active() else 0.0)
        self.pulse, self.queued = None, None                 # a frown ends any smile
        self.M = max(-1.0, self.M - self.C["mood_harm"] * amp)
        self.log.append((self.t, "frown", amp))

    def talk_over(self):
        self.harm(self.C["frown_talkover"])

    def child_pain(self):
        self.Cn = max(self.Cn, self.C["concern_pain"])
        self.pulse, self.queued = None, None
        self.M = max(-1.0, self.M - self.C["mood_pain"])
        self.log.append((self.t, "concern", 1.0))

    def distress(self):
        self.Cn = max(self.Cn, self.C["concern_distress"])
        self.pulse, self.queued = None, None

    def sudden(self, size, kind):
        n = sum(1 for (t0, k) in self.sudden_log if k == kind and self.t - t0 < self.C["habit_window"])
        self.U = max(self.U, float(np.clip(size, 0, 1)) * self.C["habituation"] ** n)
        self.sudden_log.append((self.t, kind))
        self.sudden_log = [(t0, k) for (t0, k) in self.sudden_log if self.t - t0 < self.C["habit_window"]]

    def set_engagement(self, a):   # 1 interplay, .4 its own tasks, 0 away or asleep (the behaviour layer's state)
        self.A_target = float(np.clip(a, 0, 1))

    def set_question(self, q):     # the expectant pause after a question, ask or call
        self.q = float(np.clip(q, 0, 1))

    def set_speech(self, loud):    # the voice clip's loudness this tick, 0..1
        self.loud = float(np.clip(loud, 0, 1))

    def set_wind_down(self, x):    # goodnight: 0..1
        self.wind = float(np.clip(x, 0, 1))

    def frown_active(self):
        return self.frown_t0 is not None and self.t < self.frown_t0 + self.C["frown_hold"] + self.C["offset"]

    # ---- one tick
    def step(self, seen, bringing=False):
        """Advance one tick. seen: the child's face test passed on this tick and the last (A1), as the world computes
        it. bringing (A96): her act that brings her face onto the child's line of sight is under way this tick (an unseen
        smile is held through it, Pulse.end_hold). Returns the graded face parameters for this tick."""
        C = self.C
        self.t += 1
        t = self.t
        if seen and not self.was_seen and t - self.last_seen > C["flash_refractory"]:
            self.flash_t0 = t
        if seen:
            self.last_seen = t
        self.was_seen = seen
        # the smile. A judged smile starts at once, unless (a) another smile is still on the face, or (b) the child's
        # born reading may still hold a smile it saw (it saw a positive face less than 31 ticks ago and has not seen
        # her neutral since: A2's hold): then it waits (at most queue_expiry ticks from the judgment), so every smile
        # starts from a neutral face the child's reading can rise from, and each is felt in full once. Only outward
        # facts are used: her own face and the face test's history, with the born reading's disclosed hold.
        if self.pulse is not None:
            if seen and self.pulse.seen_at is None:
                self.pulse.seen_at = t
            if self.pulse.seen_at is None:                                  # A96: brought to its line of sight, held until there
                if bringing:
                    self.pulse.bringing = True
                elif self.pulse.bringing and self.pulse.arrived is None:
                    self.pulse.arrived = t
            if self.pulse.done(t):
                self.pulse = None
                self.neutral_run = 0
        if self.pulse is None and self.queued is not None:
            clear = self.last_seen_L <= 0 or (t - self.last_seen_t) > C["reading_hold"]
            if t - self.queued.t0 > C["queue_expiry"]:
                self.log.append((t, "smile_expired", self.queued.a * 2))
                self.queued = None
            elif self.neutral_run >= C["neutral_gap"] and clear:
                self.pulse, self.queued = self.queued, None
                self.pulse.started = t
                if seen:
                    self.pulse.seen_at = t
        s = self.pulse.value(t) if self.pulse is not None else 0.0
        # the frown
        x = 0.0
        if self.frown_active():
            k = t - self.frown_t0 - C["frown_hold"]
            x = self.frown_amp * (1.0 if k < 0 else max(0.0, 1 - k / C["offset"]))
        # the slow and fast states
        self.U *= math.exp(-1 / C["surprise_tau"])
        self.Cn *= math.exp(-1 / C["concern_tau"])
        self.A += (self.A_target - self.A) / C["attention_tau"]
        self.M *= math.exp(-1 / C["mood_tau"])
        flash = 1.0 if (self.flash_t0 is not None and t - self.flash_t0 < C["flash_ticks"]) else 0.0
        blink = 0.0
        if t >= self.next_blink:
            blink = 1.0
            self.next_blink = t + int(self.rng.integers(*C["blink_every"]))
        self.state = dict(joy=s, displeasure=x, surprise=self.U, concern=self.Cn, attention=self.A, mood=self.M,
                          flash=flash, question=self.q, speech=self.loud)
        fp = self.display(s, x, flash, blink)
        L = kin.face_reading(fp)
        self.neutral_run = self.neutral_run + 1 if L == 0 else 0
        if seen:
            self.last_seen_L, self.last_seen_t = L, t
        return fp

    def display(self, s, x, flash, blink):
        """The feelings as FACS action units. The mouth corners carry only joy (smile) and displeasure (frown)."""
        U, Cn, A, M, q, loud, wd = self.U, self.Cn, self.A, self.M, self.q, self.loud, self.wind
        fp = kin.face_params(
            smile=s, cheek=s * (.5 + .5 * s), frown=x,
            brow_low=max(.9 * x, .45 * Cn), lid_tight=.8 * x,
            brow_in=min(1.0, max(.95 * U, .85 * Cn, .7 * flash, .5 * q) + .08 * A + .1 * max(M, 0)),
            brow_out=min(1.0, max(.95 * U, .7 * flash, .6 * q) + .08 * A + .06 * max(M, 0)),
            lid_up=float(np.clip(max(U, .45 * flash) + .15 * A - C_WIND * wd - .2 * max(-M, 0), -1, 1)),
            jaw=max(.5 * U * (1 - s), .45 * loud), round=.8 * U * (1 - s) * (1 - loud),
            press=.5 * Cn * (1 - loud), blink=blink, tilt=10 * q)
        assert fp["smile"] == 0 or self.pulse is not None
        return fp


C_WIND = FEEL["wind_down_lids"]


class BornReading:
    """The child's born expression reading and the sim body's face reward. For testing the parent only (the real one is
    the core's face channel and reward source). The reading updates only while the face test passes on 2 consecutive
    ticks; out of view it holds its last value 30 ticks, then reads neutral (A2). The reward: rises of the reading's
    positive part are felt as positive, rises of its negative part as negative, and no fall is ever felt; so a smile is
    felt once, at most its peak, however its onset is drawn."""

    def __init__(self, hold=30):
        self.hold, self.R, self.last, self.prev = hold, 0.0, -10 ** 9, False

    def seen(self, face_test_pass):
        s = face_test_pass and self.prev
        self.prev = face_test_pass
        return s

    def step(self, t, seen, L):
        if seen:
            new = L; self.last = t
        elif t - self.last > self.hold:
            new = 0.0
        else:
            new = self.R
        r = max(0.0, max(new, 0) - max(self.R, 0)) - max(0.0, max(-new, 0) - max(-self.R, 0))
        self.R = new
        return float(np.clip(r, -2, 2))


def run(T, gaze, events, seed=1):
    """Live T ticks. gaze[t]: the face test passes at t (bool). events: {t: [(name, args), ...]}. Returns traces."""
    F, B = Feelings(seed), BornReading()
    tr = dict(joy=[], displeasure=[], surprise=[], concern=[], attention=[], mood=[], flash=[], L=[], R=[], r=[], seen=[],
              fp=[], pulse_id=[])
    for t in range(T):
        seen = B.seen(bool(gaze[t]))
        # events are applied at the start of the tick, before the face is drawn
        F.t = t - 1
        for name, args in events.get(t, []):
            F.t = t
            getattr(F, name)(*args)
            F.t = t - 1
        fp = F.step(seen)
        L = kin.face_reading(fp)
        r = B.step(t, seen, L)
        for k in ("joy", "displeasure", "surprise", "concern", "attention", "mood", "flash"):
            tr[k].append(F.state[k])
        tr["L"].append(L); tr["R"].append(B.R); tr["r"].append(r); tr["seen"].append(seen); tr["fp"].append(fp)
        tr["pulse_id"].append(None if F.pulse is None else F.pulse.n)
    return tr


def selftest(seed=3):
    rng = np.random.default_rng(seed)
    T = 20000

    def markov_gaze(p_on, p_off):
        g, on = np.zeros(T, bool), False
        for t in range(T):
            on = (rng.random() < p_on) if not on else (rng.random() > p_off)
            g[t] = on
        return g
    out = {}
    # 1. no judgment, everything else random: nothing is ever felt
    ev = {}
    for t in range(T):
        e = []
        if rng.random() < .01: e.append(("sudden", (float(rng.random()), str(rng.integers(4)))))
        if rng.random() < .002: e.append(("child_pain", ()))
        if rng.random() < .01: e.append(("set_question", (float(rng.random() < .5),)))
        if rng.random() < .05: e.append(("set_speech", (float(rng.random()),)))
        if rng.random() < .002: e.append(("set_engagement", (float(rng.choice([0, .4, 1])),)))
        if e: ev[t] = e
    for name, (pon, poff) in dict(sticky=(.05, .05), glancy=(.3, .6), always=(1, 0)).items():
        tr = run(T, markov_gaze(pon, poff), ev)
        out[f"no_judgment_{name}_felt"] = round(float(np.abs(tr["r"]).sum()), 9)
        assert np.abs(tr["r"]).sum() == 0.0
    # 2. judgments at random, random gaze: never more felt than the worths judged; each pulse at most its worth
    ev2 = {}
    worths = []
    for t in range(T):
        if rng.random() < 1 / 150:
            w = float(rng.choice([.5, 1.0, 1.5, 2.0])); worths.append(w)
            ev2.setdefault(t, []).append(("judge", (w,)))
        if rng.random() < .01:
            ev2.setdefault(t, []).append(("sudden", (float(rng.random()), "x")))
    for name, (pon, poff) in dict(sticky=(.05, .05), glancy=(.3, .6), always=(1, 0)).items():
        tr = run(T, markov_gaze(pon, poff), ev2)
        felt = float(np.clip(tr["r"], 0, None).sum())
        per = {}
        for r, pid in zip(tr["r"], tr["pulse_id"]):
            if r > 0:
                per[pid] = per.get(pid, 0) + r
        assert felt <= sum(worths) + 1e-9
        assert max(per.values(), default=0) <= 2.0 + 1e-9
        out[f"judged_{name}"] = dict(felt=round(felt, 3), judged=round(sum(worths), 3), n_judged=len(worths),
                                     smiles_felt=len(per))
    # 3. farming by glancing: a 2-tick look every k ticks, no judgments: 0; with judgments: never above the worths
    for k in (2, 3, 5, 10, 30, 31, 45):
        g = np.array([(t % k) < 2 for t in range(T)])
        assert np.abs(run(T, g, ev)["r"]).sum() == 0.0
        f2 = float(np.clip(run(T, g, ev2)["r"], 0, None).sum())
        assert f2 <= sum(worths) + 1e-9
        out[f"glance_every_{k}"] = round(f2, 3)
    # 4. frowns: felt only when seen, at most -2 x their intensity each
    ev3 = {t: [("harm", ())] for t in range(50, T, 400)}
    tr = run(T, markov_gaze(.05, .05), ev3)
    neg = float(np.clip(tr["r"], None, 0).sum())
    assert neg >= -2 * FEEL["frown_hit"] * len(ev3) - 1e-9
    out["frowns"] = dict(n=len(ev3), felt=round(neg, 3))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=1))
