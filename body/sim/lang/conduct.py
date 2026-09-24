"""THE PARENT'S FAST LAYER AND HER INTENTS (docs/SIM_DESIGN.md 4.4-4.6, 4.10, A13, A14, A27; package P3): what she says, when,
in which register, and the acts her talk accompanies.

  INTENTS      her intents (4.5's table, 4.10's conduct): each with its register (4.4, P1's voice registers: plain, approval,
               comfort, calling, question, "no", the new word's), whether it opens an expectant pause (20 ticks: a question, an
               ask, a call), what it asks (a gaze, an act, a name), and the acts it accompanies. The acts are an interface: an
               Act names a kind and a target, and the world's MOTION (W2: the parent's acts on the G1 under the caps,
               body/sim/g1acts.py and parent_acts.py) carries it out. ACT_KINDS lists each kind with what W2 must make of it.
               StubMotion stands in until W2: it records each act and reports it done after a nominal time (STUB_TICKS, ours,
               never measured); it moves nothing.
  FastLayer    her voice's manners (4.5, 4.6): a line is composed from the intent's frames (body/sim/lang/templates.py) filled
               from what she perceives, held to the line check, picked by her own random stream (the body's seed, spawn key
               STREAM = 3; the world's is 1, the tract's 2), and said only when her voice is free and the rules allow it: the
               pauses (6 ticks between related lines, 20 after a question, an ask or a call), the call at most once per 240
               ticks, the same line not within 60, the same object named at most once per 20 (a variation set counts as one
               naming), at most one set per object per 120, idle at most one line per 40. Variation sets: 2-3 lines sharing the
               focus word, frames differing by at least a word, 6 ticks apart (Kuntay and Slobin; Onnis et al. 2008). A new
               word's set: 3 lines in the new-word register, the word last and emphasized (4.4, 4.8). Claude's steering lines
               (4.5, A14): each held to the line check (and its situation), used at most 3 times. Every refusal is logged.
  Conduct      the speech side of L2 (4.10), by its priority: the child's pain or distress (comfort), being hit ("oh!", stage 2
               "no."), a low charge (the meal's line), the child's vocal turn (a reply 3 ticks after it ends: a confirmation, a
               recast, an echo or an answer, by what her ear accepted), finishing her word (the talk-over stop: she stops and
               listens), a pending ask (judged over its window), joint attention (her gaze to the child's target, then a
               follow-in variation set naming it), and idle. Episodes (L3: the day plan, P4) and Claude's rows (P5) ask for
               intents through request(). The judgments she makes by talk (a right name, a met ask, an approximation, stage 1's
               vocal turn: 4.3's worth table) are returned for her feelings (body/sim/parent_feel.Feelings.judge); she never
               makes a feeling here. The motor judgments (a roll, a reach) and the face are the world's.

Built here from 4.10: the talk and its timing. Left to W2 and P3's day 4 / P4: L1's gaze and face each tick, the scaffolding
ladders' acts, the routines' order, and the motor judgments; the acts are requested here and carried out there.

Nothing here draws a random number but the fast layer's own stream; state() and load_state() carry everything, so a replay is
exact (a saved Conduct restored continues its lines, choices and ledger bit for bit: body/tests/test_sim_lang.py).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import consts as K
from . import templates as TP
from .lexicon import BIRTH_WORDS, NAME
from ..voice.playback import CUT_MAX

STREAM = 3                                    # the fast layer's random stream of the body's seed (ours; world 1, tract 2)
REFUSED_KEEP = 500                            # refusals kept for the digest and the instruments (the ledger keeps what she said)
NEVER = -10 ** 9

# --------------------------------------------------------------------------------------------------- acts (for W2)
ACT_KINDS = {
    "look": "her eyes (and head) to the target within 2 ticks (L1, 4.10); during='focus' only through the line's focus word, then "
            "back to the child's eyes (the joint-attention cue, 4.3)",
    "lean_in": "her face into the child's periphery, at least 15 degrees off its fovea's line, never closer than 25 cm (A3)",
    "attend": "kneel beside it and attend, one hand resting on its trunk (g1acts.attend)",
    "show": "the toy held beside her face about 40 cm before the G1's eyes, shaken for its sound (g1acts.show)",
    "point": "a point at the target, either hand (parent_acts.point)",
    "open_hand": "an open hand held out for the toy (the give ask's help level 0, 4.10)",
    "hand_over": "the toy put into the child's near or far hand (parent_acts.hand_over)",
    "touch": "a hand resting on the named part of the child, within one hand's cap (4.2)",
    "withdraw": "her hand drawn back from where she was hit (4.10)",
    "offer_bottle": "the bottle into the child's palm, her face in view (the meal's stage 1, 4.7)",
    "guide": "guide the named limb within the guide's cap (g1acts.guide, A10)",
    "pull_to_sit": "the pull-to-sit by the forearms, rising only with its own flexion (A9)",
    "wave": "a wave",
    "walk": "walk to the target (the door, the sofa, the child's side: A6's paths)",
    "cover_face": "her hands over her face (peekaboo)",
    "reveal_face": "her hands away from her face (the reveal)",
}
STUB_TICKS = {"look": 2, "lean_in": 7, "attend": 20, "show": 7, "point": 5, "open_hand": 5, "hand_over": 12, "touch": 7,
              "withdraw": 2, "offer_bottle": 12, "guide": 8, "pull_to_sit": 20, "wave": 5, "walk": 30, "cover_face": 3,
              "reveal_face": 2}                # the stub's nominal times, ours; W2 measures its own


@dataclass(frozen=True)
class Act:
    kind: str
    target: str = None                        # an object id, "child", "child_eyes", "child_periphery", a body word, "door", "sofa"
    during: str = None                        # "focus": only during the line's focus word

    def __post_init__(self):
        assert self.kind in ACT_KINDS, self.kind


class StubMotion:
    """the motion interface W2 implements: request(act, tick) -> id; status(id, tick) -> 'running' | 'done' | 'refused';
    cancel(id); state() / load_state(). The stub runs each act for its nominal time and moves nothing."""

    def __init__(self):
        self.acts = []                        # [(id, tick, kind, target, during, done_tick, cancelled)]

    def request(self, act, tick):
        i = len(self.acts)
        self.acts.append([i, int(tick), act.kind, act.target, act.during, int(tick) + STUB_TICKS[act.kind], False])
        return i

    def status(self, i, tick):
        a = self.acts[i]
        return "cancelled" if a[6] else ("done" if tick >= a[5] else "running")

    def cancel(self, i):
        self.acts[i][6] = True

    def state(self):
        return dict(acts=[list(a) for a in self.acts])

    def load_state(self, s):
        self.acts = [list(a) for a in s["acts"]]


# ------------------------------------------------------------------------------------------------------------ intents
@dataclass(frozen=True)
class Intent:
    register: str = "plain"
    expect: bool = False                      # opens a 20-tick expectant pause (a question, an ask, a call)
    ask: str = None                           # what it asks: "gaze" (a look at X), "act" (X given), "name" (a word), "call"
    acts: tuple = ()                          # Act templates: target "{o}" is the line's object, "{b}" its body part


LOOK_O = Act("look", "{o}", "focus")
EYES = Act("look", "child_eyes")
INTENTS = {
    "call": Intent("calling", True, "call", (Act("lean_in", "child_periphery"),)),
    "hall_call": Intent("calling", True, "call"),
    "greet": Intent("plain", False, None, (Act("lean_in", "child_periphery"), EYES)),
    "return": Intent("plain", False, None, (Act("walk", "child"), Act("lean_in", "child_periphery"))),
    "answer_bid": Intent("calling"),
    "label": Intent("plain", False, None, (LOOK_O, EYES)),
    "label_held": Intent("plain", False, None, (LOOK_O, EYES)),
    "label_colour": Intent("plain", False, None, (LOOK_O, EYES)),
    "show": Intent("plain", False, None, (Act("show", "{o}"), EYES)),
    "redirect": Intent("plain", False, None, (Act("point", "{o}"), LOOK_O)),
    "ask_where": Intent("plain", True, "gaze", (EYES,)),                 # never a point or a look to it: the ask tests the word
    "ask_what": Intent("plain", True, "name", (Act("show", "{o}"), EYES)),
    "ask_give": Intent("plain", True, "act", (Act("open_hand", "{o}"), EYES)),
    "confirm": Intent("approval", False, None, (EYES,)),
    "confirm_act": Intent("approval", False, None, (EYES,)),
    "recast": Intent("approval", False, None, (LOOK_O, EYES)),
    "recast_word": Intent("approval", False, None, (EYES,)),
    "echo": Intent("plain", False, None, (LOOK_O, EYES)),
    "echo_word": Intent("plain", False, None, (EYES,)),
    "reply": Intent("plain", False, None, (LOOK_O, EYES)),
    "reply_social": Intent("plain", False, None, (EYES,)),
    "narrate_fell": Intent("plain", False, None, (LOOK_O,)),
    "narrate_on": Intent("plain", False, None, (LOOK_O,)),
    "narrate_rolled": Intent("plain", False, None, (EYES,)),
    "narrate_sat": Intent("plain", False, None, (EYES,)),
    "body": Intent("plain", False, None, (Act("touch", "{b}"),)),
    "motor_sit": Intent("plain", False, None, (Act("pull_to_sit", "child"),)),
    "motor_roll": Intent("plain", False, None, (Act("guide", "far_arm"),)),
    "feed": Intent("comfort", False, None, (Act("offer_bottle", "bottle"), EYES)),
    "feed_more": Intent("comfort", True, "name", (EYES,)),
    "feed_done": Intent("comfort"),
    "leave": Intent("plain", False, None, (Act("wave", "child"), Act("walk", "door"))),
    "peekaboo_hide": Intent("plain", True, None, (Act("cover_face", "child"),)),
    "peekaboo": Intent("plain", False, None, (Act("reveal_face", "child"),)),
    "comfort": Intent("comfort", False, None, (Act("lean_in", "child_periphery"), Act("attend", "child"))),
    "hit": Intent("plain", False, None, (Act("withdraw", "child"),)),
    "no": Intent("no", False, None, (Act("withdraw", "child"),)),
    "night": Intent("comfort", False, None, (Act("walk", "sofa"),)),
    "new_word": Intent("new_word", False, None, (LOOK_O, EYES)),
}
INTRO_ACTS = {"toy": (Act("show", "{o}"), EYES), "fixture": (Act("point", "{w}"), LOOK_O), "body": (Act("touch", "{w}"),),
              "face": (EYES,), "colour": (Act("show", "{o}"), EYES), "adj": (LOOK_O, EYES), "verb": (EYES,),
              "past": (EYES,), "social": (EYES,)}
assert all(k in TP.FRAMES for k in INTENTS if k != "new_word"), [k for k in INTENTS if k not in TP.FRAMES]
assert set(TP.FRAMES) <= set(INTENTS), set(TP.FRAMES) - set(INTENTS)


def register_for(intent, text):
    """the intent's register; a plain line that asks ("?") in the question register (the synthesizer raises the pitch on "?")."""
    r = INTENTS[intent].register
    return "question" if r == "plain" and text.endswith("?") else r


def acts_for(intent, refs=(), b=None, w=None, word_class=None):
    tmpl = INTRO_ACTS.get(word_class, ()) if intent == "new_word" else INTENTS[intent].acts
    out = []
    for a in tmpl:
        tgt = a.target
        if tgt == "{o}":
            if not refs:
                continue
            tgt = refs[0]
        elif tgt == "{b}":
            tgt = b
        elif tgt == "{w}":
            tgt = w
        out.append(Act(a.kind, tgt, a.during))
    return tuple(out)


# --------------------------------------------------------------------------------------------------------- the fast layer
@dataclass
class Say:
    """what she does this tick: a line (with its clip, if a voice was given), the acts it accompanies, the judgments made,
    whether to stop her line (the talk-over), and why nothing was said (instruments)."""
    line: TP.Line = None
    clip: object = None
    acts: tuple = ()
    judgments: list = field(default_factory=list)    # [(worth, kind, word)] for parent_feel.Feelings.judge
    cut: bool = False
    listen: bool = False
    heard: list = field(default_factory=list)        # the child's words that ended this tick (transcriber.ChildWord)


class FastLayer:
    """her lines and their manners. vocab: the words she has (birth words, then the growth words as they enter); new_word: the
    day's new word (said only last, in the new-word register; it joins her words at the next night); held: the never-taught
    pairs still held out."""

    def __init__(self, seed=1, vocab=BIRTH_WORDS, stage=1):
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(STREAM,))))
        self.vocab = tuple(vocab)
        self.new_word = None
        self.held = tuple(tuple(h) for h in K.HELD_PAIRS)
        self.stage = stage
        self.last_said = {}                   # text -> tick
        self.last_named = {}                  # object id -> tick of its last naming (a set counts once)
        self.last_set = {}                    # object id -> tick of its last variation set
        self.last_call = NEVER
        self.last_line = NEVER                # tick her last line ended
        self.last_expect = NEVER              # tick her last question, ask or call ended
        self.last_focus = None                # the focus word of her last line (A27: expected as an echo)
        self.said_word = {}                   # word -> the tick she last finished saying it (the echo window, 4.8)
        self.busy_until = NEVER               # her current line sounds until this tick (exclusive)
        self.current = None                   # the line sounding
        self.queue = []                       # a set's later lines, each said 6 ticks after the last one ended
        self.recent = []                      # (tick, kind, object) she saw or heard in the last RECENT ticks
        self.steer = []                       # Claude's lines: [dict(text, situation, uses, tick_from, ttl)]
        self.refused = []                     # (tick, text, reason): the last REFUSED_KEEP lines the check or a rule refused
        self.n_lines = 0

    # ------------------------------------------------------------------ the moment
    def observe(self, p):
        self.recent = [e for e in self.recent if e[0] > p.tick - K.RECENT] + [(p.tick, k, o) for k, o in p.events]
        if len(self.refused) > REFUSED_KEEP:
            self.refused = self.refused[-REFUSED_KEEP:]

    def voice_free(self, t):
        return t >= self.busy_until

    def speaking(self, t):
        return t < self.busy_until

    def night(self):
        """a night boundary: the day's new word joins her words."""
        if self.new_word is not None and self.new_word not in self.vocab:
            self.vocab = self.vocab + (self.new_word,)
        self.new_word = None

    def open_pair(self, pair):
        """a never-taught pair's test opens (A28): it is no longer held out of her lines."""
        self.held = tuple(h for h in self.held if tuple(h) != tuple(pair))

    # ------------------------------------------------------------------ composing
    def compose(self, intent, t, p, o=None, b=None, w=None, frames=None, skip=()):
        """-> a Line for the intent (its frames filled from the percept and held to the line check; the same line not within
        SAME_LINE ticks), picked by her stream, or None. The stream is drawn only when there is a choice to make."""
        cands = []
        for fr in (frames if frames is not None else TP.FRAMES[intent]):
            got = TP.fill(fr, o=o, b=b, w=w, fixtures=p.fixtures)
            if got is None:
                continue
            text, focus, refs = got
            if text in skip or self.last_said.get(text, NEVER) > t - K.SAME_LINE:
                continue
            reg = "new_word" if (intent == "new_word" or (focus is not None and focus == self.new_word)) else \
                register_for(intent, text)
            ok, why = TP.check(text, self.vocab, self.new_word, p, refs, self.held, recent_events=self.recent)
            if not ok:
                self.refused.append((t, text, why))
                continue
            cands.append(TP.Line(text, intent, reg, focus, focus, refs, "fast", fr[0]))
        if not cands:
            return None
        return cands[int(self.rng.integers(len(cands)))] if len(cands) > 1 else cands[0]

    def variation_set(self, intent, t, p, o=None, b=None, w=None, n=None, frames=None):
        """a variation set: n lines (2-3, drawn when not given) sharing the focus word, each from a different frame -> [Line]."""
        n = n or int(self.rng.integers(K.SET_LINES[0], K.SET_LINES[1] + 1))
        out, skip = [], set()
        for _ in range(n):
            ln = self.compose(intent, t, p, o, b, w, frames=frames, skip=skip)
            if ln is None:
                break
            out.append(ln)
            skip.add(ln.text)
        return out

    # ------------------------------------------------------------------ the rules
    def allowed(self, line, t, reply=False, in_set=False):
        """the pauses and repeats (4.6) -> (ok, why)."""
        if not self.voice_free(t):
            return False, "her voice is busy"
        if in_set:
            return (t >= self.last_line + K.PAUSE_RELATED), "a set's 6-tick pause"
        if not reply and t < self.last_expect + K.PAUSE_EXPECT:
            return False, "the expectant pause"
        if line.intent in ("call", "hall_call") and t < self.last_call + K.CALL_EVERY:
            return False, "the call at most once per 240 ticks"
        if self.last_said.get(line.text, NEVER) > t - K.SAME_LINE:
            return False, "the same line within 60 ticks"
        if line.refs and line.focus is not None and line.intent in NAMING and \
                self.last_named.get(line.refs[0], NEVER) > t - K.SAME_OBJECT:
            return False, "the same object named within 20 ticks"
        return True, ""

    def commit(self, line, t, n_ticks, word_ends=(), in_set=False):
        """she starts the line at t, sounding n_ticks; word_ends: [(word, the tick its sound ends)]."""
        self.last_said[line.text] = t
        self.busy_until = t + max(1, int(n_ticks))
        self.current = line
        self.n_lines += 1
        if line.intent in ("call", "hall_call"):
            self.last_call = t
        if line.refs and line.focus is not None and not in_set and line.intent in NAMING:
            self.last_named[line.refs[0]] = t
        self.last_focus = line.focus
        for w, te in word_ends:
            self.said_word[w] = te

    def ended(self, t):
        """her line stopped sounding at t (whole, or cut by the talk-over)."""
        if self.current is not None:
            self.last_line = t
            if INTENTS[self.current.intent].expect:
                self.last_expect = t
            self.current = None
        self.busy_until = min(self.busy_until, t)

    # ------------------------------------------------------------------ Claude's lines (4.5, A14)
    def add_steer(self, text, situation, t, ttl=2000):
        """a steering row's line tied to a situation ("any", "holds:duck", "target:ball", "sees:drum"); checked now for form,
        vocabulary, the held-out pairs and praise, and again for what she perceives each time it is said. -> (ok, why)."""
        ok, why = TP.check(text, self.vocab, self.new_word, None, (), self.held, source="claude")
        if ok:
            self.steer.append(dict(text=text, situation=situation, uses=0, tick_from=int(t), ttl=int(ttl)))
        else:
            self.refused.append((t, text, "steer: " + why))
        return ok, why

    @staticmethod
    def situation(sit, p):
        kind, _, arg = sit.partition(":")
        if kind == "any":
            return True
        if kind == "holds":
            return any(p.obj(h) is not None and p.obj(h).name == arg for h in p.child_holds)
        if kind == "target":
            o = p.target_obj()
            return o is not None and o.name == arg
        if kind == "sees":
            return arg in p.names()
        return False

    def steer_line(self, t, p):
        for s in self.steer:
            if s["uses"] >= K.STEER_USES or not (s["tick_from"] <= t < s["tick_from"] + s["ttl"]):
                continue
            if not self.situation(s["situation"], p) or self.last_said.get(s["text"], NEVER) > t - K.SAME_LINE:
                continue
            ok, why = TP.check(s["text"], self.vocab, self.new_word, p, (), self.held, source="claude",
                               recent_events=self.recent)
            if not ok:
                self.refused.append((t, s["text"], "steer: " + why))
                continue
            s["uses"] += 1
            return TP.Line(s["text"], "steer", register_for("steer", s["text"]), None, None, (), "claude", s["situation"])
        return None

    # ------------------------------------------------------------------ save
    def state(self):
        return dict(rng=self.rng.bit_generator.state, vocab=list(self.vocab), new_word=self.new_word,
                    held=[list(h) for h in self.held], stage=self.stage, last_said=dict(self.last_said),
                    last_named=dict(self.last_named), last_set=dict(self.last_set), last_call=self.last_call,
                    last_line=self.last_line, last_expect=self.last_expect,
                    last_focus=self.last_focus, said_word=dict(self.said_word), busy_until=self.busy_until,
                    current=None if self.current is None else _line_d(self.current), queue=[_line_d(ln) for ln in self.queue],
                    recent=[list(e) for e in self.recent], steer=[dict(s) for s in self.steer],
                    refused=[list(r) for r in self.refused], n_lines=self.n_lines)

    def load_state(self, s):
        self.rng.bit_generator.state = s["rng"]
        self.vocab, self.new_word = tuple(s["vocab"]), s["new_word"]
        self.held, self.stage = tuple(tuple(h) for h in s["held"]), s["stage"]
        self.last_said, self.last_named, self.last_set = dict(s["last_said"]), dict(s["last_named"]), dict(s["last_set"])
        self.last_call, self.last_line, self.last_expect = s["last_call"], s["last_line"], s["last_expect"]
        self.last_focus, self.said_word = s["last_focus"], dict(s["said_word"])
        self.busy_until = s["busy_until"]
        self.current = None if s["current"] is None else _line_l(s["current"])
        self.queue = [_line_l(d) for d in s["queue"]]
        self.recent = [tuple(e) for e in s["recent"]]
        self.steer = [dict(x) for x in s["steer"]]
        self.refused = [tuple(r) for r in s["refused"]]
        self.n_lines = s["n_lines"]


NAMING = {"label", "label_held", "label_colour", "show", "redirect", "narrate_on", "new_word", "steer"}
INTENTS["steer"] = Intent("plain", False, None, (EYES,))        # Claude's line: plain (never the approval register, A14)


def _line_d(ln):
    return dict(text=ln.text, intent=ln.intent, register=ln.register, focus=ln.focus, emphasis=ln.emphasis, refs=list(ln.refs),
                source=ln.source, frame=ln.frame)


def _line_l(d):
    return TP.Line(d["text"], d["intent"], d["register"], d["focus"], d["emphasis"], tuple(d["refs"]), d["source"], d["frame"])


# ------------------------------------------------------------------------------------------------------------- conduct
class Conduct:
    """the speech side of L2. voice: anything with clip(text, register, emphasis=) -> a clip with .words [(word, first sample,
    end sample)], .pcm, .key and .digest (body/sim/voice/synth.VoiceCache; None: a line is timed at 3 ticks a word and has no
    clip). transcriber: body/sim/lang/transcriber.Transcriber (None: the child is not heard). ledger: body/sim/lang/ledger.Ledger.
    motion: W2's (default StubMotion)."""

    def __init__(self, seed=1, voice=None, transcriber=None, ledger=None, motion=None, vocab=BIRTH_WORDS, stage=1):
        from .ledger import Ledger                                   # noqa: PLC0415 (the ledger imports nothing of this)
        self.fast = FastLayer(seed, vocab, stage)
        self.voice = voice
        self.transcriber = transcriber
        self.ledger = ledger if ledger is not None else Ledger()
        self.motion = motion if motion is not None else StubMotion()
        self.routine = None                   # the routine under way (L3 sets it: "feed", "greet", "leave", "peekaboo", ...)
        self.pending = None                   # the ask she is judging: dict(kind, word, obj, tick, until)
        self.reply_due = None                 # the reply owed to the child's turn: dict(tick, kind, word, obj)
        self.target_run = [None, 0]           # the child's target and how many ticks running
        self.last_vocal_smile = NEVER
        self.turn = None                      # the child's turn under way: dict(start, in_pause, looked)
        self.sound_hist = []                  # the last NONSTOP[1] ticks: was the child sounding
        self.nonstop_since = None
        self.requests = []                    # intents asked for by L3 / Claude: [(intent, kwargs)]
        self.cuts = 0

    @property
    def stage(self):
        return self.fast.stage

    def request(self, intent, **kw):
        """an episode's or Claude's intent (L3, P4; P5), said when the priorities allow."""
        self.requests.append((intent, dict(kw)))

    # ------------------------------------------------------------------ the tick
    def tick(self, t, p, tract=None, distance_m=1.0, token=None, voice_done=None):
        """one tick. p: her Percept; tract: the child's tract samples this tick (engine units at 1 m) or None at rest;
        distance_m: from the child's mouth to her head; token: the silent token output's symbol this tick (lexicon ids) or None;
        voice_done: the tick her line stopped sounding, when the world's playback says so (a cut); otherwise the clip's own
        length ends it. -> Say."""
        f = self.fast
        f.observe(p)
        if voice_done is not None:
            f.ended(voice_done)
        elif f.current is not None and t >= f.busy_until:
            f.ended(f.busy_until)
        out = Say()
        self.target_run = [p.child_target, self.target_run[1] + 1 if p.child_target == self.target_run[0] else 1]
        # her ears (the transcriber): the child's turn, its words at the turn's end
        heard, sounding = [], p.child_sounding
        if self.transcriber is not None:
            vocab = f.vocab + ((f.new_word,) if f.new_word else ())       # her ear knows her words and the day's new one
            exp = self.transcriber.expected(p, self.pending, f.last_focus, self.routine, vocab)
            heard = self.transcriber.tick(t, tract, distance_m, token, f.speaking(t), exp, p, f.said_word, vocab)
            sounding = self.transcriber.sounding
            for cw in heard:
                self.ledger.accepted(cw, p)
        out.heard = heard
        if sounding and self.turn is None:
            self.turn = dict(start=t, in_pause=not f.speaking(t), looked=False)
            if f.speaking(t):                               # the talk-over: she finishes her word, stops, listens (4.6)
                out.cut, out.listen = True, True
                self.cuts += 1
                self.ledger.cut(t, f.current)
                f.busy_until = min(f.busy_until, t + CUT_MAX)   # the playback's cap; the world's voice_done may end it sooner
                f.queue = []
        if self.turn is not None:
            self.turn["looked"] |= p.child_target == "mama"
        self.sound_hist = (self.sound_hist + [bool(sounding)])[-K.NONSTOP[1]:]
        nonstop = len(self.sound_hist) == K.NONSTOP[1] and np.mean(self.sound_hist) > K.NONSTOP[0]
        self.nonstop_since = (self.nonstop_since if self.nonstop_since is not None else t) if nonstop else None
        # the asks she is judging (4.6: 20 ticks for a gaze, 40 for an act)
        for word, kind in self.ledger.observe(t, p):
            if self.pending is not None and self.pending["word"] == word:
                if kind != "call" or not self.ledger.understood(NAME):     # the call answered, until the name is understood
                    out.judgments.append((K.WORTH_MET_ASK, "met_ask", word))
                self.reply_due = dict(tick=t, kind="confirm", word=word, obj=self.pending.get("obj"))
                self.pending = None
        if self.pending is not None and t > self.pending["until"]:
            self.pending = None
        for cw in heard:                                    # the child's turn ended (or its tokens were read): judge, reply
            if cw.channel == "tract" or cw.word is not None:
                self._owe_reply(t, cw, p, out)
        if (self.transcriber is not None and self.transcriber.turn_end == t) or (self.transcriber is None and not sounding):
            self.turn = None                                # (with no transcriber, the world's own sounding flag ends the turn)
        got = self._choose(t, p, sounding)
        if got is not None:
            self._say(got[0], t, p, out, in_set=got[1])
        return out

    def _owe_reply(self, t, cw, p, out):
        """the child's turn ended with cw: the judgment now, the reply 3 ticks later (4.6, 4.3's worth table, A27)."""
        tgt = p.target_obj()
        near = {s.name: s for s in ([tgt] if tgt else []) + [p.obj(h) for h in p.child_holds if p.obj(h) is not None]}
        kind, obj, w = "reply", tgt, cw.word
        if w is not None:
            asked = self.pending is not None and self.pending["word"] == w
            obj = near.get(w) or (p.obj(self.pending["obj"]) if asked and self.pending.get("obj") else None) or \
                next((s for s in p.seen if s.name == w), None)
            if cw.exact and (w in near or asked):
                out.judgments.append((K.WORTH_RIGHT_NAME, "met_ask" if asked else "right_name", w))
                kind = "confirm"
                if asked:
                    self.ledger.named(t, w)
                    self.pending = None
            elif not cw.exact and self.stage >= 2 and self.ledger.exact_count(w, cw.channel) < K.EXACT_UNTIL:
                out.judgments.append((K.WORTH_APPROX, "approximation", w))
                kind = "recast"
            else:
                kind = "echo"
        if self.stage == 1 and cw.channel == "tract" and self.turn is not None and self.turn["in_pause"] and \
                self.turn["looked"] and t - self.last_vocal_smile >= K.VOCAL_TURN_EVERY:
            out.judgments.append((K.WORTH_VOCAL_TURN, "vocal_turn", None))
            self.last_vocal_smile = t
        self.reply_due = dict(tick=t + K.REPLY_AFTER, kind=kind, word=w, obj=None if obj is None else obj.id)

    def _choose(self, t, p, sounding):
        """-> (Line, in_set) by the priorities (4.10), or None."""
        f = self.fast
        if not f.voice_free(t):
            return None
        ev = {k for k, _ in p.events}
        # 1. the child's pain or distress: comfort (never a smile)
        if ev & {"pain", "distress"} and p.present:
            ln = f.compose("comfort", t, p)
            if ln is not None and f.allowed(ln, t, reply=True)[0]:
                f.queue = []
                return ln, False
        # 2. being hit by the child's own act: stage 1 "oh!", stage 2 "no." (a reflex she triggered is her defect: no line)
        if "hit_her" in ev:
            ln = f.compose("hit" if self.stage == 1 else "no", t, p)
            if ln is not None:
                return ln, False
        # the child's turn: she listens (both starting on one tick: the child has it), unless its babble never stops (A13);
        # and she holds her voice for the reply she owes it, 3 ticks after its turn
        if (sounding or self.turn is not None) and not (self.nonstop_since is not None and
                                                        t - self.nonstop_since >= K.NONSTOP[2]):
            return None
        if self.reply_due is not None and t < self.reply_due["tick"]:
            return None
        # 3. a low charge: the meal's first line (L3 runs the routine)
        if "charge_low" in ev and p.present and self.routine != "feed":
            ln = f.compose("feed", t, p)
            if ln is not None and f.allowed(ln, t)[0]:
                return ln, False
        # 4. the reply owed to the child's turn, 3 ticks after it ended
        if self.reply_due is not None and t >= self.reply_due["tick"]:
            r, self.reply_due = self.reply_due, None
            f.queue = []                                   # the turn answered; the set it broke into is not resumed
            ln = self._reply_line(r, t, p)
            if ln is not None and f.allowed(ln, t, reply=True)[0]:
                return ln, False
        # 5. a set's later lines, 6 ticks after the last one ended
        while f.queue:
            ln = f.queue[0]
            if not f.allowed(ln, t, in_set=True)[0]:
                return None
            f.queue.pop(0)
            ok, why = TP.check(ln.text, f.vocab, f.new_word, p, ln.refs, f.held, recent_events=f.recent)
            if ok:
                return ln, True
            f.refused.append((t, ln.text, "set: " + why))
        # 6. a pending ask: the expectant pause while she judges it
        if self.pending is not None:
            return None
        # 7. joint attention: a follow-in variation set naming its target (4.10; her gaze goes there in L1)
        o = p.target_obj()
        if o is not None and self.target_run[1] >= K.TARGET_TICKS and \
                f.last_named.get(o.id, NEVER) <= t - K.SAME_OBJECT and f.last_set.get(o.id, NEVER) <= t - K.SET_PER_OBJECT:
            lines = f.variation_set("label_held" if o.id in p.child_holds else "label", t, p, o=o)
            if lines and f.allowed(lines[0], t)[0]:
                f.last_set[o.id] = t
                f.queue = lines[1:]
                return lines[0], False
        # 8. the episode's act: an intent asked for by L3 or Claude, and Claude's lines for the moment
        while self.requests:
            intent, kw = self.requests.pop(0)
            got = self._compose_request(intent, t, p, kw)
            if got is not None:
                return got, False
        ln = f.steer_line(t, p)
        if ln is not None and f.allowed(ln, t)[0]:
            return ln, False
        # 9. idle: she watches (at most one line per 40 ticks; the episodes' idle lines are P4's)
        return None

    def _compose_request(self, intent, t, p, kw):
        f = self.fast
        if intent == "new_word":
            return self._introduce(kw["word"], t, p, kw.get("o"))
        o = p.obj(kw["o"]) if kw.get("o") else None
        if kw.get("o") and o is None:
            f.refused.append((t, intent, f"unseen: {kw['o']!r}"))
            return None
        if intent in ("label", "label_held", "label_colour", "show", "redirect") and o is not None:
            if f.last_set.get(o.id, NEVER) > t - K.SET_PER_OBJECT:
                return None
            lines = f.variation_set(intent, t, p, o=o, n=kw.get("n"))
            if lines and f.allowed(lines[0], t)[0]:
                f.last_set[o.id] = t
                f.queue = lines[1:]
                return lines[0]
            return None
        it = INTENTS[intent]
        if it.ask == "gaze" and o is not None and (not o.child_sees or p.child_target == o.id):
            f.refused.append((t, intent, "an ask needs its object in the child's view and not in its fovea (4.8)"))
            return None
        ln = f.compose(intent, t, p, o=o, b=kw.get("b"), w=kw.get("w"))
        if ln is None or not f.allowed(ln, t)[0]:
            return None
        if it.ask in ("gaze", "act", "name"):
            word = o.name if o is not None else ln.focus
            win = K.JUDGE_ACT if it.ask == "act" else K.JUDGE_GAZE
            self.pending = dict(kind=it.ask, word=word, obj=None if o is None else o.id, tick=t, until=t + win)
            self.ledger.ask(t, word, it.ask, None if o is None else o.id, win)
        return ln

    def _introduce(self, word, t, p, oid=None):
        """a new word's set (4.8): 3 lines, the word last and emphasized, in the new-word register, its referent shown."""
        f = self.fast
        prev, f.new_word = f.new_word, word
        o = p.obj(oid) if oid else next((s for s in p.seen if s.name == word), None)
        if o is None and TP.GROWTH_CLASS.get(word) == "colour":
            o = next((s for s in p.seen if s.colour == word), None)
        lines = f.variation_set("new_word", t, p, o=o, w=word, n=3, frames=TP.intro_frames(word))
        if len(lines) < 3 or not f.allowed(lines[0], t)[0]:
            f.new_word = prev
            f.refused.append((t, word, f"new word: {len(lines)} of its 3 lines pass"))
            return None
        f.queue = lines[1:]
        return lines[0]

    def _reply_line(self, r, t, p):
        f = self.fast
        o = p.obj(r["obj"]) if r.get("obj") else None
        w = r.get("word")
        noun = w in TP.NOUNS
        if r["kind"] == "confirm":
            ln = f.compose("confirm", t, p, o=o) if o is not None else None
            return ln or f.compose("confirm_act", t, p)
        if r["kind"] in ("recast", "echo") and w is not None:
            ln = f.compose(r["kind"] if noun else r["kind"] + "_word", t, p, o=o, w=w)
            if ln is not None:
                return ln
        tg = p.target_obj()
        return (f.compose("reply", t, p, o=tg) if tg is not None else None) or f.compose("reply_social", t, p)

    def _say(self, line, t, p, out, in_set=False):
        f = self.fast
        clip = None
        if self.voice is not None:
            clip = self.voice.clip(line.text, line.register, emphasis=line.emphasis)
            n = int(math.ceil(len(clip.pcm) / 2400))
            word_ends = [(w, t + (max(e, 1) - 1) // 2400) for w, _, e in clip.words]
        else:
            ws = TP.words(line.text)
            n = 3 * len(ws)                                  # no voice: 3 ticks a word (an instrument's timing, ours)
            word_ends = [(w, t + 3 * i + 2) for i, w in enumerate(ws)]
        f.commit(line, t, n, word_ends, in_set=in_set)
        if line.intent in ("call", "hall_call") and self.pending is None:
            self.pending = dict(kind="call", word=NAME, obj=None, tick=t, until=t + n + K.PAUSE_EXPECT)
            self.ledger.ask(t, NAME, "call", None, n + K.PAUSE_EXPECT)
        cls = TP.GROWTH_CLASS.get(line.focus) if line.intent == "new_word" else None
        acts = acts_for(line.intent, line.refs, b=line.focus, w=line.focus, word_class=cls)
        for a in acts:
            self.motion.request(a, t)
        out.line, out.clip, out.acts = line, clip, acts
        self.ledger.said(t, line, p, clip=clip, n_ticks=n, word_ends=word_ends)

    # ------------------------------------------------------------------ save
    def state(self):
        return dict(fast=self.fast.state(), routine=self.routine, pending=self.pending, reply_due=self.reply_due,
                    target_run=list(self.target_run), last_vocal_smile=self.last_vocal_smile, turn=self.turn,
                    sound_hist=list(self.sound_hist), nonstop_since=self.nonstop_since,
                    requests=[[i, dict(k)] for i, k in self.requests], cuts=self.cuts, motion=self.motion.state(),
                    ledger=self.ledger.state(), transcriber=None if self.transcriber is None else self.transcriber.state())

    def load_state(self, s):
        self.fast.load_state(s["fast"])
        self.routine, self.pending, self.reply_due = s["routine"], s["pending"], s["reply_due"]
        self.target_run, self.last_vocal_smile, self.turn = list(s["target_run"]), s["last_vocal_smile"], s["turn"]
        self.sound_hist, self.nonstop_since = list(s["sound_hist"]), s["nonstop_since"]
        self.requests, self.cuts = [(i, dict(k)) for i, k in s["requests"]], s["cuts"]
        self.motion.load_state(s["motion"])
        self.ledger.load_state(s["ledger"])
        if self.transcriber is not None and s["transcriber"] is not None:
            self.transcriber.load_state(s["transcriber"])
