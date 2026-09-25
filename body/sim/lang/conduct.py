"""THE PARENT'S FAST LAYER AND HER INTENTS (docs/SIM_DESIGN.md 4.4-4.6, 4.10, A13, A14, A27; package P3): what she says, when,
in which register, and the acts her talk accompanies.

  INTENTS      her intents (4.5's table, 4.10's conduct): each with its register (4.4, P1's voice registers: plain, approval,
               comfort, calling, question, "no", the new word's), whether it opens an expectant pause (20 ticks: a question, an
               ask, a call), what it asks (a gaze, an act, a name), and the acts it accompanies. The acts are an interface: an
               Act names a kind and a target, and the world's MOTION (W2: the parent's acts on the G1 under the caps,
               body/sim/g1acts.py and parent_acts.py) carries it out. ACT_KINDS lists each kind with what W2 must make of it.
               StubMotion stands in until W2: it records each act and reports it done after a nominal time (STUB_TICKS, ours,
               never measured); it moves nothing. Her motion declares the acts her own body can show a verb by (DOES: the stub's
               are those W2's parent_acts and ACT_KINDS already name); a verb whose act it cannot do waits (A15).
  FastLayer    her voice's manners (4.5, 4.6): a line is composed from the intent's frames (body/sim/lang/templates.py) filled
               from what she perceives, held to the line check, picked by her own random stream (the body's seed, spawn key
               STREAM = 3; the world's is 1, the tract's 2), and said only when her voice is free and the rules allow it: the
               pauses (6 ticks between related lines, 20 after a question, an ask or a call), the call at most once per 240
               ticks, the same line (the same words, its punctuation aside: templates.key) not within 60, the same object named
               at most once per 20 (a variation set counts as one naming), at most one set per object per 120, idle at most one
               line per 40. Variation sets: 2-3 lines sharing the focus word, frames differing by at least a word (two lines of
               the same words, "a rattle." / "a rattle!", never in one set), 6 ticks apart (Kuntay and Slobin; Onnis et al.
               2008). A new
               word's set: 3 lines in the new-word register, the word last and emphasized, its referent shown (4.4, 4.8, A15);
               only a word the world can show (templates.showable, against the world's inventory) that she can show now
               (templates.show_now), in lines measured to put it on the line's pitch peak (A34, the line check); up
               to NEW_PER_DAY = 3 new words a day and NEW_EVERY = 1 a minute (A14), each joining her words at the
               night. Claude's steering lines (4.5, A14):
               each held to the line check (and its situation), used at most 3 times, a line ending on the day's new word said in
               the new-word register and emphasized. Every refusal, and every request dropped, is logged.
               A word counts as said when its sound has ended: the talk-over's cut withdraws the words it stopped (4.6), from her
               echo window, her last focus word and the ledger alike.
  Conduct      the speech side of L2 (4.10), by its priority: the child's pain or distress (comfort), being hit ("oh!", stage 2
               "no."), a low charge (the meal's line), the child's vocal turn (a reply after her latency: a confirmation, a
               recast, an echo or an answer, by what her ear accepted), finishing her word (the talk-over stop: she stops and
               listens), a pending ask (judged over its window), joint attention (a follow-in variation set naming its target
               as she reads it), and idle. Episodes (L3: the day plan, P4) and Claude's rows (P5) ask for intents through
               request(). The judgments she makes by talk (a right name, a met ask, an approximation, stage 1's vocal turn: 4.3's
               worth table) are returned for her feelings (body/sim/parent_feel.Feelings.judge), and stage 2's frowns
               (Say.frown: a turn that talked over her, being hit) for Feelings.talk_over / harm; she never makes a feeling
               here. The motor judgments (a roll, a reach) and the face are the world's.
               WHAT SHE READS OF THE CHILD (A40): its head's line (the G1's trunk) and its hands, as a person sees a robot that
               shows no eyes, never its fovea's window: the Percept's child_target (held 3 ticks running, 4.10), child_holds
               and child_reaches, filled by the world through her Reader (percept.py, her error drawn from her own stream).
               Every rule below reads them.
               A right name is an exact word whose referent is where she reads the child looking, in its hand or reached
               toward (her face for "mama" when she reads it looking at her), or the answer to her name ask ("what is this?")
               begun once the question was heard (its word made before it is no answer, and voids the ask, 4.8); an
               approximation earns its smile of 1 only there too (stage 2). The ear's wider expected set (her last focus word,
               the routine's words) lets her hear and echo a word, not smile at it. An ECHO (a word said within 10 ticks of
               her saying it) is answered as imitation, as a parent answers it (Goldstein and Schwade 2008; her method): judged
               as the word said then would be, so it may earn her smile, but it counts toward nothing in the ledger, neither
               "says", "understood" nor a met ask (a name ask answered by an echo is void). "more?" at the meal is the feed's
               line, never an ask. An ask ("where is the X?", "give me the X", the call) is judged from the moment its word has
               been heard: X in the child's view and none she reads it attending then (else void), its trunk turning to an X
               or a hand reaching toward one within the window and holding 2 ticks (4.8); a give done before its word was heard
               is void, not missed; the call is a gaze ask at her face, 20 ticks from the end of its name, void when the child
               cannot see her then; a call from the hall is not judged (no look can answer it there).
               HER ATTENTION LOG (A51; Golinkoff et al. 1987; the lead's decision in P3's sixth round: one log, fail-closed, in
               place of a rule for each route by which her body could cue an answer). Every tick, before the conduct runs, her
               motion (W2; StubMotion until W2 merges) reports where her body physically points (report()): her eyes', her
               head's and each hand's target, from the tick it leaves the child (or rest) until it is back (a thing's id; the
               child itself, only for her eyes or face on its face or a hand held open or waved toward it; the part of the child
               a hand touches or her eyes rest on; a place, a raised or shaken hand's among them; her own face; or none; a hand
               holding a thing is at it), her face ("mama" on a tick its expression moves), and the status of every act asked of
               it by anyone, exactly "running", "done", "refused" or "cancelled" (any other status, or an act of hers not
               reported, is running: fail-closed); L1's own gaze (to a sudden event, to the child's act or target) is in her
               eyes' and head's fields as it happens. The conduct keeps the log (attn, the last CUE_CLEAR + 1 ticks, saved), each
               field as every thing a look that follows it could be read as attending (_words; her Reader reads a look as the
               nameable thing nearest its line, with her error, A40): a thing she sees is its word and those of every thing near
               it (Seen.near, from the world's Reader.near: within NEAR_DEG = 20 degrees of it as seen from the child's head, or
               resting on or in it, or it on or in them; P3's eighth round: a glance at the box the ball rests in held back no
               ask about the ball); the child itself (the mutual gaze every ask is made in: a look that follows it comes back to
               her face) and her face ("mama") are those and the things beside her face (Percept.face_near) and in her hands;
               and anything else, a place, a part of the child, a thing she does not see now, an id or a word she cannot name,
               or a thing (or her face) whose near the world does not give, is UNNAMED, which stands for every thing. Her face
               moving (the report, and FACE_COURSE ticks from each judgment or frown of hers, A3's longest smile) and her voice
               (a word of her line sounding) are in the log at her face. Her face, the call's X, is also where every movement of
               her body is, or beside it: for the call a tick with her eyes or head off the child, a hand not at rest, her face
               moving, her voice, or an act of hers under way but her eyes on its eyes is a cue at her face (_hits). An act she
               asks for is in the log from that tick (with what it directs her body at, as asked and as each tick's percept
               names it), before her motion's first report of it; an act in the report she did not ask for (P4's, W2's own) is
               in it, at every thing, until it is reported ended. An ask about X (a gaze ask, a give, the call about her face, a
               name ask) opens only if every tick of her life among its tick and the CUE_CLEAR = 44 ticks (6.6 s) before it is
               in the log (a tick never read is a cue at every thing; a save that does not say when she was born, every tick
               before its log) and none could cue X (a look within that time may follow her cue, not the word: Brooks and
               Meltzoff 2005 scored infants' following looks over 6.5 s from an adult's head turn), no act is running, queued
               or of unknown status but those she may make while an ask is pending and asked for herself (PENDING_OK: her eyes
               on its eyes, her open hand to it, her face into its periphery, a hand withdrawn when hit), and the log shows her
               eyes and head on the child and her hands at rest or at it. While it is pending each tick's percept judges it
               first (an answer a tick shows was made before it: a look held 3 readings and 2 ticks, a give at the end of its
               hand's path; the given toy in her hand never voids the give it answers, P3's eighth round), then the tick's log
               is held to it (_hold): her body not still, or X in the log, voids it (fail-closed: W2 broke its contract, or her
               face moved with a judgment), as do ticks the world did not tick her. Her lines' acts are held to PENDING_OK's
               (blind()), and for the call to her eyes on its eyes alone; while the call is pending she says no line (her
               voice is at her face; the reply she owes waits), the child's pain or its hit voiding it first; the call is said
               from where her face already rests, with no lean-in, and ends on its name. A copy of its movement due then is
               not made (a late copy is no copy), no new word is shown, the meal's first line waits, and L1 keeps her eyes on
               the child (eyes_on_child). Her look on a line's naming word (during='focus', 4.3) is requested as that word begins
               and cancelled as it ends, so the log shows it there. So a name ask shows nothing: "what is this?" is asked of a
               thing the child attends, as she reads it. Never a gaze or act ask about an X in her own hands (a look at her
               would meet it); and "give me the X" only with an X within the child's reach as she sees it
               (Seen.child_can_reach), so no ask is one no act of its could answer. cue_clear() is the same rule, read from the
               log, for P4's base-rate trials.
               HER IMPERFECTION (A52; consts, from human dyads, her own stream): her reply's latency jittered (Gratier et al.
               2015's switching pauses, 2.91 ticks after the turn's end on average), a turn she makes no judgment of missed 30%
               of the time (Gros-Louis et al. 2006: mothers answered over 70% of vocalizations within 2 s), and her mirrored
               copies of its arm and hand movements within 1-2 s, at most about 6 a minute (Pawlby 1977). No miss touches a
               judgment: a judged turn is always answered.
               STAGE 2's TALK-OVER (4.4, 4.6, A13): a turn begun during her line stops her at the word's end, frowns
               (Say.frown), and is answered "no." (the "no." register), not judged; never for babble that never stops.
               REDIRECTS (4.10): a redirect (the "redirect" intent, or a Claude line naming a toy the child does not attend as
               she reads it) is said only after 40 ticks with no target and while the day's follow-in namings number at least
               2 for each redirect (Tomasello and Farrar 1986).

Built here from 4.10: the talk and its timing, her reading of the child (A40), her imperfection and copying (A52), her attention
log (A51). Left to W2 and P4: L1's gaze and face each tick (W2 reads eyes_on_child and reports L1's gaze and her face in the
attention log, StubMotion's contract), the world's Seen.near and Percept.face_near (percept.Reader.near), the scaffolding ladders'
acts (4.10: the judged give is level 0 alone; its point and touch only once its window has closed unmet), the routines' order, her
spells of distraction (P4's own-tasks episode), the base-rate trials' moments (drawn where cue_clear() holds, as her asks are),
and the motor judgments; the acts are requested here and carried out there.

Nothing here draws a random number but her own streams of the body's seed (her lines' STREAM = 3, her reading's READ_STREAM = 4,
her imperfection's IMPERFECT_STREAM = 5); state() and load_state() carry everything, so a replay is exact (a saved Conduct
restored continues its lines, choices and ledger bit for bit: body/tests/test_sim_lang.py).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import consts as K
from . import templates as TP
from .lexicon import BIRTH_WORDS, NAME, PARENT_NAME
from .percept import Reader
from ..voice.playback import CUT_MAX, TICK
from ..voice.synth import SR

STREAM = 3                                    # the fast layer's random stream of the body's seed (ours; world 1, tract 2)
REFUSED_KEEP = 500                            # refusals kept for the digest and the instruments (the ledger keeps what she said)
NEVER = -10 ** 9

# --------------------------------------------------------------------------------------------------- acts (for W2)
ACT_KINDS = {
    "look": "her eyes (and head) to the target within 2 ticks (L1, 4.10); during='focus' only through the line's focus word, then "
            "back to the child's eyes (the joint-attention cue, 4.3): the conduct requests it as that word begins and cancels "
            "it as it ends",
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
    "do": "her own body does the named act while she says its word (a verb's introduction: its act from templates.NEEDS, "
          "e.g. 'clap', 'push', 'stand'; W2 refuses what it cannot do, and templates.showable() keeps such a word waiting); an "
          "act that handles a toy (templates.TOY_ACTS: 'shake' and 'hold' by the show, 'get' by the pick-up) names its toy as "
          "the Act's thing, the one the new word's set is said of, and W2 handles that toy and no other",
    "copy": "her own arm or hand makes the movement the child's just made, mirrored as she faces it (target 'kind:side', the "
            "side hers: its left arm raised, her right raised; A52), asking nothing and earning nothing",
}
STUB_FOCUS = 60                               # the stub runs a during='focus' act until the conduct cancels it (its word's end)
STUB_TICKS = {"look": 2, "lean_in": 7, "attend": 20, "show": 7, "point": 5, "open_hand": 5, "hand_over": 12, "touch": 7,
              "withdraw": 2, "offer_bottle": 12, "guide": 8, "pull_to_sit": 20, "wave": 5, "walk": 30, "cover_face": 3,
              "reveal_face": 2, "do": 7, "copy": 7}       # the stub's nominal times, ours; W2 measures its own


@dataclass(frozen=True)
class Act:
    kind: str
    target: str = None                        # an object id, "child", "child_eyes", "child_periphery", a body word, "door", "sofa"
    during: str = None                        # "focus": only during the line's focus word
    thing: str = None                         # a 'do' act's toy (the demonstration's shake or pick-up of it: TP.TOY_ACTS)

    def __post_init__(self):
        assert self.kind in ACT_KINDS, self.kind


# ---------------------------------------------------------------------------------------- her attention log (A51, for W2)
FIELDS = ("eyes", "head", "left", "right", "face")   # her motion's report every tick: where her eyes, head and each hand
                                              # physically point, and her face ("mama" on a tick its expression moves)
HANDS = ("left", "right")
LOG_FIELDS = FIELDS + ("voice",)              # her log's: those, and her voice (a word of her line sounding: the conduct's own)
ENDED = ("done", "refused", "cancelled")      # the only statuses that end an act: any other, or none reported, is running
UNNAMED = "?"                                 # a target that could be read as any thing: it stands for every thing (fail-closed)
AT_CHILD = ("child", "child_eyes", "child_periphery")   # the child itself: her eyes or face on its face, her hand held open or
                                                         # waved toward it, touching nothing (a part she touches is a place)
OUTSIDE = "?act"                              # the kind of an act in her motion's report she did not ask for (P4's, W2's own)
PENDING_OK = (("look", "child_eyes"), ("open_hand", "child"), ("lean_in", "child_periphery"), ("withdraw", "child"))
# the acts she may make while an ask about a thing is pending, and the only ones that may still be running (or queued, or
# unreported) when one opens, each one she asked for herself: her eyes on its eyes, her open hand held out to it, her face into
# its periphery (a comfort's lean-in), a hand withdrawn when hit (L1's). Every other act, and every act she did not ask for,
# could direct its attention, to its X or anywhere (A51; the lead's decision, P3's sixth round; the seventh's for an act she did
# not ask for). While the call is pending (its X her face, where every movement of her body is) only her eyes on its eyes
# (blind(acts, "mama"); P3's eighth round: the call has no lean-in of its own).
ATTN_KEEP = K.CUE_CLEAR + 1                   # ticks of the log kept: an ask's tick and the CUE_CLEAR ticks before it


def _still_act(a):
    """an open act of the log (its entry [id, kind, target, status, at]) she may make while an ask is pending: PENDING_OK's, and
    one she asked for herself (an act she did not ask for is never one: OUTSIDE)."""
    return a[1] != OUTSIDE and (a[1], a[2]) in PENDING_OK


def _act_str(a):
    """an open act of the log ([id, kind, target, status, at]) in words."""
    if a[1] == OUTSIDE:
        return f"an act she did not ask for (her motion's id {a[0]!r}, reported {a[3]!r})"
    return f"her {a[1]} at {', '.join(w for w in a[4] if w) or a[2] or 'nothing'} ({a[3]})"


def _fmt(ws):
    """a field of her log (the words a look that follows it could be read as) in words."""
    return "/".join(ws) if ws else "rest"


def _unstill(e):
    """None when a tick of her attention log shows her body as an ask needs it (her eyes and head on the child, her hands at
    rest or held open to it, no act running but PENDING_OK's she asked for herself), else what it shows."""
    bad = [f"her {f} at {_fmt(e[f])}" for f in ("eyes", "head") if "child" not in e[f]]
    bad += [f"her {f} hand at {_fmt(e[f])}" for f in HANDS if e[f] and "child" not in e[f]]
    bad += [_act_str(a) for a in e["acts"] if not _still_act(a)]
    return "; ".join(bad) if bad else None


def at_rest(target):
    """a field's target that cues no thing: none, or the child itself (a part of it she touches or looks at is a place)."""
    return target is None or target in AT_CHILD


def directs(act):
    """the attention log's fields an act of hers drives while it runs, and at what: what StubMotion reports of it, and the least
    W2 reports (W2 reports its own truth, L1's gaze with it) -> [(field, target)]. field: "eyes", "head", "hand" (one hand: the
    stub's right, else its left), "both" (both hands), or "left" / "right" (a copy's side); target: a thing's id, the child itself
    ("child", "child_eyes", "child_periphery": her eyes or face on its face, her hand held open or waved toward it), the part of
    the child she touches or guides (a body word, "trunk", "far_arm": a place, 4.3), a place word, "mama" (her own face), None
    (at rest), or UNNAMED where the act points at a place or does not say (a copy's raised arm or shaken hand, a demonstration
    that handles no named toy: fail-closed)."""
    k, tg = act.kind, act.target
    if k in ("look", "walk"):
        return [("eyes", tg), ("head", tg)]
    if k == "lean_in":
        return [("head", "child")]
    if k == "attend":
        return [("head", "child"), ("hand", "trunk")]                # one hand resting on its trunk
    if k in ("show", "point", "open_hand", "hand_over", "offer_bottle"):
        return [("hand", tg)]
    if k in ("touch", "guide"):
        return [("hand", tg)]                                         # the part she touches, the limb she guides
    if k == "wave":
        return [("hand", "child")]
    if k == "pull_to_sit":
        return [("both", "arm")]                                      # by the forearms (A9)
    if k == "withdraw":
        return [("hand", None)]
    if k in ("cover_face", "reveal_face"):
        return [("both", PARENT_NAME)]
    if k == "copy":                                                   # her arm raised or her hand shaken points at a place;
        kind, _, side = (tg or "").partition(":")                     # her hand waved or held open, toward the child
        return [(side if side in HANDS else "hand", "child" if kind in ("wave", "open_hand") else UNNAMED)]
    if k == "do":
        if tg in TP.TOY_ACTS:
            return [("hand", act.thing if act.thing is not None else UNNAMED)]
        if tg in ("wave", "open_hand"):
            return [("hand", "child")]
        if tg == "walk":
            return [("eyes", act.thing or UNNAMED), ("head", act.thing or UNNAMED)]
        return [("hand", UNNAMED)]
    return [("hand", UNNAMED)]


class StubMotion:
    """the motion interface W2 implements, and its contract:
      request(act, tick) -> id: the act asked for (W2 carries it out under the caps, or refuses it);
      status(id, tick) -> 'running' | 'done' | 'refused' | 'cancelled';
      cancel(id, tick): the act stopped from that tick (the conduct cancels her look on a naming word as the word ends);
      report(tick) -> dict(eyes, head, left, right, face, acts): HER ATTENTION LOG'S FIELDS (A51), made once a tick BEFORE the
        conduct runs (the conduct reads it first thing in its tick, and a tick whose report it never read counts as a cue at
        every thing): eyes, head, left and right, where each physically points (a thing's id; the child itself: "child",
        "child_eyes" or "child_periphery", only for her eyes or face on its face or her hand held open or waved toward it,
        touching nothing; the part of the child a hand touches or guides, or her eyes rest on, by its word ("hand", "foot",
        "arm", "trunk"); a place word, and a hand raised or shaken at nothing points at a place; "mama" for her own face; or None
        at rest), from the tick it leaves the child (or rest) until the tick it is back, whatever target the act asked for (a
        look asked for and cancelled within a tick still shows on the ticks her eyes moved; P3's seventh round), L1's own gaze
        included as it happens (to a sudden event, to the child's act or its target), and a hand holding a thing at it (the
        given toy in her hand on the tick the child gives it: the conduct judges an ask by that tick's percept before it holds
        the tick's log to it, P3's eighth round); face, "mama" on a tick her face's expression moves (a smile's or a frown's
        onset, hold or easing: the face is W2's), None while it is still (her voice is the conduct's own: it logs her words as
        they sound); and acts, {id: status} for every act asked of it by anyone (the conduct, P4, its own) that it has not yet
        reported ended, each status exactly 'running', 'done', 'refused' or 'cancelled'. Anything else ('queued', 'waiting', a
        typo), a field left out (eyes, head or a hand: at every thing; the face: moving), or an act of hers left out of the
        report counts as running, at every thing (fail-closed); an act in the report the conduct did not ask for blocks every
        ask until it is reported ended, and voids a pending one: so W2 reports every act from its request until it reports it
        ended, keeps 'running' while it waits its turn or directs the child's eyes at all (the show held beside her face, the
        offer held out, the look held), ends every act, and reads Conduct.eyes_on_child each tick (while an ask is pending her
        eyes and head stay on the child and her hands at rest or held open to it: L1 turns to nothing, W2 and P4 start nothing
        else);
      state() / load_state(); DOES, the acts her own body can show a verb by (templates.NEEDS' ("act", kind): the 'do' act's
        targets), which templates.showable() reads.
    The stub runs each act for its nominal time and moves nothing (it never refuses); its fields are what directs() gives for the
    acts running (an act asked for at a tick shows from the next report: the report of that tick was made before it), her eyes
    and head on the child at rest, her face still (the stub has no face: the conduct logs her face from its own judgments). An
    act cancelled from a tick still shows on that tick (her eyes or hand on the way back) and is 'cancelled' from the next, so
    every act shows on at least one tick. glance() stands in for W2's L1 (the world calls it for
    a sudden event: her eyes and head on it from that tick; a gaze for a tick already reported starts at the next report, keeping
    its length, so the log never loses it). Its DOES are the acts W2's parent_acts.py or ACT_KINDS already name (shake and hold:
    the show; go and walk: the walk; the wave; get: pick_up; open: the open hand). W2 declares its own (a clap, a push, ...)."""

    DOES = ("show", "walk", "wave", "pick_up", "open_hand")

    def __init__(self):
        self.acts = []                        # [id, tick, kind, target, during, end tick, stopped at (or None), thing]
        self.live = []                        # the ids not yet reported ended
        self.glances = []                     # L1's gazes (the world's, until W2): [target, from tick, until tick]
        self.reported = None                  # the last tick reported (saved: a cancel or a gaze never rewrites a report made)
        self._rep = None                      # (tick, report): a tick's report, given again if asked again

    def request(self, act, tick):
        i = len(self.acts)
        n = STUB_FOCUS if act.during == "focus" else STUB_TICKS[act.kind]
        self.acts.append([i, int(tick), act.kind, act.target, act.during, int(tick) + n, None, act.thing])
        self.live.append(i)
        return i

    def status(self, i, tick):
        a = self.acts[i]
        if tick < a[5]:
            return "running"
        return "cancelled" if a[6] is not None else "done"

    def _next(self):
        """the first tick not yet reported (None before the first report)."""
        return None if self.reported is None else self.reported + 1

    def cancel(self, i, tick=None):
        """the act stopped from tick (a look on a naming word at the word's end; None: from the next report): shown on that tick
        (on its way back) and 'cancelled' from the next, never from a tick already reported (that report showed it running)."""
        a = self.acts[i]
        s = max(x for x in (None if tick is None else int(tick), self._next(), a[1] + 1) if x is not None)
        if s < a[5]:
            a[6], a[5] = s, s + 1

    def glance(self, target, tick, ticks=2):
        """L1's reflexive gaze (W2's; the world calls it until W2): her eyes and head on target from tick for `ticks` ticks (its
        latency and hold are W2's; 2, the stub's); a gaze for a tick already reported starts at the next report, keeping its
        length (the conduct has read that tick: the log never loses it)."""
        n = max(1, int(ticks))
        start = int(tick) if self._next() is None else max(int(tick), self._next())
        self.glances.append([target, start, start + n])

    def report(self, tick):
        if self._rep is not None and self._rep[0] == tick:
            return dict(self._rep[1], acts=dict(self._rep[1]["acts"]))
        f = dict(eyes="child", head="child", left=None, right=None, face=None)
        claimed = set()
        for i in self.live:
            a = self.acts[i]
            if a[1] >= tick or self.status(i, tick) != "running":
                continue
            for fld, tg in directs(Act(a[2], a[3], a[4], a[7])):
                if fld == "hand":
                    fld = "right" if "right" not in claimed else ("left" if "left" not in claimed else "right")
                for x in (("left", "right") if fld == "both" else (fld,)):
                    if x not in claimed or at_rest(f[x]) or not at_rest(tg):   # two acts on one field: the one that could
                        f[x] = tg                                              # cue wins (it cannot time them: fail-closed)
                    claimed.add(x)
        for g in self.glances:
            if g[1] <= tick < g[2]:
                f["eyes"] = f["head"] = g[0]
        acts = {i: self.status(i, tick) for i in self.live}
        self.live = [i for i in self.live if acts[i] not in ENDED]
        self.glances = [g for g in self.glances if g[2] > tick]
        self._rep = (tick, dict(f, acts=acts))
        self.reported = tick if self.reported is None else max(self.reported, int(tick))
        return dict(f, acts=dict(acts))

    def state(self):
        return dict(acts=[list(a) for a in self.acts], live=list(self.live), glances=[list(g) for g in self.glances],
                    reported=self.reported)

    def load_state(self, s):
        self.acts = [list(a) + [None] * (8 - len(a)) for a in s["acts"]]
        for a in self.acts:                   # a save before P3's seventh round: a[6] a flag, a[5] the tick it was stopped
            if a[6] is True:
                a[6] = a[5] - 1
            elif a[6] is False:
                a[6] = None
        self.live = list(s.get("live", ()))
        self.glances = [list(g) for g in s.get("glances", ())]
        self.reported = s.get("reported")
        self._rep = None


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
    "call": Intent("calling", True, "call", (EYES,)),        # said from where her face already rests in its periphery,
                                                             # still (A3): no lean-in, whose movement a look could follow
                                                             # (A51; P3's eighth round)
    "hall_call": Intent("calling", True, None),                      # not judged: from the hall no look can answer it (4.7)
    "greet": Intent("plain", False, None, (Act("lean_in", "child_periphery"), EYES)),
    "return": Intent("plain", False, None, (Act("walk", "child"), Act("lean_in", "child_periphery"))),
    "answer_bid": Intent("calling"),
    "label": Intent("plain", False, None, (LOOK_O, EYES)),
    "label_held": Intent("plain", False, None, (LOOK_O, EYES)),
    "label_colour": Intent("plain", False, None, (LOOK_O, EYES)),
    "show": Intent("plain", False, None, (Act("show", "{o}"), EYES)),
    "redirect": Intent("plain", False, None, (Act("point", "{o}"), LOOK_O)),
    "ask_where": Intent("plain", True, "gaze", (EYES,)),                 # never a point or a look to it: the ask tests the word
    "ask_what": Intent("plain", True, "name", (EYES,)),     # of what the child attends: no show while an ask is pending (A51)
    "ask_give": Intent("plain", True, "act", (Act("open_hand", "child"), EYES)),    # her hand held out to the child, never
                                                                                     # toward the toy (A51)
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
    "feed_more": Intent("comfort", True, None, (EYES,)),        # "more?": the feed's own line, a question, never an ask
    "feed_done": Intent("comfort"),
    "leave": Intent("plain", False, None, (Act("wave", "child"), Act("walk", "door"))),
    "peekaboo_hide": Intent("plain", True, None, (Act("cover_face", "child"),)),
    "peekaboo": Intent("plain", False, None, (Act("reveal_face", "child"),)),
    "comfort": Intent("comfort", False, None, (Act("lean_in", "child_periphery"), Act("attend", "child"))),
    "hit": Intent("plain", False, None, (Act("withdraw", "child"),)),
    "no": Intent("no", False, None, (Act("withdraw", "child"),)),
    "no_talkover": Intent("no", False, None, (EYES,)),              # stage 2's "no." to a turn that talked over her (4.4, 4.6)
    "night": Intent("comfort", False, None, (Act("walk", "sofa"),)),
    "new_word": Intent("new_word", False, None, (LOOK_O, EYES)),
}
INTRO_ACTS = {"toy": (Act("show", "{o}"), EYES), "fixture": (Act("point", "{w}"), Act("look", "{w}", "focus")),
              "body": (Act("touch", "{w}"),), "face": (EYES,), "colour": (Act("show", "{o}"), EYES), "adj": (LOOK_O, EYES),
              "verb": (Act("do", "{v}"), EYES), "past": (EYES,), "social": (EYES,)}
assert all(k in TP.FRAMES for k in INTENTS if k != "new_word"), [k for k in INTENTS if k not in TP.FRAMES]
assert set(TP.FRAMES) <= set(INTENTS), set(TP.FRAMES) - set(INTENTS)


def register_for(intent, text):
    """the intent's register; a plain line that asks ("?") in the question register (the synthesizer raises the pitch on "?")."""
    r = INTENTS[intent].register
    return "question" if r == "plain" and text.endswith("?") else r


def acts_for(intent, refs=(), b=None, w=None, word_class=None):
    """the acts a line accompanies. refs: the objects the line is about (its filled slots, or the object its intent was given:
    "a rattle." shows the rattle; a verb's demonstration handles the toy its set is said of, templates.show_now); an act on "{o}"
    with no object is dropped (an intent that needs one is refused before it is said: the gaze and give asks)."""
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
        elif tgt == "{v}":
            need = TP.NEEDS.get(w, ())
            if len(need) < 2 or need[0] != "act":
                continue
            tgt = need[1]
            if tgt in TP.TOY_ACTS:                               # the toy it handles, named (A51: never an unnamed toy)
                out.append(Act(a.kind, tgt, a.during, refs[0] if refs else None))
                continue
        out.append(Act(a.kind, tgt, a.during))
    return tuple(out)


def blind(acts, x=None):
    """her acts while an ask about x is pending (A51; Golinkoff et al. 1987; the attention log's rule): only PENDING_OK's, her
    head and eyes on the child's eyes and her open hand held out to it, her face into its periphery, a hand withdrawn when hit;
    no point, show, turn, touch, guide, wave or demonstration: her hands at rest or open to it until the ask is judged. For the
    call (x her face, where every movement of her body is) her eyes on its eyes alone: no act that would put x in her log."""
    out = []
    for a in acts:
        if a.kind == "look":
            b = Act("look", "child_eyes")
        elif a.kind == "open_hand":
            b = Act("open_hand", "child")
        else:
            b = a
        if (b.kind, b.target) in PENDING_OK and b not in out and (x != PARENT_NAME or b == EYES):
            out.append(b)
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
    frown: str = None                                # stage 2: "talk_over" or "hit" (parent_feel.Feelings.talk_over / harm)
    copy: tuple = ()                                 # her copies of its movements made this tick (Act "copy", A52)


class FastLayer:
    """her lines and their manners. vocab: the words she has (birth words, then the growth words as they enter); new_words: the
    day's new words (at most NEW_PER_DAY; each said only last, one a line, in the new-word register; they join her words at
    the next night); held: the never-taught pairs still held out."""

    def __init__(self, seed=1, vocab=BIRTH_WORDS, stage=1):
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(STREAM,))))
        self.vocab = tuple(vocab)
        self.new_words = ()
        self.held = tuple(tuple(h) for h in K.HELD_PAIRS)
        self.stage = stage
        self.last_said = {}                   # a line's words (templates.key) -> the tick she last said it
        self.last_named = {}                  # object id -> tick of its last naming (a set counts once)
        self.last_set = {}                    # object id -> tick of its last variation set
        self.last_call = NEVER
        self.last_line = NEVER                # tick her last line ended
        self.last_expect = NEVER              # tick her last question, ask or call ended
        self.last_focus = None                # the focus word of the last line she finished (A27: expected as an echo)
        self.said_word = {}                   # word -> the tick she last finished saying it (the echo window, 4.8)
        self.busy_until = NEVER               # her current line sounds until this tick (exclusive)
        self.current = None                   # the line sounding
        self.spans = None                     # its words as they sound: dict(start, words=[[word, first sample, end sample]])
        self.queue = []                       # a set's later lines, each said 6 ticks after the last one ended
        self.recent = []                      # (tick, kind, object) she saw or heard in the last RECENT ticks
        self.steer = []                       # Claude's lines: [dict(text, situation, uses, tick_from, ttl)]
        self.refused = []                     # (tick, text, reason): the last REFUSED_KEEP lines the check or a rule refused
        self.last_new = NEVER                 # the tick her last new word's set began (NEW_EVERY, A14)
        self.n_lines = 0
        self.follow_in = 0                    # this life day's follow-in namings (a set once) and redirects (4.10: at least
        self.redirects = 0                    # FOLLOW_PER_REDIRECT follow-ins for each redirect; Tomasello and Farrar 1986)

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
        """a night boundary: the day's new words join her words, in the order she introduced them; the day's counts of
        follow-in namings and redirects start again."""
        self.vocab = self.vocab + tuple(w for w in self.new_words if w not in self.vocab)
        self.new_words = ()
        self.follow_in = self.redirects = 0

    def may_redirect(self):
        """4.10: follow-in naming outnumbers redirects at least 2 to 1 over the day, the redirect to be said counted."""
        return self.follow_in >= K.FOLLOW_PER_REDIRECT * (self.redirects + 1)

    def words_known(self):
        """her words and the day's new ones (what her ear and her transcript know, A15)."""
        return self.vocab + tuple(w for w in self.new_words if w not in self.vocab)

    def open_pair(self, pair):
        """a never-taught pair's test opens (A28): it is no longer held out of her lines."""
        self.held = tuple(h for h in self.held if tuple(h) != tuple(pair))

    # ------------------------------------------------------------------ composing
    def compose(self, intent, t, p, o=None, b=None, w=None, frames=None, skip=()):
        """-> a Line for the intent (its frames filled from the percept and held to the line check; the same line, the same
        words, not within SAME_LINE ticks; none whose words are in skip, a set's lines so far), picked by her stream, or None.
        The stream is drawn only when there is a choice to make."""
        cands = []
        for fr in (frames if frames is not None else TP.FRAMES[intent]):
            got = TP.fill(fr, o=o, b=b, w=w, fixtures=p.fixtures)
            if got is None:
                continue
            text, focus, refs = got
            refs = refs or ((o.id,) if o is not None else ())      # the object the line is about, slot or not ("what is this?")
            if TP.key(text) in skip or self.last_said.get(TP.key(text), NEVER) > t - K.SAME_LINE:
                continue
            reg = "new_word" if (intent == "new_word" or (focus is not None and focus in self.new_words)) else \
                register_for(intent, text)
            ok, why = TP.check(text, self.vocab, self.new_words, p, refs, self.held, recent_events=self.recent)
            if not ok:
                self.refused.append((t, text, why))
                continue
            cands.append(TP.Line(text, intent, reg, focus, focus, refs, "fast", fr[0]))
        if not cands:
            return None
        return cands[int(self.rng.integers(len(cands)))] if len(cands) > 1 else cands[0]

    def variation_set(self, intent, t, p, o=None, b=None, w=None, n=None, frames=None):
        """a variation set: n lines (2-3, drawn when not given) sharing the focus word, each from a frame differing by at least
        one word (4.5: two lines of the same words, their punctuation aside, are one frame: "a rattle." and "a rattle!") ->
        [Line]."""
        n = n or int(self.rng.integers(K.SET_LINES[0], K.SET_LINES[1] + 1))
        out, skip = [], set()
        for _ in range(n):
            ln = self.compose(intent, t, p, o, b, w, frames=frames, skip=skip)
            if ln is None:
                break
            out.append(ln)
            skip.add(TP.key(ln.text))
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
        if self.last_said.get(TP.key(line.text), NEVER) > t - K.SAME_LINE:
            return False, "the same line within 60 ticks"
        if line.refs and line.focus is not None and line.intent in NAMING and \
                self.last_named.get(line.refs[0], NEVER) > t - K.SAME_OBJECT:
            return False, "the same object named within 20 ticks"
        return True, ""

    def commit(self, line, t, n_ticks, spans, in_set=False):
        """she starts the line at t, sounding n_ticks; spans: [(word, first sample, end sample)] from the line's start. Her
        words count as said only as their sound ends (voiced()), so a word the talk-over stops is never said."""
        self.last_said[TP.key(line.text)] = t
        self.busy_until = t + max(1, int(n_ticks))
        self.current = line
        self.spans = dict(start=int(t), words=[[w, int(a), int(e)] for w, a, e in spans])
        self.n_lines += 1
        if line.intent in ("call", "hall_call"):
            self.last_call = t
        if line.refs and line.focus is not None and not in_set and line.intent in NAMING:
            self.last_named[line.refs[0]] = t
        self.last_focus = None                # set again when this line's focus word has been said (voiced)

    def voiced(self, w, te, last):
        """her word w finished sounding at te (the ledger's voiced(), each word once); last: it was the line's last word."""
        self.said_word[w] = te
        if last and self.current is not None:
            self.last_focus = self.current.focus

    def cut_point(self, t):
        """the talk-over at tick t (the child's sound heard at t; her line's samples up to the end of tick t have sounded):
        -> (the tick her line stops, the number of its words said), by the playback's own rule (body/sim/voice/playback.py
        Utterance.cut): the word sounding is finished if its end is at most CUT_MAX ticks away, else broken off at CUT_MAX
        ticks and not said; between words she stops at once; later words are never said."""
        sp = self.spans
        pos = (t + 1 - sp["start"]) * TICK
        stop = pos
        for _w, on, end in sp["words"]:
            if on < pos < end:
                stop = end if end - pos <= CUT_MAX * TICK else pos + CUT_MAX * TICK
                break
        kept = sum(1 for _w, _on, end in sp["words"] if end <= stop)
        return sp["start"] + -(-stop // TICK), kept

    def ended(self, t):
        """her line stopped sounding at t (whole, or cut by the talk-over)."""
        if self.current is not None:
            self.last_line = t
            if INTENTS[self.current.intent].expect:
                self.last_expect = t
            self.current = None
            self.spans = None
        self.busy_until = min(self.busy_until, t)

    # ------------------------------------------------------------------ Claude's lines (4.5, A14)
    def add_steer(self, text, situation, t, ttl=2000):
        """a steering row's line tied to a situation ("any", "holds:duck", "target:ball", "sees:drum"); checked now for form,
        vocabulary, the held-out pairs, praise and asks, and again for what she perceives each time it is said. -> (ok, why)."""
        ok, why = TP.check(text, self.vocab, self.new_words, None, (), self.held, source="claude")
        if ok:
            self.steer.append(dict(text=text, situation=situation, uses=0, tick_from=int(t), ttl=int(ttl)))
        else:
            self.refused.append((t, text, "steer: " + why))
        return ok, why

    @staticmethod
    def situation(sit, p):
        """a steering row's situation, as she reads the child (A40: its head's line and its hands, never its fovea's window):
        "any"; "target:X", its head's line on an X or a hand reaching toward one; "holds:X"; "reaches:X"; "sees:X", an X she
        sees."""
        kind, _, arg = sit.partition(":")
        if kind == "any":
            return True
        if kind == "holds":
            return any(p.obj(h) is not None and p.obj(h).name == arg for h in p.child_holds)
        if kind == "target":
            o = p.target_obj()
            return (o is not None and o.name == arg) or any(p.obj(r) is not None and p.obj(r).name == arg
                                                            for r in p.child_reaches)
        if kind == "reaches":
            return any(p.obj(r) is not None and p.obj(r).name == arg for r in p.child_reaches)
        if kind == "sees":
            return arg in p.names()
        return False

    def steer_line(self, t, p, redirect_ok=True):
        """Claude's line for the moment -> (Line, "follow_in" | "redirect" | None) or (None, None). A line naming a toy the child
        does not attend, as she reads it, is a redirect (4.10), said only when redirect_ok (40 ticks with no target) and the
        day's follow-in namings allow it (2 to 1); one naming only what it attends is a follow-in naming."""
        for s in self.steer:
            if s["uses"] >= K.STEER_USES or not (s["tick_from"] <= t < s["tick_from"] + s["ttl"]):
                continue
            if not self.situation(s["situation"], p) or self.last_said.get(TP.key(s["text"]), NEVER) > t - K.SAME_LINE:
                continue
            ok, why = TP.check(s["text"], self.vocab, self.new_words, p, (), self.held, source="claude",
                               recent_events=self.recent)
            if not ok:
                self.refused.append((t, s["text"], "steer: " + why))
                continue
            toys = TP.claude_toys(TP.claude_claims(s["text"])[0])
            att = {o.name for o in p.attended()}
            kind = None if not toys else ("follow_in" if set(toys) <= att else "redirect")
            if kind == "redirect" and not (redirect_ok and self.may_redirect()):
                why = (f"a redirect only after {K.REDIRECT_AFTER} ticks with no target" if not redirect_ok else
                       f"follow-in naming at least {K.FOLLOW_PER_REDIRECT} to 1 of redirects ({self.follow_in} to "
                       f"{self.redirects} today)")
                self.refused.append((t, s["text"], f"steer: names {', '.join(t_ for t_ in toys if t_ not in att)}, which "
                                                   f"the child does not attend: {why} (4.10)"))
                continue
            s["uses"] += 1
            last = TP.words(s["text"])[-1]
            focus = None if last in TP.FUNCTION else last          # her focus word, as in the frames: the last content word
            reg = "new_word" if last in self.new_words and last not in self.vocab else register_for("steer", s["text"])
            return TP.Line(s["text"], "steer", reg, focus, focus, (), "claude", s["situation"]), kind
        return None, None

    # ------------------------------------------------------------------ save
    def state(self):
        return dict(rng=self.rng.bit_generator.state, vocab=list(self.vocab), new_words=list(self.new_words),
                    held=[list(h) for h in self.held], stage=self.stage, last_said=dict(self.last_said),
                    last_named=dict(self.last_named), last_set=dict(self.last_set), last_call=self.last_call,
                    last_line=self.last_line, last_expect=self.last_expect,
                    last_focus=self.last_focus, said_word=dict(self.said_word), busy_until=self.busy_until,
                    current=None if self.current is None else _line_d(self.current),
                    spans=None if self.spans is None else dict(start=self.spans["start"],
                                                               words=[list(x) for x in self.spans["words"]]),
                    queue=[_line_d(ln) for ln in self.queue],
                    recent=[list(e) for e in self.recent], steer=[dict(s) for s in self.steer],
                    refused=[list(r) for r in self.refused], last_new=self.last_new, n_lines=self.n_lines,
                    follow_in=self.follow_in, redirects=self.redirects)

    def load_state(self, s):
        self.rng.bit_generator.state = s["rng"]
        self.vocab, self.new_words = tuple(s["vocab"]), tuple(s["new_words"])
        self.held, self.stage = tuple(tuple(h) for h in s["held"]), s["stage"]
        self.last_said, self.last_named, self.last_set = dict(s["last_said"]), dict(s["last_named"]), dict(s["last_set"])
        self.last_call, self.last_line, self.last_expect = s["last_call"], s["last_line"], s["last_expect"]
        self.last_focus, self.said_word = s["last_focus"], dict(s["said_word"])
        self.busy_until = s["busy_until"]
        self.current = None if s["current"] is None else _line_l(s["current"])
        self.spans = None if s["spans"] is None else dict(start=s["spans"]["start"],
                                                          words=[list(x) for x in s["spans"]["words"]])
        self.queue = [_line_l(d) for d in s["queue"]]
        self.recent = [tuple(e) for e in s["recent"]]
        self.steer = [dict(x) for x in s["steer"]]
        self.refused = [tuple(r) for r in s["refused"]]
        self.last_new = s["last_new"]
        self.n_lines = s["n_lines"]
        self.follow_in, self.redirects = s["follow_in"], s["redirects"]


NAMING = {"label", "label_held", "label_colour", "show", "redirect", "narrate_on", "new_word", "steer"}
INTENTS["steer"] = Intent("plain", False, None, (EYES,))        # Claude's line: plain (never the approval register, A14)


def _old_words(ws):
    """the words an older save's log gave a field or an act (one word, or a list that may hold None) -> this round's, fail-closed:
    those words and every thing (what lay near them, and beside her face, that save did not keep), or none at rest."""
    ws = [w for w in (ws if isinstance(ws, list) else [ws]) if w is not None]
    return sorted(set(ws) | {UNNAMED}) if ws else []


def _old_entry(e):
    """a tick of her log from an older save (P3's seventh round and before: each field one word, no face or voice) -> this
    round's form, fail-closed: each field's word and every thing, a face or voice it did not log moving, each act's words and
    every thing."""
    if all(isinstance(e.get(f), list) for f in LOG_FIELDS):
        return e
    out = dict(e)
    for f in LOG_FIELDS:
        out[f] = _old_words(e.get(f, UNNAMED))
    out["acts"] = [list(a[:4]) + [_old_words(a[4])] for a in e.get("acts", ())]
    return out


def _line_d(ln):
    return dict(text=ln.text, intent=ln.intent, register=ln.register, focus=ln.focus, emphasis=ln.emphasis, refs=list(ln.refs),
                source=ln.source, frame=ln.frame)


def _line_l(d):
    return TP.Line(d["text"], d["intent"], d["register"], d["focus"], d["emphasis"], tuple(d["refs"]), d["source"], d["frame"])


# ------------------------------------------------------------------------------------------------------------- conduct
ASKS_NEEDING_O = {"ask_where", "ask_give", "ask_what"}                # asks about an object: refused without one


class Conduct:
    """the speech side of L2. voice: anything with clip(text, register, emphasis=) -> a clip with .words [(word, first sample,
    end sample)], .pcm, .key and .digest (body/sim/voice/synth.VoiceCache; None: a line is timed at 3 ticks a word and has no
    clip). transcriber: body/sim/lang/transcriber.Transcriber (None: the child is not heard). ledger: body/sim/lang/ledger.Ledger.
    motion: W2's (default StubMotion), whose report of each tick, read first in it, is her attention log (A51). world: the
    world's inventory for a growth word's showing (A15; templates.showable: its objects and their colours, fixtures, events and
    her face), filled by the world from its scene (default: the living room at birth, templates.ROOM_AT_BIRTH); its acts are the
    motion's DOES. set_world() changes it (the colour twins' arrival, B2).
    reader: her reading of the child's head and hands (percept.Reader, A40), which the world calls to fill each Percept; saved
    here. imperfect: her imperfection (A52: turns missed, the reply's latency jittered, copying), always on in a life; False only
    for a test isolating another rule (then she answers every turn, REPLY_AFTER ticks after it, and copies nothing)."""

    def __init__(self, seed=1, voice=None, transcriber=None, ledger=None, motion=None, vocab=BIRTH_WORDS, stage=1, world=None,
                 imperfect=True):
        from .ledger import Ledger                                   # noqa: PLC0415 (the ledger imports nothing of this)
        self.fast = FastLayer(seed, vocab, stage)
        self.voice = voice
        self.transcriber = transcriber
        self.ledger = ledger if ledger is not None else Ledger()
        self.motion = motion if motion is not None else StubMotion()
        self.reader = Reader(seed)
        self.imperfect = bool(imperfect)
        self.imp = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(K.IMPERFECT_STREAM,))))
        self.world = None
        self.set_world(world if world is not None else TP.ROOM_AT_BIRTH)
        self.routine = None                   # the routine under way (L3 sets it: "feed", "greet", "leave", "peekaboo", ...)
        self.pending = None                   # the ask she is judging: dict(kind, word, obj, tick, open, until, trial)
        self.reply_due = None                 # the reply owed to the child's turn: dict(tick, kind, word, obj)
        self.no_target_since = 0              # the tick since which it has attended nothing, as she reads it (the redirect)
        self.attn = []                        # her attention log (A51): dict(t, eyes, head, left, right, acts) a tick, each field
                                              # by what a look that follows it would be read as (_word), acts [id, kind, target,
                                              # status, at] her acts not yet reported ended (those she asks for on the tick
                                              # among them) and those in her motion's report she did not ask for; the last
                                              # ATTN_KEEP ticks
        self.acts_open = []                   # her acts not yet reported ended: [motion id, kind, target, thing, tick, status,
                                              # at (the words it directs her body at)]; OUTSIDE's kind for one she did not ask for
        self.born = None                      # the first tick she lived (a tick of hers since then with no log is a cue at every
                                              # thing: fail-closed; none before it)
        self.face_until = NEVER               # her face moves through this tick (FACE_COURSE after a judgment or a frown: A3)
        self.focus_acts = []                  # her line's acts on its naming word (during='focus', 4.3): [the word's first
                                              # tick, its last, kind, target, thing, motion id once requested]
        self.last_vocal_smile = NEVER
        self.turn = None                      # the child's turn under way: dict(start, in_pause, looked, over)
        self.sound_hist = []                  # the last NONSTOP[1] ticks: was the child sounding
        self.nonstop_since = None
        self.requests = []                    # intents asked for by L3 / Claude: [(intent, kwargs)]
        self.cuts = 0
        self.turns = [0, 0]                   # the child's turns with no judgment: answered, missed (A52; P6's ruler, C24)
        self.copy_next = 0                    # her next copy no sooner than this tick (A52: about 6 a minute at most)
        self.copies = []                      # her copies due: [(tick, kind, her side)]

    @property
    def stage(self):
        return self.fast.stage

    @property
    def eyes_on_child(self):
        """an ask is pending (a gaze, a give, the call, a name): W2 keeps her head and eyes on the child's eyes and her hands at
        rest or held open to it, L1 turns to nothing (no gaze to its target or to a sudden event), and nothing else starts until
        it is judged (A51; her attention log must show it, or the ask is void; for the call her face moving voids it too)."""
        return self.pending is not None

    def set_world(self, world):
        """the world's inventory (plain data, saved with the conduct); its acts are her motion's."""
        w = dict(world)
        self.world = dict(objects={k: sorted(v) for k, v in sorted(dict(w.get("objects", {})).items())},
                          fixtures=sorted(w.get("fixtures", ())), events=sorted(w.get("events", ())),
                          face=sorted(w.get("face", ())), acts=sorted(getattr(self.motion, "DOES", ())))

    # ------------------------------------------------------------------ her attention log (A51)
    def _words(self, target, p):
        """a field's target by every thing a look that follows it could be read as attending (A40, A51): her Reader reads a look
        as the nameable thing nearest its line (4.10), with her error, so a look at a thing may be read as any thing near it, and
        a place or a part of the child is no thing of its own -> the sorted words:
          None (at rest) -> [];
          the child itself ("child", "child_eyes", "child_periphery": her eyes or face on its face, her hand held open or waved
            toward it, touching nothing) -> "child", and the things beside her face: a look that follows it comes back to her,
            the mutual gaze every ask is made in, read as her face or a thing beside it;
          her own face ("mama") -> "mama" and the things beside it;
          a thing she sees (an id in the tick's percept: a twin's id is its word's) -> its word, and the words of every thing
            its near names (Seen.near: within NEAR_DEG of it as seen from the child's head, or resting on or in it, or it on or
            in them) and of every thing resting on or in it or it on or in (Seen.on) (P3's eighth round: a look at the box the
            ball rests in, or at a toy a few degrees from the ball, may be read as the ball);
          anything else -> UNNAMED, which stands for every thing (fail-closed): a place (the shelf, the door, the floor, the
            window, the air a raised hand points at), the part of the child a hand touches or guides or her eyes rest on, a
            thing she does not see now, an id or a word she cannot name, and a thing (or her face) whose near the world does not
            give (None). The things beside her face (Percept.face_near) and in her hands (on "mama") are near her face."""
        if target is None:
            return []
        if not isinstance(target, str) or p is None:
            return [UNNAMED]
        if target in AT_CHILD or target == PARENT_NAME:
            base, near = {"child" if target in AT_CHILD else PARENT_NAME}, getattr(p, "face_near", None)
            rests = {s_.name for s_ in p.seen if s_.on == PARENT_NAME}
        else:
            o = p.obj(target)
            if o is None:
                return [UNNAMED]
            base, near = {o.name}, getattr(o, "near", None)
            rests = {s_.name for s_ in p.seen if s_.on == o.id}
            if p.obj(o.on) is not None:
                rests.add(p.obj(o.on).name)
        if near is None:
            return sorted(base | {UNNAMED})
        for z in near:
            q = p.obj(z) if z != PARENT_NAME else None
            base.add(PARENT_NAME if z == PARENT_NAME else q.name if q is not None else UNNAMED)
        return sorted(base | rests)

    def _at(self, act, p):
        """the words an act of hers directs her body at (directs(), each by _words), for her log from the tick she asks for it."""
        out = set()
        for _f, tg in directs(act):
            out |= set(self._words(tg, p))
        return sorted(out)

    def _request(self, act, t, p):
        """an act asked of her motion: kept open (running, whatever it reports but done, refused or cancelled) until her motion
        reports it ended (A51: an ask waits for every act of hers but PENDING_OK's), and in her log from this tick, with what it
        directs her body at, before her motion's first report of it (so a look asked for and cancelled at once is in it)."""
        mid = self.motion.request(act, t)
        at = self._at(act, p)
        self.acts_open.append([mid, act.kind, act.target, act.thing, int(t), "unreported", at])
        if self.attn and self.attn[-1]["t"] == t:
            self.attn[-1]["acts"].append([mid, act.kind, act.target, "asked", list(at)])
        return mid

    def _focus_step(self, t, start, p):
        """her acts on the naming word (4.3: her eyes on the object only through it, then back to the child's eyes): requested
        as the word begins (start: after her motion's report of the tick), cancelled from the tick after its last (at the end
        of that last tick, so the next report shows it ended), so the log shows them over the word."""
        keep = []
        for fa in self.focus_acts:
            if start and fa[5] is None and fa[0] <= t:
                fa[5] = self._request(Act(fa[2], fa[3], "focus", fa[4]), t, p)
            elif not start and fa[5] is not None and fa[1] <= t:
                self.motion.cancel(fa[5], t + 1)
                continue
            keep.append(fa)
        self.focus_acts = keep

    def _focus_clip(self, stop):
        """her line stopped at `stop` (the talk-over, or the world's playback): a naming word not reached is never looked on, one
        broken off ends there."""
        self.focus_acts = [fa[:1] + [min(fa[1], stop - 1)] + fa[2:] for fa in self.focus_acts if fa[5] is not None or
                           fa[0] < stop]

    def _sounding(self, t):
        """a word of her line sounds on tick t (her voice, at her face: the log's voice field)."""
        f = self.fast
        if f.spans is None or t >= f.busy_until:
            return False
        s0 = f.spans["start"]
        return any(s0 + a // TICK <= t <= s0 + (max(e_, 1) - 1) // TICK for _w, a, e_ in f.spans["words"])

    def _attend(self, t, p):
        """the tick's report from her motion into her attention log (A51): each of eyes, head and hands by _words (a field not
        reported is UNNAMED); her face moving where the report says so or leaves it out, or within FACE_COURSE ticks of a
        judgment of hers; her voice where a word of her line sounds; each of her open acts' status as reported (only done,
        refused or cancelled end it: anything else, or none, is running) and what it directs her body at, as asked and as this
        tick's percept names it; an act in the report she did not ask for (P4's, W2's own) kept open, OUTSIDE, until it is
        reported ended; a toy the percept shows in her hands at a hand (her first at rest, else her left) if the report names it
        at neither. Ticks since the last it read void a pending ask (fail-closed: the world did not tick her, and her body then
        could have cued the answer this tick shows); the tick's own log is held to a pending ask only once this tick's percept
        has judged it (_hold)."""
        rep = self.motion.report(t)
        got = rep.get("acts") if isinstance(rep.get("acts"), dict) else {}
        known = {a[0] for a in self.acts_open}
        keep = []
        for a in self.acts_open:
            st = got.get(a[0], "unreported")
            if st in ENDED:
                continue
            a[5] = st if isinstance(st, str) else "unreported"
            if a[1] != OUTSIDE:
                a[6] = sorted(set(a[6]) | set(self._at(Act(a[1], a[2], None, a[3]), p)))
            keep.append(a)
        for mid, st in got.items():                          # an act she did not ask for: running until reported ended
            if mid not in known and st not in ENDED:
                keep.append([mid, OUTSIDE, None, None, int(t), st if isinstance(st, str) else "unreported", [UNNAMED]])
        self.acts_open = keep
        e = dict(t=int(t))
        for fld in FIELDS[:4]:
            e[fld] = self._words(rep.get(fld, UNNAMED), p)
        face = rep.get("face", PARENT_NAME)                 # left out: her face may be moving (fail-closed)
        e["face"] = self._words(PARENT_NAME, p) if face is not None or t <= self.face_until else []
        e["voice"] = self._words(PARENT_NAME, p) if self._sounding(t) else []
        raw = {f: rep.get(f) for f in HANDS}
        for s_ in p.seen:                                   # a toy she sees in her own hands: a hand is at it, whatever the
            if s_.on != PARENT_NAME:                        # report says (the stub models no hold)
                continue
            if any(v == s_.id or (isinstance(v, str) and p.obj(v) is not None and p.obj(v).name == s_.name)
                   for v in raw.values()):
                continue
            h = next((f for f in HANDS if not e[f] or "child" in e[f]), "left")
            e[h], raw[h] = self._words(s_.id, p), s_.id
        e["acts"] = [[a[0], a[1], a[2], a[5], list(a[6])] for a in keep]
        prev = self.attn[-1]["t"] if self.attn else None
        self.attn = (self.attn + [e])[-ATTN_KEEP:]
        pd = self.pending
        if pd is not None and t > pd["tick"] and prev is not None and t - prev > 1:
            self.ledger.withdraw(t, pd["trial"], f"her attention log while it was pending: ticks {prev + 1}-{t - 1} never read "
                                                 f"(A51: fail-closed)")
            self.pending = None

    def _mark(self, t, fld, p):
        """her face or her voice at her face on tick t, logged by the conduct itself (a judgment's smile, a frown, a line
        begun)."""
        if self.attn and self.attn[-1]["t"] == t:
            self.attn[-1][fld] = self._words(PARENT_NAME, p)

    @staticmethod
    def _hits(e, x):
        """what in a tick of her log could cue X: a field at X or at a target that stands for every thing (UNNAMED), an act at
        either (her acts from the tick she asks for them; one she did not ask for, at every thing) -> (who, at what) or None.
        Her own face (X of the call) is where every movement of her body is, or beside it: a look drawn to her turning head or
        to her hand, whatever it points at, is read as her face, the nameable thing nearest its line (A40). So for her face a
        tick with her eyes or head off the child, a hand not at rest, her face moving, her voice, or any act of hers under way
        but her eyes on its eyes could cue it (P3's seventh and eighth rounds)."""
        flds = [f for f in LOG_FIELDS if x in e[f] or UNNAMED in e[f]]
        acts = [a for a in e["acts"] if x in a[4] or UNNAMED in a[4]]
        body = False
        if x == PARENT_NAME:
            moved = [f for f in ("eyes", "head") if "child" not in e[f]] + [f for f in HANDS + ("face", "voice") if e[f]]
            others = [a for a in e["acts"] if a[1] == OUTSIDE or (a[1], a[2]) != ("look", "child_eyes")]
            body = bool(moved or others)
            flds += [f for f in moved if f not in flds]
            acts += [a for a in others if a not in acts]
        if not flds and not acts:
            return None
        who = (["her " + " and ".join(flds)] if flds else []) + \
            ["an act she did not ask for" if a[1] == OUTSIDE else f"her {a[1]}" for a in acts]
        at_x = any(x in e[f] for f in flds) or any(x in a[4] for a in acts)
        return ", ".join(who), (x if at_x else "her body, by her face" if body else "a place or a thing she cannot name")

    def _pending_why(self, e, pd):
        """why a tick of her log voids the pending ask pd, or None (A51): her body not still (_unstill), or its X in the log
        (_hits: the conduct refuses every line and act that would put it there, so only W2's report, L1's or the world's face
        can). For the call her voice until its name has been heard is the ask itself."""
        x = PARENT_NAME if pd["kind"] == "call" else pd["word"]
        ee = dict(e, voice=[]) if pd["kind"] == "call" and e["t"] <= pd["open"] else e
        hit = self._hits(ee, x)
        parts = [w for w in (_unstill(e), None if hit is None else f"{hit[0]} at {hit[1]}") if w]
        return "; ".join(parts) or None

    def _hold(self, t):
        """the tick's log held to the ask still pending once this tick's percept has judged it (A51; P3's eighth round). An
        answer the percept of a tick shows was made before that tick (a look held 3 readings and then 2 ticks, a reach closing
        over 3, a give at the end of its hand's path), so her body on the tick cannot have cued it, and every tick before was
        held on its own tick: the given toy in her hand on the tick the child gives it never voids the give it answers. A tick
        whose log shows her body not still, or X in it, voids an ask still pending (fail-closed: W2 broke its contract, or her
        face moved with a judgment)."""
        pd = self.pending
        if pd is None or t <= pd["tick"] or not self.attn or self.attn[-1]["t"] != t:
            return
        why = self._pending_why(self.attn[-1], pd)
        if why is not None:
            self.ledger.withdraw(t, pd["trial"], f"her attention log while it was pending: {why} (A51: fail-closed)")
            self.pending = None

    def _ask_block(self, x, t):
        """why an ask about X may not open at t, or None (A51; the lead's decision in P3's sixth round, the seventh's naming, the
        eighth's near things and face): an act still running, queued or unreported that she may not make while an ask is
        pending, or any act in her motion's report she did not ask for; a look on a naming word still to come; a tick of her life
        among t and the CUE_CLEAR before it with no log (never read: a cue at every thing), or whose log could cue X (_hits: X,
        a thing a look at X could be read as, or a place or a thing she cannot name, which stands for every thing; for her face,
        any movement of her body, her face or her voice); her eyes or head off the child or a hand at anything but it on t."""
        for a in self.acts_open:
            if a[1] == OUTSIDE or (a[1], a[2]) not in PENDING_OK:
                who = f"an act she did not ask for (her motion's id {a[0]!r})" if a[1] == OUTSIDE else \
                    f"her {a[1]} at {a[3] or a[2] or 'nothing'}"
                return (f"{who} is still under way (her motion reports it {a[5]!r}; only done, refused or cancelled end an "
                        f"act: fail-closed)")
        for fa in self.focus_acts:
            return f"her {fa[2]} at {fa[3]} on her line's naming word is still to come (tick {fa[0]})"
        log = {e["t"]: e for e in self.attn}
        first = t - K.CUE_CLEAR if self.born is None else max(self.born, t - K.CUE_CLEAR)
        for u in range(t, first - 1, -1):
            e = log.get(u)
            if e is None:
                return (f"her attention log has no report of tick {u} (never read): a tick not read is a cue at every thing "
                        f"(fail-closed)")
            hit = self._hits(e, x)
            if hit is not None:
                when = f"is under way ({hit[0]} at {hit[1]} on this tick)" if u == t else \
                    f"({hit[0]} at {hit[1]}, tick {u}) ended {t - u - 1} ticks ago"
                return (f"her own cue at a {x} {when}: a look or a reach within {K.CUE_CLEAR} ticks of it may follow her, not "
                        f"the word (Brooks and Meltzoff 2005's 6.5 s)")
        why = _unstill(log[t])
        if why is not None:
            return f"{why}: an ask opens only with her eyes and head on the child and her hands at rest or open to it"
        return None

    def cue_clear(self, x, t):
        """an ask about X may open at t (a thin reader of her attention log: _ask_block's rule); P4's base-rate trials are drawn
        under the same rule, so the base rate is measured as the asks are."""
        return self._ask_block(x, t) is None

    def request(self, intent, **kw):
        """an episode's or Claude's intent (L3, P4; P5), said when the priorities allow (a request that cannot be said is
        dropped and logged in fast.refused)."""
        self.requests.append((intent, dict(kw)))

    def night(self):
        """a night boundary: the day's new words join her words; her ear is checked against its digest (A27: nothing changed
        it) and must know every word she now has."""
        self.fast.night()
        ear = None if self.transcriber is None else self.transcriber.ear
        if ear is not None:
            ear.verify(self.transcriber.pin)                # against the pin taken when her ear was loaded, saved with the world
            missing = ear.missing(self.fast.vocab)
            if missing:
                raise ValueError(f"her ear has no templates for {missing}: it is built with every word before birth "
                                 f"(tools/sim_parent_ear.py --words all)")

    # ------------------------------------------------------------------ her imperfection (A52)
    def _latency(self):
        """her reply's latency after the child's turn ends, in ticks: her switching pause drawn from REPLY_PAUSE_MS (Gratier et
        al. 2015), less the turn's 2 quiet ticks; at the soonest the tick she knows the turn ended (0: its 2 quiet ticks, a
        pause of 300 ms, are the floor; P3's fourth round, where a floor of 1 more tick put 43% of her replies on it and her
        realized pause at 774 ms)."""
        if not self.imperfect:
            return K.REPLY_AFTER
        m, sd, lo, hi = K.REPLY_PAUSE_MS
        s2 = math.log(1.0 + (sd / m) ** 2)
        pause = min(hi, max(lo, math.exp(self.imp.normal(math.log(m) - s2 / 2, math.sqrt(s2)))))
        return max(0, int(round(pause / (1000.0 * TICK / SR))) - K.TURN_END_REST)

    def _copying(self, t, p, out):
        """her copies of its visible arm and hand movements (A52; Ray and Heyes 2011): one seen while she attends it, when her
        copying's gap allows, is copied mirrored within 1-2 s; never while an ask is pending (her hands stay still, A51): a
        copy due then is dropped, not made late (a copy after the ask would come 1-2 s past the movement, no copy of it)."""
        if self.imperfect and p.present and p.child_in_view:
            for kind, side in p.events:
                if kind in K.COPY_KINDS and t >= self.copy_next:
                    mirror = {"left": "right", "right": "left"}.get(side, side)
                    self.copies.append((t + int(self.imp.integers(K.COPY_DELAY[0], K.COPY_DELAY[1] + 1)), kind, mirror))
                    self.copy_next = t + max(1, int(round(self.imp.exponential(K.COPY_GAP_S / (TICK / SR)))))
        due = [c for c in self.copies if c[0] <= t]
        self.copies = [c for c in self.copies if c[0] > t]
        if self.eyes_on_child:
            return
        acts = tuple(Act("copy", f"{kind}:{side}") for _t, kind, side in due)
        for a in acts:
            self._request(a, t, p)                          # open until her motion reports it ended: no ask opens meanwhile
        out.copy = acts

    # ------------------------------------------------------------------ the tick
    def tick(self, t, p, tract=None, distance_m=1.0, token=None, voice_done=None):
        """one tick. p: her Percept (child_target, child_holds and child_reaches as she reads them: self.reader, A40); tract:
        the child's tract samples this tick (engine units at 1 m) or None at rest; distance_m: from the child's mouth to her
        head; token: the silent token output's symbol this tick (lexicon ids) or None; voice_done: the tick her line stopped
        sounding, when the world's playback says so (a cut); otherwise the clip's own length ends it. Her motion's report of the
        tick (StubMotion.report: W2 makes it before this runs) is read first, into her attention log (A51). -> Say."""
        f = self.fast
        f.observe(p)
        if self.born is None:
            self.born = int(t)
        self._attend(t, p)                                  # her motion's report of this tick: her attention log (A51)
        self._focus_step(t, start=True, p=p)                # her looks on a naming word that begins now
        for w, te, last in self.ledger.voiced(t, p):        # her words whose sound has ended by now: said (4.6, 4.8)
            f.voiced(w, te, last)
        if voice_done is not None:
            f.ended(voice_done)
            self._focus_clip(voice_done)
        elif f.current is not None and t >= f.busy_until:
            f.ended(f.busy_until)
        out = Say()
        if self.stage >= 2 and any(k == "hit_her" for k, _o in p.events):
            out.frown = "hit"                               # stage 2: the frown (-1) for its own act that hit her (4.10)
        if p.child_target is not None or p.child_holds or p.child_reaches:
            self.no_target_since = t + 1
        # her ears (the transcriber): the child's turn, its words at the turn's end
        heard, sounding = [], p.child_sounding
        if self.transcriber is not None:
            vocab = f.words_known()                          # her ear knows her words and the day's new ones
            exp = self.transcriber.expected(p, self.pending, f.last_focus, self.routine, vocab)
            heard = self.transcriber.tick(t, tract, distance_m, token, f.speaking(t), exp, p, f.said_word, vocab)
            sounding = self.transcriber.sounding
            for cw in heard:
                self.ledger.accepted(cw, p)
        out.heard = heard
        self.sound_hist = (self.sound_hist + [bool(sounding)])[-K.NONSTOP[1]:]
        nonstop = len(self.sound_hist) == K.NONSTOP[1] and np.mean(self.sound_hist) > K.NONSTOP[0]
        self.nonstop_since = (self.nonstop_since if self.nonstop_since is not None else t) if nonstop else None
        if sounding and self.turn is None:
            self.turn = dict(start=t, in_pause=not f.speaking(t), looked=False, over=False)
            if f.speaking(t):                               # the talk-over: she finishes her word, stops, listens (4.6)
                out.cut, out.listen = True, True
                self.cuts += 1
                stop, kept = f.cut_point(t)
                self.ledger.cut(t, f.current, kept, stop)
                f.busy_until = min(f.busy_until, stop)      # the world's voice_done may end it sooner
                f.queue = []
                self._focus_clip(stop)
                if self.pending is not None and self.pending["tick"] == f.spans["start"] and \
                        self.pending["open_idx"] >= kept:        # her ask stopped before its word was said: withdrawn
                    self.ledger.withdraw(t, self.pending["trial"], "cut before its word was said")
                    self.pending = None
                if self.stage >= 2 and self.nonstop_since is None:   # stage 2: the talk-over frown, and "no." for its turn
                    out.frown = "talk_over"                          # (never for babble that never stops, A13)
                    self.turn["over"] = True
        if self.turn is not None:
            self.turn["looked"] |= p.child_target == "mama"
        # the asks she is judging (4.6: 20 ticks for a gaze, 40 for an act; 4.8: from the moment the word was heard)
        for word, kind, tid, result in self.ledger.observe(t, p):
            if self.pending is None or self.pending["trial"] != tid:
                continue
            if result == "met":
                if kind != "call" or not self.ledger.understood(NAME):     # the call answered, until the name is understood
                    out.judgments.append((K.WORTH_MET_ASK, "met_ask", word))
                self.reply_due = dict(tick=t, kind="confirm", word=word, obj=self.pending.get("obj"))
            self.pending = None
        if self.pending is not None and t > self.pending["until"]:
            self.pending = None
        for cw in heard:                                    # the child's turn ended (or its tokens were read): judge, reply
            if cw.channel == "tract" or cw.word is not None:
                self._owe_reply(t, cw, p, out)
        if out.judgments or out.frown:                      # her face will move (a smile, a frown): at her face in her log
            self.face_until = max(self.face_until, t + K.FACE_COURSE)
            self._mark(t, "face", p)
        self._hold(t)                                       # the tick's log held to an ask its percept has not judged (A51)
        if (self.transcriber is not None and self.transcriber.turn_end == t) or (self.transcriber is None and not sounding):
            self.turn = None                                # (with no transcriber, the world's own sounding flag ends the turn)
        self._copying(t, p, out)
        got = self._choose(t, p, sounding)
        if got is not None:
            self._say(got[0], t, p, out, in_set=got[1])
            if got[2] == "follow_in":
                f.follow_in += 1
            elif got[2] == "redirect":
                f.redirects += 1
        self._focus_step(t, start=False, p=p)               # her looks on a naming word ending now: cancelled from t + 1
        return out

    def _right(self, w, start, p):
        """is w the right word here (4.3's right name)? -> (right, asked): its referent where she reads the child looking, in
        its hand or reached toward (A40), her face for "mama" when she reads it looking at her, or the answer to her name ask: a
        word begun (start: the tick the child made it, the token or the utterance's first sound, never the tick she read it)
        once its question was heard."""
        asked = self.pending is not None and self.pending["kind"] == "name" and self.pending["word"] == w and \
            start >= self.pending["open"]
        return (w in p.attended_names() or asked), asked

    def _owe_reply(self, t, cw, p, out):
        """the child's turn ended with cw: the judgment now, the reply after her latency (4.6, 4.3's worth table, A27, A52).
        An echo (said within ECHO_WINDOW of her saying the word) is answered as imitation: judged as the word said then would
        be, so it may earn her smile (her method: a parent answers imitation, Goldstein and Schwade 2008), but it counts toward
        nothing in the ledger, neither "says" nor "understood" nor a met ask: a name ask answered by an echo is void. A turn that
        talked over her in stage 2 is answered "no." and not judged (the frown, 4.4, 4.6). A turn she makes no judgment of she
        misses at MISS_TURN (A52); a judged one she always answers."""
        tgt = p.target_obj()
        kind, obj, w = "reply", tgt, cw.word
        n_judg = len(out.judgments)
        pd = self.pending
        if w is not None and pd is not None and pd["kind"] == "name" and pd["word"] == w and cw.start < pd["open"]:
            # its word begun before her question was heard: no answer to it, and her reply would echo the answer, so the
            # ask is void (as a gaze ask is when its X is attended as its word is heard, 4.8)
            self.ledger.withdraw(t, pd["trial"], f"{w!r} begun before the question was heard")
            self.pending = None
        over = cw.channel == "tract" and self.turn is not None and self.turn.get("over") and self.stage >= 2
        if over:
            kind, w, obj = "no", None, None
        elif w is not None:
            right, asked = self._right(w, cw.start, p)
            named = {s.name: s for s in p.attended()}
            obj = named.get(w) or (p.obj(self.pending["obj"]) if asked and self.pending.get("obj") else None) or \
                next((s for s in p.seen if s.name == w), None)
            if cw.exact and right:
                label = "echo" if cw.echo else ("met_ask" if asked else "right_name")
                out.judgments.append((K.WORTH_RIGHT_NAME, label, w))
                kind = "confirm"
                if asked:
                    if cw.echo:                                # imitation: smiled at, never a met ask (the ledger)
                        self.ledger.withdraw(t, self.pending["trial"], f"answered by an echo of her own {w!r}")
                    else:
                        self.ledger.named(t, w, cw.start)
                    self.pending = None
            elif not cw.exact and right and self.stage >= 2 and self.ledger.exact_count(w, cw.channel) < K.EXACT_UNTIL:
                out.judgments.append((K.WORTH_APPROX, "approximation", w))
                kind = "recast"
            elif cw.exact or w in cw.expected:
                kind = "echo"                                  # heard in context: echoed, no smile
            else:
                kind, w, obj = "reply", None, tgt              # an approximation out of context: answered as a vocal turn
        if not over and self.stage == 1 and cw.channel == "tract" and self.turn is not None and self.turn["in_pause"] and \
                self.turn["looked"] and t - self.last_vocal_smile >= K.VOCAL_TURN_EVERY:
            out.judgments.append((K.WORTH_VOCAL_TURN, "vocal_turn", None))
            self.last_vocal_smile = t
        after = self._latency()
        missed = self.imperfect and self.imp.random() < K.MISS_TURN   # drawn for every turn, so a replay is exact
        if len(out.judgments) == n_judg and not over:
            self.turns[1 if missed else 0] += 1
            if missed:                                         # she missed it: no reply to it (A52); a reply she owes a
                return                                         # judgment made this tick (a met ask's) stands
        self.reply_due = dict(tick=t + after, kind=kind, word=w, obj=None if obj is None else obj.id)

    def _target(self, p):
        """the child's target for her follow-in naming, as she reads it (4.10, A40): where its head's line is (her reading, held
        TARGET_TICKS running: percept.Reader.look), else what a hand reaches toward, else what it holds -> a Seen, or None."""
        o = p.target_obj()
        if o is not None:
            return o
        for oid in tuple(p.child_reaches) + tuple(p.child_holds):
            if p.obj(oid) is not None:
                return p.obj(oid)
        return None

    def _choose(self, t, p, sounding):
        """-> (Line, in_set, "follow_in" | "redirect" | None) by the priorities (4.10), or None."""
        f = self.fast
        if not f.voice_free(t):
            return None
        ev = {k for k, _ in p.events}
        # while the call is pending her voice is at its X, her face (A51): no line until it is judged (the reply she owes
        # waits), but the child's pain or distress, or being hit, which void it first
        pd = self.pending
        if pd is not None and pd["kind"] == "call":
            if not ((ev & {"pain", "distress"} and p.present) or "hit_her" in ev):
                return None
            self.ledger.withdraw(t, pd["trial"], "her attention log while it was pending: her voice at mama (she answered "
                                                 "its pain or its hit; A51)")
            self.pending = None
        # 1. the child's pain or distress: comfort (never a smile)
        if ev & {"pain", "distress"} and p.present:
            ln = f.compose("comfort", t, p)
            if ln is not None and f.allowed(ln, t, reply=True)[0]:
                f.queue = []
                return ln, False, None
        # 2. being hit by the child's own act: stage 1 "oh!", stage 2 "no." (a reflex she triggered is her defect: no line)
        if "hit_her" in ev:
            ln = f.compose("hit" if self.stage == 1 else "no", t, p)
            if ln is not None:
                return ln, False, None
        # the child's turn: she listens (both starting on one tick: the child has it), unless its babble never stops (A13);
        # and she holds her voice for the reply she owes it, after her latency
        if (sounding or self.turn is not None) and not (self.nonstop_since is not None and
                                                        t - self.nonstop_since >= K.NONSTOP[2]):
            return None
        if self.reply_due is not None and t < self.reply_due["tick"]:
            return None
        # 3. a low charge: the meal's first line (L3 runs the routine); it waits while a gaze, act or call ask is judged, whose
        #    window is shorter than her 200 ticks to check (4.7), since its bottle in hand would cue the child (A51)
        if "charge_low" in ev and p.present and self.routine != "feed" and not self.eyes_on_child:
            ln = f.compose("feed", t, p)
            if ln is not None and f.allowed(ln, t)[0]:
                return ln, False, None
        # 4. the reply owed to the child's turn, after her latency
        if self.reply_due is not None and t >= self.reply_due["tick"]:
            r, self.reply_due = self.reply_due, None
            f.queue = []                                   # the turn answered; the set it broke into is not resumed
            ln = self._reply_line(r, t, p)
            if ln is not None and f.allowed(ln, t, reply=True)[0]:
                return ln, False, None
        # 5. a set's later lines, 6 ticks after the last one ended
        while f.queue:
            ln = f.queue[0]
            if not f.allowed(ln, t, in_set=True)[0]:
                return None
            f.queue.pop(0)
            ok, why = TP.check(ln.text, f.vocab, f.new_words, p, ln.refs, f.held, recent_events=f.recent)
            if ok:
                return ln, True, None
            f.refused.append((t, ln.text, "set: " + why))
        # 6. a pending ask: the expectant pause while she judges it
        if self.pending is not None:
            return None
        # 7. joint attention: a follow-in variation set naming its target as she reads it (4.10; her gaze goes there in L1)
        o = self._target(p)
        if o is not None and f.last_named.get(o.id, NEVER) <= t - K.SAME_OBJECT and \
                f.last_set.get(o.id, NEVER) <= t - K.SET_PER_OBJECT:
            lines = f.variation_set("label_held" if o.id in p.child_holds else "label", t, p, o=o)
            if lines and f.allowed(lines[0], t)[0]:
                f.last_set[o.id] = t
                f.queue = lines[1:]
                return lines[0], False, "follow_in"
        # 8. the episode's act: an intent asked for by L3 or Claude, and Claude's lines for the moment
        while self.requests:
            intent, kw = self.requests.pop(0)
            got = self._compose_request(intent, t, p, kw)
            if got is not None:
                return got, False, ("redirect" if intent == "redirect" else None)
        ln, kind = f.steer_line(t, p, redirect_ok=t >= self.no_target_since + K.REDIRECT_AFTER)
        if ln is not None and f.allowed(ln, t)[0]:
            return ln, False, kind
        # 9. idle: she watches (the episodes' idle lines, at most one per 40 ticks, are P4's)
        return None

    def _drop(self, t, intent, why):
        self.fast.refused.append((t, intent, "request: " + why))
        return None

    def _compose_request(self, intent, t, p, kw):
        """an intent asked for by L3 or Claude -> a Line, or None (dropped, and logged in fast.refused with why)."""
        f = self.fast
        if intent == "new_word":
            return self._introduce(kw["word"], t, p, kw.get("o"))
        o = p.obj(kw["o"]) if kw.get("o") else None
        if kw.get("o") and o is None:
            return self._drop(t, intent, f"unseen: {kw['o']!r}")
        it = INTENTS[intent]
        if intent in ("label", "label_held", "label_colour", "show", "redirect") and o is not None:
            if intent == "redirect" and t < self.no_target_since + K.REDIRECT_AFTER:
                return self._drop(t, intent, f"a redirect only after {K.REDIRECT_AFTER} ticks with no target (4.10)")
            if intent == "redirect" and not f.may_redirect():
                return self._drop(t, intent, f"follow-in naming at least {K.FOLLOW_PER_REDIRECT} to 1 of redirects "
                                             f"({f.follow_in} to {f.redirects} today; 4.10)")
            if f.last_set.get(o.id, NEVER) > t - K.SET_PER_OBJECT:
                return self._drop(t, intent, f"a set on {o.id!r} within {K.SET_PER_OBJECT} ticks")
            lines = f.variation_set(intent, t, p, o=o, n=kw.get("n"))
            if not lines:
                return self._drop(t, intent, "no line passes the check")
            ok, why = f.allowed(lines[0], t)
            if not ok:
                return self._drop(t, intent, why)
            f.last_set[o.id] = t
            f.queue = lines[1:]
            return lines[0]
        if intent in ASKS_NEEDING_O and o is None:
            return self._drop(t, intent, "an ask about an object needs its object")
        if it.ask == "gaze" or it.ask == "act":
            if not o.child_sees:
                return self._drop(t, intent, "an ask needs its object in the child's view (4.8)")
            if it.ask == "gaze" and o.name in p.attended_names():
                return self._drop(t, intent, f"a {o.name} already where she reads the child looking, in its hand or reached "
                                             f"toward: the ask would be met unasked (4.8, A40)")
            if any(s.name == o.name and s.on == PARENT_NAME for s in p.seen):
                return self._drop(t, intent, f"a {o.name} in her own hands: a look at her, or at her hand, would meet it "
                                             f"(A51; P3's fourth round)")
            if it.ask == "act" and not any(s.name == o.name and s.child_can_reach for s in p.seen):
                return self._drop(t, intent, f"no {o.name} within the child's reach as she sees it: no act of its could "
                                             f"answer it (4.8; P3's fourth round)")
        if it.ask == "name" and o.id not in {s.id for s in p.attended()}:
            return self._drop(t, intent, f"a name ask is of what the child attends as she reads it (its head's line on it, "
                                         f"in its hand or reached toward): while it is pending she shows nothing, so "
                                         f"'what is this?' can mean only that (A51's attention log)")
        if it.ask == "call" and p.child_target == "mama":
            return self._drop(t, intent, "she reads the child already looking at her face: nothing to call it to")
        if it.ask is not None and self.pending is not None:
            return self._drop(t, intent, "an ask is already pending")
        if it.ask is not None:                              # her attention log (A51): one rule for every ask, fail-closed
            why = self._ask_block(PARENT_NAME if it.ask == "call" else o.name, t)
            if why is not None:
                return self._drop(t, intent, why + " (A51)")
        frames = None
        if it.ask == "call":                                # her voice is at her face, the call's X: its line ends on its name,
            frames = [fr for fr in TP.FRAMES[intent]        # so no word of hers follows it while it is judged (A51)
                      if TP.words(TP.fill(fr)[0])[-1:] == [NAME]]
        ln = f.compose(intent, t, p, o=o, b=kw.get("b"), w=kw.get("w"), frames=frames)
        if ln is None:
            return self._drop(t, intent, "no line passes the check")
        ok, why = f.allowed(ln, t)
        if not ok:
            return self._drop(t, intent, why)
        return ln

    def _introduce(self, word, t, p, oid=None):
        """a new word's set (4.8): 3 lines, the word last and emphasized, in the new-word register, its referent shown (A15),
        each line measured to put the word on its pitch peak (A34: the line check); at most NEW_PER_DAY a day, each kept until
        the night (a second new word never displaces the first)."""
        f = self.fast
        if word in f.vocab:
            return self._drop(t, "new_word", f"{word!r} is already hers")
        if word not in TP.GROWTH_CLASS:
            return self._drop(t, "new_word", f"{word!r} is not in the growth queue (4.8, A15)")
        if word not in f.new_words and len(f.new_words) >= K.NEW_PER_DAY:
            return self._drop(t, "new_word", f"{word!r}: at most {K.NEW_PER_DAY} new words a life day (4.8)")
        if word not in f.new_words and t < f.last_new + K.NEW_EVERY:
            return self._drop(t, "new_word", f"{word!r}: at most 1 new word a minute ({K.NEW_EVERY} ticks, A14)")
        ear = None if self.transcriber is None else self.transcriber.ear
        if ear is not None and ear.missing((word,)):
            return self._drop(t, "new_word", f"{word!r}: her ear has no templates for it (built before birth, --words all)")
        ok, why = TP.showable(word, self.world)
        if not ok:
            return self._drop(t, "new_word", f"{word!r} waits: {why} (A15)")
        cands, why = TP.show_now(word, p, f.recent)
        if oid:
            cands = [o for o in cands if o is not None and o.id == oid]
            why = why or f"{oid!r} does not show it"
        if not cands:
            return self._drop(t, "new_word", f"{word!r} cannot be shown now: {why} (A15)")
        prev = f.new_words
        if word not in f.new_words:
            f.new_words = f.new_words + (word,)
        lines = []
        for o in cands:                                   # a colour on each toy of it in turn (never a held-out pair's noun)
            lines = f.variation_set("new_word", t, p, o=o, w=word, n=3, frames=TP.intro_frames(word))
            if len(lines) == 3:
                break
        if len(lines) < 3:
            f.new_words = prev
            return self._drop(t, "new_word", f"{word!r}: {len(lines)} lines of distinct words pass, of the 3 its set needs (the "
                                             f"check: its frames, what she sees, the held-out pairs, the pitch peak, A34; "
                                             f"4.5: frames differing by at least one word)")
        ok, why = f.allowed(lines[0], t)
        if not ok:
            f.new_words = prev
            return self._drop(t, "new_word", f"{word!r}: its set's lines pass, but not now: {why}")
        f.queue = lines[1:]
        f.last_new = t
        return lines[0]

    def _reply_line(self, r, t, p):
        f = self.fast
        o = p.obj(r["obj"]) if r.get("obj") else None
        w = r.get("word")
        noun = w in TP.NOUNS
        if r["kind"] == "no":
            return f.compose("no_talkover", t, p)
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
            n = int(math.ceil(len(clip.pcm) / TICK))
            spans = [(w, int(a), int(e)) for w, a, e in clip.words]
        else:
            ws = TP.words(line.text)
            n = 3 * len(ws)                                  # no voice: 3 ticks a word (an instrument's timing, ours)
            spans = [(w, 3 * i * TICK, (3 * i + 3) * TICK) for i, w in enumerate(ws)]
        word_ends = [(w, t + (max(e, 1) - 1) // TICK) for w, _a, e in spans]
        f.commit(line, t, n, spans, in_set=in_set)
        self.ledger.said(t, line, p, clip=clip, n_ticks=n, word_ends=word_ends)
        it = INTENTS[line.intent] if line.intent in INTENTS else None
        if it is not None and it.ask is not None and self.pending is None:
            self._open_ask(line, it, t, n, word_ends, p)
        cls = TP.GROWTH_CLASS.get(line.focus) if line.intent == "new_word" else None
        acts = acts_for(line.intent, line.refs, b=line.focus, w=line.focus, word_class=cls)
        if self.eyes_on_child:                              # an ask she is judging: only acts that keep her still and put
            pd = self.pending                               # nothing of its X in her log (A51)
            acts = blind(acts, PARENT_NAME if pd["kind"] == "call" else pd["word"])
        for a in acts:
            if a.during == "focus":                         # on the naming word only (4.3): requested as it begins
                fw = [(wa, we) for w, wa, we in spans if w == line.focus] or [(wa, we) for _w, wa, we in spans[-1:]]
                wa, we = fw[-1]
                self.focus_acts.append([t + wa // TICK, t + (max(we, 1) - 1) // TICK, a.kind, a.target, a.thing, None])
            else:
                self._request(a, t, p)
        self._focus_step(t, start=True, p=p)
        if self._sounding(t):                               # her voice from this tick, at her face (the log's voice field)
            self._mark(t, "voice", p)
        out.line, out.clip, out.acts = line, clip, acts

    def _open_ask(self, line, it, t, n, word_ends, p):
        """her ask is judged from the tick its word has been heard (4.8): a gaze or act ask from the end of its object's word, a
        name ask from the end of its question, the call (a gaze ask at her face) from the end of the child's name, each over its
        own window (4.6: 20 ticks for a gaze, the call's too; 40 for an act; 20 for a name)."""
        o = p.obj(line.refs[0]) if line.refs else None
        if it.ask == "call":
            word = NAME
        elif it.ask == "name":
            word = o.name if o is not None else line.focus
        else:
            word = o.name
        if word is None:
            return
        ws = [w for w, _te in word_ends]
        idx = len(ws) - 1 if it.ask == "name" or word not in ws else ws.index(word)
        opened = word_ends[idx][1]
        win_end = opened + (K.JUDGE_ACT if it.ask == "act" else K.JUDGE_GAZE)
        tid = self.ledger.ask(t, word, it.ask, None if o is None else o.id, win_end - opened, open_at=opened)
        self.pending = dict(kind=it.ask, word=word, obj=None if o is None else o.id, tick=t, open=opened, open_idx=idx,
                            until=win_end, trial=tid)

    # ------------------------------------------------------------------ save
    def state(self):
        return dict(fast=self.fast.state(), routine=self.routine, pending=self.pending, reply_due=self.reply_due,
                    no_target_since=self.no_target_since, attn=[dict(e, acts=[list(a) for a in e["acts"]]) for e in self.attn],
                    acts_open=[list(a) for a in self.acts_open], focus_acts=[list(a) for a in self.focus_acts],
                    born=self.born, face_until=self.face_until,
                    last_vocal_smile=self.last_vocal_smile, turn=self.turn,
                    sound_hist=list(self.sound_hist), nonstop_since=self.nonstop_since,
                    requests=[[i, dict(k)] for i, k in self.requests], cuts=self.cuts, world=dict(self.world),
                    motion=self.motion.state(), reader=self.reader.state(), imperfect=self.imperfect,
                    imp=self.imp.bit_generator.state, turns=list(self.turns), copy_next=self.copy_next,
                    copies=[list(c) for c in self.copies],
                    ledger=self.ledger.state(), transcriber=None if self.transcriber is None else self.transcriber.state())

    def load_state(self, s):
        self.fast.load_state(s["fast"])
        self.routine, self.pending, self.reply_due = s["routine"], s["pending"], s["reply_due"]
        self.no_target_since = s["no_target_since"]
        pad = lambda a, n: list(a) + [[UNNAMED]] * (n - len(a))              # noqa: E731 (a save before P3's seventh round:
        self.attn = [_old_entry(dict(e, acts=[pad(a, 5) for a in e["acts"]])) for e in s["attn"]]   # its acts' words unknown)
        self.acts_open = [pad(a, 7) for a in s["acts_open"]]
        for a in self.acts_open:                            # (a seventh-round save's words may hold None: at rest)
            a[6] = [w for w in a[6] if w is not None]
        self.born = s.get("born", NEVER)                    # unknown in an older save: every tick before its log unread
        self.face_until = s.get("face_until", NEVER)
        self.focus_acts = [list(a) for a in s["focus_acts"]]
        self.last_vocal_smile, self.turn = s["last_vocal_smile"], s["turn"]
        self.sound_hist, self.nonstop_since = list(s["sound_hist"]), s["nonstop_since"]
        self.requests, self.cuts = [(i, dict(k)) for i, k in s["requests"]], s["cuts"]
        self.world = dict(s["world"])
        self.motion.load_state(s["motion"])
        self.reader.load_state(s["reader"])
        self.imperfect = s["imperfect"]
        self.imp.bit_generator.state = s["imp"]
        self.turns, self.copy_next = list(s["turns"]), s["copy_next"]
        self.copies = [tuple(c) for c in s["copies"]]
        self.ledger.load_state(s["ledger"])
        if self.transcriber is not None and s["transcriber"] is not None:
            self.transcriber.load_state(s["transcriber"])
