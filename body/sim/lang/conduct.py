"""THE PARENT'S FAST LAYER AND HER INTENTS (docs/SIM_DESIGN.md 4.3-4.10, 12, A13, A14, A27, A28, A51; package P3): what she says,
when, in which register, the acts her talk accompanies, and her formal trials of what the child understands.

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
               listens), a formal trial under way (she says only its sentence), a pending ask (judged over its window), joint
               attention (a follow-in variation set naming its target as she reads it), and idle. Episodes (L3: the day plan,
               P4) and Claude's rows (P5) ask for intents through request(); the day plan asks for formal trials through
               probe() (never Claude: a trial's sentence requested as an intent is refused). The judgments she makes by talk
               (a right name, a met ask, a trial's first look on its target, an approximation, stage 1's vocal turn: 4.3's
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
               HER ASKS ARE HER TEACHING, NEVER A MEASURE (the lead's decision after P3's eighth round; 4.8): a met ask may earn
               her smile (4.3's worth table), and the ledger records every ask and its outcome as teaching only: no ask counts
               toward "understood", "says" or any milestone, and a word the child says while one is open (from its line to its
               window's end) or while a formal trial's is, whatever the word, never counts toward "says". In an ask her eyes
               stay on the child and she does not point or show (A51, her method: blind(), eyes_on_child), and a copy of its
               movement due then is not made (a late copy is no copy); never a gaze or act ask about an X in her own hands (a
               look at her would meet it), and "give me the X" only with an X within the child's reach as she sees it
               (Seen.child_can_reach), so no ask is one no act of its could answer.
               UNDERSTANDING IS SCORED ONLY IN FORMAL TRIALS, as infant labs score it (intermodal preferential looking: Golinkoff
               et al. 1987; looking-while-listening: Fernald et al. 2008, Bergelson and Swingley 2012; its name against a foil
               name in the same voice: Mandel, Jusczyk and Pisoni 1995). At a probe (probe(): scheduled by her day plan's stage,
               P4, never by the child's rates) she brings two things of the same kind of test into the child's view side by side
               ("present": W2 places them at matched distance and salience as far as the room allows, and is never told which
               will be named), settles, then holds still: her eyes on the child's eyes, her hands resting on her thighs, her face
               in its neutral set, no act (still; SETTLE = 44 ticks of it before the sentence, so a look that follows her placing
               them has ended: Brooks and Meltzoff 2005's 6.5 s). She says the single test sentence of the design's never-taught
               forms (consts.TRIAL_FORMS: "where is the X?" of a thing in a place or at an angle it has never been seen in, or a
               never-seen exemplar; "where is the C X?", a combination never heard, only once both its words are understood alone,
               A28), with no act: only her mouth moves. Which of the two is named and the sides are drawn by her trial stream
               (TRIAL_STREAM), each a fair coin, so chance is 50%: any child whose first look does not depend on the word, a
               follower of her gaze or her hands, a side's or a toy's favourite, scores 50% trial by trial; nothing she does
               before the sentence depends on the draw (W2's placing, the settle, the display's checks, the line check of both
               things' sentences, whichever is named). The window opens at
               the target word's onset (TRIAL_WINDOW = 20 ticks); the measure is the child's first look (her reading of its head
               line, held 3 ticks, A40) or reach to the target against the distractor: met, missed, or none (no look to either);
               void if it already attended one at the onset, or both at once, or if her attention log shows anything but her
               mouth moving from the sentence's first tick to the decision (a tick not read voids it before that tick's percept
               is read: fail-closed). The name
               test: its name or a foil (NAME_FOILS, stress-matched names she never uses, matched to its name's clip in her voice:
               its ticks, energy, loudest moment and rise), drawn by the same stream, in the same voice and stillness, no things;
               the measure is its turn to her face after its name against after the foil. Her
               judgment follows a trial as any ask's (A28: never withheld, never added): a smile for its first look on the
               target, and for its turn to its name until the name is understood; none for a turn to a foil. A trial counts
               for a word only when its thing is a fresh never-taught item (its first NOVEL_PRESENTATIONS = 3, A28), whichever
               of the two was named, and the ledger's "understood" reads only such trials (ledger.two_part, over its last 20):
               its first look on its thing when named, at least 5 times and above 1/2, and its first look where the word said
               sent it above 1/2, each one-sided p < 0.05; the second is exact for a child whose look does not depend on the
               word, whatever its favourites, drift or learning, since the named thing is a fresh fair coin each trial (the
               name: the second alone, a turn after its name or none after a foil, with at least 5 turns after its name). A
               probe dropped once its two things were brought into view counts as a presentation of them all the same, and her
               trial's line carries no object (Say.line.refs empty): W2 and the world hear its words and are told nothing more.
               HER ATTENTION LOG (A51; kept for the trials' void rule and for P4's probes, which hold her still the same way:
               still_over()). Every tick, before the conduct runs, her motion (W2; StubMotion until W2 merges) reports where her
               body physically points (report()): her trunk ("child" while it faces the child and holds still; a turn, a lean or
               a shift is movement), her eyes', her head's and each hand's target (a thing's id; the child itself,
               "child", "child_eyes" or "child_periphery"; a part of the child, a place, her own face; None at rest: her hands
               resting on her thighs), her face ("mama" on a tick its expression moves), and the status of every act asked of it
               by anyone, exactly "running", "done", "refused" or "cancelled". Fail-closed: any other status, an act left out of
               the report (its status asked of the motion itself, running unless it answers ended), a field left out (her
               face left out: moving), an act in the report she did not ask for (until reported ended), a toy her percept shows
               in her hands, her face within FACE_COURSE ticks of a judgment or frown of hers, and a tick never read all count as
               her body moving (moving()). The log keeps the last SETTLE + 1 ticks, saved.
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

Built here from 4.10: the talk and its timing, her reading of the child (A40), her imperfection and copying (A52), her formal
trials and her attention log (4.8, A51). Left to W2 and P4: L1's gaze and face each tick (W2 reads eyes_on_child and still, and
reports L1's gaze and her face in the log, StubMotion's contract), the "present" act and the world's Seen.near and
Percept.face_near (percept.Reader.near: the two things of a trial each beyond NEAR_DEG of the other and of her face, as the child
sees them, so its first look can be read as one of them), the probes' schedule by her day plan's stage and their items (which
places, angles and exemplars are never-taught, and their presentations outside trials), the scaffolding ladders' acts (4.10: the
give's level 0 is her open hand and the ask, teaching like every ask; its point and touch only once its window has closed unmet),
the routines' order, her spells of distraction (P4's own-tasks episode), and the motor judgments; the acts are requested here and
carried out there.

Nothing here draws a random number but her own streams of the body's seed (her lines' STREAM = 3, her reading's READ_STREAM = 4,
her imperfection's IMPERFECT_STREAM = 5, her trials' TRIAL_STREAM = 6); state() and load_state() carry everything, so a replay is
exact (a saved Conduct restored continues its lines, choices, trials and ledger bit for bit: body/tests/test_sim_lang.py).
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
PROMPT_KEEP = 400                             # her asks' and trials' spans kept this long for a word of the child's read late (a
                                              # turn is read at its end: ours, the trial's TRIAL_WAIT)
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
    "present": "a formal trial's two things set down side by side in the child's view (target 'left_id|right_id', the sides as "
               "the child sees them), at matched distance from its eyes and matched salience as far as the room allows, each "
               "beyond NEAR_DEG of the other and of her face as the child sees them, both hands at once; then her hands back to "
               "rest on her thighs (4.8, 12). W2 is never told which of the two will be named",
}
STUB_FOCUS = 60                               # the stub runs a during='focus' act until the conduct cancels it (its word's end)
STUB_TICKS = {"look": 2, "lean_in": 7, "attend": 20, "show": 7, "point": 5, "open_hand": 5, "hand_over": 12, "touch": 7,
              "withdraw": 2, "offer_bottle": 12, "guide": 8, "pull_to_sit": 20, "wave": 5, "walk": 30, "cover_face": 3,
              "reveal_face": 2, "do": 7, "copy": 7, "present": 12}   # the stub's nominal times, ours; W2 measures its own


@dataclass(frozen=True)
class Act:
    kind: str
    target: str = None                        # an object id, "child", "child_eyes", "child_periphery", a body word, "door", "sofa"
    during: str = None                        # "focus": only during the line's focus word
    thing: str = None                         # a 'do' act's toy (the demonstration's shake or pick-up of it: TP.TOY_ACTS)

    def __post_init__(self):
        assert self.kind in ACT_KINDS, self.kind


# ---------------------------------------------------------------------------------------- her attention log (A51, for W2)
POINTING = ("eyes", "head", "left", "right", "trunk")   # her motion's report every tick: where her eyes, head, each hand
                                              # and her trunk physically point (her trunk "child" while it faces the child and holds
                                              # still; a turn, a lean or a shift anything else: P3's tenth round, the ninth
                                              # verifier's gap)
FIELDS = POINTING + ("face",)                 # and her face ("mama" on a tick its expression moves)
HANDS = ("left", "right")
ENDED = ("done", "refused", "cancelled")      # the only statuses that end an act: any other, or none reported, is running
UNNAMED = "?"                                 # a field her motion left out or gave as no word: it points anywhere (fail-closed)
AT_CHILD = ("child", "child_eyes", "child_periphery")   # the child itself: her eyes or face on its face, her hand held open or
                                                         # waved toward it, touching nothing (a part she touches is a place)
OUTSIDE = "?act"                              # the kind of an act in her motion's report she did not ask for (P4's, W2's own)
PENDING_OK = (("look", "child_eyes"), ("open_hand", "child"), ("lean_in", "child_periphery"), ("withdraw", "child"))
# her lines' acts while an everyday ask is pending (A51, her method: blind()): her eyes on its eyes, her open hand held out to it,
# her face into its periphery, a hand withdrawn when hit; never a point, a show, a turn to a thing, a touch or a demonstration
ATTN_KEEP = K.SETTLE + 1                      # ticks of the log kept: a trial's settle and the tick after it


def _act_str(a):
    """an open act of the log ([id, kind, target, status]) in words."""
    if a[1] == OUTSIDE:
        return f"an act she did not ask for (her motion's id {a[0]!r}, reported {a[3]!r})"
    return f"her {a[1]} at {a[2] or 'nothing'} ({a[3]})"


def moving(e):
    """what moves in a tick of her attention log other than her mouth, or None when only her mouth may (4.8's formal trial, the
    lead's decision: her eyes and head on the child, her hands resting on her thighs, her trunk facing the child and still, her
    face in its neutral set, no act of hers or anyone's under way; A51): the trial's void rule and its settle, and P4's probes'
    (still_over)."""
    bad = [f"her {f} at {e[f]}" for f in ("eyes", "head") if e[f] not in AT_CHILD]
    bad += [f"her {f} hand at {e[f]}" for f in HANDS if e[f] is not None]
    if e.get("trunk", UNNAMED) != "child":
        bad.append(f"her trunk at {e.get('trunk', UNNAMED)} (turning, leaning or shifting)")
    if e["face"] is not None:
        bad.append("her face moving")
    bad += [_act_str(a) for a in e["acts"]]
    return "; ".join(bad) if bad else None


def at_rest(target):
    """a field's target that is still: none, or the child itself (a part of it she touches or looks at is a place)."""
    return target is None or target in AT_CHILD


def directs(act):
    """the attention log's fields an act of hers drives while it runs, and at what: what StubMotion reports of it, and the least
    W2 reports (W2 reports its own truth, L1's gaze with it) -> [(field, target)]. field: "eyes", "head", "hand" (one hand: the
    stub's right, else its left), "both" (both hands), or "left" / "right" (a side); target: a thing's id, the child itself
    ("child", "child_eyes", "child_periphery"), the part of the child she touches or guides, a place word, "mama" (her own
    face), None (at rest), or UNNAMED where the act does not say. Her trunk: "child" while it faces the child and holds still
    (the stub's at rest), the target it turns to, or UNNAMED while it leans or shifts."""
    k, tg = act.kind, act.target
    if k == "look":
        return [("eyes", tg), ("head", tg)]
    if k == "walk":
        return [("eyes", tg), ("head", tg), ("trunk", tg)]
    if k == "lean_in":
        return [("head", "child"), ("trunk", UNNAMED)]                # her trunk leans in (A3)
    if k == "attend":
        return [("head", "child"), ("hand", "trunk"), ("trunk", UNNAMED)]   # she kneels beside it, a hand on its trunk
    if k in ("show", "point", "open_hand", "hand_over", "offer_bottle"):
        return [("hand", tg)]
    if k in ("touch", "guide"):
        return [("hand", tg)]                                         # the part she touches, the limb she guides
    if k == "wave":
        return [("hand", "child")]
    if k == "pull_to_sit":
        return [("both", "arm"), ("trunk", UNNAMED)]                  # by the forearms (A9), her trunk leaning back
    if k == "withdraw":
        return [("hand", None)]
    if k in ("cover_face", "reveal_face"):
        return [("both", PARENT_NAME)]
    if k == "copy":                                                   # her arm raised or her hand shaken points at a place;
        kind, _, side = (tg or "").partition(":")                     # her hand waved or held open, toward the child
        return [(side if side in HANDS else "hand", "child" if kind in ("wave", "open_hand") else UNNAMED)]
    if k == "present":                                                # a trial's two things, one in each hand (4.8)
        left, _, right = (tg or "").partition("|")
        return [("left", left or UNNAMED), ("right", right or UNNAMED)]
    if k == "do":
        if tg in TP.TOY_ACTS:
            return [("hand", act.thing if act.thing is not None else UNNAMED)]
        if tg in ("wave", "open_hand"):
            return [("hand", "child")]
        if tg == "walk":
            return [("eyes", act.thing or UNNAMED), ("head", act.thing or UNNAMED), ("trunk", act.thing or UNNAMED)]
        return [("hand", UNNAMED)]
    return [("hand", UNNAMED)]


class StubMotion:
    """the motion interface W2 implements, and its contract:
      request(act, tick) -> id: the act asked for (W2 carries it out under the caps, or refuses it);
      status(id, tick) -> 'running' | 'done' | 'refused' | 'cancelled';
      cancel(id, tick): the act stopped from that tick (the conduct cancels her look on a naming word as the word ends);
      report(tick) -> dict(eyes, head, left, right, trunk, face, acts): HER ATTENTION LOG'S FIELDS (A51), made once a tick
        BEFORE the conduct runs (the conduct reads it first thing in its tick; a tick whose report it never read counts as her
        body moving): trunk, "child" while her trunk faces the child and holds still, else where it turns (a thing, a place) or
        "?" while it leans or shifts (a torso turn or lean is a movement a look can follow: P3's tenth round); eyes, head, left
        and right, where each physically points (a thing's id; the child itself, "child", "child_eyes" or
        "child_periphery", for her eyes or face on its face or her hand held open or waved toward it, touching nothing; the part
        of the child a hand touches or guides; a place word; "mama" for her own face; or None at rest, her hands resting on her
        thighs), from the tick it leaves the child (or rest) until the tick it is back, L1's own gaze included as it happens (to
        a sudden event, to the child's act or its target), a hand holding a thing at it; face, "mama" on a tick her face's
        expression moves (a smile's or a frown's onset, hold or easing: the face is W2's; her jaw with her speech is her mouth,
        not her face), None while it is still; and acts, {id: status} for every act asked of it by anyone (the conduct, P4, its
        own) that it has not yet reported ended, each status exactly 'running', 'done', 'refused' or 'cancelled'. Anything else
        ('queued', 'waiting', a typo), a field left out (the face left out: moving), an act of hers left out of the report, and
        an act in the report the conduct did not ask for count as her body moving (fail-closed), which voids a formal trial
        (4.8): so W2 reports every act from its request until it reports it ended, keeps 'running' while it waits its turn,
        ends every act, and reads Conduct.eyes_on_child (an ask or a trial: her eyes and head on the child's eyes, L1 turns to
        nothing) and Conduct.still (a trial's settle, sentence and window: also her hands resting on her thighs, her face in its
        neutral set, no act started by W2 or P4) each tick;
      state() / load_state(); DOES, the acts her own body can show a verb by (templates.NEEDS' ("act", kind): the 'do' act's
        targets), which templates.showable() reads.
    The stub runs each act for its nominal time and moves nothing (it never refuses); its fields are what directs() gives for the
    acts running (an act asked for at a tick shows from the next report: the report of that tick was made before it), her eyes
    and head on the child at rest, her face still (the stub has no face: the conduct logs her face from its own judgments). An
    act cancelled from a tick still shows on that tick (her eyes or hand on the way back) and is 'cancelled' from the next, so
    every act shows on at least one tick. glance() stands in for W2's L1 (the world calls it for a sudden event: her eyes and
    head on it from that tick; a gaze for a tick already reported starts at the next report, keeping its length, so the log
    never loses it). Its DOES are the acts W2's parent_acts.py or ACT_KINDS already name (shake and hold: the show; go and walk:
    the walk; the wave; get: pick_up; open: the open hand). W2 declares its own (a clap, a push, ...)."""

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
        f = dict(eyes="child", head="child", left=None, right=None, trunk="child", face=None)
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
    "call": Intent("calling", True, "call", (Act("lean_in", "child_periphery"),)),   # her face into its periphery (A3); her
                                                             # teaching's call: the name test is trial_name's (4.8)
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
    # a formal trial's test sentence (4.8, 12): no act at all, only her mouth moves; judged by the trial, never as an ask
    "trial_where": Intent("plain", True, None, ()),
    "trial_combo": Intent("plain", True, None, ()),
    "trial_name": Intent("calling", True, None, ()),
}
TRIAL_INTENTS = ("trial_where", "trial_combo", "trial_name")
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


def blind(acts):
    """her lines' acts while an everyday ask is pending (A51, her method; Golinkoff et al. 1987): only PENDING_OK's, her head and
    eyes on the child's eyes and her open hand held out to it, her face into its periphery, a hand withdrawn when hit; no point,
    show, turn, touch, guide, wave or demonstration until the ask is judged. (Her asks are her teaching; a formal trial holds her
    wholly still: 4.8.)"""
    out = []
    for a in acts:
        if a.kind == "look":
            b = Act("look", "child_eyes")
        elif a.kind == "open_hand":
            b = Act("open_hand", "child")
        else:
            b = a
        if (b.kind, b.target) in PENDING_OK and b not in out:
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


def _old_entry(e):
    """a tick of her log from an older save (P3's seventh and eighth rounds: each field the words a look that follows it could be
    read as; before, one word) -> this form, fail-closed: a field's lone "child" is the child, an empty one at rest, anything else
    points anywhere; her face logged at all, or left out, moving (her voice was her mouth); each act kept, open."""
    def one(v):
        if isinstance(v, list):
            return None if not v else ("child" if v == ["child"] else UNNAMED)
        return v if v is None or isinstance(v, str) else UNNAMED
    out = dict(t=e["t"])
    for f in POINTING:                                  # (a save before P3's tenth round has no trunk: UNNAMED, moving)
        out[f] = one(e.get(f, UNNAMED))
    face = e.get("face", PARENT_NAME)
    out["face"] = None if face is None or face == [] else PARENT_NAME
    out["acts"] = [list(a[:4]) for a in e.get("acts", ())]
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
    motion: W2's (default StubMotion), whose report of each tick, read first in it, is her attention log (A51); her formal trials
    are asked for by probe() (P4's day plan, by its stage). world: the
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
        self.attn = []                        # her attention log (A51): dict(t, eyes, head, left, right, face, acts) a tick, each
                                              # field its target as reported (UNNAMED where left out or no word), face "mama"
                                              # while it moves, acts [id, kind, target, status] of every act not yet reported
                                              # ended (hers, and OUTSIDE's she did not ask for); the last ATTN_KEEP ticks
        self.acts_open = []                   # acts not yet reported ended: [motion id, kind, target, thing, tick, status];
                                              # OUTSIDE's kind for one she did not ask for
        self.ended = {}                       # the acts her motion reported ended on this tick: {motion id: status}
        self.probes = []                      # formal trials asked for by her day plan (P4): [dict(form, a, b, new, shown)]
        self.trial = None                     # the formal trial under way: dict(form, phase "bring" | "settle" | "said", ...)
        self.trial_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(K.TRIAL_STREAM,))))
        self.face_until = NEVER               # her face moves through this tick (FACE_COURSE after a judgment or a frown: A3)
        self.prompts = []                     # [first tick, last tick] of each ask (its line to its window's end) and trial (its
                                              # sentence to its window's end) of the last PROMPT_KEEP ticks: a word of the child's
                                              # begun or ended within one answers her, teaching only, never "says"
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
        """an everyday ask is pending, or a formal trial holds her still: W2 keeps her head and eyes on the child's eyes, L1 turns
        to nothing (no gaze to its target or to a sudden event), and nothing else starts until it is judged (A51)."""
        return self.pending is not None or self.still

    @property
    def still(self):
        """a formal trial's settle, sentence and window (4.8): W2 holds her wholly still but her mouth, her eyes and head on the
        child's eyes, her hands resting on her thighs, her face in its neutral set, and starts no act (nor does P4); her attention
        log must show it, or the trial is void."""
        return self.trial is not None and self.trial["phase"] in ("settle", "said")

    def set_world(self, world):
        """the world's inventory (plain data, saved with the conduct); its acts are her motion's."""
        w = dict(world)
        self.world = dict(objects={k: sorted(v) for k, v in sorted(dict(w.get("objects", {})).items())},
                          fixtures=sorted(w.get("fixtures", ())), events=sorted(w.get("events", ())),
                          face=sorted(w.get("face", ())), acts=sorted(getattr(self.motion, "DOES", ())))

    # ------------------------------------------------------------------ her attention log (A51)
    def _request(self, act, t, p):
        """an act asked of her motion: kept open (running, whatever it reports but done, refused or cancelled) until her motion
        reports it ended, and in her log from this tick, before her motion's first report of it."""
        mid = self.motion.request(act, t)
        self.acts_open.append([mid, act.kind, act.target, act.thing, int(t), "unreported"])
        if self.attn and self.attn[-1]["t"] == t:
            self.attn[-1]["acts"].append([mid, act.kind, act.target, "asked"])
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

    def _attend(self, t, p):
        """the tick's report from her motion into her attention log (A51): eyes, head and hands as reported (a field left out, or
        not a word, UNNAMED); her face moving where the report says so or leaves it out, or within FACE_COURSE ticks of a
        judgment or frown of hers; each act's status as reported (only done, refused or cancelled end it: anything else is
        running; one left out is asked of the motion, _status), the acts reported ended this tick kept in `ended`; an
        act in the report she did not ask for (P4's, W2's own) kept open, OUTSIDE, until it is reported ended; a toy the percept
        shows in her hands at a hand (her first at rest, else her left) if the report names it at neither."""
        rep = self.motion.report(t)
        got = rep.get("acts") if isinstance(rep.get("acts"), dict) else {}
        known = {a[0] for a in self.acts_open}
        keep, self.ended = [], {}
        for a in self.acts_open:
            st = got[a[0]] if a[0] in got else self._status(a, t)
            if st in ENDED:
                self.ended[a[0]] = st
                continue
            a[5] = st if isinstance(st, str) else "unreported"
            keep.append(a)
        for mid, st in got.items():                          # an act she did not ask for: running until reported ended
            if mid not in known and st not in ENDED:
                keep.append([mid, OUTSIDE, None, None, int(t), st if isinstance(st, str) else "unreported"])
        self.acts_open = keep
        e = dict(t=int(t))
        for fld in POINTING:
            v = rep.get(fld, UNNAMED)
            e[fld] = v if v is None or isinstance(v, str) else UNNAMED
        face = rep.get("face", PARENT_NAME)                 # left out: her face may be moving (fail-closed)
        e["face"] = PARENT_NAME if face is not None or t <= self.face_until else None
        for s_ in p.seen:                                   # a toy she sees in her own hands: a hand is at it, whatever the
            if s_.on != PARENT_NAME:                        # report says (the stub models no hold)
                continue
            if any(v == s_.id or (isinstance(v, str) and p.obj(v) is not None and p.obj(v).name == s_.name)
                   for v in (e["left"], e["right"])):
                continue
            h = next((f for f in HANDS if at_rest(e[f])), "left")
            e[h] = s_.id
        e["acts"] = [[a[0], a[1], a[2], a[5]] for a in keep]
        self.attn = (self.attn + [e])[-ATTN_KEEP:]

    def _status(self, a, t):
        """an act her motion's report left out (hers, or one she did not ask for that it reported before): its status asked of
        the motion itself (status(), the contract's), so an ended act whose one report of its end was never read (a tick the
        world did not tick her) does not hold her unstill for good; running unless it answers done, refused or cancelled, or if
        the motion cannot say (fail-closed)."""
        try:
            st = self.motion.status(a[0], t)
        except Exception:                                   # noqa: BLE001 (a motion that cannot say: running)
            return "unreported"
        return st if st in ENDED else "unreported"

    def _mark(self, t, p):
        """her face moving on tick t, logged by the conduct itself (a judgment's smile, a frown)."""
        if self.attn and self.attn[-1]["t"] == t:
            self.attn[-1]["face"] = PARENT_NAME

    def _face(self, t, out, p):
        """a judgment or a frown this tick: her face moves for FACE_COURSE ticks from it (A3), in her log from this tick."""
        if out.judgments or out.frown:
            self.face_until = max(self.face_until, t + K.FACE_COURSE)
            self._mark(t, p)

    def still_over(self, t0, t1):
        """her attention log shows only her mouth moving on every tick from t0 to t1 (each read, none moving: moving()), or why
        not: the formal trial's rule (4.8), read for P4's probes, which hold her still the same way (12: "her eyes stay on the
        child through it") -> None when still, else why."""
        log = {e["t"]: e for e in self.attn}
        for u in range(int(t0), int(t1) + 1):
            e = log.get(u)
            if e is None:
                return f"tick {u} is not in her attention log (never read, or older than its {ATTN_KEEP} ticks: fail-closed)"
            why = moving(e)
            if why is not None:
                return f"tick {u}: {why}"
        return None

    # ------------------------------------------------------------------ her formal trials (4.8, 12; the lead's decision)
    def probe(self, form, a=None, b=None, new=None, shown=None):
        """a formal trial asked for by her day plan (P4: at a probe its stage schedules, never by the child's rates), run when she
        is free (no ask pending, no trial under way); one that cannot run is dropped and logged (fast.refused), never said.
        form: consts.TRIAL_FORMS. a, b: the two things' ids (none for "name"): for "place" and "exemplar" two things of different
        words, for "combination" a colour twin and its original (one word, two colours, one of them with the word a held-out
        pair, A55). new: {id: item} for each thing shown in a never-taught presentation (a place or an angle it has never been
        seen in, a never-seen exemplar: the world's record, P4's; a combination's item is its pair, taken here). shown: {item:
        its presentations outside trials} (P4's count, beside the ledger's own of trials: A28's first 3 count)."""
        if form not in K.TRIAL_FORMS:
            raise ValueError(f"not a trial form: {form!r} ({', '.join(K.TRIAL_FORMS)})")
        if (form == "name") != (a is None and b is None) or (form != "name" and (a is None or b is None or a == b)):
            raise ValueError(f"a {form!r} trial takes {'no things' if form == 'name' else 'two things'}: {a!r}, {b!r}")
        self.probes.append(dict(form=form, a=a, b=b, new=dict(new or {}), shown=dict(shown or {})))

    def _probe_drop(self, t, form, why, tr=None):
        """a probe that cannot run: dropped and logged, never said. Its two things brought into the child's view (the present
        act asked for and not refused) count as a presentation of their never-taught items all the same (the ledger's
        displayed: fail-closed, for her follow-in naming may label a thing in its new place)."""
        self.fast.refused.append((t, f"trial:{form}", "probe: " + why))
        if tr is not None and tr.get("act") is not None and tr.get("present") != "refused" and form != "combination":
            self.ledger.displayed(t, form, [tr["new"].get(tr["target"]), tr["new"].get(tr["distractor"])], why)

    def _trial_begin(self, t, p):
        """the next probe begins: her trial stream draws which of the two is named, the sides and (for the name) its name or a
        foil and which foil, three draws a trial whatever its form, so a replay is exact; she brings the two things into its
        view side by side ("present", W2 never told which will be named), or for the name holds still where she is."""
        pr = self.probes.pop(0)
        form, a, b = pr["form"], pr["a"], pr["b"]
        u = [float(x) for x in self.trial_rng.random(3)]
        tr = dict(form=form, phase="bring", since=int(t), run=0, act=None, tid=None, onset=None, until=None, line_start=None,
                  word_idx=None, why=None, new=pr["new"], shown=pr["shown"])
        if form == "name":
            name = u[0] < 0.5
            foil = K.NAME_FOILS[min(len(K.NAME_FOILS) - 1, int(u[2] * len(K.NAME_FOILS)))]
            tr.update(phase="settle", name=bool(name), said=NAME if name else foil)
            self.trial = tr
            return
        oa, ob = p.obj(a), p.obj(b)
        if oa is None or ob is None:
            return self._probe_drop(t, form, f"she does not see {a if oa is None else b!r}")
        if form == "combination":
            if oa.name != ob.name or oa.colour == ob.colour or not oa.colour or not ob.colour:
                return self._probe_drop(t, form, f"a combination needs one word in two colours: {oa.id}, {ob.id}")
            twins = [o for o in (oa, ob) if (o.colour, o.name) in {tuple(h) for h in K.HELD_PAIRS}]
            if len(twins) != 1:
                return self._probe_drop(t, form, f"a combination needs exactly one held-out pair between them (A55)")
            col = twins[0].colour
            if not (self.ledger.understood(col) and self.ledger.understood(oa.name)):
                return self._probe_drop(t, form, f"'{col} {oa.name}' is tested only once both its words are understood "
                                                 f"alone (A28)")
            tr["twin"] = twins[0].id
        elif oa.name == ob.name or oa.name not in TP.OBJECT_NOUNS or ob.name not in TP.OBJECT_NOUNS:
            return self._probe_drop(t, form, f"a {form} trial needs two things of different words: {oa.id}, {ob.id}")
        target, other = (a, b) if u[0] < 0.5 else (b, a)
        left, right = (a, b) if u[1] < 0.5 else (b, a)
        tr.update(target=target, distractor=other, left=left, right=right)
        tr["act"] = self._request(Act("present", f"{left}|{right}"), t, p)
        self.trial = tr

    def _key(self, tr, o):
        """the word a trial's thing stands for in the ledger: its word, or for a combination its colour and word ('blue ball')."""
        return f"{o.colour} {o.name}" if tr["form"] == "combination" else o.name

    def _display(self, tr, p):
        """why its sentence cannot be said now, or None: the two things in the child's view, in no hand, each beyond NEAR_DEG of
        the other and of her face as the child sees them (the world's Seen.near and Percept.face_near; not given: fail-closed),
        so its first look can be read as one of them; the child attending neither; for the name, the child able to see her and
        not already looking at her face."""
        if not (p.present and p.child_in_view):
            return "she is away, or cannot see the child"
        if tr["form"] == "name":
            if not p.seen_by_child:
                return "the child cannot see her"
            return "it already looks at her face" if p.child_target == PARENT_NAME else None
        ot, od = p.obj(tr["target"]), p.obj(tr["distractor"])
        for oid, o in ((tr["target"], ot), (tr["distractor"], od)):
            if o is None:
                return f"she does not see {oid!r}"
            if not o.child_sees:
                return f"{oid!r} is not in the child's view"
            if o.on in ("hand", PARENT_NAME) or oid in p.child_holds:
                return f"{oid!r} is in a hand"
        if ot.near is None or od.near is None or getattr(p, "face_near", None) is None:
            return "the world does not say what lies near them (fail-closed)"
        if od.id in ot.near or ot.id in od.near:
            return f"the two within {K.NEAR_DEG:g} degrees of each other as the child sees them"
        if ot.id in p.face_near or od.id in p.face_near:
            return f"one of them within {K.NEAR_DEG:g} degrees of her face as the child sees it"
        att = {o.id for o in p.attended()}
        if ot.id in att or od.id in att:
            return "it already attends one of them"
        return None

    def _trial_line(self, t, p):
        """her trial's single test sentence once its settle has held SETTLE ticks and its display holds, or None: "where is the
        X?" (the target's word), "where is the C X?" (a combination: the held-out pair allowed in this line alone), its name or
        a foil ("pip." / "viv.", in its name's register), each with no act. The sentence of either thing failing the line check
        (a word not hers, a new word off its peak) drops the probe whichever is named: a drop that followed the draw would name
        the sayable thing every time it ran, and a child's favourite among the two would score above chance."""
        tr = self.trial
        if tr is None or tr["phase"] != "settle" or tr["run"] < K.SETTLE:
            return None
        why = self._display(tr, p)
        if why is not None:
            tr["why"] = why
            return None
        f = self.fast
        if tr["form"] == "name":
            if tr["name"]:
                return TP.Line(f"{NAME}.", "trial_name", "calling", NAME, NAME, (), "fast", "{n}.")
            return TP.Line(TP.foil_line(tr["said"]), "trial_name", "calling", None, tr["said"], (), "fast", "{f}.")
        intent = "trial_combo" if tr["form"] == "combination" else "trial_where"
        lines = {}
        for oid in (tr["target"], tr["distractor"]):        # both sentences held to the check, whichever is named, so a
            o = p.obj(oid)                                  # probe is never dropped by which of the two her stream drew
            text, focus, refs = TP.fill(TP.FRAMES[intent][0], o=o)
            twin = p.obj(tr.get("twin")) if tr.get("twin") else None
            held = tuple(h for h in f.held if twin is None or tuple(h) != (twin.colour, twin.name))
            ok, why = TP.check(text, f.vocab, f.new_words, p, refs, held, recent_events=f.recent)
            if not ok:
                self.trial = None
                self._probe_drop(t, tr["form"], f"the sentence {text!r} (of either thing) fails the line check: {why}", tr)
                return None
            lines[oid] = TP.Line(text, intent, register_for(intent, text), focus, focus, (), "fast", TP.FRAMES[intent][0][0])
        return lines[tr["target"]]                          # (no refs: the line leaves her as Say.line, and neither W2 nor
                                                            # the world is ever told which thing is named; its words are)

    def _trial_said(self, line, t, spans, p, out):
        """her test sentence begins at t: its window opens at the target word's onset (the word that tells the two apart: the
        noun, a combination's colour, its name or the foil) and runs TRIAL_WINDOW ticks; the ledger records the trial, what it
        scores (the named word's test when its thing is a fresh never-taught item, A28; the distractor's yoked rate when its
        thing is; for the name, its test or its foil rate) and the items it presents."""
        tr = self.trial
        if tr["form"] == "name":
            word_at, score = tr["said"], [[NAME, "trials" if tr["name"] else "yoked", 1, 0]]
            target, distractor, items, sides = dict(id=None, word=tr["said"]), None, [], None
        else:
            ot, od = p.obj(tr["target"]), p.obj(tr["distractor"])
            word_at = ot.colour if tr["form"] == "combination" else ot.name
            kt, kd = self._key(tr, ot), self._key(tr, od)
            if tr["form"] == "combination":
                pair = f"{p.obj(tr['twin']).colour} {p.obj(tr['twin']).name}"
                it_t = pair if tr["target"] == tr["twin"] else None
                it_d = pair if tr["distractor"] == tr["twin"] else None
                items = [pair] if it_t else []                 # a combination is presented when it is said
            else:
                it_t, it_d = tr["new"].get(tr["target"]), tr["new"].get(tr["distractor"])
                items = [x for x in (it_t, it_d) if x]         # a place, an angle, an exemplar: when it is displayed
            fresh = lambda it: it is not None and self.ledger.presented(it) + int(tr["shown"].get(it, 0)) < \
                K.NOVEL_PRESENTATIONS                                                          # noqa: E731
            score = ([[kt, "trials", 1, 0]] if fresh(it_t) else []) + ([[kd, "yoked", 0, 1]] if fresh(it_d) else [])
            target, distractor = dict(id=ot.id, word=kt), dict(id=od.id, word=kd)
            sides = (tr["left"], tr["right"])
        ws = [w for w, _a, _e in spans]
        idx = ws.index(word_at) if word_at in ws else len(ws) - 1
        onset = int(t) + spans[idx][1] // TICK
        tr.update(phase="said", line_start=int(t), word_idx=idx, onset=onset, until=onset + K.TRIAL_WINDOW, word=word_at)
        tr["tid"] = self.ledger.trial(t, tr["form"], target, distractor, onset, K.TRIAL_WINDOW, tr["said"] if tr["form"] ==
                                      "name" else word_at, sides=sides, score=score, items=items)
        self._prompt(t, t, onset + K.TRIAL_WINDOW)
        if onset == t:                                       # the name: its onset is the line's first tick
            res, why = self._trial_judge(tr, t, p)
            if res is not None:
                self._trial_end(t, res, why, out, p)

    def _trial_judge(self, tr, t, p):
        """a tick of its window judged by her percept (A40: her reading of its head line, held 3 ticks, and its hands) -> (result,
        why) or (None, None). At the onset: void if it already attends either thing (for the name, already looks at her face),
        or cannot see them (her); after it, the first look or reach on one of them decides it (on both at once: void); for the
        name, its turn to her face."""
        if tr["form"] == "name":
            if not (p.present and p.seen_by_child):
                return "void", "the child cannot see her"
            looking = p.child_target == PARENT_NAME
            if t <= tr["onset"]:
                return ("void", "it already looked at her face at the name's onset") if looking else (None, None)
            return ("met", "its turn to her face") if looking else (None, None)
        att = {o.id for o in p.attended()}
        ht, hd = tr["target"] in att, tr["distractor"] in att
        if t <= tr["onset"]:
            if ht or hd:
                return "void", "it already attended one of them at the word's onset"
            if not all(p.obj(x) is not None and p.obj(x).child_sees for x in (tr["target"], tr["distractor"])):
                return "void", "the two not both in its view at the word's onset"
            return None, None
        if ht and hd:
            return "void", "its first look or reach on both at once"
        if ht:
            return "met", "its first look or reach on the target"
        if hd:
            return "missed", "its first look or reach on the distractor"
        return None, None

    def _trial_tick(self, t, p, out):
        """her trial under way, each tick: the present act until her motion reports it done (refused or cancelled: dropped);
        the settle, counting ticks her log shows still (moving(): only her mouth may move; a tick not read restarts it); once its
        sentence is said, each tick from the onset judged by the percept first (a look a tick's percept shows was made before
        that tick, once the tick before was read: a tick not read voids it first), then the tick's log held to it: anything but
        her mouth moving voids it; its window's end with no look: none (the name: missed, no turn). The child's pain or distress,
        or its hit, ends it (void once said)."""
        tr = self.trial
        if tr is None:
            return
        ev = {k for k, _o in p.events}
        if ev & {"pain", "distress", "hit_her"}:
            return self._trial_end(t, "void", "she answered its pain, distress or hit (4.10's priority)", out, p)
        e = self.attn[-1] if self.attn and self.attn[-1]["t"] == t else None
        read = e is not None and len(self.attn) >= 2 and self.attn[-2]["t"] == t - 1
        if tr["phase"] == "bring":
            st = self.ended.get(tr["act"])
            if st is None:
                if t - tr["since"] > K.TRIAL_WAIT:
                    self._trial_end(t, "void", f"the two not placed within {K.TRIAL_WAIT} ticks", out, p)
                return
            tr["present"] = st
            if st != "done":
                return self._trial_end(t, "void", f"her motion reported the present act {st}", out, p)
            tr["phase"] = "settle"
        if tr["phase"] == "settle":
            still = e is not None and moving(e) is None
            tr["run"] = (tr["run"] + 1 if read else 1) if still else 0
            if t - tr["since"] > K.TRIAL_WAIT:
                self._trial_end(t, "void", f"not said within {K.TRIAL_WAIT} ticks ({tr['why'] or 'her body not still'})",
                                out, p)
            return
        if t > tr["line_start"]:                            # the sentence's first tick was held as the settle's last
            if not read:                                    # the tick before never read: her body then unknown, and a look
                return self._trial_end(t, "void", "her attention log: a tick not read (fail-closed; only her mouth may "
                                                  "move)", out, p)      # this tick shows may follow it
            if t >= tr["onset"]:
                res, why = self._trial_judge(tr, t, p)
                if res is not None:
                    return self._trial_end(t, res, why, out, p)
            why = moving(e)
            if why is not None:
                return self._trial_end(t, "void", f"her attention log: {why} (only her mouth may move)", out, p)
            if t >= tr["until"]:
                return self._trial_end(t, "missed" if tr["form"] == "name" else "none",
                                       "no turn to her face in its window" if tr["form"] == "name" else
                                       "no look or reach on either in its window", out, p)

    def _trial_end(self, t, res, why, out, p):
        """the trial ends: never said, dropped and logged; said, its outcome in the ledger, and her judgment as any ask's (A28:
        never withheld, never added): a smile for the first look on the target, for its turn to its name until the name is
        understood, with her confirm after it; none for a turn to a foil (it is not its name), a miss, none or a void."""
        tr, self.trial = self.trial, None
        if tr["tid"] is None:
            return self._probe_drop(t, tr["form"], why, tr)
        self.ledger.trial_outcome(t, tr["tid"], res, why)
        if res != "met":
            return
        if tr["form"] == "name":
            if tr["name"] and not self.ledger.understood(NAME):
                out.judgments.append((K.WORTH_MET_ASK, "met_trial", NAME))
                self.reply_due = dict(tick=t, kind="confirm", word=NAME, obj=None)
            return
        out.judgments.append((K.WORTH_MET_ASK, "met_trial", tr["word"] if tr["form"] != "combination" else
                              p.obj(tr["target"]).name if p.obj(tr["target"]) is not None else tr["word"]))
        self.reply_due = dict(tick=t, kind="confirm", word=None, obj=tr["target"])

    def _prompt(self, t, t0, t1):
        """an ask or a trial of hers spans ticks t0..t1 (its line's first tick to its window's last): kept PROMPT_KEEP ticks."""
        self.prompts = [x for x in self.prompts if x[1] >= t - PROMPT_KEEP] + [[int(t0), int(t1)]]

    def _teaching(self, cw):
        """a word of the child's begun or ended while an ask of hers is open (from its line to its window's end, even once it is
        judged) or while a trial of hers is (from its sentence to its window's end), whatever the word: teaching only, never
        toward "says" (the lead's decision: a word said in answer to her, right or wrong, "mama" on answering her call, another
        word to "what is this?", the distractor's word in a trial's window, is prompted)."""
        if cw.word is None:
            return False
        a, b = int(cw.start), int(cw.tick)
        tr = self.trial
        return self.pending is not None or (tr is not None and tr["phase"] == "said") or \
            any(x[0] <= b and a <= x[1] for x in self.prompts)

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
        copying's gap allows, is copied mirrored within 1-2 s; never while an ask is pending or a trial is under way (her hands
        stay still, A51, 4.8): a copy due then is dropped, not made late (a copy after it would come 1-2 s past the movement,
        no copy of it)."""
        if self.imperfect and p.present and p.child_in_view:
            for kind, side in p.events:
                if kind in K.COPY_KINDS and t >= self.copy_next:
                    mirror = {"left": "right", "right": "left"}.get(side, side)
                    self.copies.append((t + int(self.imp.integers(K.COPY_DELAY[0], K.COPY_DELAY[1] + 1)), kind, mirror))
                    self.copy_next = t + max(1, int(round(self.imp.exponential(K.COPY_GAP_S / (TICK / SR)))))
        due = [c for c in self.copies if c[0] <= t]
        self.copies = [c for c in self.copies if c[0] > t]
        if self.eyes_on_child or self.trial is not None:
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
                self.ledger.accepted(cw, p, teaching=self._teaching(cw))
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
                tr = self.trial                             # her trial's sentence stopped before its word: void
                if tr is not None and tr["phase"] == "said" and tr["line_start"] == f.spans["start"] and \
                        tr["word_idx"] >= kept:
                    self._trial_end(t, "void", "its sentence cut before its word was said", out, p)
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
        self._face(t, out, p)                               # her face will move (a smile, a frown): in her log from now
        self._trial_tick(t, p, out)                         # her formal trial: judged by the percept, then held to her log
        self._face(t, out, p)                               # (a trial's own smile, once it is decided)
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
        # her formal trial under way (4.8): she says nothing but its test sentence, once settled; the reply she owes, the meal's
        # first line and everything below wait for its end (its pain, distress or hit, above, end it first)
        if self.trial is not None:
            ln = self._trial_line(t, p)
            return (ln, False, None) if ln is not None else None
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
        # 8. the episode's act: a formal trial her day plan asked for (4.8: she brings its things, or holds still for the name),
        #    an intent asked for by L3 or Claude, and Claude's lines for the moment
        if self.probes:
            self._trial_begin(t, p)
            if self.trial is not None:
                return None
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
        if intent in TRIAL_INTENTS:                         # said only in its trial (4.8), never requested (Claude's rows too)
            return self._drop(t, intent, "a formal trial's sentence is said only in its trial (probe(): her day plan's, by "
                                         "its stage; 4.8)")
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
                                         f"'what is this?' can mean only that (A51)")
        if it.ask == "call" and p.child_target == "mama":
            return self._drop(t, intent, "she reads the child already looking at her face: nothing to call it to")
        if it.ask is not None and self.pending is not None:
            return self._drop(t, intent, "an ask is already pending")
        ln = f.compose(intent, t, p, o=o, b=kw.get("b"), w=kw.get("w"))
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
        if self.pending is not None:                        # an ask she is judging: her eyes on the child, no point or show
            acts = blind(acts)                              # (A51, her method)
        for a in acts:
            if a.during == "focus":                         # on the naming word only (4.3): requested as it begins
                fw = [(wa, we) for w, wa, we in spans if w == line.focus] or [(wa, we) for _w, wa, we in spans[-1:]]
                wa, we = fw[-1]
                self.focus_acts.append([t + wa // TICK, t + (max(we, 1) - 1) // TICK, a.kind, a.target, a.thing, None])
            else:
                self._request(a, t, p)
        self._focus_step(t, start=True, p=p)
        out.line, out.clip, out.acts = line, clip, acts
        if line.intent in TRIAL_INTENTS and self.trial is not None:
            self._trial_said(line, t, spans, p, out)        # her formal trial's sentence: its window from its word's onset

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
        self._prompt(t, t, win_end)
        self.pending = dict(kind=it.ask, word=word, obj=None if o is None else o.id, tick=t, open=opened, open_idx=idx,
                            until=win_end, trial=tid)

    # ------------------------------------------------------------------ save
    def state(self):
        return dict(fast=self.fast.state(), routine=self.routine, pending=self.pending, reply_due=self.reply_due,
                    no_target_since=self.no_target_since, attn=[dict(e, acts=[list(a) for a in e["acts"]]) for e in self.attn],
                    acts_open=[list(a) for a in self.acts_open], focus_acts=[list(a) for a in self.focus_acts],
                    face_until=self.face_until, probes=[dict(x) for x in self.probes], prompts=[list(x) for x in self.prompts],
                    trial=None if self.trial is None else dict(self.trial), trial_rng=self.trial_rng.bit_generator.state,
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
        self.attn = [_old_entry(e) for e in s["attn"]]     # (an older save's log: fail-closed, _old_entry)
        self.acts_open = [list(a)[:6] for a in s["acts_open"]]
        self.ended = {}
        self.face_until = s.get("face_until", NEVER)
        self.probes = [dict(x) for x in s.get("probes", ())]
        self.prompts = [list(x) for x in s.get("prompts", ())]
        self.trial = None if s.get("trial") is None else dict(s["trial"])
        if "trial_rng" in s:
            self.trial_rng.bit_generator.state = s["trial_rng"]
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
