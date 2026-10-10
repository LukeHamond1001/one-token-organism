"""THE PARENT'S SENTENCES AND THE LINE CHECK (docs/SIM_DESIGN.md 4.4, 4.5, 4.8, A14, A15, A28; package P3).

FRAMES are her templates by intent (4.5's table), in infant-directed speech: at most 6 words, the focus word last (Fernald and
Mazzie 1991), the focus word said on a pitch peak and lengthened (the voice's emphasis, 4.4). A frame's slots are filled only from
what she perceives (body/sim/lang/percept.py):
  {o}  an object she sees (the child's target, what it holds, what she shows): its word
  {c}  that object's colour as she sees it (a colour frame is used only once the colour word is hers)
  {p}  what that object rests on as she sees it (a fixture word she sees)
  {b}  a part of the child's body (the child in her view)
  {n}  the child's name
  {w}  a word given by the moment: the word her ear accepted (a recast, an echo), the day's new word
Each frame names its focus: a slot, a literal word, or None (a line with no focus: "mama is here."). Wherever a line has a focus,
it is the line's last word (checked over every frame at import).

THE LINE CHECK (check()) holds every line, the templates' and Claude's alike, to the form and to what she perceives:
  form        lowercase words, single spaces, only ". ? !" and only after a word, ending in one; at most 6 words (4.5)
  words       only her vocabulary, plus the day's new word, and the new word, if said, last (4.5, A15), in a line measured to
              put it on the line's pitch peak (A34: PEAK, the table tools/sim_voice_check.py --peak writes)
  held out    never a never-taught pair before its test: both its words in one line, or its colour said of an object of its
              name (A28, A55: "it is blue." said of the blue ball is the pair, whatever the words)
  perceived   every object word names an object she sees (and each filled slot's object is one she sees); a colour word, an object
              of that colour she sees (of the named kind, when the line names one); a fixture word, a fixture she sees; the
              child's body, the child in her view; a past form ("fell", "rolled", "sat", "got", "did"), an event she saw in the
              last RECENT ticks. Only what a person in her place could see or hear.
  true        what the line says of a thing holds as she sees it (her grammar is small, so its few claims are checked, not only
              that the things exist): "it is a X." / "this is the X." / "it is your X." / "it is C." name or colour what the line
              is about (its filled object) or, for a line with none (Claude's), what the child looks at or holds; "the X is on
              the Y." an X resting on the Y ("in the Y": containment she cannot see, refused); "the X is down." / "it is down." an
              X she saw fall in the last RECENT ticks; "you see the X." an X in the child's view
  Claude's    (A14, claude_claims) no praise ("yes", "good"), no reprimand ("no": stage 2's frown line) and no approval
              register: a judgment is the fast layer's; no ask anywhere in the line ("where", "what", "give", "more",
              "look" with a word after it in its sentence, or the child's name, which is the call): an ask is judged
              (4.8's test, the worth table's met ask), so Claude asks for one through the conduct's request(), never by a
              line; and each sentence one of the forms whose claim the check can hold true as she sees it (CLAUDE_FORMS: a
              name; "oh", "uh", "look", "see" alone; "see the X" / "you see the X", an X in the child's view; "it / this /
              that / here / there is a X" or "is C", what the child looks at or holds; "the X is on the Y" / "is down" / "is
              C"; "the X fell"; "mama is here", she is present; "you roll" / "you sit", its roll or sit seen in the last
              RECENT ticks). A sentence of any other form ("the ball is here.", "it is a mat.", "sit.") claims what she
              cannot check, and is refused. The fast layer's own lines come from its frames, each said with the act or in the
              moment that makes it true (a body word with her touch on it), and are held to the rules above them.
The design's form rules (4.5) are the first three; "perceived" and "true" are the fast layer's rule that its lines are "filled
from what the parent can see" (4.5), made a check so Claude's lines are held to it too.

THE GROWTH QUEUE (4.8; GROWTH): the design's 76 words in its order, plus "ring" and "stacker" after "rattle" (4.8: "the world's
rattle, stacker and ring are named through the queue", though its list of 76 lacks the two). Each word has a class and the frames
of its introduction: a variation set of 3 lines, the word last (4.8), in the new-word register (4.4). A word enters only when its
referent or act is in the world and she can show it (A15): NEEDS names what that takes, showable() tests it against the world's
inventory (ROOM_AT_BIRTH until the world passes its own; the acts are her motion's), and show_now() against the moment (what she
perceives, and what she saw in the last RECENT ticks); the conduct asks both before it introduces a word, and logs why a word
waits. Waiting: a word whose need is absent from the world (a "bath", a "book", "toes" on a G1 whose feet have none, "hot",
"sing"; a verb whose act her motion cannot do); "happy" and "sad" (her face smiles only at a judged act and shows concern only at
the child's pain, 4.3: she cannot show either at will); the social words with no referent or act ("all", "done", "wow", "thanks",
...), whose moment is the day plan's (P4). Words that no infant-directed line can end on ("and", "to", "with", "my", "i", "has",
"not", "do") have no introduction frames and cannot enter by 4.8's rule (sentence-final); they wait for a later stage's method
(reported, never forced).
"""
import json
import os
import re
from dataclasses import dataclass

from . import consts as K
from .lexicon import BIRTH_WORDS, GROUPS, NAME, PARENT_NAME
from .percept import EVENT_KINDS

# ------------------------------------------------------------------------------------------------ the words by kind
TOYS = dict(GROUPS)["toys"]                                           # ball duck block cup bear car drum bottle (no bottle in the room: A88)
BODY = dict(GROUPS)["body"]                                           # hand foot head tummy
ROOM = dict(GROUPS)["room"]                                           # mat sofa window

# the growth queue (4.8): (word, class); the order is the design's (first-words frequency), "ring" and "stacker" named after
# "rattle" as the world's toys
GROWTH = (
    ("rattle", "toy"), ("ring", "toy"), ("stacker", "toy"), ("book", "toy"), ("red", "colour"), ("blue", "colour"),
    ("yellow", "colour"), ("big", "adj"), ("little", "adj"), ("push", "verb"), ("drop", "verb"), ("shake", "verb"),
    ("stand", "verb"), ("eyes", "face"), ("mouth", "face"), ("nose", "face"), ("clap", "verb"), ("wave", "verb"),
    ("sleep", "verb"), ("go", "verb"), ("get", "verb"), ("hold", "verb"), ("want", "verb"), ("all", "social"),
    ("done", "social"), ("and", "frame"), ("my", "frame"), ("i", "frame"), ("that", "social"), ("to", "frame"),
    ("can", "social"), ("not", "frame"), ("table", "fixture"), ("shelf", "fixture"), ("door", "fixture"), ("light", "fixture"),
    ("box", "toy"), ("bucket", "toy"), ("soft", "adj"), ("loud", "adj"), ("wow", "social"), ("thanks", "social"), ("please", "social"),
    ("sat", "past"), ("rolled", "past"), ("fell", "past"), ("got", "past"), ("now", "social"), ("again", "social"),
    ("too", "social"), ("out", "social"), ("off", "social"), ("with", "frame"), ("has", "frame"), ("do", "frame"),
    ("did", "past"), ("walk", "verb"), ("hug", "verb"), ("toes", "body"), ("arm", "body"), ("leg", "body"), ("one", "social"),
    ("two", "social"), ("green", "colour"), ("under", "social"), ("open", "verb"), ("close", "verb"), ("hot", "adj"),
    ("kiss", "verb"), ("bath", "fixture"), ("floor", "fixture"), ("tower", "toy"), ("fast", "adj"), ("slow", "adj"),
    ("hear", "verb"), ("sing", "verb"), ("song", "social"), ("happy", "adj"), ("sad", "adj"),
)
GROWTH_WORDS = tuple(w for w, _ in GROWTH)
GROWTH_CLASS = dict(GROWTH)
OBJECT_NOUNS = frozenset(TOYS) | {w for w, c in GROWTH if c == "toy"}
FIXTURE_NOUNS = frozenset(ROOM) | {w for w, c in GROWTH if c == "fixture"}
CHILD_BODY = frozenset(BODY) | {w for w, c in GROWTH if c == "body"}
HER_FACE = frozenset(w for w, c in GROWTH if c == "face")
COLOURS = frozenset(w for w, c in GROWTH if c == "colour")
OPEN_CONTAINERS = frozenset({"bucket"})   # A129: what she can see into; a toy lying in it is seen "in the bucket" (the percept's `on` is "bucket")
NOUNS = OBJECT_NOUNS | FIXTURE_NOUNS | CHILD_BODY          # a recast or an echo says "the X" of these ("ball! the ball!")
FUNCTION = frozenset(dict(GROUPS)["function"])             # never a focus word: her ear never expects them
PAST_EVENTS = {"sat": ("sat",), "rolled": ("rolled",), "fell": ("fell",), "got": ("got",),
               "did": ("rolled", "sat", "got", "gave")}

# what each growth word needs in the world to be shown (A15): ("obj", name) an object of that name; ("fixture", name); ("twins",
# colour) two toys of that colour (B2); ("act", kind) an act her motion can do; ("event", kind) an event the world makes;
# ("face", kind) her own face; ("child",) the child's body; ("never", why)
NEEDS = {
    "rattle": ("obj", "rattle"), "ring": ("obj", "ring"), "stacker": ("obj", "stacker"), "book": ("obj", "book"),
    "box": ("obj", "box"), "bucket": ("obj", "bucket"), "tower": ("obj", "tower"),
    "red": ("twins", "red"), "blue": ("twins", "blue"), "yellow": ("twins", "yellow"), "green": ("twins", "green"),
    "big": ("never", "no two toys of one kind in two sizes"), "little": ("never", "no two toys of one kind in two sizes"),
    "soft": ("obj", "bear"), "loud": ("event", "drum_hit"), "hot": ("never", "nothing hot in the room"),
    "fast": ("event", "car_moving"), "slow": ("event", "car_moving"), "happy": ("face", "smile"), "sad": ("face", "concern"),
    "push": ("act", "push"), "drop": ("event", "fell"), "shake": ("act", "show"), "stand": ("act", "stand"),
    "clap": ("act", "clap"), "wave": ("act", "wave"), "sleep": ("act", "sleep"), "go": ("act", "walk"), "get": ("act", "pick_up"),
    "hold": ("act", "show"), "want": ("never", "no act shows wanting"), "walk": ("act", "walk"), "hug": ("act", "hug"),
    "kiss": ("act", "kiss"), "open": ("act", "open_hand"), "close": ("act", "close_hand"), "hear": ("event", "sound"),
    "sing": ("never", "her voice does not sing"), "song": ("never", "her voice does not sing"),
    "eyes": ("face", "any"), "mouth": ("face", "any"), "nose": ("face", "any"),
    "toes": ("never", "the G1's feet have no toes"), "arm": ("child",), "leg": ("child",),
    "table": ("fixture", "table"), "shelf": ("fixture", "shelf"), "door": ("fixture", "door"), "light": ("fixture", "light"),
    "floor": ("fixture", "floor"), "bath": ("fixture", "bath"),
    "sat": ("event", "sat"), "rolled": ("event", "rolled"), "fell": ("event", "fell"), "got": ("event", "got"),
    "did": ("event", "rolled"),
}
NEEDS.update({                       # her face shows these only as her feelings make it (4.3), never at will
    "happy": ("never", "her face smiles only at a judged act (4.3's one law): she cannot show 'happy' at will"),
    "sad": ("never", "her face shows concern only at the child's pain or distress (4.3): she cannot show 'sad' at will"),
})
NEEDS.update({w: ("moment",) for w, c in GROWTH if c == "social" and w not in NEEDS})   # no referent or act: a moment (P4's)
TOY_ACTS = ("show", "pick_up")       # her acts that handle a toy: a verb shown by one ("shake", "hold", "get") is shown on a toy
                                     # she sees, named to her motion (the 'do' act's thing), so her attention log names it (A51)
assert all(w in NEEDS for w, c in GROWTH if c != "frame"), [w for w, c in GROWTH if c != "frame" and w not in NEEDS]

# the living room of 5.1-5.2 at birth, as the world's inventory for showable() (A15): its toys and their colours (the colour
# twins come later, before the colour words: B2, the world adds them), its fixtures (the mat, the sofa, the window, the low
# table, the cube shelves, the doorway, the floor lamp, the oak floor), the events it makes (percept.EVENT_KINDS) and her face.
# The world (W2-W3) passes its own from its scene; the acts are her motion's (StubMotion.DOES until W2).
ROOM_AT_BIRTH = dict(
    objects={"ball": ["red"], "block": ["blue"], "duck": ["yellow"], "cup": ["green"], "rattle": ["purple"], "car": ["orange"],
             "bear": ["brown"], "stacker": ["white"], "drum": ["cyan"], "ring": ["pink"],
             "book": ["black"],                # A115: the first novel toy (in the room only when the world adds it: extras.add_book);
                                               # black: no colour word of hers comes in with it
             "box": ["grey"],                  # A121: the second (extras.add_box); grey: no colour word of hers, no other kind of it
             "bucket": ["olive"]},                # A126: the third, hollow (extras.add_bucket); olive: no colour word of hers, no other kind of it
    fixtures=["mat", "sofa", "window", "table", "shelf", "door", "light", "floor"],
    events=list(EVENT_KINDS), face=["any"], acts=[])

# -------------------------------------------------------------------------------------------------------- the frames
F = lambda text, focus=None: (text, focus)   # noqa: E731
FRAMES = {
    "call": [F("{n}.", "{n}"), F("{n}. look at mama.", "mama"), F("{n}. look here.")],   # her teaching's call (4.5); the name
                                                        # test is trial_name's (its name or a foil: 4.8, 12)
    "hall_call": [F("{n}.", "{n}"), F("{n}? mama is here.")],
    "come_call": [F("mama is here. {n}.", "{n}"), F("here. here. {n}.", "{n}"), F("look. mama here. {n}.", "{n}")],   # C351: called from across the room (the focus word last).
                                                        # C351 amended twice (2026-10-10, 02:10): the first frames ('come here, {n}.', '{n}. come to mama.') never passed
                                                        # her check: a comma fails the form and 'come' is no word of hers (birth or growth), so no call was ever voiced
                                                        # (the copy's refusals: 19 of 19); her words only, no commas
    "greet": [F("hi {n}.", "{n}"), F("hi. hi {n}.", "{n}"), F("hi {n}. mama is here.")],
    "return": [F("hi {n}! mama is here."), F("hi {n}. hi.", "hi")],
    "answer_bid": [F("mama is here.")],
    "label": [F("the {o}.", "{o}"), F("a {o}.", "{o}"), F("it is a {o}.", "{o}"), F("you see the {o}.", "{o}"),
              F("look. the {o}.", "{o}"), F("this is a {o}.", "{o}"), F("here is the {o}.", "{o}"), F("a {o}. a {o}.", "{o}")],
    "label_held": [F("your {o}.", "{o}"), F("it is your {o}.", "{o}"), F("you see your {o}.", "{o}")],
    "label_colour": [F("the {c} {o}.", "{o}"), F("a {c} {o}.", "{o}"), F("it is {c}.", "{c}"), F("the {o} is {c}.", "{c}")],
    "set_near": [F("here. the {o}.", "{o}"), F("look. here. the {o}.", "{o}")],   # her lesson's setup: the toy set within reach (A90)
    "set_far": [F("look. the {o}.", "{o}"), F("look. here. the {o}.", "{o}")],     # the roll rung's setup: the toy past its reach (A109)
    "hide": [F("look. the {o}.", "{o}")],                                                  # the hide game (A129): the toy shown as she takes it to
                                                                                          # the bucket (C115: the bucket lines were untrue here)
    "hide_told": [F("the {o} is in the bucket.", "bucket"), F("look. in the bucket. the {o}.", "{o}")],   # C115: what she did, told at the
                                                                                          # drop, before the reply she owes (the focus last:
                                                                                          # "bucket", the room's newest word)
    "hand_over": [F("here. a {o}.", "{o}"), F("look. here. a {o}.", "{o}")],      # the toy into its hand (the handle rung, A90)
    "show": [F("look at the {o}.", "{o}"), F("look. a {o}.", "{o}"), F("see the {o}?", "{o}"), F("see? a {o}.", "{o}"),
             F("here is a {o}.", "{o}")],
    "redirect": [F("look! the {o}.", "{o}"), F("look at the {o}!", "{o}"), F("see? the {o}.", "{o}")],
    "ask_where": [F("where is the {o}?", "{o}"), F("the {o}? where is the {o}?", "{o}"), F("look at the {o}.", "{o}")],
    "ask_what": [F("what is this?"), F("what is it?")],
    "ask_give": [F("give me the {o}.", "{o}"), F("give mama the {o}.", "{o}"), F("the {o}. give me the {o}.", "{o}")],
    "confirm": [F("yes. the {o}!", "{o}"), F("yes! a {o}.", "{o}"), F("good. the {o}!", "{o}"), F("yes. it is a {o}.", "{o}")],
    "confirm_act": [F("yes! good.", "good"), F("good. good!", "good")],
    "confirm_got": [F("you got the {o}!", "{o}"), F("yes! you have the {o}.", "{o}"), F("you hold the {o}.", "{o}")],      # C270: her words
    "confirm_held": [F("you hold the {o}.", "{o}"), F("yes! you have the {o}.", "{o}")],                                 # for its own act
    "confirm_stood": [F("you stand!", "stand"), F("yes! you stand.", "stand"), F("up. you are up!", "up")],     # C284
    "confirm_stepped": [F("you walk!", "walk"), F("yes! you walk.", "walk"), F("walk. walk!", "walk")],
    "confirm_lifted": [F("the {o} is up.", "up"), F("up. up! the {o}.", "{o}")],
    "confirm_shook": [F("shake! you shake the {o}.", "{o}"), F("shake. shake! the {o}.", "{o}")],
    "confirm_hit": [F("bang! the {o}.", "{o}"), F("you hit the {o}!", "{o}")],
    "confirm_reach_nearer": [F("yes. get the {o}.", "{o}"), F("get it! the {o}.", "{o}")],
    "recast": [F("{w}. yes. the {w}.", "{w}"), F("yes. the {w}!", "{w}")],
    "recast_word": [F("{w}. yes. {w}.", "{w}"), F("yes! {w}.", "{w}")],
    "echo": [F("{w}! the {w}!", "{w}"), F("the {w}! a {w}.", "{w}")],
    "echo_word": [F("{w}! {w}.", "{w}"), F("{w}. {w}!", "{w}")],
    "reply": [F("oh! the {o}.", "{o}"), F("you see the {o}?", "{o}")],
    "reply_social": [F("hi {n}!", "{n}"), F("mama is here.")],                # C172 (C226's three more frames withdrawn at A172: day 64's first
                                                                                # 3,000 ticks spoke 241 lines against 172, 78% of them social: five
                                                                                # frames under the same-line rule let her say the social line
                                                                                # five times as often; the toy-in-view reply stands)
    "narrate_walk": [F("walk. walk. walk.", "walk"), F("mama walk. walk.", "walk"), F("look. mama walk. walk.", "walk")],   # C300: she shows walking (C351 amended twice: 'walks', 'goes' and the commas were never hers; the frames refused since C300)
    "narrate_sib": [F("look. walk. walk. walk.", "walk"), F("see? walk. walk.", "walk"), F("walk. walk. walk.", "walk")],   # D2: the sibling walking in its view
    "narrate_sib_get": [F("look. get the {o}.", "{o}"), F("up. the {o}.", "{o}"), F("see? the {o}.", "{o}")],   # D2: the sibling getting a toy
    "narrate_get": [F("mama gets the {o}.", "{o}"), F("up. mama has the {o}.", "{o}"), F("look. mama gets the {o}.", "{o}")],   # C300: she shows getting
    "narrate_fell": [F("uh oh. the {o} is down.", "down"), F("oh! the {o} is down.", "down")],
    "narrate_on": [F("the {o} is on the {p}.", "{p}")],
    "narrate_rolled": [F("oh! you roll.", "roll"), F("roll! you roll.", "roll")],
    "narrate_sat": [F("oh! you sit.", "sit"), F("you sit! sit.", "sit")],
    "body": [F("your {b}.", "{b}"), F("here is your {b}.", "{b}"), F("your {b}! your {b}.", "{b}"), F("this is your {b}.", "{b}")],
    "motor_sit": [F("up! stand up.", "up"), F("stand. you stand.", "stand"), F("up. up. up!", "up"), F("walk. you walk.", "walk")],   # C268
    "motor_roll": [F("roll. roll.", "roll"), F("roll. you roll.", "roll")],
    "leave": [F("bye bye {n}.", "{n}"), F("bye {n}. bye bye.", "bye")],
    "peekaboo_hide": [F("where is mama?", "mama")],
    "peekaboo": [F("peekaboo!", "peekaboo"), F("peekaboo {n}!", "{n}")],
    "comfort": [F("oh. oh {n}.", "{n}"), F("uh oh. mama is here."), F("mama is here.")],
    "turn_over": [F("oh. up.", "up"), F("up. up. up!", "up")],                   # face down in distress: she turns it over (A90)
    "hit": [F("oh!")],
    "no": [F("no.")],
    "no_talkover": [F("no.")],
    "night": [F("night night {n}.", "{n}"), F("night night.", "night")],
    # a formal trial's single test sentence (4.8, 12, the lead's decision): said with no act, only her mouth moving; since P3's
    # fourteenth round each its test word spliced into one carrier phrase, a recording before it and a tag after it, as
    # looking-while-listening's "where's the X? can you find it?" (lang/stimuli.py)
    "trial_where": [F("where is the {o}? see?", "{o}")],                # place and exemplar (M6(a); Quinn et al. 1993)
    "trial_combo": [F("where is the {c} {o}? see?", "{o}")],            # the combination never heard (M6(b), A28, A55)
    "trial_name": [F("hi. {n}. hi.", "{n}")],                           # its name; a foil in its place (foil_line)
}
TRIAL_FRAMES = ("trial_where", "trial_combo", "trial_name")   # a trial tests known words: never a new word's line (new_word_lines)

# a new word's introduction (4.8: a variation set of 3 lines, the word last, frames differing by at least one word, 4.5): each
# class's frames, of which she says only those measured to put the word on the line's pitch peak (A34: the line check's peak
# rule), 3 of distinct words drawn among them (FastLayer.variation_set: two lines of the same words, "a rattle." and "a rattle!",
# are one frame to the set, key()); a word with fewer than 3 such lines of distinct words waits. The P3 verifier's third round
# measured 17 of the first 210 lines (3 a word) off the peak; each class's list was then widened by frames of the same kind, and
# every one measured (tools/sim_voice_check.py --peak; "look at the X" dropped from the face's frames: it is the gaze ask's
# form, A51). Its fourth round found the widening had added punctuation-only twins, which left "red" on the red ball and
# "yellow" on the duck (A55's colour words, where A55 puts them) with 2 lines of distinct words on the peak: the colours took
# "{w}. {w}!", "look. {w}." / "look. {w}!", "oh! {w}!" and "the {o}. {w}!", each measured, so each colour on each of its toys
# keeps at least 3 (red on the ball 5, yellow on the duck 5). The twins stay in the lists (a line's other ending is a different
# clip, and may be the one on its peak), never two in one set. Every set of 3 of a word's lines on its peak, of distinct words,
# runs at most 3 words a second (C25; the table's spans, body/tests/test_sim_lang.py test 32).
INTRO = {
    "toy": [F("a {w}.", "{w}"), F("the {w}!", "{w}"), F("you see the {w}?", "{w}"),     # the design's set (4.5)
            F("look. a {w}.", "{w}"), F("see? a {w}.", "{w}"), F("a {w}!", "{w}")],
    "fixture": [F("the {w}.", "{w}"), F("look. the {w}!", "{w}"), F("see the {w}?", "{w}"), F("you see the {w}?", "{w}"),
                F("the {w}!", "{w}")],
    "body": [F("your {w}.", "{w}"), F("here is your {w}!", "{w}"), F("see your {w}?", "{w}"), F("your {w}!", "{w}"),
             F("you see your {w}?", "{w}")],
    "face": [F("look. {w}.", "{w}"), F("see? {w}!", "{w}"), F("the {w}.", "{w}"), F("the {w}!", "{w}"),
             F("see the {w}?", "{w}"), F("you see the {w}?", "{w}")],
    "colour": [F("it is {w}.", "{w}"), F("the {o} is {w}!", "{w}"), F("see? {w}.", "{w}"), F("{w}!", "{w}"),
               F("it is {w}!", "{w}"), F("see? {w}!", "{w}"), F("the {o} is {w}.", "{w}"), F("{w}. {w}!", "{w}"),
               F("look. {w}!", "{w}"), F("oh! {w}!", "{w}"), F("the {o}. {w}!", "{w}"), F("look. {w}.", "{w}")],
    "adj": [F("it is {w}.", "{w}"), F("{w}!", "{w}"), F("see? {w}.", "{w}"), F("it is {w}!", "{w}"), F("look. {w}.", "{w}"),
            F("see? {w}!", "{w}")],
    "verb": [F("{w}!", "{w}"), F("look. {w}.", "{w}"), F("you {w}?", "{w}"), F("see? {w}.", "{w}"), F("{w}. {w}!", "{w}"),
             F("look. {w}!", "{w}")],
    "past": [F("you {w}!", "{w}"), F("oh! you {w}.", "{w}"), F("{n} {w}.", "{w}"), F("you {w}.", "{w}"),
             F("oh. you {w}!", "{w}")],
    "social": [F("{w}!", "{w}"), F("oh. {w}.", "{w}"), F("{w}. {w}!", "{w}"), F("{w}.", "{w}"), F("oh! {w}.", "{w}")],
    "frame": [],
}
INTRO_WORD = {                                                             # words whose class frames do not fit them
    "fell": [F("uh oh. it {w}.", "{w}"), F("the {o} {w}.", "{w}"), F("oh! it {w}!", "{w}"), F("it {w}!", "{w}"),
             F("oh. it {w}.", "{w}")],
    "done": [F("all {w}.", "{w}"), F("{w}!", "{w}"), F("oh. {w}.", "{w}"), F("all {w}!", "{w}"), F("{w}.", "{w}")],
    "that": [F("look at {w}.", "{w}"), F("what is {w}?", "{w}"), F("see {w}?", "{w}")],
    "can": [F("you {w}!", "{w}"), F("{n} {w}.", "{w}"), F("you {w}?", "{w}"), F("mama {w}.", "{w}"),   # "you can!" and "you
            F("you {w}. you {w}!", "{w}")],                                                            # can?" are one frame
    "too": [F("you {w}.", "{w}"), F("mama {w}!", "{w}"), F("{n} {w}?", "{w}")],
    "now": [F("look {w}.", "{w}"), F("up {w}!", "{w}"), F("sit {w}?", "{w}")],
    "again": [F("{w}!", "{w}"), F("roll {w}.", "{w}"), F("look {w}?", "{w}")],
    "out": [F("{w}!", "{w}"), F("it is {w}.", "{w}"), F("{w}. {w}.", "{w}")],
    "off": [F("{w}!", "{w}"), F("it is {w}.", "{w}"), F("{w}. {w}.", "{w}")],
    "under": [F("{w}!", "{w}"), F("it is {w}.", "{w}"), F("look. {w}.", "{w}")],
    "thanks": [F("{w}!", "{w}"), F("oh. {w}.", "{w}"), F("yes. {w}.", "{w}")],
    "please": [F("{w}.", "{w}"), F("more? {w}.", "{w}"), F("give me. {w}.", "{w}")],
    "all": [F("it is {w}.", "{w}"), F("look. {w}.", "{w}"), F("{w}!", "{w}")],
    # "mouth", short: its face frames but "the mouth." and "the mouth!", with which 3 of its lines ran at up to 3.19 words a
    # second, over C25's 3 (measured, tools/sim_voice_check.py --growth); every set of 3 of these is under it
    "mouth": [F("look. {w}.", "{w}"), F("see? {w}!", "{w}"), F("see the {w}?", "{w}"), F("you see the {w}?", "{w}")],
}

_SLOT = re.compile(r"\{([a-z])\}")
_P = re.escape(K.PUNCT)
_FORM = re.compile(rf"[a-z]+(?:[{_P}]? [a-z]+)*[{_P}]")
_WORD = re.compile(r"[a-z]+")


def words(text):
    return _WORD.findall(text)


def key(text):
    """a line's words, its punctuation dropped: what makes two lines the same line (4.6: "the same line not within 60 ticks")
    and two lines of a variation set one frame (4.5: frames differing by at least one word)."""
    return " ".join(_WORD.findall(text))


for _k, _fs in list(FRAMES.items()) + list(INTRO.items()) + list(INTRO_WORD.items()):
    for _t, _f in _fs:                                # the focus said last (a trial's test word: before its one tag, P3's
        _ws = words(_t.replace("{", "").replace("}", ""))                                              # fourteenth round)
        assert _f is None or _ws[-2 if _k in TRIAL_FRAMES else -1] == _f.strip("{}"), (_k, _t, _f)


@dataclass(frozen=True)
class Line:
    text: str
    intent: str
    register: str = "plain"
    focus: str = None               # the focus word (said last; a trial's test word, before its tag), or None
    emphasis: str = None            # the word the voice puts on a pitch peak and lengthens (4.4): the focus
    refs: tuple = ()                # the object ids its slots were filled from
    source: str = "fast"            # "fast" (the templates) or "claude" (a steering row's line)
    frame: str = ""                 # the frame it was filled from
    shape: tuple = None             # a formal trial's test word's own rate and pitch: (word, rate, pitch), the recipe that puts
                                    # every sentence its draw could give on one timeline (stimuli.py, synth.line_ssml)


# ---------------------------------------------------------------------------------------- the new word's pitch peak (A34)
# PEAK: every line she can say with a growth word as the new word, measured by tools/sim_voice_check.py --peak (the line in the
# new-word register, its last word emphasized, 4.4): text -> [the new word, its peak F0 (Hz), the highest other word's peak F0,
# that word, the line's words, its spoken span in seconds] (a peak None where no frame of the word is voiced). On its peak: the
# new word's peak above every other word's (a tie at the tracker's resolution is no peak: P3's fourth round, where ties had
# counted). The spans give C25's words a second for a set (set_rate).
PEAK, PEAK_META = {}, {}
_PEAK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), K.PEAK_FILE)
if os.path.exists(_PEAK_PATH):
    with open(_PEAK_PATH) as _fh:
        _d = json.load(_fh)
    PEAK.update(_d["lines"])
    PEAK_META.update(_d["meta"])


def on_peak(text):
    """a new word's line (its last word the new word) -> (on its pitch peak as measured, why not)."""
    r = PEAK.get(text)
    if r is None:
        return False, "not measured on the line's pitch peak (A34; tools/sim_voice_check.py --peak)"
    w, pk, other, ow = r[:4]
    if pk is None:
        return False, f"measured with no voiced frame in {w!r} (A34)"
    if other is not None and other >= pk:
        return False, (f"measured off the line's pitch peak: {ow!r} peaks at {other:.1f} Hz, "
                       f"{'level with' if other == pk else 'over'} {w!r} at {pk:.1f} Hz (A34)")
    return True, ""


def set_rate(lines):
    """C25: the words a second of lines said together (a variation set), pooled over their spoken spans as measured (PEAK)."""
    return sum(PEAK[x][4] for x in lines) / sum(PEAK[x][5] for x in lines)


def intro_on_peak(word, o=None):
    """a growth word's introduction lines measured to put it on the line's pitch peak (A34), its frames filled with o (a Seen, for
    a frame with an object slot) -> [text]: its set is 3 of them of distinct words (key()), so a word with fewer waits."""
    out = []
    for fr in intro_frames(word):
        got = fill(fr, o=o, w=word)
        if got is not None and on_peak(got[0])[0]:
            out.append(got[0])
    return out


def new_word_lines():
    """every line the fast layer can say with a growth word as the day's new word (said last, emphasized, in the new-word
    register) -> sorted [(text, the new word)]: each word's introduction frames (INTRO, INTRO_WORD; {o} filled with every object
    word), and every frame of her intents (a trial's aside: it tests known words) whose focus a growth word fills ({o} a
    growth toy, {c} a colour, {p} a growth fixture, {b} a growth body word, {w} any growth word her ear can hear: her recasts
    and echoes; a literal focus that is a growth word: "all done."), each passing the line check's other rules with every
    other word hers. What
    tools/sim_voice_check.py --peak measures; Claude's lines ending on the new word pass only if among them."""
    from .percept import Seen                                          # noqa: PLC0415
    growth = [g for g in GROWTH_WORDS if GROWTH_CLASS[g] != "frame"]
    objs = sorted(OBJECT_NOUNS)
    frames = [(fr, None) for k, fs in FRAMES.items() if k not in TRIAL_FRAMES for fr in fs] + \
        [(fr, w) for w in growth for fr in intro_frames(w)]
    out = set()
    for fr, word in frames:
        slots = set(_SLOT.findall(fr[0]))
        o_opts = [None]
        if slots & set("ocp"):
            cols = sorted(COLOURS) if "c" in slots else [""]
            ons = sorted(FIXTURE_NOUNS) if "p" in slots else [""]
            o_opts = [Seen(n, n, c, on) for n in objs for c in cols for on in ons]
        b_opts = sorted(CHILD_BODY) if "b" in slots else [None]
        w_opts = ([word] if word else growth) if "w" in slots else [None]
        for o in o_opts:
            for b in b_opts:
                for w in w_opts:
                    got = fill(fr, o=o, b=b, w=w, fixtures=FIXTURE_NOUNS)
                    if got is None:
                        continue
                    text, focus, _ = got
                    if focus not in growth or (word is not None and focus != word):
                        continue
                    vocab = BIRTH_WORDS + tuple(g for g in GROWTH_WORDS if g != focus)
                    if check(text, vocab, focus, held=(), peak=False)[0]:
                        out.add((text, focus))
    return sorted(out)


# ------------------------------------------------------------------------------------------------------ the line check
def _new_words(new_word):
    if new_word is None:
        return ()
    return (new_word,) if isinstance(new_word, str) else tuple(new_word)


_DEIXIS = ("it", "this", "that")
_DETS = ("a", "the", "your")

# ------------------------------------------------------------------------------------------- Claude's lines (A14, 4.5)
INTERJECTIONS = frozenset({"oh", "uh"})   # C172: the child's grunts; never echoed back (life day 44: "oh" 42% of its tokens, 46% of her lines began "oh")
CLAUDE_INTERJ = (("oh",), ("uh", "oh"), ("look",), ("see",))   # a sentence of one of these alone claims nothing ("oh!",
                                                                 # "uh oh.", "look.", "see?"); never two run together
NAMEABLE = OBJECT_NOUNS | FIXTURE_NOUNS | CHILD_BODY | HER_FACE | {PARENT_NAME}
CLAUDE_DID = {"roll": "rolled", "rolled": "rolled", "sit": "sat", "sat": "sat"}   # "you roll." / "you sat.": its event, seen
CLAUDE_FORMS = ("a name, said with '.' or '!': a toy with 'a', 'the' or 'your' and a colour or none ('the duck.', 'a red "
                "block!', 'your cup.' of a toy it holds), a fixture or her face's part with 'the' ('the mat.'), its body with "
                "'your' ('your foot.'), 'mama' alone; 'oh', 'uh oh', 'look' or 'see' alone, one to a sentence; 'see the X' or "
                "'you see the X', any ending; with '.' or '!': 'it / this / that / here / there is a X', 'it / this / that is "
                "C', 'the X is on the Y' (a fixture), 'the X is down', 'the X is C', 'the X fell', 'mama is here', 'you roll', "
                "'you sit'")


def _noun_phrase(ws):
    """a name as Claude may say it -> (det, colour, noun), or None: a toy with a / the / your and at most one colour; a fixture or
    her face's part with "the"; the child's body with "your"; "mama" alone (never "a mama" or "the mama")."""
    i, det, col = 0, None, None
    if i < len(ws) and ws[i] in _DETS:
        det, i = ws[i], i + 1
    if i < len(ws) and ws[i] in COLOURS:
        col, i = ws[i], i + 1
    if i != len(ws) - 1:
        return None
    n = ws[i]
    ok = (n in OBJECT_NOUNS and det is not None) or \
        (col is None and ((n in FIXTURE_NOUNS | HER_FACE and det == "the") or (n in CHILD_BODY and det == "your") or
                          (n == PARENT_NAME and det is None)))
    return (det, col, n) if ok else None


def _sentences_marked(text):
    """[(words, its mark)] of a line that passed the form."""
    return [(words(x), m) for x, m in re.findall(r"([a-z ]+)([.?!])", text) if words(x)]


def claude_claims(text):
    """Claude's line (A14, 4.5) -> (claims, "") or (None, why): refused when it judges (praise, the reprimand), asks (an ask's
    word anywhere, "look" with a word after it in its sentence, the child's name), or has a sentence of a form whose claim the
    check cannot hold true (CLAUDE_FORMS); else the claims its sentences make, each held true against what she perceives when
    the line is said (_claude_true)."""
    for w in words(text):
        if w in K.PRAISE or w in K.REPRIMAND:
            return None, f"praise and the reprimand are the fast layer's judgments (A14): {w!r}"
        if w in K.ASK_WORDS:
            return None, f"an ask is the fast layer's: it is judged (A14: Claude requests the intent, never says the ask): {w!r}"
        if w == NAME:
            return None, ("an ask is the fast layer's: the child's name is the call (judged, at most once per 240 ticks, never "
                          "while it looks at her: A13, 4.6); Claude requests the call, the greeting or the goodnight")
    claims = []
    for ws, mark in _sentences_marked(text):
        bad = (f"not a claim she can hold true: {' '.join(ws) + mark!r} (Claude's lines take these forms only: {CLAUDE_FORMS})")
        if "look" in ws[:-1]:
            return None, "an ask is the fast layer's: 'look' with a word after it ('look at ...', 'look here') asks a look"
        if tuple(ws) in CLAUDE_INTERJ:
            continue
        k = 2 if ws[:2] == ["you", "see"] else (1 if ws[0] == "see" else 0)
        if k:
            np_ = _noun_phrase(ws[k:])
            if np_ is None or np_[2] not in OBJECT_NOUNS:
                return None, bad
            claims.append(("sees",) + np_)
            continue
        if mark == "?":                               # a question asks (the fast layer's); only "see the X?" shows
            return None, bad
        np_ = _noun_phrase(ws)
        if np_ is not None:
            claims.append(("name",) + np_)
            continue
        if len(ws) >= 3 and ws[1] == "is" and ws[0] in _DEIXIS + ("here", "there"):
            rest = ws[2:]
            if len(rest) == 1 and rest[0] in COLOURS and ws[0] in _DEIXIS:
                claims.append(("its_colour", rest[0]))
                continue
            np_ = _noun_phrase(rest)
            if np_ is None or np_[2] not in OBJECT_NOUNS:
                return None, bad
            claims.append(("is",) + np_)
            continue
        if len(ws) > 2 and ws[0] == "the" and ws[1] in OBJECT_NOUNS:
            i = 1
            n, pred = ws[i], ws[i + 1:]
            if pred == ["fell"]:
                claims.append(("down", n))
                continue
            if pred[0] == "is" and len(pred) >= 2:
                p2 = pred[1:]
                if p2[0] == "in":
                    if len(p2) == 3 and p2[1] == "the" and p2[2] in OPEN_CONTAINERS:   # A129: an open bucket she sees into
                        claims.append(("on", n, p2[2]))
                        continue
                    return None, f"not seen: a thing in the {p2[-1]} (containment she cannot see)"
                if len(p2) == 3 and p2[:2] == ["on", "the"] and p2[2] in FIXTURE_NOUNS:
                    claims.append(("on", n, p2[2]))
                    continue
                if p2 == ["down"]:
                    claims.append(("down", n))
                    continue
                if len(p2) == 1 and p2[0] in COLOURS:
                    claims.append(("colour", n, p2[0]))
                    continue
            return None, bad
        if ws == [PARENT_NAME, "is", "here"]:
            claims.append(("present",))
            continue
        if len(ws) == 2 and ws[0] == "you" and ws[1] in CLAUDE_DID:
            claims.append(("did", CLAUDE_DID[ws[1]]))
            continue
        return None, bad
    return claims, ""


def claude_toys(claims):
    """the toys a Claude line's claims name (4.10's redirect rule: a line naming a toy the child does not attend redirects)."""
    out = []
    for c in claims:
        n = c[3] if c[0] in ("name", "sees", "is") else (c[1] if c[0] in ("on", "down", "colour") else None)
        if n in OBJECT_NOUNS and n not in out:
            out.append(n)
    return out


def _claude_true(claims, percept, seen, recent_events):
    """each of a Claude line's claims held against what she perceives now and saw in the last RECENT ticks."""
    att = percept.attended()                               # as she reads the child: its head's line and its hands (A40)
    held = [o for o in (percept.obj(h) for h in percept.child_holds) if o is not None]
    fell = {ob for _, k, ob in recent_events if k == "fell"}
    kinds = {k for _, k, _o in recent_events}

    def like(o, n, col):
        return o.name == n and (col is None or o.colour == col)
    for c in claims:
        kind = c[0]
        if kind in ("name", "sees", "is"):
            det, col, n = c[1:]
            if det == "your" and n not in CHILD_BODY and not any(like(o, n, col) for o in held):
                return False, f"not true: 'your {n}': the child holds no {n}"
            if kind == "sees" and not any(like(o, n, col) and o.child_sees for o in seen.values()):
                return False, f"not true: the child does not see a {n}"
            if kind == "is" and not any(like(o, n, col) for o in att):
                return False, f"not true: it is not a {n} (it names what she reads the child looking at, holding or " \
                              f"reaching toward)"
        elif kind == "its_colour" and not any(o.colour == c[1] for o in att):
            return False, f"not true: what she reads the child attending is not {c[1]}"
        elif kind == "on" and not any(o.name == c[1] and o.on == c[2] for o in seen.values()):
            return False, f"not true: no {c[1]} on the {c[2]} as she sees it"
        elif kind == "down" and not any(o.name == c[1] and o.id in fell for o in seen.values()):
            return False, f"not seen happen: a {c[1]} falling in the last {K.RECENT} ticks"
        elif kind == "colour" and not any(o.name == c[1] and o.colour == c[2] for o in seen.values()):
            return False, f"not true: no {c[2]} {c[1]} she sees"
        elif kind == "present" and not percept.present:
            return False, "not true: she is away"
        elif kind == "did" and c[1] not in kinds:
            return False, f"not seen happen: the child {c[1]} (not in the last {K.RECENT} ticks)"
    return True, ""


def check(text, vocab, new_word=None, percept=None, refs=(), held=K.HELD_PAIRS, source="fast", register=None,
          recent_events=(), peak=True):
    """-> (ok, reason). vocab: the words she has (the birth words and the growth words entered); new_word: the day's new words
    (a word or a tuple of up to NEW_PER_DAY: each allowed, one a line, and only last); percept: what she perceives now
    (body/sim/lang/percept.Percept; None skips the perceived and true rules: for listing lines ahead, never for a line she
    says); refs: the object ids the line is about (its filled slots, or the object an intent was given); held: the never-taught
    pairs still held out; source: "fast" or "claude"; recent_events: the (tick, kind, object) she saw in the last RECENT
    ticks; peak: the new word's line must be measured on its pitch peak (A34; False only for listing the lines to measure)."""
    if not isinstance(text, str) or not _FORM.fullmatch(text):
        return False, "form: lowercase words, single spaces, '. ? !' only after a word, ending in one"
    ws = words(text)
    if len(ws) > K.MAX_WORDS:
        return False, f"too long: {len(ws)} words"
    new = _new_words(new_word)
    oov = [w for w in ws if w not in vocab and w not in new]
    if oov:
        return False, f"not her word: {oov[0]!r}"
    said_new = [w for w in ws if w in new and w not in vocab]
    if len(set(said_new)) > 1:
        return False, f"two new words in one line: {sorted(set(said_new))}"
    if said_new and ws[-1] != said_new[0]:
        return False, f"the new word {said_new[0]!r} not last"
    if said_new and peak:
        ok, why = on_peak(text)
        if not ok:
            return False, f"the new word {said_new[0]!r}: {why}"
    wset = set(ws)
    for a, b in held:
        if a in wset and b in wset:
            return False, f"held-out pair ({a}, {b})"
    claims = None
    if source == "claude":
        praise = [w for w in ws if w in K.PRAISE]
        if praise or register == "approval":
            return False, f"praise is the fast layer's: {praise[0] if praise else 'the approval register'!r}"
        claims, why = claude_claims(text)
        if claims is None:
            return False, why
    if percept is None:
        return True, ""
    seen = {s.id: s for s in percept.seen}
    for r in refs:
        if r not in seen:
            return False, f"unseen: {r!r}"
    for c in (w for w in ws if w in COLOURS):
        cands = [s for s in (seen[r] for r in refs) if s.colour == c] if refs else [s for s in seen.values() if s.colour == c]
        named = [w for w in ws if w in OBJECT_NOUNS]
        if named:
            cands = [s for s in cands if s.name in named]
        if not cands:
            return False, f"unseen: a {c} thing"
        for a, b in held:
            if a == c and any(s.name == b for s in cands):
                return False, f"held-out pair ({a}, {b}) by its referent"
    names = {s.name for s in seen.values()}
    for w in ws:
        if w in OBJECT_NOUNS and w not in names:
            return False, f"unseen: {w!r}"
        if w in FIXTURE_NOUNS and w not in percept.fixtures:
            return False, f"unseen: {w!r}"
        if w in CHILD_BODY and not percept.child_in_view:
            return False, f"unseen: the child's {w}"
        if w in PAST_EVENTS and not any(k in PAST_EVENTS[w] for _, k, _o in recent_events):
            return False, f"not seen happen: {w!r}"
    ok, why = _true(text, percept, seen, refs, recent_events)
    if ok and claims is not None:
        return _claude_true(claims, percept, seen, recent_events)
    return ok, why


def _attended(percept, seen, refs):
    """what "it" / "this" / "that" can point at: the line's own objects, else what she reads the child attending (its head's
    line, what it holds or reaches toward: A40)."""
    if refs:
        return [seen[r] for r in refs if r in seen]
    return percept.attended()


def _true(text, percept, seen, refs, recent_events):
    """the few claims her lines can make, held to what she sees (the check's "true" rule)."""
    att = _attended(percept, seen, refs)
    objs = [seen[r] for r in refs if r in seen] or list(seen.values())
    for sent in re.split(r"[.?!] ?", text):
        ws = words(sent)
        for i, w in enumerate(ws):
            if w != "is" or i == 0 or i + 1 >= len(ws):
                continue
            subj, rest = ws[i - 1], ws[i + 1:]
            if subj in _DEIXIS:
                x = rest[1] if rest[0] in _DETS and len(rest) > 1 else rest[0]
                if x in OBJECT_NOUNS and not any(o.name == x for o in att):
                    return False, f"not true: {subj!r} is not a {x} (it names what the line is about, or what she reads " \
                                  f"the child attending)"
                if x in COLOURS and not any(o.colour == x for o in att):
                    return False, f"not true: {subj!r} is not {x}"
                subj_objs = att
            elif subj in OBJECT_NOUNS:
                subj_objs = [o for o in objs if o.name == subj]
            else:
                continue
            if rest[0] in ("on", "in") and len(rest) >= 3 and rest[1] == "the":
                if rest[0] == "in" and rest[2] not in OPEN_CONTAINERS:     # A129: the bucket is open, she sees what lies in it (its `on`)
                    return False, f"not seen: a thing in the {rest[2]} (containment she cannot see)"
                if not any(o.on == rest[2] for o in subj_objs):
                    return False, f"not true: no {subj if subj not in _DEIXIS else 'such thing'} on the {rest[2]} as she sees it"
            if rest[0] == "down" and not any(k == "fell" and any(o.id == ob for o in subj_objs)
                                             for _, k, ob in recent_events):
                return False, f"not seen happen: {subj!r} down (no fall she saw in the last {K.RECENT} ticks)"
        for i in range(len(ws) - 2):
            if ws[i] == "you" and ws[i + 1] == "see":
                rest = ws[i + 2:]
                x = rest[1] if rest[0] in _DETS and len(rest) > 1 else rest[0]
                if x in OBJECT_NOUNS and not any(o.name == x and o.child_sees for o in objs):
                    return False, f"not true: the child does not see a {x}"
    return True, ""


# ---------------------------------------------------------------------------------------------------------- filling
def fill(frame, o=None, b=None, w=None, fixtures=()):
    """a frame's text with its slots filled -> (text, focus word, refs) or None when a slot has nothing to fill it.
    o: a Seen (percept.Seen); b: a body word; w: a word; fixtures: the fixture words she sees ({p} is o's resting place only when
    it is one of them)."""
    text, focus = frame
    vals = {"n": NAME}
    refs = ()
    for s in set(_SLOT.findall(text)):
        if s in "ocp":
            if o is None:
                return None
            refs = (o.id,)
            if s == "o":
                vals["o"] = o.name
            elif s == "c":
                if not o.colour:
                    return None
                vals["c"] = o.colour
            else:
                if o.on not in fixtures:
                    return None
                vals["p"] = o.on
        elif s == "b":
            if b is None:
                return None
            vals["b"] = b
        elif s == "w":
            if w is None:
                return None
            vals["w"] = w
    out = _SLOT.sub(lambda m: vals[m.group(1)], text)
    f = None if focus is None else (vals[focus[1]] if focus.startswith("{") else focus)
    return out, f, refs


def foil_line(foil):
    """a trial's foil (4.8, 12): the name test's, "hi. <foil>. hi." (the name's frame, trial_name; Mandel, Jusczyk and Pisoni
    1995) for a foil in consts.NAME_FOILS, a name she never uses for it or for anything, said in its name's register and
    stillness; an exemplar trial's, "where is the <foil>? see?" (trial_where's frame; A60b's level 2, the skeptic's design) for
    a foil in consts.NOUN_FOILS, a word she never tells, said as the form's words are. With her chatter's (foil_chatter) the
    only lines she says that are not of her words (the line check's vocabulary): refused for anything else, and never
    Claude's."""
    if foil in K.NAME_FOILS:
        return FRAMES["trial_name"][0][0].replace("{n}", foil)
    if foil in K.NOUN_FOILS.values():
        return FRAMES["trial_where"][0][0].replace("{o}", foil)
    raise ValueError(f"not a foil: {foil!r} (consts.NAME_FOILS, consts.NOUN_FOILS)")


FOIL_CHATTER = F("oh. {f}.", "{f}")     # her chatter's foil line (A60b 13, consts.FOIL_EXPOSURE): no label, no object, no act


def foil_chatter(foil):
    """her non-teaching chatter carrying a registered noun's never-told foil (the lead's decision after 67741fd: the foil heard
    as often as its noun): "oh. <foil>." (FOIL_CHATTER), a sound of hers with no referent, never a label, no act, no object
    named; never in FRAMES, so no template, set or request makes it; refused for anything but a noun's foil."""
    if foil not in K.NOUN_FOILS.values():
        raise ValueError(f"not a noun's foil: {foil!r} (consts.NOUN_FOILS)")
    return FOIL_CHATTER[0].replace("{f}", foil)


def frames_for(intent):
    return FRAMES[intent]


def intro_frames(word):
    return INTRO_WORD.get(word, INTRO[GROWTH_CLASS[word]])


def showable(word, world):
    """can a growth word enter (A15)? world: dict(objects={name: [colours]}, fixtures=[...], acts=[...], events=[...], face=[...])
    (ROOM_AT_BIRTH, the acts her motion's). -> (ok, why not)."""
    if GROWTH_CLASS.get(word) == "frame":
        return False, "no infant-directed line ends on it (4.8's rule: sentence-final)"
    need = NEEDS.get(word, ("never", "no need listed"))
    kind = need[0]
    if kind == "never":
        return False, need[1]
    if kind == "moment":
        return False, "a word with no referent or act to show: its moment is the day plan's (P4), not yet built"
    if kind == "child":
        return True, ""
    arg = need[1]
    have = {"obj": set(world.get("objects", {})), "fixture": set(world.get("fixtures", ())), "act": set(world.get("acts", ())),
            "event": set(world.get("events", ())), "face": set(world.get("face", ()))}
    if kind == "twins":
        n = sum(1 for cs in world.get("objects", {}).values() for c in cs if c == arg)
        return (n >= 2, "" if n >= 2 else f"fewer than two {arg} toys (B2's colour twins)")
    if kind == "face":
        ok = bool(have["face"]) if arg == "any" else arg in have["face"]
        return ok, "" if ok else f"her face shows no {arg}"
    ok = arg in have[kind]
    return ok, "" if ok else (f"her motion cannot {arg!r} (W2 declares what it can do)" if kind == "act" else
                              f"no {kind} {arg!r} in the world")


def show_now(word, percept, recent_events=()):
    """can she show a growth word now (A15: "the parent can show it within the minute"), from what she perceives and saw in the
    last RECENT ticks? -> (what its set is said of: [Seen], or [None] for a word with no object; "") or ([], why not). A toy, a
    colour and a quality of a toy ("soft": the bear) are shown on an object she sees (she can fetch it); a fixture she must see;
    the child's body, the child in her view; her face and her own acts, the child able to see her, and a verb shown by an act
    that handles a toy (TOY_ACTS) on a toy in the child's view and not in its hand, so the act names its toy (A51); an event,
    one she saw."""
    need = NEEDS.get(word, ("never", "no need listed"))
    kind = need[0]
    if kind == "obj":
        c = [x for x in percept.seen if x.name == need[1]]
        return (c, "") if c else ([], f"she sees no {need[1]}")
    if kind == "twins":
        c = [x for x in percept.seen if x.colour == need[1]]
        return (c, "") if c else ([], f"she sees nothing {need[1]}")
    if kind == "fixture":
        return ([None], "") if need[1] in percept.fixtures else ([], f"she does not see the {need[1]}")
    if kind == "child":
        return ([None], "") if percept.child_in_view else ([], "the child is out of her view")
    if kind == "act" and need[1] in TOY_ACTS:           # shown on a toy in the child's view, never one in its hand
        if not (percept.present and percept.seen_by_child):
            return [], "the child cannot see her"
        c = sorted((x for x in percept.seen if x.name in OBJECT_NOUNS and x.on != "hand" and x.child_sees), key=lambda x: x.id)
        return (c, "") if c else ([], f"no toy in the child's view she can {need[1].replace('_', ' ')} to show it")
    if kind in ("face", "act"):
        return ([None], "") if percept.present and percept.seen_by_child else ([], "the child cannot see her")
    if kind == "event":
        obs = [ob for _, k, ob in recent_events if k == need[1]]
        if not obs:
            return [], f"no {need[1]!r} she saw in the last {K.RECENT} ticks"
        c = [percept.obj(ob) for ob in obs if ob is not None and percept.obj(ob) is not None]
        return (c or [None]), ""
    return [], showable(word, {})[1]


def birth_lines(vocab=BIRTH_WORDS, objects=None, fixtures=ROOM):
    """every line the templates can make at birth, as (text, register, emphasis), sorted: what the voice makes ahead at a night
    boundary (4.4). objects: [(name, colour, on)] (default: each birth toy on the mat and on the sofa, no colours)."""
    from .conduct import INTENTS, register_for                        # noqa: PLC0415 (conduct imports this module)
    from .percept import Seen
    objs = objects or [(t, "", p) for t in TOYS for p in ("mat", "sofa")]
    out = set()
    for intent, frames in FRAMES.items():
        if intent not in INTENTS or intent in TRIAL_FRAMES:      # a trial's sentence is never made whole: its parts are
            continue                                             # (lang/stimuli.parts), when its probe runs
        for fr in frames:
            opts = []
            slots = set(_SLOT.findall(fr[0]))
            if slots & set("ocp"):
                opts = [dict(o=Seen(n, n, c, on)) for n, c, on in objs]
            elif "b" in slots:
                opts = [dict(b=b) for b in BODY]
            elif "w" in slots:
                noun = intent in ("recast", "echo")
                opts = [dict(w=w) for w in vocab if (w in NOUNS) == noun and w not in FUNCTION]
            else:
                opts = [{}]
            for kw in opts:
                got = fill(fr, fixtures=fixtures, **kw)
                if got is None:
                    continue
                text, focus, _ = got
                if check(text, vocab)[0]:
                    out.add((text, register_for(intent, text), focus))
    return sorted(out, key=lambda x: (x[0], x[1], x[2] or ""))
