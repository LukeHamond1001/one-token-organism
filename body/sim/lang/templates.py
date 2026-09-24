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
  words       only her vocabulary, plus the day's new word, and the new word, if said, last (4.5, A15)
  held out    never a never-taught pair before its test: both its words in one line, or its colour said of an object of its
              name (A28: "it is red." said of the red ball is the pair, whatever the words)
  perceived   every object word names an object she sees (and each filled slot's object is one she sees); a colour word, an object
              of that colour she sees (of the named kind, when the line names one); a fixture word, a fixture she sees; the
              child's body, the child in her view; a past form ("fell", "rolled", "sat", "got", "did"), an event she saw in the
              last RECENT ticks. Only what a person in her place could see or hear.
  Claude's    no praise ("yes", "good") and no approval register (A14)
The design's form rules (4.5) are the first three; "perceived" is the fast layer's rule that its lines are "filled from what the
parent can see" (4.5), made a check so Claude's lines are held to it too.

THE GROWTH QUEUE (4.8; GROWTH): the design's 76 words in its order, plus "ring" and "stacker" after "rattle" (4.8: "the world's
rattle, stacker and ring are named through the queue", though its list of 76 lacks the two). Each word has a class and the frames
of its introduction: a variation set of 3 lines, the word last (4.8), in the new-word register (4.4). A word enters only when its
referent or act is in the world and she can show it (A15): NEEDS names what that takes; a word whose need is absent from the world
(a "bath", a "book", "toes" on a G1 whose feet have none, "hot", "sing") waits, reported by showable(). Words that no
infant-directed line can end on ("and", "to", "with", "my", "i", "has", "not", "do") have no introduction frames and cannot enter by
4.8's rule (sentence-final); they wait for a later stage's method (reported, never forced).
"""
import re
from dataclasses import dataclass

from . import consts as K
from .lexicon import BIRTH_WORDS, GROUPS, NAME

# ------------------------------------------------------------------------------------------------ the words by kind
TOYS = dict(GROUPS)["toys"]                                           # ball duck block cup bear car drum bottle
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
    ("box", "toy"), ("soft", "adj"), ("loud", "adj"), ("wow", "social"), ("thanks", "social"), ("please", "social"),
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
NOUNS = OBJECT_NOUNS | FIXTURE_NOUNS | CHILD_BODY          # a recast or an echo says "the X" of these ("ball! the ball!")
FUNCTION = frozenset(dict(GROUPS)["function"])             # never a focus word: her ear never expects them
PAST_EVENTS = {"sat": ("sat",), "rolled": ("rolled",), "fell": ("fell",), "got": ("got",),
               "did": ("rolled", "sat", "got", "gave")}

# what each growth word needs in the world to be shown (A15): ("obj", name) an object of that name; ("fixture", name); ("twins",
# colour) two toys of that colour (B2); ("act", kind) an act her motion can do; ("event", kind) an event the world makes;
# ("face", kind) her own face; ("child",) the child's body; ("never", why)
NEEDS = {
    "rattle": ("obj", "rattle"), "ring": ("obj", "ring"), "stacker": ("obj", "stacker"), "book": ("obj", "book"),
    "box": ("obj", "box"), "tower": ("obj", "tower"),
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

# -------------------------------------------------------------------------------------------------------- the frames
F = lambda text, focus=None: (text, focus)   # noqa: E731
FRAMES = {
    "call": [F("{n}.", "{n}"), F("{n}. look at mama.", "mama"), F("{n}. look here.")],
    "hall_call": [F("{n}.", "{n}"), F("{n}? mama is here.")],
    "greet": [F("hi {n}.", "{n}"), F("hi. hi {n}.", "{n}"), F("hi {n}. mama is here.")],
    "return": [F("hi {n}! mama is here."), F("hi {n}. hi.", "hi")],
    "answer_bid": [F("mama is here.")],
    "label": [F("the {o}.", "{o}"), F("a {o}.", "{o}"), F("it is a {o}.", "{o}"), F("you see the {o}.", "{o}"),
              F("look. the {o}.", "{o}"), F("this is a {o}.", "{o}"), F("here is the {o}.", "{o}"), F("a {o}. a {o}.", "{o}")],
    "label_held": [F("your {o}.", "{o}"), F("it is your {o}.", "{o}"), F("you see your {o}.", "{o}")],
    "label_colour": [F("the {c} {o}.", "{o}"), F("a {c} {o}.", "{o}"), F("it is {c}.", "{c}"), F("the {o} is {c}.", "{c}")],
    "show": [F("look at the {o}.", "{o}"), F("look. a {o}.", "{o}"), F("see the {o}?", "{o}"), F("see? a {o}.", "{o}"),
             F("here is a {o}.", "{o}")],
    "redirect": [F("look! the {o}.", "{o}"), F("look at the {o}!", "{o}"), F("see? the {o}.", "{o}")],
    "ask_where": [F("where is the {o}?", "{o}"), F("the {o}? where is the {o}?", "{o}"), F("look at the {o}.", "{o}")],
    "ask_what": [F("what is this?"), F("what is it?")],
    "ask_give": [F("give me the {o}.", "{o}"), F("give mama the {o}.", "{o}"), F("the {o}. give me the {o}.", "{o}")],
    "confirm": [F("yes. the {o}!", "{o}"), F("yes! a {o}.", "{o}"), F("good. the {o}!", "{o}"), F("yes. it is a {o}.", "{o}")],
    "confirm_act": [F("yes! good.", "good"), F("good. good!", "good")],
    "recast": [F("{w}. yes. the {w}.", "{w}"), F("yes. the {w}!", "{w}")],
    "recast_word": [F("{w}. yes. {w}.", "{w}"), F("yes! {w}.", "{w}")],
    "echo": [F("{w}! the {w}!", "{w}"), F("the {w}! a {w}.", "{w}")],
    "echo_word": [F("{w}! {w}.", "{w}"), F("{w}. {w}!", "{w}")],
    "reply": [F("oh! the {o}.", "{o}"), F("you see the {o}?", "{o}")],
    "reply_social": [F("oh? hi {n}.", "{n}"), F("oh! mama is here.")],
    "narrate_fell": [F("uh oh. the {o} is down.", "down"), F("oh! the {o} is down.", "down")],
    "narrate_on": [F("the {o} is on the {p}.", "{p}")],
    "narrate_rolled": [F("oh! you roll.", "roll"), F("roll! you roll.", "roll")],
    "narrate_sat": [F("oh! you sit.", "sit"), F("you sit! sit.", "sit")],
    "body": [F("your {b}.", "{b}"), F("here is your {b}.", "{b}"), F("your {b}! your {b}.", "{b}"), F("this is your {b}.", "{b}")],
    "motor_sit": [F("sit. you sit.", "sit"), F("up! sit up.", "up"), F("up. up. up!", "up")],
    "motor_roll": [F("roll. roll.", "roll"), F("roll. you roll.", "roll")],
    "feed": [F("here is your bottle.", "bottle"), F("bottle. your bottle.", "bottle"), F("bottle?", "bottle")],
    "feed_more": [F("more?", "more"), F("more bottle?", "bottle")],
    "feed_done": [F("all done.", "done")],
    "leave": [F("bye bye {n}.", "{n}"), F("bye {n}. bye bye.", "bye")],
    "peekaboo_hide": [F("where is mama?", "mama")],
    "peekaboo": [F("peekaboo!", "peekaboo"), F("peekaboo {n}!", "{n}")],
    "comfort": [F("oh. oh {n}.", "{n}"), F("uh oh. mama is here."), F("mama is here.")],
    "hit": [F("oh!")],
    "no": [F("no.")],
    "night": [F("night night {n}.", "{n}"), F("night night.", "night")],
}

# a new word's introduction: 3 frames per class (4.8: a variation set, the word last, frames differing by at least one word)
INTRO = {
    "toy": [F("a {w}.", "{w}"), F("the {w}!", "{w}"), F("you see the {w}?", "{w}")],            # the design's set (4.5)
    "fixture": [F("the {w}.", "{w}"), F("look. the {w}!", "{w}"), F("see the {w}?", "{w}")],
    "body": [F("your {w}.", "{w}"), F("here is your {w}!", "{w}"), F("see your {w}?", "{w}")],
    "face": [F("look. {w}.", "{w}"), F("see? {w}!", "{w}"), F("look at the {w}.", "{w}")],
    "colour": [F("it is {w}.", "{w}"), F("the {o} is {w}!", "{w}"), F("see? {w}.", "{w}")],
    "adj": [F("it is {w}.", "{w}"), F("{w}!", "{w}"), F("see? {w}.", "{w}")],
    "verb": [F("{w}!", "{w}"), F("look. {w}.", "{w}"), F("you {w}?", "{w}")],
    "past": [F("you {w}!", "{w}"), F("oh! you {w}.", "{w}"), F("{n} {w}.", "{w}")],
    "social": [F("{w}!", "{w}"), F("oh. {w}.", "{w}"), F("{w}. {w}!", "{w}")],
    "frame": [],
}
INTRO_WORD = {                                                             # words whose class frames do not fit them
    "fell": [F("uh oh. it {w}.", "{w}"), F("the {o} {w}.", "{w}"), F("oh! it {w}!", "{w}")],
    "done": [F("all {w}.", "{w}"), F("{w}!", "{w}"), F("oh. {w}.", "{w}")],
    "that": [F("look at {w}.", "{w}"), F("what is {w}?", "{w}"), F("see {w}?", "{w}")],
    "can": [F("you {w}!", "{w}"), F("{n} {w}.", "{w}"), F("you {w}?", "{w}")],
    "too": [F("you {w}.", "{w}"), F("mama {w}!", "{w}"), F("{n} {w}?", "{w}")],
    "now": [F("look {w}.", "{w}"), F("up {w}!", "{w}"), F("sit {w}?", "{w}")],
    "again": [F("{w}!", "{w}"), F("roll {w}.", "{w}"), F("look {w}?", "{w}")],
    "out": [F("{w}!", "{w}"), F("it is {w}.", "{w}"), F("{w}. {w}.", "{w}")],
    "off": [F("{w}!", "{w}"), F("it is {w}.", "{w}"), F("{w}. {w}.", "{w}")],
    "under": [F("{w}!", "{w}"), F("it is {w}.", "{w}"), F("look. {w}.", "{w}")],
    "thanks": [F("{w}!", "{w}"), F("oh. {w}.", "{w}"), F("yes. {w}.", "{w}")],
    "please": [F("{w}.", "{w}"), F("more? {w}.", "{w}"), F("give me. {w}.", "{w}")],
    "all": [F("it is {w}.", "{w}"), F("look. {w}.", "{w}"), F("{w}!", "{w}")],
    "eyes": [F("look. {w}.", "{w}"), F("see? {w}!", "{w}"), F("look at the {w}.", "{w}")],
}

_SLOT = re.compile(r"\{([a-z])\}")
_FORM = re.compile(r"[a-z]+(?:[.?!]? [a-z]+)*[.?!]")
_WORD = re.compile(r"[a-z]+")


def words(text):
    return _WORD.findall(text)


def _last_is_focus(text, focus):
    if focus is None:
        return True
    return words(text)[-1] == (focus if not focus.startswith("{") else focus)


for _k, _fs in list(FRAMES.items()) + list(INTRO.items()) + list(INTRO_WORD.items()):
    for _t, _f in _fs:
        assert _f is None or words(_t.replace("{", "").replace("}", ""))[-1] == _f.strip("{}"), (_k, _t, _f)


@dataclass(frozen=True)
class Line:
    text: str
    intent: str
    register: str = "plain"
    focus: str = None               # the focus word (said last), or None
    emphasis: str = None            # the word the voice puts on a pitch peak and lengthens (4.4): the focus
    refs: tuple = ()                # the object ids its slots were filled from
    source: str = "fast"            # "fast" (the templates) or "claude" (a steering row's line)
    frame: str = ""                 # the frame it was filled from


# ------------------------------------------------------------------------------------------------------ the line check
def check(text, vocab, new_word=None, percept=None, refs=(), held=K.HELD_PAIRS, source="fast", register=None,
          recent_events=()):
    """-> (ok, reason). vocab: the words she has (the birth words and the growth words entered); new_word: the day's new word
    (allowed, and only last); percept: what she perceives now (body/sim/lang/percept.Percept; None skips the perceived rule:
    for listing lines ahead, never for a line she says); refs: the object ids a line's slots were filled from; held: the
    never-taught pairs still held out; source: "fast" or "claude"; recent_events: the (tick, kind, object) she saw in the last
    RECENT ticks."""
    if not isinstance(text, str) or not _FORM.fullmatch(text):
        return False, "form: lowercase words, single spaces, '. ? !' only after a word, ending in one"
    ws = words(text)
    if len(ws) > K.MAX_WORDS:
        return False, f"too long: {len(ws)} words"
    oov = [w for w in ws if w not in vocab and w != new_word]
    if oov:
        return False, f"not her word: {oov[0]!r}"
    if new_word is not None and new_word in ws and ws[-1] != new_word:
        return False, f"the new word {new_word!r} not last"
    wset = set(ws)
    for a, b in held:
        if a in wset and b in wset:
            return False, f"held-out pair ({a}, {b})"
    if source == "claude":
        praise = [w for w in ws if w in K.PRAISE]
        if praise or register == "approval":
            return False, f"praise is the fast layer's: {praise[0] if praise else 'the approval register'!r}"
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


def frames_for(intent):
    return FRAMES[intent]


def intro_frames(word):
    return INTRO_WORD.get(word, INTRO[GROWTH_CLASS[word]])


def showable(word, world):
    """can a growth word enter (A15)? world: dict(objects={name: [colours]}, fixtures=set, acts=set, events=set, face=set).
    -> (ok, why not)."""
    if GROWTH_CLASS.get(word) == "frame":
        return False, "no infant-directed line ends on it (4.8's rule: sentence-final)"
    need = NEEDS.get(word, ("never", "no need listed"))
    kind = need[0]
    if kind == "never":
        return False, need[1]
    if kind == "child":
        return True, ""
    arg = need[1]
    have = {"obj": set(world.get("objects", {})), "fixture": set(world.get("fixtures", ())), "act": set(world.get("acts", ())),
            "event": set(world.get("events", ())), "face": set(world.get("face", ()))}
    if kind == "twins":
        n = sum(1 for cs in world.get("objects", {}).values() for c in cs if c == arg)
        return (n >= 2, "" if n >= 2 else f"fewer than two {arg} toys (B2's colour twins)")
    if kind == "face":
        return (bool(have["face"]) if arg == "any" else arg in have["face"]), f"her face shows no {arg}"
    return (arg in have[kind], f"no {kind} {arg!r} in the world")


def birth_lines(vocab=BIRTH_WORDS, objects=None, fixtures=ROOM):
    """every line the templates can make at birth, as (text, register, emphasis), sorted: what the voice makes ahead at a night
    boundary (4.4). objects: [(name, colour, on)] (default: each birth toy on the mat and on the sofa, no colours)."""
    from .conduct import INTENTS, register_for                        # noqa: PLC0415 (conduct imports this module)
    from .percept import Seen
    objs = objects or [(t, "", p) for t in TOYS for p in ("mat", "sofa")]
    out = set()
    for intent, frames in FRAMES.items():
        if intent not in INTENTS:
            continue
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
