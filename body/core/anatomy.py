"""the body's anatomy, declared (docs/SIM_DESIGN.md section 8.2; the core refactor, steps R1 to R3): what a body senses (`Channel`), how it
acts (`Effector`) and what it feels as reward (`RewardSource`), gathered in an `Anatomy`, so that the core can serve a body other than
the diary's. `LanguageAnatomy(tok, cfg)` is the diary's body: it derives from the tokenizer and the physiology exactly the symbols
`Life.__init__` derived before step R2 and now reads from it (the rest `sil`, the display symbol `nl`, the space, the turn-end token
`eot`, the end the offset teaches `end_id`, the `reserved` the world never types and the `bans` the mouth never says) and declares the
ear, the face, the voice and the reward's three sources in their order (`FaceReward`, `WorldWordsReward`, `EffortReward`).

STEP R1 declared it; STEP R2 builds the life with it: `Life(organs, tok)`, `Life.birth(tok)` and `Life.load(path, tok)` keep their
signatures, and `anatomy_for` turns the tokenizer into the diary's `LanguageAnatomy` inside (or takes an anatomy given in its place). The
life reads its symbols from the anatomy (`life.anatomy`), and the tokenizer stays inside the language anatomy for text: the world's
typing (`symbol`), the page, the tick's record and the night's report (`decode`). STEP R3 wires the reward sources: the tick's felt
reward is the anatomy's sources, felt in their declared order on the tick's frame (body/core/world.py) and added one at a time in that
order (`_sense`, body/core/senses.py), which is the float order the reward was summed in before. body/tests/test_anatomy.py holds the
language anatomy equal to today's fields and its reward equal to today's rule. The tables, the encoder and the gate are the organs'
(`m.E`, `m.face_in`, `m.mouth_gate`) and stay None until a later step binds them; the methods that read a frame or a life raise until the
step named on them wires them. Building an anatomy builds no module, draws no random number and touches no life (SIM_DESIGN.md 8.3,
item 4); a reward source keeps no state of its own (the face's held level is the life's `level`, as before)."""
from dataclasses import dataclass, field
from typing import Optional

from .physiology import PHYSIOLOGY

KINDS = ("symbol", "vector")


@dataclass(eq=False)
class Channel:
    """ONE SENSE. kind "symbol": an alphabet of `size` symbols, each a fixed unit row of `table` [size, d] (the language ear's table is
    the lexicon, m.E); kind "vector": an observation of `size` numbers through `encoder` (the language face: the face and its change,
    each over 6, through m.face_in, a learned map born at zero). `rest_id` is the symbol of the channel's quiet, `end_id` the symbol
    the offset teaches the cortex to expect when the source's turn ends, `reserved` the symbols the world never sends on it. `partner`
    marks the turn-taking source (at most one channel of an anatomy)."""
    name: str
    kind: str
    size: int
    table: object = None                  # Tensor [size, d] (symbol channels), bound by the life
    encoder: object = None                # Module (vector channels), bound by the life
    rest_id: Optional[int] = None
    end_id: Optional[int] = None
    reserved: list = field(default_factory=list)
    partner: bool = False

    def encode(self, obs):
        """the observation as the cortex's input, [d] (the frames of step R4)"""
        raise NotImplementedError("Channel.encode: the frames are wired in step R4 (SIM_DESIGN.md 8.4)")


@dataclass(eq=False)
class Effector:
    """ONE WAY OF ACTING. `factors` is the shape of its alphabet: [K] one choice among K acts (the voice: the lexicon), [5, 5] two
    joints of five settings each (an act's row the sum of its joints' rows). `table` holds the acts' rows (the voice shares the ear's
    lexicon); `rest_id` is the act of doing nothing, `end_id` the act that closes a chunk (the voice: the space, the word's end where
    the planning actor decides), `reserved` the acts it never makes (the voice: today's `bans`). `gate` decides whether to act (the
    voice's is m.mouth_gate; its optimizer `opt_gate` and its buffer `gate_buf` keep their names). Effector 0 is the voice."""
    name: str
    factors: list
    rest_id: Optional[int] = None
    end_id: Optional[int] = None
    reserved: list = field(default_factory=list)
    table: object = None                  # Tensor [K, d], bound by the life
    gate: object = None                   # Module, bound by the life

    def gate_inputs(self, frame, life):
        """the gate's own "ear": what it reads besides the stream (the effectors of step R5)"""
        raise NotImplementedError("Effector.gate_inputs: the effectors are wired in step R5 (SIM_DESIGN.md 8.4)")

    def cost(self, act, frame):
        """the effort of an act (the effectors of step R5)"""
        raise NotImplementedError("Effector.cost: the effectors are wired in step R5 (SIM_DESIGN.md 8.4)")


@dataclass(eq=False)
class RewardSource:
    """ONE TERM OF THE FELT REWARD (step R3). Each tick a source feels (`felt`: a number, or None when it is silent this tick) and its
    feeling enters the reward as its term (`term`: clipped to +-`clip`, like a press, when a clip is declared; else the feeling as it
    is). An anatomy's sources are felt in their declared order and their terms added one at a time in that order (the float order of
    the sum); a silent source adds nothing, not even a zero. SOURCE 0 IS THE WORLD'S JUDGMENT (the diary's and the sim's: the face): it
    always answers, its feeling is the tick's felt event (the striatum's face event, the tick's record `felt`), and its term alone is
    the reward the anticipation ring and the actor's reliability read, before the later terms are added. `keys` names the physiology
    constants the source reads (none: always on); it reads them from the life's constants every tick, as the reward always did, so a
    constant changed on a living body is felt at the next tick. A subclass declares the feeling; a source keeps no state of its own."""
    name: str
    keys: tuple = ()
    clip: Optional[float] = None

    def felt(self, frame, life):
        """this tick's feeling on the world's `frame` (body/core/world.py) and the `life`'s state, or None (silent: nothing is added)"""
        raise NotImplementedError(f"RewardSource.felt: the source {self.name!r} declares no feeling (a reward source is a subclass that does)")

    def term(self, v):
        """the feeling `v` as the reward's term"""
        return v if self.clip is None else float(max(-self.clip, min(self.clip, v)))


class FaceReward(RewardSource):
    """THE FACE AS REWARD (the world's judgment; SIM_DESIGN.md 5.6, item 1). The level the world's face shows (`frame.face`, to whole
    steps within +-6) against the level the body holds (`life.level`, the life's since before the refactor): a rise in the face's size,
    or a change of its sign, is felt as the new level; a held face is silence, and its easing off is no event (the held level follows it,
    unfelt). Always answers (0 when nothing is felt). The diary declares it with clip 2, the term clipped like a press."""

    def felt(self, frame, life):
        lvl = max(-6, min(6, int(frame.face)))
        felt = 0
        if lvl != life.level:
            if abs(lvl) > abs(life.level) or lvl * life.level < 0:
                felt = lvl
            life.level = lvl
        return felt


class WorldWordsReward(RewardSource):
    """THE WORLD'S WORDS AS REWARD (world_r; 0 = off): each symbol the partner sends on the ear (`frame.obs["ear"]`, not its rest) is
    felt at world_r beside the face; under world_mask, the corollary discharge, not on a tick after the voice acted (not heard over its
    own voice). Silent otherwise."""

    def felt(self, frame, life):
        if frame.obs["ear"] != life.sil:
            wr = float(life.cfg.get("world_r", 0.0))
            if wr and not (int(life.cfg.get("world_mask", 0)) and getattr(life, "_acted_last", False)):
                return wr
        return None


class EffortReward(RewardSource):
    """THE EFFORT IN THE REWARD (cost_in_reward; 0 on the served body): the cost of the last act, symbol_cost x (1 + (fatigue /
    gate_fatigue)^2), is felt as the next tick's reward, negative, so both critics predict it and the gate reads their error alone.
    Added to the act's credit outside the critics (the earlier form, with a tonic drive of 0.25 cancelling it) it was never predicted
    away and, with the drive gone, held every act at a loss; with both gone the gate saturated at 0.98 (runs 69-72). Silent on a tick
    after no act. The feeling is the cost negated: adding it is the IEEE subtraction of the cost (x + (-c) is x - c, bit for bit)."""

    def felt(self, frame, life):
        if life.cfg.get("cost_in_reward") and getattr(life, "_acted_last", False):
            return -(float(life.cfg["symbol_cost"]) * (1.0 + (life.fatigue / float(life.cfg["gate_fatigue"])) ** 2))
        return None


class Anatomy:
    """a body's senses, effectors and reward sources, each list in its order: the channels' order is the float order of the cortex's
    input sum, effector 0 is the voice, the reward sources are summed in their order and source 0 is the world's judgment"""

    def __init__(self, channels, effectors, rewards):
        self.channels = list(channels)
        self.effectors = list(effectors)
        self.rewards = list(rewards)

    def channel(self, name):
        return next(c for c in self.channels if c.name == name)

    def effector(self, name):
        return next(e for e in self.effectors if e.name == name)

    @property
    def partner(self):
        """the turn-taking source's channel, or None"""
        return next((c for c in self.channels if c.partner), None)

    def check(self):
        """the declaration's own consistency (an error names the first fault); returns the anatomy"""
        for what, xs in (("channel", self.channels), ("effector", self.effectors), ("reward source", self.rewards)):
            names = [x.name for x in xs]
            if len(set(names)) != len(names):
                raise ValueError(f"anatomy: a {what} name is declared twice: {names}")
        if sum(1 for c in self.channels if c.partner) > 1:
            raise ValueError("anatomy: more than one partner channel")
        for c in self.channels:
            if c.kind not in KINDS:
                raise ValueError(f"anatomy: channel {c.name!r} of kind {c.kind!r}, not one of {KINDS}")
            if int(c.size) < 1:
                raise ValueError(f"anatomy: channel {c.name!r} of size {c.size}")
            if c.kind == "symbol":
                ids = [i for i in [c.rest_id, c.end_id, *c.reserved] if i is not None]
                if any(not 0 <= int(i) < int(c.size) for i in ids):
                    raise ValueError(f"anatomy: channel {c.name!r} declares a symbol outside its {c.size}")
                if c.rest_id is not None and c.rest_id in c.reserved:
                    raise ValueError(f"anatomy: channel {c.name!r}'s rest is reserved")
        if not self.effectors:
            raise ValueError("anatomy: no effector (effector 0 is the voice)")
        for e in self.effectors:
            if not e.factors or any(int(k) < 1 for k in e.factors):
                raise ValueError(f"anatomy: effector {e.name!r} of factors {e.factors}")
            n = 1
            for k in e.factors:
                n *= int(k)
            ids = [i for i in [e.rest_id, e.end_id, *e.reserved] if i is not None]
            if any(not 0 <= int(i) < n for i in ids):
                raise ValueError(f"anatomy: effector {e.name!r} declares an act outside its {n}")
            if e.rest_id is not None and e.rest_id in e.reserved:
                raise ValueError(f"anatomy: effector {e.name!r}'s rest is reserved")
        if not self.rewards:
            raise ValueError("anatomy: no reward source (source 0 is the world's judgment)")
        for s in self.rewards:
            unknown = [k for k in s.keys if k not in PHYSIOLOGY]
            if unknown:
                raise ValueError(f"anatomy: reward source {s.name!r} reads constants the physiology does not know: {unknown}")
            if s.clip is not None and not s.clip > 0:
                raise ValueError(f"anatomy: reward source {s.name!r} clipped to +-{s.clip}")
        return self


class LanguageAnatomy(Anatomy):
    """THE DIARY'S BODY: one ear on the typed page (the partner), the face as a sense, the voice as its one effector, the reward felt
    from the face (the judgment, clipped to +-2), then the world's words (world_r, not over its own voice under world_mask), then the
    effort (under cost_in_reward; 0 on the served body), summed in that order as `_sense` summed them before step R3. The symbols are derived here as `Life.__init__` derived them before step R2 (the life now copies them from
    its anatomy), from `tok` and the physiology updated by `cfg`, so they are declared by the body that is born, never found by a fixed
    string."""

    def __init__(self, tok, cfg=None):
        c = dict(PHYSIOLOGY); c.update(cfg or {})
        self.tok = tok
        self.vocab = int(tok.get_vocab_size())
        # the body's own symbols: its rest, a display symbol the world never types, the word boundary, the world's turn-end,
        # and what the offset teaches the cortex to expect
        self.sil = tok.token_to_id(str(c.get("rest_token", "<pad>")))
        self.nl = tok.token_to_id(str(c.get("display_token", "\n")))
        self.space_id = tok.token_to_id(" ")
        self.eot = tok.token_to_id(str(c.get("end_token", "<eot_human>")))
        self.end_id = self.sil if str(c.get("end_symbol", "eot")) == "rest" else self.eot
        # the reserved symbols: every `<...>` the tokenizer defines except the rest, and the display symbol
        _vocab = tok.get_vocab(); _specials = sorted(i for s_, i in _vocab.items() if s_.startswith("<") and s_.endswith(">"))
        self.reserved = [i for i in _specials if i != self.sil] + ([self.nl] if self.nl is not None else [])
        self.bans = list(self.reserved)
        ear = Channel("ear", "symbol", self.vocab, rest_id=self.sil, end_id=self.end_id, reserved=self.reserved, partner=True)
        face = Channel("face", "vector", 2)
        voice = Effector("voice", [self.vocab], rest_id=self.sil, end_id=self.space_id, reserved=self.bans)
        rewards = [FaceReward("face", clip=2),
                   WorldWordsReward("world_r", ("world_r", "world_mask")),
                   EffortReward("cost", ("cost_in_reward", "symbol_cost", "gate_fatigue"))]
        super().__init__([ear, face], [voice], rewards)

    # the tokenizer stays for text (step R2): the language body's only readers of it
    def symbol(self, ch):
        """the ear's symbol for one typed character, or None where the alphabet has none (the world's hand, `type_text`; the corpus
        as the body hears it)"""
        return self.tok.token_to_id(ch)

    def decode(self, ids):
        """symbols as text, as the tokenizer writes them (the page, the tick's record, the night's report)"""
        return self.tok.decode(ids)

    def symbols(self):
        """the symbols this anatomy declares, in one tuple (two language anatomies of one body agree on it)"""
        return (self.vocab, self.sil, self.nl, self.space_id, self.eot, self.end_id, tuple(self.reserved), tuple(self.bans))


def anatomy_for(body, cfg=None):
    """THE ANATOMY A LIFE IS BUILT WITH (step R2; SIM_DESIGN.md 8.2, "the signatures stay"): `body` is what the caller passed where the
    tokenizer always went. A tokenizer gives the diary's `LanguageAnatomy` under the constants `cfg` (updating the physiology, as the
    life's own cfg does). A language anatomy given in its place is taken if it declares the symbols its tokenizer gives under `cfg`,
    since a body's rest and ends are named by its physiology (one declared under other constants is refused, not mixed). Any other
    anatomy is refused until steps R4-R5 wire channels and effectors into the core: until then the life reads the language symbols."""
    if not isinstance(body, Anatomy):
        return LanguageAnatomy(body, cfg)
    if not isinstance(body, LanguageAnatomy):
        raise NotImplementedError(f"anatomy_for: the core serves the language anatomy only until the core refactor's steps R4-R5 wire "
                                  f"channels and effectors (SIM_DESIGN.md 8.4); given {type(body).__name__}")
    want = LanguageAnatomy(body.tok, cfg).symbols()
    if want != body.symbols():
        raise ValueError(f"anatomy_for: this language anatomy was declared under other constants than the life's (its rest, display, "
                         f"space, turn-end and end {body.symbols()[1:6]}; the life's {want[1:6]})")
    return body
