"""the body's anatomy, declared (docs/SIM_DESIGN.md section 8.2; the core refactor, step R1): what a body senses (`Channel`), how it
acts (`Effector`) and what it feels as reward (`RewardSource`), gathered in an `Anatomy`, so that the core can serve a body other than
the diary's. `LanguageAnatomy(tok, cfg)` is the diary's body: it rebuilds from the tokenizer and the physiology exactly the symbols
`Life.__init__` derives today (body/life.py: the rest `sil`, the display symbol `nl`, the space, the turn-end token `eot`, the end the
offset teaches `end_id`, the `reserved` the world never types and the `bans` the mouth never says) and declares the ear, the face, the
voice and the reward's three terms in their order.

STEP R1: declarations only. `Life` does not use this module yet; body/tests/test_anatomy.py holds the language anatomy equal to the
fields a life derives from its tokenizer. The tables, the encoder and the gate are the organs' (`m.E`, `m.face_in`, `m.mouth_gate`) and
stay None until a life binds them; the methods that read a frame or a life raise until the step named on them wires them. Building an
anatomy builds no module, draws no random number and touches no life (SIM_DESIGN.md 8.3, item 4)."""
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
    """ONE TERM OF THE FELT REWARD; an anatomy's sources are summed in their declared order, one at a time (the float order of the
    sum). `keys` names the physiology constants the term reads (none: always on)."""
    name: str
    keys: tuple = ()

    def felt(self, frame, life):
        """this tick's term (the reward sources of step R3)"""
        raise NotImplementedError("RewardSource.felt: the reward sources are wired in step R3 (SIM_DESIGN.md 8.4)")


class Anatomy:
    """a body's senses, effectors and reward sources, each list in its order: the channels' order is the float order of the cortex's
    input sum, effector 0 is the voice, the reward sources are summed in their order"""

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
        for s in self.rewards:
            unknown = [k for k in s.keys if k not in PHYSIOLOGY]
            if unknown:
                raise ValueError(f"anatomy: reward source {s.name!r} reads constants the physiology does not know: {unknown}")
        return self


class LanguageAnatomy(Anatomy):
    """THE DIARY'S BODY: one ear on the typed page (the partner), the face as a sense, the voice as its one effector, the reward felt
    from the face, then the world's words (world_r, not over its own voice under world_mask), then the effort (under cost_in_reward;
    0 on the served body). The symbols are derived as `Life.__init__` derives them (body/life.py), from `tok` and the physiology
    updated by `cfg`, so they are declared by the body that is born, never found by a fixed string."""

    def __init__(self, tok, cfg=None):
        c = dict(PHYSIOLOGY); c.update(cfg or {})
        self.tok = tok
        self.vocab = int(tok.get_vocab_size())
        # the body's own symbols (life.py): its rest, a display symbol the world never types, the word boundary, the world's turn-end,
        # and what the offset teaches the cortex to expect
        self.sil = tok.token_to_id(str(c.get("rest_token", "<pad>")))
        self.nl = tok.token_to_id(str(c.get("display_token", "\n")))
        self.space_id = tok.token_to_id(" ")
        self.eot = tok.token_to_id(str(c.get("end_token", "<eot_human>")))
        self.end_id = self.sil if str(c.get("end_symbol", "eot")) == "rest" else self.eot
        # the reserved symbols (life.py): every `<...>` the tokenizer defines except the rest, and the display symbol
        _vocab = tok.get_vocab(); _specials = sorted(i for s_, i in _vocab.items() if s_.startswith("<") and s_.endswith(">"))
        self.reserved = [i for i in _specials if i != self.sil] + ([self.nl] if self.nl is not None else [])
        self.bans = list(self.reserved)
        ear = Channel("ear", "symbol", self.vocab, rest_id=self.sil, end_id=self.end_id, reserved=self.reserved, partner=True)
        face = Channel("face", "vector", 2)
        voice = Effector("voice", [self.vocab], rest_id=self.sil, end_id=self.space_id, reserved=self.bans)
        rewards = [RewardSource("face"),
                   RewardSource("world_r", ("world_r", "world_mask")),
                   RewardSource("cost", ("cost_in_reward", "symbol_cost", "gate_fatigue"))]
        super().__init__([ear, face], [voice], rewards)
