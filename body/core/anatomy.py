"""the body's anatomy, declared (docs/SIM_DESIGN.md section 8.2; the core refactor, steps R1 to R5 and R9): what a body senses
(`Channel`), how it acts (`Effector`) and what it feels as reward (`RewardSource`), gathered in an `Anatomy`, so that the core can
serve a body other than the diary's. `LanguageAnatomy(tok, cfg)` is the diary's body: it derives from the tokenizer and the physiology exactly the symbols
`Life.__init__` derived before step R2 and now reads from it (the rest `sil`, the display symbol `nl`, the space, the turn-end token
`eot`, the end the offset teaches `end_id`, the `reserved` the world never types and the `bans` the mouth never says) and declares the
ear (`EarChannel`), the face (`FaceChannel`), the voice (`VoiceEffector`) and the reward's three sources in their order (`FaceReward`,
`WorldWordsReward`, `EffortReward`).

STEP R1 declared it; STEP R2 builds the life with it: `Life(organs, tok)`, `Life.birth(tok)` and `Life.load(path, tok)` keep their
signatures, and `anatomy_for` turns the tokenizer into the diary's `LanguageAnatomy` inside (or takes an anatomy given in its place). The
life reads its symbols from the anatomy (`life.anatomy`), and the tokenizer stays inside the language anatomy for text: the world's
typing (`symbol`), the page, the tick's record and the night's report (`decode`). STEP R3 wires the reward sources: the tick's felt
reward is the anatomy's sources, felt in their declared order on the tick's frame (body/core/world.py) and added one at a time in that
order (`_sense`, body/core/senses.py), which is the float order the reward was summed in before. STEP R4 wires the channels: each
position of the cortex's window holds every channel's observation under the channel's `field` (the diary's "x" and "face", the keys
the window always had); the cortex's input is the channels' codes summed one at a time in the declared order (`Organs.inputs`, the
ladder's bundle and the efference copy joining after the first `inner_at` channels: ear + face + bundle + own for the diary, the float
order of the sum before); a channel's code is made by the organ it names (`organ`: the ear's the lexicon `E`, the face's the learned
`face_in`, born at zero); channel 0 is the words, whose forecast head is `latent_pred`, and a later channel that declares a forecast
has a head of its own, built by the organs after every other organ (`Organs(..., channels=)`) and taught by the waking lesson.
STEP R5 wires the effectors: effector 0 is the voice (`VoiceEffector`), naming the organs it always had (the lexicon E it shares with the
ear, mouth_gate with its opt_gate and gate_buf, actor, its act the window's "xo"); its gate's own inputs are its ear (`gate_inputs`, the
life's `_voice_ear`) and its choice, act and lesson are today's code, its draws on self.gen the tick's first. A later effector
(`Effector`) names the organs the organs build for it after every other organ (`Organs(..., effectors=)`: `acts.<name>`, a fixed table of
unit rows per joint from the body's seed; `gates.<name>`, born as the voice's gate; `actors.<name>`, sized with the striatum, whose
event block for its acts is appended after the language block) and the window field of its act; each tick it chooses after the voice
(`_choose_effector`: its gate over the shared inputs and its own `gate_inputs`, its draw, then the per-joint readout of its proposal
with the striatal actor's bias per joint, each joint drawn in turn), acts (`_act_effectors`: its act's row enters the cortex's input
after the voice's own sound, its cost the body's fatigue, its act its striatal line's event) and learns (`_gate_lesson(i)`, the
voice's lesson on its own gate, buffer and baseline; its actor as the voice's). Its proposal head (act_pred) is step R6; until then it
proposes nothing and the actor's bias alone shapes its draws. The defect fixes 4, 5 and 8 are switches (physiology.py `SWITCHES`), off
by their absence. STEP R9 wires the world (body/core/world.py): a channel that declares no observation of its own is a channel of the
world's frames (`Channel.observe`: its observation in the frame the tick is lived on, `life.world.now`, under its name), and an
effector's gate inputs and cost read that frame. STEP R6 wires the motor timing part (SIM_DESIGN.md 5.4 and 5.8; body/core/timing.py): a
later effector's proposal is its act_pred head's (`propose`), corrected by the error of the forward half when it declares a body sense
(`sense`, a vector channel, and `sense_idx`, its own numbers there); it may declare an inverse model (`inverse`, act_inv, over its
sense) and a reflex (`reflex`, spinal: the act it forces this tick, which stops a chunk); its chunks and learned stops run under
chunk_gate. The voice declares none of it. body/tests/test_anatomy.py holds the language anatomy equal to today's fields, its
reward equal to today's rule, its input sum, window and heads equal to today's, and its gate's lesson equal to today's. The anatomy
names the organs and never holds them (no module, no tensor: the organs are the body's and are saved with it).
Building an anatomy builds no module, draws no random number and touches no life (SIM_DESIGN.md 8.3, item 4); a channel and a reward
source keep no state of their own (the face's held level is the life's `level`, its last face the life's `face_prev`, as before)."""
from dataclasses import dataclass, field as dc_field
from typing import Optional

import torch

from .physiology import PHYSIOLOGY

KINDS = ("symbol", "vector")
CORE_FIELDS = ("xo", "bundle", "read", "r", "end")    # the keys the core writes into a window position beside the channels' own fields


@dataclass(eq=False)
class Channel:
    """ONE SENSE. kind "symbol": an alphabet of `size` symbols, each a fixed unit row of a table [size, d] (the language ear's table is
    the lexicon, m.E); kind "vector": an observation of `size` numbers through an encoder (the language face: the face and its change,
    each over 6, through m.face_in, a learned map born at zero). `rest_id` is the symbol of the channel's quiet, `end_id` the symbol
    the offset teaches the cortex to expect when the source's turn ends, `reserved` the symbols the world never sends on it. `partner`
    marks the turn-taking source (at most one channel of an anatomy, and it is channel 0).
    Step R4: `organ` names the organs' module that makes the channel's code, the cortex's input term (a symbol channel's table, a
    vector channel's encoder): the anatomy names it and never holds it. `field` is the key under which each position of the cortex's
    window holds the channel's observation (by default the channel's name; the diary's ear "x" and face "face", the window's keys
    before R4). `forecast` declares a forecast head: channel 0 is the words, and its head is `latent_pred`; a later channel's head is
    its own (m.chan_pred[name]), foreseeing the channel's code at the next position."""
    name: str
    kind: str
    size: int
    organ: Optional[str] = None           # the name of the organs' module that encodes it (step R4)
    field: Optional[str] = None           # the window position's key for its observation (default: the name)
    forecast: bool = False                # a forecast head of its own (channel 0's is latent_pred)
    rest_id: Optional[int] = None
    end_id: Optional[int] = None
    reserved: list = dc_field(default_factory=list)
    partner: bool = False

    def __post_init__(self):
        if self.field is None:
            self.field = self.name

    def encode(self, m, obs):
        """the observation's code, the cortex's input term (step R4): the organ the channel names, among the organs `m`, applied to it
        (a symbol channel: [..] symbols -> [.., d] rows; a vector channel: [.., size] -> [.., d])"""
        return getattr(m, self.organ)(obs)

    def quiet(self, shape, device=None):
        """the channel's observation of nothing over `shape` positions (step R4): a symbol channel's rest, a vector channel's zeros. A
        dream's channels other than the words are quiet (the face a dream has always had: zeros)."""
        if self.kind == "symbol":
            if self.rest_id is None:
                raise ValueError(f"channel {self.name!r}: a symbol channel without a rest has no quiet")
            return torch.full(tuple(shape), int(self.rest_id), dtype=torch.long, device=device)
        return torch.zeros(*shape, int(self.size), device=device)

    def observe(self, life, x, who, still=False):
        """the channel's observation at the window position a step of the tick opens (`x` the symbol the step enters, `who` 0 the
        world's, 1 its own; `still`: an imagined position, nothing changing). The diary's channels declare their own (the ear, the face).
        A CHANNEL OF THE WORLD'S FRAMES (step R9, the world loop): its observation in the frame the tick is lived on (`life.world.now`,
        the one the tick's senses took), under the channel's name, the same at both halves of the tick and held at an imagined position;
        its quiet where there is no frame yet or the frame names it not. A symbol channel's observation is its symbol; a vector
        channel's its `size` numbers, as float32 on the life's device."""
        f_ = getattr(life.world, "now", None)
        o_ = None if f_ is None else f_.obs.get(self.name)
        if self.kind == "symbol":
            if o_ is None:
                if self.rest_id is None:
                    raise ValueError(f"channel {self.name!r}: a symbol channel without a rest has no quiet")
                return int(self.rest_id)
            return int(o_)
        if o_ is None:
            return torch.zeros(int(self.size), device=life.dev)
        return torch.as_tensor(o_, dtype=torch.float32, device=life.dev).reshape(int(self.size))


class EarChannel(Channel):
    """THE DIARY'S EAR (the partner, channel 0: the words): at a position the world's symbol opens, that symbol; at a position its own
    sound opens (the world was quiet), the ear's rest. Its code is the lexicon's row (organ E)."""

    def observe(self, life, x, who, still=False):
        return int(x) if who == 0 else self.rest_id


class FaceChannel(Channel):
    """THE DIARY'S FACE AS A SENSE: [face/6, its change since the last tick/6] (the change 0 at an imagined position, the face held), the
    same at both halves of a tick; its code through the learned face_in, born at zero."""

    def observe(self, life, x, who, still=False):
        return torch.tensor([life.face_now / 6.0, 0.0 if still else (life.face_now - life.face_prev) / 6.0], device=life.dev)


@dataclass(eq=False)
class Effector:
    """ONE WAY OF ACTING. `factors` is the shape of its alphabet: [K] one choice among K acts (the voice: the lexicon), [5, 5] two
    joints of five settings each. An act is one flat id, its joints' settings as mixed-radix digits (joint 0 the most significant: the
    act (a, b) of [5, 5] is 5a + b), and its row is the sum of its joints' rows. `rest_id` is the act of doing nothing (its gate's no,
    or a draw of it), `end_id` the act that closes a chunk (the voice: the space, the word's end where the planning actor decides),
    `reserved` the acts it never makes (the voice: today's `bans`; only a one-joint alphabet can reserve acts, since a factored one
    draws its joints apart). Effector 0 is the voice (`VoiceEffector`). THE FLAT ACT'S LIMIT (the R6 verifier, 2026-09-24; said here, not
    checked): the window, the striatal line and the table read a flat act as an int64 (torch.long), so one effector holds at most 27
    joints of five settings (5^27 < 2^63 <= 5^28; the 28th overflows: torch.tensor refuses the id, a flat act built of tensors wraps
    silently); the humanoid's widest limb has 7 (78125 acts). A wider limb is two effectors, as the design's limbs are.
    Step R5: an effector names its organs and never holds them (the anatomy stays stateless, as the steps before kept it): `organ` its
    acts' table (the voice's the lexicon E, the table it shares with the ear; a later effector's `acts.<name>`, fixed unit rows born
    from the body's seed), `gate` its gate (the voice's m.mouth_gate, whose optimizer `opt_gate` and buffer `gate_buf` keep their
    names; a later effector's `gates.<name>`), `actor` its striatal actor head (the voice's m.actor; a later effector's
    `actors.<name>`), `field` the window position's key for its act, the efference copy (the voice's the core's own "xo"; a later
    effector's its name). The organs build a later effector's organs under those names after every other organ (Organs(...,
    effectors=)). `n_in` is the number of its gate's own inputs beside the shared [C/sqrt(d), fatigue, mood, stress, salience, level]
    (the base effector's one: its own act last tick), `effort` the fatigue an act costs (the base's cost; a body's effector may
    declare a cost of its own; under cost_in_reward it reaches neither the reward nor the gate's lesson: see EffortReward). 8.2
    sketched gate_inputs(frame, life) and cost(act, frame): the effector's working state and the life are passed as well, since the
    anatomy keeps no state of its own and the constants are the life's. The frame is the one the tick is lived on (step R9: the
    world's; a gate's own input may read the world's observations there).
    Step R6, THE MOTOR TIMING PART (SIM_DESIGN.md 5.4 and 5.8; body/core/timing.py), built for every later effector by the organs
    (timing.<name>) and declared here: `sense` names a vector channel that holds its body sense (the arm's joint angles and
    velocities) and `sense_idx` its own numbers in it (None: all of them); the forward half then foresees that sense at the next
    position and the error of its forecast corrects the proposal. `inverse` gives it act_inv, a small network of `inv_hidden` units
    from its sense at t and t+1 to its per-joint act, learning online from its own acts (the grip and the gaze have none at birth).
    `propose` is act_pred's proposal; `reflex` is its spinal reflex (none by default), the act it forces on the world this tick,
    which stops a chunk and is sensed, never heard as its own act."""
    name: str
    factors: list
    rest_id: Optional[int] = None
    end_id: Optional[int] = None
    reserved: list = dc_field(default_factory=list)
    organ: Optional[str] = None           # its acts' table (default acts.<name>)
    gate: Optional[str] = None            # its gate (default gates.<name>)
    actor: Optional[str] = None           # its striatal actor head (default actors.<name>)
    field: Optional[str] = None           # the window position's key for its act (default its name)
    n_in: int = 1                         # its gate's own inputs (the base: its own act last tick)
    effort: float = 0.0                   # the fatigue an act costs (the base's cost)
    sense: Optional[str] = None           # step R6: the vector channel of its body sense (None: no forward half)
    sense_idx: Optional[list] = None      # its own numbers in that channel (None: all)
    inverse: bool = False                 # an inverse model (act_inv) from birth, over its sense
    inv_hidden: int = 64                  # act_inv's hidden units

    def __post_init__(self):
        if self.organ is None:
            self.organ = f"acts.{self.name}"
        if self.gate is None:
            self.gate = f"gates.{self.name}"
        if self.actor is None:
            self.actor = f"actors.{self.name}"
        if self.field is None:
            self.field = self.name

    @property
    def n_acts(self):
        """the size of its alphabet: the product of its joints' settings"""
        n = 1
        for k in self.factors:
            n *= int(k)
        return n

    def gate_inputs(self, frame, life, state):
        """its gate's own "ear" (step R5): the `n_in` numbers it reads beside the stream and the feelings, on the world's `frame` and
        the effector's working `state` in the life (life.motor); the base effector's is its own act last tick (sensed, not inferred)"""
        return [1.0 if state["acted_last"] else 0.0]

    def cost(self, act, frame, life):
        """the effort of an act (step R5), added to the body's fatigue when it acts: the base effector's is its declared `effort`"""
        return float(self.effort)

    def propose(self, life, C):
        """its proposal: the vector its per-joint readout reads, each joint's logits the readout's sharpness times the proposal's dot
        product with that joint's rows, or None (every setting of every joint equally likely, the striatal actor's bias alone shaping
        the draw). Step R6: act_pred's, the forecast of its own next act from the stream `C` (m.timing[name].pred), plus, when it
        declares a body sense, the forward half's correction of the error its sense shows now (`life._timing_propose`,
        body/core/timing.py). A body's effector may add to it (the gaze's born orienting, SIM_DESIGN.md 5.8)."""
        return life._timing_propose(self, C)

    def reflex(self, frame, life, state):
        """ITS SPINAL REFLEX (step R6; SIM_DESIGN.md 5.5): the act the reflex forces on the world this tick, or None. The base effector
        has none. A reflex stops a chunk under way; its act goes to the world while the effector's own act is its rest (no gate draw,
        no efference copy: the cortex senses it through the body, act_pred's target there is act_inv's reading of what moved it); its
        tick gives the gate no eligibility and the actor no credit. `state` is the effector's working state (life.motor), where a
        reflex that lasts keeps its count."""
        return None


@dataclass(eq=False)
class VoiceEffector(Effector):
    """THE VOICE, EFFECTOR 0 (step R5): its acts are the words' symbols, read from the lexicon it shares with the ear (organ E); its
    gate is m.mouth_gate (with opt_gate and gate_buf), its actor m.actor, its act the window's "xo"; its gate's inputs beyond the
    stream are its own ear (the partner's symbol, its trace or the sensed pace's hold, and its own act last tick: `_voice_ear`, under
    gate_ear), widened on the gate as they always were; its cost is symbol_cost a symbol. The voice's choice, act and lesson are
    today's code (the mouth mixin), its draws on self.gen the tick's first."""
    organ: Optional[str] = "E"
    gate: Optional[str] = "mouth_gate"
    actor: Optional[str] = "actor"
    field: Optional[str] = "xo"
    n_in: int = 0

    def gate_inputs(self, frame, life, state=None):
        u = frame.obs.get(life.anatomy.words.name)                  # the world's symbol this tick (a frame that names none: the rest)
        return life._voice_ear(life.sil if u is None else int(u))

    def cost(self, act, frame, life):
        return float(life.cfg["symbol_cost"])

    def propose(self, life, C):
        """the voice's proposal is the words' forecast, read in `_choose` as always (it has no act_pred)"""
        return None


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
    """THE WORLD'S WORDS AS REWARD (world_r; 0 = off): each symbol the partner sends on the ear (`frame.obs["ear"]`, not its rest; a
    frame that names none is quiet, step R9) is felt at world_r beside the face; under world_mask, the corollary discharge, not on a tick
    after the voice acted (not heard over its own voice). Silent otherwise."""

    def felt(self, frame, life):
        u = frame.obs.get("ear")
        if u is not None and u != life.sil:
            wr = float(life.cfg.get("world_r", 0.0))
            if wr and not (int(life.cfg.get("world_mask", 0)) and getattr(life, "_acted_last", False)):
                return wr
        return None


class EffortReward(RewardSource):
    """THE EFFORT IN THE REWARD (cost_in_reward; 0 on the served body): the cost of the last act, symbol_cost x (1 + (fatigue /
    gate_fatigue)^2), is felt as the next tick's reward, negative, so both critics predict it and the gate reads their error alone.
    Added to the act's credit outside the critics (the earlier form, with a tonic drive of 0.25 cancelling it) it was never predicted
    away and, with the drive gone, held every act at a loss; with both gone the gate saturated at 0.98 (runs 69-72). Silent on a tick
    after no act. The feeling is the cost negated: adding it is the IEEE subtraction of the cost (x + (-c) is x - c, bit for bit).
    THE VOICE'S EFFORT ONLY (the R5 verifier's note): it feels the voice's act (symbol_cost), and under cost_in_reward the gate's lesson
    leaves every effector's effort to the reward, so a later effector's effort reaches neither the reward nor its gate's lesson (only
    the fatigue). A body with later effectors keeps cost_in_reward 0 (the sim's effort is felt through its charge) or declares an
    effort source of its own for them."""

    def felt(self, frame, life):
        if life.cfg.get("cost_in_reward") and getattr(life, "_acted_last", False):
            return -(float(life.cfg["symbol_cost"]) * (1.0 + (life.fatigue / float(life.cfg["gate_fatigue"])) ** 2))
        return None


class Anatomy:
    """a body's senses, effectors and reward sources, each list in its order: the channels' order is the float order of the cortex's
    input sum (the ladder's bundle and the efference copy of its own acts joining after the first `inner_at` channels; by default
    after all of them), channel 0 is the words, effector 0 is the voice, the reward sources are summed in their order and source 0 is
    the world's judgment"""

    def __init__(self, channels, effectors, rewards, inner_at=None):
        self.channels = list(channels)
        self.effectors = list(effectors)
        self.rewards = list(rewards)
        self.inner_at = len(self.channels) if inner_at is None else int(inner_at)

    @property
    def words(self):
        """channel 0, the words: the lexicon's symbols (its code the row of m.E, the table the voice shares), the forecast the mouth
        reads (latent_pred), the waking lesson's targets and the symbols the night dreams"""
        return self.channels[0]

    def channel(self, name):
        return next(c for c in self.channels if c.name == name)

    def effector(self, name):
        return next(e for e in self.effectors if e.name == name)

    def sense_size(self, e):
        """step R6: the number of body-sense numbers effector `e` reads (0: none)"""
        if e.sense is None:
            return 0
        return len(e.sense_idx) if e.sense_idx is not None else int(self.channel(e.sense).size)

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
        if not self.channels:
            raise ValueError("anatomy: no channel (channel 0 is the words)")
        w = self.channels[0]
        if w.kind != "symbol" or w.organ != "E" or not w.forecast:
            raise ValueError(f"anatomy: channel 0 ({w.name!r}) must be the words: a symbol channel encoded by the lexicon (organ 'E', the table "
                             f"the voice shares and the readout reads) with its forecast (latent_pred); it is {w.kind!r}, organ {w.organ!r}, forecast {w.forecast}")
        if any(c.partner for c in self.channels[1:]):
            raise ValueError("anatomy: the partner is channel 0, the words")
        if not 0 <= int(self.inner_at) <= len(self.channels):
            raise ValueError(f"anatomy: the bundle and its own acts join the input sum after {self.inner_at} of {len(self.channels)} channels")
        fields = [c.field for c in self.channels]
        if len(set(fields)) != len(fields) or any(f_ in CORE_FIELDS for f_ in fields) or not all(isinstance(f_, str) and f_ for f_ in fields):
            raise ValueError(f"anatomy: the channels' window fields must be distinct names, none of the core's own {CORE_FIELDS}: {fields}")
        for c in self.channels:
            if c.kind not in KINDS:
                raise ValueError(f"anatomy: channel {c.name!r} of kind {c.kind!r}, not one of {KINDS}")
            if int(c.size) < 1:
                raise ValueError(f"anatomy: channel {c.name!r} of size {c.size}")
            if not (isinstance(c.name, str) and c.name.isidentifier()):
                raise ValueError(f"anatomy: channel name {c.name!r} is not an identifier (it names the channel's forecast head)")
            if not (isinstance(c.organ, str) and c.organ.isidentifier()):
                raise ValueError(f"anatomy: channel {c.name!r} names no organ to encode it ({c.organ!r})")
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
            n = e.n_acts
            ids = [i for i in [e.rest_id, e.end_id, *e.reserved] if i is not None]
            if any(not 0 <= int(i) < n for i in ids):
                raise ValueError(f"anatomy: effector {e.name!r} declares an act outside its {n}")
            if e.rest_id is not None and e.rest_id in e.reserved:
                raise ValueError(f"anatomy: effector {e.name!r}'s rest is reserved")
        # step R5: effector 0 is the voice, sharing the words' table; a later effector names the organs the organs build for it
        v, w = self.effectors[0], self.channels[0]
        if (v.organ, v.gate, v.actor, v.field) != (w.organ, "mouth_gate", "actor", "xo") or [int(k) for k in v.factors] != [int(w.size)]:
            raise ValueError(f"anatomy: effector 0 ({v.name!r}) must be the voice (VoiceEffector): one choice among the words' {w.size} symbols from "
                             f"the table it shares with the ear (organ {w.organ!r}), its gate mouth_gate, its actor 'actor', its act the window's 'xo'; "
                             f"it declares factors {v.factors}, organ {v.organ!r}, gate {v.gate!r}, actor {v.actor!r}, field {v.field!r}")
        if v.sense is not None or v.sense_idx is not None or v.inverse:
            raise ValueError(f"anatomy: the voice ({v.name!r}) has no motor timing part (its proposal is the words' forecast)")
        chan_names, chan_fields = {c.name for c in self.channels}, {c.field for c in self.channels}
        seen_fields = set()
        for e in self.effectors[1:]:
            if not (isinstance(e.name, str) and e.name.isidentifier()):
                raise ValueError(f"anatomy: effector name {e.name!r} is not an identifier (it names the effector's organs)")
            if (e.organ, e.gate, e.actor) != (f"acts.{e.name}", f"gates.{e.name}", f"actors.{e.name}"):
                raise ValueError(f"anatomy: a later effector's organs are acts.<name>, gates.<name> and actors.<name> (the organs build them); "
                                 f"{e.name!r} names {e.organ!r}, {e.gate!r}, {e.actor!r}")
            if e.name in chan_names:
                raise ValueError(f"anatomy: effector {e.name!r} shares a channel's name (the cortex's input reads both by name)")
            if not (isinstance(e.field, str) and e.field) or e.field in chan_fields or e.field in CORE_FIELDS or e.field in seen_fields:
                raise ValueError(f"anatomy: effector {e.name!r}'s window field {e.field!r} is a channel's, the core's own {CORE_FIELDS} or another effector's")
            seen_fields.add(e.field)
            if e.rest_id is None:
                raise ValueError(f"anatomy: effector {e.name!r} declares no rest (the act of its gate's no)")
            if e.reserved and len(e.factors) > 1:
                raise ValueError(f"anatomy: effector {e.name!r} reserves acts of a factored alphabet (its joints are drawn apart)")
            if int(e.n_in) < 0:
                raise ValueError(f"anatomy: effector {e.name!r} declares {e.n_in} gate inputs")
            # step R6: its body sense (a vector channel, its own numbers in it) and its inverse model over that sense
            if e.sense is not None:
                sc = next((c for c in self.channels if c.name == e.sense), None)
                if sc is None or sc.kind != "vector":
                    raise ValueError(f"anatomy: effector {e.name!r} senses its body on {e.sense!r}, not a vector channel of this anatomy")
                idx = list(e.sense_idx) if e.sense_idx is not None else list(range(int(sc.size)))
                if not idx or len(set(int(j) for j in idx)) != len(idx) or any(not 0 <= int(j) < int(sc.size) for j in idx):
                    raise ValueError(f"anatomy: effector {e.name!r}'s own numbers {e.sense_idx} in the {sc.size} of {e.sense!r}")
            elif e.sense_idx is not None:
                raise ValueError(f"anatomy: effector {e.name!r} declares its own numbers of no sense")
            if e.inverse and e.sense is None:
                raise ValueError(f"anatomy: effector {e.name!r} declares an inverse model and no body sense for it to read")
            if int(e.inv_hidden) < 1:
                raise ValueError(f"anatomy: effector {e.name!r}'s inverse model of {e.inv_hidden} units")
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
    """THE DIARY'S BODY: one ear on the typed page (the partner, channel 0: the words), the face as a sense, the voice as its one
    effector, the reward felt from the face (the judgment, clipped to +-2), then the world's words (world_r, not over its own voice under
    world_mask), then the effort (under cost_in_reward; 0 on the served body), summed in that order as `_sense` summed them before step
    R3. The cortex's input is ear + face + the ladder's bundle + its own sound (inner_at 2), the order `Organs.inputs` summed them in
    before step R4. The symbols are derived here as `Life.__init__` derived them before step R2 (the life now copies them from its
    anatomy), from `tok` and the physiology updated by `cfg`, so they are declared by the body that is born, never found by a fixed
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
        # the channels in the input sum's float order (step R4): the ear, the face, then the ladder's bundle and its own sound (inner_at 2);
        # their window fields are the keys the window always had, their codes the organs that always made them
        ear = EarChannel("ear", "symbol", self.vocab, organ="E", field="x", forecast=True, rest_id=self.sil, end_id=self.end_id,
                         reserved=self.reserved, partner=True)
        face = FaceChannel("face", "vector", 2, organ="face_in", field="face")
        voice = VoiceEffector("voice", [self.vocab], rest_id=self.sil, end_id=self.space_id, reserved=self.bans)   # effector 0 (step R5)
        rewards = [FaceReward("face", clip=2),
                   WorldWordsReward("world_r", ("world_r", "world_mask")),
                   EffortReward("cost", ("cost_in_reward", "symbol_cost", "gate_fatigue"))]
        super().__init__([ear, face], [voice], rewards, inner_at=2)

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
    since a body's rest and ends are named by its physiology (one declared under other constants is refused, not mixed); it may declare
    channels beyond the diary's two (step R4 wires channels into the core) and effectors beyond the voice (step R5 wires effectors);
    since step R9 its later channels observe the world's frames and its later effectors act on the world (body/core/world.py). Any other
    anatomy is refused: the core's words are a language's, its rest and ends, the typing, the page and the night's report read through
    a tokenizer (the sim's words channel is the parent's word tokens, declared on a language anatomy of its own; SIM_DESIGN.md 5.8)."""
    if not isinstance(body, Anatomy):
        return LanguageAnatomy(body, cfg)
    if not isinstance(body, LanguageAnatomy):
        raise NotImplementedError(f"anatomy_for: the core's words are a language's (its rest, its ends, the typing, the page and the night's "
                                  f"report through a tokenizer): a body declares them on a LanguageAnatomy, with any later channels and "
                                  f"effectors (SIM_DESIGN.md 8.2); given {type(body).__name__}")
    body.check()
    want = LanguageAnatomy(body.tok, cfg).symbols()
    if want != body.symbols():
        raise ValueError(f"anatomy_for: this language anatomy was declared under other constants than the life's (its rest, display, "
                         f"space, turn-end and end {body.symbols()[1:6]}; the life's {want[1:6]})")
    return body
