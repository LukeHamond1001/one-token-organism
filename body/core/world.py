"""the world as the body meets it, and the loop that runs the two (docs/SIM_DESIGN.md 8.2; the core refactor, steps R3 and R9).

`Frame`: one tick of the world as the body meets it (step R3 declared it, for the reward sources, body/core/anatomy.py
`RewardSource.felt(frame, life)`). STEP R9, THE WORLD LOOP:
- `World`: what a world offers a body. `frame()` its frame this tick, `apply(acts)` the body's acts, `pause()` when the night falls
  and `resume()` in the morning, `save_state()` and `load_state(blob)`; `now`, the frame the body lives this tick (set at the tick's
  senses: a channel of the world's frames observes it there, `Channel.observe`); `lapse(n)`, the ticks it runs without the body (the
  deadline switch only).
- `DiaryWorld`: the language body's world, today's page queue and face (the life holds them where it always did; the world reaches
  them). body/serve.py wraps it. Its pause is what the sleep switch always did (the queue let go); it moves nothing else.
- `SimWorld`: the interface the simulated world implements (body/sim/world.py, to come; SIM_DESIGN.md 5 and 6.1). Abstract.
- `WorldLoop`: frame, tick, apply, one pass a tick. Lockstep, the world waiting for the child; the deadline switch (a test switch,
  never the default) lets the world run on without it.
- `PaceLog`: the ops guard's record (SIM_DESIGN.md section 3, graft 5): the loop the serve runs around the DiaryWorld appends one line
  every 1000 ticks and one each morning to logs/diary_pace.jsonl.

A life is born with a world (`Life(..., world=None)`: the DiaryWorld unless one is given; `life.world`) and asks it for the tick's frame
at the senses' phase (`_sense`, body/core/senses.py), where the queue was always read, so the draw order does not move. The sleep switch
pauses the world before the night and resumes it after (`_sleep_now`, body/core/night.py). The world is an interface object: it is not
saved with the body (the body's save never held the queue), `--full` leaves it to the tests (tools/determinism_check.py's skip list, since
R1), nothing here draws a random number, and the pace log is logging only, outside the life, so the digests do not move. The diary's
frame holds no tensor and no module."""
import abc
import json
import os
import time
import weakref
from dataclasses import dataclass, field


@dataclass(eq=False)
class Frame:
    """ONE TICK OF THE WORLD, as the body meets it. `tick` is the world's tick; `obs` maps a channel's name to its raw observation (the
    language body's: the ear's symbol this tick, or its rest; a channel the frame does not name is quiet); `face` is the teacher's face
    level (a scaffold at birth); `truth` is for the teacher and the instruments only and never enters the body (the diary's: who typed
    the symbol, which the page shows)."""
    tick: int
    obs: dict
    face: float
    truth: dict = field(default_factory=dict)


class World(abc.ABC):
    """A WORLD, AS THE BODY MEETS IT (step R9). One tick, lockstep: `frame()` shows the world as it is (no time passes), the body lives
    the tick on it, `apply(acts)` takes the body's acts ({effector name: act}, a rest being an act) and moves the world on by a tick.
    `pause()` freezes the world where it stands when the night falls (nothing is moved) and `resume()` goes on from the same state in the
    morning. `save_state()` gives the world's state as bytes and `load_state(blob)` gives it back exactly; a world's state is saved beside
    the body's save, never inside it. `now` is the frame the body lives this tick: the tick's senses set it (`_sense`), and a channel of
    the world's frames observes it (`Channel.observe`); none before the first tick. `lapse(n)` (the deadline switch only): n ticks of the
    world pass without the body; by default n moves with no act (every effector at its rest)."""
    now = None

    @abc.abstractmethod
    def frame(self):
        """this tick's Frame"""

    @abc.abstractmethod
    def apply(self, acts):
        """the body's acts this tick, {effector name: act}"""

    @abc.abstractmethod
    def pause(self):
        """the night falls: the world stands still"""

    @abc.abstractmethod
    def resume(self):
        """the morning: the world goes on from where it stood"""

    @abc.abstractmethod
    def save_state(self):
        """the world's state, as bytes"""

    @abc.abstractmethod
    def load_state(self, blob):
        """the world's state given back exactly, from `save_state`'s bytes"""

    def lapse(self, n):
        """n ticks of the world pass without the body (the deadline switch): n moves with no act"""
        for _ in range(int(n)):
            self.apply({})


class DiaryWorld(World):
    """THE DIARY'S WORLD (the language body's; step R9): the page's queue of typed symbols (who typed each) and the face the caregiver's
    hand holds. They stay where they always were, on the life (`life.queue`, `life.queue_who`, `life.face_now`: the tools read them
    there), and the world reaches them through a weak reference (it never keeps its body alive). body/serve.py wraps it: the page's two
    hands (`type_text`, `set_face`, the life's own) reach the body through it, and the serve's loop is a WorldLoop around it.
    - frame(): the queue's next symbol (or the rest when it is empty), who typed it (truth: the page shows it, nothing inside reads it) and
      the face the hand holds, at the life's tick: what `_sense` read at that point before R9, in the same order.
    - apply(acts): nothing moves. The voice's symbol is on the page already: the tick's record writes it, as it always did.
    - pause(): the queue is let go, what the sleep switch always did before the night. resume(): nothing more; the morning hears what is
      typed from then on, as before.
    - save_state() / load_state(blob): the queue, who typed each symbol and the face, as JSON. The body's save never held them (a reload
      begins with an empty queue, as always), and still does not.
    - lapse(n) (the deadline switch only): n symbols of the queue go by unheard."""

    def __init__(self, life):
        self._life = weakref.ref(life)

    @property
    def life(self):
        life = self._life()
        if life is None:
            raise RuntimeError("DiaryWorld: its life is gone")
        return life

    def frame(self):
        life = self.life
        u = life.queue.popleft() if life.queue else life.sil
        who = (life.queue_who.popleft() if life.queue_who else "") if u != life.sil else ""
        return Frame(life.ticks, {life.anatomy.words.name: u}, life.face_now, {"who": who})

    def apply(self, acts):
        return None

    def pause(self):
        life = self.life
        life.queue.clear(); life.queue_who.clear()

    def resume(self):
        return None

    def save_state(self):
        life = self.life
        return json.dumps({"queue": [int(u) for u in life.queue], "who": [str(w) for w in life.queue_who], "face": float(life.face_now)}).encode()

    def load_state(self, blob):
        st = json.loads(bytes(blob).decode())
        life = self.life
        life.queue.clear(); life.queue.extend(int(u) for u in st["queue"])
        life.queue_who.clear(); life.queue_who.extend(str(w) for w in st["who"])
        life.face_now = float(st["face"])

    def lapse(self, n):
        life = self.life
        for _ in range(int(n)):
            if not life.queue:
                break
            life.queue.popleft()
            if life.queue_who:
                life.queue_who.popleft()

    # the page's two hands (the life's own, where the tools find them)
    def type_text(self, s, who=""):
        return self.life.type_text(s, who=who)

    def set_face(self, expr):
        return self.life.set_face(expr)


class SimWorld(World):
    """THE SIMULATED WORLD'S INTERFACE (step R9; SIM_DESIGN.md 5, 6.1 and 8.2), for body/sim/world.py to implement over the MuJoCo scene
    (the high chair: the arm, the mitten, the head, four objects, the parent robot across the table). Every method is the sim's to write,
    and the class cannot be instantiated until each is written. The world waits for the child (lockstep): sim time passes only in
    `apply`, so nothing the body learns is fitted to wall-clock pace.
    - frame(): the world at this tick, rendered and sensed, each channel's raw observation under the channel's name (a channel of the
      world's frames observes it there); `face`, the parent's face level (the scaffold); `truth`, object poses and contacts, for the
      teacher and the instruments only. No sim time passes, and it draws nothing from the body's random streams.
    - apply(acts): one tick, 150 ms of sim time, with the body's acts ({effector name: act}; an effector at its rest holds, by the servo
      law); each act is sensed in the next tick's frame. `acts` empty: every effector at rest (a lapse).
    - pause(): the night falls. The world freezes exactly where it is: nothing is moved, no scripted posture, the teacher's clock stops.
    - resume(): the morning. The world goes on from the same state.
    - save_state() / load_state(blob): the whole world (the physics, the teacher's state and clocks, the world's own random streams) as
      bytes, and back exactly (the sim's exact-replay test); saved beside the body's save, never inside it."""

    @abc.abstractmethod
    def frame(self):
        """the world at this tick, as a Frame (no sim time passes)"""

    @abc.abstractmethod
    def apply(self, acts):
        """one tick of sim time with the body's acts"""

    @abc.abstractmethod
    def pause(self):
        """the night: frozen exactly where it is"""

    @abc.abstractmethod
    def resume(self):
        """the morning: on from the same state"""

    @abc.abstractmethod
    def save_state(self):
        """the whole world, as bytes"""

    @abc.abstractmethod
    def load_state(self, blob):
        """the whole world, back exactly"""


class PaceLog:
    """THE PACE LOG (SIM_DESIGN.md section 3, graft 5: the ops guard; step R9). The language body must not slow down while the sim runs
    beside it, and its serve wrote no tick timing, so the loop around its world records it: one JSON line every `every` ticks it lived
    (kind "ticks": the body's tick count, the ticks, the seconds they took inside the tick, the wall seconds from each tick's start to
    the next's over `periods` intervals, and how many ran over the period) and one line each morning (kind "night": the seconds of the
    pass that held it, the dusk tick and the night). Nights are left out of the tick lines, and so is the interval across one. Logging
    only, outside the life: nothing in the body reads it, and a line that cannot be written is said once and skipped, never an error
    in the loop. The sim's guard reads the file."""

    def __init__(self, path, period, every=1000, stamp=time.time):
        self.path = str(path); self.period = float(period); self.every = int(every); self.stamp = stamp
        self._prev = None; self._n = 0; self._busy = 0.0; self._wall = 0.0; self._periods = 0; self._over = 0
        self.lines = 0; self.failed = 0

    def record(self, t0, t1, tick, nights, night=False):
        """one pass of the loop, from `t0` to `t1` (seconds on the loop's clock), after which the body has lived `tick` ticks and
        `nights` nights; `night`: the pass held the night"""
        if night:
            self._write({"kind": "night", "t": round(float(self.stamp()), 3), "tick": int(tick), "night": int(nights),
                         "seconds": round(float(t1 - t0), 3)})
            self._prev = None                                 # the interval across the night is left out
            return
        if self._prev is not None:
            self._wall += float(t0 - self._prev); self._periods += 1
        self._prev = t0
        busy = float(t1 - t0)
        self._n += 1; self._busy += busy
        if self.period > 0 and busy > self.period:
            self._over += 1
        if self._n >= self.every:
            self._write({"kind": "ticks", "t": round(float(self.stamp()), 3), "tick": int(tick), "n": self._n, "busy": round(self._busy, 3),
                         "wall": round(self._wall, 3), "periods": self._periods, "over": self._over, "period": self.period})
            self._n = 0; self._busy = 0.0; self._wall = 0.0; self._periods = 0; self._over = 0

    def _write(self, row):
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
            self.lines += 1
        except Exception as e:                                # logging only: the loop goes on
            self.failed += 1
            if self.failed == 1:
                print(f"[pace] {self.path}: {e!r} (the pace log is skipped while it fails)", flush=True)


class WorldLoop:
    """THE WORLD LOOP (step R9; SIM_DESIGN.md 8.2): `frame = world.frame(); acts = life.tick(frame); world.apply(acts)`, one pass a tick,
    over the life's own world (`life.world`). The frame is taken inside the tick, at its senses' phase (`_sense` asks the world; the
    diary's is built there from the queue, where the queue was always read, so the draw order does not move), and the acts the tick
    returns go to the world. The night comes inside a tick (the sleep switch), which pauses the world and resumes it in the morning; that
    tick's acts reach the world after the resume (the loop applies them when the tick returns), and the world, still between, meets
    them where it stood at dusk.
    LOCKSTEP, the default: the world waits for the child however long its tick takes (SIM_DESIGN.md 6.1). `period` is the wall seconds
    of a tick; the caller keeps it (the serve pads each pass to its --period; a viewer's 1x), and the loop measures against it.
    THE DEADLINE SWITCH (`deadline`, off; a test switch, never the default: SIM_DESIGN.md section 3, item 11): the world does not wait.
    Its clock runs a tick every `period` seconds while it is awake; at each pass the ticks it ran past the ones the body lived lapse
    unseen (`world.lapse`), the body lives the tick that is due, and a body ahead of the clock waits for it. The night stops the clock:
    after a pass that held a night it starts again at the morning, and nothing lapses for the night.
    `pace`: a PaceLog, given each pass's start and end (the serve's around the DiaryWorld: logs/diary_pace.jsonl)."""

    def __init__(self, life, period=0.0, deadline=False, pace=None, clock=time.monotonic, sleep=time.sleep):
        self.life = life
        self.world = life.world
        self.period = float(period)
        self.deadline = bool(deadline)
        if self.deadline and not self.period > 0:
            raise ValueError("WorldLoop: the deadline switch needs the world's period (the wall seconds of its tick)")
        self.pace = pace
        self.clock = clock; self.sleep = sleep
        self.lived = 0; self.lapsed = 0                       # the ticks the body lived through the loop, and those that went by without it
        self._base = None; self._next = 0                     # the deadline's clock: its start, and the world tick the body lives next

    def step(self):
        """one pass; returns the tick's acts"""
        life, world = self.life, self.world
        if life.world is not world:
            raise RuntimeError("WorldLoop: the life's world is no longer the loop's")
        t0 = self.clock()
        if self.deadline:
            if self._base is None:
                self._base = t0
            due = int((t0 - self._base) // self.period)
            if due > self._next:
                world.lapse(due - self._next); self.lapsed += due - self._next; self._next = due
            elif due < self._next:                            # ahead of the world's clock: wait for the tick
                self.sleep(max(0.0, self._base + self._next * self.period - t0))
                t0 = self.clock()
        dusk = life.last_night
        acts = life.tick()
        world.apply(acts)
        t1 = self.clock()
        self.lived += 1
        night = life.last_night is not dusk                   # the night's report is new (a night that failed leaves one too)
        if self.deadline:
            self._next += 1
            if night:
                self._base = t1; self._next = 0               # the night stopped the world's clock: it starts again at the morning
        if self.pace is not None:
            self.pace.record(t0, t1, life.ticks, life.nights, night=night)
        return acts
