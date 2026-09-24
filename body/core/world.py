"""the world as the body meets it (docs/SIM_DESIGN.md 8.2; the core refactor): `Frame`, what the world shows the body at one tick.

STEP R3 declares the frame, for the reward sources (body/core/anatomy.py, `RewardSource.felt(frame, life)`). For the language body the
frame is still built inside the tick from the queue (`_sense`, body/core/senses.py), so the draw order does not move. The world itself
(`World`: frame, apply, pause, resume, save_state, load_state) and the diary's (`DiaryWorld`, today's queue and face, which serve.py
wraps) come with step R9, the world loop. A frame holds no tensor and no module, and building one draws no random number."""
from dataclasses import dataclass, field


@dataclass(eq=False)
class Frame:
    """ONE TICK OF THE WORLD, as the body meets it. `obs` maps a channel's name to its raw observation (the language body's: the ear's
    symbol this tick, or its rest); `face` is the teacher's face level (a scaffold at birth); `truth` is for the teacher and the
    instruments only and never enters the body."""
    tick: int
    obs: dict
    face: float
    truth: dict = field(default_factory=dict)
