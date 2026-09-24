"""the organism's roles, as mixins of `Life` (body/life.py): the split of 2026-09-23 (review 2026-09-22 section 4, step 2).

physiology.py   PHYSIOLOGY, every constant grouped by organ (re-exported by body/life.py)
senses.py       the feelings' recovery, _sense, _hear, the offset, the world's hands (type_text, set_face)
memory.py       the recall's query and read, the bags and the slow context (the keys), its own utterance as an episode
cortex.py       _step, the window, the stream, the waking lesson, the calibrated readout
mouth.py        the sensed pace (M1-M5), imagination for choice, _choose, _act, _feel_and_learn, the gate's lesson
critics.py      the fast value, _learn_values (dopamine, the least-squares critics), the reliability gain, the face organ
actor.py        the chooser's eligibility and lesson, the actor's reliability
night.py        dreams, night (NREM, REM, the value replay, the fade), the night's device, the sleep switch's call
persistence.py  save, load, birth
instruments.py  _bookkeep (the page, the record, the sleep switch), gauge, state, anticipation, insides
anatomy.py      not a mixin: the body's anatomy declared (Channel, Effector, RewardSource, Anatomy, LanguageAnatomy; docs/SIM_DESIGN.md
                8.2), step R1 of the core refactor; since step R2 `Life` is built with one (`life.anatomy`, from the tokenizer by
                `anatomy_for`) and reads its symbols and its text (the tokenizer) there; since step R3 the tick's reward is its reward
                sources (FaceReward, WorldWordsReward, EffortReward), felt in their order and summed in it (`_sense`); since step R4
                its channels (EarChannel, FaceChannel) are the window's fields, the cortex's input (their codes summed in their order,
                `Organs.inputs`) and, for a later channel that declares one, a forecast head of its own (`Organs.head`); since step
                R5 its effectors: the voice is effector 0 (VoiceEffector: the lexicon E, mouth_gate, actor and "xo", its choice, act
                and lesson today's), and each later effector (Effector) names the organs the organs build for it (acts.<name>,
                gates.<name>, actors.<name>) and is chosen, acts and learns after the voice (`_choose_effector`, `_act_effectors`,
                `_gate_lesson(i)`); physiology.py's SWITCHES hold the defect fixes 4, 5 and 8, off by their absence
world.py        not a mixin: `Frame`, the world at one tick as the body meets it (docs/SIM_DESIGN.md 8.2; step R3: the reward sources
                read it; the language body builds it inside the tick from the queue); the World and DiaryWorld come with step R9

Every method was moved verbatim; `Life` keeps `__init__` (the organs and the state, in their order), `tick` and (step R2) the
read-only `tok`, its anatomy's tokenizer. The mixins hold no state and no class attributes, and no method name is defined twice, so
the order of the bases decides nothing.
"""
