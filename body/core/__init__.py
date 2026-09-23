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

Every method was moved verbatim; `Life` keeps `__init__` (the organs and the state, in their order) and `tick`. The mixins hold no
state and no class attributes, and no method name is defined twice, so the order of the bases decides nothing.
"""
