"""the second body (BODY_SPEC.md): a small organism raised only by live conversation.

model.py  the organs: lexicon, hippocampus, PFC ladder, cortex, mouth gate, value ladder
life.py   the body alive: Life, its __init__ (the organs and the state) and its tick
core/     Life's other methods as mixins by role: physiology (the constants), senses, memory, cortex, mouth, critics, actor,
          night, persistence, instruments (core/__init__.py has the map)
serve.py  THE DIARY protocol on a port (page, /type, /face, /state, /save)
tests/    one test per organ; each fails when its organ stops doing its job
"""
