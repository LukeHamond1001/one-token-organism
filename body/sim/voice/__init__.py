"""THE VOICES (docs/SIM_DESIGN.md 4.4, 4.9; the owner's decision 5 of 2026-09-24; package P1).

  synth.py      the parent's voice: macOS speech through a small Swift server (synth_server.swift), every line SSML in its
                register, 16 kHz clips with their words' onsets and ends, a content-addressed cache and a ledger of digests
  playback.py   a parent's line spoken tick by tick into the world, its words handed to the words channel, a talk-over cut
The child's voice is its vocal tract, body/sim/tract.py (the design's 17 places it beside the ears, body/sim/ears.py): the voice
effector's physics (10 articulators, a 24-section tube, the glottal source, turbulence, the breath).

THE CHILD'S VOICE AND THE WORD SCAFFOLD: what we chose, and why.
  - The child's voice is its tract (the owner's decision 5; the decision log's B6: a child-sized throat, resting pitch about
    265 Hz). Every sound the child makes comes from its articulators, and it hears it through its own ears from its head's front
    (body/sim/ears.py: the rigid sphere's level for a source on its surface 90 degrees from each ear, +19.8 dB over the same
    sound from 1.5 m in front, read at the midline; the first design's "about 14 dB" was the prototype's 0.3 m level floor, not
    a measurement), an act at tick t heard at t + 1. Nothing of the child is synthesized by the engine (the first design's
    synthesized child voice and its letter runs are gone from the amended design, and are not built).
  - The 79-row table (body/sim/lang/lexicon.py) stays at birth on both sides. In: the words channel (channel 0), the parent's
    own label beside the sound, unchanged from section 4.9. Out: a SILENT effector beside the tract, with its own gate, whose
    tokens and letters the parent reads as an exact transcript (4.9); the child never hears it. Both go on a copy at a night boundary: the input by 4.9's two tests; the output when the input test passes, at least 10
    words reach "says" from the tract alone, and a copy day without the token output keeps at least 80% of the right names (the
    voice study's proposal, finding 3 of the amendment).
  - Why the silent output stays: early imitation through the tract is near chance. In the voice study an inverse model learned
    from 16,000 ticks of the child's own babble read the parent's words into echoes the parent accepted for 2 of 12 words, so
    without the output no word would reach the parent for a long time and stage 2's judgments (right names, met asks) would have
    nothing to judge. The tract still earns stage 1's vocal-turn smile, and the parent hears it.
  - The parent's ear for the tract (body/sim/parent_ear.py, the study's prototype; P3v builds it) listens only for the words it
    expects in the situation,
    and accepts one only when the sound is nearer that word than a bank of the child's own babble, recorded before birth and
    fixed, by a margin set so held-out babble passes 2% of the time: in the study real other speakers' words then passed 100%,
    where a fixed distance cutoff passing 1% of babble passed 0% of them. It is to hear through the same cochlea as the child
    (ears.cochlea); until P3v the prototype hears through the study's (body/sim/parent_ear_cochlea.py, kept at the merge of
    main's G1 prototypes so it runs as it was measured). The bank stays fixed (adding rejected near-words in life would lock those pronunciations out), and the
    margin is re-set for the real context sets (P3, P6). The babbler's accepted approximations are written down as chance, and
    the tract's first accepted word is milestone M6t.
  - Every figure of the parent's ear in the design (4.9: the margin m = -0.35, 94-100% and 72-96% recognized, 2 of 12 echoes
    accepted, the tract's reach table) was measured in the voice study through the prototype cochlea (allout/lang/ear2.py),
    whose filters were 1.6 ERB wide; the committed cochlea's are 1.00 ERB. They are the study's numbers, not this cochlea's:
    P3v measures them again on ears.cochlea before any is used.
  - The risk: tokens are the easier road to reward. The removal test watches it.
"""
