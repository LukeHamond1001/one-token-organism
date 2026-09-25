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
  - The parent's ear for the tract (body/sim/parent_ear.py, built in P3 on the child's own cochlea, ears.cochlea) listens only
    for the words she expects in the moment, and accepts one only when the sound is nearer that word than a bank of the child's
    own babble, recorded before birth and fixed, by a margin m set so held-out babble passes 2% of the time, and near enough the
    nearest of all her words (P3's delta, set so the held-out voices' other words pass 2% of the time); a word is
    exact only when it is also the nearest of all her words. Her templates are made through this voice's cache (VoiceCache.clip
    with prosody=: other voices, the old child pitch) and kept in its ledger. Nothing she hears is added to her ear. The babbler's
    accepted approximations are written down as chance; the tract's first accepted word is milestone M6t
    (tools/sim_parent_ear.py measures the margins, the held-out voices and the cost).
  - The design's figures for the tract in the parent's ear (4.9: 2 of 12 echoes accepted, the tract's reach table) are still the
    voice study's, measured through its own cochlea (allout/lang/ear2.py, filters 1.6 ERB wide): P3v measures them again on this
    ear. Her ear's own figures (the margins, the other voices' recognition, its cost) were measured again on ears.cochlea in P3.
  - The risk: tokens are the easier road to reward. The removal test watches it.
"""
