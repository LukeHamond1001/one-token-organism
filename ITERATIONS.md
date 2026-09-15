# The iterations (2026-09-14)

The organs are all present; each is made to do what its original does, one change at a time, measured on a copy before the served
body takes it at a night's save. Rulers: HELD-OUT (tools/heldout_stage4.txt), the old lines (days 110-125), the fact prefixes and the
fact sentences (probe_lm.py --lmloss), the questions by the pause (qa_by_gap.py, probe_lm.py --qa), the branch after a shared start
(branch_probe.py), the day's tally (answers before B, fact words in its turn, own questions, smiles, frowns), the actor's slope, duty.

## Landed
- The write floor (write_floor 1e-30): the answer's onset written after long turns; the branch 9 of 9 on the fast bag alone once it is (the matched test, 2026-09-15). On the served body since night 188.
- The slow context in the memory's key (key_ctx 0.5): separated siblings at the branch but cost the held-out 0.024 and four answers in the matched test; reverted at night 191; kept as an instrument.
- The question rulers read twenty symbols and read the store along the question as awake; the depth measure for parents.

## Failed, kept as instruments, off
- The exchange replay (dream_pair): no preference at the branch after three copy nights.
- The cortex-state key (key_form cortex, pattern-separated): 0/30; the state is the whole window.
- The context at weight 1.0 (loops); the query horizon 0.9 (no peak gain); the follow gain 20-1000 (the thread is cut earlier).

## To test in the language world (a copy run of 20-30 minutes each)
1. The episode chain kept per utterance, not in a slot's sixteen links. Ruler: the branch.
2. Store capacity and the tag tables. Ruler: fact prefixes across days.
3. The chooser: the striatum selecting among candidate continuations at a branch, with the no-go path (response inhibition), trained by the answer smile. Ruler: the branch, fact answers in conversation.
   DERIVED (2026-09-14, 23:35, before any run): the machinery exists. The actor is a linear head on the striatal delay line's expansion,
   its vote scaled by its earned voice; the planner deliberates among the cortex's candidates when they are torn, valuing each by
   imagination. It fails at a fact's start for two reasons the arithmetic settles: (a) the reward is starved, one to five answer smiles a
   day against thirty question-to-answer maps, at actor_lr 0.02 with deltas of 0.1-0.5, so a map gains 0.002-0.01 a smile and needs
   hundreds; (b) the "torn" test reads the logits with the store's vote in them, so where the store is sure and wrong nothing
   deliberates. What to change: the actor's rate at rewarded moments (a dopamine burst is large; a constant to sweep), the torn test on
   the cortex alone, the candidates from the cortex and the recall together. PREREQUISITE: copy days carry no reward, so none of this is
   measurable on a copy until the day's smiles are replayed with the lines (the caregiver log holds each smile's time and word): the
   reward-replay instrument (day_on_copy --rewards) comes first, and it makes 5, 7, 11 and 13 measurable on copies too.
   MEASURED (2026-09-15, 06:40): the reward replay works (277 faces land); the actor's earned voice does not move in a day at 0.02, 0.2
   or 1.0 (the voice is gated by a correlation over two hours of ticks); and the actor's raw vote at the branch is saturated, 'b' or
   'd' at +1.00 whatever the question, 0 of 9. DESIGN for the rebuild: a striatal head that scores only the CANDIDATES at a torn
   moment (the cortex's and the recall's top few) with a softmax over them, its input the delay line's expansion (which carries the
   question) and the candidates' embeddings; trained by the reward prediction error (the critic's baseline) with the eligibility on
   the candidate chosen, decaying over the answer's ticks; its weights bounded by decay so it cannot saturate; its vote entering the
   readout at a gain earned by its correlation with reward as now. Rulers: the actor's raw vote at the branch (branch_probe) on copies
   with rewards, then fact answers in conversation.
4. The actor's earned voice measured again with the chooser in place. Ruler: the actor's slope.
5. The handoff: the rulers read with the store off. Ruler: answers with the store off.
6. Old memories in the night's draw. Ruler: old lines, held-out.
7. Reward-weighted replay. Ruler: the fact sentences per night.
8. Waking plasticity gated by the smile and by surprise. Ruler: dusk-to-night deltas.
9. Event boundaries from surprise as well as silence. Ruler: the branch, held-out.
10. The learned working-memory cue replacing the hand-shaped context. Ruler: the branch, held-out.
11. Tonic dopamine from the running reward rate setting vigor and the gate's rate. Ruler: duty, smiles per line.
12. An acetylcholine-like signal tipping the store between writing and recalling by surprise. Ruler: held-out, answers by the pause.
13. The exploration gain driven by uncertainty in place of a constant. Ruler: duty, answer smiles.
14. Awake replay during the pauses. Ruler: the fact sentences, held-out.
15. Synaptic homeostasis in the night (renormalizing, not only strengthening). Ruler: held-out, old lines.
16. Noise: a copy day of dropped and swapped letters. Ruler: all four on the corrupted copy.
17. The rephrased-question ruler (before any recording).
18. The exchange replay revisited once the chooser exists. Ruler: the branch.

## Body only (a simulated body first; each with the same rulers carried over)
19. A learned sensory front end.
20. Continuous action with a forward model (the cerebellum), and reflexes and pattern generators under it.
21. Credit over seconds: eligibility traces, dopamine ramps.
22. The patience signal: a serotonin-like tone choosing the timescale the actor and critic listen to.
23. The stress switch: a gain on the slow bands' reach, moderate stress raising it, high stress cutting it.
24. Imagination as search over candidate actions (REM's critic where a choice depends on it).
25. Habit against goal-directed control: the arbitration with repetition.
26. The cognitive map: place and grid cells.
27. Drives (the hypothalamus): hunger, energy, temperature as reward's ground.
28. Threat and pain (the amygdala): a fast value tag, one-trial fear.
29. Sensory gating and attention (the thalamus).
30. Interoception (the insula).
31. A circadian clock beside fatigue.
32. Joint attention and social reward (oxytocin).
33. Real sensor noise.
34. Development: plasticity that opens and closes on a schedule; neurogenesis as the store's growth.
