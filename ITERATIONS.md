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
- The chooser (actor_form softmax): learned the action prior from ten relevant smiles a day and thinned the mouth; one served day lost (2026-09-15).
- The exchange replay (dream_pair): no preference at the branch after three copy nights.
- The cortex-state key (key_form cortex, pattern-separated): 0/30; the state is the whole window.
- The context at weight 1.0 (loops); the query horizon 0.9 (no peak gain); the follow gain 20-1000 (the thread is cut earlier).

## To test in the language world (a copy run of 20-30 minutes each)
1. The episode chain kept per utterance, not in a slot's sixteen links. Ruler: the branch.
   MEASURED (2026-09-14, 21:04, day 207 on a copy): the branch equal (6 of 9, 8 of 12 both ways), the answers 11/11/5 against 8/8/7
   across the pauses: within a day's noise. NEUTRAL; off. The floor and the capacity, not the chain's form, were the answer's gates.
2. Store capacity and the tag tables. Ruler: fact prefixes across days.
   DERIVED (2026-09-15, 13:15): the store holds 8192 slots and a day writes about 3500 world symbols, so two fifths of it turns over
   every day; a fact's chain survives only until stronger writes displace it, and the facts come once in three days. Bracketed on
   the served body: the prefixes 20 of 26 at 11:48 and 10 of 26 at 12:36 the same day, the store full at every strength above
   0.54. The hippocampus holds weeks, not two days. Mechanism: the slot count as a constant (store_cap), raised to 32768, four
   times the writes of a fact's cycle. Predicted: the prefixes hold their morning count through the day; the tick's read costs
   four times as much (the matmul over the keys), to be measured before the switch.
   MEASURED (13:20): day 228 wrote 1240 new slots into the full store and evicted 1238, every one at strength 0.54-0.59 (the
   memories heard once; the kept mean 0.86); of the 20 prefixes the morning save answered, 14 read a slot at weight near 1.0 that
   the dusk no longer held. The dusk's body with the union of both stores (9430 slots, what a larger cap would have held) answers
   19 of 26 against the dusk's own 10 and the morning's 20, and 14 of 30 questions at two rests against the dusk's 9. The read at 32768 slots costs 5.1 ms against 1.5 (measured), the
   self-confidence now in row blocks. CONFIRMED; store_cap 32768 goes on the served body at night 201's save (the boundary),
   the falsifier the next dusk: the prefixes hold their morning count through day 230 or the cause was not the eviction.
   LIVE (15:25): the capacity went on at night 201's save (14:11). The first day at it (the log's 231; facts 21-30 taught): the
   prefixes 11 of 26 in the morning -> 22 at dusk, the held-out 0.659 -> 0.667 (the first dusk above its morning in the stage),
   the store 8192 -> 10318 slots, nothing evicted. The prediction held and more: LANDED.
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
   ON THE SERVED BODY from night 198's save (2026-09-15, 10:45) and OFF from 11:50: its first live day collapsed (48 smiles, 99 junk characters, duty 0.158; the guard tripped). Read on a copy: the head learns the day's action prior through the state's mean direction and votes it back into the mouth, which loops and thins. FAILED for this world's reward density; kept as code, off; it belongs to a body with denser, more specific reward.
4. The actor's earned voice measured again with the chooser in place. Ruler: the actor's slope.
   MOOT (2026-09-15): the chooser failed live (iteration 3); the earned voice stays measured as it was (0.036-0.063, unmoved by a day).
5. The handoff: the rulers read with the store off. Ruler: answers with the store off. READ ALREADY by qa_by_gap's 'cortex alone' column: 0 of 30 every night; the cortex answers no question on its own yet.
6. Old memories in the night's draw. Ruler: old lines, held-out.
   DERIVED (2026-09-15, 10:50): the night draws its 1024 utterances by strength, which is recency (0.97 a night), from a memory of
   about 4096 utterances, twenty days of speech; the held-out lines are in the fourth stage's style and the days are now fact-heavy
   short questions, so each night's lesson leans to the newest style and the held-out drifts (0.689 to 0.654 over the six nights
   since the revert; the old lines near 0.48 for a week) while the fact sentences climb. Biology's replay reaches remote memories
   too. A share of the draw taken uniformly over the whole memory (dream_old_share, a constant) is the mechanism; predicted: the
   old lines and the held-out up by one to two hundredths a night, the fact sentences' climb slower by less than that. Measured
   by two copy nights from the same save, with and without the share.
   MEASURED (11:20): rejected. With the share the held-out fell 0.013 and the old lines 0.005; without, flat. The memory's twenty days are all of stage five, so its old is the new's style; the held-out's drift is the day's erosion, not the draw's.
7. Reward-weighted replay. Ruler: the fact sentences per night.
   DERIVED (2026-09-15, 12:15): an utterance enters the memory at strength 1 and fades 0.97 a night; the night draws by strength, so by
   recency alone. Dopamine tags what precedes it, and replay favours the tagged: the reward the child earns in the turn after a world
   line belongs to that line. Mechanism: when the next world utterance is pushed, the previous one's strength is raised by
   reward_gain times the smiles' dopamine integrated since it (a constant, disclosed). Predicted: the fact exchanges, which draw the
   answer smiles, are replayed about twice as often; the fact sentences climb faster by a hundredth or two a night; the held-out
   unchanged or a hundredth down. Measured: a copy day with the rewards replayed, then a copy night, against the same without the
   weighting; the fact sentences and the held-out after the night.
   RUNNING (14:15): arm T (reward_gain 2.0) lived day 224 at the old cap; the flags file took the capacity at 13:22, so arm C's day
   runs at 32768. The night's inputs (the utterance memory and its strengths) do not depend on the store's cap, so the fact
   sentences and the held-out compare cleanly; the questions and the prefixes, the store's, do not, and are not read for this pair.
8. Waking plasticity gated by the smile and by surprise. Ruler: dusk-to-night deltas.
   DERIVED (2026-09-15, 11:25): the day's lesson runs every 24 ticks at 1e-5 on the last 32 ticks, all day, scaled only by stress;
   each dusk the held-out sits two to three hundredths under the morning and the night restores one to two, a net drift down. In
   the brain the waking write into cortex is gated by dopamine and acetylcholine: the rewarded and the surprising moments are
   written, the rest weakly. Mechanism: the lesson's rate multiplied by (wake_base + wake_dopa x |dopamine| + wake_novel x the
   running surprise), constants disclosed; with wake_base 0.3 the ordinary tick learns at a third, a smile's tick at full or more.
   Predicted: the dusk's fall on the held-out halves, the fact sentences (rewarded exchanges) keep their climb. Measured by two copy
   days with the rewards replayed, gated and ungated, the held-out read before and after the day.
   MEASURED (13:16): day 224 lived twice on the save after night 198 (the held-out 0.654, the prefixes 21 of 26 before the day).
   Gated (0.3 + 1.0 x |dopamine| + 0.5 x surprise): held-out 0.638, prefixes 22, questions 17 (pause 2) and 14 (pause 8).
   Ungated: held-out 0.644, prefixes 18, questions 17 and 13. The prediction FAILED: the dusk's fall on the held-out did not halve
   (it was the same within a hundredth, both arms two hundredths under the morning); the prefixes' four are within a day's noise
   and were mostly the eviction below (iteration 2), not the lesson. Not adopted; the constants stay at their off values.
9. Event boundaries from surprise as well as silence. Ruler: the branch, held-out.
   DERIVED (2026-09-15, 13:30): a boundary ends an episode tag; the tag scopes the follow links the recall runs along. Today the
   boundary is the offset (silence). A surprise boundary would split a line at its surprising symbol, "yes. take some with honey"
   into two episodes. The branch reads the chain within a sentence, unchanged either way; the held-out is the cortex, which the
   night trains on the utterances whole from the utterance memory, not on the store's episodes. Predicted: neither ruler moves by
   more than its noise (the branch 1 of 9, the held-out 0.005). NOT RUN: no ruler on this world can see it.
10. The learned working-memory cue replacing the hand-shaped context. Ruler: the branch, held-out.
   DERIVED (13:30): the fast bag is a fixed code (shift and decay); a learned cue needs a signal that says which keys should be
   near and which far. The cortex's state as the key (iteration 27 of the spec) was the learned cue without such a signal, and its
   state was global: the branch fell to 1 of 9. The signal the dentate gyrus has is pattern separation learned unsupervised from
   the input's statistics, not from reward; on a lexicon of a hundred symbols and lines of forty, the fixed shift-and-decay already
   separates by position and content. Predicted: no lift on the branch (8-9 of 9 at the floor). NOT RUN until a ruler fails on it.
11. Tonic dopamine from the running reward rate setting vigor and the gate's rate. Ruler: duty, smiles per line.
   DERIVED (13:30): the running reward (rbar) exists; vigor would scale the gate's readiness by it. The rulers it moves are duty
   (0.33) and smiles a line (1.28), neither a demo ruler; the answers by the pause and the prefixes are the store's, not the
   gate's. Predicted: duty follows the smile rate within a day, the demo rulers unmoved. DEFERRED to the body-general list.
12. An acetylcholine-like signal tipping the store between writing and recalling by surprise. Ruler: held-out, answers by the pause.
   DERIVED (13:30): the write is already scaled by surprise; the recall's share of the forecast is constant. The tip would suppress
   the recall while a novel line is heard (encoding) and free it in the familiar (retrieval). The question is familiar when asked,
   so the answers by the pause are read under the same recall either way; the held-out is teacher-forced on the cortex. Predicted:
   both rulers within noise. What it would change is the mouth's interference during a new line, which no ruler reads. DEFERRED.
13. The exploration gain driven by uncertainty in place of a constant. Ruler: duty, answer smiles.
   DERIVED (13:30): the choice's exploration is a constant (explore_choice 1.0). By uncertainty it would speak less where the
   forecast is sure and more where it is not; the answer is spoken where the recall is sure, so the answer smiles stay; the
   junk (own symbols where nothing is sure) would fall. Predicted: the demo rulers unmoved, the own-symbol junk down by a third.
   DEFERRED behind the capacity's falsifier; one copy day with the rewards replayed when the served body is steady.
14. Awake replay during the pauses. Ruler: the fact sentences, held-out.
   DERIVED (13:30): the night replays the utterances whole a thousand times a night and the cortex alone still answers 0 of 30:
   the cortex's failure at the question is structural (the answer sits across a pause the window carries but the lesson does not
   bridge), not a shortage of replay. Replay in the day's pauses adds the same lesson earlier. Predicted: the fact sentences'
   cortex loss unchanged within 0.01 (0.88 already), the answers by the cortex alone still 0. NOT RUN.
15. Synaptic homeostasis in the night (renormalizing, not only strengthening). Ruler: held-out, old lines.
   DERIVED (13:30): the held-out drifts down (0.689 to 0.654 over six nights) as each night's lesson leans to the newest style; a
   renormalisation (weight decay in the night's steps) shrinks all weights alike and does not choose between the old style and the
   new. Predicted: the held-out within 0.005 of the plain night on a copy night. NOT RUN; the drift's cure is the draw (iteration 6
   failed on it) or a steadier world, not the weights' norm.
16. Noise: a copy day of dropped and swapped letters. Ruler: all four on the corrupted copy.
   DERIVED (13:30): a measurement, not a mechanism: a day whose lines lose one letter in twenty. The store's keys are the last
   five symbols, so a dropped letter breaks the chain at that point and the follow links recover it two symbols on. Predicted:
   the prefixes and the answers within 2 of the clean day's; the held-out (read on clean text) within 0.005. TO RUN once the
   served body is steady under the capacity (a day's copy at the usual cost).
35. The fact told once (the demo's central scene, before any recording). Ruler: tools/once_told.py, ten never-typed facts.
   DERIVED (2026-09-15, 15:40, before the run): a fact told once is written at the surprise's strength (about 0.55) as a chain from
   the question's last symbols through the pause into the answer; the read is by content, and a never-typed question ("what is
   sour?") has no near key among the taught ones (the taught questions share "what is " and differ by one word, the branch's
   problem), so its recall is less crowded than a taught fact's. The night fades the strength by a tenth and evicts nothing at the
   capacity; the cortex's lesson on the day's utterances moves the forecast, which the recall's confidence outweighs on a unique
   chain. Predicted: at least 6 of 10 answered at two rests the same day, and the same count within one after the night; the
   cortex alone 0. Falsified if the same-day count is under 4 (the single write too weak against the forecast) or the night's count
   falls by three or more (the night's lesson overrides the store's chain).
17. The rephrased-question ruler (before any recording). DONE 2026-09-15: tools/heldout_rephrased.txt, qa_by_gap --set=rephrased; first reading 8 and 7 of 30 at two and four rests against 16-18 on the taught wording.
18. The exchange replay revisited once the chooser exists. Ruler: the branch.
   MOOT (2026-09-15): the chooser failed live; the exchange replay stays as measured (failed).

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
