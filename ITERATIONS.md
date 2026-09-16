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
   DERIVED AGAIN (2026-09-15, 22:05): under the hold the answer chains carry the question in their keys and merge less, and the
   store grows 2200 slots a day (18966, 21209, 23310 over nights 206-208) against 1500 before; at that rate it fills in four
   days, and the eviction of the weakest (the newest one-shots) returns. The fade's horizon is nineteen nights, so the capacity
   should stand above 19 x 2200 = 42000: the next constant is 65536 (the read 13 ms a tick, measured; the save 1.6 GB). To be
   switched at a night's save if two more nights confirm the rate; the falsifier as before (the prefixes hold through a day).
   ARMED (2026-09-15, 23:58): the rate held (2200, 1983, 1957 a day; the store 27250 after night 210): --store-cap 65536 goes on
   the served body at night 211's save (the 32768 set kept as BASE_FLAGS_cap32k.txt). The falsifier as before.
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
   MEASURED (15:40): with the tag, the night took the held-out 0.661 -> 0.669 and the fact sentences to 0.888; without, 0.646 ->
   0.646 and 0.881. The direction is the predicted one on both rulers, but the two dusks differ by 0.015 from the day alone (the
   tag does nothing to the cortex by day; two copy days from one save diverge on the store's reads), so one pair cannot resolve a
   hundredth. INCONCLUSIVE; not adopted. Next: a matched night pair from the tagged day's save, the control with the tags zeroed
   (the strengths above one clipped to one), the same seed: the only difference the draw's weights. The chain had removed its
   day saves, so the matched pair needs the tagged day lived again (an hour and a half of copy time): QUEUED behind the demo's
   gates (the told-once test, the recording), not before.
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
   RE-DERIVED (2026-09-16, 00:55): since the hold went on (night 206) the child answers before the other voice on three
   questions in four and speaks whole utterances of its own, and the held-out has drifted down five nights running (0.657,
   0.649, 0.640, 0.644, 0.635 after the nights; 0.647, 0.638, 0.629, 0.619 at the dusks). The day's lesson learns from windows
   full of its own fluent speech (heard at 0.3, the targets the world's quiet), a conflict the babbling child never posed. If
   that is the drift's source, the day's lesson at a third (wake_base 0.3) cuts the dusk's fall from about 0.025 to about 0.01
   on day 243 lived on the save after night 211 with the rewards replayed, the facts by the cortex likewise; falsified if the
   two dusks lie within 0.005 of each other (then the source is the day's content, not the lesson's rate).
   MEASURED (2026-09-16, 02:46): day 243 relived on the save after night 211 with the rewards replayed: ungated, the held-out
   0.635 -> 0.645 and the facts 0.877 -> 0.885; gated at a third, 0.650 and 0.881. Neither arm fell, while the served body fell
   to 0.619 on that same day; the two arms lie within 0.005, the falsifier. INCONCLUSIVE on the mechanism, and the instrument's
   limit is now plain: a copy day's mood sinks to -5 under replayed rewards, which are not contingent on what the copy's child
   says (the served child's smiles land on the copy's silence), so the copy speaks under a floor-flat readout and never
   reproduces the served day's fluent turns, which are the suspected source. The gate stays off. The served held-out flattened
   at night 212 (0.633, the dusk's fall 0.009); watched, not acted on. A faithful test needs a caregiver in the loop on the copy
   (smiles contingent on the copy's own words, as live_qa does for questions), an instrument to build if the drift resumes.
   LIVE A/B DERIVED (2026-09-16, 09:35): the copy days cannot reproduce the served day (the mood artifact), so the test moves to
   the served body at a boundary: the day's lesson at a third (wake_base 0.3) from night 221's save for two days. The ruler is
   the dusk's fall (the dusk's held-out minus the morning's): the last four days -0.029, -0.018, -0.019, -0.029. Predicted: the
   falls halve, at or above -0.015 on both days, and the morning values stop declining; the facts by the cortex at dusk fall
   less (0.836-0.849 lately against 0.883 in the morning). Falsified if either day falls -0.025 or more, or the parent's counts
   (answered before B, answer smiles) drop by a fifth: then back to 1.0 at the next save (BASE_FLAGS_wake1.txt).
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
   REVISED (16:58, after the live ruler): the derivation above was wrong about the demo rulers: the live mouth answers 3 of 30
   where the greedy readout answers 20, because the choice samples each word's first symbol at a fixed sharpness. Biology's
   selection is a winner-take-all whose noise falls as the evidence rises (the basal ganglia's threshold under dopamine; the
   drift-diffusion of a decision: more evidence, less variance at the bound). Mechanism: the readout's sharpness at the choice
   scaled by the forecast's certainty, sharp x (1 + sharp_conf x |pred|), the norm the same certainty the gate's salience reads
   (a constant, disclosed; 0 = as now). A sure recall (norm near 1) is read at double or more the sharpness; a babble (norm 0.2)
   nearly as now. Predicted: with sharp_conf 3, the taught questions answered live at 12 or more of 30 per pass, the never-typed
   6 or more of 10, the day's own-symbol junk not up (a sure forecast was decisive already; the unsure is untouched). The bound
   first: sharp_base 60 and 100 on the live ruler; if 100 does not lift the live count past 10, the choice's sharpness is not
   the cause and the trace of a single question tick by tick comes next.
   RE-DERIVED (2026-09-16, 04:20, from the rehearsals on the served body): three facts told once before a night answered 3 of 3
   minutes later and 1 of 3 after the night, live; read greedily on the morning save they answer 3 of 3 verbatim and seven facts
   never told 0 of 7. The memory is intact; the live loss is the choice: after "what has wool?" the cortex's habit ("i do.",
   "yes.") and the sure recall ("a sheep has wool") compete for the first symbol, the sum's argmax is the recall's, the margin
   is thin, and the sample lands on the habit half the time. The copy measurements of sharp_conf were made at a floor mood (the
   readout at 8, where a fourfold sharpening only restores the healthy 25) on crowded taught questions; this is the uncrowded
   case at a healthy mood (the readout 23-24), where the certainty is high exactly at the answer. Predicted with sharp_conf 3 on
   the served body: the once-told facts answered live after a night at 2 of 3 or better (from 1 of 3), the same day 3 of 3; the
   taught live count not down; the parent's answer smiles and answered-before-B not down; junk not up. Falsified if the
   after-night count stays at 1 of 3 over two sets of three, or the parent's counts fall by a fifth.
   MEASURED LIVE (07:22): under sharp_conf 3 the told facts answer 7 of 18 after a night and 7 of 12 the same day across five
   sessions (the taught questions 16 of 30); the greedy readout holds them. The losses sit at word boundaries ("a pear is
   shall", "a lemon i" then "do."): the cortex's habit wins the sample where the recall's margin is thin. Predicted with
   sharp_conf 8 (a sure forecast read at nine times the sharpness, an unsure one at five): the nine told facts answered mid-day
   at 6 of 9 or better; the day's junk and the parent's counts unchanged. Falsified under 5 of 9 (then back to 3), or if the
   parent's known-word growth stalls over two days (the exploration lost).
   MEASURED (08:10, the sixth rehearsal, begun at mood +4.6): under sharp_conf 8 the nine told facts answered 4 of 9 after their
   nights, the three told twice 1 of 3: FALSIFIED (under 5 of 9). Across the six rehearsals no value separates: after a night
   1 of 3 at 0; 3 of 3, 1 of 6, 2 of 6 at 3; 4 of 9 at 8. The choice's sharpness is not the lever of the live variance. What
   the rehearsals show instead: the answer usually begins right and derails at a common-word branch ("a pear is a " never
   reaches "fruit"; "owls hoot", "the sea is salty", "a hill is steep" survive), and the same-day recall (7 of 12) beats the
   overnight (11 of 27). OFF from night 219's save (no measured lift; the law). The demo's told fact must have a distinctive
   continuation and a question unlike the session's others.
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
   MEASURED (16:12, on a copy of the save after night 202): the same day 10 of 10 at two rests and 9 at eight; after a night on the
   copy 10 of 10 and 8 of 10, every answer the fact's sentence verbatim ("a lemon is sour", "the sea is salty", "a sheep has wool");
   the cortex alone 0 throughout; the taught questions on the same copy 17 of 30 (the served 18). LANDED, above the prediction.
   The never-typed facts answer better than the taught ones because their question's key is uncrowded: the taught questions have
   been asked in ordinary talk with other answers ("what is hot?" -> "the tub is hot"), so the store holds several continuations
   under one key and the recall splits among them. The taught questions' 60% is the crowding of the key, not the memory's strength.
36. The questions asked of the live mouth (the demo's form: the gate and the sampled choice, not the greedy probe). Ruler:
   tools/live_qa.py, the question typed as the parent, twelve seconds of the child's turn read from the page.
   DERIVED (2026-09-15, 16:20, before the run): the greedy probe reads the argmax at every symbol with the recall in the forecast;
   the live mouth speaks when its gate opens and samples its choice (explore_choice 1.0), and the parent's silence after the
   question is the only cue. The taught questions answer 18-20 of 30 greedily on the dusk copy; the live form loses some to the
   gate's timing and the sampling. Predicted: 12-15 of 30 taught per pass (three passes), the never-typed facts 6-8 of 10 on the
   told-once copy. Falsified if the taught count is under 9: then the demo's gate is the choice and the gate, not the memory.
   MEASURED (16:56): FALSIFIED. Taught: 5, 1, 3 of 30 over three passes (7 ever); never-typed: 4, 0, 1 of 10. The answers are
   there but late and garbled ("one little fishice is co", "lemon is sourlwet", "he sea is salty"): each word's first symbol is
   sampled from the readout at a fixed sharpness (25) and the word then runs greedily, so a three-word answer needs three lucky
   starts where the forecast's margin is thin; and the instrument typed the next question over the child's speech (the typist
   waits for its quiet). The memory is not the demo's gate; the choice is. Next: the instrument waits for quiet; the bound with
   the readout near-greedy (sharp_base 60, 100); then the certainty-scaled decisiveness (iteration 13's form, revised).
37. The hold of working memory in the quiet (bag_rest_decay). Ruler: the live mouth; the answers by the pause at 8 and 16.
   DERIVED (2026-09-15, 17:20, from the live ruler's trace, before the run): after "what is hot?" the readout's argmax is 't' at a
   sure forecast for five ticks, but the gate, taught caution by the talk-over frowns, stays at its floor for twelve to forty ticks;
   the world's context fades 0.8 a tick in the quiet, so by then its norm is a tenth, the recall (a dot product, the norm its
   temperature) has flattened to the generic mean, and the child says "yes.". The answer lives eight ticks and the gate opens
   later. In a brain the cue of a question is held through the pause by prefrontal delay activity; a context fades as symbols
   displace it, not by the clock. Mechanism: the fast bags fade by bag_decay per symbol of their own kind and by bag_rest_decay
   per quiet tick (a constant; 0 = the old rule). The keys are unchanged in direction (the onset's key is the question's still,
   merged into the same slot), so the store's structure does not move; only the cue's norm through the pause. Predicted at 0.97
   (a norm of 0.54 after twenty quiet ticks): the greedy answers at eight rests rise to the count at two (20 of 30) and hold near
   it at sixteen; the live mouth answers 12 or more of 30 taught and 6 of 10 never-typed; a line's context is a tenth after 75
   quiet ticks, the typist's gap, so the mouth does not chain across lines. Falsified if the live count stays under 8 (then the
   gate's latency or the sampling is the gate, not the fade) or the branch or held-out fall.
   THE BOUND (17:53): with the instrument waiting for the child's quiet, sharp_base 25 answers 4 and 6 of 30 (8 ever), sharp_base
   60 answers 2 and 1: sharper is worse. The sampling is not the cause; the trace's reading stands (the gate's latency against the
   cue's fade, iteration 37). A confound found in the instrument: it gave no smiles, so the copy's mood sank through the run and
   its readout's sharpness fell to the floor (8); the instrument now smiles at an answer as the caregiver does (--smile 1).
   The third point (18:10): sharp_base 100 answers 4 and 4 of 30 taught (as at 25) and 3 and 4 of 10 never-typed (against 2 and
   1 at 25), the mood at -6 throughout for lack of smiles. Sharpness alone moves the uncrowded questions a little and the
   taught ones not at all: the lever is the cue's hold through the gate's latency (iteration 37), measured next with smiles.
   MEASURED (19:00, the dusk copy of 232): greedy by the pause at 0.97: 19, 20, 23, 20 of 30 at 2, 8, 16 and 32 rests (against
   20 and 14 at 2 and 8 without the hold): the answer survives the pause, the first half of the prediction met. Live, with the
   instrument smiling at answers only: the control 4 and 3 of 30; with the hold 14 and 4 of 30 (15 ever) — the first pass past
   the predicted 12, the second sunk with the copy's mood at -6: the instrument's smiles were a tenth of the world's (the
   caregiver smiles at every known word) and the readout's sharpness fell to its floor. The never-typed 3 and 2 of 10. The
   instrument now smiles as the caregiver does; the mood-fair pair (control against 0.97, and the never-typed) runs next.
   THE TRACE WITH THE HOLD (19:13): after "what is hot?" the forecast's norm stays at 1.1-1.4 for eleven ticks with 't' on top
   (against a fall to 0.5 and 'y' without the hold); the gate opens at the eleventh tick (2.2 s) and the child says "th m for..."
   (this question is crowded on the copy: the greedy readout says "the duck does." to it too). With sharp_conf 3 on top of the
   hold, 10 and 8 of 30 at mood -6 (the answers-only smiles): no lift over the hold alone; the decisiveness is not the lever.
   MEASURED, MOOD-FAIR (19:49; the instrument smiling as the caregiver does, 90-121 smiles a run): the control answers 3 and 2 of
   30 live; the hold 12 and 12 of 30 (18 ever), the predicted count met on both passes at the copy's floor mood; the never-typed
   6 and 1 of 10 (7 ever). LANDED on copies. ON THE SERVED BODY from night 206's save (~20:10; BASE_FLAGS, guard_args and
   serve_command carry --bag-rest-decay 0.97; the pre-hold set kept). The falsifier live: the parents' fact words in the child's
   turn before B (2-3 of 10 a day) and the whole-fact anticipation smiles rise over the next days; the held-out and the branch
   hold; the child does not chain across lines through the pauses (junk and talk-over frowns not up).
   ON THE SERVED SAVE (20:34; the save after night 206, the caregiver's smiles): with the hold 10 and 9 of 30 live (12 ever),
   without it 1 of 30. The instrument undercounted the hold's answers ('ce is coldd', the 'i' said in the question's last tick);
   fixed after this reading. The first day under the hold on the served body is read at night 207 (the parent's counts).
   NIGHT 207, THE FIRST DAY LIVE (21:05): the parent's questions answered before B 36 of 91 (25 the day before), the answer
   smiles 10 (9), duty 0.294 (0.337), frowns 40 (38); but the greedy questions at two rests 21 -> 16 and the rephrased 13 -> 8
   while the count at eight rests rose 14 -> 17. DERIVED (21:10): a flaw in the form. The query shifts the world's context a lag
   for every symbol the body says, but under the hold that context faded at the quiet rate while the body answered, whereas the
   keys were written with it fading 0.8 per symbol of the other voice's answer: the query's geometry during its own answer no
   longer matched the keys. The context must fade by the symbol rate for every symbol, whoever says it, and by the quiet rate
   only when no one speaks (bag_own_fade 1, a disclosed switch; 0 = the form served from night 206). Predicted on the save after
   night 207: the questions at two rests back to 20 or more, at eight and sixteen unchanged or up; the live count unchanged or
   up. Falsified if the two-rest count stays at 16: then the drop was the day's writes, not the geometry.
   AN INSTRUMENT'S LESSON (21:40): a save carries its own constants (the blob's cfg), and a flags file that lacks a key leaves the
   save's value standing, so "the pre-hold flags" on the save after night 207 ran at 0.97 and printed the hold's own numbers as
   the control (11 of 30 live, interleaved, both arms identical). A control must pass the zero explicitly (--bag-rest-decay 0).
   Rerun queued. The interleaved live with the hold on that save: 11 of 30, the copy's mood at the floor throughout.
   MEASURED (21:48, the save after night 207, greedy by the pause): the hold as served 16, 17, 18 of 30 at 2, 8, 16 rests; with
   own symbols at the symbol rate 19, 20, 18; without the hold 19 and 14; the branch 9 of 9 in both forms. The form's fix restores
   the two-rest count to the save's own ceiling (the no-hold 19) and keeps the pause's gain (20 against 14 at eight). Within one
   of the prediction; the falsifier not met. ON THE SERVED BODY from night 208's save (~22:10): --bag-own-fade 1 beside the hold
   (BASE_FLAGS, guard_args, serve_command; the previous set kept as BASE_FLAGS_pre_ownfade.txt).
   THE FORM'S SECOND FLAW (23:09, night 209, the first day at form 1): the ten facts retaught that day answered 3 of 10 at two
   rests while the twenty taught under the served form answered 7 of 10 each. On the log, the child spoke before the other
   voice on nine of those ten questions ("ice", "fish live", "birds", "rain falls": the live answering itself), and form 1 faded
   the world's context by those own symbols, so the other voice's answer was written under a context faded to nothing: one
   fact's retellings landed in two key forms. The keys must be the world's alone; the symbol-rate fade for own symbols belongs
   in the query, beside the efference copy's shift (form 2, bag_own_fade 2; test 64: the state as form 0, the query as form 1).
   On the save after night 209 form 2 reads 17 and 17 of 30 at two and eight rests, form 1's numbers exactly (the same query).
   ON THE SERVED BODY from night 210's save (~00:05). Predicted: the facts retaught from now on keep 7 of 10 or better within a
   night of their retelling; facts 1-10 recover when retold (about three days); the parent's answer smiles hold or rise.
   REHEARSED ON THE SERVED BODY (2026-09-16, 03:24; tools/rehearse.py: the typist frozen, the questions typed over the page, the
   smiles contingent): the taught questions 1-15 answered 8 of 15 in the child's turn, the answers arriving 0.6 to 7.4 s after
   the question ("the sun", "water is wet", "cats drink milk", "rain falls from the sky", "an apple is red", "an ant is little",
   "we eat bread and"); three never-typed facts told once at the session's start answered 3 of 3 ("a lemon i", "a sheep has
   wool", "the sea") at 2 to 5 s. The mood fell -2.5 -> -6.0 over sixteen minutes: the rehearsal's parent smiled only inside
   short windows where the typist smiles at every known word whenever said; the later answers came under a floor-flat readout.
   The rehearsal's parent is made attentive through its waits before the after-night run.
   LIVE, AFTER A NIGHT (2026-09-16, 06:19; the rehearsals on the served body): a fact told once answered after its night 1 of 3,
   3 of 3 (two nights, decisiveness on), 1 of 6 (mid-day, the mood -2 to -4); the same day 3 of 3 and 1 of 3. Greedily on each
   morning's save the told facts answer and the untold do not. The store keeps the once-told fact; the live choice recalls it
   unreliably, and the mood is the strongest modulator. The parent's natural repetition is the next mechanism to measure: the
   fact told twice in a session (a merge, the slot's strength doubled), asked mid-day after its night. Predicted: 2 of 3 or
   better live; falsified at 1 of 3 or worse over two sets.
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
