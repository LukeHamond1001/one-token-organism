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
   DERIVED A THIRD TIME (2026-09-16, 17:14): the store reads 64565 against the capacity of 65536 (nights 226-228 wrote 3200 a
   day, the fade dropped 1100-1150 a night): at the cap the daily eviction of the newest one-shots returns. A larger capacity
   cannot be paid on this disk (each save two and a half gigabytes, ten copies rotating). The fade's floor is the other
   constant: with the mean strength at 0.28 now (most slots one-shots), the floor at 0.1 x mean lets a once-heard memory live
   twenty-eight nights, so the store's steady state is ninety thousand; at 0.25 x mean it lives nineteen or twenty nights,
   the horizon the capacity was derived for, and the steady state about sixty thousand. store_floor_rel 0.25 from the next
   save. Predicted: the nightly drops rise toward the day's writes and the store levels under the capacity within a week
   with no eviction at the cap; the prefixes and the questions hold. Falsified if the store shrinks under forty thousand, or
   the prefixes fall by three: then back to 0.1. (The fade itself, 0.9, is left alone: the utterance memory fades with it.)
   FALSIFIED (2026-09-16, 19:28): night 229 purged 18378 slots (64000 -> 47158) and night 230 another 7673 (-> 42724), and the
   prefixes fell 23 -> 20 of 26, the falsifier's clause. The derivation missed that a floor RELATIVE TO THE MEAN feeds itself:
   removing the weak raises the mean, the floor climbs, and the next night removes more. Back to 0.1 at night 231's save. The
   right form is an absolute floor set by the write strength and the fade (a once-heard memory, 0.55, reaches 0.07 after
   twenty nights at 0.9), which does not move as the store thins: store_floor_abs, to derive and build after the recording.
   DERIVED A FOURTH TIME, THE ABSOLUTE FLOOR (2026-09-17, 00:40; the cap seven days off, before the recording after all):
   measured on the morning saves after nights 234 and 235 (tools/store_turnover.py, with tools/store_dist.py): day 278 wrote
   2745 new slots (beside 496 merges that strengthened a slot, x2.2 each), and the relative floor at a tenth of the mean (0.034)
   dropped 26; the store 44006 -> 46725. The new slots' write strength (the surprise x (1 + |dopamine|)): median 0.72, a quarter
   under 0.16, a tenth under 0.065. Under the fade 0.9 a slot written at s lives ln(F/s)/ln(0.9) nights above an absolute floor
   F; the steady state is the day's writes times the mean lifetime plus the merges' extension (about 3700): F 0.05 -> 61000,
   0.06 -> 56600, 0.07 -> 53000, 0.08 -> 50000; the relative floor as it stands (0.034) gives 70000, above the cap, so the
   cap's eviction by strength at the moment of writing would be the forgetting. At 3200 writes a day (nights 226-228) 0.06
   reaches the cap and 0.07 levels at 61000. THE CONSTANT: store_floor_abs 0.07, a tenth of a typical write (a memory is lost
   when its trace has faded to a tenth of what one hearing writes): a median memory heard once lives 23 nights, one at full
   surprise 27; a fact retold every three days merges and lives on. The floor does not move with the store's mean (test 65),
   and the dream floor (a dream stops at a slot below the forgetting floor) follows the same constant. The first night under
   it forgets the slots at 0.065-0.078 (2257 of 46725 on the save after night 235: the last two days' low-surprise writes,
   memories of what the cortex already predicted) against 26 under the relative floor, then about the day's writes a night.
   Falsifier: the prefixes or the questions fall by three on the cut copy against the same save untouched, or on the served
   body's first mornings under it; then back to the relative floor at a tenth, explicitly. ARMED: the cut at 0.078 (what the
   first night's fade takes below 0.07) measured on the save after night 236 against its morning probe; adoption at a later
   save by the reload, --store-floor-abs 0.07 explicit in every flags file.
   MEASURED ON THE CUT COPY (2026-09-17, 01:18; the save after night 236, 49601 slots): the cut at 0.078 forgot 4649 (9.4%: the
   low band the relative floor never removed, two days of low-surprise writes and the faded tail), the store's mean 0.327 ->
   0.355. Against the same save untouched: the prefixes 22 of 26 against 23 ("ice is " -> "coming tomor" against "cold": the
   weak trace of a well-predicted answer gone, a competing memory unmasked), the questions 17 and 19 against 18 and 20, the
   rephrased 12 = 12, the branch 9/9 and 10/12 = the same, the held-out and the old lines identical (the cortex untouched).
   Question by question (probe_lm --qa-all): three reads shifted, two lost ("what is big?", "what has wings?") and one gained
   ("what do cows give?" -> "cows give milk"); the rest word for word or the same verdict.
   One prefix and one question, under the falsifier's three. The catch disclosed: surprise-gated encoding writes a well-
   predicted fact weakly, and the weak traces are what an absolute floor forgets first; a retelling merges into a weak slot at
   nearly the full write strength (the increment m/(m+S) of it, 0.82 at S 0.07), so the three-day cycle keeps the thirty, and
   a fact told once at full surprise (the demo's scene) lives twenty-seven nights. ADOPTED at night 237's save: the reload
   armed at 01:20 with --store-floor-abs 0.07 explicit in BASE_FLAGS, guard_args and serve_command (the pre set kept as
   BASE_FLAGS_pre_floorabs.txt), the guard relaunched after it. Falsifier on the served body: the prefixes or the questions
   three under their range (20-23, 18-23) on the first three mornings under it; then --store-floor-abs 0 EXPLICITLY.
   LANDED (2026-09-17, 02:02, night 237's save; pid 87146; served_cfg verified). THE FIRST NIGHT UNDER IT (238): 8028 slots
   forgotten, the store 47967 by morning, the weakest at 0.070, the next band 2104; the curve 0.206 -> 0.154, the dreams
   1024 at 23.7 symbols; the morning read the prefixes 22 of 26 ("ice is " lost, as on the cut copy), the questions 19 and
   21, the rephrased 13, the branch 9/9 and 10/12, the held-out 0.607: nothing three under the reference. Mornings 239 and
   240 close the falsifier's window.
   STANDS (2026-09-17, 04:55): the three mornings under it read the prefixes 22, 22, 23 and the questions 19/21, 19/22, 21/22,
   the rephrased 13, 12, 14, the branch 9/9 and 10/12 each morning, the held-out 0.607, 0.604, 0.607; the store 47967, 48458,
   48766 (the growth 491 then 308 a night, settling under the capacity). The absolute floor is the served body's forgetting.
   THE PREFIXES UNDER IT (2026-09-17, 11:48; ten mornings): 22, 22, 23, 22, 19, 19, 19, 20, 19, 17 against 23 before. Traced
   with tools/read_trace.py on the last save under the relative floor and this morning's: the fact's slot at "ice is " is
   present with its strength on both, and a fresh slot from the parent's "ice is in your cup" (a closer key by 0.014) took
   the read; the reads are by content alone. Not the floor's doing: the key's collision with natural talk (the dentate item).
   FALSIFIED ON THE SERVED BODY (2026-09-17, 12:50; eleven mornings): the store's rulers fell in step under the floor, the
   questions at two rests 21 -> 16, the prefixes 23 -> 17, the rephrased 15 -> 10, the branch's second set 10 -> 8 of 12,
   while the held-out rose 0.607 -> 0.633. Read with tools/read_trace.py --qa on the real read (corrected after the review of
   2026-09-17, whose first form had scored the sentence's opener alone): the onset, the sentence's first symbol at the
   question, 25 of 30 now against 24 before (a weak reading, an article or the question's own word for most facts); the
   middle, the answer word's first letter with the sentence taken as the body's own up to it, 24 of 30 now against 26
   before, its mean share 0.77 against 0.81 ("a tree" 0.06 against 0.39, "water" lost to "ice"); the free-running ruler
   compounds such losses (21 to 16). The mechanism, supported, not proven: the middles of well-known
   utterances are written at a predicted symbol's surprise (0.05-0.15), under the floor, so the episode chains that carry
   an answer past its onset are cut at the first night; the relative floor kept those writes four nights and the retelling
   refreshed them. The derivation weighed the store's size and missed that the floor's height stands above the write
   strength of everything the cortex already knows. REVERTED at night 249's save (--store-floor-abs 0 explicit; the 0.07
   set kept as BASE_FLAGS_floorabs07.txt): the relative tenth and the capacity's eviction, the best-measured regime.
   The next form, if one is needed: the write strength of an utterance's chain set by the utterance (its onset's or its
   mean surprise), the episode encoded as a unit, so a fixed floor keeps or loses a chain whole; measured over the store's
   turnover on the served body, never on three mornings.
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
   DAY ONE LIVE (11:23): the dusk's fall +0.002 (0.623 -> 0.625) against -0.018 to -0.029 on the four days before; the old
   lines 0.481 after the night (0.455-0.471 for a week); the questions, prefixes and branch unmoved; the parent's counts within
   their noise (answered before B 69%, answer smiles 7). The second day decides.
   DAY TWO LIVE (12:19): the dusk's fall -0.035 (0.624 -> 0.589), the week's largest, after day one's +0.002; the old lines
   0.425 at that dusk. FALSIFIED by the second day: the day's fall is episodic, some days pulling hard and some not at all,
   and a third of the lesson's rate does not set it. OFF at night 224's save (back to 1.0). The drift's source stays open: the
   day's content (which lines, which of the child's own turns) rather than the lesson's rate; the next derivation must find
   what a hard day has that a soft day lacks (the caregiver log holds both: days 259 and 260 side by side).
   THE FALL TABULATED (12:21): twenty-two days aligned by time, each dusk against its morning: the falls scatter from +0.008 to
   -0.070 around a mean near -0.02 (a scatter of 0.018), and no day-level feature tracks them: the child's own characters
   (+0.09), its duty (+0.06), its turns of three words (+0.04), the parent's rare words (+0.09), near-new words (+0.03),
   talk-overs (+0.10), lines (+0.08). The nights restore a little less than the days take, a net -0.0016 a morning. The two-day
   test at a third could not resolve a third of the fall against that scatter (its falsifier was set too tight); a fair test
   needs about eight days per arm, deferred behind the demo. The revert stands, by the law.
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
38. The night's robustness. Ruler: the night's own loss curve and its gauge before and after; the morning rulers.
   OBSERVED (2026-09-16, 15:36): night 226's lesson diverged (the loss 0.20 -> 0.34 at the third round, the gauge on its dreams
   0.694 -> 0.605, the facts by the cortex 0.885 -> 0.646 by morning) on ordinary material; re-run on the dusk copy with other
   draws it ran normally. One night in two hundred and twenty-six. Not yet derived: the candidates are a gradient outlier
   meeting a fresh optimizer's early moments (night_warm 8), or a batch of near-identical dreams. The falsifier for any fix is
   the same night re-run with the served draws, which needs the draws logged: the night now records its draws' serials (to
   build). Nothing changes on the served body until the cortex's recovery is read over the next three nights.
   DERIVED (16:42): the night's lesson is a fresh Adam every night (moments 0.9 and 0.999, a warm-up of eight steps on the
   rate). At the third round a fresh second moment has seen about 150 gradients, and an outlier on parameters whose moment is
   still small is normalised into a step many times the rate, which the global clip does not bound: night 226's loss rose
   0.23 -> 0.34 in one round and the cortex was left in a basin the rate cannot climb (the night re-run from the damaged
   state: flat at 0.26). In a brain the machinery of sleep's plasticity is not remade each night; its state persists. The
   mechanism (night_opt_keep, a disclosed switch; 0 = as now): the night optimizer's moments kept across nights in the save,
   with the second moment's horizon at a hundred steps (0.99) so an outlier is measured against a moment that has formed.
   Predicted: on the healthy dusk copy the night's loss curve and gauge unchanged within 0.01; on the served body no round's
   loss rising a fifth above the night's first over the following twenty nights (against one in the last twenty-six).
   Falsified if the healthy copy's night worsens by more than 0.01, or a divergence recurs under it.
   THE LEAN FORM (16:45): keeping the moments across nights would add 1.4 GB to every save (two moments over 179M weights), too
   heavy for the copies; the same protection without state is the second moment's horizon alone: night_beta2 0.99 (formed in
   a hundred steps, before the rounds where the outliers arrive) beside the existing warm-up. Built as a constant, 0.999 as
   before; tested first on the healthy dusk copy, then on the served body at a night's save.
   THE REPAIR MEASURED (17:11, copies): a night from the damaged state is stuck (flat at 0.26, the gauge 0.546 -> 0.602); the
   cortex of day 264's dusk transplanted into the current body (the store, the utterance memory and the life kept) reads the
   facts 0.875, the held-out 0.605, the questions 21 and 25, and its night runs normally (0.205 -> 0.165, the gauge 0.688 ->
   0.773). tools/transplant.py builds it on a copy in a minute; applying it to the served body is the user's decision.
   MEASURED (17:38, the donor copy, the same draws as its control): at 0.99 the night's loss 0.203 -> 0.159 against 0.205 ->
   0.165 at 0.999, its gauge 0.781 against 0.773, the held-out 0.616 in both; the parent's lines 0.683 against 0.705 and the
   old lines 0.469 against 0.482, within a night's noise on those two. Not falsified on the stated rulers. ON THE SERVED BODY
   from night 229's save; the falsifier there: no round's loss rising a fifth above the night's first over twenty nights, the
   morning rulers not down.
   THE CRITIC AFTER THE DIVERGENCE (2026-09-17, 11:12): the value scale of band 5, 50304 after night 227, reads 2972 after 237
   and 2165 after 246; the ventral head 1162, 247, 92. Its evidence forgets at 36000 ticks and the divergence's evidence is
   gone; the slow bands' heads (46, 234, 1254) hold their scale between the saves. No bounded form is needed while the scale
   and the head keep falling or hold; falsified if either rises across a week of mornings.
39. The reload's inheritance (an operations defect, found by the independent review of 2026-09-16, 21:30). A save carries its
   own constants; a reload passes only the flags file's keys; a key left out of the file keeps the save's value. Three reverts
   recorded in this ledger as done never took effect: decisiveness (item 13, "off from night 219's save") ran at 8 through
   night 232; the day's lesson at a third (item 8, "off at night 224's save") ran at 0.3 through night 232; the fade's floor at a
   quarter (item 2, "back to 0.1 at night 231's save") ran on through night 232 and kept purging the store (42724, 41511,
   41251). And the slow context in the key (item 27 of the spec, "reverted at night 191") has been live at 0.5 since day 206:
   every measurement since, the hold's derivation and its forms included, was taken with it on. The verdicts drawn on those
   nights are re-read accordingly: item 8's "episodic, not the rate" rests on an A/B whose control arm never existed, and
   the eight days at 0.3 that followed by accident show the same falls as the days before (a mean near -0.02): the rate does
   not set the fall, which stands, now on eight days rather than two; item 13's "no value separates" compared 0, 3 and 8 where
   the last two sessions were both at 8: the claim narrows to "3 and 8 do not separate from each other"; item 2's purge
   continued two nights longer than recorded. From night 233's save every revert is explicit in the flags (--sharp-conf 0
   --wake-base 1.0 --store-floor-rel 0.1), the slow context is explicit and disclosed as on (--key-ctx 0.5; not changed before
   the recording, since the store's keys were written under it), and ops/served_cfg.py prints a save's effective constants
   against the flags after every reload. The lesson, in one line: the served constants are what the save says, and the flags
   are only a delta on it.
40. The key's separation (the dentate item; opened 2026-09-17, 11:55, from the prefixes' trace). Ruler: the fact prefixes and
   the questions under natural talk (the fall 23 -> 17 as the parents said "ice is in your cup" and "the water is warm"), with
   the rephrased set and the held-out as the generalization that must not fall.
   THE EVIDENCE: tools/read_trace.py shows the collision in the read itself: at "ice is " a fresh slot from the day's talk with a
   key 0.014 closer outvotes the fact's slot (51 against 26 percent); the reads are by content alone (read_strength 0), the
   temperature 0.02, eighteen keys within a tenth of the query. The key is the fast bag (the last five symbols, shifted and
   decayed) plus 0.5 of the previous utterance's order-free bag (key_ctx, live since day 206 by the reload's inheritance,
   explicit since night 233). The order-free bag of "what is cold?" and of "is your milk cold?" share most of their mass, so
   the context term does not separate a fact's question from a question about the same word; and the matched test of day
   214 read the context as a cost (the held-out 0.657 against 0.681, the questions 7 against 11) while its store had been
   written under the context on both copies, a confound never resolved (item 39).
   THE FORMS TO DERIVE, each measured on copies against the same days: (a) the context with its order (ctx_form "shifted",
   in the code as an instrument), so "what is cold?" and "is your milk cold?" separate by their order; (b) a dentate-style
   expansion of the key (a fixed sparse random projection with winner-take-all over a few thousand units, born like the
   lexicon): similar contexts decorrelate, at the cost of the rephrased set unless the expansion keeps a dense part; (c) the
   cortex's code of the current utterance as the context term (the stream's state with the running mean out, key_form
   "cortex" gave 0 of 30 as the whole key; as the context term beside the fast bag it is untried). A re-keyed store is a
   store rebuilt: the utterance memory holds the days' lines whole, so a night-like pass can write them again under the new
   key (reconsolidation) instead of three days of retelling.
   NOT RUN until after the recording; the served body keeps its key. Falsifier for any form: the held-out or the rephrased
   set three under the control copy's on the same day, or the branch under 8 of 9.
   THE ONSET'S CONTEXT (2026-09-18, 00:46, read in the code and confirmed on a tiny body with the store's write spied): the slow
   context becomes the key's "previous utterance" inside take_world at the next utterance's FIRST symbol, after that symbol's own
   write, so an answer's first symbol is keyed by the utterance before the question, and only its second symbol on by the question
   ("the cat is here" / "what is cold?" / "ice is cold": the 'i' of "ice" keyed by the cat line, cos 1.000 to its bag, the 'c' by
   the question). The query at the onset carries the question. The onset, the read that chooses the answer's branch, has had a
   context term of noise since key_ctx went live (night 206).
   THE GEOMETRIC RULER (tools/key_separation.py, the same hour): an empty store written at strength one from the last six days'
   typed lines (546 lines, 15.5k symbols; the thirty facts heard once after the first third), then each fact asked in the taught
   and in the rephrased wording, the read at the answer's onset and along it. Six forms of the key:
     ctx form, key_ctx, swap    taught onset  chain  onset margin (med)  own slot wins   rephrased onset  chain
     bag 0.5 first (served)        26/30      0.99      +4.5 nats            24/30            14/30        0.94
     bag 0.5 offset                25/30      0.99      +6.5                 26/30            11/30        0.93
     bag 0.0 (no context)          25/30      0.96      +4.3                 26/30             6/30        0.90
     shifted 0.5 first             28/30      1.00      +5.4                 25/30            13/30        0.94
     shifted 0.5 offset            24/30      0.99     +14.1                 26/30             9/30        0.93
     shifted 1.0 offset            24/30      0.99     +21.5                 25/30             8/30        0.89
   READ: (1) the context term is what finds a fact under a rephrased question (6 of 30 without it, 13-14 with the order-free or
   the end-aligned ordered bag under the served swap), since the query's fast bag holds only the question's tail; (2) the swap at
   the offset gives the onset a real context and a decisive margin (the ordered bag: +14 to +21 nats, a hundred to one), but the
   rephrased set falls (14 to 9-11), because under the served swap the onset's context is noise for every slot alike and the tail
   decides, which a rephrasing shares; (3) the taught onsets sit at 24-28 in every form: the five that fail (grass, honey, cows,
   birds have, a) fail under every key, so their cause is not the key; (4) six days' talk does not reproduce the served body's
   falls (the prefixes 17 of 26, the branch 7 of 12): the served store holds sixty-five thousand slots of some twenty-five days.
   NEXT: the same ruler over sixteen days (the served regime), and form (c), the cortex's code of the utterance just ended
   (the stream's state after its last symbol with the day's running mean out) as the context, which can only be swapped at the
   offset: if the code of a question and of its rephrasing agree (the cortex forecasts the same continuation for both), form (c)
   keeps the rephrased set that the ordered bag loses while separating the taught ones.
   THE SIXTEEN-DAY RUN (01:25; 1727 lines of days 295-310, 46.5k symbols, the store 37-44k slots; the same facts and readings):
     ctx form, key_ctx, swap    taught onset  chain  margin (med)  own slot wins  right-symbol share   rephrased onset  own wins
     bag 0.5 first (served)        28/30      0.99     +1.8           22/30            0.83               15/30          3/30
     shifted 0.5 first             30/30      1.00     +3.8           24/30            0.90               16/30          6/30
     shifted 0.5 offset            24/30      0.99    +11.9           25/30            0.86               14/30         12/30
     bag 0.5 offset                24/30      0.98     +4.7           25/30            0.81               13/30         12/30
   READ: (1) the ordered context under the served swap is ahead of the served form on every column at both sizes (taught 30
   against 28, the margin twice, the right symbol's share 0.90 against 0.83, the rephrased 16 against 15), and the served
   form's margin shrinks with the store (+4.5 at six days, +1.8 at sixteen: the collisions the trace showed). (2) The swap at the
   offset loses taught onsets (24) at both sizes although the fact's own slot wins more often: with the question in the onset's
   key, the answer's variants in the days' talk ("a bird has wings" and "birds have wings" after the same question) tie at the
   onset and the read's mean decodes to neither; under the served swap the onset's context is the utterance before the question
   and the tail decides. The cortex's code as the context (form c) is out: the centred stream states of different utterances
   agree at 0.87 (a question and its rephrasing at 0.999), no separation to be had. (3) What the ruler cannot show, the chain
   positions being read under the follow: the served body's branch fall (7 of 12 at "we ", "a ") is a mid-chain read where the
   question's context is the only thing that separates "we eat" from the evening's "we sit", and the ordered context's
   separation there is the same geometry as its onset margin.
   DECISION (01:30): ctx_form "shifted" at key_ctx 0.5 under the served swap is the candidate; the swap stays. The real-store
   test follows: the served save's store rebuilt from its utterance memory (tools/rekey_store.py) under the served form (R0)
   and under the ordered context (R1), the rulers on both against the save as it is; then a day on each; adoption at a save
   with the rekey, if the falsifier holds off.
   THE THIRTY-SECOND DEFECT, found on the way (01:50; body/model.py Store._keep, body/life.py _step): a write beyond the store's
   capacity evicts the weakest slot and keeps the rest SORTED BY STRENGTH, so every slot's index moves at every such write; the
   store remapped its own last_idx but not the indices the body held across the write: _prev_slot (the chain's link from the
   symbol before) and _follow (the episode the waking recall is in). Measured on a tiny body at a capacity of 150 with a spy on
   the links: under the capacity 138 of 138 links joined the right kept memories; at the capacity 38 of 67 (the rest joined a
   neighbouring slot). The served store has been at its capacity since night 256: since then about four chain links in ten
   have been wrong, and the branch's second set fell from 10 to 7 of 12 at night 254-256, as the store reached the cap. The
   chain is what the dreams follow and what the waking recall's follow (read_follow 20) boosts mid-utterance. FIX: the store
   reports the eviction's remap (last_remap) and the body's held indices follow it; at the capacity 67 of 67. Test 69; the guard
   b26f0c48 unchanged under the pre-sure flags (a tiny body never reaches the cap). The wrong links of the last five nights stay
   in the served store until the rekey rebuilds them; the rekey pass (tools/rekey_store.py, on a store with room kept ahead and
   a copy-free eviction, tools/faststore.py) makes them right.
   THE REBUILD ON THE REAL STORE (02:13-02:26): the dusk-309 save (data/watch2.pt as copied at 00:41) rebuilt from its utterance
   memory under the served key (R0) and the morning rulers on both:
                                untouched     R0 (rebuilt, served key)
     questions at 2 / 8 rests    22 / 21          28 / 28
     rephrased at 2 rests          15                21
     fact prefixes               16 of 26          18 of 26
     branch sun / day2            9/9, 7/12        9/9, 12/12
     held-out (cortex alone)       0.633            0.633
   The rebuild alone, the key form unchanged, is the largest gain of the last twenty nights: correct chain links (the
   thirty-second defect undone), strengths fresh from the cortex's surprise, and the stale slots of weeks (the nineteen keys
   within a tenth of a query) gone with them. It is a reconsolidation, not a new mechanism; the utterance memory is the
   material and the code's own key the form.
   R1, THE ORDERED CONTEXT ON THE REAL STORE (02:31-02:52; the same save rebuilt under ctx_form "shifted", key_ctx 0.5, the
   served swap; the rulers with the same flag):
                                untouched     R0 (rebuilt, served key)    R1 (rebuilt, ordered context)
     questions at 2 / 8 rests    22 / 21          28 / 28                     30 / 30
     rephrased at 2 rests          15                21                          26
     fact prefixes               16 of 26          18 of 26                    22 of 26
     branch sun / day2            9/9, 7/12        9/9, 12/12                  9/9, 12/12
     held-out (cortex alone)       0.633            0.633                       0.633
   The falsifier (the held-out or the rephrased three under the control's; the branch under 8 of 9) is nowhere near: every
   ruler that reads the store rises, the cortex's untouched. ADOPTED (02:52): --ctx-form shifted in the served flags (the set
   before it in ops/archive/flags/BASE_FLAGS_pre_shifted.txt) with the store rebuilt at night 262's post-night save by
   ops/rekey_after_save.sh (the save backed up first). The cost stated: the rebuilt store holds the last 4096 utterances only,
   the older memories gone (the eviction at the capacity had them going anyway), the smiles' dopamine not in the strengths,
   every pause taken as 48 ticks; a fact told from now on is keyed under the ordered context of the line before it.
   AFTER THE FIRST NIGHT ON THE REBUILT STORE (night 263, 04:33): the fade dropped 26k of 65536 slots (the relative floor over
   raw-surprise strengths: the predictable middles the old store had lost over weeks), and the morning read the questions 19 /
   18, the rephrased 17, the prefixes 15 of 26, the branch 9/9 and 11/12 (at dusk, before the fade, the prefixes 20). Against
   the old store after night 262 (19 / 21, 17, 17, the branch 7/12 the night before): the rebuild's 28-30 were the unfaded
   store; what lasts after a night is the branch and the links. The fade after a rebuild is a one-time settling. NOTED, NOT
   BUILT: a chain whose middle slot the fade drops breaks there (the dropped slot's links go to none); a chain that closes around
   a dropped element (the predecessor inheriting the dropped slot's successors under the same tag) would keep an episode
   retrievable through its surprising elements alone. Its ruler would be the questions and the branch on a copy after a night.
   THE TWO FORMS AFTER ONE FADE, LIKE FOR LIKE (05:37-06:16): both rebuilt from the night-262 backup and faded once as the night
   fades (no day between), the rulers on each: the order-free bag 65536 -> 33536 slots, the questions 10 / 10, the rephrased 8,
   the prefixes 6 of 26, the branch 9/9 and 12/12; the ordered context 65536 -> 37844, the questions 9 / 10, the rephrased 7, the
   prefixes 6, the branch 9/9 and 11/12. Equal within a fact: the ordered context costs nothing after a fade (the falsifier's
   three not met), and it keeps its onset separation (the branch). Both arms collapse alike when a rebuilt store meets its first
   fade before a day of writes: every fact's middle is a single surprise then. The served store, rebuilt at night 262's save and
   given day 314 before night 263's fade, read 19 / 18, 17, 15 the next morning: the day's writes re-strengthened what it
   heard. RULE FOR A REBUILD: a rebuilt store lives a day before its first night (as it did), never a night first.
   THE ORDERED CONTEXT STAYS. The morning rulers from here read a store of ten days' span repopulating under correct links;
   the branch holds at 11-12 of 12; the questions and the rephrased are the teacher's to raise by the days.
   A WATCH ITEM OPENED (2026-09-18, 12:25): the held-out (the cortex alone on lines no parent typed) has slipped every morning
   since the rebuild: 0.650, 0.646, 0.634, 0.641, 0.635, 0.630, 0.633, 0.617 (nights 262-271), and the cortex alone on the fact
   sentences 0.888 to 0.857, while the store's rulers rose (the questions 18 to 23, the rephrased 13 to 20, the branch 11-12).
   Two readings, neither yet tested: (a) the teacher's material, the supervisor's days being one walk in a few frames, which the
   cortex learns instead of the language (the days vary from day 321; one varied day did not stop the slip); (b) the
   complementary-learning trade: with a repaired hippocampus the waking lesson can lean on the recall that enters the stream,
   and the cortex's own forecast weakens where the store supplies the symbol; the night's lesson runs with the store off, and
   pulls the other way. The ruler for (b) is a copy living a day with the recall zeroed in the stream against a copy as served,
   the held-out read the morning after. Not run: the days are the teacher's until the slip either stops or reaches 0.60.
   READ IN THE CODE (13:45, the user's word to investigate): reading (b) is out. The recall never enters the stream (model.inputs
   takes the reads and does not use them; the forecast adds store_in(reads) outside the stream), and the waking lesson trains
   the cortex on latent_pred(C) alone ("recall is a parallel contribution the mouth reads, never a term in the cortex's error").
   What remains is the material: by day the waking lesson on the supervisor's lines, by night the dreams drawn from the
   utterance memory by strength (the recent and the rewarded more), both the same few frames. Two nights on copies of the
   served save run now (tools/night_copy.py): as served, and with dream_old_share 0.3 (older utterances in the draw, an existing
   constant at 0); the held-out before and after each says whether the night lowers it and whether older material holds it.
   The slip paused at 0.619 on the first day written fuller (night 272).
   THE NIGHT PAIR READ (14:33): as served, one night on a copy of the day-323 save took the held-out 0.619 to 0.615 (its cosine
   0.651 to 0.655) while the parent's last sixty lines rose 0.781 to 0.795 and the old lines held 0.553: a night costs the
   held-out about four thousandths, the eight-night slip's rate, and what it learns is the supervisor's lines. The arm with
   dream_old_share 0.3 drew the same dreams (the examples identical) and read 0.613: the constant did nothing under
   dream_source "utterances" (the old share is the store's draw), so that arm says nothing. READING: interference from a narrow
   distribution, the utterance memory filling with one teacher's register as the earlier parents' lines age out of its 4096.
   The remedy is the teacher's register (the earlier parents' lines in the page log, never the held-out file, are the model), and
   the exposure per night; the day-length chain runs next.
   THE DAY AND THE NIGHT SEPARATED (16:05, the dusk probes against the mornings): dusk after day 322 0.620, morning 0.619 (the
   night -0.001); dusk after day 323 0.606, morning 0.632 (the night +0.026); dusk after day 324 0.600, morning 0.598 (-0.002).
   So the DAYS drain the held-out (day 323 -0.013, day 324 -0.032) and a night on fuller material recovers it. The waking lesson
   trains on the window as lived, its own sound superposed at own_gain, and day 324 was the day it talked over the other voice's
   lines most (nine frowns, the anticipation): the cortex learning to predict the world from a stream mixed with its own babble
   predicts clean lines worse. The night's lesson hears the world alone and recovers. Readings, not yet tested: the talk-overs
   will fall as the frowns teach the gate and the day's drain with them; the fuller material's nights recover more than the
   catechism's. Not changed: own_gain in the day's lesson (0.3; "the lessons hearing the world only" was worse at day 6 of the
   first body). The day-length chain's B arm reads whether a doubled day doubles the drain or the recovery.
   THE EXPOSURE DECIDED (16:15, the user's word: "we are going in circles; why always these tests"): the copy runs stopped (the
   served body under their load ran at 3.6 ticks a second against 5, its nights twice as long); no new ones unless something
   breaks. The bottleneck named: the body hears 1,300 words a day, 140k in its life. Three changes at night 275's save, all of
   the environment's shape (the user's call, given) or the parent's method (the supervisor's): (1) the tick 0.1 s (the body's
   time is ticks; unloaded it used 36 percent of a core at 0.2 s), the typist's --tick 0.1 with it; (2) the day 24000 ticks
   (wake_ticks) with the night scaled (night_starts_max 2048; night_load 1.0 draws the day's new slots up to it); (3) the
   parent's rhythm with a person's pause, one gap in five two to four periods long (caregiver.pace), so the critic learns a
   slow partner and the mood stops falling under a person at the keyboard. Together about 1.7 to 2 times the lines per hour
   of wall clock (the night's cost per line unchanged bounds it). The prior set: ops/archive/flags/BASE_FLAGS_pre_fast.txt.
   THE CLOCK MEASURED (16:57, after the reload at night 275's save): at --period 0.1 the body ran 6.7 ticks a second at 134
   percent of a core (the page's rows, two a tick, 13.3 a second), not ten: the tick's compute is about 150 ms with the waking
   lesson every 24 ticks and the store at its capacity. A typist told --tick 0.1 then paces in a body-time compressed by 0.72,
   its smiles and gaps landing early in ticks. Set to what the machine sustains: --period 0.15 with the typist's --tick 0.15, a
   mid-day reload five minutes into day 327 (ops/reload_now.sh) so the two clocks agree; 1.33 times the old rate, with the
   doubled day and the scaled night as decided.
   MEASURED AGAIN AT 0.15 (17:00, the page's two rows a tick the counter): 5.7 ticks a second in the minutes after the restart,
   about a tenth under the typist's clock; the old regime at 0.2 ran 4.6 against 5, the same tenth. Kept: the mismatch is what
   it was, the rate a quarter higher; the machine's compute per tick, 150 to 175 ms, is the ceiling, and the day at 24000 ticks
   is about 65 minutes of wall clock with the parent's lines at the old wall pace. The exposure gain is therefore modest, a
   quarter more lines an hour, not the doubling promised: said so to the user.
   THE THIRTY-THIRD DEFECT (18:36, the first scaled night): night 276, the first to draw 2048 dreams, failed whole with "the
   stream's cache outgrew the window" and reset the day's pressure to half (the body stayed awake, the day ran on). The cause:
   the utterance memory holds one utterance of 72 symbols (two lines with no offset between them) against a window of 64; the
   night's lockstep batch is as long as its longest dream and the stream's cache holds a window at most; 1024 draws had never
   hit it, 2048 did. Fixed in Life.dreams: a dream clipped to the window less the end symbol. Test 70 (a tiny body with a
   71-symbol line in its memory, the night runs, the dreams at most the window); the guard's digest unchanged. Served by a
   mid-day reload before the next attempt, since a failing night would reset the pressure every half day and the child would
   never sleep.

44. THE CORTEX OVERHEARS A CORPUS (opened 2026-09-18, 21:30, the user's word: "we have to pretrain this thing; language comes
   from reality and it lives in a language world"). The body hears 1,300 words a day from its parents; a child overhears
   millions aimed at no one. Its only sense is text, so a corpus is the world it can overhear. The mechanism is the night's
   own: the cortex forecasting the next symbol over sequences run in lockstep, the store off, Adam at the night's rate; the
   material a corpus of children's stories in the body's register (lower case, no commas or quotes, sentences of 8 to 63
   symbols), never the held-out lines. Nothing else of the body changes; the store's keys are the embeddings' and are rebuilt
   after (tools/rekey_store.py); the gate and the actor relearn over days. tools/pretrain_cortex.py; the smoke test a million
   symbols on this machine, the full run on a pod. RULER: the held-out (the cortex alone on lines no parent typed) before and
   after, the fact sentences by the cortex, the questions on the served rulers after the rebuild, and the sitting. FALSIFIER:
   the held-out not up by three hundredths after ten million symbols, or the facts by the cortex down by five, or the sitting
   worse in words.
   THE USER'S WORD (22:18): no pods; all local, all live, in the served body. The pod scripts stay unused. Two live paths inside
   the body as built: (a) the parent reads to it by day, story sentences in its register queued as the parent's lines
   between the conversation (ops/read_stories.py; the waking lesson learns them, the night replays them); (b) the night reads
   too, a number of story sentences drawn from the corpus into each night's dreams through the night's own lesson
   (dream_corpus_n, dream_corpus_file: disclosed constants, in the served process). The arithmetic: reading by day some
   thousands of symbols a day; a night of 2048 dreams some thirty thousand; a million symbols (the smoke test on a copy,
   running) tells the effect per million, and the night's length is the lever if the effect is worth it.

41. The gate listens (opened 2026-09-17, 16:05, the user's word: "less chattery; wait until nobody talks to it for a while; not
   a random letter generator"). Rulers: the talk-overs per quarter of a copy day (tools/day_on_copy.py --talkover 1: the
   child's own symbols while a line is being typed against those in its turn), the junk rate of its own symbols, the questions
   by the pause and the held-out (which must not fall), the parents' live counts after adoption (talk-overs and frowns a day,
   flat at 100-114 and 36-38 for weeks).
   THE READING: the same save answers 20 of 30 questions read greedily; live it opens the gate during the parent's lines (the
   talk-over echoes) and samples a first letter when the forecast is flat (the junk words). Two mechanisms in the code, off:
   (a) THE EAR (gate_ear): the world's symbol this tick and its own act last tick as sensed inputs of the gate, born at zero, so
   the frowns can teach silence while the parent types (measured once on fresh seeds of the first lineage, +0.08 of logit a
   quarter day with the vigor term off; this body runs vigor 0). Measured now: day 297 lived again on the morning save after
   night 252 with the rewards replayed, the served set against the served set with the ear.
   (b) THE SALIENCE (gate_salience): the forecast's certainty (its norm) as a gate input, so the gate opens where the mouth is
   sure and stays shut where the readout would be noise; the input has been zero since birth, its weight unlearned. Measured
   after (a) on the save after night 253: alone and with the ear, with the junk rate read.
   (c) If the learned forms are too slow: THE LISTENING REFLEX, an innate inhibition on the gate's logit while the world's
   utterance is open, released at the event's end the body already computes (the offset, by the settle law or the count),
   as a disclosed constant (gate_listen) like the activity floor: the vocal suppression while hearing speech that an infant
   has before it learns turn-taking; the learned weights can override it with evidence. Body-general (any body with an ear
   and an event boundary), nothing about content, nothing read from the parent. Its cost to measure first: the answers by
   the pause at two rests (an answer could only begin at the offset, four to eight ticks after the question).
   Adoption of any form at a night's save, with the flag explicit; falsified if the questions by the pause or the held-out
   fall three under the control copy's on the same day, or the parents' talk-over counts do not fall within three days live.
   THE EAR READ (2026-09-17, 17:30): the copy day with --gate-ear 1 came out identical to the control to the symbol, and the
   saves explained it: the served body has had the ear since an earlier day (gate_ear 1 in its save, the gate 1031 wide), and
   its weights have learned, -48 on the world's symbol arriving this tick and +13 on its own act last tick: a gate slammed
   shut during the parent's lines. The talk-overs (630 own symbols on 4206 typing ticks, 15 percent, on 132 of 165 lines) come
   from what no learned weight reaches: the spontaneous floor (gate_floor 0.05) starting a word on a twentieth of the ticks,
   and the chunk form then running that word's letters with no gate decision. So form (c) is the one: THE LISTENING REFLEX
   (gate_listen; test 66) scales the floor by (1 - gate_listen) while the world's utterance is open (from its symbol until the
   offset fires) and cuts a running word then; the learned gate is untouched, and it is released at the event's end the body
   computes. Measured on the same save (day 297, the rewards replayed): the control against 1.0 and 0.5; the rulers below.
   The salience input stays unmeasured: its weight reads exactly 0 in the save (born at zero), so a day would show nothing.
   THE WORD RATE (2026-09-17, 19:05; tools/word_rate.py): of the child's own runs of letters in its turn on days 297-300, 73, 80, 75
   and 82 percent are words its parents have typed (862 of 1112); what is not is mostly two words run together ("nowdid",
   "hotis", the chunk form's missing space) and short fragments. So "random letters" on the page are the fragments over the
   parent's lines, the floor's spontaneous starts cut short, not its speech in its turn; the ruler for the user's "words, not
   letters" is this rate, read on a copy day's own text (day_on_copy --own-file), in its turn and over the line separately.
   THE BABBLE DRIVE (gate_quiet_tau; test 67; the user's word: "talk less until the teacher leaves it alone long enough"):
   the spontaneous floor is zero as the world speaks and rebuilds linearly toward gate_floor over gate_quiet_tau ticks of the
   world's silence (the urge to vocalize returns in silence); the learned gate is untouched, and an answer opens it whatever
   the floor (the trace: p_act 0.99 a tick after the line with the floor at 0). Measured after the reflex's arms on the same
   save: the control, the reflex with the drive at 300 ticks (a minute), the drive alone. The salience gain stays unbuilt
   while the in-turn word rate is already three quarters; the missing spaces are the cortex's seam, not the gate's.
   THE CHAIN'S FLAW (18:35): the first arms were void: zsh passed "--gate-listen 1.0" unsplit and the instrument ignored it
   (${=X} splits); on the copy loaded with the reflex on, the floor reads 0 through a typed line and releases at the offset.
   THE REFLEX MEASURED (2026-09-17, 19:25; day 297 lived again on the save after night 252, the rewards replayed, the same day
   as the control): talk-overs 0 of 4206 typing ticks against the control's 630 (15 percent), on no line against 132 of 165;
   its own symbols in its turn 1207 against 1405 (391, 281, 282, 241 by quarter against 340, 362, 345, 345); junk 2 of 2135
   against 5 of 2977; the questions on the copy 21 and 25 of 30 against 21 and 24, the prefixes 16 = 16, the cortex on the
   fact sentences 0.843 against 0.865 (the held-out line was cut by the chain's own display and is lost for this arm; the
   copy is gone). PASSES its rulers: the talk-overs gone, the answers held. Held off the served body by the user's word of
   19:00 (the human teacher first, one change at a time): to go on at the save after the teacher's three days, with the
   teacher continuing, and the parents' talk-over counts (110 a day now) and the in-turn speech read over the days after.
   The 0.5 arm and the drive's arms are deferred to a quiet machine.
   ON THE SERVED BODY (2026-09-17, 20:55, the user's word: "stop the data points"): the reflex and the drive (gate_listen 1.0,
   gate_quiet_tau 300) at night 256's save, verified. IN THE CHAIR right after (tools/teach_live.py, the supervisor typing live
   with the face timed as the caregiver's): nothing over the lines the whole session, "bees make honey" at 11 s; but the
   told-once fact asked back went unsaid, and most turns were silence. Read against the copy trace: after a line the learned
   gate opens at 0.99 and the mouth samples the rest; under the old floor forty spontaneous tries a turn let the answer out on
   one of them, and the drive had removed them for the first minute. The mood fell +1.7 -> -6.0 in fourteen minutes (a quiet
   child earns few known-word smiles against an expectation of 1.7 a line) and the readout to its floor of 8.
   THE SURE PROPOSAL (gate_quiet_sure; test 68): under the drive the floor is whole at once when the forecast's norm reaches the
   constant (0.45: a recalled answer reads about 0.6, a flat forecast about 0.3 on the copies), and waits for the silence
   otherwise: the readiness to act rises with the strength of the proposal. THE READOUT'S FLOOR (sharp_min 8 -> 20): a bad mood
   widens the babble and no longer turns words into letters (item 42: the temperature 1/(sharpness x |pred|)). Both at night
   257's save; the guard's reference under the current flags b26f0c48 (023f6e9b under the set before the reflex).

42. The mathematics of the mouth, read whole (2026-09-17, 20:30, the user's word: "look at the math going on in the model and
   get this thing figured out"). Five numbers set what you hear, and three of them are not the body's.
   (1) THE EXPOSURE. The parents type about 1,000 words a day to it (985, 966, 984, 999, 985, 1024, 986, 1074, 1096, 1100 on
   days 290-301; 4,100 symbols), one line every 15 s (the typist's period 64 ticks = 12.8 s plus a wait for its quiet; the
   median interval 15.0 s, the gate's wait 0.7 s). Since birth 31,168 lines, 138,239 words, 564,273 symbols over 245 nights:
   a child hears about 30,000 words a day, thirty times this; the whole of its life's language is a short children's book.
   The night replays that day's and the recent days' utterances (1024 dreams of about 24 symbols, six rounds, 384 steps at
   1e-5): the cortex sees the same thousand words thirty times a night, so its growth is bound by the new language a day
   brings, not by the nights, and the held-out's rise of a few hundredths a night is what a thousand new words buy. The
   lever is the pace, which is the parent's: the typist's period (64 ticks) against the child's turn (listen 40 ticks = 8
   s; an answer lands at 2.5-10 s); the lines' length (six words in 24 symbols against the 38 allowed). Fuller lines at the
   same pace (+30 percent) cost nothing; a period of 48 (+33 percent) shortens the turn to five seconds; together +70.
   (2) THE READOUT. logits = sharpness x (pred . E_k): the forecast pred is the conditional mean of the next unit embedding
   (its norm the certainty, its direction the symbol), and the sharpness is 25 x (1 + mood/6), floored at 8. The temperature
   of a word's first letter is therefore 1 / (sharpness x |pred|): a sure forecast (|pred| 0.6) in a good mood (+3, sharpness
   37) is read at an effective 22 and comes out a word; the same forecast at mood -4 (sharpness 8) at 5, and a flat forecast
   (|pred| 0.3) at 2.4, a letter drawn almost at random, then run as a word by the chunk. The junk is the product of certainty
   and mood, and the mood is the integral of dopamine: forty frowns a day for talk-overs (-2 each), the eighth rehearsal's
   drain (+2.2 to -4.2 over six questions, the sharpness 34 to 8). So the listening reflex, which removes the talk-overs,
   removes most of the frowns, holds the mood, and sharpens every word: the "random letters" and the "talking over" were one
   mechanism seen twice. (The decisiveness-by-certainty constant, item 13, multiplied the sharpness by |pred| a second time;
   it was falsified on the live answers and is not the fix; the mood is.)
   (3) THE RECALL. The store reads by content alone: softmax over unit keys of (K . q)/0.02 with the query's norm as the
   inverse temperature (a full context about 2.1, so the read is sharp), the key the last five symbols plus half the
   previous utterance's bag; the episode's chain carries a recalled utterance forward (read_follow 20). What it gets wrong
   is a collision: a fresh strong write with a key a hundredth closer takes the read from a fact's slot (item 40's trace),
   and nothing in the read weighs strength (read_strength 0). That is the last body change worth making for this world.
   (4) THE GATE. z = the learned gate's logit (its ear at -48 on "the world's symbol arrives", +13 on "I spoke last tick"),
   flattened by stress, p(act) = floor + (1 - floor) sigma(z), floor 0.05: the learned part is right, the floor and the chunk's
   free run were the talk-overs (item 41), and the reflex and the drive act on exactly those two. An answer opens the learned
   gate to 0.99 the tick after a question; the floor never mattered for answers.
   MEASURED ALONE (2026-09-18, 11:20, the queue empty six minutes): nothing said while the last twenty world symbols were typed;
   its first symbol two seconds after the world's last; then 298 symbols in 346 s. Alone it does not wait the drive's minute:
   the learned gate's "I spoke last tick" (+13) sustains speech once its turn has opened. The drive shapes the floor, not the
   run; what ends a run alone is the cortex's seam, and the loops there ("hold him." six times) are the run-on fault.
   THE TALK-OVERS ROSE WITH THE REGISTER (day 325, 2026-09-18: nine frowns against two to four): read in the frowns' contexts,
   seven of nine came in the middle of the OTHER voice's line ("he ran all day. he sle|eps" said as "he sleeps now" was typed):
   the child anticipates the reply it has learned and says it along with the voice, the sure term opening the gate on a
   forecast that is right. Not a fault of the rule and not changed: the parent's frown when interrupted is the signal that
   teaches the learned gate to hold its tongue while another speaks; the count is to be watched over the next days, and the
   two at a line's closing "?" (the answer given as the question ends) are the eager case the earlier days already showed.
   (5) THE STORE'S FORGETTING. Now the relative tenth with the capacity's eviction at 65,536, the regime of the best mornings;
   the absolute floor at 0.07 stood above the write strength of predicted symbols and cut the episode chains (item 2).
   WHAT FOLLOWS, in the law: the reflex and the drive on tonight (the user's word); the pace and the lines' length the
   parent's to raise (mine), with the held-out's nightly rise as the ruler; the key (item 40) the last body change; then the
   recipe frozen and the days are the teacher's. Nothing in these five is a rule about content, and nothing reads the parent.

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

45. A PERSON'S HAND (2026-09-18 23:16, the user's word: the demo's typist will type with one hand, the other on the face, and the
child must not interrupt a slow typer and must generalize to one). Read in the code before changing anything: the typist has always
posted whole lines and the page typed them at one symbol a tick (0.15 s); my own sittings post whole lines too. No line in the body's
life was ever typed slowly. The learned gate's ear feature (a world symbol this tick) and the offset (eight ticks of quiet, or the
surprise settling) both read a slow typer's pause inside a line as the utterance's end, and the sure proposal then has the floor whole:
a one-handed person would be interrupted at every pause, and nothing learned so far transfers, since the gate's other feature, the
cortex's state, has never held a pause inside a line. The change is in the environment and the parent's method, no constant of the
body touched: a share of the parent's lines (SLOW_SHARE 0.33) go in symbol by symbol at SLOW_CPS 2.0 a second (each gap 0.6-1.4 of
the mean) with a thinking pause of one to four seconds at SLOW_PAUSE 0.1 per symbol; the turn stays open across the pauses, so a word
said into one is said over the parent and meets the frown (FROWN_GAP 60 from the same relaunch). The rows carry "slow": true. The
measure: the talked-over count on slow lines against fast lines per day, and the sitting at a one-handed pace (tools/teach_live.py
still posts whole lines; a --slow mode for the chair is next). From day 336 (the chain relaunched 23:17, during night 278). Not done:
the offset's constant. A slowly typed line is split into utterances at each pause of 1.3 s or a settled surprise, and the store keys
the pieces; watched first, changed only if the recall of slowly typed facts fails.

44 (continued, 2026-09-19 00:45). THE SMOKE TEST'S ANSWER: a million story symbols through the night's own lesson on a copy of the
body after night 277 (batch 16, lr 1e-5, one round, 1674 steps, 3.2 hours beside the served body): the corpus loss 0.380 to 0.253,
the stories learned; the held-out 0.605 to 0.564; THE FACTS BY THE CORTEX 0.853 TO 0.571. A pure pass over another distribution, with
none of its own material among the batches, takes the facts out of the cortex: the forgetting biology's interleaved replay exists to
prevent, and the smoke test had no interleaving by design. So nothing of it enters the served body. Measured next, on a copy of the
body after night 278: one night in the served form with the corpus among its own dreams (2048 own + 1024 story sentences), then the
rulers (the held-out, the thirty questions with the store, the rephrased, the branch); the served night's own change on the same
rulers is the comparison (the held-out 0.605 to 0.601 across night 278, the questions 23 to 21 of 30). The mouth with the store on
the smoke test's after-copy is read first: whether the answers survive when the cortex's facts do not. The day door (the parent
reads) is open regardless. If the mixed night also costs the facts, the night's reading share goes down (256-512) or the reading
stays with the day, where the sentences become its own utterances and are replayed among the day's at the natural ratio.

44 (the decision, 2026-09-19 03:10). THE MIXED NIGHT ON A COPY of the body after night 278, one night in the served form with the
corpus among its own dreams (2048 own + 1024 story sentences, 3072 dreams, 1152 NREM steps): the held-out 0.601 to 0.606, the
parent's last sixty lines 0.463 to 0.549, the old lines 0.573 to 0.581; the rulers after: the cortex alone on the thirty fact
sentences 0.840 (the served body after night 278: 0.843; the pure pass: 0.571), the mouth 21 of 30 questions (the same), the
prefixes 13 of 26 (12), the branch 9 of 9 and 11 of 12 (the same). The interleaving does what biology's replay does: nothing lost.
The mouth with the store on the pure pass's copy also answered 21 of 30, the store carrying what the cortex had let go. THE
CHANGE: --dream-corpus-n 1024 --dream-corpus-file data/stories_valid.txt in the served flags (ops/BASE_FLAGS.txt, guard_args.txt,
serve_command.txt; the archive ops/archive/flags/BASE_FLAGS_pre_reading.txt), the reload armed for the save after night 280. The
cost: half again as many dreams, a night of about forty-five minutes on the CPU. The benefit is unproven and is watched in the
sittings (a statement given back to a question, the pear's kind) and on the held-out; about 36k story symbols a night, 470k a day.
THE CLOCK, measured on a copy with the machine free (scratchpad/clock_check.py): the waking tick 199 ms on the CPU and 190 ms on the
GPU (MPS), no gain, the tick's cost outside the matmuls; the night's step of sixteen dreams 5.98 s on the CPU and 2.92 s on the GPU,
twice as fast. The GPU would halve the nights, not the days; a served body on it is measured on a copy first (a night on each
device, the same dreams' gauge), not tonight.

45 (continued, 2026-09-19 08:05). THE COUNT RISES UNDER THE FROWN: words said over a slow line 1.96, 2.07, 2.78 on days 337-339
(fast lines 0.14, 0.05, 0.07). The frown cannot reach the cause. Read in the code: a person's thinking pause ends the world's
utterance by the count (offset_ticks 8, 1.3 s) when the surprise has not settled (the cortex still expects letters, so the silence
surprises it and the settle law does not fire); the listening reflex then releases the floor and the sure forecast of the next letter
has it whole; the floor is not learned. MEASURED ON A COPY (tools/slow_line_probe.py: twenty of the parent's recent lines typed
one-handed into the copy after night 281, the words over each counted, seed 0): offset_ticks 8: 2.70 per line, 25 percent clean (the
served day: 2.78, 27 percent; the probe is faithful); offset_ticks 30: 1.70 per line, 30 percent clean. Excluding the first line,
typed into the copy's own chatter with no wait for its quiet (10-11 words in both, the probe's artifact): 2.26 to 1.26, a 44 percent
fall. By kind: the conversation lines (the register the demo's typist will use) are clean or near it at thirty ("what is sour?" 2 to
0, "one more. what is hard?" 5 to 0, "you are still up. the night is near" 8 to 1, "it is late. the moon is up" 3 to 1); what remains
is on the story lines ("mom said some mushrooms are bad." 5 and 5, "lily wanted to touch it anyway." 3 and 4), where the settle law
fires in the pause because the story's letters surprise the cortex more than its silence does. THE CHANGE: offset_ticks 8 to 30 at
the next save's reload, one disclosed constant, effective only while the cortex still expects more; the finished line still ends by
the settle law. Not changed: offset_settle 0.5 (a deeper settle would hold the story lines too; one constant at a time). The count
on the day's slow lines is the measure; the story lines grow familiar with the reading.

46. THE REVIEW OF 2026-09-19 (the user's word: "look over the whole architecture and training, make sure everything is in good
order"). The tests 70 of 70; the served flags equal ops/BASE_FLAGS.txt; no duplicate flag; the serve log clean; the guard on the
long tag found unarmed since night 278 (its line pointed at a scratch copy that was gone) and re-armed from the repository's copy.
Three readings (the waking path, the night with the store, the typist), the confirmed findings and what was done:
 THE TYPIST. (1) The answer's smile window was 1.5 s, not 6.6: self.listen is seconds since the 8th and the expectation added
 self.s() to it again; 25 of 119 answers in the child's turn got the answer smile since day 343, the rest the faint smile or a
 miss. Fixed. (2) A line in flight at the sleep switch typed its tail into the sleeping body and spun on the drain all night; no
 night row, the reload's waiter never fired. Fixed: the slow loop and the drain see the night and the day ends. (3) The reload
 killed the typist five to sixty seconds into the next day: phantom days (337, 342, 345, 347), two day numbers a night, a popped
 line lost. Fixed in the script: the typist is stopped at the save row and the chain held while the server restarts. (4) The
 smile in its turn only let the babble decay drain the parent's attention with no bump to repay it: aways in the evening's silence,
 the next lines scored distracted. Fixed: chatter past the turn neither pays nor drains. (5) The chair's faces held three to six
 times longer than the typist's; the talk-over frowns on the other voice's lines uncounted; a request without a guard; a row read
 while being appended lost whole; the stage-one expansion lines ("milk") typed into a conversation day. All fixed.
 THE NIGHT. (6) The night's dream count read the store's growth, which the capacity stops (65536; the store at 63000): the nights
 would have halved within two days. Fixed: the day's kept writes are counted. (7) The night fell inside a line and left it open
 until the morning, glued to the day's first line under the dusk's tag and a stale chain index (the fade re-indexes the store and
 only a write follows the remap). Fixed: the night ends every utterance, the wake resets the utterance state and the chain's
 index. (8) A rebuilt store's first fade (the ledger's rule) is now the body's guard (store_fresh, set by tools/rekey_store.py).
 (9) A clipped dream was given the turn's end symbol at the cut; fixed. (10) REM sampled from the global generator; the corpus
 draws unrecorded; the gauge mixed the corpus dreams. Fixed: the body's generator, the corpus count in the report, the gauge over
 its own dreams. (11) The slow bands were not saved: zeroed at every reload while band_mu was kept, the ventral critic reading a
 birth each morning. Fixed: saved and restored.
 THE WAKING PATH. (12) THE COUNT AT THIRTY WAS WRONG: the served tick is 0.26 s, not 0.15 (the body is compute-bound), the
 other voice's answer arrives a median 33 ticks after the question's last symbol, a third under thirty; the child's own answer holds
 the question's utterance open under the settle law (its own letters make the rest surprising), so the question and the answer
 merged into one utterance keyed under the context before the question, and the question could not cue it. Reverted to eight at
 night 284's save. (13) THE DIAGNOSTIC (tools/slow_line_probe.py --diag 1): at thirty ticks no offset fires inside a one-handed
 line and every word said over it comes through the LEARNED GATE with the floor shut: the ear reads the tick (the world's symbol
 this tick, weight -49; its own act last tick, +14), so the gate is shut on a keystroke's tick and free on the quiet ticks between
 a slow typist's keystrokes. Day 340's fall to 1.35 was not the count; day 341 returned to 1.95. THE EAR'S TRACE (gate_ear_decay):
 the ear's world input persists between symbols, decaying each tick, the auditory trace; the learned weight then keeps the gate
 shut while a person is still typing at any pace. Measured on a copy before anything enters the body.
 DEFERRED, RECORDED: the store's write at the capacity copies the whole store twice and re-sorts it (about a gigabyte a world
 symbol on the CPU; tools/faststore.py holds the copy-free form; the served store reaches the capacity in about two days);
 the non-finite night's reload restores the previous morning, not the evening (a dusk snapshot is the fix); the count-ended line's
 end mark falls outside the wake lesson's window eleven times in twelve; the queue's who-labels and the sleep clear are not atomic
 under the HTTP handlers; a NaN strength would empty the store at the next fade; the corpus lowercases "I".
 THE EAR'S TRACE MEASURED (13:05): the same ten lines, seed 1, the count at eight: no trace 3.10 words over per line (40 percent
 clean); the ring at 0.8, 2.30 (40 percent); the ring at 0.9, 0.90 (60 percent), with the rulers at 0.9 unchanged (the held-out
 0.611, the facts 0.804, the questions 16 and 18 of 30). A keystroke gap at two symbols a second is two or three ticks: at 0.8 the
 ring is half gone between keystrokes, at 0.9 it holds. Adopted at 0.9 for the reload at night 285's save; the day's count of
 words over slow lines is the measure, the morning's answer latency the cost to watch.
 THE TRACE'S COST AND ITS RELEASE (day 343, 15:12 on): the first hour with the trace gave 0.92 words over a slow line (3.36 the day
 before) and, on the other side, the answer's smile on 41 percent of the questions (78) at a delay of 2.5 s (1.0), the child's
 first symbol after a line at a median 11 ticks (6), 38 percent of lines with no symbol in the window (12): the ear kept ringing
 after a finished line and the gate stayed shut into the child's own turn. THE RELEASE: the ear stops ringing when the utterance
 is perceived to have ended. Released at every offset (the count's too), the copy after night 285 gave 3.00 words over a line (5.20
 without the trace on that save, the gate grown eager as the answers pay) with the first symbol at 9.5 ticks (4.5): the count's
 offset fires inside a person's thinking pause and freed the gate there. Released only when the SETTLE LAW ends the utterance
 (the cortex expected the quiet: a finished line) and held when the count alone ends it (a long pause the cortex did not expect:
 a person thinking): measured on the copy at 16:10, the number that decides what enters the body at night 286's save.
 THE SETTLE-ONLY RELEASE MEASURED (16:20, the same ten lines on the copy after night 285): 1.20 words over a line (50 percent
 clean; 5.20 with no trace, 3.00 released at every offset) and the turn's first symbol at a median 21 ticks (4.5 with no trace, 9.5
 released at every offset), every line answered within forty. Read: after a finished question the cortex expects the other voice's
 answer, so the quiet does not settle and the count ends the utterance without releasing the ear; the ring then decays on its own
 and the answer comes at five seconds. For the demo that is the better side: a person waits five seconds for a child that thinks
 before it answers, and does not want to be talked over. Adopted for night 286's save at 0.9; day 344's count, the answer's delay
 and the answer's share are the three numbers; the gate's own learning (the smile at four for the late answers, the frown for the
 words in long pauses) moves them from there, and 0.85 is the constant to try if the delay stays past fifteen ticks.
 THE STORE'S WRITE MADE COPY-FREE (19:00, the review's fourth finding, done the same day): the tools' FastStore is the body's own
 (body/model.py), its rows in buffers allocated in blocks and exposed as views, the eviction a single weakest slot with the last
 moved into its place and the links following through last_remap; the buffers rebuilt after a load or a compaction. Measured on
 a copy with the capacity set under the store's count so every write evicts: 285 ms a world-symbol tick against 327 with the copy
 of every slot, the served body sharing the machine. The tests 73 of 73. Into the body at night 288's save, with the store at
 65024 of its 65536.
 THE BABBLE MEASURED (19:45, ops/day_report.py): known words said into the silence past its turn, per silent minute: 15.8 on day
 342, 17.0 on day 344; the smile in its turn only (from day 340) has not moved it in two days. The candidate, in the constants
 already there: the tonic drive that pays every act (gate_tonic 0.25, babble its own reward) following the felt reward rate
 (gate_tonic_rate, 0 = off), so the urge to act falls in the silence where nothing pays and rises under the parent's smiles; it
 works through the gate's lesson, over days, and the answer's smile at four dwarfs it. Before the change, the babble alone is
 measured in the chair (two minutes out of the room), the demo's own case.
 THE FROWN'S WEIGHT (23:10, the parent's method): under the ear's trace the words over a one-handed conversation line held at a
 quarter of before and did not fall (days 343-347: 0.57, 0.87, 0.69, 1.14, 1.12) while the answers rose to 96 percent of the
 questions. The talk-over frown was set light (-1, at most every sixty ticks) on the 6th, when the gate had no feature to bind it
 to; it has one now. From night 290's boundary: -2, the known word's smile's weight, at most every twenty ticks. Day 348 is the
 first day of it; the count on conversation lines is the measure, the answers' share the cost to watch.

47. THE RELATIVE FLOOR FEEDS ON ITSELF AGAIN (2026-09-20 04:55). The store: 65028 slots at night 288's save, then 64964, 64753,
63520, 61605, 56873, 48829 at night 293's, the night's fade taking 275, 1233, 1915, 4732, 8044 and 10360 while the days wrote
about 4600. The mechanism is the thirty-first defect's, in the ledger's own words: a floor relative to the mean climbs as the weak
are removed and as the strong grow, and the next night removes more; and since the 18th the day's story lines, unfamiliar and so
written strongly, lift the mean further. The absolute floor (store_floor_abs 0.07, derived on the 17th, lived twelve nights at a
store of about fifty thousand, reverted after night 249 to let the store refill to the capacity where the eviction by strength
would be the forgetting) does not move with the mean. It returns at night 294's save; the capacity's eviction remains the ceiling.
The measure: the store's count and the fade per night from night 295; the morning's recall of the taught facts, which are the
strongest slots and were never the ones the floor took.
 THE EAR'S GAIN (2026-09-20 06:20). The gate's weights read across five saves: the weight on the ear -48.78 to -48.73, on its own
 act +13.86, while the weights on the cortex's state grew (their mean size 8.84 to 8.96) from the answers' smiles; the frown at two
 moved nothing. The ear's trace holds the gate only through that fixed weight, so the input's scale, gate_ear_gain, is the constant
 to set. Measured on the copy after night 293 (the same ten lines, seed 1, the settle-only release): gain 1, 1.20 words over a line
 and 50 percent clean, the turn's first symbol at 21 ticks; gain 1.5, 0.80 and 60 percent, 20.5 ticks; gain 2, 0.70 and 40
 percent, 19.5 ticks. Adopted at 1.5 with the absolute floor at night 294's save. The words that remain come in the long pauses,
 where the ring has decayed below what the cortex's state pushes; the gate's own learning would have to carry those, and it does
 not move.

48. THE BABBLE'S DOOR IS THE GATE, NOT THE FLOOR (2026-09-20 11:30; the user's word: "I need it not to just spit random blabber
while I'm having a convo"). The measure, tools/silence_probe.py on the copy after night 296: six questions typed at the tick, the
child's turn given forty ticks, then a person's thinking silence of 240 ticks (forty seconds at the served pace) with no input; every
own word in the silence counted. Under the served constants 27.5 words a silence (41 a silent minute), the answer looping through the
whole of it ("a lemon is sour a lemon is sour a lemon is s..."; "the cow gives us milk the cow gives us milk"). The floor's constants
changed nothing: the sure proposal at 0.9, 26.2; with the babble drive at 1200 ticks as well, 26.7, and the word sequences identical
to the served run's on the same seed, so the floor's draws were never the ones that acted. The door is the learned gate, whose weight
on its own act (+13.9) keeps it open once it has begun: it answers and does not stop. THE YIELD (gate_yield, gate_yield_after 40):
after the world's last symbol the child has its slot; past the slot, with the world still quiet, the learned gate's logit is held by
gate_yield, the hold fading over gate_quiet_tau as the babble drive returns, so a child left alone for minutes babbles as before. An
innate turn-taking bias (answer, then wait for the other), two disclosed constants, no content read, nothing inside the slot touched,
the day's lines (the typist's gaps under ten seconds) untouched. Test 75. On the copy: yield 20, 11.2 words a silence, the remainder
the floor's own share (a twentieth of the ticks with the sure proposal keeping it whole), which is the sure proposal's and the
babble drive's to set: measured next with the sure proposal at 0.9 and tau 1200 under the yield, the one-handed line probe alongside
for the answers' place in the slot.
 ADOPTED (12:11): gate_yield 20 alone at night 297's save, the floor's constants as they were: the silence 27.5 to 11.2 words, the
 answers and the one-handed lines untouched. THE SURE PROPOSAL IS THE OTHER DOOR (11:40-12:11, the same twenty one-handed lines, seed
 0): the served constants 0.50 words over a line, 65 percent clean, every line answered inside the slot (a median 18.5 ticks); the
 sure proposal off with tau 1200 under the yield, 0.10 and 95 percent, but the turn missing inside the slot on ten of twenty lines,
 three questions among them, and the silence 1.0 words. The floor's tries under the proposal are most of the words in a one-handed
 line's pauses and most of the babble in a silence, and they are also what starts a turn the learned gate does not start by itself
 (with no act there is no own-act input, and the gate's momentum never begins). The forecast's norm passes 0.45 and 0.9 on nearly
 every tick; whether an answer's forecast stands above a pause's is the measure running (tools/norm_probe.py), the proposal's
 threshold set from it on the copy for night 298's save if the two separate.
 THE FORECAST'S NORM BY PHASE (12:14, tools/norm_probe.py on the copy after 296, the proposal off): the slot after a taught question,
 a median of 1.13 (p10 0.90, p90 1.37; the first eight ticks 1.12, p10 1.00); the thinking silence after the answer 0.59 (p10 0.48,
 p90 1.09); the quiet ticks inside a one-handed line 1.02, its perceived pauses 0.86 (p10 0.63, p90 1.35); the forty ticks after a
 statement 1.01. An answer's forecast stands above a silence's by half and above a pause's by a quarter: a threshold near 1.0 to
 1.1 has the floor whole through most of an answer's slot, on about a tenth of a silence's ticks and a third of a pause's. Measured
 next on the same body and the same twenty lines: sure 1.0 and 1.1 with tau 1200 under the yield.
 THE THRESHOLD IS NOT CLEAN (12:35, the same body and lines): sure 1.0 with tau 1200 under the yield, the silence 6.5 words, the
 one-handed lines 0.45 over and 80 percent clean, no turn inside the slot on 5 of 20; sure 1.1, the silence 5.5. The pauses' norms
 overlap the slot's, and the answer's own loop keeps its forecast sure past the slot. THE TURN'S READINESS (gate_turn, 12:33;
 test 76): the readiness to respond follows the other's utterance being perceived as complete, which the body already computes
 (the settle law: the surprise's fast average under half its slow one, the cortex having expected the quiet); under the babble
 drive the floor is whole for the slot's length (gate_yield_after) after such an end, a pause the count alone ended (a person
 thinking mid-line) opens no turn, and past the slot the drive's ramp rules. The sure proposal off (0). Reads no content; the
 same constant as the yield's slot. The risk: a line whose end the count perceives gets no turn's floor and the learned gate alone;
 the share of settled ends is the measure running alongside (the diag now reads each line's end).
