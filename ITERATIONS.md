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
