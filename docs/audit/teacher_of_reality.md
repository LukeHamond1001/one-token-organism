# The teacher of reality: one reward, her face. The lead's view and plan (2026-09-26)

The owner's word (2026-09-26): "i want g1 to only get dopime from parent in sim no charging or anything? when parent smiiles with face
it gets it when ot frowns it gets cortisol. we need parent to become the teacher of reality. we need to use what we learned from llm
to solve robtics. I want to solve grounded reward live enironment rl robotics."
And after the first draft: "also add pain is cortisol tp plan. no bottle no charger. objects are for teaching".

**The rule, in one line.** Dopamine, up or down, comes only from her face. Cortisol comes from her frown and from its own pain.
Every object in the room is there for her to teach with; nothing in it meets a need.

Read against `sim` at 0a42821 (`body/sim/anatomy.py`, `body/sim/lane.py`, `body/sim/parent_feel.py`, `body/sim/lang/conduct.py`,
`body/core/`). Sources were checked through Europe PMC, arXiv and the publishers' pages. Nothing here is built yet.

## What I think

**Yes. It is the right bet for the goal, for four reasons.**
1. **One reward that a person can give in the real world is the body-general form.** Any body and any skill get the same channel,
   and no reward is written for any task. It is the owner's goal (a body-general architecture) and the ratified two-button design,
   with a face as the buttons.
2. **It is what made the language model work, made live.** Prediction learns *what is*: the forecast of every sense, with no reward,
   as pretraining is. Her face teaches *what is good*, as feedback from people does after pretraining. Here both run at once, in one
   life, in a body.
3. **Biology does it.** A caregiver's contingent smiles and touches shape an infant's behaviour:
   - adults' social responses increased three-month-olds' vocalizing (Rheingold, Gewirtz & Ross 1959);
   - mothers' contingent, but not non-contingent, reactions made 8-month-olds' babbling more mature, and the change persisted
     (Goldstein, King & West 2003);
   - at 12 months, infants read their mother's face to decide whether to cross a drop that looks unsafe. About three quarters crossed
     when she posed joy, almost none when she posed fear, and anger held them back too (Sorce et al. 1985).
4. **It makes First 1 stronger.** Its claim becomes "no reward but her face".

**Three things must be true, or it fails.**
1. **Her smile must answer the body's acts. Today it answers only talk.**
   - The design's motor rows were never built: a whole roll, its own reach and hold, sitting alone, peekaboo answered by an act, and a
     guided act repeated (§4.3's worth table).
   - `lang/conduct.py` makes only verbal judgments: a met ask, a right name, an approximation, a vocal turn, a scaffolded give.
   - Her percept already sees the events these rows need: `got`, `rolled`, `sat`, `gave`, `fell`, `child_reaches` (`lane.py:293–337`).
   - With her face as the only reward and no motor rows, the body would learn nothing about the world from reward. This is the first
     build.
2. **She must teach like a good teacher, not pay like a vending machine.**
   - She judges within the tick.
   - She shapes: approximations first, then the bar rises.
   - She stops smiling at a mastered act. Today's worth for a first motor act falls as 1 + e^(−n/10) with a **floor of 1**, so a
     child could farm her smile forever. Human trainers give mostly positive reward, and a learner that values the future then loops
     to farm it: the "positive circuits" problem (Knox & Stone 2015).
   - She praises improvement against the child's *current* level, as people naturally do (MacGlashan et al. 2017).
   - She frowns only at its own harmful acts, and not before stage 2.
3. **The world must still teach what is.** The forecast of every channel stays the main learner, and her face only steers. That was
   the language model's lesson (§3).

**The honest costs.**
- **Removing the charge removes the body's own stakes.** That was 2026-08-29's point about feeling. For a robotics result it costs
  nothing.
- **Pain no longer pays.** It raises cortisol and marks the moment, but it never moves dopamine, so pain alone does not make an act
  less valued. The body learns to avoid harm from her frown. Its reflexes and the motors' own limits still guard the joints, and
  T-D (§2) tests what pain alone does.
- **In the sim her face is a scripted reward function wearing a face.** It becomes a person's only with a person and the real robot
  (First 1's Level 2; A49).

## 1. Smile as dopamine; frown and pain as cortisol

The core already has the smile's and the frown's paths. Pain's needs one small change, the last row.

| her face | in the body (existing code) | biology |
|---|---|---|
| a smile's rise, seen | `FaceIncrement` gives a positive term → the TD error δ = r + γV′ − V > 0 (`body/core/critics.py:155`): the dopamine burst → mood += 0.25 δ (`body/core/mouth.py:835`); the amygdala's face+ head learns what predicts it | the dopamine prediction error (Schultz, Dayan & Montague 1997) |
| a frown's rise, seen | a negative term → δ < 0, the dopamine dip → **stress** += 0.5 × \|δ\| (`mouth.py:836`), with a half-life of 240 ticks (36 s) → the waking lesson's plasticity × (1 + stress/10) (`body/core/cortex.py:280`) and each gate's choice flattened (`mouth.py:382`); the amygdala's face− head tags the moment, so it is written to memory more strongly, replayed first at night, and an act followed by net harm is not taught as an act to make (`body/core/amygdala.py`, the night's side) | the lateral habenula drives dopamine's dip for bad outcomes (Matsumoto & Hikosaka 2007); the amygdala strengthens the storage of arousing events, pleasant or unpleasant (McGaugh 2004) |
| its own pain (a joint past its limit, or a blow past F_pain, from its own sensors) | **new:** `JointPain` stays a source, but its term never enters the reward, so there is no dopamine. Each pain tick adds 0.5 × 1 to **stress** (the frown's gain on a −1 dip), and it reaches the amygdala: its pain− head learns what predicts pain, and the tag marks the moment, which is written more strongly and replayed first at night. The withdrawal reflex and the born cry stay as they are | newborns' cortisol rises to a painful heel-stick and, unlike their response to handling, does not habituate when repeated (Gunnar et al. 1991); a local anaesthetic cut circumcision's cortisol response (Stang et al. 1988); glucocorticoids, with noradrenaline, strengthen the storage of emotionally arousing events (McGaugh 2004; Buurstede et al. 2022) |

**What "cortisol" means here.**
- Real cortisol is slow: it acts over tens of minutes to hours through gene expression (de Kloet, Joëls & Holsboer 2005). It works
  through the basolateral amygdala to strengthen the storage of the day's arousing events (McGaugh 2004).
- The core's `stress` is the fast arm, over seconds to a minute. The slow arm is the amygdala's tag carried into the night.
- So the owner's frown-as-cortisol exists at birth in two parts, and pain joins it through the same two.
- **The change for pain:** `RewardSource` gains `dopamine` (default on, so the language body's sources and digests are unchanged).
  A source with it off adds nothing to the reward. Its |term| goes to stress at `stress_gain`, and it still reaches the amygdala
  (its heads and the tag), as `amyg` says.
- **What pain alone then does:** the reflex withdraws, the cry calls her, learning is sharper for about half a minute, and the
  moment is remembered and dreamt first, so the forecast learns what brings pain. **What it does not do:** lower the value of the act
  that caused it, because dopamine is hers alone. T-D tests this.
- Test T-C (§2) measures whether one frown makes an act rarer that day and after the night. A slow cortisol trace becomes a candidate
  only if T-C fails.

Every change of her face is felt only when the child sees it. The born reading updates only while the face test passes on 2 ticks
(A1). So the child must look at her to learn from her: social referencing, grown as infants grow it, by use.

## 2. The plan

**Phase 1: one reward (the body and the world)**
- **The rewards:** `FaceIncrement` is the only source that pays (dopamine). `JointPain` stays as a cortisol source (`dopamine` off, §1).
  `ChargeRelief` goes, and the amygdala's heads fall from 5 to 3 (face +, face −, pain −).
- **The charge is removed:**
  - the charge channel (2 numbers), the drain and the charger;
  - the bottle, which existed only to feed, with its dock, the pad's ring and the charging contact rule (`CHARGER_PREFIX`), and its
    word, lines and sound in her inventory;
  - her feeding episodes (`lang/dayplan.py`);
  - the cry's charge trigger and her `charge_low` event;
  - the page's charge meter and the runner's fields.
- **Pain stays as a sense and as cortisol, never as dopamine:** the pain flags, the pain event line (the amygdala's low road), the
  withdrawal reflex and the born cry, plus §1's cortisol path. They protect the body and call her. Her concern face still reads 0.
- **The paperwork:**
  - the tests (world, lane, language, ears, voice, eyes);
  - the pins;
  - SIM_DESIGN §6 (one term), §5.3 (the charger gone), §4.3, First 1's claim and risks;
  - the decision log (A88, the owner's decision; A89 for the teacher's rules below).

**Phase 2: the teacher (§2c).** Built in four parts, in order: her eyes and her notebook (2a), her judgment (2b), her lessons
(2c), her hands (2d, the boundary decision A90). Her rulers are reported from the first plumbing day.

**Phase 3: S5b plumbing (all learning rates 0, one day and a night).** Every path must fire:
- a smile gives δ > 0;
- a frown gives δ < 0, stress, the face− tag and a first place in the night's draw;
- a pain gives stress and the pain− tag, with δ untouched;
- no charge appears anywhere;
- the digests are pinned.

**Phase 4: birth.** Seed 1, the born rates, one life with no resets, watched live on the /sim page and reported each day.

**Phase 5: the tests, registered before birth.**
- **First 1, reworded:** "...from a scripted parent's sparse smiles and frowns for its own completed acts, felt only while it looked
  at her, and no other reward; its own pain raises cortisol and never moves dopamine".
- **T-A, the motor kiwi:** an act she smiled at once is made more often the next morning, after one night, than on the day before
  it. It is the robot's form of the kiwi sentence (learned after one telling, kept through a night).
- **T-B, social referencing grown by use:** after its own act, it looks to her face within 20 ticks more often than at birth (Sorce
  et al. 1985; First 2 already holds Walden & Ogan 1988).
- **T-C, a frown felt:** an act frowned at once is made less often that day and after the night.
- **T-D, pain as cortisol:** after a pain she did not see, the forecast of pain rises before the act that caused it, and the moment is
  replayed that night. The act is **not** made less often, because pain moves no dopamine. This is a registered prediction, reported
  either way. If painful acts persist, the fix is her frown at blows she sees, never a pain reward.

**Phase 6: reality.**
- A49's removal test: her smile read from its own pixels (Level 2).
- Then a person's face and the real G1, the owner's call at each step (no purchase without their word).

## 2b. The room: objects are for teaching

Every object in the room is there for her to teach with or for a test. The bottle and its charger were the one thing that met a need,
and they go (Phase 1). A new object enters only for a lesson she gives with it or for a test registered before birth.

| object | its sound (`make_g1room.TOYS`) | what she teaches with it |
|---|---|---|
| ball | a rubber bounce when it lands or is struck | its name and colour; reach and hold; it rolls, which sets up First 2's ball behind the table |
| block | a hollow wooden knock on contact | name; grasp from a surface; banging makes a knock (cause and effect through its own forecast) |
| duck | a squeak when squeezed | name; a squeeze makes a squeak |
| cup | a clink when it strikes something | name; a container, and later W5b's containers |
| rattle | beads, louder with speed | name; its own shaking makes the sound, and the sound scales with speed |
| car | a wheel rattle while it moves | name; push, and it goes |
| bear | a soft bell when it moves | name; give and take with her |
| stacker | the rings' clack | name; later, putting on and taking off |
| drum | a boom when hit on top | name; hit, and it booms |
| ring | a crinkle when handled | name; the easiest hand-over (6 of 6 in the grasp study) |

The planned richer room (W5b, A53) is all for teaching too:
- textures, so a thing is not known by colour alone;
- containers;
- a cover that a Dex3 hand can lift, for hiding games and M3's search test (First 2);
- at least 24 examples of each registered noun;
- one new object every 3 life days from an inventory fixed before birth.

Her smiles for acts (Phase 2) are given with these objects. She marks the object with her eyes, the child acts on it, and she smiles
at the completed act.

## 2c. The teacher: what "very good" means, and her build

The owner's word (2026-09-26): "what im trying to build is our llm archecture to use to solve robotics rl. thats main goal. so sim
teachers going to have to be very good."

She is the whole reward. The body can learn from reward only what she can **see**, **judge** and **set up**. Today she is a talker,
not a coach: her judgments are of words, and her motor time is "show a toy near it and call". A very good teacher has four parts,
each on the world's side, each reading only what a person could see, never the body's inside (the reading-mode rule: the child's
outward acts are a person's to read; its store and its rates are not).

**2a. Her eyes and her notebook.** Her percept already reads got, rolled, sat, gave, fell, lost_toy, its reaches, its head's line,
pain and distress (`lane.py:293–337`). To add:
- the end of each movement: each hand's distance to each toy, and the head's line to the target (the reach's and the turn's progress);
- a lift (a held toy above its surface), a shake (a held toy's speed) and a hit (a toy's contact that sounds: the drum, the block);
- posture in more grades: back, side, front, head up on its front, sitting;
- `hit_her` (A25c) and the copying readers (A52), the two readers still unbuilt;
- **her notebook:** per skill and object, the child's best so far and its last 10 tries. A teacher's record of the student, on the
  world's side, saved with her state.

**2b. Her judgment (the reward).** Scripted, deterministic, within the tick, every judgment logged with its reason.
- The worth table's motor rows, built at last: a whole roll 2; a reach that gets a toy 2; a lift 1; a shake or a hit that sounds 1;
  sitting alone a moment 2; a give after her ask 1; peekaboo answered by an act 1; a guided act repeated itself within 40 ticks,
  that act's worth.
- **Shaping by improvement:** an approximation earns 1 while the full act is unmastered: a reach that ends nearer the toy than its
  best of the last 10 tries, a turn that ends nearer the target. The bar rises with the child (policy-dependent feedback,
  MacGlashan et al. 2017).
- **Habituation to zero** per (act, object): the n-th smile is worth 2·e^(−n/10), no floor; a new object or a new place starts n
  again. This is A2's "falling with mastery" with its floor of 1 removed (the positive circuits, Knox & Stone 2015).
- **Marked and seen:** her eyes go to the object and back around the act, as for words. At the judgment she says "yes! you got it."
  at once from where her face already is: a sound onset that the born orienting turns toward (A43), so its eyes come to her face
  and the smile is seen, before social referencing is learned. The pulse waits for the look as now (20 ticks).
- **Frowns from stage 2:** hitting her, a toy thrown or knocked off a surface, a blow past F_pain. Never failure, pain or distress.
- **Formal trials:** no judgment, no face (A60b), unchanged.

**2c. Her lessons (the curriculum, L3).** A ladder in the order infants climb it. Each rung has the setup that makes the act
likely, the reading that scores it, and the step up. She picks the rung where the child is nearly there by her notebook (its best
within one step of the bar), stays while it improves, steps up when her smile for it has habituated under 0.5, and steps down after
a block with no progress. Floor play and motor time run the rungs; show time and the words stay as they are.

| rung | her setup (acts she has) | scored by | its step |
|---|---|---|---|
| orient | the call from its periphery; a toy shaken 15° off the fovea (A3) | its head's line on the target | farther off the line |
| reach and touch | the rattle set beside the hand on its own side (`put_near`), then 5 cm farther each mastered level, then across the midline | the hand's distance at the movement's end; a touch | the distance |
| grasp and hold | the toy into the palm's path; the ring first (6 of 6 hand-overs in the grasp study) | got; held 10 ticks | harder toys |
| lift, shake, hit | the rattle in its hand; the drum under its hand | a lift; the toy's own sound | louder, longer |
| give and take | her open hand; "give me the X" | gave | farther |
| roll | the toy shown beside its head on the far side; her brief turn from its front (kept) | a half roll (side), then rolled | unassisted |
| head up on its front | a toy at its eye level while prone | head up | longer |
| sit | a toy above its chest at arm's length (the pull and the prop stay closed: 34 kg) | sat | longer |

**2d. Her hands (the scaffolds; A25c reopened in part, at a boundary: A90).** `guide` (a forearm along a path, within 65 N: A8, A10)
and `knee_over` (within 76 N) reopen: their controllers are built and tested (`OPENED_LATER`), and a person can do both. `pull_to_sit`
and `prop` stay closed: a person cannot lift 34 kg. A guide is used only after a block with no progress at a rung, and the smile
goes to the child's own repeat within 40 ticks, never to the guide. Every guide is counted in the intervention log printed beside
First 1 (A63), whose claim adds "with her guidance, counted". Guidance speeds a learner's teaching (Thomaz & Breazeal 2008).

**Her rulers, reported each day beside the child's:** judgments an hour by kind; the share the child saw; the latency from the act;
each rung's bar and the child's best; farming (the same act repeated past habituation); guides an hour; her cost in ms a tick (the
tick's budget is 150 ms). The teacher is judged by her rulers as the diary's was (the human-teacher rule).

**What she is not.** No language model in her judgment or her face. Her curriculum is chosen from her notebook by the rules above
and logged, so a life replays exactly. A language model may later write her talk, as the diary's did, logged and replayed.

## 3. What the language model taught us, carried to the robot

| the lesson | where it came from | on the G1 |
|---|---|---|
| prediction does most of the learning; reward steers | the diary learned language by forecasting it; her face chose what it said | the forecast of all 9 channels stays the main learner; her face steers the acts |
| one telling and a night | the kiwi sentence, kept through a night | T-A, the motor kiwi |
| a contingent teacher beats drills | "conversations, not drills" (09-12); "a human teacher" (09-17) | her judgments within the tick, answering what the child just did |
| a reward that over-pays gets farmed | the graded onset over-paid by 49%, fixed by the increment rule; the self-press body collapsed into "I love me" (08-29) | habituation to zero; no self-reward; her smile only for completed, visible acts |
| the teacher must not leak the answer | the Clever Hans rounds (A51, A60) | formal trials with no feedback remain the only measure of understanding |
| the night consolidates | REM stays on; the night stands | the tagged day dreamt first; frowned acts not taught as acts to make |
| judge it by sitting with it | "judge by conversation" (09-18) | watch it live on /sim; report what it does, then the rulers |

**Left out, on purpose.**
- **Practising in dreams against its own forecast of her face**, the trick of training on a learned model of the teacher's reward,
  is not used. The amygdala never pays (the owner's decision 2), and optimizing against a learned stand-in for the teacher overshoots
  the real one (Gao, Schulman & Hilton 2023).
- **No reward for moving, looking, closeness, novelty or progress:** §6's list stands.
- **No language model in her loop** (no cloud, and local speed). Her talk and judgments stay local and scripted, and every one is
  logged.

## 4. Risks and how we will see them

| risk | how we see it |
|---|---|
| sparse reward over 10 effectors is slow; First 1's 60 life hours slip | smiles an hour and the life hours to each milestone, reported daily |
| smile farming through a loop | her ledger: the same act kind smiled at again and again; her habituation must hold it at zero |
| pain pays nothing, so it repeats a painful act she does not see | the pain flags an hour (already logged); T-D; her frowns for blows she sees |
| frowns chill exploration (stress flattens choice; the amygdala turns orienting away) | acts an hour before and after frowns; no frowns before stage 2 |
| it never looks at her, so it never feels her | the share of her smiles it saw; T-B |
| the claim is a scripted face | printed beside every result until Level 2 and a person |

## 5. Parked

The consciousness instrument (T1 of `consciousness_math.md`) waits until Phase 3 passes. The owner turned the work to this plan.

## Sources

- Csibra G, Gergely G. Natural pedagogy as evolutionary adaptation. Philos Trans R Soc B 2011. PMID 21357237.
- Buurstede JC et al. Hippocampal glucocorticoid target genes associated with enhancement of memory consolidation. Eur J Neurosci
  2022. PMID 33840130.
- de Kloet ER, Joëls M, Holsboer F. Stress and the brain: from adaptation to disease. Nat Rev Neurosci 2005. PMID 15891777.
- Gao L, Schulman J, Hilton J. Scaling laws for reward model overoptimization. arXiv 2210.10760 (ICML 2023).
- Gunnar MR, Hertsgaard L, Larson M, Rigatuso J. Cortisol and behavioral responses to repeated stressors in the human newborn. Dev
  Psychobiol 1991. PMID 1797593.
- Goldstein MH, King AP, West MJ. Social interaction shapes babbling. PNAS 2003. PMID 12808137.
- Knox WB, Stone P. Interactively shaping agents via human reinforcement: the TAMER framework. K-CAP 2009.
- Knox WB, Stone P. Framing reinforcement learning from human reward: reward positivity, temporal discounting, episodicity, and
  performance. Artificial Intelligence 2015.
- MacGlashan J et al. Interactive learning from policy-dependent human feedback. ICML 2017; arXiv 1701.06049.
- Matsumoto M, Hikosaka O. Lateral habenula as a source of negative reward signals in dopamine neurons. Nature 2007. PMID 17522629.
- McGaugh JL. The amygdala modulates the consolidation of memories of emotionally arousing experiences. Annu Rev Neurosci 2004.
  PMID 15217324.
- Millar WS, Watson JS. The effect of delay on infant learning reinvestigated. Child Dev 1979.
- Rheingold HL, Gewirtz JL, Ross HW. Social conditioning of vocalizations in the infant. J Comp Physiol Psychol 1959. PMID 13641468.
- Schultz W, Dayan P, Montague PR. A neural substrate of prediction and reward. Science 1997. PMID 9054347.
- Sorce JF, Emde RN, Campos JJ, Klinnert MD. Maternal emotional signaling: its effect on the visual cliff behavior of 1-year-olds.
  Dev Psychol 1985.
- Stang HJ, Gunnar MR et al. Local anesthesia for neonatal circumcision: effects on distress and cortisol response. JAMA 1988.
  PMID 3339788.
- Thomaz AL, Breazeal C. Teachable robots: understanding human teaching behavior to build more effective robot learners. Artificial
  Intelligence 2008.
- Walden TA, Ogan TA. The development of social referencing. Child Dev 1988.
