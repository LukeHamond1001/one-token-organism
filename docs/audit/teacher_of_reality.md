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

**Phase 2: the teacher of reality (her judgments of the body's acts)**
- **Her smiles for acts** are judged on the events her percept already reads:
  - `got`: its own reach and hold, not a hand-over of hers;
  - `rolled`;
  - `sat`;
  - `gave`, after her ask;
  - peekaboo answered by an act;
  - a guided act it repeats itself within 40 ticks.

  Readers still to be built: `hit_her` (A25c) and the copying readers (A52).
- **Her rules, each with its source:**
  - **Contingent:** judged within the tick, the smile begun at once. Infants learn a contingency when it is immediate, not at a 3 s
    delay (Millar & Watson 1979).
  - **Marked:** her eyes go to the thing and back, around an act she teaches, as they already do for words (natural pedagogy:
    Csibra & Gergely 2011).
  - **Shaped:** an approximation earns 1 until the full act has been done 3 times. This is her word rule (P3) carried to acts; a
    reach toward a toy counts as an approximation, from `child_reaches`.
  - **Habituating to zero:** the n-th smile for the same kind of act is worth 2·e^(−n/10), with no floor. A new toy, a new place or a
    faster act counts as a new kind. This replaces the floor of 1, closing the positive circuit (Knox & Stone 2015). She praises
    improvement against its current level (MacGlashan et al. 2017).
  - **Frowns, from stage 2:** hitting her; a toy it knocks or throws off a surface; a blow its own act makes past F_pain; and talking
    over her, as now. Never a failure, never its pain, never its distress: those draw concern, which reads 0. A frown is held 10
    ticks, never waits for a look, and is felt only if seen.
  - **Guidance:** her guided acts (the roll's guided arm), each counted in the intervention log. People who can guide a learner
    teach it faster (Thomaz & Breazeal 2008).
- **Her ledger beside the life:** smiles and frowns an hour, their latency after the act, the share the child saw, and her guides.
  The teacher is judged by her own rulers (the human-teacher rule).

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
