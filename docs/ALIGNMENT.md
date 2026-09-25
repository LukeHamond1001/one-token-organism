# Alignment: both projects, 2026-09-25

One page that says what each project is for, what is decided, where each stands this morning, the order of the work left, and who does what. Anything not on this page is open to the lead's judgment under the laws in `docs/SIM_DESIGN.md` §1 and the owner's standing words in memory.

## 1. The two projects, one sentence each

1. **The language body** (the 179M-parameter organism at `data/watch2.pt`, paused after the kiwi demo): finish as **one film on X** that makes an average person, on a phone, amazed — it learns a word from one telling, keeps it through a night, and the film then shows what each part of the machine does.
2. **The humanoid** (a stock Unitree G1 in a MuJoCo reality with the same brain architecture and a human parent): **on this Mac only**, learn **fast**, and do **robotics firsts no architecture has done** — and along the way find **math that humans use**, by making the architecture predict human measurements it was never fitted to.

## 2. What is locked

### The film
- One version, on X, about 1:50. Order: **the growing-up montage (7.2 s) → the kiwi demo on the real talk page, light theme (to 42.1 s) → the owner's line to camera (42.1–52.6) → the 3D animation (52.6–110.1), which ends the film.** No G1 teaser, no closing card, no self-introduction.
- Outside light, inside dark: the demo is the real page in its light theme, re-rendered from the session logs and its recorded frames; the animation is the dark graphite studio.
- The animation explains **what each part does**, for a person with no ML knowledge. Real 3D (three.js), Apple/Anthropic-clean, no neon, no biology words, no monitors, no AI clichés. Not real math, but true to the mechanism: every claim on screen is in the claims ledger (`video/film5_design/SPEC.md`, `facts.md`).
- Every word the body says on screen is from its logs. Nothing is staged.
- The owner's line: *"ChatGPT read trillions of words. This one has been told about fifty thousand short sentences, total. So how did it remember that overnight?"* — and the four sentences that answer it are the animation's.
- Music: Grand_Project, "Technology Modern Electronic" (Pixabay licence in `video/music/LICENSE.txt`); the beat grid is the track's; film time = track time − 14 s so the animation lands on the track's full entry.
- The authority for words, pace and production is `video/film5_design/DIRECTORS_CUT.md` (`gen_cut.py`); the cut itself is `video/film5_design/preview.py` → `timeline.json`.

### The humanoid
- **Local only. No cloud, ever. No spending.** The cloud speed test is dead.
- **Speed and firsts, not human pace.** The human-pace ledger (SIM_DESIGN A58) is an optional report, never a target.
- The laws: no hand-written rules, cheats or controllers; biology's mechanisms at the stage biology has them; every constant disclosed with its source; nothing fitted to the environment's pace or the child's rates; one seed; no baseline runs; REM stays on; exact determinism and the eight pinned digests (`tools/pins/digests.txt`) hold on every commit; the language body stays bit-identical; the sim is the robot.
- The body: the stock G1 (43 joints, 34.4 kg), D435-placed stereo eyes with a software fovea, ears 15.8 cm apart, touch as the real G1 has it (Dex3), contact and pain from joint efforts, ten effectors each with a gate; the parent is a physical body of cited masses and human-strength actuators; understanding is scored only in formal time-matched trials.
- Structural decisions already taken (they do not reopen): plain bounded steps for act_pred (no Adam); the leaky bounded cerebellum (leak = rate/3); the physical parent; the top-heavy configural face bias; only trials count.

## 3. Where each stands this morning

### The film
| Piece | State |
|---|---|
| Montage + demo (light page) | **Rendered**, 42.13 s at 1920×1080/30 (`video/film5/demo.mp4`); checked frame by frame by the lead today: clean. |
| The owner's line (10.5 s) | **Not recorded.** The recording guide (backdrop, light, clothes, phone, mic) is in DIRECTORS_CUT.md; the prompter is `python3 ops/demo_video.py prompter video/film5`. |
| The animation (9 shots, 57.5 s) | **First full build done** (`ops/anim/architecture_v4.html` + `ops/anim/v4/`). The lead's review (`video/film5_design/REVIEW_1.md`): 5 must-fixes (the kiwi thread and the necklace, S2c's fusion, S6a's smile, label sizes, whole-machine framing and C's labelled map), 5 should-fixes. |
| Sound | The cue and music pipeline is built (`ops/film_sound.py`; −14 LUFS, ducking under the voice). |
| Transcript | `TRANSCRIPT.md` is stale against the current cut; to regenerate. |

### The humanoid (branches in scratchpad worktrees)
| Branch | State |
|---|---|
| sim-core | **R6 READY** (fix 8, plain bounded steps). **Cerebellum merged** (3655706). **R6h built and verified** (four commits to 8788abc: movement units, fatigue per effector, forward error into the gates, act_inv batched, born encoders, orienting and VOR hooks, the spinal pattern generator, the born cry, the disclosed drives, the C61 numbering). Open: the pattern generator's period (C54) is the lead's number, not a sourced one; the world must read `acts.cord` and `acts.vor`; the cerebellum is off in SIM_CFG until the lead names the G1's mossy-fibre list; born movement units were measured without physics (W4's job). |
| sim-world | W1, W3 READY earlier; **W2 (the physical parent) being rebuilt** structurally; C6 (catch rate 36% vs 95%) and C7 (turn 28° in 2 s) to re-measure after it; W3 to reopen for the real D435 views, the camera model and lighter meshes. |
| sim-face | The top-heavy face bias: **fix in verification** (round 2). Nothing silent gets merged. |
| sim-parent | P3 time-matched trials: **fix 3 in verification**. Acceptance is the invariance property: flipping the draw changes nothing for any non-knower. |
| Design | `docs/SIM_DESIGN.md` at cfa7cf5 (A1–A66, B1–B21, C1–C72): the owner's bar of 2026-09-25 folded in after a state-of-the-art check and a skeptic's review — local only, no spending, ever (A61); fast learning measured as sample efficiency, life hours and exposures to each milestone and the overnight gain, never an "N× faster" figure (A62); the registered firsts (A63); the local speed levers with build steps R10, P7, W7, R8 (A64); B3 (a) yes, (b) and (c) no (A65); the human-pace ledger an optional report (A66). In flight: the PFC-maturation study. |

## 4. The order of the work left

### The film (target: a full preview for the owner today; the final when the face piece exists)
1. The fix pass from REVIEW_1.md, three builders on disjoint files (§5).
2. The lead's second review of every shot, start, middle and end, at phone size.
3. Regenerate `TRANSCRIPT.md` from the current timeline.
4. The final render (`--ss 2 --mb 4`), `cut.mp4`, the placeholder edit with the music, and the preview to the owner.
5. The owner records the line; the final `edit`; the owner posts. Then commit the film tools.

### The humanoid
1. Close what is in verification: W2, the face bias, P3.
2. One design-sync commit at that boundary: R6's plain steps, the cerebellum's leak and bound, R6h's constants and cost, W2's and the face's outcomes, C6/C7 re-measured, C54.
3. The lead's calls owed to the build: the G1's mossy-fibre list (cerebellum on at birth), C54's period from a source or disclosed as ours, and the world reading `acts.cord`/`acts.vor`.
4. R7 (the tagger, event lines, recall into action) and R8 (night on frames, twitches).
5. W3 reopened; W4–W6; P4–P6.
6. The local speed levers (§9), measured, with no reduced precision and no looser world.
7. S5a plumbing, S5b, birth.
8. **The firsts, as registered (SIM_DESIGN A63; each bar frozen with a digest before birth, the exact replay released, nothing claimed until a skeptic has read it against the log).**
   - **First 1.** In one life with no resets, the stock G1 learns to orient, reach and grasp (the full claim adds rolling) within 60 life hours, from a scripted parent's sparse smiles felt only while it looks at her, every intervention counted and printed beside the claim; level 2 repeats it on the smile read by its own camera; the long-run clause at 200 life hours: savings, plasticity and retention hold. No learner has done these together from random weights in a real robot's whole body.
   - **First 2.** It passes 3 of 4 hard tests of understanding it was never taught (the ball behind the table, the toy under the cover, her silent head turn, a look to her at a new toy); a test counts only if the born state fails it.
   - Dropped as firsts, with reasons in the design: speed alone, a face reward alone, lifelong learning alone, compositional words, one core with two bodies, a first word from its own tract. The real G1 is the endgame. "A word told once and acted on after a night" waits for a skeptic's review before it is registered.
   - Wall time at the lever pace (A64, A65): 60 life hours ≈ 1.6–3.3 calendar days running around the clock; 200 ≈ 5–11 days.
9. **The discovery line.** Each mechanism must predict a human measurement it was never fitted to (a signature: the shape of a curve, a ratio, an order of firsts). Measured on copies at night boundaries, reported with the prediction written down before the measurement. A signature no one put in is the discovery.

## 5. Who does what

**Opus 5.5 builders and verifiers** (subagents, one job each, on disjoint files): building shots and parts; building mechanisms behind switches; the verification rounds (tests threaded and pinned, the eight digests, the old saves, the main tree unchanged); renders and mechanical passes; literature checks for constants; long measurement runs. Their rules: never edit a file they do not own; never `rm` a path built from a variable; scratch under 1 GB and stop below 8 GB free; sim checks at nice 19, one at a time; never load `data/watch2.pt`; never touch `data/backups` or `data/first_lineage`; nothing pushed.

**The lead (Fable 5.1)**: the story and every word on screen; the personal review of each build against the cut (the owner's word: no subagents for the review of the three-part film); the structural decision whenever a build loops; the design doc and the laws; the science framing (the firsts and the human signatures); the alignment and the daily reporting; anything that touches the owner's data or needs the owner's yes.

**The owner**: only two things. Record the line when the preview is right. Say the word to push or post. Everything else is decided here, measured on copies, changed at boundaries, and reported.
