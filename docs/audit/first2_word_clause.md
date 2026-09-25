# A skeptic's review: "a word told once, acted on after a night, by a robot body" (2026-09-25)

Research only, by an Opus 5.5 reviewer for the lead; the lead's decision is at the end. Read against SIM_DESIGN A61–A66, C69–C72,
section 12, and the firsts research (`$S/firsts/research.md`). Texts read are saved in `$S/firsts/kiwi_lit/`.

## A correction first

The language body was not told once before its night. In `video/take1/events.jsonl`, "a kiwi is fuzzy" was typed 5 times, "what is
fuzzy?" asked 8 times, and the body said the answer itself 9 times, all before the sleep. Its first right answer came after 1 telling;
after the night it answered at the first ask (take 2). **"Learned after one telling, kept through a night" is true; "told once, kept
through a night" is not.** (The film says exactly the true thing: "One telling, and it has it. We tell it four more times. Then it
sleeps. It kept it.")

## 1. Has it been done?

One exposure, word to object, acted on, in a body:
- Hill et al. 2021, ICLR, arXiv:2009.01719. A simulated 3D agent, told once by a text string that an object is a dax, can re-identify
  and manipulate it as instructed. Fast mapping was its trained task (RL, task reward plus shaping, 1e9 steps), inside one episode of
  about 80–120 steps; offline consolidation is future work. Prior art for one-shot binding with action.
- Lampinen et al. 2021, NeurIPS, arXiv:2105.14039: new object names kept over distractor phases inside an episode; trained by RL.
- Jiang et al. 2023, VIMA, arXiv:2210.03094: new nouns defined in the prompt with support images, then acted on by a simulated arm;
  pretrained T5, imitation on 600K+ trajectories.
- Twomey, Morse, Cangelosi & Horst 2016, Interaction Studies, doi:10.1075/is.17.1.05two: iCub with hand-built colour and shape
  features in SOMs, speech-to-text, Hebbian links; each label said 5 times per trial; retention tested after the block; no sleep.
- Morse et al. 2015, PLoS ONE, doi:10.1371/journal.pone.0116012: iCub bound names to objects through posture; name said 3 times, tested
  10–15 s later by orienting and reaching.
- Engineered and foundation robots: Hwu, Kashyap & Krichmar 2020, IJCNN (a Toyota HSR fetched a new object after one training trial;
  YOLOv3 labels, typed names); Lee, Mo & Han 2025, VAP, arXiv:2512.20014 (a pretrained VLA acts on a user's own cup from a few images).
  These keep a name overnight for free.
- Not one-shot: Tani's group 2025, doi:10.1126/scirobotics.adp0751; Vong et al. 2024, doi:10.1126/science.adi1374 (no body).

One exposure, kept across sleep-like consolidation, then acted on:
- Hwu & Krichmar 2020, Biol Cybern, doi:10.1007/s00422-019-00808-7: one-trial paired associates with training epochs standing for
  sleep replay; a network, tested within the session.
- Bruce et al. 2017, arXiv:1711.10137: one traversal, offline replay, then a real robot navigated; ImageNet encoder, no words.
- Jayasinghe et al. 2026, arXiv:2606.17493: offline consolidation of skills; no single exposure, no words.
- Not found: any embodied learner that heard a word once, slept, then acted on it; or any from random weights without pretrained
  speech or vision.

Infants: 24-month-olds map fast but keep poorly over 5 minutes without ostensive naming (Horst & Samuelson 2008,
doi:10.1080/15250000701795598). A nap helped 16-month-olds generalise two taught words (Horváth et al. 2015, doi:10.1111/jsr.12306).

## 2. A first as worded? No.

Pretrained robots are told a name, act on it later, and keep it overnight trivially. Only a qualified form is unclaimed. A reviewer
rejects it for:
- Language through the side door: the scaffold delivers the word's exact letters, a perfect transcript, and Claude writes lines. The
  claim holds only with the input scaffold silenced (A29); words, objects and tellings fixed before birth, outside the steering, held out
  by the line check.
- Her cues: her gaze (A51), the order she places objects (A43's onsets), her voice's side, a smile after one ask cueing the next.
- Trainable by construction: smiles for probe asks she reads (A40) reward the routine; A45's heading key can bind the word to where it
  was said (Morse's posture effect), a place pass; the told object's extra attention alone can shift looks.
- Chance: not 0.5 by assumption (no landing is a miss); it is the body's own rate.
- The night's role: no ablation is allowed (REM on, no side seeds), so the claim is "kept across a night", never "consolidated by it".

## 3. Wording and bar

"From random weights, with no pretrained language, speech or vision, in one life, the stock G1 model heard a new word once, as sound
alone, from a scripted parent naming an object in its view; it never heard the word again; after a night it looked at [reached for,
handed over] that object when asked for it, above its own rate for a never-told word at matched moments, on the registered items, first
ask only. Its born state fails the same protocol."

- Telling: one line in the new-word register, the word last and on the pitch peak ("look. a W.", A34), not a variation set. The object
  in her hand within the fovea's reach (A22), her eyes to it on the word (A51 allows this for a label). Every telling counts; the
  fovea's place at the word recorded. W absent from every line heard before, held out after, not among the 79 born rows.
- Items: object pairs from A53's calendar, seen and handled equally, never named before the probe, each target once and distractor once.
- Sleep: the live night (A46). Reported: ticks from telling to test; whether the telling's frames were written, kept and dreamt.
- Test, in the next day's first floor-play block: both objects placed at once by one fixed choreography, symmetric about the midline,
  never where the target was told. The ask comes a registered delay later, when she reads it facing her. "where is the W?" and "where is
  the F?": F never told, matched in syllables and stress, same voice and register, a variant other than the telling's. Order
  counterbalanced, same block and posture. She stays still, eyes on the child, no point (A51).
- Acted on: primary, the fovea's first landing within 20 ticks of the word, held 2 ticks (A28); she cannot read it (A40), so the probe
  earns nothing. After M2, the first hand contact within 40 ticks; after M3 and "give" understood, a hand-over within 40. A headline
  "acted on" needs the reach or the hand-over. Only the first W ask per word counts.
- Chance and power: per item, the F ask's rate on that object and the born state's, pooled by A28's Poisson-binomial, one-sided
  p < 0.01. Planning figures (replaced by C36 from the measured chance): at 1/2, 7 of 7 words (p 0.0078), 10 of 11 (0.0059) or 12 of 14
  (0.0065); with a second distractor at 1/3, 5 of 5 (0.0041), 6 of 7 (0.0069) or 7 of 9 (0.0083). With 3 probes a day and W and F each
  counting, about one word a day.
- Born state, every learning rate 0 at S5b, through the telling, the night and both asks: it must not pass W against F at p < 0.01, nor
  show a place bias. Recall's maps are born at zero (7.6), so a born pass means a leak, and the test leaves the claim.

Earliest: M6, once the input scaffold is silenced (4.9). The look needs M1 and a comprehension route that already passes "where is the
X?" for known words by A19; before that, C36 reports "not testable yet". The reach needs M2, the hand-over M3. M6 is forecast at 15–80
life hours; the silencing is not forecast, so no hour is promised. A level with the scaffold on is a disclosed report, never the claim.

Risk: recall into action returns acts keyed by heading, so it can miss a moved object. A pass needs recalled object codes to steer the
fovea, and nothing born does that.

## 4. Recommendation

Merge it into First 2 as a word clause with its own bar (above), outside the 3 of 4, reported "not testable yet" until M6 with the
scaffold silenced. Do not register it as a separate first. As worded it is false: Hill and Lampinen bound a word in one exposure and
acted on it in simulated bodies, the iCub mapped words fast with pre-built features, and pretrained robots keep names overnight. What
stays unclaimed (from scratch, sound only, never trained, one life, a night between) is the kind of claim First 2 makes. "The robot's
kiwi" would overstate both this test and the kiwi itself. The film and the design should say "learned after one telling, kept through a
night".

## The lead's decision (2026-09-25)

Accepted in full. The word clause joins First 2 with the bar above at the next design sync (SIM_DESIGN A63's First 2), reported "not
testable yet" until M6 with the scaffold silenced; it is not a separate first. Every document says "learned after one telling, kept
through a night"; `docs/audit/value.md` line 15 and the firsts research's F5 are corrected at the sync. The film already says the true
thing.
