"""THE PARENT'S LANGUAGE (docs/SIM_DESIGN.md 4.4-4.10). Built: lexicon.py, the born table of 79 rows and the words channel's
schedule (P1); and P3's fast layer:
  consts.py       her constants (the teacher's method: timings, the line check's limits, the ledger's tests, her ear's margins,
                  what she expects by situation), each with its source
  percept.py      what she perceives each tick (what the world must hand her: only what a person in her place could see or hear),
                  and her Reader of the child's head line and hands (A40: never its fovea's window)
  templates.py    her frames by intent, the growth queue and what each word needs to be shown, and the line check (with the
                  new word's pitch-peak table, peak_lines.json: A34)
  conduct.py      her intents and the acts they accompany (the interface W2's motion implements, with its report of her
                  attention each tick, A51's log, fail-closed; a stub until then), her voice's manners (FastLayer), the speech
                  side of L2 (Conduct), and her formal trials of what the child understands (4.8, 12: the only place it is
                  scored; her everyday asks are her teaching)
  stimuli.py      a formal trial's stimuli, time-matched by construction (P3's twelfth round; its fourteenth: one carrier
                  phrase a form, the test word spliced between one recording before it and one tag after it, under the level
                  ceiling): what a child can time from in a sentence (timeline), whether every sentence a trial's draw could
                  give is one (same), the parts and the splice, and the recipe measured before birth (trial_lines.json,
                  tools/sim_voice_check.py --trial)
  transcriber.py  how she hears the child: its tract through her ear (body/sim/parent_ear.py) at each turn's end, its silent token
                  output as a transcript
  ledger.py       every line she says, every word she accepts, her asks (teaching) and her trials, each word's standing (heard,
                  understood from trials alone, says), saved with the world and checked on a replay
To come: day.py (P4: the day plan, routines, stages, leaving and returning, the never-taught pairs' probes), the digest and the
steering check (P5)."""
