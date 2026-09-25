"""THE PARENT'S BODY: HER CONSTANTS FILE (docs/SIM_DESIGN.md 4.1, 4.2, 4.10, A3, A4, A6, A7, A9, A10, A22, A25; section 10's rows
"the parent's force caps", "the guide's cap", "the parent's soft contact and yield", "the parent's own pain", "the prop's easing and
catch", "the parent's paths"; the build plan's W2). Every constant of her motion is here, with who set it and why. They are the
teacher's method (section 1: "the parent's timings, feelings, priorities and force caps live in its constants files"), set before
birth, never tuned to the child's rates, never scaled to the child (A25), and her caps may only tighten. Her talk's constants are
the fast layer's (body/sim/lang/consts.py, P3).

HER STRENGTH: A PERSON'S (A25; the owner's B16: she can never lift, slide or sit up the 34 kg G1; it must rise by itself). The
design wrote the caps "from ergonomic norms recalled from memory: the NIOSH patient-handling limit of 35 lb (Waters 2007), and the
Snook and Ciriello push and pull tables; W2 checks them against the sources". Checked 2026-09-24:
  - Waters 2007 (Am J Nurs 107(8):53-58, "When is it safe to manually lift a patient?"): the revised NIOSH lifting equation "yields a
    recommended 35-lb. maximum weight limit for use in patient-handling tasks": 35 lb = 15.88 kg = 155.7 N. The design's 160 N
    sustained was that limit rounded UP; a cap may only tighten, so both hands' sustained cap is the source's 156 N.
  - Snook and Ciriello 1991 (Ergonomics 34(9):1197-1213, "The design of manual handling tasks: revised tables of maximum acceptable
    weights and forces"; the tables as reprinted in "Using the Snook Push/Pull Tables", ergonomiesite.be, read 2026-09-24; Garg,
    Waters, Kapellusch and Karwowski 2014, Int J Ind Ergon 44:281-291, find the 1991 female values still valid): two-handed forces
    acceptable to women, at the lowest handle height tabulated (64 cm; a kneeling woman's hands at a body on the floor are lower
    still), over 2.1 m, one exertion every 5 min / 30 min / 8 h:
        sustained pull  75% of women 13 / 14 / 18 kg (128 / 137 / 177 N); 50% of women 17 / 18 / 23 kg (167 / 177 / 226 N)
        sustained push  75% of women 13 / 14 / 17 kg (128 / 137 / 167 N); 50% of women 17 / 18 / 23 kg (167 / 177 / 226 N)
        initial pull    75% of women 26 / 27 / 28 kg (255 / 265 / 275 N)
        initial push    75% of women 19 / 20 / 21 kg (186 / 196 / 206 N); 50% of women 23 / 24 / 25 kg (226 / 235 / 245 N)
    So the sustained 156 N lies between the 75th-percentile woman's acceptable force (128-177 N by frequency) and the median
    woman's (167-226 N): a person of ordinary strength, not a strong one; which percentile the parent is is written down for the
    design (the owner's "a person of real strength"). The brief 200 N (an initial force, held at most 2 s) lies within the
    75th-percentile woman's initial push (186-206 N) and under her initial pull (255-275 N): kept.
  - ONE HAND: the tables are two-handed; no one-handed acceptable force was found. Garg et al. 2014 cite one-handed dynamic pulling
    STRENGTHS (Garg and Beller 1990, men: a mean 180-360 N at 1.1-0.7 m/s; women 65% of men: Garg et al. 1988, Fothergill et al.
    1991), so a woman's one-handed dynamic pull averages about 117-234 N. The one-hand caps (100 N sustained, 150 N for 2 s) are
    under that: kept as ours, flagged (no acceptable-force source).
The caps bound HER effort, whatever the direction (a squeeze counts as a push): the sum of her holds' force magnitudes and of her
body's contact forces on the G1 (each contact's normal and friction force together), so her holds give way to what her body
already presses (the W2 verifier's finding: holds alone were counted, and friction never).
Every constant that shapes her behaviour is here; parent_motion.py names only its own guards (a planning fault, a cache)."""

# ------------------------------------------------------------------------------------------------------ her caps (A25; ours, checked)
CAP_ONE = 100.0                 # N, one hand, sustained (ours; under a woman's one-handed dynamic pulling strength, see above)
CAP_ONE_BRIEF = 150.0           # N, one hand, for at most BRIEF_S (ours; the same)
CAP_TWO = 156.0                 # N, both hands together, sustained: NIOSH's 35 lb patient-handling limit (Waters 2007); the
                                # design's 160 tightened to its own source
CAP_TWO_BRIEF = 200.0           # N, both hands, for at most BRIEF_S: within Snook and Ciriello's initial push for 75% of women
BRIEF_S = 2.0                   # s: the brief caps' longest use (A25); after it, the sustained cap until she has rested
BRIEF_REST_S = 300.0            # s under the sustained caps before a brief cap is hers again: Snook and Ciriello's initial forces
                                # (the source of CAP_TWO_BRIEF) are for one exertion every 5 min at most, their lowest tabulated
                                # frequency (the W2 verifier's finding: 2 s, the burst's own length, was looser than the source)

# ------------------------------------------------------------------------------------------------------------- the capped spring
HOLD_K = 2000.0                 # N/m: her hand's stiffness on a held point (ours: the order of a human arm's endpoint stiffness
                                # in a steady effort, Mussa-Ivaldi, Hogan and Bizzi 1985, Perreault et al. 2001, recalled, not
                                # checked). The cap bounds the force; K only sets how far a hold gives before the cap
HOLD_C = 60.0                   # N s/m: its damping on the held point's velocity relative to her hand (ours; stable at 2 ms
                                # steps on the lightest held link, measured by body/tests/test_sim_parent.py)
AT_CAP = 0.98                   # a hold is "at its cap" on a step when its force is at least this share of the cap (ours)
AT_CAP_TICKS = 2                # a hold at its cap on every step of this many ticks stops (A10, A25: "a hold or guide at its cap
                                # for 2 ticks stops")
HOLD_SLIP_M = 0.12              # m: a held point this far from her hand's reach has slipped from her grip (ours: a hand's length)

# ------------------------------------------------------------------------------------------------------ her acts' own caps (A8-A10)
TOUCH_N = 5.0                   # N: a hand resting on the child (attend, "touch"): the relaxed hand's weight, 0.6% of a 60 kg
                                # woman's body weight (Winter's segment table: the hand 0.006 M), with some of the forearm's
                                # (ours)
ATTEND_PARTS = "tummy|leg|foot" # attend's resting hand: on its trunk (4.2), or, where only its feet leave her room to kneel (A6:
                                # a child in a corner), on its leg or its foot, the first her hand reaches (ours)
TOUCH_DEPTH_M = 0.02            # m: a resting hand's target lies this far inside the surface along its normal (the cap decides
                                # the force; ours)
GUIDE_CAP_FACTOR = 1.5          # the guide's cap: min(1.5 x the limb's own push at that pose, CAP_ONE) (A10; ours)
GUIDE_MAX_TICKS = 8             # a guide's path lasts at most one movement unit (A10)
GUIDE_MAX_SPEED = 0.30          # m/s, the guide's fastest pace (A10); its pace is set on the yielding arm (A25): GUIDE_SPEED
GUIDE_SPEED = 0.30              # m/s: the guide's pace, set in W2 on the limp arm A25 names (tools/sim_parent_motion.py guide_pace:
                                # the arm's actuators zeroed, the near forearm raised GUIDE_RAISE_M), the fastest of 0.30 / 0.25 /
                                # 0.20 / 0.15 / 0.12 / 0.10 / 0.08 m/s whose peak stayed within the arm's own push (93.9 N there):
                                # peaks 53.5 / 50.3 / 50.3 / 47.2 / 47.1 / 47.0 / 36.1 N, all within it and within the design's
                                # 65 N, so A10's fastest. Under the resting servo law the same guides peak 99.0 / 98.5 / 98.5 /
                                # 93.4 / 90.1 / 88.7 / 76.9 N against its 94.2 N push: an arm that holds meets her cap and stops
                                # her (A10, A25). The W2 verifier's finding: 0.15 m/s had been set on the resting arm, with a
                                # 1.3 N margin. Never tuned to the child's learning (A25)
ROLL_ARM_N = 65.0               # N: the far arm guided across the chest (A8)
ROLL_KNEE_N = 76.0              # N: the far knee bent over, one hand (A8; section 4.2's "bend a leg: 60-76 N")
CAP_RAMP_NPS = 100.0            # N/s: a pull or a turn grows its force at this rate, so it reaches CAP_TWO_BRIEF at BRIEF_S
                                # (ours: "a slowly growing force", A7, A9)
TURN_MAX_S = 2.0                # s: the brief turn from its front lasts at most this (A7)
GUIDE_RAISE_M = 0.12            # m: a guide raises the near forearm this far (ours: within one movement unit at GUIDE_SPEED,
                                # 8 ticks x 0.30 m/s = 0.36 m, and inside the arm's reach off the mat)
FOREARM_HOLD = (0.07, 0.0, 0.0) # where she takes a G1 forearm (its elbow link's frame): 7 cm down it from the elbow, ...
FOREARM_HOLD_N = (0.0, 0.0, 1.0)    # ... her palm on its surface along this normal (the link's +z: its top as it lies; ours)
ROLL_ARM_M = 0.15               # m: the far arm drawn this far toward her and up across its chest (A8; ours)
KNEE_OVER_M = 0.15              # m: the far knee drawn this far toward her and up (A8; ours)
PULL_LEAD_M = 0.30              # m: the pull's spring target lies this far ahead of the held forearm, so its force is set by the
                                # ramp and her caps, never by how far the spring stretches (HOLD_K x 0.30 = 600 N, over every cap;
                                # ours)
TURN_LIFT_M = 0.12              # m: the turn's spring target lies this far along its push (ours: as PULL_LEAD_M, HOLD_K x
                                # 0.12 = 240 N, over the one-hand brief cap it is given) ...
TURN_OVER_UP = 0.4              # ... its push straight up while the chest faces down, then this much up with the rest toward
                                # its back, away from her (ours) ...
TURN_SIDE_Z = 0.3               # ... switched when its chest's normal rises past this (a unit vector's z: face down -1, on its
                                # side 0; ours). It has turned when it lies on its back, or on its side with its chest past this

# ------------------------------------------------------------------------------------------------------ the prop and the catch (A9)
PROP_MAX_DEG = 30.0             # the prop engages only with the trunk within this of vertical (A9)
PROP_EASE = (1.0, 0.7, 0.4, 0.2)    # the prop's cap steps (x CAP_TWO), then hovering (A9)
PROP_STEADY_DEG = 20.0          # the trunk within this of vertical ...
PROP_STEADY_TICKS = 10          # ... for this many ticks steps the cap down (A9)
HOVER_M = 0.05                  # m: the hovering hands' gap (A9)
CATCH_DEG = 35.0                # the catch: the trunk past this from vertical ... (A9)
CATCH_HEAD_MPS = 0.5            # ... or the head dropping faster than this (m/s) (A9)
REACTION_TICKS = 2              # her reaction: 300 ms, a person's; never changes (A9, C6)
LAY_BACK_S = 2.0                # s: laying the child back gently eases her hold to nothing over this (ours)
# The catch's springs pull at once, up to her brief caps while she has them (else her sustained ones), toward where the trunk sat
# most upright while she propped it (a person catching pushes back; she does not merely stop where her hands met it); they pull
# at most BRIEF_S, and a trunk not back within PROP_MAX_DEG by then is laid back gently. C6 counts a fall as caught when it is
# stopped short of CATCH_STOP_DEG (back within PROP_MAX_DEG, or held until she lays it back) and neither its trunk, head nor
# pelvis strikes the mat, the floor, the furniture or a toy at F_pain (A12's 10 ms filter) while it falls, never counting its own
# links pressing each other (tools/sim_parent_motion.py catch, Strike)
CATCH_STOP_DEG = 60.0           # a fall stopped short of this is caught (ours: past 60 deg its upper body lies closer to the mat
                                # than to upright, and its head is within about 0.2 m of the mat)

# ---------------------------------------------------------------------------------------------- her contacts with the child (A4)
SOFT_SOLREF = (0.05, 1.0)       # her collision shapes' contact: soft, like flesh and cloth (A4), taken by every contact with the
                                # G1 through contact priority SOFT_PRIORITY (the world's own mechanism, A21), the G1 untouched
SOFT_PRIORITY = 2
YIELD_STEPS = 2                 # a contact over the act's cap for 2 physics steps ... (A4)
YIELD_M_PER_TICK = 0.02         # ... stops that segment and backs it off 2 cm a tick along the contact (A4)
YIELD_MAX_M = 0.12              # m: backed off this far in one push and still pressed on, she gives the act up (ours: a hand's
                                # length) and gets out of its way at her own pace (WALK_MPS standing, SHUFFLE_MPS kneeling,
                                # REACH_MPS a hand), still along the contact and never into furniture (along it), until the push
                                # ends; at rest, where it pushed her is where she stays
YIELD_BACK_M_PER_TICK = 0.01    # m a tick: a chain backed off comes back at half the yield's rate once the child has left it
                                # alone for her reaction time (REACTION_TICKS), and only where it stays CLEAR_M from the child
                                # (A4: "the act resumes when the force is gone"; ours)
# Her body backs off on the floor plan away from the child's centre of mass, turned toward the contacts' own way out as far as they
# agree (no constant: the W2 verifier's seed 6 had two contacts pointing opposite ways, a leg between her shins, flip her way out
# every step); her arms along their contacts (an arm low by the floor drawn out along it). A BLOW, a contact over her own pain
# (HER_PAIN_N, 4.10: the child's kick), moves the struck chain out at once by as far as her own shapes went in (the contact's
# depth; a toy in her hand is left out), each step from the first, along the contacts' own way out as far as they agree, the rest
# away from the child's centre of mass through where she is struck (her head leaning over it goes up and back; risen, she comes
# down again only clear of the child): her body gives with a blow as a person's does, never a wall, as her hands are never let
# into the child. No speed constant of its own: the give is what the child's own motion put into her (the W2 verifier's finding:
# W2's BLOW_GIVE_M_PER_STEP, 5 m/s, had been reasoned from a head struck at 2 m/s). An arm is given at most YIELD_MAX_M in a tick
# (a forearm pinned under the child, given its depth every step, had run 1.6 m in a tick), and an arm pushed back farther than
# YIELD_MAX_M gives the rest to her whole body at her own pace (SHUFFLE_MPS kneeling, WALK_MPS standing).
# A4's 2 cm a tick is kept for a press under her pain. The lead's decision is asked for: the design's A4 says "2 cm a tick" only
PATIENCE_TICKS = 40             # ticks: an act held up this long in a row by the child pressing against her is given up (6 s; ours)
HAND_CLEAR_M = 0.003            # m: her hand (its palm, fingers and thumb) kept this far outside the G1's collision surface (its
                                # convex hulls, which enclose its drawn meshes) while it holds or rests on it: the spring is her
                                # grip, so her hand's shapes never pass through the child (A4; ours)
HAND_FREE_M = 0.01              # m: ... and this far from every other link of the child (they move on their own: a dangling wrist
                                # swung 1.6 cm nearer her thumb in a tick than its velocity carried it, W2). Both measured as her
                                # hand is drawn at the tick's end (the child now and a tick ahead) and again within the tick (the
                                # held link every 20 ms, every other link every step: the W2 verifier's finding, a babbling limb
                                # passed into her fingers), the hand moved out along the held surface's normal by what it lacks; a
                                # hand pushed off its hold a hand's length (HOLD_SLIP_M) has slipped. Her hand not holding touches
                                # the G1 with its palm, fingers and thumb (their collision shapes on) and is moved out of it at once
                                # by as far as it went in (ours)
HOLD_SHAPE = (0.10, 0.0)        # her hand's (curl, thumb) on the child: open and flat, the fingers along its surface (a curled
                                # grip's fingers passed 3.6 cm into its torso: the W2 verifier's finding; ours)
HER_PAIN_N = 150.0              # N: her own pain, on any segment, as a 10 ms mean (4.10; a person's, ours)
HER_PAIN_STEPS = 5              # the 10 ms mean: 5 physics steps (the child's pain filter's length, A12)
WITHDRAW_M = 0.12               # m: a hand withdrawn when hit draws back this far (ours)
SUPPORT_BEND_DEG = 45.0         # leaning (lean + spine) this far, her free hand rests on her own thigh (a hand hanging from a
                                # far lean reached down onto the child; the floor beside her knee was where its arm lay: W2; ours)
SUPPORT_OFF_DEG = 35.0          # ... and lets it hang again once she straightens past this (ours: 10 deg under, so it does not
                                # flicker at the edge)
OFFER_HOLD_TICKS = 40           # ticks: her open hand held out for a toy (the give ask's level 0), or her hands over her face
                                # (peekaboo), wait at most this long for the next act (ours: A4's own patience with the child's
                                # hand, 40 ticks for a hand-over's release); the act is under way while they wait (P3's contract)
GESTURE_HOLD_TICKS = 4          # ticks: a hand held out (an open hand, a raised arm, a fist) stays this long before it rests
                                # (ours: 0.6 s, long enough to be seen at the child's 150 ms tick)
DO_WALK_M = 1.0                 # m: 'do walk' (a verb's showing): a few steps, about three strides, away and turned back to the
                                # child (ours)
IDLE_RELAX_TICKS = 10           # a hand left out by a finished act relaxes after this long unless it holds something (ours)
CLEAR_M = 0.03                  # m: her legs, trunk and head kept this far from the child in every planned frame (A4)

# ------------------------------------------------------------------------------------------------------------- her paths (A6)
GRID_M = 0.05                   # the floor grid of her paths (A6)
CLEAR_FURNITURE_M = 0.15        # a path keeps this far from furniture and walls (A6)
CLEAR_CHILD_M = 0.25            # ... from the child's body (A6: outside its leg sweep)
CLEAR_TOY_M = 0.08              # ... from toys (A6)
BODY_R_M = 0.20                 # m: her own half-width on the floor plan, walking (ours: a woman's hip breadth, about 0.37 m)
WALK_MPS = 0.8                  # walking (A6); parent_poses.walk's gait runs at its own SPEED and is timed to this
SHUFFLE_MPS = 0.25              # shuffling on the knees (A6)
TURN_DEG_PER_S = 180.0          # turning in place (ours: a half turn in a second)
KNEEL_DOWN_S = 2.4              # s: standing to sitting on her heels at the least (ours: parent_poses.kneel_down's three parts)
STAND_UP_S = 2.4                # s: the way back up at the least (ours)
KNEEL_SEG_MPS = 0.8             # m/s: no segment of hers moves faster than this kneeling down or getting up (ours: her stepping foot
                                # and her knee coming down set the pace; with parent_poses.kneel_down's own path, 3.1 s all the way)
KNEEL_OFF_M = (0.72, 0.78)      # m: her kneeling spot from the G1's torso centre line, beside its chest (A6)
KNEEL_OFF_TRY = (0.75, 0.72, 0.78, 0.84, 0.90)   # the offsets she tries in turn: A6's range first, then a little further out when the
                                # child's arm lies where her knees would go (W2: its arms sink and move out from the birth pose)
KNEEL_ALONG_M = 0.10            # m: ... toward its feet from its trunk's middle (section 4.2's measured spot)
HIPS_ALONG_M = (0.30, 0.40, 0.20)   # m toward its feet from its trunk's middle, tried in turn: beside its hips, where she kneels
                                # to bend its far knee over (A8) or to take both its forearms (A9): from beside its chest the
                                # far knee lies beyond her reach (W2: 12 cm short). Ours
PULL_FEET_M = (0.35, 0.40, 0.45, 0.50, 0.55)   # m beyond its feet (0.70 m from its pelvis along it) where she kneels on her
                                # heels for the pull-to-sit, facing its head (A9: "she kneels at the G1's feet and holds both
                                # forearms"), nearest first: her knees about 0.34 m ahead of her heels' spot, so from just beyond
                                # its soles; a spot is taken only where every kneeling frame keeps 3 cm from it (A4) and one trunk
                                # of hers reaches both its forearms (C7). Ours (the W2 verifier's finding: W2 had knelt beside its
                                # hips, which A9 does not say, and refused even there)
KNEEL_CLEAR_FRONT_M = 0.6       # m: kneeling down needs this clear in front, so she kneels a step back and shuffles in (4.1)
KNEEL_FOOTPRINT_R = 0.35        # m: toys within this of her knees' line where she kneels are cleared first (ours)
TOY_ASIDE_M = 0.35              # m: a cleared toy is set down this far beyond her footprint, away from the child (ours)
TRUNK_DEG_PER_S = 30.0          # deg/s: her trunk's unhurried lean, down or up (ours; the lean phase's pace, and her return to
                                # upright when her hands come back to rest: at the old fixed 4 ticks a 67 deg lean came back at
                                # 110 deg/s and her arm swung 0.6 m in a tick, W2)
REACH_MPS = 0.5                 # m/s: her hand's unhurried reach (ours)
REACH_MIN_S = 0.45              # s: the shortest hand move (ours: 3 ticks)
APPROACH_M = 0.10               # m: a hand comes onto the child, or a toy, along the surface's normal from this far out (ours)

# ------------------------------------------------------------------------------------------------------------ her face (A3, A22)
FACE_MIN_M = 0.25               # her face never closer than this to the child's eyes (A3)
FACE_OFF_LINE_DEG = 15.0        # when she leans in or calls, her face arrives at least this far off the fovea's line (A3)
LEAN_DIST_M = (0.30, 0.60)      # m: where she puts her face when she leans in: from the child's eyes (ours: the lean-in
                                # distance, where the grades are visible in its fovea, section 4.3). She kneels where a pose inside
                                # human ranges puts it there (beside its chest or its shoulders), the least bend first; never
                                # widened after measuring
LEAN_MAX_DEG = 70               # her kneeling lean's range in her searches (hip flexion kneeling tall; ours: parent_kin's range)
LEAN_ALONG_M = (-0.10, 0.0, 0.10, -0.20)   # m toward its feet from its trunk's middle: the spots she leans in from, beside its chest
                                # and shoulders first (her face within LEAN_DIST_M at the least bend there, W2: a lean of 50 deg
                                # beside its shoulders, 60 beside its chest; ours)

# ------------------------------------------------------------------------------------------------------ her plans (4.1: "W2")
REPLAN_TICKS = 7                # an act's plan is solved when it starts and again about once a second (7 ticks, 1.05 s) while
                                # its target moves, warm-started from the last solution, interpolated in between (4.1)
REPLAN_MOVED_M = 0.02           # m: the target has moved when it is this far from where the last solve put it (ours)
FOCUS_TICKS = 3                 # ticks: a look "during the focus word" holds on the object this long, then the next look (ours:
                                # a focus word lasts 2-3 ticks at her 3.6 words a second, 4.4)
LOOK_TICKS = 2                  # her gaze reaches its target within 2 ticks (4.10's L1)
SHOW_SHAKE_M = 0.02             # m: a shown toy is shaken this far either way ... (4.10: "shaken for its sound"; ours)
SHOW_SHAKE_HZ = 2.5             # ... at this rate (ours)
SHOW_DIST_M = 0.40              # m: a toy shown before the G1's eyes (4.2, 4.10)
HANDOVER_PALM_N = 0.3           # the hand-over's release: the child's palm touch at least 0.3 N ... (A4)
HANDOVER_CLOSED_DEG = 30.0      # ... and its fingers closed at least 30 deg (A4) ...
HANDOVER_HOLD_TICKS = 2         # ... for 2 ticks, ...
HANDOVER_MAX_TICKS = 40         # ... or after 40 ticks (A4)
TAKE_BACK_N = 0.3               # taking a toy back: she closes only after the child's palm force stayed under 0.3 N ... (A4)
TAKE_BACK_TICKS = 2             # ... for 2 ticks
FEED_FULL = 0.98                # she holds the bottle in the palm until the charge reaches this, or she is asked to stop (ours)
