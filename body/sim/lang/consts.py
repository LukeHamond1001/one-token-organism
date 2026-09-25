"""THE PARENT'S CONSTANTS (docs/SIM_DESIGN.md 4.4-4.10, A13-A15, A27, A28 and section 10's "the parent's timings, judgments and
conduct" and "the parent's ear" rows; package P3). Every number here is the teacher's method: ours, set before birth, disclosed,
never fitted to the child's rates or the environment's pace, and never loosened after birth. The body reads none of them.

Each line gives its value, its unit, and where it comes from: a section of the design (which gives its own source: Fernald and
Mazzie 1991, Goldstein et al. 2003, Kuntay and Slobin, Onnis et al. 2008, Tomasello and Farrar 1986, Wood and Middleton 1975),
or "ours" where the design leaves the number to the build.
"""
from .lexicon import BIRTH_WORDS, PARENT_NAME

TICKS_PER_DAY = 24000                 # a life day (section 10)

# ----------------------------------------------------------------------------------------------------- turn-taking (4.6)
PAUSE_RELATED = 6                     # ticks between related lines: a variation set's lines (4.5, 4.6)
PAUSE_EXPECT = 20                     # ticks of expectant pause after a question, an ask or a call (4.6)
JUDGE_GAZE = 20                       # a gaze ask ("where is the X?", "look at the X") is judged over 20 ticks (4.6, 4.8)
JUDGE_ACT = 40                        # an act ask ("give me the X", an action word) over 40 ticks (4.6, 4.8)
HOLD = 2                              # a gaze ask is met when its trunk turns to X or its hand reaches toward X (or takes it), as
                                      # she reads them, and that holds 2 ticks (4.8, A40): her reading of its trunk is the one
                                      # held TARGET_TICKS running (percept.Reader.look), so a turn meets it after 3 + 1 ticks
TURN_END_REST = 2                     # the child's turn ends when its voice has rested 2 ticks after sounding (4.6)
REPLY_AFTER = 3                       # she replies 3 ticks after the child's turn ends on average (4.6; Goldstein et al.
                                      # 2003): the mean of REPLY_PAUSE's draws (2.91), and her latency made perfect for a test
CALL_EVERY = 240                      # the call at most once per 240 ticks (4.6, A13)
SAME_LINE = 60                        # the same line not within 60 ticks (4.6)
SAME_OBJECT = 20                      # the same object named at most once per 20 ticks; a variation set counts as one naming (4.6)
SET_PER_OBJECT = 120                  # at most one variation set per object per 120 ticks (4.5)
SET_LINES = (2, 3)                    # a variation set is 2-3 lines sharing the focus word (4.5); a new word's set is 3 (4.8)
NEW_PER_DAY = 3                       # at most 3 new words a life day, each joining her words at the night (4.8, A14, A15)
NEW_EVERY = 400                       # and at most 1 a minute (A14; 60 s at 150 ms a tick), whoever asks: a new word's set of 3
                                      # lines takes its minute (4.8)
REDIRECT_AFTER = 40                   # she redirects ("look! the drum.", with a point) only after 40 ticks with no target (4.10)
FOLLOW_PER_REDIRECT = 2               # and only while her follow-in namings number at least 2 for each redirect, the one to be
                                      # said counted, over the life day (4.10; Tomasello and Farrar 1986); a Claude line naming a
                                      # toy the child does not attend (as she reads it) is a redirect, one naming what it attends
                                      # a follow-in naming
TARGET_TICKS = 3                      # the child's target as she reads it (4.10, A40): the object nearest its head camera's line,
                                      # held 3 ticks running (percept.Reader.look: her reading moves to a thing, or to nothing,
                                      # only once read() has given it 3 ticks running; every rule reads this held reading); or
                                      # what its hand holds or reaches toward
SETTLE = 44                           # a formal trial's settle (4.8, 12; the lead's decision: understanding is scored only in
                                      # formal trials): once the two things are placed, she holds still (her eyes on the child's
                                      # eyes, her hands resting on her thighs, her face in its neutral set, no act) for 44 ticks
                                      # (6.6 s) before the test sentence, so a look that follows her placing them has ended
                                      # before its window: 6.5 s is the response period over which infants' looks after an
                                      # adult's head turn were scored as following it (Brooks and Meltzoff 2005, Dev Sci
                                      # 8:535-543: "each 6.5-s trial began with the onset of the adult head movement"), in whole
                                      # ticks. Her attention log keeps the last SETTLE + 1 ticks
FACE_COURSE = 77                      # her face moves for at most 77 ticks after a judgment of hers or a frown (A3, and FEEL in
                                      # parent_feel.py: a smile waits at most 40 ticks from its judgment, reaches its apex in 2,
                                      # is held until seen for at most 20 ticks and 10 more once seen, and eases off over 5:
                                      # 40 + 2 + 30 + 5; a frown's 10 + 5 within it). Her attention log holds her face moving for
                                      # that long from the tick she judges, whatever her motion reports: no trial's settle or
                                      # window runs through it (only her mouth may move)
ECHO_WINDOW = 10                      # anything the child says within 10 ticks of her saying it is an echo (4.8, A27)
VOCAL_TURN_EVERY = 60                 # stage 1: a vocal turn in a pause while looking earns a smile at most once per 60 ticks (4.6)
NONSTOP = (0.7, 40, 20)               # babble that never stops: sounding on over 70% of 40 ticks; she waits 20, then speaks (A13)
RECENT = 40                           # ticks an event she saw or heard stays sayable ("uh oh. the ball is down."): ours

# ------------------------------------------------------------------------------------- her reading of the child (A40, C58)
# A real G1 shows no eyes: she reads where it looks as a person reads a robot's head, from the line its head's camera faces (the
# G1 has no neck, so the trunk's line: A22) and its hands, never its fovea's window (percept.Reader). The fovea's window stays the
# child's own and the instruments'.
READ_STREAM = 4                       # her reading's random stream of the body's seed (ours; world 1, tract 2, her lines 3)
READ_ERR_DEG = 4.0                    # her error reading its head's line: yaw and pitch each normal with this sd, drawn per
                                      # reading, a mean angular error of 5.0 degrees (4.0 x sqrt(pi / 2)): Poppe, Rienks and
                                      # Heylen 2007 (Perception 36:971-979), observers judging which target another's head is
                                      # oriented toward, a mean angular error close to 5 degrees (C58)
FOVEA_REACH_DEG = (38.0, 20.0)        # its target is within the fovea's reach of that line, +-38 x +-20 degrees (4.10, A40)
NEAR_DEG = 20.0                       # a look at a thing could be read as any thing within 20 degrees of it as seen from the
                                      # child's head (5 x READ_ERR_DEG; percept.Reader.near fills Seen.near and Percept.face_near
                                      # from it): measured with the Reader itself, 4,000 trials each, a head resting exactly on
                                      # one thing is read as another for a look held 3 readings then 2 ticks within 40 ticks at
                                      # 8 degrees 12%, 12 degrees 0.8%, 16 degrees 0.08%, and at 18 and 20 degrees in none (P3's
                                      # eighth round). So a formal trial's two things lie beyond it of each other and of her face
                                      # as the child sees them, or its sentence waits (4.8): its first look can be read as one of
                                      # them, never the other, and never as her face (the lead's decision)
CAMERA_FIELD_DEG = (88.3, 58.0)       # "in the child's view" as she reads it: before its head camera's field (the D435's grey
                                      # imagers, A38), about the line she reads
REACH_TICKS = 3                       # reaching toward: the hand's path over the last 3 ticks closing on the object (4.10)
REACH_CLOSE_M = 0.03                  # closing on it on each of those ticks, and by at least 3 cm in all (ours: a movement a
                                      # person sees at a metre; the closest-closing object, one a hand)

# ------------------------------------------------------------------------------------- her imperfection (A52, C57)
# Her method, from human dyads, fixed before birth, never fitted to the child's rates; drawn from its own stream. No miss ever
# touches a judgment she has made: a judged act or word still gets its smile within the tick, and its reply.
IMPERFECT_STREAM = 5                  # her imperfection's random stream of the body's seed (ours)
MISS_TURN = 0.30                      # she answers a turn of the child's with no judgment 70% of the time: mothers responded
                                      # within 2 s to over 70% of their infants' prelinguistic vocalizations (Gros-Louis, West,
                                      # Goldstein and King 2006); the rest she misses (no reply)
REPLY_PAUSE_MS = (730.0, 543.6, 50.0, 2974.0)   # her switching pause after the child's sound: 838 switching pauses of mothers
                                      # and infants at 2-5 months, mean 730 ms, sd 543.6, range 50-2974 (Gratier et al. 2015,
                                      # Front Psychol 6:1167); drawn lognormal with that mean and sd (the shape ours), clipped to
                                      # that range; her reply comes round(pause / 150 ms) - TURN_END_REST ticks after the turn's
                                      # end, at the soonest on the tick she knows it ended (after its 2 quiet ticks: a pause of
                                      # 300 ms at least): 2.91 ticks on average, a realized pause of 736 ms (sd 500; the draws'
                                      # 725, sd 509, clipped). Gratier's mean pools both directions; the mothers' own pauses
                                      # (infant then mother) were 135.4 ms shorter than the infants' (their regression), a
                                      # split the paper gives no mean for, so she keeps the pooled figure (her replies, if
                                      # anything, a little slow)
COPY_GAP_S = 10.0                     # she copies its visible arm and hand movements (a raise, a wave, a shake, an open hand),
                                      # mirrored, at most as often as mothers match their infants' acts, about 6 times a minute
                                      # (Pawlby 1977, as cited by Ray and Heyes 2011 and de Klerk et al. 2019): after each copy
                                      # the next waits an exponential gap of mean 10 s
COPY_DELAY = (7, 13)                  # within 1-2 s of the movement (4.10): uniform in 7..13 ticks
COPY_KINDS = ("arm_raise", "wave", "shake", "open_hand")   # the movements she copies (percept events, the side as object)
# Her spells of distraction during her own tasks (A52) belong to the day plan's own-tasks episode (P4), from the same sources.
# The day plan's (P4, body/sim/lang/day.py to come): disclosed here with their sources, read by no code of P3's.
CALLS_UNANSWERED = (3, 720)           # after 3 calls unanswered in 720 ticks she carries on where the child looks (A13)
IDLE_EVERY = 40                       # idle (watching): at most one line per 40 ticks (4.10; the episodes' idle lines are P4's)
WIND_DOWN_EVERY = 40                  # winding down: at most one line per 40 ticks, no asks, no new toys (4.7)
BID_ANSWER = 5                        # away: she answers a bid from afar within 5 ticks ("mama is here.") (4.7, 4.10)
HALL_CALL_EVERY = 600                 # away: she calls from the hall about every 600 ticks (4.7, 4.10)

# ------------------------------------------------------------------------------------------------ the line check (4.5)
MAX_WORDS = 6                         # at most 6 words a line (4.4, 4.5; Fernald and Mazzie 1991's short phrases)
PUNCT = ".?!"                         # the only punctuation (4.5; the form's pattern, templates._FORM, is built from it)
PRAISE = ("yes", "good")              # Claude may not write praise: "yes", "good", the approval register (A14)
REPRIMAND = ("no",)                   # nor stage 2's "no." (4.10: the frown's line): a judgment too, the fast layer's (A14)
ASK_WORDS = ("where", "what", "give", "more")   # nor an ask (A14), anywhere in the line: "where is ...", "what is ...", "give
                                      # me ...", nor the meal's "more?" (the feed routine's own line); nor "look" with a word
                                      # after it in its sentence ("look at ...", "look here"), nor the child's name (the call:
                                      # an ask, judged, at most once per 240 ticks, A13); templates.claude_claims
STEER_USES = 3                        # each of Claude's lines is used at most 3 times (4.5)

# The never-taught pairs (A28 as amended by A55, B2): the colour twins are the targets, the blue ball, the red block, the yellow
# cup and the green car; each pair held out of every line (templates, Claude's, recasts, echoes) until its test opens, as words
# in one line and as a colour said of an object of that name (the referent check: "it is blue." said of the blue ball). So each
# colour word is heard on its original and other kinds, never with its held-out noun: "red" on the red ball, never a block.
HELD_PAIRS = (("blue", "ball"), ("red", "block"), ("yellow", "cup"), ("green", "car"))

# The new word on its pitch peak (A34): she says the day's new word only in a line measured to put it on the line's pitch peak,
# the new word emphasized in the new-word register (4.4): its peak F0 (tools/sim_voice_check.py's f0_word: the largest of its
# voiced frames median-filtered over 3, octave errors dropped) above every other word's in the line: a tie at the tracker's
# resolution (one lag of its autocorrelation, about 4 Hz near 255 Hz) is no peak (P3's fourth round). The measure is
# tools/sim_voice_check.py --peak, over every line she can say with a growth word as the new word (templates.new_word_lines),
# written to PEAK_FILE; a line not measured, or measured off its peak, fails the line check.
PEAK_FILE = "peak_lines.json"         # beside this file

# ------------------------------------------------------------------------------------------------ the ledger (4.8)
UNDERSTOOD_LAST = 20                  # understood (4.8): over its last 20 scored formal trials, its thing fresh and never-taught
                                      # (A28), whichever of the two was named (about 10 of each: which is named is a fair coin),
UNDERSTOOD_MIN = 5                    # (i) its first look on its thing at least 5 times when named, and above 1/2, and (ii) its
UNDERSTOOD_P = 0.05                   # first look where the word said sent it above 1/2, each a one-sided binomial p < 0.05
                                      # (ledger.two_part): (ii) exact whatever the child's favourites, drift or learning (the
                                      # named thing a fresh fair coin each trial), (i) the credit on this word, not the other's;
                                      # the name: (ii) with at least 5 turns after its name. Her everyday asks never count (the
                                      # lead's decision). P3's ninth round measured the rule it replaces, against a yoked or foil
                                      # rate estimated over all time and treated as known: a voice-turner's name understood in 15
                                      # of 40 lives within 100 name trials, a growing one's in 94% within 120
CLAIM_P = 0.01                        # section 12's claims: (i) and (ii) over a form's trials, each counted once, one-sided
                                      # p < 0.01 (A19, A28; Ledger.pooled)
ASKS_KEEP = 10                        # her everyday asks' last 10 outcomes kept in each word's record (teaching only)
SAYS_TIMES = 3                        # says: 3 times over at least 2 life days, never within ECHO_WINDOW of her saying it (4.8)
SAYS_DAYS = 2
EXACT_UNTIL = 3                       # an approximation earns a recast and a smile until the exact word is said 3 times (4.6, A27)

# ------------------------------------------------------------------------ formal trials (4.8, 12, A28; the lead's decision)
# Understanding is scored only in formal trials, as infant labs score it: intermodal preferential looking (Golinkoff et al.
# 1987), looking-while-listening (Fernald et al. 2008; Bergelson and Swingley 2012), and name recognition against a foil name in
# the same voice (Mandel, Jusczyk and Pisoni 1995). Her everyday asks are her teaching: they may earn her smile, and count toward
# nothing ("understood", "says" or any milestone).
TRIAL_STREAM = 6                      # her trials' random stream of the body's seed (ours; world 1, tract 2, her lines 3, her
                                      # reading 4, her imperfection 5): which of the two things is named, the sides, name or foil
TRIAL_FORMS = {                       # the design's never-taught forms (section 12, A28, A55), each said as one test sentence
    "place": "a known word's thing seen in a place within its view it has never been seen in, or from a new angle, beside "
             "another known thing: 'where is the X?' (M6(a))",
    "exemplar": "a never-seen exemplar of X beside a known non-X: 'where is the X?' (M6; Quinn, Eimas and Rosenkrantz 1993)",
    "combination": "known words in a combination never heard, the colour twin beside its original: 'where is the C X?' (M6(b), "
                   "A28, A55); run only once both its words are understood alone (A28)",
    "name": "its name or a foil name, same voice, same stillness, no things (Mandel, Jusczyk and Pisoni 1995)",
}
TRIAL_WINDOW = 20                     # a trial's window: 20 ticks (3 s) from the target word's onset (the lead's decision; 4.6's
                                      # 20 ticks for a gaze ask, opened at the onset as looking-while-listening measures from it:
                                      # Fernald et al. 2008); the first look (her reading of its head line, held TARGET_TICKS, or a
                                      # reach) to the target or the distractor decides it. Every sentence the trial's draw
                                      # could have given is made on one identical timeline (P3's twelfth round: TRIAL_FILE,
                                      # stimuli.py), so its onset and her sentence's end are the same ticks whichever is named
TRIAL_EAR_TICKS = 1                   # the child hears her sound at most a tick after it plays: its ears' 7.7 ms of latency and
                                      # the 25 ms frame that ends 10 ms after it (body/sim/ears.py), with no room echo there yet
                                      # (B9's first-order echo, W5's to add, would lie within 30 ms in this room). The name test
                                      # holds a foil's "no turn" past its window by every tick from the name's onset to her
                                      # clip's end and this one (P3's thirteenth round: a turn timed from any moment of her
                                      # sound, at any level or precision, lands at most that many ticks later after one name than
                                      # after another, so it can never be in the window after its name and outside the span
                                      # after a foil); must grow if W5's echo outlasts a tick (ours)
TRIAL_WAIT = 400                      # a probe whose things are not placed, or whose settle or display does not hold, within 400
                                      # ticks (a minute) of its start is dropped and logged, never said (ours)
CHANCE_2AFC = 0.5                     # a pair trial's chance: 50% by counterbalancing (her trial stream draws which of the two is
                                      # named and the sides, each a fair coin, so any child whose first look does not depend on
                                      # the word said goes where the word sends it on each trial with probability 1/2 exactly);
                                      # the design's "or its yoked rate where higher" is the ledger's two-part test on one
                                      # stretch of trials (ledger.two_part), never a rate carried from other times
NAME_FOILS = ("viv", "vib", "pew")    # the name's foils (Mandel, Jusczyk and Pisoni 1995's name against a foil in the same voice):
                                      # one stressed syllable as "pip", names she never uses for it or for anything (none within
                                      # edit distance 1 of any of her 128 words), said in the name's register and stillness,
                                      # and matched to "pip." in her voice as measured (P3's tenth round, over the 993 such
                                      # consonant-vowel-consonant names: "X." in the calling register, emphasized as the name
                                      # is, with its clip's ticks equal (5) and its energy within 5% of "pip."'s: four pass,
                                      # and "hyd", spelled as her "hi" with a consonant after, is left out): "pip." 738
                                      # ms, "viv." 717, "vib." 729 and "pew." 727; energy 99%, 103% and 101%; the loudest 10
                                      # ms 93%, 93% and 90% of the name's; its rise to half of that 100 ms, as the name's;
                                      # median F0 242, 242 and 276 Hz against 250; the word by the voice's marks 380, 420 and
                                      # 430 ms against 360. Left, disclosed (C66): the name's loudest moment 7-10% above each
                                      # foil's. The ninth round's "tib", "vek" and "jem" ran 6 ticks, 490-570 ms and 109-176%
                                      # of its energy: a child drawn to short or quiet sounds would have turned after its name
                                      # more. test_sim_lang holds them to FOIL_MATCH where the engine is present
FOIL_MATCH = dict(energy=0.05, peak=0.12, rise_ms=10, f0=0.12)   # the foils' match to the name's clip, relative (energy, the
                                      # loudest 10 ms, F0) and in ms (the rise to half the loudest), its ticks equal
# Time-matched stimuli by construction (P3's twelfth round, the lead's decision on C67; body/sim/lang/stimuli.py): every sentence a
# trial's draw could give is said on one timeline, its test word at the engine's own per-word rate and its pitch contour matched,
# and (its thirteenth round) on one carrier recording before its test word, checked on the rendered audio; a set that is not one
# timeline is not used for a trial.
TRIAL_FILE = "trial_lines.json"       # the recipe, beside this file (tools/sim_voice_check.py --trial --write)
TRIAL_LOUD_DB = -36.0                 # each tick of a trial sentence loud (its RMS above -36 dB of the engine's full scale: within
TRIAL_SILENT_DB = -60.0               # 20 dB, a tenth of the pressure, of her plain speech's level, synth.SYNTH_RMS's -16) or
                                      # silent (its loudest 10 ms below -60: the edge of hearing at a metre, where -16 is 62 dB
                                      # SPL, so -60 is 18 dB SPL, under the words channel's 20, lexicon.AUDIBLE_DB), none between,
                                      # the loud ticks the same in every sentence (ours). A tick's RMS is at most its loudest 10
                                      # ms, so her sound and her mouth (the tick's loudness the face shows, playback.Utterance.
                                      # mouth) start, stop and pause on the same ticks of the clip at any level in the band: on
                                      # the clip's tick grid only, not as the child's ears hear it (their 7.7 ms and 25 ms frames
                                      # carry a word's last sound into the next tick), nor 10 ms by 10 ms, nor above -36, where no
                                      # two words are alike: the trial's scoring answers that (ledger: intention to treat; the
                                      # name's span, TRIAL_EAR_TICKS; P3's thirteenth round). The engine's own floor after a line
                                      # is -75 to -85
TRIAL_RATE_SPAN = 2.0                 # a test word's own rate at most a factor of 2 from its natural rate (the emphasis's 35% of
                                      # the engine's default): the span her own registers' rates run (synth.REGISTERS, 0.15 for
                                      # comfort and the new word to 0.30 for "no."), so a test word is never said faster or
                                      # slower than her voice says anything (ours)
TRIAL_F0 = 0.12                       # the test words' pitch contours matched: the word's F0 at the 10th, 50th and 90th
                                      # percentile of its voiced frames each within 12% of the form's (the median over its
                                      # words), FOIL_MATCH's tolerance for the foils' F0
NOVEL_PRESENTATIONS = 3               # a never-taught item counts only on its first 3 presentations (A28): a place, an angle or an
                                      # exemplar displayed in a trial (as either thing), a combination's pair in each of its
                                      # trials, whichever is named (P3's eleventh round: counted when said, it had depended on
                                      # her coins)

# ------------------------------------------------------------------------------------------ the parent's ear (4.9, A27)
EAR_SHIFTS = (0, 1, 2, 3, 4)          # the child's bands shifted down 0-4 (she adapts to a shorter tract) (4.9)
EAR_NCEP = 12                         # cepstra c1-c12, mean-normalized (4.9)
EAR_TRIM_DB = 25.0                    # an utterance's frames within 25 dB of its loudest (the voice study's; ours)
EAR_SLOPE = 2.0                       # Itakura's constraint: a template 1/2 to 2 times the utterance's length (the study's)
# Her margins by the size of her expected set, 1-8 (a larger set takes size 8's), fixed before birth and never loosened after, set
# by one principle: a sound that is not the context word passes as the context word at most 2% of the time (A27 for m; P3 for delta).
#   m      d(word) - d(the babble bank) < -m: held-out babble passes as the context word at most 2% of the time (A27)
#   delta  the word is the nearest of all her words, or within delta of it: the held-out voices' words she does not expect pass as
#          the context word at most 2% of the time (P3: without it they passed as an expected word on 36-95% of trials by size)
# Measured by tools/sim_parent_ear.py on her ear as built (ears.cochlea; the 550 templates of the 50 birth words; a bank of 241
# turns of the stand-in babbler, seed 8; held out: 93 turns of seed 7 x 200 random sets a size for m, the 8 other voices' 50 words
# with their own templates held out x 80 sets a size for delta), 2026-09-24. The stand-in is the voice study's babbler; W4's
# babbler records the bank before birth (P3v), and both tables are measured again on it then. (The study's m = -0.35 was on its
# own cochlea's scale, 1.6 ERB filters and another floor.) Checked on the ear with all 128 words (1,408 templates, the P3
# verifier's item 10): m over sets of the 120 words she can come to have is more lenient at every size (-4.50 to -5.30), so these,
# the stricter, hold as her vocabulary grows (the tool keeps the stricter per size); delta only tightens as words are added. What
# delta costs the tract's own words is measured in 4.9 (it takes "hi", "duck" and "pip" to 0%): delta stands only after P3v.
EAR_FALSE_PASS = 0.02
EAR_M = {1: -4.15, 2: -4.20, 3: -4.20, 4: -4.40, 5: -4.40, 6: -4.40, 7: -4.45, 8: -4.85}
EAR_DELTA = {1: 3.30, 2: 3.35, 3: 3.55, 4: 3.55, 5: 3.65, 6: 3.90, 7: 3.90, 8: 3.90}
EAR_M_SOURCE = "tools/sim_parent_ear.py, the stand-in babbler (seeds 8 and 7), 2026-09-24; P3v measures it again on W4's"
EAR_VOICES = (                        # her templates (A27): (label, voice, pitch, rate); each word said alone
    ("parent_plain", "com.apple.voice.compact.en-US.Samantha", 1.15, 0.25),     # her own voice, plain (4.4)
    ("parent_approval", "com.apple.voice.compact.en-US.Samantha", 1.35, 0.25),  # her own voice, approval pitch (4.4)
    ("child_pitch", "com.apple.voice.compact.en-US.Samantha", 1.45, 0.25),      # the same synthesizer at the old child pitch
    ("Flo", "com.apple.eloquence.en-US.Flo", 1.0, 0.25),                        # 8 other macOS voices (other synthesizers,
    ("Sandy", "com.apple.eloquence.en-US.Sandy", 1.0, 0.25),                    # not recordings of people), each at its own
    ("Shelley", "com.apple.eloquence.en-US.Shelley", 1.0, 0.25),                # pitch and the plain register's rate (ours:
    ("Eddy", "com.apple.eloquence.en-US.Eddy", 1.0, 0.25),                      # the study's clips' rate was not recorded)
    ("Reed", "com.apple.eloquence.en-US.Reed", 1.0, 0.25),
    ("Junior", "com.apple.speech.synthesis.voice.Junior", 1.0, 0.25),
    ("Kathy", "com.apple.speech.synthesis.voice.Kathy", 1.0, 0.25),
    ("Fred", "com.apple.speech.synthesis.voice.Fred", 1.0, 0.25),
)
EAR_DISTANCE_M = 1.0                  # the babble bank is heard as from 1 m (ours; the features are blind to level, the audibility
                                      # threshold is not)

# What she expects the child to say, by situation (A27), fixed before birth and never widened. The transcriber takes the union of
# the rows that hold, kept to the words she knows (her ear knows only the vocabulary: A15).
EXPECT_ROUTINES = {                   # the words of the routine under way (A27)
    "greet": ("hi", PARENT_NAME),
    "wake": ("hi", PARENT_NAME),
    "return": ("hi", PARENT_NAME),
    "feed": ("more", "bottle"),
    "leave": ("bye",),
    "peekaboo": ("peekaboo",),
}
EXPECT_AWAY = (PARENT_NAME,)          # "mama" while she is away or where its eyes cannot reach her (A27)
# (the names of what she reads the child attending, where its head's line is, in its hand or reached toward (A40, never its
# fovea's window), the word of a pending ask, and the focus word of her last line are read from the moment:
# body/sim/lang/transcriber.expected_words)

# ------------------------------------------------------------------------------------- the token output's reading (4.9)
EDIT_MAX = 1                          # letters within edit distance 1 of a word (2 for words of 6 letters or more) (4.9)
EDIT_MAX_LONG = 2
EDIT_LONG = 6
PREFIX_MIN = 2                        # or a prefix of at least 2 letters of what the child sees or holds (4.9)

# ------------------------------------------------------------------------------------------------ the worth table (4.3)
WORTH_RIGHT_NAME = 2                  # a met ask; a right name (exact); the call answered, until the name is understood
WORTH_MET_ASK = 2
WORTH_APPROX = 1                      # stage 2: an approximation of a word, until the exact word has been said 3 times, and only
                                      # where the exact word would be a right name (its referent where she reads it looking, in
                                      # its hand or reached toward, A40; her face for "mama"; or the answer to her name ask):
                                      # never looser than the exact word
WORTH_VOCAL_TURN = 1                  # stage 1: a vocal turn in a pause while looking, at most once per 60 ticks
WORTH_SCAFFOLD_GIVE = 1               # a give after the give ask's scaffolding (her point or touch, once the judged trial's window
                                      # has closed unmet), for its release; counted toward nothing in the ledger (4.10; P4 builds
                                      # the ladders)

assert all(w in BIRTH_WORDS for ws in EXPECT_ROUTINES.values() for w in ws)
