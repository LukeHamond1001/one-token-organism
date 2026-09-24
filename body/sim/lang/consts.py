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
HOLD = 2                              # a gaze ask is met when X lands in the fovea and stays 2 ticks (4.8)
TURN_END_REST = 2                     # the child's turn ends when its voice has rested 2 ticks after sounding (4.6)
REPLY_AFTER = 3                       # she replies 3 ticks after the child's turn ends (4.6; Goldstein et al. 2003)
CALL_EVERY = 240                      # the call at most once per 240 ticks (4.6, A13)
SAME_LINE = 60                        # the same line not within 60 ticks (4.6)
SAME_OBJECT = 20                      # the same object named at most once per 20 ticks; a variation set counts as one naming (4.6)
SET_PER_OBJECT = 120                  # at most one variation set per object per 120 ticks (4.5)
SET_LINES = (2, 3)                    # a variation set is 2-3 lines sharing the focus word (4.5); a new word's set is 3 (4.8)
NEW_PER_DAY = 3                       # at most 3 new words a life day, each joining her words at the night (4.8, A14, A15)
NEW_EVERY = 400                       # and at most 1 a minute (A14; 60 s at 150 ms a tick), whoever asks: a new word's set of 3
                                      # lines takes its minute (4.8)
REDIRECT_AFTER = 40                   # she redirects ("look! the drum.", with a point) only after 40 ticks with no target (4.10)
TARGET_TICKS = 3                      # the child's target: what its fovea's central ray hits 3 ticks running (4.10, B14)
ECHO_WINDOW = 10                      # anything the child says within 10 ticks of her saying it is an echo (4.8, A27)
VOCAL_TURN_EVERY = 60                 # stage 1: a vocal turn in a pause while looking earns a smile at most once per 60 ticks (4.6)
NONSTOP = (0.7, 40, 20)               # babble that never stops: sounding on over 70% of 40 ticks; she waits 20, then speaks (A13)
RECENT = 40                           # ticks an event she saw or heard stays sayable ("uh oh. the ball is down."): ours
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
                                      # me ...", the meal's "more?" (a name ask, judged); nor "look" with a word after it in its
                                      # sentence ("look at ...", "look here"), nor the child's name (the call: an ask, judged, at
                                      # most once per 240 ticks, A13); templates.claude_claims
STEER_USES = 3                        # each of Claude's lines is used at most 3 times (4.5)

# The never-taught pairs (A28, B2): each held out of every line (templates, Claude's, recasts, echoes) until its test opens, as
# words in one line and as a colour said of an object of that name (the referent check: "it is red." said of the red ball).
HELD_PAIRS = (("red", "ball"), ("blue", "block"), ("yellow", "cup"), ("green", "car"))

# ------------------------------------------------------------------------------------------------ the ledger (4.8)
UNDERSTOOD_LAST = 10                  # understood: met on at least 5 of the last 10 asks (4.8) ...
UNDERSTOOD_MIN = 5
UNDERSTOOD_P = 0.05                   # ... and above the child's own base rate, one-sided binomial p < 0.05 (4.8)
BASE_MIN = 10                         # the base rate counts only once it has 10 trials (ours: the design names no minimum, and a
                                      # rate from 1 or 2 random moments would let 5 lucky asks pass)
SAYS_TIMES = 3                        # says: 3 times over at least 2 life days, never within ECHO_WINDOW of her saying it (4.8)
SAYS_DAYS = 2
EXACT_UNTIL = 3                       # an approximation earns a recast and a smile until the exact word is said 3 times (4.6, A27)

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
EXPECT_AWAY = (PARENT_NAME,)          # "mama" while she is away or out of its view (A27)
# (the names of what is in the child's fovea or hand, the word of a pending ask, and the focus word of her last line are read
# from the moment: body/sim/lang/transcriber.expected_words)

# ------------------------------------------------------------------------------------- the token output's reading (4.9)
EDIT_MAX = 1                          # letters within edit distance 1 of a word (2 for words of 6 letters or more) (4.9)
EDIT_MAX_LONG = 2
EDIT_LONG = 6
PREFIX_MIN = 2                        # or a prefix of at least 2 letters of what the child sees or holds (4.9)

# ------------------------------------------------------------------------------------------------ the worth table (4.3)
WORTH_RIGHT_NAME = 2                  # a met ask; a right name (exact); the call answered, until the name is understood
WORTH_MET_ASK = 2
WORTH_APPROX = 1                      # stage 2: an approximation of a word, until the exact word has been said 3 times, and only
                                      # where the exact word would be a right name (its referent in the child's fovea or hand,
                                      # her face for "mama", or the answer to her name ask): never looser than the exact word
WORTH_VOCAL_TURN = 1                  # stage 1: a vocal turn in a pause while looking, at most once per 60 ticks

assert all(w in BIRTH_WORDS for ws in EXPECT_ROUTINES.values() for w in ws)
