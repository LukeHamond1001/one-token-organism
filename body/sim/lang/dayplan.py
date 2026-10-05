"""HER DAY: THE EPISODE LAYER, L3 (docs/SIM_DESIGN.md 4.7, 4.10, 5.3, A17; package P4, the lead's first build at S5a). What the
parent does across a life day, around what is urgent (the conduct's own priorities, 4.10, come first every tick). Nothing here
reads the child's inside or its rates: the plan is laid out at dawn from her own stream and the day's length, and every line and act
goes through her conduct and her motion as any other. There is no meal: the child has no charge (A88, the owner's decision
2026-09-26; the bottle and its dock are gone from the room).

THE DAY (4.7's table; a life day is DAY_TICKS = 24,000 ticks; a shorter day scales every length by its share of that):
  wake        the first 300 ticks: she comes from the sofa (the world's dawn sends her), greets it once she kneels beside it
              ("hi pip."), then calls it.
  blocks      from 300 to the winding down, in an order drawn from her stream: floor play 3 x 4,000-5,000 ticks, motor time
              2 x 1,000-1,500, show time 1,500, away 2-4 x 400-1,200, her own tasks 3,000; their lengths drawn, then scaled
              together to fill the day exactly. Floor play opens the day; away is never first or last, never two running.
  winding     the last 1,000 ticks: her lids lowered (her feelings' wind-down), no asks, no new words, no new toys.
  goodnight   the last 300: "night night pip." and the sofa.
  floor play  2-3 focus toys among her birth words' toys (drawn); every PLAY_GAP ticks (drawn 150-300) while her voice is
              free, nothing is pending and no act of hers is open: one of showing a focus toy (a variation set with her show),
              peekaboo, an ask of her teaching (where is it / give it / what is it, each gated by her conduct), or the call.
              Following in on what it attends is her conduct's own (4.10) and comes first.
  motor time  HER LESSONS (A90; teacher_of_reality.md 2c): the rung the child is nearly at, by her conduct's book of the smiles
              she has given (never its inside): the reach and grasp rung sets a focus toy within its reach ("set_near", her
              motion's bring_back at lesson_dist: 0.10 m out from its near hand, 0.05 farther each time "got" on that toy has
              been smiled at since, at most LESSON_LEVELS, back a level after NO_PROGRESS ticks with none); once "got" on it is
              mastered (2 x MASTERED_N smiles) the handle rung puts it into its hand ("hand_over") for lifts, shakes and hits;
              once those are, the give rung asks for it ("ask_give"). Floor play gives a lesson on LESSON_SHARE of its offers.
              The roll rung (A109): once it rolls, every other reach-rung lesson sets the toy beside its far shoulder past its
              reach ("set_far"), so a roll brings it within reach. The head-up and sit rungs have no setup act yet.
  show time   her growth words (templates.GROWTH, in order) that the room can show and she can show now, at most the
              conduct's NEW_PER_DAY and NEW_EVERY, and "what is this?" of what it attends.
  away        "bye bye pip." and a wave, the walk to the door; calls from the hall every AWAY_CALL ticks (never judged);
              back after the block, or early after 3 of its vocal turns in 40 ticks. Never begun within 100 ticks of its
              pain (its cry heard) or during distress: the block becomes floor play.
  own tasks   she sits on the sofa (her attention .4); her conduct still answers its turns; back to it at the block's end.
Not yet here: the formal trials by stage and the tests of understanding by milestone (4.8, 12, A55, C36: the conduct's probe(),
asked by this layer once their table is registered before birth), hand-overs of toys and the ball rolled to its hands (her motion's
hand_over and roll acts, until her conduct's intents carry them), and the scaffolding ladders (closed at birth, A25c).

Every length and gap here is 4.7's or ours, disclosed; none is set from a rate of the child's. state() / load_state() carry the
stream, the day's layout and every counter, so a replay is exact.
"""
import math

import numpy as np

from . import consts as K
from . import templates as TP

PLAN_STREAM = 7                        # her day plan's random stream of the body's seed (ours; world 1, tract 2, lines 3, reading 4,
                                       # imperfection 5, trials 6)
DAY_TICKS = 24000                      # a life day (4.7)
WAKE = 300                             # the wake episode (4.7)
WIND = 1000                            # the winding down (4.7)
GOODNIGHT = 300                        # goodnight (4.7)
BLOCKS = (("floor", 3, 4000, 5000), ("motor", 4, 1500, 2000), ("show", 1, 1500, 1500), ("away", 1, 400, 600),
          ("tasks", 1, 600, 600))      # 4.7's table: kind, how many, shortest, longest. TRAINING MODE (2026-09-29, the owner's word: fix
                                       # fast; her pace is the lead's): away 2-4 x 400-1,200 and her own tasks 3,000 cut to one short block
                                       # each, so the play blocks (drawn, then scaled to the day) carry about a quarter more of the day
PLAY_GAP = (30, 60)                    # ticks between her floor play's offers (ours). C269 (2026-10-04, the owner's word: the teacher wastes
                                       # not a second): 150-300 until then (22 to 45 s between offers); life day 85's audit: she neither
                                       # acted nor spoke on 33% of the day's ticks, 6,009 of them in stretches of 3 s or more
SIT_TRIES_PER_BLOCK = 3         # C224: the pull-to-sit offered this many times a motor block when her reach or her hold refused it (ours)
SIT_TURNS_PER_BLOCK = 1         # C254: a child on its front at the sit's offer is turned onto its back first, this many turns a block (ours)
CARRY_AFTER, CARRY_WINDOW = 2, 1500   # C277: this many of her acts refused for want of a spot to kneel beside it within this many ticks, and
PRONE_HURT_TICKS, PRONE_HURT_PAIN = 150, 3   # C278: on its front this long with this many pain ticks, it is laid on its back (the carry); ours
CARRY_WAIT_TICKS = 8                   # C281: the carry waits this long at most for her hands to come off it (ours)
CARRY_LAY_CLEAR_M = 1.3                # C281: it is laid this far from where she kneels at the least (its body is 1.3 m long); ours
CARRY_CLEAR_M = 1.0                   # she carries it back to its mat (her own place this far from the mat's centre); ours
FLOOR_STAND_GAP = 500           # C273: in floor play she stands it up this often when it lies on its back (75 s; ours: a stand takes
                                # about 250 ticks, so a third of her floor play is standing and stepping in her hands)
STAND_AGAIN_GAP = 100           # C268: after a stand done, the next one this many ticks on (15 s of rest; ours)
SIT_RETRY_GAP = 200             # C224: ... the next try this many ticks after the refusal (30 s: a parent tries again in a minute; ours)
SIT_RETRY_WHY = ("cannot reach", "lost their hold", "slipped", "no spot she can kneel", "did not arrive",
                 "the guide sat at its cap",
                 "beyond her reach where she kneels", "her hands are busy",   # C224: the refusals of a moment, not of the child; C230: the gather's guide at its cap too
                 "sat at her cap")               # C257 (2026-10-03): and the pull that sat at her cap for want of the child's own flexion (A9: she never hauls
                                                 # it up; day 73's two pulls, the first in two days, both ended so): the sit rung is learned by repetition, and a
                                                 # parent plays 'up! up!' a few times running; SIT_TRIES_PER_BLOCK a block, SIT_RETRY_GAP apart
                                               # (the arm's own push of the moment: day 65's first offer, her spot and hold good under C227,
                                               # stopped there); C236: a toy in her kneeling footprint she could not clear (day 66, 3,190,004)
AT_HAND_M = 0.85                       # C272: a toy this near where she kneels is at her hand: taken without getting up (ours: inside
                                       # her kneeling reach, the plan's stretch about 0.9 m)
AT_HAND_SHARE = 0.75                   # C272: of her picks among several toys, this share falls on the toys at her hand when some are
                                       # (ours: a parent on the floor plays with what lies by her and fetches now and then)
LESSON_DIST0 = 0.10                    # the reach rung's first distance out from its near hand, m (4.10's ladder, level 1; A90)
LESSON_STEP = 0.05                     # farther each mastered level (teacher_of_reality.md 2c)
LESSON_LEVELS = 7                      # up to 0.40 m (about the arm's reach)
NO_PROGRESS = 1000                     # a level with no new "got" for this long steps back one (about a block)
ASK_GAP = (120, 200)                   # C283: her asks' own cadence, ticks (18 to 30 s; ours): an ask is her words and her eyes, so it does not
                                       # wait for her hands' play (one offer in about 220 ticks on the day-91 copy: her acts are long)
ASK_SHARE = 0.55                       # C283: of floor play's other offers, this share are her asks ('where is the ball?', 'what is this?',
                                       # 'give me the ball.'): the word tested and paid by her smile. Life day 90: none of her 1,062 lines was
                                       # a where, what or give ask (five 'look at the X.'); the mix until then gave asks a quarter of what the
                                       # lessons, shows and peekaboo left, about five a day. With LESSON_SHARE 0.25 (0.4 until then). Ours
LESSON_SHARE = 0.25                    # floor play's offers that are lessons (the rest: shows, peekaboo, asks, the call)
GREET_BY = 200                         # the greeting at the latest by this tick of the wake (ours)
CALL_AFTER_GREET = 60                  # the wake's call this long after the greeting (ours)
AWAY_CALL = 600                        # her calls from the hall (4.7: about every 600 ticks)
AWAY_AFTER_PAIN = 100                  # nor within 100 ticks of its pain (4.7)
DOOR_XY, DOOR_NEAR_M = (2.6, -1.5), 1.0   # C131: the room's door to the hall (make_g1room: the right wall's gap, y -1.95 to -1.05) and how
                                       # near it the child may lie when she leaves: nearer, she does not go (life day 32: the child crawled
                                       # into the doorway while she was out and she stood in the hall 5,000 ticks, no path back; ours)
BIDS_BACK = (3, 40)                    # back early after 3 of its vocal turns in 40 ticks (4.7)
TASKS_ATTENTION = 0.4                  # her attention at her own tasks (parent_feel: .4)
BIRTH_TOYS = ("ball", "block", "duck", "cup", "car", "bear", "drum")   # her birth words' toys in the room (lexicon; no bottle: A88)


def _bucket_at_hand(seen, lane):
    """C152 (2026-09-30): the hide game's bucket is at hand when she sees it or knows its place (lane.toys): since C138 the hide act
    carries the bucket beside the child when it stands beyond its reach, so C130's gate (the bucket within the child's reach, before the
    carry existed) only starved the game: life days 38 to 41 ran 0 to 1 hide a day, and no find has ever been judged"""
    return "bucket" in seen or "bucket" in set(getattr(lane, "toys", ()) or ())


def _lvl_key(o, lvl):
    """C261: her book's key for a lesson toy's got at the reach rung's level `lvl` (conduct._book_key; level 0: the toy's own name)"""
    return f"{o}@{int(lvl)}" if int(lvl) > 0 else o


def _worn(book, o, lvl=None):
    """C141: a toy her smiles have worn out: every act on it that her book pays (got, lifted, shook, hit) has habituated under
    HABIT_FLOOR (conduct._motor_judgments: the n-th smile is worth w e^(-n/HABIT_TAU)). Day 37: the box alone all day (the duck lay
    against the wall), 189 lifts and 73 hits of it and 4 smiles: a person brings another toy when the child has had the box for days"""
    for k in ("got", "lifted", "shook", "hit"):                        # (C261: its got read at the reach rung's level it stands at)
        w = K.MOTOR_WORTH.get(k, (0, None))[0]
        n = int(book.get(k, {}).get(_lvl_key(o, lvl or 0) if k == "got" else o, 0))
        if w * math.exp(-n / K.HABIT_TAU) >= K.HABIT_FLOOR:
            return False
    return True


def _may_give(seen, o):
    """a give may be asked of o (her conduct's own check, 4.8): in the child's view and within its reach as she sees it, not in her hands"""
    s = seen.get(o)
    return bool(s is not None and s.child_sees and s.child_can_reach and s.on != TP.PARENT_NAME)


class DayPlan:
    def __init__(self, seed=1, day_ticks=DAY_TICKS):
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(PLAN_STREAM,))))
        self.day_ticks = int(day_ticks)
        self.day = -1
        self.blocks = []                   # [[start, end, kind, info]] for the day
        self.kind = None                   # the episode now
        self.away = False                  # she is away (the hall) or on her way there
        self.greeted = None                # the tick she greeted it this morning
        self.called = False
        self.focus = []                    # the floor play's focus toys
        self.next_play = 0
        self.next_call = 0
        self.level = {}                    # her lessons (A90): toy -> the reach rung's level
        self.level_t = {}                  # toy -> the tick its level was set
        self.got_seen = {}                 # toy -> the "got" smiles counted at its last lesson
        self.reach_turn = True             # C261: the next lesson on a toy whose reach ladder is open is a reach lesson (they alternate with the later rungs)
        self.roll_turn = True              # A109: the next reach-rung lesson is the roll rung (once it rolls); they alternate
        self.sit_due = False               # A161: a motor block entered with the child on its back owes one pull-to-sit (its first offer)
        self.sit_turns = 0                 # C254: the block's turns of a prone child for the sit
        self.block_i = -1                  # C220: the block the day is in (two motor blocks laid end to end are two blocks, each owing a sit)
        self.sit_tries = 0                 # C224: the pull-to-sit's tries this block (a refusal at her reach or her hold is tried again)
        self.hide_turn = False             # A129: every other lesson on a toy it grasps at will is the hide game (the bucket in the room)
        self.last_pain = -10 ** 9
        self.bids = []                     # its vocal turns heard while she is away (ticks)
        self.night_said = False
        self.log = []                      # (tick, what): instruments

    # ------------------------------------------------------------------ the day's layout
    def _scale(self, n):
        return max(1, int(round(n * self.day_ticks / DAY_TICKS)))

    def lay_out(self, day, lane=None):
        """the day's blocks, drawn at dawn from her stream: 4.7's lengths, scaled together to fill the time between the wake and
        the winding down exactly"""
        self.day = int(day)
        items = []
        for kind, n, lo, hi in BLOCKS:
            k = int(self.rng.integers(n[0], n[1] + 1)) if isinstance(n, tuple) else n
            for _ in range(k):
                items.append([kind, float(self.rng.uniform(lo, hi)) if hi > lo else float(lo)])
        first = next(x for x in items if x[0] == "floor")                # floor play opens the day
        others = [x for x in items if x is not first and x[0] != "away"]
        aways = [x for x in items if x[0] == "away"]
        others = [others[i] for i in self.rng.permutation(len(others))]
        seq = [first] + others                                          # the absences into distinct gaps between blocks: never
        gaps = sorted(self.rng.choice(len(seq) - 1, size=len(aways), replace=False).tolist())   # first, last or two running
        order = []
        for i, x in enumerate(seq):
            order.append(x)
            if i in gaps:
                order.append(aways[gaps.index(i)])
        a, b = self._scale(WAKE), self.day_ticks - self._scale(WIND)
        total = sum(x[1] for x in order)
        t, self.blocks = a, []
        self.block_i = -1                                                   # (C220: a new day's blocks)
        for i, (kind, n) in enumerate(order):
            e = b if i == len(order) - 1 else t + max(1, int(round(n * (b - a) / total)))
            self.blocks.append([t, e, kind, {}])
            t = e
        pool, newest = list(BIRTH_TOYS), None                            # A119 (2026-09-28): the toys she has named join the day's
        if lane is not None:                                             # pool (the growth queue's toys in the room, once said: the
            f = lane.conduct.fast                                        # book, the rattle, the ring, the stacker), and the newest of
            known = [w for w in TP.GROWTH_WORDS if TP.GROWTH_CLASS.get(w) == "toy" and w in lane.toys and (w in f.vocab or w in f.new_words)
                     and w not in TP.OPEN_CONTAINERS]                    # (C119: the bucket is never a focus toy, nor the newest)
            pool += [w for w in known if w not in pool]                  # them is in every day's focus until another is named: a person
            newest = known[-1] if known else None                        # keeps offering the new toy (life day 15: the book's acts all in
        k = int(self.rng.integers(2, 4))                                 # the day's first quarter, then out of its reach; no lesson
        rest = [w for w in pool if w != newest]                          # brought it back, her focus drawn from the birth toys alone)
        fresh = [w for w in rest if lane is None or not _worn(lane.conduct.book, w)]   # C141: the toys her smiles have worn out are
        src = fresh if len(fresh) >= k else rest                         # drawn only when fewer fresh ones are left than the day takes
        draw = self.rng.choice(src, size=min(k - (1 if newest else 0), len(src)), replace=False).tolist()
        self.focus = sorted(o for o in draw + ([newest] if newest else []) if o not in TP.OPEN_CONTAINERS)   # C119: the bucket stays where it stands (her grasp and put miss it by 17 to 50 cm): the hide game's container, never a toy of the day
        self.greeted, self.called, self.night_said = None, False, False
        self.next_play = self._scale(WAKE)
        self.log.append((day, "laid out", [(s, e, k) for s, e, k, _ in self.blocks], self.focus))

    def block_index(self, t):
        """the index of the day's block the tick is in, -1 outside every block (wake, wind, goodnight; C220)"""
        for i, (s, e, _kind, _x) in enumerate(self.blocks):
            if s <= t < e:
                return i
        return -1

    def episode(self, t):
        D = self.day_ticks
        if t < self._scale(WAKE):
            return "wake"
        if t >= D - self._scale(GOODNIGHT):
            return "goodnight"
        if t >= D - self._scale(WIND):
            return "wind"
        for s, e, kind, _ in self.blocks:
            if s <= t < e:
                return kind
        return "floor"

    # ------------------------------------------------------------------ one tick (before her conduct's)
    def tick(self, t, t_day, lane, world):
        c, pm, p = lane.conduct, world.parent, lane._p
        if self.day < 0 or t_day == 0 and self.day != lane.day:
            self.lay_out(lane.day, lane)
        kind = self.episode(t_day)
        if kind != self.kind:
            self._enter(kind, t, t_day, lane, world)
            self.kind = kind
        bi = self.block_index(t_day)
        if bi != self.block_i:
            self.sit_tries = 0                                              # (C224: a new block, its tries afresh)
            self.sit_turns = 0                                              # (C254: and its turn for the sit)
            if self.block_i >= 0 and bi >= 0 and kind == "motor" and self.blocks[bi][2] == "motor" and self.blocks[self.block_i][2] == "motor":
                # C220 (2026-10-02): A SECOND MOTOR BLOCK LAID AGAINST THE FIRST OWES ITS OWN SIT. Life day 61's plan put two motor blocks end
                # to end (5927-7566, 7566-9039): one episode as the day runs them (the kind never changed), so the second block's offer never
                # came; the first's pull was refused at her kneel ('her left hand cannot reach it from here, 38 cm short') and the child,
                # on its back the whole morning, was offered the sit once in two blocks. A161 owes one a block
                self.sit_due = True                                         # (C254: owed whatever its posture; a prone child is turned first)
                self.log.append((t, "motor block two: a sit owed again (C220)" + ("" if self._lying_on_back(lane) else "; the child not on its back: turned over first (C254)")))
            self.block_i = bi
        if any(k == "pain" for k, _o in p.events):
            self.last_pain = t
        if p.child_sounding and self.away:
            self.bids = [b for b in self.bids if b > t - BIDS_BACK[1]] + [t]
        if not hasattr(self, "stood_seen"):
            self.stood_seen = set()
        # C277 (2026-10-05): A CHILD SHE CANNOT KNEEL BESIDE IS CARRIED BACK TO ITS MAT. Day 89: a stand that slipped left it at the
        # room's edge by the wall (2.1, -1.0), and for 5,000 ticks her acts were refused 'no spot she can kneel at', 'no spot to kneel
        # beside the child: every side is blocked' (ten refused against nine done); A110 carried it back only at dawn. A parent picks
        # the baby up and puts it back on its mat. Her arms cannot carry this body (her caps), so the carry is the world's, as at
        # dawn (world.carry_to_mat: laid on its back at the mat's centre): after CARRY_AFTER such refusals within CARRY_WINDOW, with
        # none of her hands on it, no act of hers under way, and her own place clear of the mat. An environment's act, disclosed
        # C281: THE HELD WALK OVER, IT IS CARRIED FROM HER HANDS BACK TO ITS MAT (parent_motion._ctl_stand's carry_due): laid on its
        # back at the mat's centre, or beside the centre where that is clear of where she kneels (CARRY_LAY_CLEAR_M)
        cp_ = getattr(pm, "carry_pending", None)
        if cp_ is not None and t - int(cp_) > CARRY_WAIT_TICKS:
            pm.carry_pending = cp_ = None                                   # (her hands did not come off it in time: no carry)
        if cp_ is not None and not any(h_.kind == "stand" for h_ in pm.holds) and \
                not any(pm.arms[sd_].get("mode") == "hold" for sd_ in "LR"):
            mat_ = np.asarray(world.m.geom_pos[world.m.geom("mat").id][:2], float)
            her_ = np.asarray(pm.base["at"], float)[:2]
            spot_ = next((mat_ + np.array(o_) for o_ in ((0.0, 0.0), (0.0, 0.5), (0.0, -0.5), (0.5, 0.0), (-0.5, 0.0), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5))
                          if float(np.linalg.norm(mat_ + np.array(o_) - her_)) >= CARRY_LAY_CLEAR_M), None)
            pm.carry_pending = None
            if spot_ is not None:
                was_ = [round(float(x), 2) for x in world.d.qpos[:2]]
                world.carry_to_mat(to=spot_)
                world.toys_beside(list(self.focus))                         # C286: its toys with it
                self.log.append((t, "the walk over: carried to its mat (C281)", was_))
                print(f"the walk over: carried to its mat at tick {t} from {was_} (C281)", flush=True)
        if not hasattr(self, "blocked_seen"):
            self.blocked_seen, self.blocked_at = set(), []
        for mid, st in (getattr(c, "ended", None) or {}).items():
            if st == "refused" and mid not in self.blocked_seen:
                self.blocked_seen.add(mid)
                try:
                    why_ = str(c.motion._act(mid).get("why", ""))
                except Exception:
                    why_ = ""
                if "no spot" in why_ or "every side is blocked" in why_ or "cannot get up without touching the child" in why_:   # (C280)
                    self.blocked_at.append(t)
        self.blocked_at = [x for x in self.blocked_at if x > t - CARRY_WINDOW]
        # C278: A CHILD HURTING ON ITS FRONT IS LAID ON ITS BACK. Day 89's last 4,000 ticks: on its front 1,084 of them after a slipped
        # stand and a stand sat down, 42 pain ticks (its wrists under it), her turn refused twice ('no spot she can kneel at puts a
        # far grip of the turn in her reach'). On its front PRONE_HURT_TICKS running with PRONE_HURT_PAIN pain ticks among them, the
        # carry lays it on its back where it lies or on its mat (world.carry_to_mat, as C277: the world's act, disclosed); a prone
        # child that does not hurt is left to its tummy time
        if self._posture(lane) == "front":
            self.prone_run = getattr(self, "prone_run", 0) + 1
            self.prone_pain = getattr(self, "prone_pain", 0) + int(any(k == "pain" for k, _o in p.events))
        else:
            self.prone_run = self.prone_pain = 0
        hurt_ = getattr(self, "prone_run", 0) >= PRONE_HURT_TICKS and getattr(self, "prone_pain", 0) >= PRONE_HURT_PAIN
        if (len(self.blocked_at) >= CARRY_AFTER or hurt_) and kind in ("floor", "motor", "show") and not self.away and not pm.holds and \
                not any(v is not None for v in pm.holding.values()) and \
                not any(a[5] not in ("done", "refused", "cancelled") for a in c.acts_open):
            mat_ = world.m.geom_pos[world.m.geom("mat").id][:2]
            if float(np.hypot(pm.base["at"][0] - mat_[0], pm.base["at"][1] - mat_[1])) >= CARRY_CLEAR_M:
                was_ = [round(float(x), 2) for x in world.d.qpos[:2]]
                if world.carry_to_mat():
                    world.toys_beside(list(self.focus))                     # C286: its toys with it
                    self.log.append((t, "carried back to its mat (C277)" if not hurt_ else "laid on its back: hurting on its front (C278)", was_))
                    print(f"{'laid on its back' if hurt_ else 'carried back to its mat'} at tick {t} from {was_} (C277, C278)", flush=True)
                self.blocked_at = []; self.prone_run = self.prone_pain = 0
        if kind == "motor" and not self.sit_due and self.sit_tries < SIT_TRIES_PER_BLOCK:
            for mid, st in (getattr(c, "ended", None) or {}).items():
                try:
                    a_ = c.motion._act(mid)
                except Exception:
                    a_ = {}
                if a_.get("kind") == "stand_up" and st == "done" and mid not in self.stood_seen:
                    # C268: IT STOOD AND WALKED: AGAIN, after a rest. Standing and stepping are learned by doing them many times
                    # (a parent stands a baby up a dozen times in a play); the block owes the next one STAND_AGAIN_GAP on, as long
                    # as the block runs
                    self.stood_seen.add(mid); self.sit_due = True
                    self.next_play = max(self.next_play, t + STAND_AGAIN_GAP)
                    self.log.append((t, "motor_sit done (stood): owed again (C268)", a_.get("why", "")[:80]))
                    break
                if a_.get("kind") in ("pull_to_sit", "stand_up") and st == "refused" and any(x in str(a_.get("why", "")) for x in SIT_RETRY_WHY):
                    # C224 (2026-10-02): THE SIT IS TRIED AGAIN IN THE BLOCK WHEN HER REACH OR HER HOLD FAILED. Days 61 and 62: every pull-to-sit
                    # offered (one a block, A161; two a day with C220) was refused at her kneel or her hold, 'her left hand cannot reach it from
                    # here (38, 64 cm short)', 'her hands lost their hold on it before she began (it slipped)', 'no spot she can kneel at lets her
                    # do it (pull)': a supine child waves its arms a hand's length in a tick, and the one offer met a bad moment, where the same
                    # pull on the day's copy a few hundred ticks on sat it up (06:45). A parent tries again in a minute; the block's offer is
                    # owed again after SIT_RETRY_GAP, SIT_TRIES_PER_BLOCK tries a block
                    self.sit_due = True
                    self.sit_tries += 1
                    self.next_play = max(self.next_play, t + SIT_RETRY_GAP)
                    self.log.append((t, "motor_sit refused at her reach or hold: tried again (C224)", a_.get("why", "")[:80]))
                    break
        if self.away:
            self._away_tick(t, t_day, lane, world, kind)
            return
        for mid_, st_ in (getattr(c, "ended", None) or {}).items():           # C288: a toy just shown or handed is in its view: the ask now
            if st_ == "done":
                try:
                    if c.motion._act(mid_).get("kind") in ("show", "hand_over", "bring_back"):
                        self.next_ask = min(getattr(self, "next_ask", 0), t)
                except Exception:
                    pass
        if kind in ("floor", "motor", "show") and t >= getattr(self, "next_ask", 0) and c.pending is None and c.trial is None and \
                c.fast.voice_free(t):                                       # (C289: also while her hands hold it standing: on its feet the
            # room is in its view, and her voice is free; day 93's first 6,400 ticks: 3 asks, a toy in view of a child on its back 8% of ticks)
            # C283: HER ASKS ON THEIR OWN CADENCE: of a toy the child sees (a focus toy first), 'where is the X?' twice in three,
            # 'what is this?' once; asked over whatever her hands are doing, never while they hold the child
            seen_ = {s_.id: s_ for s_ in p.seen}
            sees_ = [o_ for o_, s_ in seen_.items() if s_.child_sees and o_ not in TP.OPEN_CONTAINERS and s_.on != "mama"]
            first_ = [o_ for o_ in sees_ if o_ in self.focus] or sees_
            if first_:
                o_ = first_[int(self.rng.integers(len(first_)))]
                c.request(["ask_where", "ask_where", "ask_what"][int(self.rng.integers(3))], o=o_)
                self.log.append((t, "ask (C283)", o_))
                self.next_ask = t + int(self.rng.integers(*ASK_GAP))
            else:
                self.next_ask = t + 10                                      # C286: nothing in its view now: asked as soon as something is
                                                                            # (the whole gap was waited each time, and a toy was in its
                                                                            # view on 40 ticks of 500: no ask in day 92's first 7,500)
        busy = (c.pending is not None or c.trial is not None or not c.fast.voice_free(t)
                or any(a[5] not in ("done", "refused", "cancelled") for a in c.acts_open))
        if kind == "wake":
            if self.greeted is None and (not pm.phases or t_day >= self._scale(GREET_BY)) and not busy:
                c.request("greet"); c.routine = "greet"; self.greeted = t; self.log.append((t, "greet"))
            elif self.greeted is not None and not self.called and t >= self.greeted + CALL_AFTER_GREET and not busy:
                c.request("call"); self.called = True; c.routine = None
        elif kind == "motor" and t >= self.next_play and not busy:
            self._sit_or_lesson(t, lane)
        elif kind == "floor" and t >= self.next_play and not busy:
            self._play(t, lane, kind)
        elif kind == "show" and t >= self.next_play and not busy:
            self._show(t, lane)
        elif kind == "goodnight" and not self.night_said and not (c.pending is not None or c.trial is not None or not c.fast.voice_free(t)):
            c.request("night"); self.night_said = True; self.log.append((t, "night"))   # A117: said over an act of hers still running (a person
                                                                                       # says goodnight while she puts a toy away): with the
                                                                                       # fetch's farther rings a fetch may run into the last
                                                                                       # GOODNIGHT ticks (lane 8's 2,400-tick day: 30 of them)
        elif kind == "tasks" and t_day == self._block_end(t_day) - 1:
            c.request("return")

    def _block_end(self, t_day):
        for s, e, _k, _ in self.blocks:
            if s <= t_day < e:
                return e
        return self.day_ticks

    def _enter(self, kind, t, t_day, lane, world):
        """an episode begins"""
        c, feel = lane.conduct, lane.feel
        self.log.append((t, "episode", kind))
        feel.set_engagement(TASKS_ATTENTION if kind == "tasks" else 1.0)
        feel.set_wind_down(1.0 if kind in ("wind", "goodnight") else 0.0)
        if kind == "away":
            if t - self.last_pain < AWAY_AFTER_PAIN or any(k == "distress" for k, _o in lane._p.events):
                self.log.append((t, "away skipped: its pain or its distress (4.7)"))
                return
            cxy = lane.last.get("child_xy")
            if cxy is not None and float(np.hypot(cxy[0] - DOOR_XY[0], cxy[1] - DOOR_XY[1])) < DOOR_NEAR_M:
                self.log.append((t, "away skipped: the child at the door (C131)"))
                return
            c.request("leave"); c.routine = "leave"
            self.away, self.bids, self.next_call = True, [], t + AWAY_CALL
        elif kind == "tasks":
            c.routine = None
            c.motion.request(_Plain("walk", "sofa"))
        elif kind in ("floor", "motor", "show", "wind"):
            c.routine = None
            self.next_play = t + int(self.rng.integers(*PLAY_GAP))
            if kind == "motor":
                self.sit_due = True                                     # A161: the block's first offer is the pull-to-sit; C254: owed
                self.sit_turns = 0                                      # whatever its posture (a child on its front is turned over first)
                if not self._lying_on_back(lane):
                    self.log.append((t, "motor block: the child not on its back: turned over first for the sit (C254)"))

    def _away_tick(self, t, t_day, lane, world, kind):
        c = lane.conduct
        end = kind != "away" or len([b for b in self.bids if b > t - BIDS_BACK[1]]) >= BIDS_BACK[0]
        if end:
            c.request("return"); c.routine = "return"
            self.away = False
            self.log.append((t, "back", "early: its bids" if kind == "away" else "the block's end"))
            return
        if t >= self.next_call and c.fast.voice_free(t):
            c.request("hall_call"); self.next_call = t + AWAY_CALL

    @staticmethod
    def _posture(lane):
        """the child's posture as her motion's model of it reads it ('back', 'front', 'side', 'sitting'; None when unknown)"""
        ch = getattr(getattr(lane.conduct, "motion", None), "child", None)
        return getattr(ch, "posture", None)

    @staticmethod
    def _lying_on_back(lane):
        """the child on its back as her motion's model of it reads it (the pull-to-sit is from lying on its back, A9); False when unknown"""
        ch = getattr(getattr(lane.conduct, "motion", None), "child", None)
        return getattr(ch, "posture", None) == "back"

    def _sit_or_lesson(self, t, lane):
        """A161 (2026-10-01): the motor block's first offer, with the child on its back, is the pull-to-sit (motor_sit: her 'up! sit up.'
        over both forearms taken, the pull growing to her brief cap and rising only with its own flexion, A9; once per block, twice a
        day); every later offer of the block is her reach-rung lesson (A90). Life days 50 to 56 the child lay on its back the whole day
        and no one offered it the sit: the pull-to-sit and the prop had stayed closed since birth (NOT_AT_BIRTH)"""
        if self.sit_due and self._lying_on_back(lane) and \
                not [v for v in getattr(lane.conduct.motion, "holding", {}).values() if v is not None]:   # C277: with a toy in her hand the
            self.sit_due = False                                            # toy's lesson first (day 89: 'her hands are busy' three times)
            if self.sit_tries == 0:
                self.sit_tries = 1                                          # (C224: the block's first try)
            lane.conduct.request("motor_sit")
            self.log.append((t, "motor_sit"))
            self.next_play = t + int(self.rng.integers(*PLAY_GAP))
            return
        if self.sit_due and self._posture(lane) in ("front", "side") and self.sit_turns < SIT_TURNS_PER_BLOCK:
            # C254 (2026-10-03): A CHILD ON ITS FRONT IS TURNED ONTO ITS BACK FOR THE SIT. Life days 71 and 72: the child lay on its front
            # half the day (its rolling: half_roll 142 a day) and both motor blocks found it so at their entry, so no pull-to-sit was
            # offered at all (days 68 to 72: 3, 3, 1, 0, 0 offers; the sit chain of C235 to C238 unexercised). The block's sit is owed
            # whatever its posture; at the offer a child on its front or side is turned onto its back first (turn_over: her 'up. up. up!'
            # and the turn, A7), SIT_TURNS_PER_BLOCK a block, and the sit stays owed while the block runs (offered when it lies on its back,
            # the lesson meanwhile). A parent sitting a baby up turns it over first
            self.sit_turns += 1
            lane.conduct.request("turn_over")
            self.log.append((t, "motor_sit: the child on its front, turned onto its back first (C254)"))
            self.next_play = t + SIT_RETRY_GAP
            return
        self._lesson(t, lane)                                               # (C254: the sit stays owed while the block runs)

    def _lesson(self, t, lane):
        """her lesson (A90; the module's doc): the rung the child is nearly at on a focus toy she sees, read from her conduct's book"""
        c, p = lane.conduct, lane._p
        seen = {s.id: s for s in p.seen}
        held = [v for v in getattr(c.motion, "holding", {}).values() if v is not None and v not in TP.OPEN_CONTAINERS]
        if held:                                                        # C122: a toy left in her hand is the lesson's toy (day 28: the
            focus = [held[0]]                                           # block stayed in her left hand from the morning on; her peekaboo
        else:                                                           # refused "her hands are busy" 21 times): its rung as any toy's
            focus = [o for o in self.focus if o in seen and not c.left_where_it_lies(o)]   # A117: not a toy she could not get to
            if not focus:                                               # C121: none before her eyes (day 28's afternoon: she knelt at its
                known = set(getattr(lane, "toys", ()) or ())            # head looking at its eyes and asked a call six times instead of
                focus = [o for o in self.focus if o in known and not c.left_where_it_lies(o)]   # a lesson): the toys she knows the place of
            its = set(getattr(p, "child_holds", ()) or ())              # C132 (day 33): a toy in its hand she does not see (C121's known
            free = [o for o in focus if (o not in seen or seen[o].on != "hand") and o not in its]   # toys) was the lesson's toy 9 times
            focus = free or focus                                       # in 4,000 ticks, each hand-over refused: never a toy it holds
            fresh = [o for o in focus if not _worn(c.book, o, self.level.get(o, 0))]          # C141: a toy her smiles have worn out is offered only when no
            if not fresh:                                               # fresh one is at hand; else a fresh toy she knows the place of
                known = set(getattr(lane, "toys", ()) or ())
                fresh = [o for o in sorted(known) if o not in TP.OPEN_CONTAINERS and o not in its and not _worn(c.book, o, self.level.get(o, 0))
                         and not c.left_where_it_lies(o) and (o not in seen or seen[o].on != "hand")]
            if not fresh:                                               # C261: none pays where it stands: a toy with a farther level
                fresh = [o for o in focus if int(self.level.get(o, 0)) < LESSON_LEVELS - 1]   # to go (its next reach lesson steps it there)
            focus = self._at_hand(lane, fresh or focus)                 # C272: the toys at her hand first
        if not focus:
            c.request("call")
        else:
            o = focus[int(self.rng.integers(len(focus)))]
            book = c.book
            lvl = int(self.level.get(o, 0)); top = LESSON_LEVELS - 1
            got = sum(int(v) for k_, v in book.get("got", {}).items() if k_ == o or str(k_).startswith(o + "@"))   # her smiles for its got, every level
            got_lvl = int(book.get("got", {}).get(_lvl_key(o, lvl), 0))     # and at the level it stands at (C261)
            raw = int(getattr(c, "got_raw", {}).get(o, got))                # every got she saw on it, smiled at or not
            handled = sum(int(book.get(k, {}).get(o, 0)) for k in ("lifted", "shook", "hit"))
            rolled = int(book.get("rolled", {}).get("", 0))
            # C261 (2026-10-03): THE REACH LADDER STAYS OPEN AND ITS BAR RISES (conduct._book_key's note: day 75's ladder, every toy at her
            # cap, the levels all back at 0 since days 0 to 43, 17 motor smiles a day). The rung is the toy's while a farther level
            # remains or the top level's got is not yet mastered twice over; once its first gots are mastered (2 x MASTERED_N, as
            # before) its reach lessons alternate with the later rungs' (reach_turn). The level steps farther after MASTERED_N smiled
            # gots at it (mastery, then on: before, after every smiled got), or after any got at a level where her smile is worn out
            # (nothing left to earn there); back a level after NO_PROGRESS ticks with no got
            reach_open = lvl < top or got_lvl < 2 * K.MASTERED_N
            reach_now = got < 2 * K.MASTERED_N or (reach_open and self.reach_turn)
            if reach_now and rolled >= K.MASTERED_N and lane.posture in ("back", "front") and self.roll_turn:
                self.roll_turn = False                                  # A109 (C91): the roll rung, every other lesson once it rolls (the
                self.reach_turn = False                                 # roll smiled at MASTERED_N times): the toy set beside its far
                if hasattr(c, "set_level"):                             # shoulder past its reach, so a roll brings it within reach and
                    c.set_level[o] = lvl                                # the toy is the reward (its own "got", worth 2 on that toy)
                c.request("set_far", o=o)
                self.log.append((t, "lesson", "roll", o, rolled))
            elif reach_now:
                self.roll_turn = True                                   # the reach and grasp rung
                self.reach_turn = False
                seen_ = int(self.got_seen.get(o, raw))
                worn_here = K.MOTOR_WORTH["got"][0] * math.exp(-got_lvl / K.HABIT_TAU) < K.HABIT_FLOOR
                if lvl < top and (got_lvl >= K.MASTERED_N or (worn_here and raw > seen_)):
                    lvl += 1; self.level_t[o] = t                       # mastered here (or nothing left to earn here and it got it): farther
                elif raw <= seen_ and t - int(self.level_t.get(o, t)) > NO_PROGRESS and lvl > 0:   # nothing for a block: a level back
                    lvl -= 1; self.level_t[o] = t
                self.level[o] = lvl; self.level_t.setdefault(o, t); self.got_seen[o] = raw
                c.motion.lesson_dist = LESSON_DIST0 + LESSON_STEP * lvl
                if hasattr(c, "set_level"):
                    c.set_level[o] = lvl
                c.request("set_near", o=o)
                self.log.append((t, "lesson", "reach", o, lvl, round(c.motion.lesson_dist, 2)))
            elif _bucket_at_hand(seen, lane) and o != "bucket" and self.hide_turn and (o not in seen or seen[o].on != "hand") \
                    and ("bucket" not in seen or seen["bucket"].child_sees):   # C158: the hide in the child's view (the bucket before its camera when she sees it; unseen, C138 carries it beside the child)   # C152: the bucket seen or its place known (C138 brings it beside the child; C130's gate, within its reach only, predates the carry) (day 31: four hides 2 m from it)   # A129: the hide game, every other lesson once it grasps the toy at will; C115: never on the toy in its hand (her fetch never takes it, A4: day 26 asked three of four hides on the held book, refused)
                self.hide_turn = False                                  # toy at will (got mastered): the toy let go into the bucket in
                self.reach_turn = True                                  # C261
                c.request("hide", o=o)                                  # its view; its hand into the bucket after is "found" (worth 2)
                self.log.append((t, "lesson", "hide", o, got))
            elif handled < 3 * K.MASTERED_N or o in held:               # the handle rung: the toy into its hand (C125: and the toy in
                                                                        # her own hand, which she cannot ask the child to give her)
                self.reach_turn = True                                  # C261
                self.hide_turn = _bucket_at_hand(seen, lane)                # C152
                c.request("hand_over", o=o)
                self.log.append((t, "lesson", "handle", o, handled))
            else:                                                       # the give rung
                self.reach_turn = True                                  # C261
                self.hide_turn = _bucket_at_hand(seen, lane)                # C152
                if _may_give(seen, o):
                    c.request("ask_give", o=o)
                    self.log.append((t, "lesson", "give", o))
                else:                                                   # C126: a give is asked of a toy in its view and its reach
                    c.request("hand_over", o=o)                         # (4.8; life day 29: 36 gives asked of a toy it could not see,
                    self.log.append((t, "lesson", "handle", o, handled))   # each refused, more than the day's 23 lessons): else the
                                                                        # toy into its hand first, the give asked at a later lesson
        self.next_play = t + int(self.rng.integers(*PLAY_GAP))

    def _at_hand(self, lane, toys):
        """C272: her pick among several toys falls AT_HAND_SHARE of the time on those within AT_HAND_M of where she kneels (taken
        without getting up); day 88: each toy shown was fetched from across the room, a get-up, a walk and two kneelings (about
        200 ticks) a toy, half her ticks in transit"""
        m = getattr(lane.conduct, "motion", None)
        base = getattr(m, "base", None)
        if m is None or not base or len(toys) < 2 or getattr(m, "d", None) is None:
            return toys
        at = base["at"]
        near = [o for o in toys if o in getattr(m, "toys", {}) and
                float(np.hypot(m.d.xpos[m.toys[o]][0] - at[0], m.d.xpos[m.toys[o]][1] - at[1])) <= AT_HAND_M]
        if near and len(near) < len(toys) and float(self.rng.random()) < AT_HAND_SHARE:
            return near
        return toys

    def _play(self, t, lane, kind):
        c, p = lane.conduct, lane._p
        if kind == "floor" and t >= getattr(self, "next_floor_stand", 0) and self._lying_on_back(lane) and \
                not [v for v in getattr(c.motion, "holding", {}).values() if v is not None]:
            # C273: THE STAND IS PART OF HER FLOOR PLAY. Day 88: 8,000 ticks of floor play before the day's first motor block, the
            # stand (C268) not offered once; standing and stepping are learned by doing them, and the owner's word is walking first.
            # Every FLOOR_STAND_GAP, with the child on its back and her hands empty, her play's offer is the stand
            self.next_floor_stand = t + FLOOR_STAND_GAP
            c.request("motor_sit")
            self.log.append((t, "floor play: the stand (C273)"))
            self.next_play = t + int(self.rng.integers(*PLAY_GAP))
            return
        seen = {s.id: s for s in p.seen}
        focus = [o for o in self.focus if o in seen and not c.left_where_it_lies(o)]   # A117: not a toy she could not get to
        held = [v for v in getattr(c.motion, "holding", {}).values() if v is not None and v not in TP.OPEN_CONTAINERS]
        known = set(getattr(lane, "toys", ()) or ())                    # C125: floor play asked a lesson only with a focus toy before
        can = bool(focus or held or [o for o in self.focus if o in known and not c.left_where_it_lies(o)])   # her eyes, so the lesson's
        roll = float(self.rng.random())                                 # own fallbacks (C121 the toys she knows the place of, C122 the toy
        if can and roll < LESSON_SHARE:                                 # in her hand) ran in motor time only; a lesson among her play (A90)
            self._lesson(t, lane)
            return
        roll = (roll - LESSON_SHARE) / (1.0 - LESSON_SHARE) if can else roll
        if kind == "floor" and roll < ASK_SHARE and focus:                # C283: her asks first
            o = focus[int(self.rng.integers(len(focus)))]
            ask = ["ask_where", "ask_where", "ask_what", "ask_give"][int(self.rng.integers(4))]
            can = [x for x in focus if (_may_give(seen, x) if ask == "ask_give" else seen[x].child_sees)]
            if o not in can and can:                                    # C126: an ask is of a toy in the child's view (4.8), a give of
                o = can[int(self.rng.integers(len(can)))]               # one in its reach too: the toy of her ask chosen among those
            if can:
                c.request(ask, o=o)
            else:                                                       # none: the toy shown (into its view), the ask another time
                c.request("show", o=o)
        elif focus and roll < 0.85:
            free = [o for o in focus if o not in seen or seen[o].on != "hand"] or focus   # A117: never the toy in its hand (her fetch never takes a
            free = self._at_hand(lane, free)                            # C272: the toys at her hand first
            o = free[int(self.rng.integers(len(free)))]                 # toy from it, A4: life day 13's 4 shows refused for the car)
            c.request("show", o=o)
        elif kind == "floor" and roll < 0.95 and held:                  # C125: her peekaboo needs both her hands (refused "her hands
            self._lesson(t, lane)                                       # are busy" 26 times on day 28, twice on day 29): with a toy in
            return                                                      # her hand the toy's lesson instead (C122)
        elif kind == "floor" and roll < 0.95:
            c.request("peekaboo_hide"); c.routine = "peekaboo"
        else:
            c.request("call")
        self.next_play = t + int(self.rng.integers(*PLAY_GAP))

    def _show(self, t, lane):
        c, p = lane.conduct, lane._p
        f = c.fast
        for w in TP.GROWTH_WORDS:
            if w in f.vocab or w in f.new_words:
                continue
            ok, _why = TP.showable(w, c.world)
            if not ok:
                continue
            got, _why = TP.show_now(w, p, f.recent)
            if got:
                o = got[0].id if got[0] is not None else None
                c.request("new_word", word=w, o=o)
                self.log.append((t, "new word asked", w))
                break
        else:
            att = p.attended()
            if att:
                c.request("ask_what", o=att[0].id)
            else:
                # C150 (2026-09-30): a show block with no new word to show and nothing the child attends is play with a lesson (her
                # focus toys shown, the lesson's rung, a call), not a kneel in silence. Life day 40's last show block: 1,888 ticks
                # kneeling 1.5 m from a child on its back, 112 looks at its eyes, no act; every growth word she could show was known
                self._play(t, lane, "show")
                return
        self.next_play = t + int(self.rng.integers(*PLAY_GAP))

    # ------------------------------------------------------------------ the save
    def state(self):
        return dict(rng=self.rng.bit_generator.state, day_ticks=self.day_ticks, day=self.day,
                    blocks=[[s, e, k, dict(i)] for s, e, k, i in self.blocks], kind=self.kind, away=self.away,
                    greeted=self.greeted, called=self.called,
                    focus=list(self.focus), next_play=self.next_play, next_call=self.next_call, last_pain=self.last_pain,
                    bids=list(self.bids), night_said=self.night_said, log=[list(x) for x in self.log[-200:]],
                    level=dict(self.level), level_t=dict(self.level_t), got_seen=dict(self.got_seen), roll_turn=bool(self.roll_turn),
                    sit_due=bool(self.sit_due),                                       # C207: the motor block's owed pull-to-sit survives a resume
                    block_i=int(self.block_i),                                        # C220
                    sit_tries=int(self.sit_tries),                                    # C224
                    sit_turns=int(getattr(self, "sit_turns", 0)),                     # C254
                    hide_turn=bool(self.hide_turn),
                    reach_turn=bool(self.reach_turn))                                 # C261

    def load_state(self, s):
        self.rng.bit_generator.state = s["rng"]
        self.day_ticks, self.day = int(s["day_ticks"]), int(s["day"])
        self.blocks = [[int(a), int(b), str(k), dict(i)] for a, b, k, i in s["blocks"]]
        self.kind, self.away = s["kind"], bool(s["away"])
        self.greeted, self.called = s["greeted"], bool(s["called"])
        self.focus, self.next_play, self.next_call = list(s["focus"]), int(s["next_play"]), int(s["next_call"])
        self.last_pain, self.bids, self.night_said = int(s["last_pain"]), [int(b) for b in s["bids"]], bool(s["night_said"])
        self.log = [tuple(x) for x in s["log"]]
        self.level = {k: int(v) for k, v in s.get("level", {}).items()}
        self.level_t = {k: int(v) for k, v in s.get("level_t", {}).items()}
        self.got_seen = {k: int(v) for k, v in s.get("got_seen", {}).items()}
        self.roll_turn = bool(s.get("roll_turn", True))
        self.reach_turn = bool(s.get("reach_turn", True))                           # (C261; older saves: a reach lesson first)
        self.sit_due = bool(s.get("sit_due", False))                                 # (C207; older saves: none owed)
        self.sit_tries = int(s.get("sit_tries", 0))                                  # (C224; older saves: none tried)
        self.sit_turns = int(s.get("sit_turns", 0))                                  # (C254; older saves: none turned)
        self.block_i = int(s.get("block_i", -1))                                     # (C220; older saves: the block re-read at the next tick,
                                                                                     #  a second motor block owing its sit then)
        self.hide_turn = bool(s.get("hide_turn", False))               # (C125: lost at each resume before; older saves: none)


class _Plain:
    """an act of her own body with no line (her walk to the sofa for her tasks)"""
    def __init__(self, kind, target=None, during=None, thing=None):
        self.kind, self.target, self.during, self.thing = kind, target, during, thing
