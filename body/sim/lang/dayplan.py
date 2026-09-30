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
import numpy as np

from . import consts as K
from . import templates as TP

PLAN_STREAM = 7                        # her day plan's random stream of the body's seed (ours; world 1, tract 2, lines 3, reading 4,
                                       # imperfection 5, trials 6)
DAY_TICKS = 24000                      # a life day (4.7)
WAKE = 300                             # the wake episode (4.7)
WIND = 1000                            # the winding down (4.7)
GOODNIGHT = 300                        # goodnight (4.7)
BLOCKS = (("floor", 3, 4000, 5000), ("motor", 2, 1000, 1500), ("show", 1, 1500, 1500), ("away", 1, 400, 600),
          ("tasks", 1, 600, 600))      # 4.7's table: kind, how many, shortest, longest. TRAINING MODE (2026-09-29, the owner's word: fix
                                       # fast; her pace is the lead's): away 2-4 x 400-1,200 and her own tasks 3,000 cut to one short block
                                       # each, so the play blocks (drawn, then scaled to the day) carry about a quarter more of the day
PLAY_GAP = (150, 300)                  # ticks between her floor play's offers (ours)
LESSON_DIST0 = 0.10                    # the reach rung's first distance out from its near hand, m (4.10's ladder, level 1; A90)
LESSON_STEP = 0.05                     # farther each mastered level (teacher_of_reality.md 2c)
LESSON_LEVELS = 7                      # up to 0.40 m (about the arm's reach)
NO_PROGRESS = 1000                     # a level with no new "got" for this long steps back one (about a block)
LESSON_SHARE = 0.4                     # floor play's offers that are lessons (the rest: shows, peekaboo, asks, the call)
GREET_BY = 200                         # the greeting at the latest by this tick of the wake (ours)
CALL_AFTER_GREET = 60                  # the wake's call this long after the greeting (ours)
AWAY_CALL = 600                        # her calls from the hall (4.7: about every 600 ticks)
AWAY_AFTER_PAIN = 100                  # nor within 100 ticks of its pain (4.7)
BIDS_BACK = (3, 40)                    # back early after 3 of its vocal turns in 40 ticks (4.7)
TASKS_ATTENTION = 0.4                  # her attention at her own tasks (parent_feel: .4)
BIRTH_TOYS = ("ball", "block", "duck", "cup", "car", "bear", "drum")   # her birth words' toys in the room (lexicon; no bottle: A88)


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
        self.roll_turn = True              # A109: the next reach-rung lesson is the roll rung (once it rolls); they alternate
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
        draw = self.rng.choice(rest, size=min(k - (1 if newest else 0), len(rest)), replace=False).tolist()
        self.focus = sorted(o for o in draw + ([newest] if newest else []) if o not in TP.OPEN_CONTAINERS)   # C119: the bucket stays where it stands (her grasp and put miss it by 17 to 50 cm): the hide game's container, never a toy of the day
        self.greeted, self.called, self.night_said = None, False, False
        self.next_play = self._scale(WAKE)
        self.log.append((day, "laid out", [(s, e, k) for s, e, k, _ in self.blocks], self.focus))

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
        if any(k == "pain" for k, _o in p.events):
            self.last_pain = t
        if p.child_sounding and self.away:
            self.bids = [b for b in self.bids if b > t - BIDS_BACK[1]] + [t]
        if self.away:
            self._away_tick(t, t_day, lane, world, kind)
            return
        busy = (c.pending is not None or c.trial is not None or not c.fast.voice_free(t)
                or any(a[5] not in ("done", "refused", "cancelled") for a in c.acts_open))
        if kind == "wake":
            if self.greeted is None and (not pm.phases or t_day >= self._scale(GREET_BY)) and not busy:
                c.request("greet"); c.routine = "greet"; self.greeted = t; self.log.append((t, "greet"))
            elif self.greeted is not None and not self.called and t >= self.greeted + CALL_AFTER_GREET and not busy:
                c.request("call"); self.called = True; c.routine = None
        elif kind == "motor" and t >= self.next_play and not busy:
            self._lesson(t, lane)
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
            c.request("leave"); c.routine = "leave"
            self.away, self.bids, self.next_call = True, [], t + AWAY_CALL
        elif kind == "tasks":
            c.routine = None
            c.motion.request(_Plain("walk", "sofa"))
        elif kind in ("floor", "motor", "show", "wind"):
            c.routine = None
            self.next_play = t + int(self.rng.integers(*PLAY_GAP))

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
            free = [o for o in focus if o not in seen or seen[o].on != "hand"]   # A109: a toy in its hand is not the one to set out for it
            focus = free or focus                                       # (her fetch never takes a toy from it, A4)
        if not focus:
            c.request("call")
        else:
            o = focus[int(self.rng.integers(len(focus)))]
            book = c.book
            got = int(book.get("got", {}).get(o, 0))
            handled = sum(int(book.get(k, {}).get(o, 0)) for k in ("lifted", "shook", "hit"))
            rolled = int(book.get("rolled", {}).get("", 0))
            if got < 2 * K.MASTERED_N and rolled >= K.MASTERED_N and lane.posture in ("back", "front") and self.roll_turn:
                self.roll_turn = False                                  # A109 (C91): the roll rung, every other lesson once it rolls (the
                c.request("set_far", o=o)                               # roll smiled at MASTERED_N times): the toy set beside its far
                self.log.append((t, "lesson", "roll", o, rolled))       # shoulder past its reach, so a roll brings it within reach and
            elif got < 2 * K.MASTERED_N:                                # the toy is the reward (its own "got", worth 2 on that toy)
                self.roll_turn = True                                   # the reach and grasp rung
                lvl = int(self.level.get(o, 0))
                if got > int(self.got_seen.get(o, 0)):                  # it got the toy since: a level farther
                    lvl = min(LESSON_LEVELS - 1, lvl + 1); self.level_t[o] = t
                elif t - int(self.level_t.get(o, t)) > NO_PROGRESS and lvl > 0:   # nothing for a block: a level back
                    lvl -= 1; self.level_t[o] = t
                self.level[o] = lvl; self.level_t.setdefault(o, t); self.got_seen[o] = got
                c.motion.lesson_dist = LESSON_DIST0 + LESSON_STEP * lvl
                c.request("set_near", o=o)
                self.log.append((t, "lesson", "reach", o, lvl, round(c.motion.lesson_dist, 2)))
            elif "bucket" in seen and seen["bucket"].child_can_reach and o != "bucket" and self.hide_turn and (o not in seen or seen[o].on != "hand"):   # C130: a hide within its reach only (day 31: four hides 2 m from it)   # A129: the hide game, every other lesson once it grasps the toy at will; C115: never on the toy in its hand (her fetch never takes it, A4: day 26 asked three of four hides on the held book, refused)
                self.hide_turn = False                                  # toy at will (got mastered): the toy let go into the bucket in
                c.request("hide", o=o)                                  # its view; its hand into the bucket after is "found" (worth 2)
                self.log.append((t, "lesson", "hide", o, got))
            elif handled < 3 * K.MASTERED_N or o in held:               # the handle rung: the toy into its hand (C125: and the toy in
                                                                        # her own hand, which she cannot ask the child to give her)
                self.hide_turn = "bucket" in seen and seen["bucket"].child_can_reach
                c.request("hand_over", o=o)
                self.log.append((t, "lesson", "handle", o, handled))
            else:                                                       # the give rung
                self.hide_turn = "bucket" in seen and seen["bucket"].child_can_reach
                if _may_give(seen, o):
                    c.request("ask_give", o=o)
                    self.log.append((t, "lesson", "give", o))
                else:                                                   # C126: a give is asked of a toy in its view and its reach
                    c.request("hand_over", o=o)                         # (4.8; life day 29: 36 gives asked of a toy it could not see,
                    self.log.append((t, "lesson", "handle", o, handled))   # each refused, more than the day's 23 lessons): else the
                                                                        # toy into its hand first, the give asked at a later lesson
        self.next_play = t + int(self.rng.integers(*PLAY_GAP))

    def _play(self, t, lane, kind):
        c, p = lane.conduct, lane._p
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
        if focus and roll < 0.5:
            free = [o for o in focus if o not in seen or seen[o].on != "hand"] or focus   # A117: never the toy in its hand (her fetch never takes a
            o = free[int(self.rng.integers(len(free)))]                 # toy from it, A4: life day 13's 4 shows refused for the car)
            c.request("show", o=o)
        elif kind == "floor" and roll < 0.65 and held:                  # C125: her peekaboo needs both her hands (refused "her hands
            self._lesson(t, lane)                                       # are busy" 26 times on day 28, twice on day 29): with a toy in
            return                                                      # her hand the toy's lesson instead (C122)
        elif kind == "floor" and roll < 0.65:
            c.request("peekaboo_hide"); c.routine = "peekaboo"
        elif kind == "floor" and roll < 0.9 and focus:
            o = focus[int(self.rng.integers(len(focus)))]
            ask = ["ask_where", "ask_give", "ask_what"][int(self.rng.integers(3))]
            can = [x for x in focus if (_may_give(seen, x) if ask == "ask_give" else seen[x].child_sees)]
            if o not in can and can:                                    # C126: an ask is of a toy in the child's view (4.8), a give of
                o = can[int(self.rng.integers(len(can)))]               # one in its reach too: the toy of her ask chosen among those
            if can:
                c.request(ask, o=o)
            else:                                                       # none: the toy shown (into its view), the ask another time
                c.request("show", o=o)
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
        self.next_play = t + int(self.rng.integers(*PLAY_GAP))

    # ------------------------------------------------------------------ the save
    def state(self):
        return dict(rng=self.rng.bit_generator.state, day_ticks=self.day_ticks, day=self.day,
                    blocks=[[s, e, k, dict(i)] for s, e, k, i in self.blocks], kind=self.kind, away=self.away,
                    greeted=self.greeted, called=self.called,
                    focus=list(self.focus), next_play=self.next_play, next_call=self.next_call, last_pain=self.last_pain,
                    bids=list(self.bids), night_said=self.night_said, log=[list(x) for x in self.log[-200:]],
                    level=dict(self.level), level_t=dict(self.level_t), got_seen=dict(self.got_seen), roll_turn=bool(self.roll_turn),
                    hide_turn=bool(self.hide_turn))

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
        self.hide_turn = bool(s.get("hide_turn", False))               # (C125: lost at each resume before; older saves: none)


class _Plain:
    """an act of her own body with no line (her walk to the sofa for her tasks)"""
    def __init__(self, kind, target=None, during=None, thing=None):
        self.kind, self.target, self.during, self.thing = kind, target, during, thing
