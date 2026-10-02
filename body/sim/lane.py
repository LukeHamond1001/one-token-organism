"""THE PARENT'S LANE (S5a; docs/SIM_DESIGN.md 4.3-4.10, A1-A3, A29, A40, A49, A51): her conduct (body/sim/lang/conduct.py, P3),
her voice (body/sim/voice, P1), her feelings and face (body/sim/parent_feel.py) and her body's acts (body/sim/parent_motion.py,
W2) joined to the G1's world tick by tick. Nothing here is the child's: what reaches the body is the frame's words symbol, the face
pair and her voice's sound at its ears, as the world gives them.

The world (body/sim/world.G1World) holds one as world.lane (ParentLane(world) sets it) and calls:
  tick(world)          in apply, by day, after the physics and the child's tract and before the ears: one tick of hers. Her
                       Percept of the tick from world truth (below), her conduct's tick on it (the child's tract samples and
                       token of the tick, her distance, her playback's stop), her new line begun and its first samples played,
                       her feelings and her face set on her body, the born reading of her face (A1, A2, A49), and the words
                       channel's symbol. Returns her voice as the ears' source {"parent": (Pa at 1 m, her mouth)}.
  after_apply(world)   at the tick's end, day or night: by night her day's lane is quiet (the words at rest, the face held as
                       the reading holds it, her line ended at dusk).
  word_now, face_seen  the next frame's words symbol (the anatomy's born table: rest 0, end 1, space 2, letters 3-28, the 50
                       words 29-78) and face pair [the reading, its change this tick]
  dusk(world), dawn(world), state(), load_state(s)

HER PERCEPT (percept.py's fields, filled only from what a person in her place could see or hear; A40: never the child's inside,
never its fovea's window):
  present         she is awake (at birth she never leaves the room; the day plan's absences are P4's)
  child_in_view   a ray from her face to the G1's torso meets the G1 first
  seen_by_child   her face lies within the child's camera field about its head's line (CAMERA_FIELD_DEG; a person knows when a
                  baby cannot see her)
  child_target    Reader.look over the toys she sees and her face ("mama"), from the G1's head camera: the midpoint of its two
                  eye cameras and their axes (forward, left, up), the G1 having no neck (A22)
  child_holds     the toys touching a link of its hands (its wrist's yaw link, which carries the palm, and the Dex3's fingers)
  child_reaches   Reader.reaches from its two grasp points (her planner's, parent_motion.Child.grasp), after look (A40's order)
  seen            each toy a ray from her face meets first: its word and colour (the room's inventory at birth,
                  templates.ROOM_AT_BIRTH), what it rests on (her hand "mama", the child's "hand", else the fixture its
                  contacts name), child_sees from look's field, child_can_reach (in its hand, or within its arm's reach of a
                  shoulder: the chain's length in the model, shoulder to grasp point), near (Reader.near)
  fixtures        the words of the room's fixtures with shapes in the model (mat, sofa, window, table, shelf, floor)
  events          fell (a toy falling faster than FALL_MPS, not in a hand; once a drop), got (a toy come into its hand with
                  no hand-over of hers in the last HANDOVER_TICKS), lost_toy (a toy gone from its hand), gave (a toy come
                  into her hand from its within GAVE_TICKS), rolled (its lying posture turned between back and front), sat
                  (it came to sit), pain (what a person perceives of its pain: its born cry sounding, which she hears, or
                  a blow to its body past the base's pain force, F_pain, while she sees it; never its joints' own pain flags,
                  which are its inside), distress (lying
                  face down DISTRESS_TICKS running, A13's first sign). Not made
                  here: hit_her and reflex_hit (A25c: her body passes no contact to it), and the arm and hand movements her
                  copying reads (arm_raise, wave, shake, open_hand: A52's copying waits for their readers)
  child_sounding  its tract sounded this tick (the transcriber's own reading replaces it)
Every threshold here is ours unless a source is named; each is disclosed.

HER FACE (A1, A2, A49; A158): the face test (body/sim/eyes.face_test: A1's four conditions for either eye) passing this tick and the
last is "seen" (her feelings' input: a smile waits for a look); the born reading takes 2 x (smile - frown) of the face she shows
(parent_kin.face_reading) while she is with the child (present: awake and in the room), looked at or not (A158: her approval reaches it
as a tone and a touch would), holds it READING_HOLD ticks once she is away, then reads 0. Until the child's own mouth-corner
reader works this is the world's value, a disclosed scaffold (A49).

THE WORDS CHANNEL (A29): her playback hands each word to the channel as its sound starts (voice/playback.py); a symbol goes out
only while she is audible at the child's nearer ear (lexicon.audible, her plain speech's 62 dB at 1 m and the ears' 1/r paths),
and only while the scaffold is on (her conduct's scaffold).

Nothing here draws a random number but her own streams (the conduct's, the reader's, her feelings' blinks). state() and
load_state() carry all of it, the clip of a line under way included, so a replay continues exactly.
"""
import math

import mujoco
import numpy as np

from body.sim import anatomy as AN
from body.sim import ears as EA
from body.sim import eyes as EY
from body.sim import extras as X                    # A129: the bucket's sizes
from body.sim import parent_feel as PF
from body.sim import parent_kin as kin
from body.sim.lang import conduct as C
from body.sim.lang import dayplan as DP
from body.sim.lang import consts as K
from body.sim.lang import lexicon as LX
from body.sim.lang import percept as PC
from body.sim.lang import templates as TP
from body.sim.lang.transcriber import Transcriber
from body.sim.voice import synth as V
from body.sim.voice.playback import TICK, Utterance

SOURCE = "parent"                      # her voice's name among the ears' sources
ROOM_EDGE_X = 2.6                      # the room's wall with the door to the hall (make_g1room: ROOM_X)
FALL_MPS = 0.5                         # a toy falling faster than this, not in a hand: "fell" (once a drop; ours)
REST_MPS = 0.05                        # a toy slower than this has come to rest: its next drop is a new one (ours)
HANDOVER_TICKS = 40                    # a toy she let go of within 40 ticks is her hand-over, never its own "got" (A2's 40 ticks)
HIDDEN_OUT_TICKS = 10                  # A129: a hidden toy out of the bucket with no hand on it for 10 ticks (1.5 s) is hidden no more (a tip, a
                                       # fall; the lane's reading of a hold takes GOT_HOLD ticks to settle: ours)
GAVE_TICKS = 3                         # a toy come into her hand from its within 3 ticks: "gave" (ours)
DISTRESS_TICKS = 100                   # face down this many ticks running: distress (A13's "face down over 100 ticks")
TUMMY_TIME_TICKS = 1200                # C216: off its back (its front or its side, C217) this many ticks running (3 min), still or not: tummy time is over and she turns it
                                       # over (tummy time in short sessions for a young infant, the AAP's guidance; ours). Life day 60: the
                                       # child, rolling onto its front since day 58, lay prone 87% of a motor block, moving, so no
                                       # distress (C137's stillness) and no turn: it saw the mat, no face and no toy named, and the
                                       # pull-to-sit is from its back
CRY_DOWN_TICKS = 10                    # or face down and crying this many ticks running (A90: a parent hears a baby crying on its tummy
                                       # and turns it at once; the second plumbing day waited 100 ticks while its wrists hurt)
# HER EYES ON ITS ACTS (A89, the teacher's build 2a; percept.EVENT_KINDS): every threshold ours, disclosed
TICK_S = TICK / V.SR                   # a tick, 0.15 s
LIFT_M = 0.05                          # a held toy 5 cm above where it lay: "lifted"
SHAKE_MPS = 0.3                        # a held toy faster than this on two ticks running: "shook"
SHAKE_GAP = 20                         # a shake counted once in 20 ticks (3 s) a toy
HIT_GAP = 10                           # a hit counted once in 10 ticks a toy
THROW_MPS = 1.0                        # a toy leaving its hand faster than this: "threw" (stage 2's frown)
HAND_REST_MPS = 0.05                   # a hand slower than this has come to rest
HAND_MOVE_MPS = 0.15                   # a hand faster than this is moving
MOVED_TICKS = 5                        # "got" needs that hand moved, or a reach toward the toy, within the last 5 ticks (its own reach and hold)
GOT_HOLD = 3                           # and the toy kept in that hand's touch this many ticks running: a hold, not a graze (the plumbing day
FOUND_WINDOW, FOUND_TOUCHES = 6, 3     # C116: a find: its hand on the hidden toy on 3 of the last 6 ticks (a toy in the bucket rattles; ours)
BUCKET_NEAR_M = 0.30                   # C174: a hand within this of the bucket's centre is "at the bucket" for the record's bucket_hand ruler (ours)
                                       # of 2026-09-26 counted 15 "got" in 1,500 ticks of babble against the toys beside its hands)
REACH_BOOK_M = 0.6                     # a movement's end within this of a toy is a reach at it (about the arm's length)
BOOK_LAST = 10                         # her notebook keeps the child's last 10 reaches a toy
NEARER_M = 0.01                        # a reach ends nearer when it beats the best of those by 1 cm
HEAD_UP_M = 0.08                       # on its front with the head 8 cm above the pelvis, held HEAD_UP_TICKS: "head_up", once in HEAD_UP_GAP
HEAD_UP_TICKS, HEAD_UP_GAP = 5, 100    # (the plumbing day of 2026-09-26: a thrashing body on its front crossed 8 cm 14 times in 100 ticks)
ROLL_GAP = 40                          # a half roll counted once in 40 ticks (rocking on its side is one act, not many)
STILL_M, STILL_TICKS = 0.10, 40      # C137: a face-down child is "still" when its pelvis has not moved 10 cm in 40 ticks (6 s); only then is
                                       # its distress hers to answer with a turn: a crawling child is going somewhere (ours)
SIT_HOLD, SIT_GAP = 5, 100             # C134: a sit is the sitting held 5 ticks (0.75 s), counted once in 100 ticks (day 34: "sat" judged 22
                                       # times in 1,200 ticks, three of them 3 ticks apart, the trunk bobbing up and down on its back; ours)
CRAWL_M = 0.20                         # A125: on its front, its pelvis carried this far along the floor from where its prone spell began
CRAWL_GAP = 60                         # (or from the last crawl counted): "crawled", once in this many ticks (ours; a body length is 1.3 m)
PEEKABOO_ACT = (10, 5)                 # an act begun within 10 ticks of her reveal by a hand that rested the 5 ticks before (A2)
READING_HOLD = PF.FEEL["reading_hold"]  # the born reading holds its last value 30 ticks out of view (A2)
FIXTURE_WORDS = ("mat", "sofa", "window", "table", "shelf", "floor")   # her fixture words that name shapes in the room
SPEECH_DB = 20.0 * math.log10(V.SPEECH_PA / 20e-6)                   # her plain speech at 1 m (62 dB SPL)


def _table_maps():
    """her lexicon's ids to the anatomy's born table and back, read from the table itself (born_table(LX.BIRTH_WORDS))"""
    tok = AN.born_table(LX.BIRTH_WORDS)
    v = tok.get_vocab()
    to_an = np.zeros(LX.N_TABLE, np.int64)
    for i, s in enumerate(LX.TABLE):
        if i < LX.N_BIRTH:
            key = AN.word_key(s)
        elif s == LX.SPACE:
            key = " "
        else:
            key = s
        to_an[i] = v[key]
    assert len(set(to_an.tolist())) == LX.N_TABLE, "the two tables do not map one to one"
    to_lx = np.zeros(LX.N_TABLE, np.int64)
    to_lx[to_an] = np.arange(LX.N_TABLE)
    return to_an, to_lx


LX_TO_AN, AN_TO_LX = _table_maps()
AN_REST = int(LX_TO_AN[LX.ID[LX.REST]])


def _pl(x):
    """a plain, picklable copy (numpy to lists and numbers)"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: _pl(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_pl(v) for v in x]
    return x


class ParentLane:
    """her lane over a G1World (module doc). voice: body/sim/voice/synth.VoiceCache (or anything with its clip()); None: her
    lines are timed at 3 ticks a word and make no sound and no symbol (the conduct's instrument timing). ear: her ear
    (body/sim/parent_ear.ParentEar) or None (the child's turns are heard with no word). level2: her registered nouns (A60b)."""

    def __init__(self, world, seed=1, voice=None, ear=None, level2=(), stage=1, imperfect=True, plan=True, day_ticks=DP.DAY_TICKS):
        m = world.m
        self.voice = voice
        self.conduct = C.Conduct(seed=seed, voice=voice, transcriber=Transcriber(ear), motion=world.parent, stage=stage,
                                 imperfect=imperfect, level2=level2, world=self._inventory(m))
        world.parent.bind_conduct(self.conduct)
        self.feel = PF.Feelings(seed)
        self.words = LX.Words()
        self.utt = None                                   # her line under way (playback.Utterance)
        self.voice_done = None                            # the tick a cut line's sound stopped, for the conduct's next tick
        self.word_now = AN_REST
        self.face_seen = np.zeros(2)
        self.reading = 0.0                                # the born reading (A2), its last value
        self.reading_t = -10 ** 9                         # the tick it last updated in view
        self.test_prev = False                            # the face test passed last tick
        self.fp = dict(kin.FACE_NEUTRAL)                  # the face she shows
        self.toy_z = {}                                   # toy -> its last height (the fall)
        self.falling = set()                              # toys in a drop already reported
        self.child_had = ()                               # the toys in its hands last tick
        self.child_held_at = {}                           # toy -> the last tick it was in its hands
        self.her_had = {}                                 # her hands' toys last tick
        self.released = {}                                # toy -> the tick her hand let it go
        self.hidden = {}                                  # A129: toy -> the tick it lay hidden in the bucket from her hand
        self.hidden_out = {}                              # A129: a hidden toy out of the bucket -> the first tick it was seen out
        self.hidden_seen = {}                             # C158: toy -> whether the bucket stood before the child's camera as it was hidden
        self.found_log = []                               # C158: (tick, toy, seen) of the finds, the last 50 (the record's "found")
        self.posture = None                               # its lying posture, back or front, last seen stable
        self.face_down = 0                                # ticks lying face down running
        self.off_back = 0                                 # C217: ticks running off its back (front or side), tummy time's clock
        self.tummy_over = False                           # C217: tummy time's end told this spell off its back
        self.cry_down = 0                                 # ticks lying face down and crying running
        self.distressed = False                           # distress said of this face-down spell
        self.found_ticks = {}                             # C116: the ticks its hand touched each hidden toy of late
        # her eyes on its acts (A89, 2a) and her notebook of its reaches
        self.first = True                                 # the first tick: what is already in its hands is nothing it got
        self.hand_prev = {}                               # side -> its grasp point last tick
        self.hand_speed = {"left": [], "right": []}       # side -> its last 6 ticks' speeds (m/s)
        self.hand_moved_t = {"left": -10 ** 9, "right": -10 ** 9}   # side -> the last tick it moved
        self.toy_prev = {}                                # toy -> its place last tick
        self.toy_speed = {}                               # toy -> its speed this tick
        self.toy_rest_z = {}                              # toy -> its height when it last lay still, in no hand
        self.lifted = set()                               # toys lifted in this hold
        self.last_shook = {}                              # toy -> the last tick a shake was counted
        self.last_hit = {}                                # toy -> the last tick a hit was counted
        self.reach_t = {}                                 # toy -> the last tick a hand reached toward it
        self.touch_run = {}                               # toy -> ticks running in a hand's touch (a hold: GOT_HOLD)
        self.got_arm = {}                                 # toy -> whether the hand had moved, or reached, when the touch began
        self.book = {}                                    # her notebook: toy -> the hand's distance at the end of its last BOOK_LAST reaches
        self.side_from = None                             # the lying posture it turned onto its side from (a half roll, once)
        self.last_half_roll = -10 ** 9                    # the last tick a half roll was counted
        self.sit_run, self.last_sat = 0, -10 ** 9         # C134: ticks sitting so far, the last tick a sit was counted
        self.still_from, self.still_t = None, -10 ** 9    # C137: where and when its prone stillness began
        self.crawl_from = None                            # A125: its pelvis on the floor plan when its prone spell began, or its last crawl
        self.last_crawl = -10 ** 9                        # the last tick a crawl was counted
        self.head_up = False                              # its head up on its front, this spell
        self.head_up_run = 0                              # ticks running with its head up on its front
        self.last_head_up = -10 ** 9                      # the last tick a head-up was counted
        self.reveal_t = -10 ** 9                          # the tick of her last peekaboo reveal
        self.peekaboo_done = -10 ** 9                     # the reveal an act has answered
        self.last = {}                                    # instruments: the tick's percept summary and her output
        self.n_lines = 0
        self.plan = DP.DayPlan(seed, day_ticks) if plan else None      # her day (L3, P4)
        self.day, self.day_start = 0, int(world.tick)
        self._geom(world)
        world.lane = self

    # ------------------------------------------------------------------ the model's names, once
    @staticmethod
    def _inventory(m):
        names = {m.body(b).name[4:] for b in range(m.nbody) if m.body(b).name.startswith("toy_")}
        objects = {k: v for k, v in TP.ROOM_AT_BIRTH["objects"].items() if k in names}
        gnames = [m.geom(g).name or "" for g in range(m.ngeom)]
        fixtures = [f for f in FIXTURE_WORDS if any(n == f or n.startswith(f + "_") for n in gnames)]
        return dict(objects=objects, fixtures=fixtures, events=list(PC.EVENT_KINDS), face=["any"])

    def _geom(self, world):
        m = world.m
        self.toys = sorted(m.body(b).name[4:] for b in range(m.nbody) if m.body(b).name.startswith("toy_"))
        self.toy_body = {t: m.body("toy_" + t).id for t in self.toys}
        root = m.body_rootid
        self.toy_of_root = {int(root[b]): t for t, b in self.toy_body.items()}
        self.parent_bodies = frozenset(b for b in range(m.nbody) if m.body(b).name.startswith("parent_"))
        self.g1_root = int(root[m.body("pelvis").id])
        hand = {}
        for side in ("left", "right"):
            hand[side] = frozenset(b for b in range(m.nbody) if m.body(b).name.startswith(f"{side}_hand_")
                                   or m.body(b).name == f"{side}_wrist_yaw_link")
        self.hand = hand
        self.hand_any = hand["left"] | hand["right"]
        fx = {}
        for g in range(m.ngeom):
            n = m.geom(g).name or ""
            for f in FIXTURE_WORDS:
                if n == f or n.startswith(f + "_"):
                    fx[g] = f
        self.fixture_geom = fx
        self.fixtures = frozenset(fx.values())
        self.head_b = m.body("parent_head").id
        self.cam = {sd: m.camera(f"eye_{sd}").id for sd in "LR"}
        self.shoulder = {s: m.body(f"{s}_shoulder_pitch_link").id for s in ("left", "right")}
        chain = ("shoulder_roll_link", "shoulder_yaw_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
        self.arm_reach = {s: float(sum(np.linalg.norm(m.body_pos[m.body(f"{s}_{c}").id]) for c in chain)
                                   + np.linalg.norm([0.13, 0.06, 0.0])) for s in ("left", "right")}

    # ------------------------------------------------------------------ what she sees
    def _visible(self, m, d, face, tt, at):
        """tt in her line of sight from her face: the ray to its body's origin meets it first. An open container's origin is its floor
        plate (extras.add_bucket), so a toy lying in it took the ray and the bucket went unseen at the moment it held a toy (C129: day
        30's hides, her look at the bucket done 30 times and the telling refused "unseen: 'bucket'"): for a container the ray goes to
        the middle of its opening, and she sees it when the ray meets the container or nothing at all (a clear line into its mouth)"""
        b = self.toy_body[tt]
        root = int(m.body_rootid[b])
        if tt in TP.OPEN_CONTAINERS:
            Rt = d.xmat[b].reshape(3, 3)
            rim = np.asarray(at, float) + Rt @ np.array([0.0, 0.0, X.BUCKET_WALL + X.BUCKET_H])
            hit = self._ray_first(m, d, face, rim)
            if hit is None or hit == root:
                return True
            rel = Rt.T @ (d.xpos[hit] - np.asarray(at, float))         # C130: a toy heaped in the bucket above its rim (day 31's hide
            return bool(max(abs(rel[0]), abs(rel[1])) <= X.BUCKET_IN + 0.03 and rel[2] < X.BUCKET_H + 0.15)   # of the box on the book
                                                                        # hidden before) took the ray: a thing lying in the bucket is the
                                                                        # bucket seen, with what is in it
        return self._ray_first(m, d, face, at) == root

    def _ray_first(self, m, d, a, b):
        """the root body a ray from a to b meets first past her own body, or None when it meets nothing before b"""
        v = np.asarray(b, float) - np.asarray(a, float)
        dist = float(np.linalg.norm(v))
        if dist < 1e-9:
            return None
        u = v / dist
        p = np.asarray(a, float).copy()
        gid = np.array([-1], dtype=np.int32)
        gone = 0.0
        for _ in range(6):
            h = mujoco.mj_ray(m, d, p, u, EY.EYE_GROUPS, 1, -1, gid)
            if h < 0 or gone + h > dist + 0.02:
                return None
            body = int(m.geom_bodyid[int(gid[0])])
            if body in self.parent_bodies:                   # her own head or hand in the way of her own look: past it
                p = p + u * (h + 1e-3)
                gone += h + 1e-3
                continue
            return int(m.body_rootid[body])
        return None

    def _contacts(self, m, d):
        """per toy: the child's hand sides touching it, her own touching it, and the fixture words under it"""
        touch = {t: dict(child=set(), fixture=set(), toys=set()) for t in self.toys}
        for i in range(d.ncon):
            c = d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                t = self.toy_of_root.get(int(m.body_rootid[int(m.geom_bodyid[g])]))
                if t is None:
                    continue
                ob = int(m.geom_bodyid[o])
                for side in ("left", "right"):
                    if ob in self.hand[side]:
                        touch[t]["child"].add(side)
                f = self.fixture_geom.get(int(o))
                if f is not None:
                    touch[t]["fixture"].add(f)
                t2 = self.toy_of_root.get(int(m.body_rootid[ob]))
                if t2 is not None and t2 != t:
                    touch[t]["toys"].add(t2)                          # a toy against a toy (a hit with what it holds, A89)
        return touch

    def _head(self, d):
        """the G1's head camera as a person reads it: the eyes' midpoint and its forward, left and up axes (rows)"""
        R = d.cam_xmat[self.cam["L"]].reshape(3, 3)
        pos = (d.cam_xpos[self.cam["L"]] + d.cam_xpos[self.cam["R"]]) / 2
        return pos.copy(), np.stack([-R[:, 2], -R[:, 0], R[:, 1]])

    def _percept(self, world, t):
        m, d = world.m, world.d
        pm = world.parent
        mouth, fwd, face = EY.mouth_point(m, d)
        present = not pm.asleep and float(pm.base["at"][0]) < ROOM_EDGE_X     # awake and in the room (the hall is away)
        head, axes = self._head(d)
        pos = {tt: d.xpos[b].copy() for tt, b in self.toy_body.items()}
        in_view = self._ray_first(m, d, face, d.xpos[m.body("torso_link").id]) == self.g1_root
        ang = PC._angles(face - head, axes)
        seen_by_child = ang is not None and abs(ang[0]) <= K.CAMERA_FIELD_DEG[0] / 2 and abs(ang[1]) <= K.CAMERA_FIELD_DEG[1] / 2
        visible = [tt for tt in self.toys if self._visible(m, d, face, tt, pos[tt])]
        things = [(tt, pos[tt]) for tt in visible] + [("mama", face)]
        rd = self.conduct.reader
        target, before = rd.look(head, axes, things)
        self._before = before                                              # C158: what stands before the child's camera this tick
        touch = self._contacts(m, d)
        holds = tuple(tt for tt in self.toys if touch[tt]["child"])
        ch = PM_child(world)
        grasp = {"left": ch.grasp["L"], "right": ch.grasp["R"]}
        self._bucket_hand = None                                             # C174 (2026-10-01): THE HAND AT THE BUCKET, a ruler: when a hand is
        bb_ = self.toy_body.get("bucket")                                    # within BUCKET_NEAR_M of the bucket's centre, [its height above the
        if bb_ is not None:                                                  # rim, its distance in from the rim's edge] for the nearer hand
            cb = d.xpos[bb_]; rim = float(cb[2]) + X.BUCKET_WALL + X.BUCKET_H
            near_ = [(float(np.linalg.norm(g[:2] - cb[:2])), g) for g in (ch.grasp["L"], ch.grasp["R"])]   # (negative: outside). After each of
            dist_, g_ = min(near_, key=lambda x: x[0])                       # day 44's three hides the child's hand lay on the bucket 87 to 92
            if dist_ <= BUCKET_NEAR_M:                                       # ticks and never on the toy inside: did it reach over the rim?
                self._bucket_hand = [round(float(g_[2]) - rim, 3), round(X.BUCKET_IN - dist_, 3)]
        reaches = rd.reaches(grasp, [(tt, pos[tt]) for tt in visible], holds)
        her = {tt for tt in (pm.holding or {}).values() if tt is not None}
        in_bucket = set()                                                  # A129: the toys lying inside the bucket (A126), by its own frame
        bucket_b = self.toy_body.get("bucket")
        if bucket_b is not None:
            Rt = d.xmat[bucket_b].reshape(3, 3); ct = d.xpos[bucket_b]
            for tt in self.toys:
                if tt != "bucket":
                    loc = Rt.T @ (pos[tt] - ct)
                    if abs(loc[0]) < X.BUCKET_IN and abs(loc[1]) < X.BUCKET_IN and X.BUCKET_WALL < loc[2] < X.BUCKET_WALL + X.BUCKET_H + 0.02:
                        in_bucket.add(tt)
        self._in_bucket = in_bucket
        on_pairs = []
        seen = []
        for tt in visible:
            if tt in her:
                on = "mama"
            elif tt in holds:
                on = "hand"
            elif tt in in_bucket:
                on = "bucket"                                              # A129: seen in the bucket (templates.OPEN_CONTAINERS: she sees into it)
            else:
                fs = sorted(touch[tt]["fixture"])
                on = fs[0] if fs else ""
            if on in self.fixtures:
                pass
            reach = tt in holds or any(np.linalg.norm(pos[tt] - d.xpos[self.shoulder[s]]) <= self.arm_reach[s] for s in self.shoulder)
            seen.append((tt, on, reach))
        near = PC.Reader.near(head, things, on=on_pairs)
        cols = self.conduct.world["objects"]
        seen_t = tuple(PC.Seen(id=tt, name=tt, colour=(cols.get(tt) or [""])[0], on=on, child_sees=tt in before,
                               child_can_reach=bool(reach) and tt not in her, near=near.get(tt)) for tt, on, reach in seen)
        events = self._events(world, t, pos, holds, her, ch, in_view, grasp, touch, reaches)
        raw = world.tract_raw
        sounding = raw is not None and bool(np.any(raw))
        p = PC.Percept(tick=t, present=present, child_in_view=bool(in_view), seen_by_child=bool(seen_by_child),
                       child_target=target, child_holds=holds, seen=seen_t, fixtures=frozenset(self.fixtures) if present else frozenset(),
                       events=tuple(events), child_sounding=sounding, child_reaches=tuple(reaches), face_near=near.get("mama"),
                       face_down=self.last.get("posture") == "front",     # A102: face down this tick (her turn's standing reason)
                       her_hold=bool(pm.holds))                            # A111: her hands on it (a hold of hers engaged)
        return PC.check_events(p), mouth, face

    def _events(self, world, t, pos, holds, her, ch, in_view, grasp=None, touch=None, reaches=()):
        ev = []
        grasp = grasp or {"left": ch.grasp["L"], "right": ch.grasp["R"]}
        touch = touch or {tt: dict(child=set(), fixture=set(), toys=set()) for tt in self.toys}
        if self.first:                                                  # the first tick: what already lies against its hands it did
            self.first = False                                          # not get, and nothing has moved yet
            self.child_had = tuple(holds); self.her_had = {tt: t for tt in her}
            for tt in self.toys:
                self.toy_z[tt] = float(pos[tt][2]); self.toy_prev[tt] = pos[tt].copy(); self.toy_rest_z[tt] = float(pos[tt][2])
            for side in ("left", "right"):
                self.hand_prev[side] = np.asarray(grasp[side], float).copy()
            self.posture = ch.posture if ch.posture in ("back", "front") else None
            self.last["posture"] = ch.posture
            return ev
        for tt in reaches:
            self.reach_t[tt] = t
        self._hands(t, pos, holds, grasp, ev)
        for tt in self.toys:                                            # a toy's fall, once a drop
            z0 = self.toy_z.get(tt)
            z = float(pos[tt][2])
            self.toy_z[tt] = z
            if z0 is None:
                continue
            v = (z0 - z) / (TICK / V.SR)
            if v > FALL_MPS and tt not in holds and tt not in her and tt not in self.falling:
                ev.append(("fell", tt))
                self.falling.add(tt)
            elif abs(v) < REST_MPS:
                self.falling.discard(tt)
        for tt, tk in list(self.her_had.items()):                      # her hand let a toy go: its hand-over's tick
            if tt not in her:
                self.released[tt] = t
        in_bucket = set(getattr(self, "_in_bucket", ()))                      # A129 (the percept's reading this tick): the toys in the bucket
        for tt in in_bucket:                                               # a toy is HIDDEN once it lies in the bucket from her hand
            if tt not in self.hidden and (tt in her or t - self.released.get(tt, -10 ** 9) <= HANDOVER_TICKS):
                self.hidden[tt] = t
                self.hidden_seen[tt] = bool("bucket" in (getattr(self, "_before", None) or ()))   # C158: hidden in its view, or not
        for tt in list(self.hidden):
            if tt in in_bucket or tt in holds:
                self.hidden_out.pop(tt, None)
            elif t - self.hidden_out.setdefault(tt, t) >= HIDDEN_OUT_TICKS:   # out of the bucket by other means (a tip, a fall) for
                self.hidden.pop(tt); self.hidden_out.pop(tt, None); self.hidden_seen.pop(tt, None)   # HIDDEN_OUT_TICKS with no hand on it: no longer hidden
        for tt in self.toys:                                            # its own reach and hold (A89): the hand moved, or reached for the
            if tt in holds:                                             # toy, as the touch began, and the toy kept in its touch GOT_HOLD ticks
                run = self.touch_run.get(tt, 0) + 1
                if run == 1:
                    self.got_arm[tt] = (any(t - self.hand_moved_t[s] <= MOVED_TICKS for s in touch[tt]["child"])
                                        or t - self.reach_t.get(tt, -10 ** 9) <= MOVED_TICKS) and \
                        t - self.released.get(tt, -10 ** 9) > HANDOVER_TICKS
                self.touch_run[tt] = run
                if tt in self.hidden:                                   # C116 (2026-09-29): a toy rattling in the bucket touches the hand
                    ft = [x for x in self.found_ticks.get(tt, []) if t - x < FOUND_WINDOW] + [int(t)]   # on alternate ticks (day 29's
                    self.found_ticks[tt] = ft                           # first hide: its hand on the hidden car at +152, +154, +156,
                else:                                                   # shaking it, and no three ticks running): the find is its hand
                    ft = []                                             # on the hidden toy on FOUND_TOUCHES of the last FOUND_WINDOW ticks
                if tt in self.hidden and (run == GOT_HOLD or len(ft) >= FOUND_TOUCHES):   # A129: its hand on the hidden toy: FOUND
                    self.found_log.append((int(t), tt, bool(self.hidden_seen.pop(tt, False)))); del self.found_log[:-50]   # C158: the find, witnessed or blind
                    ev.append(("found", tt)); self.hidden.pop(tt); self.found_ticks.pop(tt, None); self.released[tt] = -10 ** 9   # (her release is spent: a found toy still in the bucket is not hidden again; worth 2 in her book; the hand-over window does not apply:
                if run == GOT_HOLD and self.got_arm.get(tt):            # the toy was in the bucket, not in her hand)
                    ev.append(("got", tt))
            else:
                self.touch_run[tt] = 0; self.got_arm[tt] = False
        for tt in self.child_had:
            if tt not in holds:
                ev.append(("lost_toy", tt))
                if self.toy_speed.get(tt, 0.0) > THROW_MPS:
                    ev.append(("threw", tt))                            # it left its hand fast (stage 2's frown, A89)
        self._toys(t, pos, holds, her, touch, ev)
        impacts = {str(e[1])[4:] for e in getattr(world.sounds, "last_events", ()) if e[2] != "motion" and str(e[1]).startswith("toy_")}
        for tt in sorted(impacts):                                      # a toy it held or touched already struck something and sounded
            if tt in self.child_had and t - self.last_hit.get(tt, -10 ** 9) >= HIT_GAP and \
                    (touch[tt]["child"] or tt in holds or any(o in holds for o in touch[tt]["toys"])):
                ev.append(("hit", tt)); self.last_hit[tt] = t
        for tt in sorted(her):
            if tt not in self.her_had and t - self.child_held_at.get(tt, -10 ** 9) <= GAVE_TICKS:
                ev.append(("gave", tt))
        for tt in holds:
            self.child_held_at[tt] = t
        self.child_had = tuple(holds)
        self.her_had = {tt: t for tt in her}
        post = ch.posture
        if post in ("back", "front"):
            if self.posture is not None and post != self.posture:
                ev.append(("rolled", None))
            self.posture = post
            self.side_from = None
        elif post == "side" and self.posture is not None and self.side_from != self.posture:
            self.side_from = self.posture                                       # onto its side from its back or front, once in
            if t - self.last_half_roll >= ROLL_GAP:                             # ROLL_GAP (rocking is one act)
                ev.append(("half_roll", None)); self.last_half_roll = t
        self.sit_run = self.sit_run + 1 if post == "sitting" else 0
        if self.sit_run == SIT_HOLD and t - self.last_sat >= SIT_GAP:      # C134: held, and once in SIT_GAP (a bob is not a sit)
            ev.append(("sat", None)); self.last_sat = t
        if post == "front":                                                 # A125 (the crawl rung): its pelvis carried CRAWL_M along the
            pxy = np.asarray(ch.pelvis[:2], float)                          # floor while on its front, from where the spell began (or
            if self.crawl_from is None:                                     # the last crawl counted), once in CRAWL_GAP: "crawled"
                self.crawl_from = pxy
            elif float(np.linalg.norm(pxy - self.crawl_from)) >= CRAWL_M and t - self.last_crawl >= CRAWL_GAP:
                ev.append(("crawled", None)); self.last_crawl = t; self.crawl_from = pxy
        else:
            self.crawl_from = None
        up = post == "front" and float(ch.head[2] - ch.pelvis[2]) > HEAD_UP_M
        self.head_up_run = self.head_up_run + 1 if up else 0
        if self.head_up_run == HEAD_UP_TICKS and t - self.last_head_up >= HEAD_UP_GAP:   # held, once in HEAD_UP_GAP
            ev.append(("head_up", None)); self.last_head_up = t
        self.head_up = up
        if self.reveal_t <= t <= self.reveal_t + PEEKABOO_ACT[0] and self.peekaboo_done < self.reveal_t:
            for side in ("left", "right"):                              # peekaboo answered by an act (A2): a hand that rested, moving
                h = self.hand_speed[side]
                if len(h) > PEEKABOO_ACT[1] and h[-1] > HAND_MOVE_MPS and all(x < HAND_REST_MPS for x in h[-PEEKABOO_ACT[1] - 1:-1]):
                    ev.append(("peekaboo_act", None)); self.peekaboo_done = t
                    break
        self.last["posture"] = post
        self.face_down = self.face_down + 1 if post == "front" else 0
        self.off_back = self.off_back + 1 if post in ("front", "side") else 0   # C217: ticks running off its back (its front or its side)
        if post == "back":
            self.tummy_over = False
        self.cry_down = self.cry_down + 1 if (post == "front" and world.crying) else 0
        if post != "front":
            self.distressed = False
        pxy = np.asarray(ch.pelvis[:2], float)
        moved = float(np.linalg.norm(pxy - self.still_from)) if self.still_from is not None else 0.0
        if post != "front" or moved > STILL_M:
            self.still_from = pxy.copy(); self.still_t = t                  # C137: where and when its last prone stillness began
        still = post == "front" and t - self.still_t >= STILL_TICKS         # face down and not going anywhere for STILL_TICKS
        if not self.tummy_over and self.off_back >= TUMMY_TIME_TICKS:
            ev.append(("tummy_time_over", None)); self.tummy_over = True    # C216/C217: tummy time's end, still or not, on its front OR its
                                                                            # side (her turn_over answers it, no concern); once a spell off its back
        if not self.distressed and still and (self.face_down >= DISTRESS_TICKS or self.cry_down >= CRY_DOWN_TICKS):
            ev.append(("distress", None)); self.distressed = True           # once a face-down spell (her turn_over answers it); C137: not
                                                                            # to a child crawling under its own power (day 36: 28 turns
                                                                            # refused at a crawling child; it rolled over itself)
        if world.crying or (in_view and float(world._sensed["true_base_peak"]) > world.f_pain):
            ev.append(("pain", None))                                   # its cry heard, or a blow to its body she sees
        return ev

    def _hands(self, t, pos, holds, grasp, ev):
        """its hands as she sees them (A89): each hand's speed, the tick it last moved, and the end of a movement, where her
        notebook takes the hand's distance to the nearest toy it does not hold (a reach at it); nearer than its best of the last
        BOOK_LAST by NEARER_M: "reach_nearer" """
        for side in ("left", "right"):
            g = np.asarray(grasp[side], float)
            prev = self.hand_prev.get(side)
            sp = 0.0 if prev is None else float(np.linalg.norm(g - prev)) / TICK_S
            self.hand_prev[side] = g.copy()
            h = self.hand_speed[side]
            h.append(sp); del h[:-6]
            if sp > HAND_MOVE_MPS:
                self.hand_moved_t[side] = t
            ended = len(h) >= 3 and sp < HAND_REST_MPS and h[-2] >= HAND_REST_MPS and any(x > HAND_MOVE_MPS for x in h[-4:-1])
            if not ended:
                continue
            near = [(float(np.linalg.norm(g - pos[tt])), tt) for tt in self.toys if tt not in holds]
            if not near:
                continue
            d, tt = min(near)
            if d > REACH_BOOK_M:
                continue
            last = self.book.setdefault(tt, [])
            if last and d < min(last) - NEARER_M:
                ev.append(("reach_nearer", tt))
            last.append(round(d, 4)); del last[:-BOOK_LAST]

    def _toys(self, t, pos, holds, her, touch, ev):
        """the toys as she sees them (A89): each one's speed; where it lay still in no hand; a lift and a shake of one in its hand"""
        for tt in self.toys:
            p_ = pos[tt]
            prev = self.toy_prev.get(tt)
            sp = 0.0 if prev is None else float(np.linalg.norm(p_ - prev)) / TICK_S
            self.toy_prev[tt] = p_.copy()
            last_sp = self.toy_speed.get(tt, 0.0)
            self.toy_speed[tt] = sp
            if tt in holds:
                z0 = self.toy_rest_z.get(tt)
                if z0 is not None and float(p_[2]) > z0 + LIFT_M and tt not in self.lifted:
                    ev.append(("lifted", tt)); self.lifted.add(tt)
                if sp > SHAKE_MPS and last_sp > SHAKE_MPS and t - self.last_shook.get(tt, -10 ** 9) >= SHAKE_GAP:
                    ev.append(("shook", tt)); self.last_shook[tt] = t
            else:
                self.lifted.discard(tt)
                if tt not in her and sp < REST_MPS:
                    self.toy_rest_z[tt] = float(p_[2])

    # ------------------------------------------------------------------ one tick of hers (by day)
    def tick(self, world):
        t = int(world.tick)
        m, d = world.m, world.d
        p, mouth, face = self._percept(world, t)
        self._p = p
        if self.plan is not None:
            try:
                self.plan.tick(t, t - self.day_start, self, world)    # her day's episode (L3), before her conduct's tick
            except Exception as e:                                    # C123 (2026-09-29): an error in her day plan never stops the life:
                self.plan_errors = int(getattr(self, "plan_errors", 0)) + 1   # on day 28 a lesson on a toy out of her view raised a KeyError
                if self.plan_errors <= 3 or self.plan_errors % 1000 == 0:     # (C121's change) and the life died 1,500 ticks after every
                    import traceback                                          # restart, four times over; the plan skips the tick, the
                    print(f"PLAN ERROR {self.plan_errors} at tick {t}: {type(e).__name__}: {e}", flush=True)   # error is printed
                    traceback.print_exc()
        raw = world.tract_raw
        tract = None if raw is None or not np.any(raw) else np.asarray(raw, float)
        tp = d.xpos[m.body("torso_link").id]
        tR = d.xmat[m.body("torso_link").id].reshape(3, 3)
        ear_l, ear_r, cmouth = EA.head_from_torso(tp, tR)
        dist = float(np.linalg.norm(cmouth - d.xpos[self.head_b]))
        tok = world.words_out
        token = None if tok is None else int(AN_TO_LX[int(tok)])
        if token == LX.ID[LX.REST]:
            token = None
        vd, self.voice_done = self.voice_done, None
        out = self.conduct.tick(t, p, tract=tract, distance_m=dist, token=token, voice_done=vd)
        scaffold = self.conduct.scaffold
        words = self.words if scaffold else None
        if out.cut and self.utt is not None:
            self.utt.cut(words)
        if out.line is not None:
            self.n_lines += 1
            if out.clip is not None:
                if self.utt is not None and not self.utt.done:
                    self.utt.cut(words)                     # (the conduct says a line only when her voice is free)
                self.utt = Utterance(out.clip, t)
        pa = np.zeros(TICK, np.float32)
        if self.utt is not None:
            if self.utt.start + self.utt.pos // TICK == t:
                was = self.utt.done
                pa = self.utt.tick(t, words)
                if self.utt.done and not was and self.utt.cut_at_tick is not None:
                    self.voice_done = self.utt.start + (max(self.utt.stop_at, 1) - 1) // TICK
            if self.utt.done:
                self.utt = None
        # her feelings and her face (4.3): her judgments, frowns and concern; the face test's two ticks (A1)
        for w, kind, _word in out.judgments:
            self.feel.judge(w, kind)
        if out.frown in ("hit", "threw"):
            self.feel.harm()                                # stage 2: its own act hit her, or it threw a toy (A89)
        elif out.frown == "talk_over":
            self.feel.talk_over()
        if out.line is not None and out.line.intent == "peekaboo":
            self.reveal_t = t                               # her reveal: an act within PEEKABOO_ACT ticks answers it (A2)
        for k, _o in p.events:
            if k == "pain":
                self.feel.child_pain()
            elif k == "distress":
                self.feel.distress()
        self.feel.set_engagement(1.0 if p.present else 0.0)
        self.feel.set_question(1.0 if self.conduct.pending is not None else 0.0)
        loud = 0.0 if self.utt is None and not np.any(pa) else float(np.sqrt(np.mean(pa.astype(np.float64) ** 2)))
        self.feel.set_speech(min(1.0, loud / V.SPEECH_PA))
        test = any(v[0] for v in EY.face_test(m, d, world.gaze).values())
        seen = test and self.test_prev
        self.test_prev = test
        bringing = any(a_[1] == "lean_in" and a_[2] == "child_line" and a_[5] not in C.ENDED for a_ in self.conduct.acts_open)
        fp = self.feel.step(seen, bringing)                 # A96: a smile she brings to its line of sight is held while she does
        self.fp = dict(kin.FACE_NEUTRAL) if self.conduct.still else fp       # a trial's face: its neutral set (4.8)
        world.parent.set_face(self.fp)
        self._read_face(t, seen or p.present)              # A158: her expression reaches the child while she is with it, looked at or not
        # the words channel: a symbol only while she is audible at its nearer ear and the scaffold is on (A29)
        path = EA.paths(mouth, ear_l, ear_r)[0]
        audible = LX.audible(SPEECH_DB, [-20.0 * math.log10(max(float(x), 1e-6)) for x in path])
        sym = self.words.tick(t, audible) if scaffold else LX.ID[LX.REST]
        self.word_now = int(LX_TO_AN[int(sym)])
        self.last = dict(posture=self.last.get("posture"), target=p.child_target, holds=p.child_holds, seen=len(p.seen), in_bucket=sorted(getattr(self, "_in_bucket", ())),
                         events=[list(e) for e in p.events], line=None if out.line is None else out.line.text,
                         heard=[cw.word for cw in out.heard], judged=[list(j) for j in out.judgments], cut=bool(out.cut),
                         face_test=bool(test), reading=self.reading, word=self.word_now, in_view=p.child_in_view,
                         seen_by_child=p.seen_by_child, present=p.present,
                         found=[[o_, s_] for tk_, o_, s_ in self.found_log if tk_ == int(t)],   # C158: this tick's finds, each with whether the hide was in its view
                         bucket_hand=getattr(self, "_bucket_hand", None), hidden=sorted(self.hidden),   # C174: the nearer hand at the bucket [above the rim, in from its edge]; the toys hidden now
                         child_xy=[float(world.d.qpos[0]), float(world.d.qpos[1])])   # C131: its pelvis on the floor plan (her plan reads it)
        return {SOURCE: (pa.astype(np.float64), mouth)}

    def _read_face(self, t, with_it):
        """the born reading (A1, A2, A49; A158): 2 x (smile - frown) of the face she shows while she is with the child (awake, in the
        room: Percept.present), looked at or not, as a parent's approving tone and touch reach an infant looking elsewhere (Fernald
        1993; the reward's carrier read from the world, A49's scaffold); held READING_HOLD ticks once she is away, then 0; the frame's
        pair is [the reading, its change]. Until A158 (2026-10-01) it read only while the face test held: days 49 to 55 then paid about 0"""
        prev = self.reading
        if with_it:
            self.reading = kin.face_reading(self.fp)
            self.reading_t = t
        elif t - self.reading_t > READING_HOLD:
            self.reading = 0.0
        self.face_seen = np.array([self.reading, self.reading - prev])

    def after_apply(self, world):
        if world.night:
            self.word_now = AN_REST
            self.face_seen = np.array([self.reading, 0.0])

    # ------------------------------------------------------------------ night and morning
    def dusk(self, world):
        """her line ends where it is (its words withdrawn as a cut withdraws them), her feelings rest, and her conduct's night
        boundary is kept (the day's new words join hers; her ear checked)"""
        if self.utt is not None and not self.utt.done:
            self.utt.cut(self.words if self.conduct.scaffold else None)
        self.utt = None
        self.words.queue = []
        self.conduct.night()
        self.feel.set_engagement(0.0)
        self.word_now = AN_REST

    def dawn(self, world):
        self.feel.set_engagement(1.0)
        self.day += 1
        self.day_start = int(world.tick)
        self.distressed = False; self.face_down = 0; self.cry_down = 0   # A107 (C89): a face-down spell starts anew at waking: a child
        self.off_back = 0; self.tummy_over = False                       # (C217: and tummy time's clock)
        self.conduct.left = {}                                           # A117: the toys she left where they lay: the room tidied (B8)
        if self.conduct.dawn(int(world.tick)):                           # C142: her stage advances at a dawn (stage 2: the words her ear
            print(f"dawn {self.day}: her stage 2 begins (C142): {self.conduct.book_log[-1][4]}", flush=True)   # accepts smiled, the frowns)
                                                                         # that slept prone and wakes prone is found so, and its distress
                                                                         # (after DISTRESS_TICKS) owes her turn again; the flag saved True
                                                                         # across a night gave day 6 no turn at all

    # ------------------------------------------------------------------ the save
    def state(self):
        f = self.feel
        pulse = lambda q: None if q is None else dict(n=q.n, t0=q.t0, a=q.a, kind=q.kind, seen_at=q.seen_at, started=q.started,
                                                       bringing=q.bringing, arrived=q.arrived)
        feel = dict(rng=f.rng.bit_generator.state, t=f.t, pulse=pulse(f.pulse), queued=pulse(f.queued), frown_t0=f.frown_t0,
                    frown_amp=f.frown_amp, U=f.U, Cn=f.Cn, A=f.A, M=f.M, A_target=f.A_target, q=f.q, loud=f.loud, wind=f.wind,
                    sudden_log=[list(x) for x in f.sudden_log], flash_t0=f.flash_t0, last_seen=f.last_seen, was_seen=f.was_seen,
                    neutral_run=f.neutral_run, last_seen_L=f.last_seen_L, last_seen_t=f.last_seen_t, next_blink=f.next_blink,
                    log=[list(x) for x in f.log[-200:]], state=_pl(getattr(f, "state", None)))
        utt = None
        if self.utt is not None:
            c = self.utt.clip
            utt = dict(clip=dict(key=c.key, text=c.text, register=c.register, pcm=np.asarray(c.pcm).copy(),
                                 words=[list(w) for w in c.words], digest=c.digest, gain_db=float(c.gain_db), meta=_pl(dict(c.meta))),
                       play=_pl(self.utt.state()))
        return dict(conduct=self.conduct.state(), feel=feel, words=dict(queue=[[q.tick, q.order, q.sym, q.word] for q in self.words.queue],
                    order=self.words.order, dropped=self.words.dropped, withdrawn=self.words.withdrawn),
                    utt=utt, voice_done=self.voice_done, word_now=self.word_now, face_seen=self.face_seen.copy(),
                    reading=self.reading, reading_t=self.reading_t, test_prev=self.test_prev, fp=_pl(self.fp),
                    toy_z=dict(self.toy_z), falling=sorted(self.falling), child_had=list(self.child_had),
                    child_held_at=dict(self.child_held_at), her_had=dict(self.her_had), released=dict(self.released), hidden=dict(self.hidden), hidden_seen=dict(self.hidden_seen), found_ticks={k: list(v) for k, v in self.found_ticks.items()},
                    posture=self.posture, face_down=self.face_down, last_posture=self.last.get("posture"), n_lines=self.n_lines,
                    off_back=self.off_back, tummy_over=self.tummy_over,   # C217
                    plan=None if self.plan is None else self.plan.state(), day=self.day, day_start=self.day_start,
                    eyes=dict(first=self.first, hand_prev={k: v.tolist() for k, v in self.hand_prev.items()},
                              hand_speed={k: list(v) for k, v in self.hand_speed.items()}, hand_moved_t=dict(self.hand_moved_t),
                              toy_prev={k: v.tolist() for k, v in self.toy_prev.items()}, toy_speed=dict(self.toy_speed),
                              toy_rest_z=dict(self.toy_rest_z), lifted=sorted(self.lifted), last_shook=dict(self.last_shook),
                              last_hit=dict(self.last_hit), reach_t=dict(self.reach_t), book={k: list(v) for k, v in self.book.items()},
                              side_from=self.side_from, head_up=self.head_up, reveal_t=self.reveal_t, peekaboo_done=self.peekaboo_done,
                              last_half_roll=self.last_half_roll, head_up_run=self.head_up_run, last_head_up=self.last_head_up,
                              sit_run=self.sit_run, last_sat=self.last_sat,
                              still_from=None if self.still_from is None else [float(x) for x in self.still_from], still_t=self.still_t,
                              cry_down=self.cry_down, distressed=self.distressed, touch_run=dict(self.touch_run),
                              got_arm=dict(self.got_arm)))

    def load_state(self, s):
        self.conduct.load_state(s["conduct"])
        f, fs = self.feel, s["feel"]
        f.rng.bit_generator.state = fs["rng"]

        def pulse(q):
            if q is None:
                return None
            p = PF.Pulse.__new__(PF.Pulse)
            p.n, p.t0, p.a, p.kind, p.seen_at, p.started = q["n"], q["t0"], q["a"], q["kind"], q["seen_at"], q["started"]
            p.bringing, p.arrived = bool(q.get("bringing", False)), q.get("arrived")   # A96 (a save from before it: never brought)
            return p
        f.t, f.pulse, f.queued = fs["t"], pulse(fs["pulse"]), pulse(fs["queued"])
        for k in ("frown_t0", "frown_amp", "U", "Cn", "A", "M", "A_target", "q", "loud", "wind", "flash_t0", "last_seen", "was_seen",
                  "neutral_run", "last_seen_L", "last_seen_t", "next_blink"):
            setattr(f, k, fs[k])
        f.sudden_log = [tuple(x) for x in fs["sudden_log"]]
        f.log = [tuple(x) for x in fs["log"]]
        if fs["state"] is not None:
            f.state = dict(fs["state"])
        w = s["words"]
        self.words.queue = [LX._Due(int(a), int(b), int(c), str(e)) for a, b, c, e in w["queue"]]
        self.words.order, self.words.dropped, self.words.withdrawn = w["order"], w["dropped"], w["withdrawn"]
        self.utt = None
        if s["utt"] is not None:
            c = s["utt"]["clip"]
            clip = V.Clip(c["key"], c["text"], c["register"], np.asarray(c["pcm"], np.int16).copy(),
                          [(str(a), int(b), int(e)) for a, b, e in c["words"]], c["digest"], float(c["gain_db"]), dict(c["meta"]))
            pl = dict(s["utt"]["play"])
            pl["last"] = np.asarray(pl["last"], np.float32)
            self.utt = Utterance.restore(clip, pl)
        self.voice_done, self.word_now = s["voice_done"], int(s["word_now"])
        self.face_seen = np.asarray(s["face_seen"], float).copy()
        self.reading, self.reading_t, self.test_prev = float(s["reading"]), int(s["reading_t"]), bool(s["test_prev"])
        self.fp = dict(s["fp"])
        self.toy_z = dict(s["toy_z"]); self.falling = set(s["falling"]); self.child_had = tuple(s["child_had"])
        self.child_held_at = dict(s["child_held_at"]); self.her_had = dict(s["her_had"]); self.released = dict(s["released"])
        self.hidden = dict(s.get("hidden", {}))                         # A129 (a save from before it: nothing hidden)
        self.hidden_seen = {k: bool(v) for k, v in dict(s.get("hidden_seen", {})).items()}; self.found_log = []   # C158 (older saves: unknown)
        self.found_ticks = {k: [int(x) for x in v] for k, v in dict(s.get("found_ticks", {})).items()}   # C116 (older saves: none)
        self.posture, self.face_down = s["posture"], int(s["face_down"])
        self.off_back, self.tummy_over = int(s.get("off_back", 0)), bool(s.get("tummy_over", False))   # (C217; older saves: fresh)
        self.last = dict(posture=s["last_posture"])
        e = s.get("eyes")
        if e is not None:
            self.first = bool(e["first"])
            self.hand_prev = {k: np.asarray(v, float) for k, v in e["hand_prev"].items()}
            self.hand_speed = {k: [float(x) for x in v] for k, v in e["hand_speed"].items()}
            self.hand_moved_t = {k: int(v) for k, v in e["hand_moved_t"].items()}
            self.toy_prev = {k: np.asarray(v, float) for k, v in e["toy_prev"].items()}
            self.toy_speed = {k: float(v) for k, v in e["toy_speed"].items()}
            self.toy_rest_z = {k: float(v) for k, v in e["toy_rest_z"].items()}
            self.lifted = set(e["lifted"]); self.last_shook = {k: int(v) for k, v in e["last_shook"].items()}
            self.last_hit = {k: int(v) for k, v in e["last_hit"].items()}; self.reach_t = {k: int(v) for k, v in e["reach_t"].items()}
            self.book = {k: [float(x) for x in v] for k, v in e["book"].items()}
            self.side_from, self.head_up = e["side_from"], bool(e["head_up"])
            self.reveal_t, self.peekaboo_done = int(e["reveal_t"]), int(e["peekaboo_done"])
            self.last_half_roll = int(e.get("last_half_roll", -10 ** 9)); self.head_up_run = int(e.get("head_up_run", 0))
            self.sit_run, self.last_sat = int(e.get("sit_run", 0)), int(e.get("last_sat", -10 ** 9))
            sf = e.get("still_from"); self.still_from = None if sf is None else np.asarray(sf, float); self.still_t = int(e.get("still_t", -10 ** 9))
            self.last_head_up = int(e.get("last_head_up", -10 ** 9))
            self.cry_down, self.distressed = int(e.get("cry_down", 0)), bool(e.get("distressed", False))
            self.touch_run = {k: int(v) for k, v in e.get("touch_run", {}).items()}; self.got_arm = {k: bool(v) for k, v in e.get("got_arm", {}).items()}
        self.n_lines = int(s["n_lines"])
        if self.plan is not None and s.get("plan") is not None:
            self.plan.load_state(s["plan"])
        self.day, self.day_start = int(s.get("day", 0)), int(s.get("day_start", 0))


def PM_child(world):
    """the G1 as her planner sees it (parent_motion.Child: world truth for her, never the body's)"""
    from body.sim import parent_motion as PM                  # noqa: PLC0415
    return PM.Child(world.m, world.d, world.parent.scene.g1_set)
