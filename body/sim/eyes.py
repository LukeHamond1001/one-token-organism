"""THE G1'S EYES: THE D435'S THREE VIEWS (docs/SIM_DESIGN.md 3.4, 3.7, A1, A38, A42; W3, reopened at S5a as W3r): the head's
RealSense D435 as the robot carries it (body/sim/g1scene.py adds the cameras at load): its two monochrome imagers 50 mm apart, each
rendered at 336 x 192 px (88.3 x 58 deg, 3 px a degree at the centre), and its colour camera 15 mm beside the left one at 238 x 134
px (69.4 x 42.5 deg), all three into one offscreen buffer with one read-back, multisampling off (a render is a function of the state).
Each grey image is the imager's visible response to the render (LUMA: Rec. 601's weights until the OV9282's own are read, C45); the
render has no near-infrared light, which the real imagers also see (a disclosed gap). Then, in software, as the real G1 could:
  PERIPHERY  each grey eye's whole field averaged 6 x 6: 56 x 32 px, coded as 7 x 4 cells of 8 px x luminance ON and OFF against
             mid-grey (56 an eye); the colour camera's whole image as 5 x 3 cells x red-green and blue-yellow, ON and OFF (60):
             eye_p, 172 numbers.
  FOVEA      a 64 x 64 px window of each grey eye's native image (about 21 deg, a newborn's acuity: A42), placed by the world's gaze
             state (the gaze effector's conjugate yaw and pitch, the vergence setting the two windows apart; the VOR counter-turning
             it), read by THE BORN BANK (A42; Hubel and Wiesel 1963): centre-surround ON and OFF (a difference of Gaussians, centre
             sd 1 px, surround 3 px) and oriented energy (an even and an odd Gabor as a complex cell) at 4 orientations and periods
             of 3 and 6 px (1.0 and 0.5 cycles a degree), each pooled over 8 x 8 px cells: 8 x 8 cells x 10 maps, 640 an eye. THE
             COLOUR WINDOW: the same 21 deg at the left eye's gaze direction in the colour camera's image (64 px), 8 x 8 cells x
             red-green and blue-yellow ON and OFF (256); what lies outside that camera's field reads nothing: eye_f, 1,536 numbers.
  No depth channel and no stereo algorithm: the body gets both eyes and learns what they share (the owner's decision 4).
  face_fovea reads nothing: at birth there is no born face detector on the pixels (C39, option a); the born template below is an
  instrument. face_periph is THE BORN FACE CUE'S STAND-IN (A157; `face_cue`): [fired, yaw, pitch], 1 and her mouth's direction from
  the fovea's centre (rad, + right / + up) while her face lies in either eye's image under A1's conditions but the fovea's, else
  zeros: the orienting the newborn has toward a face (CONSPEC, Johnson and Morton 1991) read from the world as the smile's level is
  (A49's scaffold law), until a born detector on the pixels works; its removal test is A49's. onset_periph is THE BORN VISUAL ONSET CUE (A43; `_onset`): the grey peripheries' sudden local change, habituating per
  place, suppressed while the trunk turns fast; its constants ours or recalled until C49 reads them from their sources. The camera model (A38: exposure,
  Poisson-Gaussian noise, the head's motion blur, the colour camera's rolling rows, gamma) waits for its constants (C45).
The images themselves go to the frame's truth (the page, the stills, the eye check), never to the body.

THE FACE TEST (A1; the reward carrier's gate, world truth, never a channel of the body): an eye sees the parent's face this tick
when all four hold: the parent's mouth point lies inside that eye's fovea window; a ray from the eye to it over the geoms the eyes
render hits nothing first but her face within 2 cm of it or her own lips (the child's own hand, a toy, the parent's hand or hair
block it; her lips are her mouth, met first when her face is turned); the face is turned within 75 deg of
the eye; and its front (an ellipse 0.17 x 0.21 m) covers at least 20 fovea px x the cosine of the turn. Either eye counts. It goes
to the frame's truth (`face_test`), where the face channel's gate (the born reading's 2 consecutive ticks and its 30-tick hold,
P3/S5a) and the parent read it.
THE BORN FACE TEMPLATE (3.4, A1; an instrument since C39's option (a)): CONSPEC's three dark blobs (`face_template`; its constants
below), read on a grey fovea or a colour render by the tools and the eye tests. Measured on her face of human proportions it rarely
detected her face (C39), so at birth no born face detector on the pixels reaches the body: face_fovea reads nothing. From C39's
option (a) to day 55 the born route to faces was orienting to sudden change and sound with her face brought into the child's line, and
measured over days 49 to 55 her face passed A1's test on 1 to 3% of ticks and paid nothing (her smiles judged 90 to 155 a day, the
face's felt reward about 0; a copy read: her face inside the camera field and within the gaze's reach on every tick, the gaze held
17 deg below her mouth); so A157 fills face_periph from the world as A49 fills the smile: the detector's disclosed stand-in.

NO LAMP AT THE EYES (the W1 verifier's fifth finding). The G1 carries no lamp, so the eyes see by the room's lights alone: the room
has no headlight (MuJoCo's lamp at the viewing camera), and `Eyes` refuses to render with one on. The room's own indirect light,
which a lamp at the eye had stood in for, is its lights' ambient term (body/sim/make_g1room.py, ROOM_INDIRECT), so a face bent
over the child is lit from below and the side as a real room lights it, and it goes out with its lights. For W5's night: darken
the lights' terms, never switch every light off (MuJoCo then draws the scene unlit, at full brightness), and the emissive surfaces
(the ceiling, the window, the lamp shade, the dock) still glow (body/tests/test_sim_eyes.py, eyes 10); her eyes' highlights no
longer do (drawn, not lamps: the W1 verifier's third round).

SHADOWS. The room has a sun, a key spot and a fill. `shadows`: "sun" (the sun's shadow only: the body's eyes, 3.4), "all" or
"none". The sun's shadow stays in the body's eyes: it is part of the owner's complete reality (decision 3), never decided by the
eye check or by the tick (5.1, B3). The eye check only reports what the fovea tells apart with it and without (tools/
sim_eye_check.py, 2026-09-24, 1,650 views at midday: a small nonlinear readout 0.865 with it and 0.864 without; the face 0.76 and
0.82; linear 0.62 and 0.64). It costs about 2.5x the render; dropping it for speed is the owner's render choice (SIM_DESIGN.md 9
and B3), never ours."""
import hashlib
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from body.sim import world as W  # noqa: E402

G = W.G
kin = G.kin
POOL = 6                        # the periphery: each grey eye's native image averaged 6 x 6, 56 x 32 px (3.4; anatomy, ours)
CELL_P, CELL_F = 8, 8           # the retina's cells: 8 px of the periphery (7 x 4 per eye, about 12.5 x 14.5 deg) and 8 px of the fovea
                                # (8 x 8 per eye, 2.6 deg) (3.4; anatomy, ours)
COL_CELLS = (3, 5)              # the colour camera's periphery: 5 x 3 cells across its image, about 14 deg each (3.4; anatomy, ours)
LUMA = np.array([0.299, 0.587, 0.114])   # the grey imagers' visible response over the render's red, green and blue: Rec. 601's luma
                                # weights, ours until the OV9282's own response is read (C45)
CS_SIGMA = (1.0, 3.0)           # THE BORN BANK (A42; section 10): the centre-surround cells' centre and surround, Gaussian sd in px
                                # (the centre 1 px, 0.33 deg; the surround 3 x: ours)
GABOR_PERIODS = (3.0, 6.0)      # the oriented cells' periods in px: 1.0 and 0.5 cycles a degree at 3 px a degree (a newborn's
                                # limit and half of it: Dobson and Teller 1978)
GABOR_ORIENTS = 4               # 0, 45, 90, 135 deg
GABOR_SIGMA = 0.56              # each Gabor's envelope sd in periods: a one-octave bandwidth (ours)
BANK_MAPS = 2 + 2 * GABOR_ORIENTS          # per pixel: CS ON, CS OFF, and the oriented energy at 4 orientations x 2 scales
ONSET_WEBER = 0.2               # THE VISUAL ONSET CUE (A43; C49): a periphery pixel's change this tick over the scene's mean luminance
                                # (the level the retina adapts to), beyond
                                # the image's median change, past a newborn's contrast threshold at its best (10-20%: Banks and
                                # Salapatek 1978, recalled; its top: a change seen in the periphery)
ONSET_EPS = 0.02                # the adaptation level's floor (a black scene's change is not infinite: ours)
TURN_SUPPRESS = 0.5             # rad/s: no onset while the trunk turns faster (the head's own motion changes everything: ours)
ONSET_NEAR = 2                  # a change is an onset only where no change passed ONSET_WEBER within 2 periphery px (about 4 deg) the
                                # tick before: an appearance, not a motion under way (ours)
HAB_STEP, HAB_TAU = 0.5, 10.0   # habituation per place: each onset there halves the next (Sokolov 1963), recovering over 10 s (ours)
EYE_SAMPLES = 0                 # the eyes' multisampling: off, so a render is a function of the state alone (the lead's decision:
                                # determinism before smoothness; the camera model's noise dwarfs the aliasing)
FACE_TURN_DEG = 75.0            # the face test (A1; innate, ours)
FACE_FRONT_M = (0.17, 0.21)
FACE_MIN_PX = 20.0 * (W.EYE_F_PX / (48.0 / math.tan(math.radians(29.0)))) ** 2   # A1's 20 px at the first build's 1.5 px a degree,
                                # the same solid angle at the fovea's 3 px a degree (80 px): the test's reach unchanged (A1, A42)
RAY_SLACK_M = 0.02              # a ray's first hit within 2 cm of the mouth point is the face itself
MOUTH_PARTS = ("parent_mouth", "parent_lip_lo", "parent_teeth")   # her drawn mouth's geoms: a ray's first hit on them is her mouth
_MOUTH_GEOMS = {}
EYE_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)   # the eyes see groups 0-2: never collision proxies, sites or markers
# THE BORN FACE TEMPLATE (3.4, 3.7, A1, A22; innate, ours): CONSPEC's three dark blobs (Johnson and Morton 1991; Goren 1975), two
# for the eyes above one for the mouth, darker than the face around them (newborns turn to this configuration only in that
# polarity: Farroni et al. 2005), in a face-shaped ellipse. Its layout, in the ellipse's width W and height H from its centre, y up,
# from a person's proportions (the eyes at the head's mid-height and about 0.45 W apart; the mouth about three quarters down the
# face; recalled, not measured on the parent): the ellipse FACE_FRONT_M's shape (A1's face front); the eyes discs of 0.20 W at
# (+-0.22 W, +0.12 H); the mouth an ellipse 0.36 W x 0.10 H at (0, -0.25 H). Read on luminance (the retina's L, mean RGB) at a
# bank of sizes (TEMPLATE_WIDTHS px wide, sqrt 2 apart, the largest fitting the fovea): a place matches when the correlation of
# its pixels with the template over the ellipse is at least TEMPLATE_R (a quarter of the pixels' variance explained) and the
# blobs are darker than the face by at least TEMPLATE_CONTRAST (Weber; a newborn's contrast threshold at its best spatial
# frequencies is of this order or higher: Banks and Salapatek 1978, recalled). None of these was set on the parent's face or
# on any hit or false alarm; the eye check (tools/sim_eye_check.py, C3) measures them. On her face as built to a real face's
# proportions and photometry (the W1 verifier's fourth round; body/sim/parent_face.py) the template, unchanged, matches HER FACE
# (centred on it, at its size) in 1 of 48 fovea readings at 0.3-2 m under the three lights (dusk, 1.25 m, r 0.51, where her face
# is the smallest size's width) and by chance elsewhere in the window in 5 more (0.3-0.6 m, the window upside down scoring as
# high), and never in the periphery (tools/sim_face_template.py, body/tests/test_sim_eyes.py eyes 15): a finding for the design
# (C3), which the template's own source decides; neither side is fitted to the other.
TEMPLATE_EYES = ((-0.22, 0.12), (0.22, 0.12))   # the eye blobs' centres, (x / W, y / H)
TEMPLATE_EYE_D = 0.20                           # their diameter / W
TEMPLATE_MOUTH = ((0.0, -0.25), (0.36, 0.10))   # the mouth's centre (x / W, y / H) and its size (width / W, height / H)
TEMPLATE_WIDTHS = (8, 11, 16, 23)               # px
TEMPLATE_R = 0.5
TEMPLATE_CONTRAST = 0.10
EYE_P_SIZE = 2 * (G.EYE_W // POOL // CELL_P) * (G.EYE_H // POOL // CELL_P) * 2 + COL_CELLS[0] * COL_CELLS[1] * 4   # 172
EYE_F_SIZE = 2 * (W.FOVEA_PX // CELL_F) ** 2 * BANK_MAPS + (W.FOVEA_PX // CELL_F) ** 2 * 4                      # 1,536
COL_F_PX = (G.COL_H / 2) / math.tan(math.radians(G.COL_FOVY) / 2)                          # the colour camera's focal length, px
COL_WIN = int(round(W.FOVEA_PX * COL_F_PX / W.EYE_F_PX))                                   # the colour window: the fovea's 21 deg


def grey(img):
    """an imager's grey image (rows x cols, 0-1) from the render's colour (uint8): its visible response (LUMA, C45)"""
    return (img.astype(np.float64) @ LUMA) / 255.0


def on_off(x):
    return np.maximum(x, 0.0), np.maximum(-x, 0.0)


def cells(a, rows, cols):
    """the means of a map (rows_px x cols_px [x k]) over a rows x cols grid of cells (np.array_split's edges)"""
    out = []
    for rb in np.array_split(np.arange(a.shape[0]), rows):
        out.append([a[rb[0]:rb[-1] + 1, cb[0]:cb[-1] + 1].mean(axis=(0, 1)) for cb in np.array_split(np.arange(a.shape[1]), cols)])
    return np.array(out)


def retina_grey(L, cell):
    """a grey image's retinotopic code: cells of `cell` px, each cell's mean luminance against mid-grey, ON and OFF (Kuffler;
    Schiller) -> (rows / cell, cols / cell, 2)"""
    h, w = L.shape[0] // cell * cell, L.shape[1] // cell * cell
    c = L[:h, :w].reshape(h // cell, cell, w // cell, cell).mean(axis=(1, 3)) - 0.5
    return np.stack(on_off(c), axis=-1)


def retina_colour(img, rows, cols):
    """a colour image's opponent code over a grid of cells: red-green and blue-yellow (Hering; De Valois), ON and OFF -> (rows,
    cols, 4)"""
    c = img.astype(np.float64) / 255.0
    rg = c[..., 0] - c[..., 1]
    by = c[..., 2] - (c[..., 0] + c[..., 1]) / 2
    m = cells(np.stack([rg, by], axis=-1), rows, cols)
    rgp, rgn = on_off(m[..., 0]); byp, byn = on_off(m[..., 1])
    return np.stack([rgp, rgn, byp, byn], axis=-1)


def _gauss(sd, r):
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-x ** 2 / (2 * sd * sd)); g /= g.sum()
    return np.outer(g, g)


def _bank_kernels():
    """THE BORN BANK's kernels (A42): the difference of Gaussians, and even and odd Gabors at GABOR_ORIENTS orientations and the
    two periods, each even one zero-mean"""
    r = int(math.ceil(3 * CS_SIGMA[1]))
    dog = _gauss(CS_SIGMA[0], r) - _gauss(CS_SIGMA[1], r)
    gab = []
    for lam in GABOR_PERIODS:
        sd = GABOR_SIGMA * lam
        rr = int(math.ceil(3 * sd))
        y, x = np.mgrid[-rr:rr + 1, -rr:rr + 1].astype(np.float64)
        env = np.exp(-(x * x + y * y) / (2 * sd * sd))
        for k in range(GABOR_ORIENTS):
            th = math.pi * k / GABOR_ORIENTS
            u = x * math.cos(th) + y * math.sin(th)
            ev = env * np.cos(2 * math.pi * u / lam); ev -= env * (ev.sum() / env.sum())
            od = env * np.sin(2 * math.pi * u / lam)
            n = math.sqrt((ev * ev).sum())
            gab.append((ev / n, od / n))
    return dog, gab


BANK = _bank_kernels()


def bank(L):
    """THE BORN BANK on a grey fovea (64 x 64, 0-1; A42): per pixel the centre-surround ON and OFF responses and the oriented energy
    (an even and an odd Gabor's, as a complex cell) at 4 orientations x 2 scales, the window's edge continued by its nearest pixel;
    pooled over CELL_F x CELL_F px cells -> (8, 8, 10)"""
    from scipy.signal import fftconvolve
    dog, gab = BANK
    r = max(dog.shape[0], max(e.shape[0] for e, _o in gab)) // 2
    P = np.pad(L, r, mode="edge")
    crop = lambda a: a[r:r + L.shape[0], r:r + L.shape[1]]
    cs = crop(fftconvolve(P, dog, mode="same"))
    maps = list(on_off(cs))
    for ev, od in gab:
        e, o = crop(fftconvolve(P, ev, mode="same")), crop(fftconvolve(P, od, mode="same"))
        maps.append(np.sqrt(e * e + o * o))
    M = np.stack(maps, axis=-1)
    n = L.shape[0] // CELL_F
    return M[:n * CELL_F, :n * CELL_F].reshape(n, CELL_F, n, CELL_F, BANK_MAPS).mean(axis=(1, 3))


def _template(width):
    """the born face template at one size: (the template over the ellipse, zero-mean and unit-norm, 0 outside; the ellipse's
    support; the face's pixels outside the blobs; the blobs' pixels), each rows x cols, from 4 x 4 samples in each pixel"""
    aspect = FACE_FRONT_M[0] / FACE_FRONT_M[1]
    W_, H_ = float(width), float(width) / aspect
    h = int(round(H_))
    sub = (np.arange(4) + 0.5) / 4
    ys = ((np.arange(h)[:, None] + sub[None, :]).reshape(-1) - h / 2) / H_       # y down, in H, from the centre
    xs = ((np.arange(width)[:, None] + sub[None, :]).reshape(-1) - width / 2) / W_
    X, Y = np.meshgrid(xs, -ys)                                                   # y up
    ell = (X / 0.5) ** 2 + (Y / 0.5) ** 2 <= 1.0
    blob = np.zeros_like(ell)
    for ex, ey in TEMPLATE_EYES:
        blob |= ((X - ex) ** 2 + ((Y - ey) * H_ / W_) ** 2) <= (TEMPLATE_EYE_D / 2) ** 2
    (mx, my), (mw, mh) = TEMPLATE_MOUTH
    blob |= ((X - mx) / (mw / 2)) ** 2 + ((Y - my) / (mh / 2)) ** 2 <= 1.0
    blob &= ell
    frac = lambda a: a.reshape(h, 4, width, 4).mean(axis=(1, 3))
    fe, fb = frac(ell), frac(blob)
    support = fe >= 0.5
    t = np.where(support, 1.0 - 2.0 * fb / np.maximum(fe, 1e-9), 0.0)
    t[support] -= t[support].mean()
    t /= np.sqrt((t ** 2).sum())
    return t, support.astype(float), (support & (fb == 0)).astype(float), (support & (fb >= 0.5)).astype(float)


def _kernels(width):
    """the template's four read-outs as one matrix (pixels x 4): the template, and the means over the ellipse, the face, the blobs"""
    t, sup, skin, blob = _template(width)
    return t.shape[0], float(sup.sum()), np.stack([t.reshape(-1), sup.reshape(-1) / sup.sum(), skin.reshape(-1) / skin.sum(),
                                                   blob.reshape(-1) / blob.sum()], 1)


TEMPLATES = {w_: _kernels(w_) for w_ in TEMPLATE_WIDTHS}


def face_template(img):
    """THE BORN FACE TEMPLATE on an image's pixels (rows x cols x 3 uint8, or a grey eye's rows x cols in 0-1): an instrument since
    C39's option (a) (no born face detector at birth; never in the frame): its best match over places and sizes, (r, contrast,
    width, column, row), the match's centre in the image's px (r -inf when no place has the contrast); a match is r >= TEMPLATE_R
    with contrast >= TEMPLATE_CONTRAST (`template_match`). Only the pixels are read."""
    from numpy.lib.stride_tricks import sliding_window_view
    L = img.astype(float).mean(axis=-1) / 255.0 if img.ndim == 3 else np.asarray(img, float)   # a colour render, or a grey eye's (0-1)
    best = (-np.inf, 0.0, 0, 0.0, 0.0)
    for w_, (h, n, K) in TEMPLATES.items():
        if h > L.shape[0] or w_ > L.shape[1]:
            continue
        win = sliding_window_view(L, (h, w_))
        rows, cols = win.shape[:2]
        P = win.reshape(rows * cols, h * w_)
        st, mean, m_skin, m_blob = (P @ K).T
        m2 = (P * P) @ K[:, 1]
        sd = np.sqrt(np.maximum(m2 - mean * mean, 0.0) * n)
        r = np.where(sd > 1e-6, st / np.maximum(sd, 1e-6), 0.0)
        con = (m_skin - m_blob) / np.maximum(m_skin, 1e-6)
        ok = np.where(con >= TEMPLATE_CONTRAST, r, -np.inf)
        k = int(np.argmax(ok))
        if not np.isfinite(ok[k]):
            k = int(np.argmax(r))
        cand = (float(ok[k]), float(con[k]), w_, k % cols + w_ / 2, k // cols + h / 2)
        if cand[0] > best[0] or best[2] == 0:
            best = cand
    return best


def template_match(best):
    return bool(best[0] >= TEMPLATE_R and best[1] >= TEMPLATE_CONTRAST)


def _dilate(mask, r):
    """a boolean map grown by r px in each direction (a square neighbourhood)"""
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sh = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            if dy > 0: sh[:dy] = False
            elif dy < 0: sh[dy:] = False
            if dx > 0: sh[:, :dx] = False
            elif dx < 0: sh[:, dx:] = False
            out |= sh
    return out


def periphery(L):
    """a grey eye's periphery: its native image averaged POOL x POOL (56 x 32)"""
    h, w = L.shape[0] // POOL * POOL, L.shape[1] // POOL * POOL
    return L[:h, :w].reshape(h // POOL, POOL, w // POOL, POOL).mean(axis=(1, 3))


def colour_window(img, gaze, m, d):
    """THE COLOUR WINDOW (3.4): the fovea's 21 deg at the left eye's gaze direction in the colour camera's image (COL_WIN px square),
    what lies outside its field reading nothing (mid-grey: no opponent signal) -> (COL_WIN, COL_WIN, 3) uint8"""
    cx, cy = window_centre("L", gaze)
    cl, cc = m.camera("eye_L").id, m.camera("eye_C").id
    v_cam = np.array([(cx - G.EYE_W / 2) / W.EYE_F_PX, -(cy - G.EYE_H / 2) / W.EYE_F_PX, -1.0])     # the ray, left camera's frame
    v = d.cam_xmat[cl].reshape(3, 3) @ v_cam                                                        # in the world
    u = d.cam_xmat[cc].reshape(3, 3).T @ v                                                          # in the colour camera's frame
    half = COL_WIN / 2
    out = np.full((COL_WIN, COL_WIN, 3), 128, np.uint8)
    if u[2] >= 0:
        return out
    px, py = G.COL_W / 2 + COL_F_PX * u[0] / -u[2], G.COL_H / 2 - COL_F_PX * u[1] / -u[2]
    x0, y0 = int(round(px - half)), int(round(py - half))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(G.COL_W, x0 + COL_WIN), min(G.COL_H, y0 + COL_WIN)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return out


def window_centre(side, gaze):
    """the fovea window's centre (column, row) in eye `side`'s native image for a gaze (yaw, pitch, vergence), in px"""
    yaw = gaze[0] + (gaze[2] / 2 if side == "L" else -gaze[2] / 2)
    return G.EYE_W / 2 + W.EYE_F_PX * math.tan(yaw), G.EYE_H / 2 - W.EYE_F_PX * math.tan(gaze[1])


# THE LOOKS FOR THE GROUNDING ORGAN (A202; body/core/grounding.py): the G1's appearance is its colour, as the colour window and the
# colour periphery code it (the two opponent axes, ON and OFF: 4 numbers), contrast-coded against its surround
GROUND_CENTRE = 2       # the fovea's colour cells counted as the thing looked at: the central 4 x 4 of 8 x 8 (ours: the fovea's inner
                        # 10 deg, where a toy at arm's length fills it)
GROUND_K = 4            # the look's numbers: red-green ON, OFF, blue-yellow ON, OFF
GROUND_FLOOR_WIN = 0.02 # A202h: the fovea's look above this is the look (body/core/grounding.GROUND_FLOOR's number; ours)
GROUND_SAL_FLOOR = 0.06 # A202h: the least colour contrast of a periphery cell against the scene to count as the thing in view (ours:
                        # the beige room's cells read 0.02 to 0.04 against their mean)


GROUND_FIG = 0.02       # A202m: the least departure of a fovea window cell from the window's mean for the window to hold a figure (ours)


def _local_contrast(c):
    """A202m: each cell's opponent-code distance from the mean of its 4-neighbours (the edge cells' from those they have) -> [rows*cols]"""
    rows, cols, k = c.shape
    out = np.zeros(rows * cols)
    for r in range(rows):
        for q in range(cols):
            nb = [c[r2, q2] for r2, q2 in ((r - 1, q), (r + 1, q), (r, q - 1), (r, q + 1)) if 0 <= r2 < rows and 0 <= q2 < cols]
            out[r * cols + q] = float(np.linalg.norm(c[r, q] - np.mean(nb, axis=0)))
    return out


def ground_appearance(eye_f, eye_p=None):
    """the look of what the fovea holds: the colour window's cells' mean opponent code less the scene's (the colour periphery's cells'
    mean; A202g: the whole 21-degree window against the room, so a toy anywhere in the window is the look, where the central 4 x 4
    against the window's own ring read a toy held 15 degrees off as its colour's negative); near zero on the empty floor -> [4].
    Without the periphery (a test's), the window's centre against its ring as before"""
    n = W.FOVEA_PX // CELL_F
    c = np.asarray(eye_f, dtype=np.float64)[-n * n * GROUND_K:].reshape(n, n, GROUND_K)
    if eye_p is not None:
        rows, cols = COL_CELLS
        cells_ = np.asarray(eye_p, dtype=np.float64)[-rows * cols * GROUND_K:].reshape(-1, GROUND_K)
        scene = cells_.mean(axis=0)
        win = c.reshape(-1, GROUND_K).mean(axis=0) - scene
        # A202h (2026-10-07): THE LOOK IS WHAT STANDS OUT IN VIEW WHEN THE FOVEA HOLDS LESS. Day 108's copies: with her shows brought before
        # its eyes (C304-C306) the toy still lay 30 to 60 degrees from the fovea's line (the eyes' own acts wander; the fovea's 21-degree
        # window is narrow), the fovea's look stayed below the floor and nothing bound. An infant's attention is captured by the salient
        # colourful thing held before it whether or not its fovea has arrived (exogenous attention: Posner 1980; the saliency map: Itti and
        # Koch 2001), and a word heard then binds to it. The look is the colour periphery's cell that stands out most against the scene
        # when it stands out more than the window does and past GROUND_SAL_FLOOR; else the window's look as above. The cue's cells (each
        # against the scene) share its form, so the heard word draws the eyes to the salient thing and the fovea learns to follow
        # A202m (2026-10-07): A THING IS A FIGURE AGAINST ITS GROUND. Day 110's rows: 'the' (193 hearings), 'see', 'you', 'oh', 'good' all
        # carried her sweater's blue-green, the fovea resting on her body while she spoke, and 'drum' was primed 22 times with the drum
        # before its eyes twice: a window filled edge to edge by one surface (her sweater, the wall) stood out against the room and read
        # as a look. A thing in view is a figure against its surround (centre-surround contrast: the retina's ganglion cells, Kuffler
        # 1953; figure-ground); a surface that fills the window has no figure. The window's look counts only when its cells are not
        # one surface (a cell's departure from the window's mean past GROUND_FIG), and a periphery cell stands out only against its
        # neighbours (its local contrast past GROUND_SAL_FLOOR), not merely against the room's mean; a view with neither is no thing
        # (None), and the organ neither binds nor names on it
        cw = c.reshape(-1, GROUND_K)
        fig = float(np.max(np.linalg.norm(cw - cw.mean(axis=0), axis=1)))
        if fig > GROUND_FIG and float(np.linalg.norm(win)) > GROUND_FLOOR_WIN:   # the fovea holds a figure: its look against the room
            return win                                                      # (two toys in view: the one its eyes are on, not the brighter)
        loc = _local_contrast(cells_.reshape(rows, cols, GROUND_K))
        k_ = int(np.argmax(loc))
        if float(loc[k_]) > GROUND_SAL_FLOOR:
            return cells_[k_] - scene
        return None
    a, b = GROUND_CENTRE, n - GROUND_CENTRE
    inner = c[a:b, a:b].reshape(-1, GROUND_K).mean(axis=0)
    mask = np.ones((n, n), dtype=bool); mask[a:b, a:b] = False
    outer = c[mask].mean(axis=0)
    return inner - outer


def ground_periphery(eye_p, gaze):
    """every colour periphery cell's look (its opponent code less the cells' mean) and its direction from the fovea's centre (yaw,
    pitch, rad, + right / + up: the cell's centre in the colour camera's image, whose axis is the head's, less the gaze) ->
    (feats [15, 4], dirs [15, 2])"""
    rows, cols = COL_CELLS                                               # (3, 5): retina_colour(imgs["C"], *COL_CELLS)'s rows and columns
    c = np.asarray(eye_p, dtype=np.float64)[-rows * cols * GROUND_K:].reshape(rows, cols, GROUND_K)
    feats = (c - c.reshape(-1, GROUND_K).mean(axis=0)).reshape(-1, GROUND_K)
    dirs = np.zeros((rows * cols, 2))
    for r in range(rows):
        for k in range(cols):
            x, y = (k + 0.5) * G.COL_W / cols, (r + 0.5) * G.COL_H / rows
            dirs[r * cols + k] = (math.atan((x - G.COL_W / 2) / COL_F_PX) - float(gaze[0]), math.atan((G.COL_H / 2 - y) / COL_F_PX) - float(gaze[1]))
    return feats, dirs


def window_corner(side, gaze):
    """the window's top-left pixel (column, row), inside the image"""
    cx, cy = window_centre(side, gaze)
    half = W.FOVEA_PX / 2
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    return min(G.EYE_W - W.FOVEA_PX, max(0, x0)), min(G.EYE_H - W.FOVEA_PX, max(0, y0))


def fovea(img, side, gaze):
    x0, y0 = window_corner(side, gaze)
    return img[y0:y0 + W.FOVEA_PX, x0:x0 + W.FOVEA_PX]


def project(m, d, side, point):
    """a world point in eye `side`'s image: (column, row, depth along the axis) or None behind the eye"""
    c = m.camera(f"eye_{side}").id
    v = d.cam_xmat[c].reshape(3, 3).T @ (np.asarray(point, float) - d.cam_xpos[c])
    if v[2] >= 0:
        return None
    return G.EYE_W / 2 + W.EYE_F_PX * v[0] / -v[2], G.EYE_H / 2 - W.EYE_F_PX * v[1] / -v[2], -v[2]


def gaze_at(m, d, point):
    """the gaze (yaw, pitch, vergence) that puts a world point at both windows' centres (an instrument's aim; never the body's)"""
    out = []
    for side in "LR":
        c = m.camera(f"eye_{side}").id
        v = d.cam_xmat[c].reshape(3, 3).T @ (np.asarray(point, float) - d.cam_xpos[c])
        out.append((math.atan2(v[0], -v[2]), math.atan2(v[1], -v[2])))
    (yl, pl), (yr, pr) = out
    return np.array([(yl + yr) / 2, (pl + pr) / 2, yl - yr])


def mouth_point(m, d):
    """the parent's mouth point and her face's forward axis and centre, in the world (world truth: the face test's)"""
    b = m.body("parent_head").id
    R = d.xmat[b].reshape(3, 3)
    local = np.array([kin.head_surface_x(0, kin.MOUTH_Z) + .0015, 0.0, kin.MOUTH_Z])
    centre = np.array([kin.head_surface_x(0, 0.15), 0.0, 0.15])
    return d.xpos[b] + R @ local, R[:, 0].copy(), d.xpos[b] + R @ centre


def mouth_geoms(m):
    """her drawn mouth's geoms (the lips, their opening, the teeth), by name, once per model"""
    key = id(m)
    if key not in _MOUTH_GEOMS or _MOUTH_GEOMS[key][0] is not m:
        _MOUTH_GEOMS[key] = (m, frozenset(g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(MOUTH_PARTS)))
    return _MOUTH_GEOMS[key][1]


def _face_front(m, d, eye, mouth, fwd, centre):
    """A1's conditions on her face from one eye but the window's: "" when it holds, else the first that failed: a ray from the eye to her
    mouth meets something first ("blocked"), the face turned past FACE_TURN_DEG ("turned away"), its front under FACE_MIN_PX ("too small")"""
    to = mouth - eye
    dist = float(np.linalg.norm(to))
    gid = np.array([-1], dtype=np.int32)
    hit = mujoco.mj_ray(m, d, eye, to / dist, EYE_GROUPS, 1, -1, gid)
    if 0 <= hit < dist - RAY_SLACK_M and int(gid[0]) not in mouth_geoms(m):   # something before the mouth (its own surface is at dist;
        return "blocked"                                                        # her own lips, met first when her face is turned, are
                                                                                # her mouth: the W1 fix 2's C2 finding)
    to_eye = eye - centre
    cos_turn = float(fwd @ to_eye / np.linalg.norm(to_eye))
    if cos_turn < math.cos(math.radians(FACE_TURN_DEG)):
        return "turned away"
    fd = float(np.linalg.norm(to_eye))
    area = math.pi / 4 * (FACE_FRONT_M[0] * W.EYE_F_PX / fd) * (FACE_FRONT_M[1] * W.EYE_F_PX / fd) * cos_turn
    if area < FACE_MIN_PX:
        return "too small"
    return ""


def face_test(m, d, gaze):
    """A1's test for each eye: {side: (passes, why)}; why names the first condition that failed ("" when it passes)"""
    mouth, fwd, centre = mouth_point(m, d)
    out = {}
    for side in "LR":
        c = m.camera(f"eye_{side}").id
        eye = d.cam_xpos[c].copy()
        p = project(m, d, side, mouth)
        if p is None:
            out[side] = (False, "behind the eye"); continue
        x0, y0 = window_corner(side, gaze)
        if not (x0 <= p[0] < x0 + W.FOVEA_PX and y0 <= p[1] < y0 + W.FOVEA_PX):
            out[side] = (False, "not in the fovea"); continue
        why = _face_front(m, d, eye, mouth, fwd, centre)
        out[side] = (not why, why)
    return out


def face_cue(m, d, gaze):
    """THE BORN FACE CUE'S STAND-IN (A157; the frame's face_periph, orienting's face cue: body/core/cord.py, anatomy's OrientCue "face"):
    [1, yaw, pitch] while the parent's face lies in either eye's image (anywhere in it: the periphery or the window) and A1's conditions
    but the window's hold from that eye (`_face_front`), yaw and pitch her mouth's direction from the fovea's centre (the gaze that would
    centre it, `gaze_at`, less the gaze; rad, + right / + up, the cue's declared sense); zeros otherwise. Inside the fovea's zone the cord
    pulls nothing (OrientCue.zone), so the cue is the newborn's orienting toward a face in the periphery (Goren, Sarty and Wu 1975; Johnson,
    Dziurawiec, Ellis and Morton 1991: CONSPEC's subcortical route), read from the world as the smile's level is (A49) until a born
    detector on the pixels works; its removal test is A49's. Never a channel of what she shows: her face's place alone."""
    mouth, fwd, centre = mouth_point(m, d)
    for side in "LR":
        c = m.camera(f"eye_{side}").id
        p = project(m, d, side, mouth)
        if p is None or not (0 <= p[0] < G.EYE_W and 0 <= p[1] < G.EYE_H):
            continue
        if _face_front(m, d, d.cam_xpos[c].copy(), mouth, fwd, centre):
            continue
        ga = gaze_at(m, d, mouth)
        return np.array([1.0, float(ga[0] - gaze[0]), float(ga[1] - gaze[1])])
    return np.zeros(3)


HAND_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")   # A208: the Dex3 palms ride these (the URDF's hand_palm_link meshes)
HAND_CUE_MPS = 0.15             # A208: a hand moving faster than this is reached with; its direction draws the eyes (ours)


def hand_cue(m, d, gaze, prev):
    """A208 (2026-10-07): THE BORN HAND CUE (the frame's hand_periph; the anatomy's OrientCue "hand", standing). Infants of two to four
    months look at their own moving hands and keep the arm where they can see it (White, Castle and Held 1964; van der Meer 1997), and
    reaching comes to be guided by the eye on the hand: the thing reached for enters the fovea with the hand. [1, yaw, pitch] while a
    hand of its own moves faster than HAND_CUE_MPS and lies in the left eye's image: the faster hand's direction from the fovea's centre
    (gaze_at less the gaze; rad, + right / + up); zeros otherwise. `prev` {body: last tick's position} is the eyes' memory (state).
    Read from the body's own kinematics, as the face cue is read from the world until a born detector works (A157); never a channel"""
    out = np.zeros(3); best = HAND_CUE_MPS
    for name in HAND_BODIES:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            continue
        pos = d.xpos[b].copy(); last = prev.get(name)
        prev[name] = pos
        if last is None:
            continue
        v = float(np.linalg.norm(pos - last)) / W.TICK_S
        if v <= best:
            continue
        pr = project(m, d, "L", pos)
        if pr is None or not (0 <= pr[0] < G.EYE_W and 0 <= pr[1] < G.EYE_H):
            continue
        ga = gaze_at(m, d, pos)
        out = np.array([1.0, float(ga[0] - gaze[0]), float(ga[1] - gaze[1])]); best = v
    return out


class Eyes:
    """the G1's three views over a G1World (A38, A42): `render()` the two grey imagers' and the colour camera's native images into one
    buffer with one read-back, `see()` this tick's codes (rendered again only when something they would see has moved: the state, the
    gaze, the scene's run-time fields), `close()` the GL context. Attaching sets the world's `eyes`, so its frames carry eye_p (172)
    and eye_f (1,536) at the anatomy's sizes, face_fovea (zeros: no born face detector on the pixels, C39 option a) and face_periph (A157: the born face cue read from the world, `face_cue`),
    onset_periph (the born visual onset cue, A43), and A1's face test in the truth."""

    def __init__(self, world, shadows="sun", samples=EYE_SAMPLES):
        from mujoco import gl_context
        self.world = world
        m = self.m = world.m
        self.w, self.h = G.EYE_W, G.EYE_H
        self.cw, self.ch = G.COL_W, G.COL_H
        self.W = 2 * self.w + self.cw
        self.ctx = gl_context.GLContext(max(self.W, 64), max(self.h, 64))
        self.ctx.make_current()
        room_samples = int(m.vis.quality.offsamples)
        m.vis.quality.offsamples = int(samples)                        # the eyes' multisampling, fixed in their own context
        try:
            self.con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
        finally:
            m.vis.quality.offsamples = room_samples
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        self.scn = mujoco.MjvScene(m, maxgeom=5000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        self.opt = G.eye_option()
        self.pert = mujoco.MjvPerturb()
        self.cams = {}
        for side in "LRC":
            c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FIXED; c.fixedcamid = m.camera(f"eye_{side}").id
            self.cams[side] = c
        self.buf = np.zeros((self.h, self.W, 3), np.uint8)
        self.set_shadows(shadows)
        self._cache = None
        self.timing = {"renders": 0, "render_s": 0.0, "code_s": 0.0}
        self.prev_per = None                                            # the onset cue's memory: last tick's peripheries,
        self.prev_ex = None                                             # last tick's changes past the threshold,
        self.hab = {sd: np.zeros((G.EYE_H // POOL, G.EYE_W // POOL)) for sd in "LR"}   # its habituation per place,
        self.onset = (None, np.zeros(3))                                # and this tick's reading (world tick, [fired, yaw, pitch])
        self.hand_prev = {}                                             # A208: the hands' last positions (the hand cue's speed)
        world.eyes = self

    def set_shadows(self, shadows):
        if shadows not in ("sun", "all", "none"):
            raise ValueError(f"shadows: 'sun', 'all' or 'none'; given {shadows!r}")
        self.shadows = shadows
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows != "none"
        self._cache = None

    def render(self):
        """the three views' native images {"L", "R": 192 x 336 x 3, "C": 134 x 238 x 3; uint8, row 0 at the top}, one read-back"""
        t0 = time.perf_counter()
        m, d = self.m, self.world.d
        if m.vis.headlight.active:
            raise ValueError("the eyes see by the room's lights alone: the model's headlight (a lamp at the eye) is on")
        self.ctx.make_current()
        cast = m.light_castshadow.copy()
        if self.shadows == "sun":
            for i in range(m.nlight):
                m.light_castshadow[i] = cast[i] if m.light(i).name == "sun" else 0
        try:
            for x, side, w_, h_ in ((0, "L", self.w, self.h), (self.w, "R", self.w, self.h), (2 * self.w, "C", self.cw, self.ch)):
                mujoco.mjv_updateScene(m, d, self.opt, self.pert, self.cams[side], mujoco.mjtCatBit.mjCAT_ALL, self.scn)
                mujoco.mjr_render(mujoco.MjrRect(x, 0, w_, h_), self.scn, self.con)
            mujoco.mjr_readPixels(self.buf, None, mujoco.MjrRect(0, 0, self.W, self.h), self.con)
        finally:
            m.light_castshadow[:] = cast
        img = np.flipud(self.buf)
        self.timing["renders"] += 1; self.timing["render_s"] += time.perf_counter() - t0
        return {"L": img[:, :self.w].copy(), "R": img[:, self.w:2 * self.w].copy(), "C": img[self.h - self.ch:, 2 * self.w:].copy()}

    def see(self):
        """this tick's eyes: {"eye_p": 172, "eye_f": 1,536, "face_fovea": [0], "face_periph": the born face cue's stand-in (A157, `face_cue`), "onset_periph": [0, 0, 0],
        "truth": the images, the grey foveae, the colour window and A1's face test}. eye_p: per grey eye its periphery's 7 x 4 cells
        x luminance ON and OFF (56), then the colour camera's 5 x 3 cells x red-green and blue-yellow ON and OFF (60). eye_f: per grey
        eye its fovea's 8 x 8 cells x the born bank's 10 maps (640), then the colour window's 8 x 8 cells x the two opponent axes ON
        and OFF (256). Each flattened row by row, channels last."""
        w = self.world
        m, d = self.m, w.d
        hl = m.vis.headlight
        key = hashlib.blake2b(b"".join(np.ascontiguousarray(x).tobytes() for x in (
            d.qpos, d.mocap_pos, d.mocap_quat, w.gaze, m.geom_pos, m.geom_quat, m.geom_size, m.geom_rgba, m.light_active,
            m.light_diffuse, m.light_ambient, m.light_specular, m.light_dir, m.light_pos, m.light_castshadow, m.mat_rgba,
            m.mat_emission, np.array([hl.active], float), hl.ambient, hl.diffuse, hl.specular))).digest()
        if self._cache is None or self._cache[0] != key:
            imgs = self.render()
            t0 = time.perf_counter()
            Lg = {s: grey(imgs[s]) for s in "LR"}
            per = {s: periphery(Lg[s]) for s in "LR"}
            fov = {s: fovea(Lg[s], s, w.gaze) for s in "LR"}
            cwin = colour_window(imgs["C"], w.gaze, m, d)
            n = W.FOVEA_PX // CELL_F
            eye_p = np.concatenate([retina_grey(per[s], CELL_P).reshape(-1) for s in "LR"]
                                   + [retina_colour(imgs["C"], *COL_CELLS).reshape(-1)])
            eye_f = np.concatenate([bank(fov[s]).reshape(-1) for s in "LR"] + [retina_colour(cwin, n, n).reshape(-1)])
            self.timing["code_s"] += time.perf_counter() - t0
            truth = {"images": imgs, "periphery": per, "fovea": fov, "colour_window": cwin, "face_test": face_test(self.m, w.d, w.gaze),
                     "windows": {s: window_corner(s, w.gaze) for s in "LR"}}
            self._cache = (key, eye_p, eye_f, face_cue(self.m, w.d, w.gaze), truth)      # A157: the born face cue's stand-in (the truth
        _, eye_p, eye_f, cue, truth = self._cache                                         # last: the page reads it there, sim_page.py)
        return {"eye_p": eye_p.copy(), "eye_f": eye_f.copy(), "face_fovea": np.zeros(1), "face_periph": cue.copy(),
                "onset_periph": self._onset(truth["periphery"]).copy(), "hand_periph": self._hand(), "truth": truth}

    def _hand(self):
        """A208: the born hand cue once a world tick (hand_cue; its memory of the hands' places kept in the eyes' state)"""
        w = self.world
        if getattr(self, "_hand_at", None) == w.tick:
            return self._hand_out.copy()
        self._hand_out = hand_cue(self.m, w.d, w.gaze, self.hand_prev); self._hand_at = w.tick
        return self._hand_out.copy()

    def _onset(self, per):
        """THE VISUAL ONSET CUE (A43), once a world tick: the grey peripheries' change since the last tick, each pixel's over the scene's
        mean luminance (the level the retina is adapted to), less the median change over both eyes (the whole image moving with the head), times one less the place's
        habituation; the strongest place past ONSET_WEBER fires, unless the trunk turns faster than TURN_SUPPRESS (the torso unit's
        gyro, as sensed). -> [fired, yaw, pitch]: its direction from the fovea window's centre (rad, + right, + up)"""
        w = self.world
        if self.onset[0] == w.tick:
            return self.onset[1]
        out = np.zeros(3)
        decay = math.exp(-W.TICK_S / HAB_TAU)
        for sd in "LR":
            self.hab[sd] *= decay
        if self.prev_per is not None:
            adapt = 0.5 * (np.mean([per[sd].mean() for sd in "LR"]) + np.mean([self.prev_per[sd].mean() for sd in "LR"])) + ONSET_EPS
            ch = {sd: np.abs(per[sd] - self.prev_per[sd]) / adapt for sd in "LR"}   # over the scene's mean luminance (the retina adapts)
            med = float(np.median(np.concatenate([ch["L"].ravel(), ch["R"].ravel()])))
            turn = float(np.linalg.norm(w._sensed["imu_torso"][3:6]))
            best = None
            ex = {sd: (ch[sd] - med) > ONSET_WEBER for sd in "LR"}
            for sd in "LR":
                drive = (ch[sd] - med) * (1.0 - self.hab[sd])
                if self.prev_ex is not None:                            # an appearance: nothing near it was changing last tick
                    near = _dilate(self.prev_ex[sd], ONSET_NEAR)
                    drive = np.where(near, 0.0, drive)
                k = int(np.argmax(drive))
                if best is None or drive.flat[k] > best[0]:
                    best = (float(drive.flat[k]), sd, divmod(k, drive.shape[1]))
            if best[0] > ONSET_WEBER and turn < TURN_SUPPRESS:
                _v, sd, (r, c) = best
                x, y = c * POOL + POOL / 2, r * POOL + POOL / 2
                yaw_w = w.gaze[0] + (w.gaze[2] / 2 if sd == "L" else -w.gaze[2] / 2)
                out = np.array([1.0, math.atan((x - G.EYE_W / 2) / W.EYE_F_PX) - yaw_w, math.atan((G.EYE_H / 2 - y) / W.EYE_F_PX) - w.gaze[1]])
                h = self.hab[sd]
                h[max(0, r - 1):r + 2, max(0, c - 1):c + 2] += HAB_STEP * (1.0 - h[max(0, r - 1):r + 2, max(0, c - 1):c + 2])
        self.prev_per = {sd: per[sd].copy() for sd in "LR"}
        self.prev_ex = None if self.prev_per is None or "ex" not in locals() else ex
        self.onset = (w.tick, out)
        return out

    def state(self):
        return dict(prev_ex=None if self.prev_ex is None else {sd: self.prev_ex[sd].copy() for sd in "LR"},
                    prev_per=None if self.prev_per is None else {sd: self.prev_per[sd].copy() for sd in "LR"},
                    hab={sd: self.hab[sd].copy() for sd in "LR"}, onset=[self.onset[0], self.onset[1].copy()],
                    hand_prev={k_: v_.copy() for k_, v_ in self.hand_prev.items()})   # A208

    def load_state(self, s):
        self.prev_per = None if s["prev_per"] is None else {sd: np.array(s["prev_per"][sd], dtype=np.float64) for sd in "LR"}
        self.hab = {sd: np.array(s["hab"][sd], dtype=np.float64) for sd in "LR"}
        self.prev_ex = None if s.get("prev_ex") is None else {sd: np.array(s["prev_ex"][sd], dtype=bool) for sd in "LR"}
        self.onset = (s["onset"][0], np.array(s["onset"][1], dtype=np.float64))
        self.hand_prev = {k_: np.array(v_, dtype=np.float64) for k_, v_ in (s.get("hand_prev") or {}).items()}   # A208 (older saves: none)
        self._hand_at = None
        self._cache = None

    def close(self):
        if self.world.eyes is self:
            self.world.eyes = None
        self.con.free()
        self.ctx.free()
