"""THE FILM'S CUT (A123, 2026-09-28): the frames tools/sim_life.py --film saved (JPEGs named by tick) joined with the captions drawn
from the life's tick records, into an mp4. Nothing is burned into the frames as they are taken; the captions come from
ticks.jsonl by the frame's tick: her last line, the child's acts judged (with their worth), its token if it spoke, and the day's clock.

    python3 tools/sim_film_cut.py --frames data/g1_seed1/film --ticks data/g1_seed1/ticks.jsonl --from 1032000 --to 1034000 \\
        --fps 8 --out /tmp/clip.mp4 [--every 1] [--no-captions]

The film itself stays out of git (the film rule). Needs ffmpeg and PIL."""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

DAY_TICKS = 24000
TICK_S = 0.15


def load_rows(path, t0, t1):
    rows = {}
    with open(path, "rb") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            t = r.get("t")
            if t is None or t < t0 or t > t1 or r.get("night"):
                continue
            rows[t] = r                                                     # the last row of a tick wins (a day lived twice)
    return rows


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def caption(im, t, rows, last_line, day0):
    W, H = im.size
    d = ImageDraw.Draw(im, "RGBA")
    r = rows.get(t, {})
    day = r.get("day", day0)
    life_s = (t - day * DAY_TICKS) * TICK_S
    clock = f"day {day}  {int(life_s // 60):02d}:{int(life_s % 60):02d}"
    big, small = font(22), font(16)
    d.rectangle([0, H - 64, W, H], fill=(0, 0, 0, 150))
    d.text((12, H - 58), clock, font=small, fill=(230, 230, 230, 255))
    if last_line:
        d.text((12, H - 34), f"she: {last_line}", font=big, fill=(255, 235, 170, 255))
    judged = [x for x in r.get("judged", []) if x]
    if judged:
        txt = "  ".join(f"{x[1]} {x[2] or ''} +{float(x[0]):.1f}".replace("  ", " ") for x in judged[:3])
        tw = d.textlength(txt, font=big)
        d.rectangle([W - tw - 24, 10, W - 6, 44], fill=(20, 120, 40, 190))
        d.text((W - tw - 16, 16), txt, font=big, fill=(255, 255, 255, 255))
    tok = r.get("token")
    if tok and tok not in ("<rest>", "<end>", "<space>"):
        d.text((12, 14), f"it: {tok}", font=big, fill=(170, 220, 255, 255))
    if r.get("pain"):
        d.ellipse([W - 30, H - 90, W - 12, H - 72], fill=(220, 60, 60, 230))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--ticks", required=True, help="the life's ticks.jsonl")
    ap.add_argument("--from", dest="t0", type=int, required=True)
    ap.add_argument("--to", dest="t1", type=int, required=True)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-captions", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.frames, "*.jpg")))
    files = [f for f in files if a.t0 <= int(os.path.basename(f)[:-4]) <= a.t1][::max(1, a.every)]
    if not files:
        sys.exit(f"no frames in {a.frames} between {a.t0} and {a.t1}")
    rows = load_rows(a.ticks, a.t0 - 200, a.t1)
    day0 = next((r["day"] for r in rows.values()), 0)
    last_line = None
    for t in sorted(rows):
        if t < a.t0 and rows[t].get("line"):
            last_line = rows[t]["line"]
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(files):
            t = int(os.path.basename(f)[:-4])
            for tt in range(max(a.t0, t - a.every * 3 + 1), t + 1):
                if rows.get(tt, {}).get("line"):
                    last_line = rows[tt]["line"]
            im = Image.open(f).convert("RGB")
            if not a.no_captions:
                im = caption(im, t, rows, last_line, day0)
            im.save(os.path.join(tmp, f"{i:06d}.png"))
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "%06d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out]
        subprocess.run(cmd, check=True)
    print(f"{len(files)} frames ({a.t0} to {a.t1}, every {a.every}) at {a.fps} fps -> {a.out} ({os.path.getsize(a.out) // 1024} KB)")


if __name__ == "__main__":
    main()
