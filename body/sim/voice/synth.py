"""THE PARENT'S VOICE (docs/SIM_DESIGN.md 4.4 and 4.5; package P1): macOS speech through a small Swift server, 16 kHz clips with
their words' onsets and ends, and a content-addressed cache on disk whose every clip's digest is kept for good.

  SynthServer   builds synth_server.swift with swiftc (once per source digest) and runs it at nice 10, off the tick loop; one warm
                synthesizer answers one request at a time (a cache miss makes the lockstep world wait; it costs no sim time)
  VoiceCache    clip(text, register, emphasis) -> Clip. Keyed by sha256 of the request (the voice, the line as SSML with the
                register's pitch and rate, the clip format); the samples are 16 kHz int16 PCM; a size limit (300 MB) drops the least
                recently used clips. The LEDGER (ledger.jsonl) keeps every clip's digest (its samples and word marks) and is
                never trimmed: a clip synthesized again after it was dropped must equal its digest bit for bit, or the voice
                refuses it (VoiceChanged) and the life pauses at that tick, never swapping a line (the decision log's rule for
                the voice server)
  Clip          the samples as pascals at 1 m in front of the parent's mouth (for the ears' spatializer, body/sim/ears.py), each
                word with its first and last sample, and the register's level

DETERMINISM, and which part gives it. The engine is deterministic on this Mac: the same request synthesized twice, in two server
processes with the cache bypassed, gives the same float32 samples bit for bit (measured, test_sim_voice.py). That holds for one OS
build and one voice asset; an OS update may change the voice. So the cache is what makes a replay exact across time: a replay reads
the clips the life heard (each checked against its digest as it is read), and a clip made again after it was dropped is checked
against the ledger's digest (sha256 of the int16 PCM and the word marks), so a changed engine can never silently change a replayed
day. A clip's words are computed from its samples and marks as it is read, so the cache holds no derived number. The resampling
to 16 kHz (scipy's polyphase filter) and the rounding to int16 are exact functions of the engine's samples.

EVERY LINE IS SSML. The engine ignores an utterance's own rate and pitch when it is given SSML. So each line goes as SSML with
the register's pitch and rate as one prosody around it (rate r is r / 0.5 of the engine's default): a plain line so made is the
same clip, bit for bit, as the utterance at that pitch and rate (measured), SSML reaches below the utterance rate's floor, and a
focus word can be emphasized inside it (the parent spec's D3). The engine's rates come in steps (measured: utterance rates 0.15
and 0.2 are one pace, as are 0.3 and 0.35; SSML's 40 and 45%, 50 to 60%, 65 and 70%), so the "no." register's 0.3 speaks at
plain's pace: it is set apart by its pitch and its one word.

THE CONSTANTS (disclosed; section 10's "voices" row):
  voice             compact Samantha (com.apple.voice.compact.en-US.Samantha): the owner's default (B10); none of the installed
                    voices is at enhanced or premium quality
  registers         REGISTERS below: pitch and rate as the design's 4.4 table and the parent spec's D3 table (ours; Fernald 1989's
                    infant-directed contours); calling +6 dB is a level, applied here, not by the engine. Measured over 40 of the
                    birth lines (tools/sim_voice_check.py): plain 3.65 words a second over the spoken span (the design's 3.6),
                    F0 211 Hz; approval 246 Hz; comfort 3.16 words a second, 188 Hz; calling 229 Hz at +6 dB
  new word          rate 0.2 (the parent spec D3): 3.39 words a second, over the spec's target of 3; with its new word emphasized
                    2.84 (the emphasis is the conduct's to ask for: clip(..., emphasis=word))
  emphasis          EMPHASIS: SSML pitch +30%, rate 70% on the focus word (the parent spec D3; ours): measured F0 +37%, length +91%
  level             SPEECH_PA: the plain register's speech at 1 m in front of the mouth, 62 dB SPL (0.0252 Pa RMS over the sounding
                    10 ms frames): ANSI S3.5-1997's "normal" vocal effort at 1 m (62.35 dB), recalled from memory, not checked
                    against the standard. SYNTH_RMS is the engine's own RMS over those frames, 0.159 over the 331 birth template
                    lines of the all-out study (tools/sim_voice_check.py; test_sim_voice.py re-measures it on its lines)
  sample rate       16 kHz (the ears' rate, section 3.4); the engine's own 22,050 Hz resampled with scipy's polyphase filter
  word end          a word ends at its last 10 ms frame above -40 dB of the clip's loudest frame, and at the latest where the
                    next word starts
  cache             body/sim/voice/cache/ by default; 300 MB; least recently used dropped first (by the file's access mark)
"""
import hashlib
import json
import math
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
SWIFT_SRC = HERE / "synth_server.swift"
DEFAULT_CACHE = HERE / "cache"

SR = 16000
HOP = 160                                  # 10 ms frames for the word ends and the level
PARENT_VOICE = "com.apple.voice.compact.en-US.Samantha"
FORMAT = 1                                 # the clip format's version (part of the key)

# register: (pitch multiplier, rate, level in dB over the plain register)
REGISTERS = {
    "plain":    (1.15, 0.25, 0.0),
    "approval": (1.35, 0.25, 0.0),
    "comfort":  (1.05, 0.15, 0.0),
    "calling":  (1.25, 0.25, 6.0),
    "question": (1.15, 0.25, 0.0),         # the engine raises the pitch on "?" itself
    "no":       (1.00, 0.30, 0.0),         # stage 2's short, low "no." (the parent spec D3)
    "new_word": (1.15, 0.20, 0.0),         # a new word's lines, slower (the parent spec D3)
}
SPEECH_PA = 0.0252                         # 62 dB SPL re 20 uPa: plain speech at 1 m (ANSI S3.5-1997 "normal", from memory)
SYNTH_RMS = 0.159                          # the engine's RMS over the plain register's sounding frames (331 lines, measured)
PA_PER_UNIT = SPEECH_PA / SYNTH_RMS        # engine units -> pascals at 1 m
EMPHASIS = ("+30%", "70%")                 # a focus word's SSML prosody: pitch, rate (the parent spec D3; ours)
ENGINE_RATE = 0.5                          # the engine's default utterance rate (AVSpeechUtteranceDefaultSpeechRate): SSML's 100%
WORD_END_DB = -40.0
CACHE_LIMIT = 300 * 2 ** 20
NICE = 10


class VoiceChanged(RuntimeError):
    """a clip made again differs from the digest the ledger kept: the engine changed under the life."""


class SynthError(RuntimeError):
    pass


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class SynthServer:
    """the Swift server as a child process: one request at a time, a JSON line each way."""

    def __init__(self, bin_dir=None, nice=NICE):
        self.bin_dir = Path(bin_dir or DEFAULT_CACHE / "bin")
        self.nice = nice
        self.proc = None
        self.lock = threading.Lock()
        self.n = 0

    def binary(self):
        src = SWIFT_SRC.read_bytes()
        exe = self.bin_dir / f"synth_server-{_sha(src)[:12]}"
        if not exe.exists():
            self.bin_dir.mkdir(parents=True, exist_ok=True)
            tmp = exe.with_name(exe.name + f".build{os.getpid()}")
            r = subprocess.run(["swiftc", "-O", str(SWIFT_SRC), "-o", str(tmp)], capture_output=True, text=True)
            if r.returncode != 0:
                raise SynthError("swiftc failed: " + r.stderr[-2000:])
            os.replace(tmp, exe)
        return exe

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        nice = self.nice
        self.proc = subprocess.Popen([str(self.binary())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                     preexec_fn=(lambda: os.nice(nice)) if nice else None)

    def ask(self, req):
        with self.lock:
            self.start()
            self.n += 1
            req = dict(req, id=self.n)
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                self.proc = None
                raise SynthError("the voice server stopped")
            out = json.loads(line)
            if not out.get("ok"):
                raise SynthError(out.get("error", "the voice server refused"))
            return out

    def info(self, voice=PARENT_VOICE):
        return self.ask({"op": "info", "voice": voice})

    def voices(self):
        return self.ask({"op": "voices"})["voices"]

    def synth(self, text, voice=PARENT_VOICE, rate=0.25, pitch=1.15, ssml=False):
        """-> (float32 samples at the engine's rate, the rate, [(frame, loc, len)], wall seconds). The marks' text ranges are the
        engine's UTF-16 offsets into `text`: the lines are ASCII, where they are the string's own indices."""
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=".f32", prefix="synth_", dir=self.bin_dir)
        os.close(fd)
        try:
            out = self.ask({"op": "synth", "voice": voice, "rate": float(rate), "pitch": float(pitch), "volume": 1.0,
                            "text": text, "ssml": bool(ssml), "out": path})
            x = np.fromfile(path, np.float32)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        assert len(x) == out["n"], (len(x), out["n"])
        return x, float(out["sr"]), [tuple(m) for m in out["marks"]], float(out["secs"])

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None


def clip_digest(pcm, marks):
    """sha256 of a clip's int16 samples and its engine's word marks: what the ledger keeps."""
    return _sha(pcm.astype("<i2").tobytes() + json.dumps([list(m) for m in marks]).encode())


def to_16k(x, sr):
    """the engine's samples -> 16 kHz int16 (an exact function of the engine's samples)."""
    sr = int(round(sr))
    if sr != SR:
        g = math.gcd(SR, sr)
        x = resample_poly(np.asarray(x, np.float64), SR // g, sr // g)
    return np.clip(np.round(np.asarray(x, np.float64) * 32767.0), -32768, 32767).astype(np.int16)


def frame_rms(pcm):
    """the RMS of each 10 ms frame of an int16 clip, in engine units."""
    y = pcm.astype(np.float64) / 32767.0
    nf = max(1, int(math.ceil(len(y) / HOP)))
    y = np.pad(y, (0, nf * HOP - len(y)))
    return np.sqrt((y.reshape(nf, HOP) ** 2).mean(1))


def _letters(s):
    return "".join(ch for ch in s.lower() if "a" <= ch <= "z")


def words_of(pcm, sr_in, marks, text):
    """the engine's word marks -> [(word, first sample, end sample)] at 16 kHz. A mark whose text holds no letter is dropped."""
    e = frame_rms(pcm)
    thr = e.max() * 10 ** (WORD_END_DB / 20) if e.max() > 0 else np.inf
    active = e > thr
    ws = []
    for fr, loc, ln in marks:
        w = _letters(text[loc:loc + ln])
        if w:
            ws.append((w, int(round(fr * SR / sr_in))))
    out = []
    for i, (w, on) in enumerate(ws):
        nxt = ws[i + 1][1] if i + 1 < len(ws) else len(pcm)
        f0, f1 = on // HOP, max(on // HOP + 1, int(math.ceil(nxt / HOP)))
        act = np.where(active[f0:f1])[0]
        end = min((f0 + act[-1] + 1) * HOP, nxt) if len(act) else nxt
        out.append((w, on, int(min(max(end, on + 1), len(pcm)))))
    return out


@dataclass
class Clip:
    key: str
    text: str
    register: str
    pcm: np.ndarray                           # int16, 16 kHz
    words: list                               # [(word, first sample, end sample)]
    digest: str
    gain_db: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def seconds(self):
        return len(self.pcm) / SR

    def pa(self):
        """the clip as sound pressure at 1 m in front of the mouth (pascals), at the register's level."""
        return self.pcm.astype(np.float32) * np.float32(PA_PER_UNIT / 32767.0 * 10 ** (self.gain_db / 20))


def _pct(x):
    return f"{x:.4g}%"


def line_ssml(text, pitch, rate, emphasis=None):
    """a line as SSML: the register's pitch multiplier and rate as one prosody around it (the engine ignores an utterance's own
    rate and pitch when it is given SSML, and honours SSML's rate below the utterance rate's floor), and, if emphasis names one
    of its words, its last occurrence on a pitch peak and slower (infant-directed speech's focus word; Fernald and Mazzie 1991;
    the parent spec's D3). A plain line in SSML is the same clip, bit for bit, as the utterance with that pitch and rate
    (measured on the three test lines)."""
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if emphasis:
        low, w = esc.lower(), emphasis.lower()
        i = -1
        for k in range(len(low) - len(w), -1, -1):
            if low[k:k + len(w)] == w and (k == 0 or not low[k - 1].isalpha()) and \
                    (k + len(w) == len(low) or not low[k + len(w)].isalpha()):
                i = k
                break
        if i < 0:
            raise ValueError(f"{emphasis!r} is not a word of {text!r}")
        p, r = EMPHASIS
        esc = f'{esc[:i]}<prosody pitch="{p}" rate="{r}">{esc[i:i + len(w)]}</prosody>{esc[i + len(w):]}'
    return f'<speak><prosody pitch="{_pct((pitch - 1) * 100) if pitch < 1 else "+" + _pct((pitch - 1) * 100)}" ' \
           f'rate="{_pct(rate / ENGINE_RATE * 100)}">{esc}</prosody></speak>'


def request(text, register="plain", emphasis=None, voice=PARENT_VOICE):
    """-> (the key, the request, the register's level in dB)."""
    p, r, g = REGISTERS[register]
    req = {"format": FORMAT, "sr": SR, "voice": voice, "ssml": line_ssml(text, p, r, emphasis)}
    key = _sha(json.dumps(req, sort_keys=True, separators=(",", ":")).encode())
    return key, req, g


class VoiceCache:
    """the clips on disk, addressed by their request; the ledger of digests kept for good."""

    def __init__(self, root=DEFAULT_CACHE, limit=CACHE_LIMIT, server=None):
        self.root = Path(root)
        self.clips = self.root / "clips"
        self.clips.mkdir(parents=True, exist_ok=True)
        self.limit = int(limit)
        self.server = server
        self._own_server = server is None
        self.ledger_path = self.root / "ledger.jsonl"
        self.ledger = {}
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                if line.strip():
                    j = json.loads(line)
                    self.ledger[j["key"]] = j
        self.hits = self.misses = 0
        self.wall_miss = 0.0
        self.size = sum(p.stat().st_size for p in self.clips.glob("*/*"))

    def _server(self):
        if self.server is None:
            self.server = SynthServer(self.root / "bin")
        return self.server

    def _paths(self, key):
        d = self.clips / key[:2]
        return d / f"{key}.pcm", d / f"{key}.json"

    def clip(self, text, register="plain", emphasis=None, voice=PARENT_VOICE):
        """the line in a register (and, if emphasis names one of its words, with that word emphasized) -> Clip."""
        key, req, gain = request(text, register, emphasis, voice)
        pp, pj = self._paths(key)
        if pp.exists() and pj.exists():
            meta = json.loads(pj.read_text())
            pcm = np.fromfile(pp, np.int16)
            marks = [tuple(m) for m in meta["marks"]]
            if clip_digest(pcm, marks) == meta["digest"]:
                now = time.time()
                os.utime(pp, (now, now))
                self.hits += 1
                return Clip(key, text, register, pcm, words_of(pcm, meta["engine_sr"], marks, req["ssml"]), meta["digest"],
                            gain, meta)
            for q in (pp, pj):                                   # a damaged file: made again, and checked by the ledger
                self.size -= q.stat().st_size
                q.unlink()
        t0 = time.perf_counter()
        x, sr_in, marks, secs = self._server().synth(req["ssml"], voice, ENGINE_RATE, 1.0, ssml=True)
        pcm = to_16k(x, sr_in)
        digest = clip_digest(pcm, marks)
        old = self.ledger.get(key)
        if old is not None and old["digest"] != digest:
            raise VoiceChanged(f"{text!r} ({register}) synthesized again differs from the ledger's digest {old['digest'][:12]}")
        meta = dict(req, key=key, digest=digest, n=len(pcm), marks=marks, engine_sr=sr_in, engine_secs=secs)
        pp.parent.mkdir(parents=True, exist_ok=True)
        tmp = pp.with_name(pp.name + ".tmp")
        pcm.tofile(tmp)
        os.replace(tmp, pp)
        pj.write_text(json.dumps(meta))
        if old is None:
            self.ledger[key] = dict(key=key, digest=digest, n=len(pcm), text=text, register=register, voice=voice,
                                    ssml=req["ssml"])
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(self.ledger[key]) + "\n")
        self.size += pp.stat().st_size + pj.stat().st_size
        self.misses += 1
        self.wall_miss += time.perf_counter() - t0
        if self.size > self.limit:
            self.trim()
        return Clip(key, text, register, pcm, words_of(pcm, sr_in, marks, req["ssml"]), digest, gain, meta)

    def trim(self, keep=0.9):
        """drop the least recently used clips until the cache holds at most keep x its limit (the ledger stays)."""
        files = sorted(self.clips.glob("*/*.pcm"), key=lambda p: p.stat().st_mtime)
        for p in files:
            if self.size <= keep * self.limit:
                break
            j = p.with_suffix(".json")
            for q in (p, j):
                try:
                    self.size -= q.stat().st_size
                    q.unlink()
                except OSError:
                    pass

    def warm(self, lines):
        """pre-synthesize (text, register) pairs, as at a night boundary; returns the misses' count."""
        m0 = self.misses
        for text, register in lines:
            self.clip(text, register)
        return self.misses - m0

    def close(self):
        if self._own_server and self.server is not None:
            self.server.close()
            self.server = None
