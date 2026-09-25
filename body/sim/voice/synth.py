"""THE PARENT'S VOICE (docs/SIM_DESIGN.md 4.4 and 4.5; package P1): macOS speech through a small Swift server, 16 kHz clips with
their words' onsets and ends, and a content-addressed cache on disk whose every clip's digest is kept for good.

  SynthServer   builds synth_server.swift with swiftc (once per source digest, into BIN_DIR, a build product outside git) and runs
                it at nice 10, off the tick loop; one warm synthesizer answers one request at a time (a cache miss makes the
                lockstep world wait; it costs no sim time). Its clips pass through a temporary file in the system's temporary
                folder, removed at once. The decision log's rule for the voice server is here: a request not answered within
                DOWN_S = 60 s raises VoiceDown (a server that dies is started again and asked again until then; one that hangs is
                killed at the deadline), and the world pauses the life at that tick; a line is never skipped or swapped
  VoiceCache    VoiceCache(root): root is the life's voice folder, saved beside the body (the design's 4.4: "saved beside the
                body, so a replay is exact"), never the source tree. clip(text, register, emphasis) -> Clip. Keyed by sha256 of
                the request (the voice, the line as SSML with the register's pitch and rate, the clip format); the samples are
                16 kHz int16 PCM. Two stores: kept/ holds every clip the life has HEARD (clip(..., heard=True), the default: what
                the parent says), never trimmed, so a replay reads the very samples the life heard; clips/ holds lines made
                ahead (warm(), as at a night boundary, heard=False), with a size limit (300 MB) that drops the least recently
                used. The LEDGER (ledger.jsonl) keeps every clip's digest (its samples and word marks) and is never trimmed.
                Every read is checked: the clip's own digest, its request (the key and the SSML it was made from), and the
                ledger's digest for the key; a clip that fails any is made again, and a clip made again must equal the ledger's
                digest bit for bit, or the voice refuses it (VoiceChanged) and the life pauses at that tick, never swapping a line
                (the decision log's rule for the voice server). A ledger whose last line was cut short (a full disk, a kill)
                drops that line on load; a line the ledger lost is restored from the stored clip's own record when the clip
                holds to it, and a damaged clip so recorded is made again held to that record's digest
  Clip          the samples as pascals at 1 m in front of the parent's mouth (for the ears' spatializer, body/sim/ears.py), each
                word with its first and last sample, and the register's level

DETERMINISM, and which part gives it. The engine is deterministic on this Mac: the same request synthesized twice, in two server
processes with the cache bypassed, gives the same float32 samples bit for bit (measured, test_sim_voice.py). That holds for one OS
build and one voice asset; an OS update may change the voice. So the life's own folder is what makes a replay exact across time: a
replay reads from kept/ the clips the life heard (each checked against its digest, its request and the ledger as it is read); the
ledger detects a changed engine (a line made again that differs from its digest: a line made ahead and dropped, or a kept clip
damaged) and pauses the life rather than let it hear a changed voice; whether to go on in the new voice is the owner's call. A
clip's words are computed from its samples and marks as it is read, so the cache holds no derived number. The resampling to 16 kHz
(scipy's polyphase filter) and the rounding to int16 are exact functions of the engine's samples. Disk: the birth's 331 lines are
13.3 MB (40 KB a line, measured); kept/ grows only by lines the life hears for the first time (Claude's fresh lines, the growth
words' lines), so its growth is bounded by what the parent says: at the design's density (4.6: at most 2,500 words a life day,
4.0 words a line, so at most about 625 lines) it grows by at most about 25 MB a life day even if every line were new, and by the
day's fresh lines in practice (VoiceCache.kept_size reports it); the design's section 9 counts it under the disk rule.

EVERY LINE IS SSML. The engine ignores an utterance's own rate and pitch when it is given SSML. So each line goes as SSML with
the register's pitch and rate as one prosody around it (rate r is r / 0.5 of the engine's default): a plain line so made is the
same clip, bit for bit, as the utterance at that pitch and rate (measured), SSML reaches below the utterance rate's floor, and a
focus word can be emphasized inside it (the parent spec's D3). The engine's rates come in steps (measured: utterance rates 0.15
and 0.2 are one pace, as are 0.3 and 0.35; SSML's 15% and below (a floor), 20%, 22 to 28%, 30 to 36%, 40 and 45%, 50 to 60%,
65 and 70%), so the "no." register's 0.3 speaks at plain's pace: it is set apart by its pitch and its one word. What the engine
does with a prosody nested in the line's (measured, 2026-09-24): its rate is read against the engine's default, not the line's
(40% inside the line's 40% changes nothing), its pitch against the line's (+30% inside raises the word's F0 by 1.30); a "." left
just outside it is read aloud as "period" (the clip is, bit for bit, the line with "period" written out), and "?" and "!" so left
get marks of their own; after an SSML break the next word's mark comes 90-210 ms before its sound. So the focus word's prosody
holds its punctuation, its rate is written as a share of the line's, no line has a break, and a mark without a letter is refused
(SpokenMark). The first build (87a4194 and before) wrote the word's rate as 70% of the default and left the punctuation outside:
on "." lines the parent said "period" after every emphasized word (the "+91% length" it reported was the word and "period"), and
on "?" and "!" lines, where 70% of the default was faster than the line's 40%, the new word came out 15% shorter.

THE CONSTANTS (disclosed; section 10's "voices" row):
  voice             compact Samantha (com.apple.voice.compact.en-US.Samantha): the owner's default (B13); none of the installed
                    voices is at enhanced or premium quality
  registers         REGISTERS below: pitch and rate as the design's 4.4 table and the parent spec's D3 table (ours; Fernald 1989's
                    infant-directed contours); calling +6 dB is a level, applied here, not by the engine. Measured over the 331
                    birth template lines, every ending (tools/sim_voice_check.py): plain 3.56 words a second over the spoken
                    span, F0 203 Hz; approval 239 Hz; comfort 3.08 words a second, 184 Hz; calling 219 Hz at +6 dB
  new word          the parent introduces a new word utterance-finally, on an exaggerated pitch peak (Fernald and Mazzie 1991:
                    mothers put the focused new word on the utterance's pitch peak, in final position, in speech to infants and
                    not consistently in speech to adults) and lengthened (Albin and Echols 1996: word- and sentence-final
                    lengthening in infant-directed speech), on every line she uses for it: the new_word register (rate 0.15; D3's
                    0.2 ran over C25 once the word stopped saying "period") with the new word emphasized, request() refusing a
                    new-word line without it. Measured over the 331 birth lines in it, the line's last word taken as the new word
                    (tools/sim_voice_check.py): on ".", "?" and "!" lines alike the word is 1.14-1.15 times as long as in the same
                    line unemphasized (1.12 at the least) and 1.32-1.33 times the plain line's; its F0 1.29-1.30 times both (1.22
                    at the least); 2.93, 2.90 and 2.51 words a second pooled over the ".", "?" and "!" lines, 2.91 over all,
                    and 2.40-2.83 over the design's variation set for a new word ("a X." / "the X!" / "you see the X?") (C25: at
                    most 3, pooled; a line of 4-6 words alone runs up to 3.9)
  emphasis          EMPHASIS: the focus word's prosody, pitch +30% on the line's and rate 0.7 x the line's (the parent spec D3's;
                    ours). In the plain register: the word 1.17 times as long, F0 1.29-1.30 times, on every ending
  level             SPEECH_PA: the plain register's speech at 1 m in front of the mouth, 62 dB SPL (0.0252 Pa RMS over the sounding
                    10 ms frames): ANSI S3.5-1997's "normal" vocal effort at 1 m (62.35 dB), recalled from memory, not checked
                    against the standard. SYNTH_RMS is the engine's own RMS over those frames, 0.159 over the 331 birth template
                    lines of the all-out study (tools/sim_voice_check.py; test_sim_voice.py re-measures it on its lines)
  sample rate       16 kHz (the ears' rate, section 3.4); the engine's own 22,050 Hz resampled with scipy's polyphase filter
  word end          a word ends at its last 10 ms frame above -40 dB of the clip's loudest frame, and at the latest where the
                    next word starts
  cache             the life's voice folder (given by the world); lines made ahead: 300 MB, least recently used dropped first
                    (by the file's access mark); heard lines kept
"""
import hashlib
import json
import math
import os
import select
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
BIN_DIR = HERE / "build"                   # the server's binary, one per source digest (a build product, gitignored)

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
    "new_word": (1.15, 0.15, 0.0),         # a new word's lines, slower (the parent spec D3's 0.2, slowed to 0.15 for C25)
}
SPEECH_PA = 0.0252                         # 62 dB SPL re 20 uPa: plain speech at 1 m (ANSI S3.5-1997 "normal", from memory)
SYNTH_RMS = 0.159                          # the engine's RMS over the plain register's sounding frames (331 lines, measured)
PA_PER_UNIT = SPEECH_PA / SYNTH_RMS        # engine units -> pascals at 1 m
EMPHASIS = ("+30%", 0.7)                   # the focus word: pitch on the line's, rate x the line's (the parent spec D3; ours)
ENGINE_RATE = 0.5                          # the engine's default utterance rate (AVSpeechUtteranceDefaultSpeechRate): SSML's 100%
WORD_END_DB = -40.0
CACHE_LIMIT = 300 * 2 ** 20
NICE = 10
DOWN_S = 60.0                              # the decision log's rule: a voice server down this long pauses the life at that tick


class VoiceChanged(RuntimeError):
    """a clip made again differs from the digest the ledger kept: the engine changed under the life."""


class SynthError(RuntimeError):
    pass


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class VoiceDown(SynthError):
    """the voice server has not answered for DOWN_S seconds (it hung, or it died and would not come back): the life pauses at this
    tick (the decision log's rule for the voice server); a line is never skipped or swapped."""


class SynthServer:
    """the Swift server as a child process: one request at a time, a JSON line each way. A request is answered within DOWN_S
    seconds or VoiceDown is raised: a server that dies is started again and asked again until then; one that hangs is killed at
    the deadline. cmd: the server's command (default: the built synth_server; a test gives its own)."""

    def __init__(self, bin_dir=None, nice=NICE, cmd=None, tmp_dir=None, down_s=None):
        self.bin_dir = Path(bin_dir or BIN_DIR)
        self.nice = nice
        self.cmd = list(cmd) if cmd else None
        self.tmp_dir = tmp_dir                    # the clips' temporary files (default: the system's temporary folder)
        self.down_s = DOWN_S if down_s is None else float(down_s)
        self.proc = None
        self.lock = threading.Lock()
        self.n = 0
        self.starts = 0
        self._buf = b""

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
        self.proc = subprocess.Popen(self.cmd or [str(self.binary())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
                                     preexec_fn=(lambda: os.nice(nice)) if nice else None)
        self._buf = b""
        self.starts += 1

    def _kill(self):
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        self.proc = None
        self._buf = b""

    def _readline(self, deadline):
        """one reply line, or None if the server closed its end; raises TimeoutError at the deadline."""
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError
            r, _, _ = select.select([fd], [], [], left)
            if not r:
                raise TimeoutError
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def ask(self, req):
        if self.cmd is None:
            self.binary()                                            # a first build (swiftc) is not the server being down
        deadline = time.monotonic() + self.down_s
        with self.lock:
            self.n += 1
            msg = (json.dumps(dict(req, id=self.n)) + "\n").encode()
            while True:
                try:
                    self.start()
                    self.proc.stdin.write(msg)
                    self.proc.stdin.flush()
                    line = self._readline(deadline)
                except TimeoutError:
                    self._kill()
                    raise VoiceDown(f"the voice server did not answer in {self.down_s:.0f} s")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    line = None
                if line is None:                                     # it died: started again and asked again, until the deadline
                    self._kill()
                    if time.monotonic() + 0.5 >= deadline:
                        raise VoiceDown(f"the voice server stopped and did not come back in {self.down_s:.0f} s")
                    time.sleep(0.5)
                    continue
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
        fd, path = tempfile.mkstemp(suffix=".f32", prefix="synth_", dir=self.tmp_dir)
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
                self.proc.stdin.write(b'{"op": "quit"}\n')
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


class SpokenMark(SynthError):
    """the engine marked a word with no letter in it: a punctuation mark it read as a word. Measured: a "." left outside the
    prosody that closes the word before it is spoken aloud as "period" (bit for bit the clip of the line with "period" written
    out), and "?" and "!" so left get marks of their own. No line of the parent's may say it, and its words' ends would be wrong."""


def words_of(pcm, sr_in, marks, text):
    """the engine's word marks -> [(word, first sample, end sample)] at 16 kHz. A mark whose text holds no letter is refused
    (SpokenMark): the parent's lines hold only words and ". ? !", so such a mark is punctuation the engine read as a word."""
    ws = []
    for fr, loc, ln in marks:
        w = _letters(text[loc:loc + ln])
        if not w:
            raise SpokenMark(f"the engine read {text[loc:loc + ln]!r} as a word in {text!r}")
        ws.append((w, int(round(fr * SR / sr_in))))
    return ends_of(pcm, ws)


def ends_of(pcm, ws):
    """[(word, first sample)] of a 16 kHz clip -> [(word, first sample, end sample)]: each word's end at its last 10 ms frame
    within 40 dB of the clip's loudest (WORD_END_DB) before the next word's first sample (words_of's rule; a formal trial's
    spliced sentence finds its words' ends by it again, lang/stimuli.splice)."""
    e = frame_rms(pcm)
    thr = e.max() * 10 ** (WORD_END_DB / 20) if e.max() > 0 else np.inf
    active = e > thr
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


def _signed(x):
    return f"{'+' if x >= 0 else ''}{x:.4g}%"


def _word_at(esc, word, text):
    """the last whole-word occurrence of word in the escaped line -> (its first index, the end of the punctuation that follows
    it: a word's own punctuation goes inside its prosody)."""
    low, w = esc.lower(), word.lower()
    i = -1
    for k in range(len(low) - len(w), -1, -1):
        if low[k:k + len(w)] == w and (k == 0 or not low[k - 1].isalpha()) and \
                (k + len(w) == len(low) or not low[k + len(w)].isalpha()):
            i = k
            break
    if i < 0:
        raise ValueError(f"{word!r} is not a word of {text!r}")
    j = i + len(w)
    while j < len(esc) and esc[j] in ".?!":
        j += 1
    return i, j


def line_ssml(text, pitch, rate, emphasis=None, shape=None):
    """a line as SSML: the register's pitch multiplier and rate as one prosody around it (the engine ignores an utterance's own
    rate and pitch when it is given SSML, and honours SSML's rate below the utterance rate's floor), and, if emphasis names one
    of its words, its last occurrence on a pitch peak and slower: infant-directed speech's focus word, a new word said
    utterance-finally on an exaggerated pitch peak (Fernald and Mazzie 1991) and lengthened (Albin and Echols 1996), the parent
    spec's D3. The focus word's prosody holds the punctuation that follows it (a "." left outside it is read aloud as "period":
    SpokenMark), and its rate is EMPHASIS's share of the line's own (the engine reads a nested prosody's rate against its
    default rate, not the enclosing one: measured, 40% inside 40% changes nothing; its pitch it reads against the enclosing
    one: +30% inside the line's raises the word's F0 by 1.30). A plain line in SSML is the same clip, bit for bit, as the
    utterance with that pitch and rate (measured on the three test lines).
    shape: (word, rate, pitch), a formal trial's test word said at its own rate (percent of the engine's default rate, the
    engine's own per-word rate: its steps, tools/sim_voice_check.py --trial) and pitch (percent over the line's), so every
    sentence a trial's draw could give runs on one timeline in her voice (docs/SIM_DESIGN.md 4.8, P3's twelfth round; its
    values from body/sim/lang/trial_lines.json); on the emphasized word it takes the place of EMPHASIS's values, on another
    word (a combination's colour) it is that word's own prosody. None: the SSML is as it was, bit for bit."""
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    marks = []                                                          # [first index, end, pitch, rate]
    if emphasis:
        i, j = _word_at(esc, emphasis, text)
        p, f = EMPHASIS
        marks.append([i, j, p, _pct(f * rate / ENGINE_RATE * 100)])
    if shape:
        w, r_pct, p_pct = shape
        i, j = _word_at(esc, w, text)
        m = next((m for m in marks if m[0] == i), None)
        if m is None:
            marks.append([i, j, _signed(float(p_pct)), _pct(float(r_pct))])
        else:
            m[2], m[3] = _signed(float(p_pct)), _pct(float(r_pct))
    for i, j, p, r in sorted(marks, reverse=True):                     # the later word first: the earlier's index holds
        esc = f'{esc[:i]}<prosody pitch="{p}" rate="{r}">{esc[i:j]}</prosody>{esc[j:]}'
    return f'<speak><prosody pitch="{_pct((pitch - 1) * 100) if pitch < 1 else "+" + _pct((pitch - 1) * 100)}" ' \
           f'rate="{_pct(rate / ENGINE_RATE * 100)}">{esc}</prosody></speak>'


def request(text, register="plain", emphasis=None, voice=PARENT_VOICE, prosody=None, shape=None):
    """-> (the key, the request, the register's level in dB). A new word's line must name its new word as emphasis: the parent
    introduces a new word on its pitch peak and lengthened, on every line she uses for it (". ? !" alike; the design's 4.4 and
    C25, measured by ending in tools/sim_voice_check.py). prosody: (pitch, rate) in place of a register's, at the plain level,
    for the parent's ear's templates only (body/sim/parent_ear.py: other voices, the old child pitch; lang/consts.EAR_VOICES);
    never a line she says, whose register is always one of REGISTERS. shape: a formal trial's test word's own rate and pitch
    (line_ssml)."""
    if register == "new_word" and not emphasis:
        raise ValueError(f"a new word's line must emphasize its new word: {text!r}")
    p, r, g = REGISTERS[register] if prosody is None else (float(prosody[0]), float(prosody[1]), 0.0)
    req = {"format": FORMAT, "sr": SR, "voice": voice, "ssml": line_ssml(text, p, r, emphasis, shape)}
    key = _sha(json.dumps(req, sort_keys=True, separators=(",", ":")).encode())
    return key, req, g


class VoiceCache:
    """the life's clips on disk, addressed by their request: kept/ (heard, never trimmed) and clips/ (made ahead, 300 MB, least
    recently used dropped); the ledger of digests kept for good."""

    def __init__(self, root, limit=CACHE_LIMIT, server=None):
        self.root = Path(root)
        self.clips = self.root / "clips"
        self.kept = self.root / "kept"
        self.clips.mkdir(parents=True, exist_ok=True)
        self.kept.mkdir(parents=True, exist_ok=True)
        self.limit = int(limit)
        self.server = server
        self._own_server = server is None
        self.ledger_path = self.root / "ledger.jsonl"
        self.ledger = {}
        self.recovered = 0                        # ledger lines restored from a stored clip's own record (a lost or cut ledger)
        if self.ledger_path.exists():
            raw = self.ledger_path.read_bytes()
            done, nl, tail = raw.rpartition(b"\n")
            if tail.strip():                      # the last line has no newline: an append cut short (a full disk, a kill)
                try:
                    json.loads(tail)
                    with open(self.ledger_path, "ab") as f:
                        f.write(b"\n")            # whole, only its newline lost
                except ValueError:
                    os.truncate(self.ledger_path, len(done) + len(nl))   # a half line: dropped; its clip's record restores it
                    raw = done + nl
            for line in raw.decode().splitlines():
                if line.strip():
                    j = json.loads(line)          # a damaged line before the last is not an interrupted append: it raises
                    self.ledger[j["key"]] = j
        self.hits = self.misses = self.refused = 0
        self.wall_miss = 0.0
        self.size = sum(p.stat().st_size for p in self.clips.glob("*/*"))

    @property
    def kept_size(self):
        return sum(p.stat().st_size for p in self.kept.glob("*/*"))

    def _server(self):
        if self.server is None:
            self.server = SynthServer()
        return self.server

    def _paths(self, key, store=None):
        d = (store or self.clips) / key[:2]
        return d / f"{key}.pcm", d / f"{key}.json"

    def _ledger_add(self, key, digest, n, text, register, voice, ssml):
        self.ledger[key] = dict(key=key, digest=digest, n=n, text=text, register=register, voice=voice, ssml=ssml)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(self.ledger[key]) + "\n")

    def _read(self, key, req, store, text, register):
        """a stored clip, checked: its own digest, its request (key and SSML), and the ledger's digest for its key. -> (pcm, meta)
        or None, and the digest a clip made again must equal (the ledger's; or, when the ledger has lost this line, the stored
        record's). A clip failing a check is removed (made again by the caller, and checked against that digest). A clip whose
        own record holds but whose line the ledger lost (a lost ledger, or its last line cut short) is served, and the ledger
        line is restored from it: it is what the life heard, and deleting it would let a changed engine replace it unchecked."""
        pp, pj = self._paths(key, store)
        if not (pp.exists() and pj.exists()):
            return None, None
        led, record = self.ledger.get(key), None
        try:
            meta = json.loads(pj.read_text())
            pcm = np.fromfile(pp, np.int16)
            mine = meta.get("key") == key and meta.get("ssml") == req["ssml"] and meta.get("voice") == req["voice"]
            record = meta["digest"] if mine else None
            whole = mine and clip_digest(pcm, [tuple(m) for m in meta["marks"]]) == meta["digest"]
        except (ValueError, KeyError, TypeError):
            whole = False
        if whole and led is None:
            self._ledger_add(key, meta["digest"], len(pcm), text, register, req["voice"], req["ssml"])
            self.recovered += 1
            return (pcm, meta), None
        if whole and led["digest"] == meta["digest"]:
            return (pcm, meta), None
        self.refused += 1
        for q in (pp, pj):
            if store is self.clips:
                self.size -= q.stat().st_size
            q.unlink()
        return None, (led["digest"] if led is not None else record)

    def _write(self, key, pcm, meta, store):
        pp, pj = self._paths(key, store)
        pp.parent.mkdir(parents=True, exist_ok=True)
        for q, data in ((pp, pcm.tobytes()), (pj, json.dumps(meta).encode())):
            tmp = q.with_name(q.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, q)
        if store is self.clips:
            self.size += pp.stat().st_size + pj.stat().st_size

    def clip(self, text, register="plain", emphasis=None, voice=PARENT_VOICE, heard=True, prosody=None, shape=None):
        """the line in a register (and, if emphasis names one of its words, with that word emphasized) -> Clip. heard: the
        life hears it (the parent says it): the clip is kept for good; heard=False: made ahead (warm()), trimmed by the limit.
        (The parent's ear's templates are made with heard=True: kept for good, as part of the life's fixed ear.) prosody, shape:
        see request()."""
        key, req, gain = request(text, register, emphasis, voice, prosody, shape)
        got, expect = self._read(key, req, self.kept, text, register)
        if got is None:
            got, ahead = self._read(key, req, self.clips, text, register)
            expect = expect or ahead
            if got is not None:
                pp, _ = self._paths(key, self.clips)
                now = time.time()
                os.utime(pp, (now, now))
                if heard:                                            # heard now: moved to kept/, never dropped
                    self._write(key, got[0], got[1], self.kept)
                    for q in self._paths(key, self.clips):
                        self.size -= q.stat().st_size
                        q.unlink()
        if got is not None:
            pcm, meta = got
            self.hits += 1
            return Clip(key, text, register, pcm, words_of(pcm, meta["engine_sr"], [tuple(m) for m in meta["marks"]],
                                                           req["ssml"]), meta["digest"], gain, meta)
        t0 = time.perf_counter()
        x, sr_in, marks, secs = self._server().synth(req["ssml"], voice, ENGINE_RATE, 1.0, ssml=True)
        pcm = to_16k(x, sr_in)
        digest = clip_digest(pcm, marks)
        old = self.ledger.get(key)
        ref = old["digest"] if old is not None else expect
        if ref is not None and ref != digest:
            raise VoiceChanged(f"{text!r} ({register}) synthesized again differs from the digest kept for it {ref[:12]}")
        words = words_of(pcm, sr_in, marks, req["ssml"])                # SpokenMark before anything is kept
        meta = dict(req, key=key, digest=digest, n=len(pcm), marks=marks, engine_sr=sr_in, engine_secs=secs)
        if old is None:
            self._ledger_add(key, digest, len(pcm), text, register, voice, req["ssml"])
        self._write(key, pcm, meta, self.kept if heard else self.clips)
        self.misses += 1
        self.wall_miss += time.perf_counter() - t0
        if self.size > self.limit:
            self.trim()
        return Clip(key, text, register, pcm, words, digest, gain, meta)

    def trim(self, keep=0.9):
        """drop the least recently used lines made ahead until they take at most keep x the limit (kept/ and the ledger stay)."""
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
        """make (text, register) pairs, or (text, register, emphasis) triples (body/sim/lang/templates.birth_lines), ahead, as at
        a night boundary (not heard: trimmed by the limit); returns the misses' count."""
        m0 = self.misses
        for ln in lines:
            self.clip(ln[0], ln[1], emphasis=ln[2] if len(ln) > 2 else None, heard=False)
        return self.misses - m0

    def close(self):
        if self._own_server and self.server is not None:
            self.server.close()
            self.server = None
