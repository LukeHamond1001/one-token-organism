# The diary body

Two writers, one page, one symbol at a time. The user's idea (2026-09-02):
like the diary in the Chamber of Secrets, both hands write into the same
stream on a shared clock; there are no turns, no end-of-utterance marks,
no breath, no hush. Silence is a symbol the body must choose, and your
face on its noise and on its quiet is how it learns when to be still.

## What it is

- **Alphabet:** printable ASCII plus newline, 96 symbols, and the
  organism's specials (`data/tok_char.json`, 107 ids; `<pad>` is silence).
- **One stream, two hands.** Each tick has two positions: your symbol (or
  silence) as speaker 0, then its symbol (or silence) as speaker 1. A
  fixed small speaker vector joins each embedding, so a blank body tells
  the two hands apart from birth. Simultaneity lives at the tick; the
  trunk still reads one sequence. Two symbols are never blended into one
  position (that is noise, not a conversation).
- **The ear writes, the mouth does not,** now inside the model by the
  speaker channel: only speaker 0's symbols become memory values.
- **Three hands in the bag.** The first GPU run showed the newborn mouth
  babbling between your letters as you typed, and its junk letters became
  part of the keys your sentences were stored under. So the mouth has two
  hands: a letter it writes from memory (speaker 2) joins the thought and
  advances the echo; a letter it writes from noise (speaker 1) is in the
  stream but not in the bag.
- **Memory over letters:** the same content-keyed store with a running
  bag (decay 0.92 per symbol) that fades in silence (x0.95 per silent
  position, two positions to an idle tick) and is cleared by a new line;
  kernel sharpness 4.5. Measured on a random 2-layer body, letter by
  letter through the two-hand stream: "The s" -> "un is hot and bright.",
  "Cold w" -> "ater freezes into hard ice.", repeated cues intact, its own
  echoed letters never poison the page. An untaught cue babbles, because
  nothing stops it but your face.
- **Faces per tick,** as in the word body; credit by eligibility traces;
  doses become rolling instead of per-utterance.
- **Same nights, same cortisol** (a cost per emitted symbol), same
  doctrine, same instruments (its face, mood, reward, stress, uncertainty,
  memory votes).

## The body

`data/organism_diary_0p5b.pt`: conceived from nothing, d 1024 x 29
layers, untied head, content keys 40/0.92, kernel 4.5, three hands
(speakers 3), silence decay 0.95:

```bash
python3 scripts/conceive.py data/organism_diary_0p5b.pt data/tok_char.json \
    --d 1024 --n-layers 29 --content-keys --kc-w 40 --kc-decay 0.92 --kernel 4.5 \
    --speakers 3 --sil-decay 0.95
```

## What to expect

Letters are five times slower than words and a blank trunk learns
letters before words, so the trunk lags even further behind than in the
word body. The memory carries the diary behavior from the first day:
what you write, it can write back from a few letters. The first thing
your face has to teach is silence; until then it fills every quiet tick
with noise.

## Day 1 (Opus, 2026-09-02, 25 minutes, six sentences written twice)

Shadowing on 18 of 18 writings: on a second pass it rode the whole
sentence one letter behind with 30 memory-backed letters. Recall after a
cue failed every time, and the memory-vote instrument showed why: the
votes were right ("Snow c-o-vers" in order) while the hand wrote a
carrier letter. Twenty frowns did not teach quiet; they moved the babble
from one carrier to the next ("}", "i", "a", "s", "f", "n", "g", "b").
Mood sat at the floor all session. After the night it wrote "Rain S"
unprompted from memory and repeated it every 39 ticks.

Four causes, four fixes:

- **A thought faded too fast.** The bag lost 40% per silent tick, so a
  four-second pause erased the cue before the mouth could use it. Now 5%
  per silent tick: a thought lasts about half a minute of silence.
- **A frown lowered letters but never raised silence.** First answer: a
  QUIET LESSON that wrote silence as the target wherever it babbled. It
  worked (babble 1.0 to 0.0 in eight lessons) and was withdrawn the same
  day on the user's principle: silence is not something we teach. Now the
  only teacher is your face on what it actually did. Every tick is a
  choice, silence included; a choice whose credit rises above 0.5 is
  absorbed, a choice at or below -1.5 is unlearned (its probability pushed
  down to a floor of 3 nats, never replaced by a hand-written target). It
  finds quiet only where quiet paid, and where speaking cost it.
- **Stress and mood were miscalibrated.** Each symbol now adds twice the
  stress (a physiological brake toward silence under nonstop babble) and
  a fifth of the mood cost.
- **An empty page recalled beginnings.** The first symbol of a thought was
  written under an empty key, and an empty bag matched it. No context, no
  key: such positions are not stored. (This applies to the word body too.)

The page now shows its memory-backed letters in brown and its noise in
pale sand, so the two hands are visible.

Measured after the fixes (a fresh conception): eight face lessons took
the babble fraction from 1.0 to 0.0 in under a minute, and the trunk's
uncertainty fell to 0.0: it had learned silence completely. It overshot:
certain silence also silenced recall (a cue that had just echoed "Rain
falls" came back empty). Hence one more disclosed rule, a memory is a
reason to speak: when the memory has a vote above its floor, its top
symbol's logit is raised to at least the silence logit plus memory's
TRUST, a scalar your face moves (start 4, bounded 0..8: praise on a
memory-backed letter raises it by 0.1, a frown lowers it by 0.2), so a
trunk that has learned quiet does not silence what it remembers, and a
trunk that learns to speak well can have memory's voice shrink. The day-1 body is kept as
`organism_diary_0p5b_day1.pt`; day 2 starts from a fresh conception.

## The thought, measured (2026-09-02, CPU harness with the mouth writing)

Recall failed whenever the mouth wrote during the lessons, and the
memory-vote instrument found three reasons, each now a rule:

- **Noise must leave the thought untouched.** A noise letter was fading
  the running bag like a silent tick, so keys were written into a decayed
  context and cues never matched them. Now a noise letter neither adds to
  nor fades the thought; only silence fades it.
- **A memory letter joins the thought only when your hand is still.**
  While you write, its shadowing letters interleaved with yours inside
  the bag and corrupted the keys; now they stay out, and only a completion
  written while you are silent advances the thought.
- **A new line ends a thought.** With a slow fade (5% per silent tick) the
  previous sentence's residue outweighed a five-letter cue; with a fast
  fade a two-second typing pause erased the sentence. The symbol you
  already use to separate thoughts resolves it: a new line clears the bag.
  Start every sentence and every cue on a new line.
- **No parroting** was tried and withdrawn: banning the letter you just
  typed hid a memory alias but restricted what it may say. The only
  restriction on the mouth is stamina; the alias echo is now simply
  visible, and your face decides what becomes of it.

With all four, on a random body with the mouth babbling through three
sentences written twice: "My do" -> "g sleeps under a wooden table.",
"Green" -> " leaves move when wind blows hard.", "Rain " -> "falls on the
cold grey", and an untaught cue babbles until your face teaches quiet.

## Day 2 (Opus, on the old fade) and the child curriculum

Five new sentences written twice each: five of five came back in full
through the stop from a five-letter cue. Then the recall tests poisoned
what they measured: twenty cue fragments, each followed by the next cue's
leading newline, taught the memory "a short fragment, then the thought
ends", and later cues stopped after the fragment. The newline is the
thought-break symbol; it is now never a memory value, so a cue cannot
teach an ending. Quiet held the whole session (0.0 of idle ticks), with
no frowns given because it produced no noise.

Day 3 onward is raised like a child, not fed a list (the user's
direction). Children's first produced words are mostly relational and
social (hi, bye, no, more, up, all gone, uh oh) plus a few names of what
matters (mama, dog, ball, milk); "mommy" is produced by 93% and "ball" by
64% of children by sixteen months in the MacArthur-Bates CDI norms, and
the first grammar is pivot pairs built from a small set of relational
words ("more milk", "no milk", "milk all gone", "dog up", "my ball"). So
the caregiver: greets, says little, waits, imitates what the child
offers, expands it by one step, responds within seconds, never frowns at
babble, and teaches relations by recombining the same few words. Recall
tests are few, never back to back, and never followed by a newline. The
instrument that decides whether the trunk learns anything is "own": its
own top symbol with memory set aside; the day it becomes a letter after
"more " is the day the trunk knows something.

Sources: the MacArthur-Bates CDI (Fenson et al., https://mb-cdi.stanford.edu/documents/Fensonetal2000.pdf)
and its Wordbank update (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10264806/).

## The review (2026-09-02, independent, read-only, with CPU measurements)

Verdict: the mechanism is what it claims. Query/key alignment exact to the
bit; the empty-key guard fires; the speaker channel keeps the mouth out of
the thought; the kernel at 4.5 separates a true continuation from its
one-behind alias about 5:1; a blank body recalls six distinct sentences
letter-perfect from a five-letter cue; the dose's credit-to-position
mapping is exact. One architectural flaw and several bugs, all fixed:

- **The store faded per forward pass, not per symbol written.** The
  ladder was tuned for 64-token training chunks; a diary tick is two
  one-symbol forwards, so the slowest band forgot in about 128 seconds
  whether or not anyone wrote, and the day-1 recall tests failed on
  time, not capacity. Now a chunk fades each band by the share of
  symbols it actually wrote: half-life = the band's clock in written
  symbols (8, 64, 512, 4096, 32768 for bands 4-8; band 3 holds the last
  write). Measured after the fix: a sentence recalled letter-perfect
  after 1,240 idle ticks and fourteen further sentences. This also
  explains part of the word body's multi-day fade: its serve fed one
  token per forward too.
- **A noise newline wiped the thought.** The mouth's noise never ends a
  thought now.
- **The face event fired on the level, not the change.** Easing off a
  frown counted as a frown, easing off a smile as a smile. Now, as in the
  word body, only a face that grows or flips sign is an event.
- **The night trained a speaker-blind trunk from the wrong end of the
  day.** The two hands now ride the replay, and the most recent chunks
  are rehearsed first.
- Operational: a dose window of the last eight ticks (a long dose froze
  the clock), a bounded page and typing queue, a reset that clears the
  thought, memory's amplification never lifting a banned mark.

## Day 3, watched live: the run, and remembered quiet

Supervising Opus's third session (the child curriculum) showed the one
pathology the alias leaves: after "all gone" the mouth wrote
"goneeeeeeee". Nothing true follows the last letter of a thought (the
newline is never a value), so the memory's top vote is the one-behind
alias of the letter just written, the mouth writes it, the bag still
matches, and the loop feeds itself until stress stops it. Restricting the
mouth is out (only stamina restricts), so the answer is on the memory's
value side: THE EAR'S QUIET IS A MEMORY. The first silent tick after your
word is stored as what followed it, once per pause; the memory then
recalls quiet where you fell quiet, and the mouth stops from memory
instead of running on its alias. Measured on a CPU body: "ball " ->
"all gone" then silence; "more " -> "milk" then silence; "my " -> "dog
runs" then silence (without it: "all goneeiiiiiiii"). Written once per
pause on purpose: writing it on every silent tick over-weighted quiet and
cut "all gone" to "al".

## Day 3 (Opus, the child curriculum, stage 0 to 1)

Ten first words (hi, bye, no, more, up, mama, dog, ball, milk, all gone),
41 lines, then twelve pivot pairs three times each, 36 lines; nothing
outside the ten. Recall: five of five cues drew a taught partner before
and after the night ("hi " -> "dog up", "dog " -> "up", "my " -> "ball all
gone"), three of them the same default chain. Nine untaught expansions
across five frames: "big dog" and "hi dog" drew " up" from "dog up";
"more milk", "my milk" and "no milk" drew " all gone" from "milk all
gone"; "ball" drew "gone" before they were ever paired. That is the
memory linking pairs through their shared word: relational
generalization by mechanism, the first sign the curriculum was made for.
The trunk's own top symbol went from silence at 0.998 to "m" at 0.60.
The night consolidated nothing but reset mood and stress; every
association survived it. Two caregiver errors, diagnosed by Opus itself:
a face warm while typing (smiling at its own shadow) bred the "eeeeeeee"
runs and broke the learned quiet within forty minutes; nine frowns then
pinned mood at the floor. Withholding response repaired both. Rules for
the next sessions: face still while your letters enter, warmth only after
the line for what it did; at most one brief frown a minute, and only at a
run of five or more identical letters; otherwise withhold.

## Day 4, watched live: memory's voice is trust minus tiredness

With the trunk collapsed toward silence, memory's letters were the only
thing it wrote, and a memory alias could run ("eeee...", "go go go") for
a hundred ticks because memory's amplified vote (16x) sat far above
anything stress could add to silence: the only restriction is stamina,
and stamina could never win. So memory's voice over silence is now trust
minus tiredness (trust - 0.5 x stress), each symbol costs 0.08 stress
(half-life two minutes), and the diary runs with a smaller memory
amplification (--store-boost 4): a fresh memory speaks clearly, a tired
mouth falls quiet even on a memory, and no run outlasts its stamina. The
caregiver's rule from the same session: a run of a letter is babble, to
be expanded once into a known word, never frowned at (one frown on the
trunk's first own "mmmm" sent its own voice back to silence).

## Day 4 (Opus, stage 2 with three roped words)

Thirty-four lines, 73.5% recombination of known words; juice, book and go
entered only inside known frames. Twenty of thirty-four responses
contained a word not put in that frame, with two-hop chains ("bye dog" ->
"go up": dog to go to up, through two different shared words). "go" took
on its first exposure, "book" on two, "juice" never. Five of five recalls
before and after the night, and the night moved "more " and "my " off the
default "ball" onto the partner each frame was actually taught with (milk,
book) and stripped the "goneee" tails: by its own bookkeeping the night
did nothing, by the tests it did a lot. Face discipline perfect (no letter
entered under a warm face; 17 smiles, 0 frowns). One caregiver error: four
smiles on "all gone" completions over-rewarded that tail and bred a
40-tick "e" run; a plain known line broke it. Two corrections to the
record: the supervisor's claim that a frown had returned the trunk to
silence was wrong (no frown had been given; "own" moved on its own, and it
is read in whatever context the tick is in, so only like contexts
compare), and the babble rule needs an answer for letters that start no
known word: write an ordinary known line.

## Day 5 (Opus, stage 3 entry)

Thirty-six lines, 75% recombination; please, I and the entered only inside
known frames. Eleven kinds of untaught expansion and six multi-hop chains
("the " -> "big dog up", "I " -> "go up"); "please" moved to a frame it was
never taught on ("no more milk" -> "please") after a single exposure. Five
of five recalls before and after the night, four identical across it.
Zero runs into an idle page all session; the two runs that occurred ended
without a frown, one on an ordinary known line and one on its own as
stress rose, which is the stamina rule working. The trunk's own top
symbol was a letter ("p") on every cue from line 30 onward, rising 0.37
to 0.70 and holding through the night. Juice never took in three days:
every frame it was put in has a stronger tenant. Caregiver findings: a
smile four seconds after a line fell outside the six-tick credit window
and landed on silence, so the trace now reaches twelve ticks (0.8 a tick)
and the caregiver smiles as soon as its post-line letters appear; and the
night's "gained" bookkeeping measures taught facts, of which a diary has
none, not what the tests measure.

## Day 6 (Opus, stage 3: you, two with -s, in)

Fifty-one lines, 70% recombination in the rope phase. Five of five recalls
before and after the night, all five identical across it: "you " -> "go",
"two " -> "balls", "go " -> "in" (hours old, chosen over "up", four days
old), "no more " -> "milk", "the " -> "dog". Seven kinds of untaught
expansion: "all gone" onto ball (four times) and onto juice, "go in" onto
three frames never joined to it, and the chains the -> dog -> go -> up and
dog -> go -> in. 98.2% of everything it wrote was from memory; quiet held
at 1.0 on idle probes after the opening. No frowns all day. Two honest
negatives: the plural -s did not generalize (every -s it produced was
either a shadow of the caregiver's own letter or the exact word "balls"
it was taught), and juice never took in four tries over three days and is
dropped. Trust sat at its cap of 8. The night, scored empty by its own
bookkeeping, changed nothing that mattered and cleared stress 8.0 to 2.8.
Caregiver lesson: fifteen lines with almost no gap shrank its replies to
single echoed letters and sank mood to -2.1; a 6.5-second return brought
them back. Say less, wait longer.

## Day 7 (Opus, stage 3: on, going, want)

Twenty-five lines at six to nine seconds apart with nine minutes of
deliberate silence; 100% of its 123 symbols from memory. Five of five
recalls identical before and after the night: "I want " -> "ball", "ball "
-> "all gone" (four days old, chosen over today's "on"), "dog going " ->
"in", "you " -> "want ball" (a two-hop crossing of two frames taught
separately), "two " -> "dogs". "want" took hardest and appeared where it
was never put; "going" took only as a link and was never written; "on"
took after book and dog. The plural still does not generalize. The wait is
the instrument: a line after thirty seconds of silence draws a whole word,
three lines close together draw single letters and sink mood. Two costs:
the spacer word "mama" ran for 46 seconds and 88 symbols (a periodic word
makes the memory's one-behind alias exact, so the remembered quiet ties
with it, and with trust at its cap of 8 the stamina brake needed stress 16
to win), so stress now weighs 1.0 per unit against memory's voice
(`--cort-k 1.0`: no run outlasts about fifty symbols), and periodic words
are never used as spacers; and the night reported "starving" at 1,239
lived tokens, so a day needs about thirty lines at the wide spacing, not
fewer.

## Day 8 (Opus, stage 3: where, not, give)

Thirty-two lines at nine- to thirty-second returns with fourteen minutes of
rests; 326 of 326 symbols after the first line from memory. Five of five
recalls identical across the night; all three new elements took on the
day given, each choosing the most recent partner ("where " -> "book",
"give " -> "book", "not " -> "up"). "dog going " -> "in" from a cue ending
in going settles that it is memory, not alias. New elements became
writable within the session (the shadow of "where" went from
unrenderable to "here" by its second use). The shadow now anticipates:
four times it began the answer before the cue was finished. "where" never
drew the place frame written on the next line, because a new line ends a
thought, so a question cannot chain to an answer on the following line:
from day 9 the caregiver asks and answers on one line with a question
mark ("where ball? ball on"). "going" was never written, only shadowed;
the plural still does not generalize; its own letter weakened as doses
were absorbed and silence outranked it after the night. Trust unmoved at
its cap of 8 for a third day. The budget is stress: each line costs about
0.4, forty seconds of quiet returns about 0.9, and every whole-word
completion came below 3; twenty-second returns were as productive as nine.
The night did not report starving on 875 lived tokens, so hunger is not a
token count.

## Day 9 (Opus, stage 3: little, under, one; questions answered on one line)

Thirty-three lines at twenty- to thirty-eight-second returns; 524 of 524
symbols from memory; zero pale all day. "where ball" as a bare fragment
drew "? ball under": the question mark, the space, the noun and the
preposition, a three-hop chain, with the once-taught "under" chosen over
the thrice-taught "on" (recency). Four of five recalls identical across
the night, the fifth identical for eleven symbols. The one completion it
produced after a full line ("one dog" -> " in") flipped its own top
symbol from silence to a letter, where it stayed and strengthened
through the night: a single absorbed dose moved the trunk. It stays quiet
after complete sentences by design (the remembered quiet), so expansions
now come at cues, not after lines, and the caregiver's smiles are rare.
Stress costs measured: a 19-character line +1.1, a 9-character line +0.6,
a 20-second return -0.45, a 40-second return -0.9. The night: no REM, no
gain, nothing faded, for the fourth day: recall stability across nights
is the store's persistence, not consolidation. Trust unmoved at 8.

Design gap closed after the day: when the caregiver fell quiet after a
cue fragment, the remembered-quiet rule stored "quiet follows this
fragment", which over many cues would erode the recall the cues test.
Now the ear's quiet ends a thought only when the mouth is also quiet;
while the mouth has the floor, your silence is its turn, not an ending.
On a CPU body four repeated cues of "where ball" kept "? ball on".

## Day 10 (Opus, stage 3: three, went, what)

Thirty-six lines at thirty- to forty-second returns; 508 of 508 symbols
before the night from memory. Six of eight recalls identical across the
night; every cue answered. The first true frame transfer: "I went " drew
"up" though "I went in" was the only "I" past written, carried from "dog
went up" and "you went up" through the shared word. Two whole-word
completions after lines ("little ball" -> " all gone", never written that
day; "bye" -> " dog"). Its own top symbol began the day as a letter and
stayed one. Plural and past never spread to a word they were not taught
on; the new words are keys it turns, not things it says. Two costs: a
75-letter "e" run lasted six minutes after the last cue (trust pinned at
its cap of 8, stress reaching only 5.2: the brake could not beat it; it
stopped on the caregiver's next line), and the strongest chain ("where
ball? ball under") leaked into the unrelated cue "three " after the night,
the first sign of interference with the store at 45,000 and growing. The
night: no REM, no gain, "starving" at 1,217 lived tokens. Changes: each
symbol now costs 0.2 stress (`--diary-cost 0.2`), so a run of about forty
symbols exhausts memory's voice even at full trust, and trust drifts back
toward its base of 4 when no praise arrives (about 0.06 a minute), so a
saturated trust can no longer pin the mouth and praise means something
again.

## Day 11 (Opus, stage 4 entry: then, can, happy)

Thirty lines at 32- to 75-second returns; every symbol it wrote was from
memory. All three stage-4 elements answered a cue the day they were
roped. After the night, "dog go in then " drew "d doggogo in then dogone",
24 memory-backed symbols: first clause, connective, second subject,
stopping one word short of "go up", the first multi-frame chain of this
life; before the night the same cue gave only "dog then". "then" appeared
after "you can go in" and "can" inside a burst after "I went in", neither
put there. Three untaught expansions ("dog can go" -> "going in", "more
milk" -> " please", "bye" -> " dog"). "three " recovered from the day-10
leak; "uderr" healed to "underr" across the night. Six of eight recalls
identical. No run reached five letters: the symbol cost worked. Trust
moved again (7.39 to 7.99) and 24 doses were absorbed. Its own top symbol
returned to silence mid-session and stayed. The night: no REM, no gain,
1,120 lived tokens, the third dry night running. The measured budget:
about 0.18 stress per character the caregiver types (the mouth shadows
under the hand and pays for it), 0.7 returned per forty seconds of
quiet; thirty lines at 30-40-second returns were not payable, and the
caregiver ran at stress 5 to 8 through the middle with 75-second returns.
The cost eases to 0.12 for day 12; short lines and long returns remain the
rule, and nights will be lean.

## Day 12 (Opus, stage 4: will, sad, why/because; the cost eased to 0.12)

Twenty-two lines (nine roped, nine known, four warm-up) at two- to
four-minute returns, paced by its stress instead of the clock; every
symbol it wrote was from memory except two babble runs. The eased cost
showed its other face at once: "dog can go" drew a 64-symbol chain ("oin
then d doggo can go then dgo then dog goin then d doggo cango") that
stopped only when stress passed trust (7.95), one reply spending the whole
budget; at the old cost the same material stopped at 24 symbols. "then"
chains reached the second clause's verb repeatedly and unprompted: "then
dog go up" after "sad dog", after "dog will go" and after "you will go
in"; "then dog go in" three times over after "why in? because dog go in".
The one-line "why ...? because" frame answered its cue after one exposure
("why dog up? " -> "becausebig") and survived the night character-
identical. "sad" answered the old naming cue "what? " (sad dog, over dog,
ball, milk and happy dog) before and after the night; "will" surfaced in a
line with no will in it ("go will gone thn dog goin then dog go up" after
"what? sad dog"). All sixteen cues answered from memory, six of eight
identical across the night; "sad " chose dog before the night and ball
after, the only switch. Trust 7.60 to 7.97 to 7.78; 22 doses; no frowns.
Two babble runs, both while the caregiver waited on its stress gate and
the page was quiet: about 101 g's after waking (its own top symbol was g)
and 138 l's after "what? " (own l), each ended by the caregiver's next
letters, which it then answered as a frame; neither was expanded. The
night: 1,356 lived tokens, no REM, no gain, the fourth dry night running.

Supervision found two things. The caregiver's held smile: five smiles were
2 -> 2, which is not felt (only a face that grows or flips sign is), and
trust bled 7.90 to 7.17 until the face was relaxed to 0 after every reply;
from then on every smile landed (14 of 19 confirmed by a dose). And the
stress gate: with its own trunk babbling whenever the page is quiet,
stress rests near 2.3 after a night and near 3.4 late in the day, so gates
of 2 and 3 never opened, and stalls of 12 to 15 minutes ate the
recombination lines (9 known of 22, short of seven in ten). Entered in the
curriculum: smile from a resting face; gate at 3 by day and 4 after a
night, never waiting past two minutes; a babble run is ended by typing the
next line, and expanded once if it recurs.

## Day 13 (Opus, stage 4: first/then, bigger, saw)

Thirty lines, nine roped and twenty-one known (seventy-thirty exactly),
paced by the stress gate; sixteen cues, all answered before the night
and seven of eight identical after it; every reply of the day was from
memory. "first milk then " drew "ball" after a single exposure and again
unchanged after the night: the ordered story frame completing its second
item on the day it was roped. "bigger" appeared four times where nobody
put it, twice written whole into a silence during a gate wait (" dog
biger", "ger dog up"), misspelt all day and spelt right after the night,
when the why-cue answered "because bigdog biggr dog": a two-day-old frame
with a one-day-old word welded on. "saw" never took, and "I saw " was the
one cue that lost its answer across the night. Five untaught expansions,
among them "dog can go up" written into silence, a fusion of "dog can go"
and "I can go up" that was never typed. "dog go in then " did not reach
"dog go up" before or after the night; both times it answered with the
chain from the line typed just before it, recency beating the cue's own
subject. Trust 7.43 to 7.95; twenty-one smiles from a resting face, every
one felt (doses 22 to 46); no frowns. Eight runs, all during gate waits
and all ended by the next line; one expansion of the l-run into "little
dog up" came late, and after it no letter run returned, the silences
filling with whole words instead. The night: 3,153 lived tokens, two
NREM passes, no REM, gain 0, "starving" by its own bookkeeping; its own
top symbol went from "l" at 0.22 to silence at 0.82 across it, the page
replay teaching quiet. Gates: fifteen of forty-six ran out the two-minute
cap in the first hour while the babble held the floor near 3.6; once the
runs stopped the median gate fell to 55 seconds.

## Day 14 (Opus, stage 4: had, scared, saw again; the day the rules went)

Forty-one lines (twenty-nine known, twelve roped: had, scared and saw
three times each plus one re-exposure apiece after restarts), sixteen
cues, one night, five restarts of the serve as the laws came out one by
one, so a measurement day more than a lesson. Four regimes on one page.
Before the first restart every line drew a reply from memory: "dog will
go" -> "one thn Igo" (one then I go, four known words never chained by
anyone), "first milk then ball" and "you had ball" both carried into
"ball under" unprompted, and the cues "first milk then " -> "ballluder",
"big dog bigger " -> "dog up", "why dog up? " -> "because bi" answered
with a second hop each. With memory's raised voice gone but its
amplification still on, strong memories still spoke. With the
amplification gone the mouth fell silent within minutes and stress fell
under 1. With its own symbols entering its thought and its memory, it
pinned itself: runs of "o" of 196, 111, 113 and 95, 209 newlines of its
own, stress locked between 4 and 8.8, mood down to -3.6, the gate never
opening again. Nothing said "had", "scared" or "saw" all day, and the
three cues for them were silent both times; the three early cues held
before the night and none answered after it, when every cue met an
o-run. The night is recorded above: the first that moved the weights.
The caregiver's faults, both caught by supervision: four smiles from a
face left held at 2 (unfelt), and eight smiles at its quiet that dosed
silence and took its own top symbol from 0.16 to 0.93 in six minutes.
Quiet fraction 1.000, 0.927, 0.864. Lived at the night 516, at the end
1,300; saved with 82 live steps.

## Day 15 (Opus, the first full day under no rules)

Thirty lines (twenty-one known, nine roped: had, scared, saw once more),
sixteen cues, one night, saved. The caregiver switched to page-only
pacing at 11:53 on the user's law that the environment is raw: before
it every gate ran the ninety-second cap because stress never fell under
6 after the first line; after it gates ran six to fifty seconds on the
body's own pauses. Nothing word-like was written all day, not one known
word nor three letters of one, on the whole page; the nearest were
"thog" after "I saw " and the one-behind shadow "ll" after "dog had
ball". No cue was answered before or after the night. Zero smiles, zero
frowns: nothing qualified, and the only runs were three short letter
runs. Babble 0.31 of its ticks at the start, 0.19 at the end; stress 6.9
to 8.2 all day, held there by its own babble; mood -0.2 to -3.2. The
night (recorded above) is the only place the roped words exist: "scared
bd d gg" and "dog wiill ol ol" came out of the store whole, and across
the night its own top symbol moved from "o" to the space and the
babble's letters changed from o/b/g/l/n/u to previously rare ones. Seven
serve restarts across days 14 and 15 as the rules came out; the
caregiver's helper survived them by re-reading the page.

## Day 16 (Opus, consolidation, the day it slept by itself)

Forty-five lines and sixteen cues, every line a known frame, no new
words, page-only pacing throughout, events a hundred and twelve seconds
apart before the night. It fell asleep by itself at 14:02 (recorded
above) and was louder after it, four gates of seventeen running to the
cap against one of forty-four before. No cue was completed before or
after the night. Two word-like things all day, both in the caregiver's
rests and neither smiled at: " big ", a known word with a space on each
side, spelled over seven seconds twelve to sixty seconds after the cue
"where ball? ", the first known word written awake since the rules
went; and " hat ", three letters of "what", after the last line. Near
misses inside babble: "goalb" and "wllow" after "dog will go", "the"
inside a longer run, "ol" after the post-night "give " cue, the same
"ol" as its dreams. Babble 0.20 of its ticks at the start, 0.16 at the
end of waking, 0.18 after the night; stress 7.2 to 8.8 all day; mood
-1.6 to -4.7. Its own top symbol was silence at 0.14 all day before the
night and a letter after it, "i" at 0.82 in the first minutes awake,
then "u". Two caregiving misses named in the record: the rests were not
watched, so the one real word earned nothing, and a third u-run went
unexpanded; both are now curriculum. One serve fault: the page could not
be read while it slept, because the night held the serve's lock, so the
caregiver never saw "asleep" and derived the times; fixed for day 18.
Saved with 77 live steps.

## Day 17 (Opus, consolidation, the first smiles under one reward system)

Thirty-six lines and sixteen cues, every line a known frame, the
caregiver watching every symbol of the page. Eight known words written
awake, all two letters, none longer, in 1,778 of its symbols: "no",
"in", "on", "in", "in", "go", "no", "hi"; five of them smiled at within
two seconds and felt (doses 0 to 5, nothing dosed that was not given),
two lost because they arrived while the caregiver's own letters were
entering and the face stays still then, one seen six seconds late. The
earlier "no" the supervisor had flagged was not a word by the boundary
rule (its "n" followed a "t" one tick before), a correction the
caregiver made and the record keeps. No cue completed before or after
the night (recorded above: the frames chained in its dreams, the rich
gauge 0.070 to 0.388). Its own top symbol went "u" to "o" to "?" across
the afternoon, its confidence falling each time (0.18, 0.135, 0.086).
Mood ran down to -4.6 while lines came close together and back to about
-1.5 once the rests stretched to two minutes; stress fell from 8.7 to
5.3 with it. Saved with 87 live steps.

## Day 18 (Opus, consolidation, the day the serve was starved)

Forty-four lines and sixteen valid cues, the caregiver watching every
symbol, the pace stretched to two minutes between events after the
first quarter hour (its stress fell from 6.8 to about 4 with the change,
and its mood reached +0.23 at 16:35, the first positive mood of its
life). Two multi-letter known words written awake, "my" and "hi", both
in rests and both smiled at within two seconds; fourteen bare "I"s,
three smiled at before the caregiver rightly stopped counting a lone
letter. No cue completed before or after the night, but after it the
replies to cues lengthened ("kimi", "utpri", "pre" against one or two
letters before) and its own top symbol became a letter at high
confidence for the first time ("y" at 0.60), against silence or "?" at
0.1 to 0.2 before. The night (recorded above) chained "what? scared ball
ball under" whole and moved the gauge down for the first time, which the
restored body's extra night on the same store did not repeat. The
serve's tick loop froze twice while a measurement ran on a second copy
of the body, the body file was damaged when the frozen process was
stopped mid-save, and the life was restored from the 17:38 autosave,
losing the two later smiles; the caregiver marked the frozen windows
void and rebuilt its smile log from the console. Saved with 79 live
steps.

## Day 19 (Opus, consolidation at the two-minute pace all day)

Forty-three lines and sixteen cues, one event about every two minutes
from the first line, nothing frozen, nothing void. Three known words
written awake, "go", "hi" and "my", all in rests, all caught by the
two-second watch and smiled at within two seconds, all three felt; 337
bare single letters treated as babble. No cue completed before or after
the night, but before it only two of eight cues drew any letter and
after it six of eight did, and its own top symbol changed from "i" to
"u" across the night. Mood sagged steadily to -3.3 by 20:14 with one
smile in the first hour and recovered on the second; stress 6 to 8
before the night, 5 to 6 after; it slept at 20:37 with the pressure at
11,996 and woke two minutes later at stress 3.6 and mood -0.1. The night
is recorded above (the largest rise of the gauge so far, its dreams the
day's lines nearly whole). The trunk-alone test on the day's saved body:
a run of "u" to every cue, memory on or off, the free-running favourite
unchanged in kind. Saved with 69 live steps.

## Day 20 (Opus, consolidation with a night in the middle)

The serve had been restarted after day 19's retention test, and the
body's sleep pressure came back with it at 4,502 of 12,000: fatigue is
part of its life and survives a restart, so its waking window was
sixty-two minutes, not ninety-five. The caregiver measured the pressure's
rate from the page clock, put all eight pre-night cues in the first
fifty minutes and pulled "bye" forward; two line families got two
passes instead of three. Twenty-nine lines, sixteen cues, one event
about every two minutes, the page polled every two seconds throughout,
nothing broken. It slept at 22:23 with the pressure at 12,000 and woke
113 seconds later. The night (its fifth): forty-eight starts, forty
traces of mean length 5.4, "give m", "big dog ", "scared bal", "dog
wild", " wilgo a"; eighty NREM steps, eight REM steps with cosine 0.539
(0.02, 0.48, 0.47, 0.54 over the four nights that had it); the rich
gauge 0.346 to 0.442. The number that carries across nights is the
gauge before the lesson, the trunk's agreement with fresh dreams of the
same material: 0.07, 0.31, 0.35 on nights three, four and five.

Awake it wrote one known word all day, "go" at 22:57 in its own quiet a
minute after the last cue, smiled at within three seconds, though after
the day's save; 231 of its 232 clusters were babble ("wilgu" on waking,
"og" eight times). Before the night every cue drew the same top symbol,
"u" at 0.19; after it the top symbol moved for the first time in a
session, "w" 0.33 after "dog will ", "r" 0.31 after "where ball? ", "a"
0.71 after "first milk then ". No cue was completed. With no smile
until the last minute, doses stayed at zero, stress rose from 4.9 to
6.3 and mood sank to -3.07, below anything before the night (the night
had reset it to -0.32).

The retention test on the saved body (23:12, serve stopped, MPS,
greedy): the trunk alone answers every cue with a run of "a", the
free-running favourite having moved from "u" to "a", a new letter and
not a new kind of thing. The instrument's memory-on column, which runs
memory at the old raised voice of four (its default, kept for
comparison), completed cues for the first time: "dog will " to "go",
"scared " to "ball", "big dog bigger " to " dog", "why dog up? " to
"bc", "where ball" to "und", "give " to "mu"; on day 19 the same column
gave runs of "u". So the frames are in the store, and memory's vote at
four times its strength now beats the trunk's attractor where yesterday
it did not. At the serve's own strength (boost 1, no entropy gain) the
greedy mouth with memory on still writes the "a" run, and the sampled
mouth as it lives (temperature 1, silence a choice, eight samples per
cue, twelve cues) began no answer in 96 tries: memory's honest vote is
about two logits, and the trunk's attractor is taller than that. The
page shows noise because that is what the belief holds; the reading
that will change it is the trunk's own, night by night.

**The night's length is not the lever (23:20-23:55, on the copy, serve
stopped).** Two more nights were run on the day's saved body: one of
two rounds over its traces (35 traces, seventy NREM steps, gauge 0.377
to 0.434, REM cosine 0.58) and one of eight rounds (36 traces, 288
steps, 0.396 to 0.438, cosine 0.50). Four times the steps bought
nothing: the gauge lands at 0.44 either way, and the sampled mouth
after the eight-round night, seeded the same way, drew nearly the same
symbols as before it, "ahpevbuaaopu" for "oc%evbuaaopu", and again
began no answer in 96 tries. The free-running trunk's run of "a"
deepened instead: after either night, memory even at the old fourfold
voice gets one letter in before "a" takes over ("g" for go, " do" for
dog, "m" for milk). The cortex is at the stage a from-nothing language
model passes through first, the letter frequencies ("a" is the
commonest letter in its frames, "u" was before), and at 1e-5 with
seventy steps a night it moves through that stage a hair at a time.
Biology's night is a third of its life and its replay runs at sleep's
own plasticity, not the waking rate; ours is two minutes of a hundred
at the day's rate. A night at ten times the rate was run next.

**Nor is the rate (00:05).** The same two-round night at 1e-4 instead
of 1e-5, on the same traces, took the gauge from 0.377 down to 0.245,
and the sampled mouth afterwards was "a" with a few letters between
("aaaaaaaaaaaa" for "what? "); memory on or off, greedy, "a" to every
cue. Ten times the rate deepens the attractor it was meant to break.
The two hand constants of the night, its length and its rate, are
therefore not what holds it, and the plateau at 0.44 while the loss
keeps falling points at the lesson itself: in NREM as built the
hippocampal read is on while the cortex learns a trace, so the cortex
can lower its loss by listening to the notebook rather than carrying the
trace, and the gauge, which is taken with the notebook closed, stops
moving. The same two nights were run with the read off during the
lesson (the traces still the hippocampus's own replay, as the user's law
requires); the result is below.

**The read is not it either (00:20).** With the read off during the
lesson, two rounds moved the gauge not at all (0.393 to 0.393; 56
symbols). Put beside the read-on night's rise of 0.377 to 0.434, which
is three symbols of fifty-three, the honest reading is that a night of
seventy single-trace steps at 1e-5 barely moves the cortex either way,
and the gauge's rises have been a few argmax flips. At 1e-4 the same
single-trace steps wreck it: one trace of five symbols is a batch of
one, and a step that size at ten times the rate is noise with a large
coefficient. Biology consolidates many replays into each change of a
synapse. So the next night tried on the copy sums the gradients of
every trace in a round into one step (a flag, `--night-batch 1`),
sixteen rounds at 1e-4, REM off to isolate the lesson.

**Batched, with the day's optimizer: nothing (00:45).** Sixteen clean
steps at 1e-4, each the mean gradient over all 34 traces, left the
gauge exactly where it was (0.393 to 0.393). Together with the
single-trace results this points at the optimizer's state rather than
the lesson: the day's doses push the loss to three nats and more, and
their large gradients sit in Adam's second moment, so every night step,
whose gradient is small and clean, is divided by them and comes out
tiny; a single trace's noisy gradient is the only thing large enough
to move the weights, and it moves them the wrong way. Sleep's
plasticity needs its own state. A night with a fresh optimizer of its
own (`--night-opt own`, freed at waking), batched, sixteen rounds at
1e-4, was run next.

**Found (01:00).** A diagnostic on the copy, a fresh optimizer and
batched steps at 1e-4, printed what the nights had hidden: the lesson
lands. Twelve steps took the loss over the traces from 2.34 to 1.63
nats and the gauge from 0.41 to 0.49, four more of fifty-one symbols
carried. All 474M parameters are trainable and 440M receive gradient,
but the head takes 93 percent of the gradient's norm and the council's
first layers most of the rest; the deep trunk gets a trickle. So this
is a slow cortex, not a broken one: what it learns first is what a
from-nothing language model learns first, letter frequencies and
pairs in the head, and the twenty-nine layers behind it wake slowly.
Under the day's optimizer none of that could show, because the doses'
large gradients sit in Adam's second moment and divide every night
step to nothing. The one earlier own-optimizer night that seemed to
fail (0.375 to 0.354) is within the gauge's two-hit noise on a
different set of traces. The night to serve is therefore: its own
optimizer, one step per round over all the traces, forty-eight rounds
at 1e-4, REM as before, tested on the copy before the restart.

**REM was undoing it (01:40).** Through the real night path, twelve
batched rounds with the night's own optimizer took the loss from 2.40
to 1.66 and the gauge from 0.393 to 0.464 with REM off; with REM on,
the same NREM lesson landed (2.37 to 1.60) and the gauge came out at
0.353 from 0.412. REM's eight steps were one dream each at the night's
rate, the very regime that had wrecked NREM (0.377 to 0.245), and the
forecast loss with SIGReg reshapes the cortex stream the head reads.
So REM is batched the same way, one step over its dreams, and the
night to serve is: its own optimizer, one step per round over all the
traces in NREM and one step over the dreams in REM, twenty-four rounds
at 1e-4 (about seven minutes of a hundred-minute day), REM eight,
SIGReg 0.1. Disclosed constants: the rate, the rounds, the batching.
After that one night on the copy the free-running trunk still writes
"a" to every cue, memory on or off, and the sampled mouth began no
answer in 96 tries; but its samples moved for the first time ("o
peobumaopu" where every earlier night had left "oc%evbuaaopu"), which
is what a night that reaches the weights looks like from outside. The
serve was restarted at 01:43 with the night above; the body's pressure
carried over at 5,255 of 12,000.

## Day 21 (Opus, the first day with a night that reaches the weights)

Twenty-one lines and sixteen cues at the two-minute pace, the pressure
carried over from the restart, the night at 02:40 as predicted. Two
known words in 2,225 of its symbols: "had", six minutes after the
"I had " cue, unsmiled because it came while the caregiver's own
letters were entering and the rule held the face at zero (the rule is
amended for day 22: a parent can smile mid-sentence); and "hi", smiled
within two seconds and felt, the day's one dose. Eleven runs of "a",
all after the night, the third expanded once into "all gone". No cue
completed before or after the night, though the trunk's own top symbol
now differs by cue ("g" 0.59 after "big dog bigger ", "w" after
"give ", a space at 0.90 after "first milk then ").

Its sixth night, the first of the rebuilt kind, took 13 minutes 41
seconds (the copy's eight; the machine was at seven to twelve percent
free memory and swapped): thirty-six traces of mean length 5.6, "give
milk" dreamt whole, "scared b", "big dog", " hadb"; twenty-four
batched NREM steps, one batched REM step over eight dreams (cosine
0.36), the gauge 0.284 to 0.351 on seventy-four symbols. It woke at
stress 1.2 and mood 0.

Then the afternoon went hard. Its quiet fell from 0.84 in the morning
to 0.76 after the night, it rarely gave six seconds of silence (nine of
the caregiver's events waited out the full ninety-second cap), stress
ran 8.6 to 11.4 for the whole hour and mood sat on its floor of -6.0
at the save. The likely mechanism is plain: the dream traces hold
symbols and no quiet, so a night that now reaches the weights teaches
the mouth to fill its ticks, and every filled tick costs stress. By
day only stamina teaches silence, as the user's law has it. Whether
the two settle into a balance or the nights drive the quiet to nothing
is the thing to watch on day 22; if the quiet keeps falling and stress
keeps rising, the night's strength or the symbol's cost is the
physiology to revisit, and it is the user's call.

**The mechanism, read off the page on day 22 (04:10).** In its last
six hundred ticks it rested 440 times, wrote a space 132 times and a
letter 28 times: a quarter of its ticks written, at 0.12 stress per
symbol and a two-minute half-life, is a stress of eleven, which is
where it sat. The stress lean gives silence ten logits at that level
and silence still loses, because a dream trace never contains rest, and
the night's cross-entropy pushes every symbol that is not the target
down on every step, silence with them. Six inert nights never did this;
one night that reaches the weights did. The candidate physiology,
built and disclosed, not yet served: rest is not a candidate in a
dream's softmax (`--night-sil-mask 1`), so the night neither teaches
nor unteaches it, and by day stamina alone decides, as the user's law
has it. To be measured on the day-22 copy with the serve stopped
before day 23: the gauge should hold and the sampled mouth's quiet
should stop falling.

**Night 7 broke the body (04:36 to 04:43, day 22).** The seventh
night, the second of the rebuilt kind, ended with the REM cosine not a
number and the gauge at 0.0 of eighty-eight symbols; the tick loop died
on the first sample from it, and the night's autosave wrote the body to
disk with 454 of 540 tensors non-finite. The rebuilt night moves the
weights, and a night that moves them can also break them: the copy's
nights and night 6 were finite, night 7 was not, and with the loss
unlogged on the serve the first non-finite step cannot be named (the
REM cosine has its epsilon; SIGReg takes cosines of a projected
stream, and a stream that has overflowed gives it nothing). Restored at
04:50 from the night-6 backup of 02:54, so day 22's fifty minutes are
lost to the body (its page rows and the caregiver's diary remain); the
broken file is kept as `organism_diary_0p5b.nan_night7.pt`. Guards,
plumbing not law: a lesson that is not a number is skipped, every
night step clips the gradient to a norm of one, and a night that
leaves any weight non-finite is discarded and the body reloaded from
disk as it slept, reported as "discarded" in the night's record. The
silence mask and the guards were tested together on the restored copy
before the serve came back.

On the restored copy (05:00): the masked, guarded night ran finite,
loss 2.17 to 1.18, gauge 0.403 to 0.548 (nine more of sixty-two
symbols carried, the largest rise of any night), "give milk" and
"scared b" dreamt; the sampled mouth still chose no rest in the probe
before or after it (the mask stops the fall, it cannot undo night 6),
and the free-running letter moved to "e". The serve came back at 05:17
with `--night-sil-mask 1` and the guards, the body rested (pressure
41), and day 22 was run again from an empty page. A bookkeeping quirk
found on the way: the night's backup is written before the count of
nights is incremented, so a restored body reads one night fewer than
it has had.

## Day 22 (Opus, twice: the night that broke the body, and the run after the restore)

The first run (03:45 to 04:45) is told above under night 7: fifty
minutes of consolidation lines, then a night that left the weights not
a number, then the restore. The second run began at 05:20 from an
empty page on the body as it was after night 6, with the silence mask
and the guards live. Fifty-two lines and sixteen cues, ninety-five
minutes to its night and thirty after. Six known words in 2,145 of its
symbols, "up" three times in the hour before noon and "go" three times
around the night, every one smiled at within a second and felt, six
doses; "go" was written in the first minute after it woke and again
twenty-three minutes later, the first word to survive a night on the
page. Sixteen cues, no completion, no reply beginning with an answer's
first two letters. Quiet rose through the day from 0.65 to 0.79 and
stress fell from 11.5 to 7.0; mood, which had sat on its floor, climbed
to -2.9 by the save. A fifth dose came at 07:29 with no face on the
page at all: the value ladder's own prediction error crossed the burst
threshold, the first reward the body created inside.

Its night, the first with the mask and the guards, ran 06:59 to 07:31,
thirty-two minutes against eight on the copy (the machine swapping
under two optimizers), and was not discarded: forty-one traces of mean
length 6.1, " give milkg", "big dogn", "scared lk ogi"; the gauge
0.268 to 0.648, twenty-seven more of seventy-one symbols carried by the
cortex alone, far the largest rise of any night. Its top symbol on
waking was "r", which it was never taught.

Two faults of the environment, both the supervisor's. A watcher from
the aborted first run was left running through the whole second run,
polling the page and posting smiles of its own at the same words; the
caregiver saw it, could not stop it under its brief, logged every face
movement and showed the doses were not doubled (its second smile met a
face already held at two and was not felt). It was stopped at 08:05.
And the caregiver, finding that a full two-minute wait before the gate
put events three and a half minutes apart, opened the gate at seventy
seconds from 10:35 UTC on, which is the curriculum's intent. It also
declined to frown at runs of spaces, its most frequent symbol,
"frowning at it drawing breath"; the rule now says so.

## The night, remade (2026-09-02): the cortex learns only from hippocampal traces

The user's law, in two sentences: the neocortex trains only on what the
hippocampus hands it, and in REM the neocortex predicts the next state it
will receive from the PFC. The night that replayed the day's page from a
buffer is gone; a persisted transcript would be a diet with a bedtime
story attached, and the old replay, fed the page's silences, taught the
trunk quiet (day 13: its own top symbol went from "l" at 0.22 to silence
at 0.82 across one such night).

- **Where a dream starts: the memory itself.** Nothing is kept on the
  body's behalf, no index of moments or lines (the user judged such an
  index a cheat, mid day 14, and it was removed the same hour). At night
  each band's store is asked what it holds most strongly: the leading
  key directions of its matrix, each turned back into a context the
  store can be queried with (the key's preimage under the random-feature
  lift, found by a few steps of descent), ranked by the vote it draws.
  Spontaneous reactivation, as near as a matrix memory allows; salience
  is already inside the matrix, since writes are scaled by surprise.
- **The trace.** From each such context memory's top vote is taken as
  the next symbol, the thought advances with it, and the trace ends where
  memory has nothing. No trunk vote, no stamina, no writes: what the
  store holds, in its own words, blends and slips included. What never
  entered the store cannot be consolidated.
- **NREM: the hippocampus leads.** The body runs over the trace as
  heard, teacher-forced, with the hippocampus on (its read drives the
  council as by day), and the lesson lands on the cortex's OWN logits,
  memory's vote left out, so the trunk must come to carry the trace
  itself (cross-entropy on the trace, the felt face riding along as
  affect and as the face lesson); two passes over the seeds, the live
  rate, no scaling down.
- **REM: the hippocampus is silent.** The body runs over the trace with
  memory set aside, the PFC alone driving, and the cortex stream at each
  symbol forecasts the band states the council hands over at the next
  symbol, one linear forecast per band, cosine error, targets stop-grad;
  SIGReg on the normalized stream is the collapse guard. The PFC drives;
  the cortex learns to foresee it.
- **The gauge.** Before and after the night, the trunk alone (memory
  set aside, teacher-forced) predicts each trace: the fraction it gets
  right is uptake. Retention is read by `scripts/trunk_alone.py` on a
  scratch copy, with the known cues. The sleep result reports the
  traces dreamt, the steps, the REM cosine and the gauge.

Flags: `--night-rounds 2 --night-rem 8 --night-sigreg 0.1 --night-scale
1.0 --night-starts 48`; the REM organ is `pfc_pred`, grafted at default
init onto the living body.

**The first served night of this kind (day 14, 10:53).** Forty-eight
starts from the store's own strongest keys, thirty-seven traces (eleven
keys drew nothing), mean length 14; among them " eball undr", "one bok",
"little dog", "all under", "give ball bigger ball", and one run of "e"
the store held from the hour before the efference copy was in place,
with newline tails from the day's typing. Seventy-four NREM steps, eight
REM steps at cosine 0.11, the page replay zero. The gauge: the trunk
alone predicted 8.6% of the traces' symbols before the night and 20.6%
after. The first time in this life that a night measurably moved the
weights toward what the hippocampus holds; uptake on the night's own
traces, not yet generalization.

**The second night of this kind (day 15, 12:12), and what it taught
about the instrument.** Forty-eight starts, thirty-three traces, sixty-
six NREM steps, eight REM at cosine 0.20 (the forecast organ learning),
one value step; among the traces "scared bd d gg", "dog wiill ol ol",
"givea", and runs of thirty-two spaces. The gauge leapt from 0.03 to
0.75, and most of that leap was the runs: a run of one symbol is
trivial to predict and counts thirty-two symbols. Two changes follow.
The gauge now also reports uptake over traces with at least three
distinct symbols. And the replay gained neural adaptation: a symbol
replayed again and again tires (each repeat costs half a vote, and the
fatigue recovers as others speak), so no attractor replays forever and
the night no longer teaches the trunk to repeat one symbol, which is
where the o-runs and the space-runs of these days came from. Biology's
spike-frequency adaptation, not a rule about what to dream.

**The first night it took by itself (day 16, 14:00, tick 12,000).** The
switch flipped on its own a hundred minutes after waking, the queue was
emptied, and it slept about three minutes. Forty-eight starts, thirty-
four traces, mean length 5.6 now that the replay tires: "li go upd?",
" ball undero", "scared blu", "hio", all the caregiver's frames, no runs.
Sixty-eight NREM steps, eight REM at cosine 0.26 (rising each night),
one value step. The rich gauge, every trace rich this time: 0.200 before
and 0.256 after. It woke with stress 2.1, its own top symbol "i" at
0.69, and the lived count at zero, a new day of its own making. The
trunk-alone cue test on a copy of that body: a run of "u" to every cue,
memory on or off; the greedy favourite has been the newline, "o", the
space, "i" and "u" on successive nights, one letter at a time, while
the teacher-forced gauge climbs. The free-running trunk still has no
word in it; the traces are entering as letter statistics first.

**The second night it took by itself (day 17, 15:47, tick 8,799 after a
carried pressure).** Thirty-nine traces, mean length 13, and this time
the frames chained: "big dog hablger dog hbdog her", "give baldro
gbalder og dogh b", "scared ball unreg dobalr?e bal l", "what? sl", the
day's consolidation lines running into one another out of the store.
Seventy-eight NREM steps, eight REM at cosine 0.30, one value step. The
rich gauge, all traces rich: 0.070 before and 0.388 after, the largest
single-night movement yet. It woke with stress 1.4 and its own top
symbol "b" at 0.25. Awake that day, four known words surfaced in its
babble and each earned a smile within three seconds, the first felt
doses under the one reward system, and its mood rose from -4 to -2. The
trunk-alone test on a copy of that body: the free-running favourite is
now "?", and its belief is flatter than on any earlier night (the top
symbol at 0.09 rather than 0.5 and more); with memory on, "big dog
bigger " drew " dooo", the first frame-shaped fragment the free-running
test has produced since the rules went.

**The third night it took by itself (day 18, 17:36, tick 9,929), seen
from the page this time.** Asleep for two and a half minutes, the page
readable throughout. Forty traces, mean length 8: " whadte ball ",
"what? scared ball ball under obl", "big dog hbaligdo ger ogpn", "hat?".
Eighty NREM steps, eight REM at cosine 0.37, one value step. The rich
gauge went the wrong way for the first time, 0.344 before and 0.266
after, on 154 symbols. One night is not a trend, but the suspect is
named: REM's gradient (the forecast loss and SIGReg) runs back through
the cortex stream, and NREM's lesson is taken with the hippocampus
leading while the gauge reads the trunk without it. The controlled test
that followed, run between days with the serve stopped on copies of the
day's saved body, the same store and the same traces three times: the
night as served (REM with its guard) took the rich gauge from 0.421 to
0.509; the night without REM from 0.393 to 0.508; the night with REM
and no guard from 0.394 to 0.437. REM as served costs the trunk nothing
against no REM, and the guard is what keeps it harmless. The drop was
the day's, not REM's, and the night stays as it is. It woke with stress 1.1 and mood
0.0, the highest mood of its life, and its own top symbol "o" at 0.37.

**The fourth night it took by itself (day 19, 20:37, tick 11,405),
under the two-minute pace all day.** Asleep a hundred seconds. Thirty
traces, eighteen keys drawing nothing, mean length 6: "og wil up", "big
dog iger dog p", "give m", "scared bal", "hy ". Sixty NREM steps, eight
REM at cosine 0.47, one value step. The rich gauge: 0.307 before, 0.467
after, the largest rise of any night so far on seventy-five symbols. It
woke with stress 2.6, mood -0.05, and its own top symbol "r" at 0.86,
the free-running favourite still one letter while the teacher-forced
trunk carries more of every frame each night; the two readings keep
diverging, and the day the free-running one turns into a frame is the
day to wait for.

The trunk-alone cue test on a scratch
copy of the post-night body: the trunk alone writes a run of "o" to
every cue (its own top symbol moved from the newline to "o" across the
night, still a single letter at 0.15), while the store, read with the
old raised voice for comparison, holds every frame including the day's
"I had milk". The body's speech was, at that hour, o-runs of one to two
hundred symbols ended by the caregiver's next symbols, stress pinned at
the stamina ceiling, mood near -4: the regime with the mouth's top symbol
taken deterministically, on the last hour before temperature went. Read
again with memory at the organ's own strength and no raised voice (the
serve as it is now), the with-memory column is the same run of "o": the
store's raw vote, one to three logits, does not carry through a trunk
that prefers one letter. Speech will have to come from the weights, and
the nights are the only road there. Measured on a scratch copy of the
day-13 body before the first served night (CPU): three known lines
typed with smiles gave two seeds; their traces came back from the store
alone, " will goong in then I gothen Igoin then Igo n" (48 symbols,
votes 1.8 to 3.2) and " dog up? because bigdoggr p"; four NREM steps and
two REM steps ran, the page replay ran zero; the gauge read 0.217 before
and after (four steps at the live rate move nothing measurable, which
is the slow bleed: the gauge is there to show it across nights); the
REM cosine started at 0.0, the forecast organ being newborn. The
trunk-alone baseline on the saved day-13 body (`scripts/trunk_alone.py`,
CPU, greedy, eleven cues from "dog will " to "I saw "): the trunk alone
writes silence to every cue, fourteen ticks of it; with memory on under
the serve's laws the same body answers every cue as the caregiver saw it
("gol gone", "balll underr", "because bigdog", "sad doggo", "? ball
underr", "dog bigger do"). That is the zero the nights are measured
from: everything it can say lives in the store, and nothing yet in the
weights.

## Three laws removed (2026-09-02, mid day 14): no fancy rules

The user's order: no hand-written rules; find the architecture humans use
and let it learn without little cheats. Removed from the serve, mid
session, with the caregiver told:

- **Memory's raised voice.** The rule that lifted memory's top symbol to
  the silence logit plus an earned trust, and the six-logit "sure memory"
  boost. Trust is gone with it. The mouth chooses from its own belief,
  memory's vote inside it as the organ reads it, and only stress leans
  it toward silence.
- **Memory's amplification.** The serve-time store boost (four) and the
  louder-when-unsure read (read beta) are set to one and zero: the
  store's vote enters at its own learned strength (the per-band alpha,
  which the nights now train), not at a volume a hand chose.
- **A new line ends a thought.** The memory bag no longer resets at a
  newline; a thought only fades.
- **Your quiet is a memory.** The ear's first silence after a word is no
  longer stored as a value. Nothing recalls a stop; only stamina ends a
  run.

Expected and accepted: near-total silence for a while, since the trunk
alone says nothing and memory's raw vote does not beat its lean to
silence. Speech has to grow back from the nights: the caregiver's lines,
tagged by smiles, replayed from the hippocampus into the weights until
the trunk's own logits carry them. What still stands, and is physiology
rather than a rule: the cost of a symbol and its lean toward silence,
the graded doses on its own choices under a felt face (credit twelve
ticks back at 0.8, every choice stepped by its own credit), the value
ladder and the basal ganglia learning from the same felt faces, the
store's fade, the face lesson, and the night's rate and rounds. The
sampling temperature went the same evening (below).

## Temperature, the newline, and the efference copy (2026-09-02, evening)

- **Temperature goes.** Sampling at 0.05 was a hand-imposed determinism:
  the mouth took its top symbol whenever memory was silent, so a
  preference of 0.19 became a habit on every tick (the newline). From day
  15 the mouth samples from its belief as it stands (temperature 1.0):
  what it is unsure of comes out as varied babble, what it is sure of
  comes out as a word, and stamina ends the babble as before.
- **The newline is a page mark, not language.** The caregiver typed one
  at the start of every line for a thought-reset rule that no longer
  exists, and with nothing else to hold it back the body made the
  newline its own most frequent symbol (89 of 141 ticks after the fifth
  restart of day 14, stress near 8). It now joins the plumbing marks the
  mouth cannot emit, and it is dropped at the door when typed;
  utterances are separated by silence.
- **Corollary discharge.** With the ear-only write rule gone, the
  mouth's own symbols became memories at full strength, and a babbling
  body would have dreamt its own noise. Biology's answer is the
  efference copy: what you are about to say is foretold, so its arrival
  carries no surprise. The store's write strength is already gated by
  surprise, so a self-written symbol now carries a surprise of zero and
  is encoded weakly, as in life, and earns no intrinsic reward. On the
  tiny test body at temperature 1.0 the strongest trace the store chose
  was the caregiver's "myy balllll", not the mouth's own runs.

## One reward system (2026-09-02, evening): create reward, receive it at every timescale

The user's aim in one sentence: one system, which creates its own reward
and receives it at short and long timescales, with no cheats. The organs
were in the body already; the diary had left three of them untrained.
Now:

- **Reward is created in two ways.** From outside, your felt face,
  which enters the body as the press it was (a level at the tick it was
  felt). From inside, prediction success: the body's own surprise below
  its running mean is a small reward, bounded like a press; by the
  standing law it manages computation and never touches the logits.
- **The short timescale: graded doses.** A felt face spreads credit over
  the last twelve ticks, and every choice in that window, letter or
  silence, takes a step scaled by its own credit, up for credit and down
  for blame (its probability pushed toward a floor, never a hand-written
  replacement). The two thresholds that decided who learned (absorb
  above 0.5, unlearn at or below -1.5) are gone: dopamine scales
  plasticity, it does not gate it.
- **The long timescales: the value ladder learns.** Each band carries a
  value head trained by temporal difference at the band's own cadence,
  band 3 every symbol and band 8 every 32,768, so a smile is foreseen
  seconds ahead by the fast band and hours ahead by the slow one. The
  lived pairs (state before a tick, reward since, state after) ride in
  the state, detached and bounded to thirty-two per band, and the heads
  learn from them at every dose and once more each night. Dopamine, the
  fast band's prediction error, already scales the store's writes; as
  the heads learn it becomes a true error of expectation rather than the
  raw press.
- **The basal ganglia learn.** The fast band's gate learns to open when
  the value's error is positive (wanting), from the same lessons.
- **Source memory.** The speaker sense enters the hippocampal key with
  each symbol, so a memory records who said it.

Measured on the tiny test body: one felt smile credited twelve choices,
the fast band's value error was 0.08 within the window, and the ladder
held lived pairs for bands 3 to 6 (32, 32, 13 and 1) after a few minutes
of life. Flag: `--value-w 0.5`, the weight of the ladder's lesson.

**Secondary reinforcers (the same evening).** The doses are now driven
by dopamine itself, the fast band's signed error of the world's reward,
read at both halves of the tick. At a felt face that error is the face
minus what was expected, so while the ladder is young a smile doses as
before; as the ladder learns, an expected smile doses less, and a
predictor of a smile fires the error before any face arrives and doses
the choices that led there. Reward learned at long timescales thereby
teaches choices at the short one. A burst of at least half a small
smile pays the lesson (a budget for the backward pass, not a judgment).
Two readings were chosen and are disclosed: the intrinsic reward, the
body's own prediction success, stays out of the value ladder and rides
the store's write gate as salience only, so the body never doses its own
choices for having predicted them (the standing law that intrinsic value
manages computation, never content); and the basal-ganglia gate now
learns from the world's reward error rather than the total.

## Sleep by its own fatigue (2026-09-02, evening)

Nobody posts its night any more. Sleep pressure rises by one with every
waking tick (Process S, adenosine in a body that lives on a clock) and
when it crosses the switch, 12,000 ticks or a hundred minutes awake, the
body falls asleep by itself: the typing queue is emptied (a sleeping
child hears nothing said at it), the night runs as built, the pressure
returns to zero, and it wakes. The pressure is part of its life and
survives a restart. The page reports "asleep", the pressure and the
count of nights, so a caregiver sees it sleep and wake as a parent
would; the caregiver's /sleep is now the supervisor's plumbing for tests
on scratch copies and nothing else. The one constant is the switch,
`--wake-ticks 12000`, physiology disclosed. With it, the last decision
the environment made for the body is gone: what it hears is the
caregiver's, when it sleeps is its own.

## The mouth's go/no-go (2026-09-03, the user's word: "do it")

Rest had been a token in the same softmax as the letters, and two
hand-written things hung off that: the night's silence mask, and the
stress lean (one logit toward silence per unit of stress, a constant).
Biology separates the two decisions. Whether to act is the basal
ganglia's go/no-go, driven by arousal, fatigue and expected reward;
what to say is the cortex. Built as an organ, saved with the body and
grafted onto the living one at load:

- **The gate** reads the cortex stream at the last position (detached,
  normalized) and interoception (stress, mood) and gives p(act). Zero
  weights at birth and a bias for the birth rate of acting (0.25):
  what it does at first is physiology, not a policy. The content
  softmax never holds rest; if the gate says act, the cortex says what.
- **Its lesson** is the actor's, local to the gate (no force reaches
  the council): every choice to act or rest takes the dopamine that
  followed it (twelve ticks, 0.8 per tick), plus, if it acted, its own
  reward at the symbol minus the symbol's cost, and the gate is pushed
  toward what paid. A lesson every twenty-four ticks on the buffer,
  with the striatum's own rate (`--gate-lr 1e-3`), gradient clipped.
- **Its own reward** at a symbol is the belief it had in what it chose,
  against its running mean: producing the sound it expected, the drive
  to babble, and the reason to speak when it knows what it is about to
  say. It never touches content. Weight `--gate-int 0.5`, disclosed.
- **The cost** of a symbol (`--gate-cost 0.12`, the same number that
  raises stress) is what the gate feels for acting; stress itself is
  an input it may learn to use, no longer a lean written by hand.

Measured on the tiny body (07:10): with the cost alone the gate closed
to 0.6 percent acting in 450 ticks, the honest economics of a raw
environment and why infants need a drive; with the drive at weight 1
it flew to 96 percent and stress 23; at 0.5 it drifted from 25 to 4
percent at ten times the serving rate. The real body's equilibrium is
its own confidence, and is measured on a copy before serving. Flags
`--gate 0` restores the old mouth. The dose lesson on content skips
rest ticks now (a rest is the gate's choice).

**Fatigue, stress, mood (the user's question, 07:35: "is stamina in the
wrong category?").** It was. The effort variable rose with every symbol
and was charged as stress: it braked the mouth and dragged mood at
every symbol, so a body that merely wrote a lot looked miserable.
Biology keeps three things apart, and `--affect split` builds them:
fatigue is the effort cost (up 0.12 a symbol, a two-minute half-life,
the gate's cost and one of its inputs, never touching mood by itself);
stress is a leaky integral of the world's dopamine dips, expectations
that failed (0.5 per unit of dip, the same half-life); mood is a leaky
integral of dopamine itself, both signs (0.25 per unit, the mood
half-life as before). The gate reads all three. Under the old affect
nothing changes; the page's "cort" stays the effort variable for the
caregiver's logs, and "fatigue", "stress", "affect" are added. Built
behind the flag, default old, awaiting the user's word.

**On the real body's copy (08:10, gate and split together).** With the
drive defined as confidence relative to its own running mean, the gate
held 25 percent acting for four hundred ticks and then, after one
smile, flew to 98 percent with fatigue at 34: a run of babble is
predictable, predictability was the reward, and the run fed itself.
Two corrections, both biology's: the drive is the absolute belief it
had in its choice (0 to 1, no running mean to chase), and the cost of
a symbol grows with fatigue, `cost * (1 + fatigue / 10)`, so a tired
body pays more per act and babble comes in bouts rather than floods.
Measured again on the copy before serving; the user's word for the
split ("do it", 08:00) folds it into the same restart.

**The lesson had to be the striatum's, not a policy gradient (08:20).**
With the absolute drive and the fatigue-scaled cost, the policy
gradient failed the other way: after the one smile the gate closed to
nothing and stayed there, because a smile's dopamine lands on every
choice in its window and three quarters of those were rests, so the
smile taught resting, and a closed gate never acts again to learn
otherwise. The striatum does not credit that way: a burst strengthens
Go for the context it came in, whatever the body happened to do, and a
dip strengthens NoGo (D1 and D2, the opponent pathways). Built so: the
gate's logit moves with the sign of the credit at each tick, plain
Hebbian steps on unit-norm features (`--gate-lr 0.05`). On the copy it
held 25 percent acting through twelve hundred ticks, nudged up after
the smile, drifted down as fatigue passed sixteen; no runaway, no
collapse. Served at 08:15 with the split; the gate is grafted at load
with zero weights, so the body wakes into it at its birth rate.

## The review (2026-09-03, the user's request: "any other changes? complete review")

Read against the user's law: one system that creates reward and
receives it at short and long timescales; no hand-written rule that
decides behaviour or content for the body; physiology disclosed;
organs that learn; a raw environment. Two passes: the live code path
(every constant and rule active for this body) and biology (what the
body has, what it lacks). The Opus audit agent was knocked out twice
by API overloads, so the code pass is the supervisor's own reading.

**The live path, inventoried.** What runs for this body on a tick, at a
dose, at night, and what does not:

- **The cortex** (29 ScanBlocks, d 1024, PFC-first: the council
  deliberates on the raw symbol, the trunk decodes the bundles). Learns
  at night (NREM cross-entropy on traces, batched, own optimizer, rate
  1e-4, 24 rounds, rest masked; REM forecast of the band states, one
  batched step, SIGReg 0.1) and at doses (rate 1e-5, credit-scaled,
  blame to three nats, content only). ORGAN. The head takes most of the
  gradient; the deep trunk wakes slowly. Nothing to change; instrument.
- **The routing organ** (route mode, `route_cap` 0.125, `ponder_aux`
  0.5): about one symbol in eight gets an extra council cycle and a
  deeper decode, the threshold self-tuned as a running quantile, the
  deep logits trained at half weight. ORGAN, learned; active; not
  previously disclosed here. Biology: variable deliberation. Keep.
- **The planner and imagination** (`plan_cand` 4, `imag_k` 4): four
  candidate plans gated by a learned softmax, rolled four steps forward
  and mixed into the cortex stream through two zero-initialized gates,
  trained by a foresight loss (cosine to the future stream at fixed
  horizons) whenever the body trains. ORGAN, active in every forward,
  its gates still near zero; not previously disclosed here; the cost is
  compute. Biology: prospection. It overlaps REM's forecast organ
  (`pfc_pred`); one of the two is redundant. Candidate: measure the
  gates and the foresight fidelity; retire whichever does not learn.
- **The hippocampus** (content-keyed store, RFF lift, delta rule; keys =
  the bag of the last eight symbols with decay 0.7 and the speaker
  sense; values = the next symbol; writes amortized every four forwards
  with the pending symbols buffered, none lost). Write strength:
  bookkeeping marks never stored (`kc_skip` 11, PLUMBING); the write
  gated by surprise against its running mean (`write_surprise` 1.0,
  PHYSIOLOGY: novelty encodes); the mouth's own symbols carry zero
  surprise (corollary discharge, PHYSIOLOGY); dopamine and the body's
  own prediction success multiply the write (`dopamine` 1.0,
  `intrinsic_w` 0.5, PHYSIOLOGY: salience). The store fades 0.9 a night
  (PHYSIOLOGY). Its read reaches the mouth two ways: a learned council
  slot (`store_in`, ORGAN) and a direct vote added to the logits at the
  organ's own strength (`store_boost` 1, `read_beta` 0). The direct vote
  is the one shortcut here: biology's hippocampus drives cortex, not the
  mouth. Candidate, later: retire it once the cortex listens (measure
  cue completion with and without it).
- **The night's selection** (`_dream_starts`: the store's strongest key
  directions by low-rank SVD, 48 starts, 40 Adam steps at 0.05 to find
  each key's preimage; `_dream_trace`: winner-take-all with neural
  adaptation 0.5 per repeat recovering by 0.7, stop floor
  `store_boost_min` 0.15). PHYSIOLOGY and PLUMBING; disclosed. The stop
  floor is the one number that decides how long a dream runs; a
  learned or relative floor (the vote against the store's own mean)
  would remove the constant. Candidate, low priority.
- **The inherited night** (`Organism.sleep`, written for the earlier
  bodies): for this body its candidate lists are empty (no taught
  facts, no study, no conscience file), the pursuit and the retention
  curve never engage, the page replay is off (`night_no_page`), so no
  NREM pass of the old kind runs. What still runs: the store decays
  and the working state resets to a fresh wake (PHYSIOLOGY), the
  autosave and the night backup (PLUMBING), an empty report card.
  LEGACY, verified inert except those. Candidate: strip the dead
  branches from the diary's night for legibility; no behaviour change.
- **The reward system.** The face enters as press levels (reward table
  0, +1, +2, -1, -2: PHYSIOLOGY); the fast band's signed prediction
  error is dopamine; every choice of the last twelve ticks takes it at
  0.8 per tick (eligibility, PHYSIOLOGY); a burst of at least 0.5 pays
  a dose (a budget for the backward pass, PLUMBING, disclosed); the
  lesson is graded by credit, clipped at plus or minus two, blame to
  three nats (PHYSIOLOGY, disclosed). The value ladder learns TD at
  every band from lived pairs (32 kept per band) at doses and once a
  night (`value_w` 0.5); the band gates learn Go/NoGo from the value's
  error (`bg_w` 0.01); the face organ forecasts the caregiver's face at
  every tick (`face_lr` 2e-5). ORGANS. Reward created inside: a dose
  without a face was seen on day 22 (07:29). Nothing here is a rule.
- **The mouth.** Content sampled at temperature 1 from its belief
  (bookkeeping marks and the newline barred, PLUMBING); whether to act
  is the gate's (ORGAN, above), with its own reward at a symbol (the
  belief it had in its choice, weight 0.5), a cost 0.12 per symbol that
  grows with fatigue (`/10`), the opponent rule at rate 0.05, a lesson
  every 24 ticks. PHYSIOLOGY, all disclosed. Temperature 1 is itself a
  choice; biology's action selection is sharper than the belief, and a
  learned decisiveness (tonic dopamine, vigor) would be the organ.
  Candidate, later, once the cortex has beliefs worth sharpening.
- **The feelings** (split): fatigue up 0.12 a symbol with a two-minute
  half-life; stress 0.5 per unit of dopamine dip, two-minute half-life;
  mood 0.25 per unit of dopamine, ten-minute half-life; sleep pressure
  one per tick, the switch at 12,000. PHYSIOLOGY, disclosed. Stress does
  nothing yet but inform the gate.
- **Numerical plumbing.** Atomic saves; night backups (three kept); a
  non-finite lesson skipped, gradients clipped at one in the night and
  in the gate, a night that leaves a weight non-finite discarded and the
  body reloaded; a failed night retried after half a day, not every
  tick; the night counted before its autosave. One remaining hazard:
  the serve's memory during the night (a second optimizer, the machine
  swapping; night 8 took 32 minutes).

**Rules found.** None that decide content or behaviour for the body.
The three constants closest to the line, each disclosed: the dose
burst threshold (0.5), the dream stop floor (0.15), and temperature 1.

**Undisclosed before this review, now disclosed:** the routing organ,
the planner and imagination, the surprise-gated write, the write
cadence, the reward table, the inherited night's inert branches.

**Biology's side: what the body lacks, ranked.**

1. **REM as generation.** REM forecasts along hippocampal traces; the
   cortex never runs free. Biology's REM is the cortex generating from
   memory with the hippocampus decoupled, which is also where the
   free-running attractor ("a", "u", "r") would be worked on and where
   old memories would be interleaved as the store fades. Depends on the
   cortex being able to generate anything but one letter; the nights
   are now moving it. Build when the trunk-alone probe first gives a
   frame.
2. **Stress as a modulator.** Acute stress should raise plasticity and
   exploration; ours only informs the gate. Candidate: stress scales the
   dose's rate and the gate's temperature. Measure first.
3. **Interleaving.** With the store fading 0.9 a night, frames older
   than about ten days leave the dreams; only the cortex's own
   generations can supply them (item 1).
4. **The direct memory vote** on the logits (above): retire once the
   cortex listens.
5. **The planner** overlapping REM's forecast organ: keep one.
6. **Action selection sharper than the belief** (learned decisiveness):
   later.
7. **The environment.** The smile is the only contingent response. A
   parent also answers babble in kind (repeats "ba ba", expands a word
   into a phrase); the curriculum's expansion covers letter runs only.
   This is the environment, the user's call.

**Two things measured this morning that bear on the review.** With the
gate live, the body earned six felt smiles in its first twenty minutes
of day 23 (four times the previous day's rate) and its mood went
positive for the first time in three days; and the gate is opening,
0.25 to 0.45 by 08:44 with fatigue at 16, because its belief in what it
writes is near one, so the drive outweighs a cost that only doubles at
fatigue ten. The fatigue-scaled cost bounds it near fatigue thirty.
Whether that is a healthy babbling bout or a flood is the day's
question; the weight of the drive (0.5) is the disclosed constant to
lower if it floods. It flooded: 0.63 by 08:50 with fatigue 22 and mood
falling, its belief in its babble near one. The drive's weight was
lowered to 0.25 (a save, then a restart at 08:51 that carried the
gate's learned weights; the gate read 0.34 on waking), which balances
the fatigue-scaled cost near fatigue ten. The biological refinement
this points at, built at 09:05 on the user's word (`--gate-habit 0.9`):
the reward for producing the sound one expected habituates to
repetition. The gate keeps a per-symbol memory of how often each symbol
was just chosen (decaying 0.9 per act); its own reward at a symbol is
scaled by one minus that, so a run pays less each time and a fresh
symbol pays in full, which is what keeps an infant's babble varied. To
be measured on the day-23 copy at drive weights 0.5 and 0.25 before it
is served for day 24.

## The waking cortex (2026-09-03, built on the user's question, not yet served)

"Shouldn't the cortex always be predicting the next state it receives,
on everything?" Yes: that is what a cortex is, waking and sleeping. In
this body it predicted at every tick but learned from the error only
at doses by day, since the law that the cortex learns only from
hippocampal traces (a reaction to day 1, when it learned its own
babble) had removed the waking lesson. Built behind `--wake-lesson 1`,
default off: every twenty-four ticks one step on the last thirty-two
symbols at the live rate, the day's optimizer; cross-entropy on the
symbols the world wrote, the mouth's own symbols inputs but never
targets (corollary discharge: what it wrote itself was foretold by its
efference copy) and its quiet never a target; plus the forecast of its
next band state with SIGReg, the same organ REM trains, so the cortex
learns the consequences of its own actions in its latent state while
awake. Guards as in the night. Nothing is learned when the world has
not written in the window. Smoked on the tiny body (the world's
cross-entropy fell 5.07 to 4.89 across two lessons). To be measured on
the copy and served on the user's word; it reverses an earlier law.

## The rebuild (2026-09-03, the user: "stop and pause and make all these changes")

The days paused after night 9 (the counter's seventh) while the
architecture was brought to what biology does, all behind flags,
measured on the day's copy before serving:

- **The latent cortex** (`--cortex latent`). The cortex predicts the
  embedding it receives next, one after the other, not the word: a
  forecast organ `latent_pred` on the stream, the target the next
  symbol's embedding (stop-grad), one minus the cosine as the loss,
  SIGReg on the forecasts as the collapse guard; the same lesson at
  night on the traces and by day in the waking lesson. Cross-entropy is
  retired, and with it the trained head: the mouth reads the lexicon
  itself, logits = sharpness x cosine between the forecast and each
  embedding (`--read-sharp 10`, a physiology constant standing in for a
  learned decisiveness). The readout starts from scratch, so the first
  babble under it is flat.
- **Generative REM** (`--rem-generate 1`, `--rem-steps 8`). From each
  dream's first three symbols the cortex runs free on its own readout,
  the hippocampus decoupled, and learns to forecast the band state it
  will receive at its own next step (targets stop-grad, SIGReg on the
  stream). The rollout is the body's own, not the store's: the
  free-running attractor is worked on here, and old material comes back
  through what the cortex itself brings up.
- **Reward where biology puts it.** Under the latent cortex a dose no
  longer pulls content toward a rewarded symbol; the dose forward still
  teaches the value ladder, the basal ganglia's band gates and the
  face. Reward reaches behaviour through the mouth's gate, the store's
  dopamine-scaled writes, and the ladder.
- **Stress as a modulator** (`--wake-mod 1`): the waking lesson's
  weight grows with stress/10 (acute stress encodes harder) and the
  gate's logit is divided by the same (stress flattens the choice:
  exploration). Chronic stress's withdrawal already emerges through the
  gate reading it.
- **The waking cortex** and **the habituating drive**, both above, go
  live with the rest.
- **The lean night** (`_rest`): the inherited `Organism.sleep`, with
  the earlier bodies' pursuits, report cards and conscience (all inert
  here), is no longer called; the night's plumbing is written out in
  full: the store fades, the working state wakes fresh, the day's
  buffers clear, fatigue rests, the body is saved and backed up.

Proposed launch line (served only after the copy test):

```bash
python3 scripts/diary.py data/organism_diary_0p5b.pt data/tok_char.json --dev mps --port 8018 \
    --temp 1.0 --store-read-beta 0 --store-boost 1 --store-boost-min 0.15 \
    --live-lr 1e-5 --store-decay 0.9 --save data/organism_diary_0p5b.pt --diary-period 0.5 \
    --diary-cost 0.12 --cort-k 1.0 --value-w 0.5 --wake-ticks 12000 \
    --night-rounds 24 --night-rem 8 --night-sigreg 0.1 --night-starts 48 \
    --night-opt own --night-batch 1 --night-lr 1e-4 --night-sil-mask 1 \
    --gate 1 --gate-lr 0.05 --gate-int 0.5 --gate-habit 0.9 --gate-cost 0.12 --gate-fatigue 10 \
    --gate-every 24 --affect split --wake-mod 1 --wake-lesson 1 --wake-every 24 --wake-window 32 \
    --cortex latent --read-sharp 10 --rem-generate 1 --rem-steps 8
```

## The second body (2026-09-03, from scratch)

The user's word at 10:15: "stop what we are doing now and start from
scratch." The first lineage is archived (`data/organism_diary_0p5b.pt`
and its night backups, never to be overwritten). The second body is
built from BODY_SPEC.md in `body/`: 6.5 million parameters, born at
10:27 as `data/body2.pt`, served on port 8018 at four ticks a second (a
day of 12,000 ticks is fifty minutes; a night about a minute). Nine
organ tests pass in four seconds (`python3 -m body.tests.test_organs`).
Two lessons the tests taught at birth: the striatum learns from the
error against a baseline (a constant cost had closed the gate), and a
recalled memory must tire or dreams loop. The feelings recover on the
body's own clock (half-lives in ticks). Days are given by the raw
caregiver as a script (`body/caregiver.py`: page-only rules, a line or
cue every sixty seconds, smiles within three seconds) while the agent
API is unreliable; agent caregivers can take over on the same protocol.
Probes between days: `python3 -m body.probe data/body2.pt`. The day
log: `data/body2_caregiver.jsonl`.

**Day 1 and night 1 (10:28 to 11:50).** Forty-seven events at a line or
cue a minute; two smiles ("oN" taken as "on", after which the match was
made exact). The waking lesson's forecasts rose through the day (next
symbol 0.03 to 0.08 in cosine, next state 0.015 to 0.17) while the
readout stayed nearly flat (top symbol 4 to 6 percent) and the gate
held near its birth rate at fatigue 10. Night 1 at 11:19 took ten
seconds: sixteen dreams of mean length 18.8, recognizably its lines
with stutters (" dddog ddoger biger iger", "red ddog dddogoger iger",
"rst "), NREM loss 1.25 to 0.75 over 24 rounds, REM five steps, and
the gauge from 0.007 to 0.537: after one night the cortex alone
forecasts more than half of its dreams' next symbols. The probe after
it: greedy and alone the mouth runs on "g" (the letter-frequency
stage), the sampled mouth is still noise (quiet 0.8), no cue started.
The readout's top symbol rose to 12 percent and the waking forecast of
the next symbol to 0.21 within minutes of waking.

## The fast days (2026-09-03, 12:30 to 13:35)

`body/fastlife.py` runs the same body and the same raw rules on the
body's own clock in one process: a day in about a minute, a night in
ten seconds. Eight runs from birth in an hour, each ended by a
measurement and a fix, each fix a line in the spec's mathematics:

- **Run 3:** the gate went to zero and stayed: a policy gradient's
  absorbing state (no acts, no gradient). Spontaneous activity as a
  floor, p(act) ≥ 0.05.
- **Run 4:** the gate sat at the floor. The stream feature had norm 16
  (Hebbian steps of tens of logits) and a stale baseline kept the credit
  negative for hundreds of lessons after the night reset fatigue. Unit
  features, a baseline that forgets in ten lessons and resets with the
  night, a tonic drive that makes babble the no-reward equilibrium.
  Then the probe showed the hippocampus recalling "l" after "dog will ":
  strength in the read let the most-reinforced memory answer every cue;
  quiet ticks were writing pause timing into the keys; its own babble
  polluted them. Recall by content alone, quiet only fading the context,
  own symbols entering at a third; a collision test.
- **Run 5:** REM's retention vanished on night 3: the PFC's input maps
  learning at the critic's rate moved the states everything forecasts.
  A slow PFC (the cortex's rate).
- **Run 6:** the cortex's own forecast plateaued at 0.30 while the night's
  loss fell: NREM fed the store's recall as input and the cortex learned
  to copy it. NREM with the read off.
- **Run 7 (12 days, every fix in):** the cortex alone, after each night,
  0.26, 0.31, 0.28, 0.27, 0.33, 0.51, 0.70, 0.69, 0.73, 0.70, 0.70, 0.76
  argmax on its dreams (cosine 0.23 to 0.76), retained between nights;
  REM rising within every night; smiles 0, 0, 0, 0, 0, 1, 0, 5, 0, 1, 3, 0
  (go, on, all); by day 8 the cue replies carry the frames' letters
  ("all" after "scared "). The mouth alternates between a space attractor
  and an "l" attractor: at a fixed sharpness of 10 the readout is nearly
  greedy on a blended forecast, so babble has no variety.
- **Run 8 (20 days, running):** decisiveness from tonic dopamine, the
  readout's sharpness 5 + 5 × mood/6 (songbirds: vocal variability high
  when unrewarded, falling as reward comes), replacing the last stand-in
  constant of the mouth.

The slow lineage on port 8018 kept its hourly days for comparison and
for the page; it will be reborn from the settled recipe.

**Milestone 3 (2026-09-03, 15:45, run 10, day 14, tick 168,210).** After
the caregiver typed "dog will " the body wrote "go", and the caregiver's
page-only rule recorded a cue completion and smiled. The same day it
wrote "dog", its first three-letter word, and "ba" after "give "; the
next day "baa" after "where ball? ". Run 10 carried every fix through
the world's-next-symbol target but not the two forecast fixes found
after it (recall to the forecast, the cortex on its own error), so its
answer came through recall and sampling with an off-by-one still in its
cortex; by then its cortex carried its dreams at 0.87 argmax. One
completion in a day of sixteen cues: the first, not fluency. The body
is kept as `data/body2_run10_day15.pt`. Run 14, with the forecast fixes,
is the candidate for the settled recipe.

## Status

- model: speaker channel, ear-writes by speaker, running bag with
  silence decay — built and measured on CPU (2026-09-02).
- serve: `scripts/diary.py` — the tick loop (two forwards per tick: your
  symbol as the ear, its symbol as the mouth), the page (your letters
  black, its memory letters brown, its noise pale; arrow keys move your
  face; Enter is a new line), doses on its actual choices by the word
  body's thresholds over the last eight ticks, the same nights, save and
  reset. Numbers: stress +0.03 per symbol written (half-life 120 s), mood
  -0.002 x stress per symbol and +0.5 x a felt face; a face registers on
  whole-unit crossings (int); characters outside the alphabet are dropped;
  the typing queue holds 600 symbols. Smoked on a tiny CPU body (2026-09-02): ticks, both
  hands on the page, faces felt, a night, a morning.
- first serve: the word body is parked after seven days; the diary body
  runs on port 8018:

```bash
python3 scripts/diary.py data/organism_diary_0p5b.pt data/tok_char.json --dev mps --port 8018 \
    --temp 1.0 --store-read-beta 0 --store-boost 1 --store-boost-min 0.15 \
    --live-lr 1e-5 --store-decay 0.9 --save data/organism_diary_0p5b.pt --diary-period 0.5 \
    --diary-cost 0.12 --cort-k 1.0 --value-w 0.5 --wake-ticks 12000 \
    --night-rounds 24 --night-rem 8 --night-sigreg 0.1 --night-starts 48 \
    --night-opt own --night-batch 1 --night-lr 1e-4 --night-sil-mask 1 \
    --gate 1 --gate-lr 0.05 --gate-int 0.25 --gate-cost 0.12 --gate-fatigue 10 \
    --gate-every 24 --affect split
```

Since day 23 (2026-09-03 08:15): the mouth's go/no-go gate and the
three feelings (fatigue, stress, mood), both above. The stress lean is
gone; the silence mask stays harmless (rest is never in the content
softmax now).

The night's flags since day 21 (2026-09-03, measured on the day-20
copy, above): sleep's plasticity has its own optimizer state
(`--night-opt own`, freed at waking), many replays make one change
(`--night-batch 1`: one step per round over all the traces in NREM and
one step over the dreams in REM), at sleep's own rate (`--night-lr
1e-4`) for twenty-four rounds, about eight minutes. Under the day's
optimizer and single-trace steps at 1e-5 the night had been inert
(two or three argmax flips of fifty), and REM's single-dream steps at
the higher rate undid NREM. `IGA_NIGHT_DEBUG=1` prints each round's
loss. `--night-scale` is inert under Adam and left at 1.

The live rate is ten times the word body's on purpose: the first thing
frowns must teach a newborn mouth is silence, which is the cheapest
attractor a trunk can fall into, so here the collapse the word body had
to avoid is the lesson.

The principle (the user's, 2026-09-02): the only restriction on the mouth
is stamina, and no hand-written rules. What remains: the bookkeeping
marks it cannot emit (plumbing, not language); stress leaning it toward
silence (cort_k logits per unit) and weighing on mood; the face lesson
every tick. Memory's voice is the organ's own: its read enters the logits
at its learned strength, nothing raises it, and nothing resets or stores
on the caregiver's behalf (the three laws removed mid day 14, above). The night replays the day as lived, silences
included (runs kept to one tick), so a quiet tick can be learned as
thinking time. Instruments per tick: its face, mood, stress, uncertainty,
memory's votes, and the trunk's own top symbol with memory set aside (the
measurement that will show the day the trunk itself begins to propose
letters). Reflexes it drops: the breath, the hush, the end-is-an-end
rule, the bag reset (silence fades the bag instead).

## The forecast's norm is its certainty (2026-09-03)

Run 14's day-6 sequence probe: the store's first letters 6/8, the mouth's 1/8, and the
mouth's wrong answers were all 'l' or 'g'. 'l' does not follow a space in any line the
body has heard. It is the commonest letter it has heard. The readout adds log(heard
prior) to sharpness × cosine, and with the cortex trained by one minus the cosine, the
forecast's norm meant nothing: a flat forecast and a sure one read the same, and the
prior's commonest letter won every tie. Adding a unit recall to a unit cortex forecast
does not help either: two directions of equal weight, and the readout normalised the sum.

The mathematics wants calibration. Train the forecast by squared error to the unit
embedding received and its minimiser is the conditional mean of the next embedding:
norm 1 when one symbol follows, small when many can. Read the lexicon by the dot
product, sharpness × (forecast · E) + log prior, sharpness 10 + 10 × mood/6: a sure
forecast beats the prior's largest log-gap (about 4.6), an unsure one lets the prior
babble, as a baby babbles its language's sounds. What the mouth reads is the cortex's
calibrated forecast plus the recall's unit direction times its confidence: two
calibrated votes, and their agreement is sharp because the readout is a dot product.

The second finding of the same probe: after its own first letter the store kept
recalling the first letter. Its own symbols entered the key at 0.3 and the key barely
moved. Raising the weight to 0.6 broke the collision test, and the reason was the real
pollution mechanism: the bag decayed per tick, so the babble between the world's
symbols shifted the world's weights in every key by however much it happened to
babble while listening. The context is now two bags summed into the key: the world's,
decaying per world symbol (a pause or its own babble leaves the world's context as it
was), and its own, decaying per tick (its babble fades with time, its last own symbol is
what remains). Measured on the collision test with a continuation check: own weight
0.3 to 0.5 keeps every cue's recall, 0.7 loses one, and 0.5 moves the key after its own
first letter half the time. 0.5 it is.

Ten organ tests pass. Commit ea479e6. Run 15 from birth on this recipe; the day-6
sequence probe is the test: the mouth's first letter should follow the store's.
Run 10, the control, finished twenty days with its cues still babble.

## Two calibrations more (2026-09-03, run 15)

Run 15 on the calibrated recipe: milestone 1 two nights earlier than run 7 (gauge 0.52
after night 4), then day 5 fell to two smiles and the mouth said mostly spaces. The
day-5 sequence probe: the store right on 5 of 8 first letters at confidence 0.11 to
0.17, the mouth a space on 6 of 8.

Both are the same kind of error as the last one, a quantity read off the wrong
measure. The store's confidence was the largest attention weight; with the key now
carrying its own bag, the eight slots of one line no longer merge (their own-bag
parts differ), so the mass splits eight ways to 0.11, every slot saying 'g'. The
recall is the attended mean of unit values, and the norm of that mean is the
agreement: eight slots saying 'g' give norm 0.98. That is the confidence, and the
store's own reference for the dream floor is measured the same way.

The space: the readout added log(heard prior) to the forecast's dot products. But a
forecast trained by squared error to unit targets is the conditional mean, whose dot
with E_k is the probability of k; the unconditional mean is the heard distribution
itself. Adding the prior counted frequency twice, and in every flat context the space,
the commonest symbol, won. The prior is gone; the sharpness rises to 25 + 25 × mood/6
because the dots are now probabilities, not cosines, and the lexicon's pairwise
cosines of 0.06 need a symbol at probability 0.5 to outweigh fifty strangers at their
noise. The forecast head is born near zero (norm 0.1): the default init gave a norm of
4 to 9 of pure noise, one deterministic junk symbol until the first lessons.

Ten tests pass; the tiny body's night now moves its gauge 0.13 to 0.65 where the
cosine-trained tiny body's did not move at all. Run 16 from birth.

## The stutter (2026-09-03, run 16)

Run 16 on the two calibrations: twelve smiles on day 1 with whole words among them
(bigger, milk, will), the gauge over 0.5 after night 2 where run 7 needed six nights,
and on day 2 the mouth began 6 of 8 cues with the right letter. Then it stuttered it:
"bbbbbb" for "where ball? ", "dd ddd" for "big dog bigger ".

The mathematics, with the key a decayed bag: the world's context W has norm about 1.67
(0.8 per symbol). After its own first letter b at weight w, the query is W + w·b. The
stale key is W itself, cosine |W|/|W + w·b|; the continuation key is 0.8·W + b. At
w = 0.5 the stale key wins, 0.96 to 0.94, and the mouth says b again; at w = 1 the
continuation wins, 0.99 to 0.86, decisively at temperature 0.02. Full weight had been
ruled out because its own babble in the keys broke recall by content at 0.6.

The two weights are two different things, and biology keeps them apart. Corollary
discharge suppresses the hearing of self-produced sound: a memory of the world's
sequence is stored under the world's context alone, never under its own babble. The
efference copy is the sequencing system's full knowledge of what it just said: a memory
is read with the world's context plus its own symbols in full. Both bags fade with time
(0.8 per tick): a pause ends a context, as working memory does. The persistence I had
given the world's bag this morning was a mistake: typing is one symbol per tick, so the
world's weights in a key never depended on the babble between them, and a persistent
bag stored every line's first letters under the previous line's tail.

On the collision test with a continuation check: every cue recalled, and after its own
first letter the store continues (o after g, i after m, n after o) at confidence about
1. The test itself had to learn the design: read before the body answers, since the
tiny body says "g" as the cue's space enters, and with that g in its query the store
rightly recalls "o". Eleven checks pass. Run 17 from birth.

## Run 17, and the shape of a tick (2026-09-03)

Run 17 on the corollary-discharge keys: day 1, 427 smiles and the cues completed with
the words themselves, the first letter said as the cue's last symbol enters (the fast
caregiver's record began a tick late; fixed). Night 1 took the gauge from 0.00 to 0.59.
The day-6 probe: first letters 7 of 8 by store and mouth alike, second letters 8 of 8,
the sampled mouth 27 of 48 started and 19 of 48 full (scared → "dog had milk", give →
"milk then", why dog up? → "because", six of six each). Milestone 3 is met every day by
the hippocampus and the efference copy: the body echoes what it heard. Then by day 9
the gate stood at 0.997, every chain ran into "ball under", the smiles fell to 96, and
one night lowered the gauge from 0.62 to 0.41.

Three findings from the probe, all measured on the day-6 body.

The crowd. For "dog will " the true key matched at cosine 0.998 with weight 0.42, and
six slots at 0.966, identical to each other (mutual cosine 1.000), all "ball " then "o",
took 0.08 each and outvoted it. Two causes. A speaker embedding had been summed into
every symbol of the bag: a constant every key shared, which pushed every cosine toward
1, so keys sharing only " ll " with "dog will " sat at 0.966. The bags are content
alone now. And the merge judged by the single best-matching key: a slot with the same
key and the value "u" (ball under) sat first among the ties, blocked the merge, and each
hearing of "ball on" added a voter. A memory now merges into the best-matching slot
among those that say the same.

The shape of a tick. The gauge said the cortex alone forecast 0.83 of the dreams' next
symbols; typed "dog will go" awake, the same cortex forecast "d" after nearly every
symbol. The window was the difference: awake, every world symbol was followed by the
body's own entry, mostly a rest, so the stream read "d · o · g ·", while the dreams the
night trains on read "d o g". The cortex had learned one format and lived in another.
Biology's time step is the tick, and all the sounds of a tick superpose: a position is
now the world's embedding plus half its own (corollary discharge in the stream). A quiet
listener's stream is the dream's format exactly; its own voice is an attenuated
superposition the day's lesson teaches it to see through.

The bags were also fading twice a tick, once at the world's step and once at its own
(0.64 a tick where the spec said 0.8); a tick fades them once now.

Ten checks pass; the tiny night lifts its gauge 0.27 to 0.71. Commit 1b759ac. Run 18 from
birth, with a new instrument in the day-6 probe: the cortex alone along typed lines under
the dream construction and under life's, which must now agree.

## Run 18: the tick confirmed, and order (2026-09-03)

Run 18 on one position per tick. The day-6 trace, the cortex alone along typed lines
under the dream construction and under life's, scored right (my first scoring compared
life's forecasts against the wrong offset): "dog will go" 8 of 10 awake, "why dog up?
because big dog" 18 of 26, the same as in the dream construction. The cortex the night
trains now lives in the format it dreams in. The sampled mouth started 45 of 48 cues and
completed "dog will go" 6 of 6.

What it could not do was stop repeating a letter: "balll", "boookkk", "bbaall". The
decode after its own "ball": the query was nine-tenths "l", the key for the second l of
"ball" matched at 0.976 and the key after "ball" at 0.962. A bag of symbols cannot tell
"bal" from "ball"; the difference is a weight, and a little extra context tips it. The
hippocampus is not blind to order: theta sequence coding gives each lag its own code.
The mathematics of that is the lag code of holographic reduced representations: each
symbol shifts the whole context through a fixed permutation of the dimensions before it
enters at lag 0, so "l" at lag 0 and "l" at lag 1 are orthogonal directions and "bal"
and "ball" are far apart. The read query shifts the world's context by as many lags as
the body has said since, and what it said leaves the query when the world speaks again.

And the fade diagnostic on run 17's body: with the cosine read a context faded to norm
0.01 by 24 quiet ticks still recalled "m" at confidence 0.87, the faint tail of the last
line amplified into a full direction, which is how the chains ran across lines through
the pauses. The store reads by dot product now, keys unit, the query as it is: its norm
is the inverse temperature of recall, and confidence falls with the pause (0.88 at the
cue, 0.81 six ticks later, 0.30 at twelve).

Ten checks pass, the collision test now at confidence 1.0 for every cue and the
continuation check too. Commit 0022416. Run 19 from birth. Open: a night that lowers the
gauge while its loss falls (run 17 night 9, run 18 night 6), and the gate near 1.0 when
smiles flood.

## REM through the trunk (2026-09-03, run 19)

Run 19 on the lag code: no stutters from day 1 ("ball? ", "ball u", "milk t", "dog up",
"becaus"), and the nights: +0.39, −0.11, −0.05, +0.23 on the gauge. A night that lowers
the gauge while its own loss falls had happened on runs 17 and 18 too. The instrument
law: a scratch copy of run 19's day-2 body slept four times.

The night as it was: gauge 0.66, after NREM 0.85, after REM 0.78. The same night
without REM: 0.85 stays. REM without SIGReg: 0.78 again, so SIGReg was not the cause.
REM with the stream detached, the PFC's forecast heads learning alone: 0.85 stays. The
PFC's forecast objective, trained through the cortex's trunk for six rounds at the
night's rate, undid a third of what NREM had consolidated. Two objectives on one trunk
in alternation, not jointly, oscillate.

The principle was already written for the cortex: each area learns from its own error.
The PFC's forecast heads learn from the stream; they do not rewrite it, in REM and in
the day's lesson alike. Your question of this morning, whether both the PFC and the
neocortex should learn in REM, is answered by the measurement: the cortex cannot serve
the PFC's objective in REM without losing its own, and the dorsolateral PFC is in any
case the part of the brain that REM switches off. And SIGReg, the collapse guard, is
retired: with the fixed lexicon as the target and the bundle objective off the trunk,
nothing can collapse the stream; the function stays, the term is zero.

Ten checks pass; the scratch night now reads 0.66, 0.84, 0.84. Run 20 from birth once
run 19's day-6 probe has measured the lag code and the dot-product read.

## Run 19 at day 6, and the price of a word (2026-09-03)

The day-6 probe of run 19 (lag code, dot-product read): first letters 7 of 8 by the
store, second letters 8 of 8, and the sampled mouth started 48 of 48 cues and completed
40 in full: "dog will " gives "go in", "where ball? " gives "ball under", "I had " gives
"milk then", "big dog bigger " gives "dog up? because", "why dog up? " gives "because big
dog", six of six each. No stutters. The cortex alone, without the store, forecasts 53 of
82 next symbols awake against 58 under the dream construction: the tick fix holds, and
consolidation is visible at day 6.

What the body cannot do is stop. Gate 0.96 to 0.97 through days 2 to 6, fatigue pinned
at its ceiling of 40, and every answer chains into the next line it once heard after
that one. Mood only 0.3 to 0.9: the smiles are predicted and dopamine is small, so the
reward flood is not the reason. The arithmetic is. The cost of a symbol was linear in
fatigue, 0.12 (1 + fatigue/10), which is 0.59 at the ceiling, and a confident, varied
recitation earns the tonic 0.25 plus a novelty term near 0.45 whatever the caregiver
does. A linear cost never beats that drive at any fatigue the body can reach. Effort
cost is convex in every account of it; with (fatigue/10)² the cost passes the drive near
fatigue 22, a duty cycle of about a half, and the spec's promise of babble in bouts
bounded by fatigue becomes arithmetic rather than hope.

Run 20 carries the REM fix alone in a second slot (its day-6 probe measures that); run
21 carries both. Run 19 retired at day 7 with its measurement taken.

## Run 20 at day 6: the night holds (2026-09-03)

Run 20 carries the REM fix alone. Six nights, gauge after NREM and after REM identical
every night: 0.75, 0.76, 0.82, 0.86, 0.88, 0.88. The day-6 probe: the sampled mouth
started 48 of 48 cues and completed 47 in full. "scared " gives "ball under", "first milk
then " gives "ball under", "give " gives "book then", six of six. The cortex alone,
without the store, forecasts 53 of 82 next symbols awake against 59 under the dream
construction, and 22 of 26 on "why dog up? because big dog". The gate sat at 0.98: the
effort cost of run 21 is the remaining measurement.

## Run 21 at day 6, and the recipe is settled (2026-09-03)

Run 21 carries the REM fix and the convex effort cost. The economy: gate 0.52 to 0.55
every day, fatigue settled at 22, where the arithmetic said the cost would meet the
drive, mood 0.3 to 0.6, smiles 207 to 283 a day where run 20's mouth earned 440 by never
pausing. The nights: 0.73, 0.86, 0.82, 0.83, 0.88, 0.89, REM leaving every one of them
where NREM put it. The day-6 probe: first letters 6 of 8, second letters 8 of 8, the
sampled mouth started 47 of 48 cues and completed 37 in full, quiet half the ticks, the
answers now in bouts ("ball under", "milk then", "dog up? because", "because big dog");
the cortex alone awake 55 of 82 next symbols.

Every flaw measured today has its fix, its test, its paragraph and its run. The recipe
of commit e9be497 is the one the served body is reborn from: the old slow body, seven
nights on a recipe four fixes behind, is archived as data/body2_oldrecipe_n7.pt. Runs 20
and 21 continue to twenty days for the question that matters next: whether the cortex
takes over from the hippocampus, measured by the cortex alone at days 15 and 20.

## The reborn body's first day (2026-09-03, evening)

Born at 16:58 from the settled recipe, 6.5M parameters, an empty store, on the page at
port 8018 at a quarter second a tick, the scripted raw caregiver at its side at human
pace. Smiles on "up", "dog", "in", "go" within the first half hour, and then the cue
"scared " completed with "ball": milestone 3 on the served body, day 1, no chance in it.

## Day 15, and the invariance (2026-09-03, night)

Runs 20 and 21 at day 15. Run 21 (the settled recipe): the sampled mouth started 48 of 48
cues and completed 47 in full; the cortex alone awake 61 of 82 next symbols, up from 55
at day 6, with "why dog up? because big dog" at 24 of 26. Run 20 (no effort cost):
42 of 48 full, the cortex alone 56 of 82, flat since day 6.

What the cortex alone cannot do is chain its own speech: without the store it loops,
"will will will", "red red red", "up? up? up?". The measurement on run 20's day-15
body: its own symbols entered the stream at half weight superposed on a rest, and the
cortex alone gave identical output whether that weight was 0.5, 1.0, or its own sound
replaced the rest outright. Identical: the cortex's forecast did not depend on its own
symbols at all. That is a learned invariance, and the day's lesson taught it. Its own
symbols sat in the lesson's inputs while the target was always the world's next symbol,
which they never predict, so the cortex learned to ignore its own voice, and what it
learned after the world's "g" could not apply after its own.

Biology's corollary discharge suppresses learning from self-produced sound, not the
hearing of it. So the lessons now hear the world's symbols only, and life hears its own
sound as it hears the world's: the tick's input is whatever sounded, its own voice
replacing the rest when the world is quiet, superposed when both sound. Unlearned
about, its own "g" is the world's "g" to the cortex. Commit a914f08; run 22 from birth,
and the alone column of its day-6 probe is the test.

The first generalization probe, eight cues never heard whose answers follow the
corpus's patterns ("big ball bigger ", "where dog? ", "why ball up? "): nothing at day
15 on either run. The store returns the nearest heard chain, "ball under" for most, with
one partial success where the frame carries the answer: "why ball up? " gives "because
bi…". Twenty-two fixed lines cannot teach "big X bigger X"; that needs the cortex free of
the invariance, a wider world, and time. Recorded as the honest state.

The reborn body's first night, 17:49: forty dreams ("dog will go inder dog up", "g dog
bigger dog up?"), gauge 0.09 before, 0.76 after NREM, 0.76 after REM, 192 slots kept.

## Run 21, twenty days (2026-09-03, night)

The settled recipe's long run is in. Twenty nights, every one of them held (the gauge
after REM equal to the gauge after NREM, 0.73 on night 1, 0.90 on night 20); the gate
0.48 to 0.58 every day; 190 to 280 smiles a day. Day 20's probe: the sampled mouth
started 46 of 48 cues and completed 46 in full, the cortex alone awake 60 of 82 next
symbols (55 at day 6, 61 at day 15, a plateau near three quarters), "dog will go" at 9
of 10 and "why dog up? because big dog" at 23 of 26. The unheard cues: still nothing.
The body is a faithful echo of its world with a cortex that has learned the world's
sequences and cannot yet use them for its own speech; run 22 is the test of that.

## The instrument that echoed (2026-09-03, late)

Run 22 at day 6 answered no: the cortex alone still looped, and its next-symbol trace was
lower (48 of 82). Then two readings of the same body and cue disagreed: the trace said
the cortex forecast "d" after "dog will ", the probe's alone column said "w". The probe's
store held eight slots. "Memory off" emptied the store at load, and the cue's own typing
wrote the cue back into it; the loops were that store echoing the cue's own words, "will"
after "dog ". Every "alone" loop since run 17 was the instrument. The variants that gave
identical output at gains 0.5, 1.0 and replaced did so because the store dominated all
three. Memory off now means no writes and no reads.

Measured honestly, the cortex alone of run 21 at day 20, no hippocampus at all: "dog
will " gives "go up? because", "scared " gives "ball onder", "where ball? " gives "ball
onder", "first milk then " gives "ball ball", "why dog up? " gives "because big". At day
6 the same recipe's cortex could not ("inder dog up?"). That is the answer to the
question that mattered: the cortex takes over from the hippocampus across days, under
the lesson as it was. Run 22's change, its own voice at full weight and the lessons
hearing the world only, made the cortex alone worse at day 6 ("big big big") and is
reverted, on measurement, with an apology to the diary for the entry before this one.

Two things stay from the cycle. The honest probe. And the mouth's read at a quiet-world
tick: the forecast made at the last filled position, the one holding its own last symbol,
which is trained to foresee what follows it; a freshly appended rest foresaw what follows
a pause. Commit aca73c0; run 23 from birth on the settled recipe plus that read.

The reborn body's second day at human pace: 196 smiles, every cue answered from the
page ("dog will go", "book then ball", "milk then ball", "ball under", "dog up? scared",
"because" once as "cared ball"), and the second night 0.52 to 0.86 with 210 slots kept.
Run 23 at day 15, with the store: 44 of 48 in full; the cortex's trace 59 of 82, where
run 21 had 61 at the same day.

## Run 23, twenty days (2026-09-03, night)

The settled recipe with the mouth's read at the last filled position. Twenty nights held
(0.73 to 0.89), gate 0.47 to 0.56, cues completed daily. Day 20: with the store, 47 of 48
cues started and 43 in full; the cortex's next-symbol trace 61 of 82, the plateau run 21
reached; the cortex alone, no hippocampus, opens "dog will " with "go" and "big dog
bigger " with "big dog dog" and little else, where run 21's cortex alone at the same day
completed five of eight. The honest range for the cortex alone at day 20 is one to five
cues of eight, varying by run, on a next-symbol accuracy of three quarters that does not
vary. The read fix did no harm and stays; the served body took it at its day-3 boundary.

What is settled tonight: grounded reward at two timescales (the face's dopamine on the
gate, the PFC ladder's values on its clocks), a hippocampus that stores the world's
sequences under the world's context and completes them through the efference copy, a
cortex that consolidates them at night and holds them awake, a mouth that speaks in
bouts, and no rule anywhere that authors a word. What is not: generalization beyond the
twenty-two lines, and a cortex that carries its own speech as reliably as the
hippocampus does. Those are the next measurements, not the next guesses.

## The ladder, measured (2026-09-03, late night)

Your law again, and the one claim in it I had never measured: reward at long timescales.
The PFC ladder has eight bands at clocks 1 to 16384 ticks, each with a value head trained
by TD. Whether the slow ones learn anything was an assumption. An instrument: one fast day
on a copy of run 21's day-20 body, the reward and every band's value recorded per tick,
the realized return computed at each band's own horizon.

The result. At horizons 1 to 256 ticks the value equals the mean return and correlates
with the realized return at 0.10, 0.02, −0.11, −0.23, −0.03: the critics learned
constants, so dopamine has been reward minus a baseline, not an anticipation. At 1024 to
16384 ticks the values diverge, the slowest reading 7072 against a true return near 85
with correlation −0.995, the known instability of semi-gradient TD with bootstrapping as
the discount nears one. And a ridge fit from each band's state to its return, held out,
reads nothing up to 1024 ticks; the slow bands' positive fit is the ramp of a state that
fills at one over its clock across the day against a return-to-go that shrinks.

Two corrections, both from the biology and the mathematics rather than from a knob. The
band update wrote at rate gate over clock, so a slow band could never load; it is now the
gated working memory of the basal-ganglia model, the gate loading the band and the clock
forgetting it, with the gate's rest set so an untrained band is exactly the leaky average
it was. And the bands at or above 1024 ticks learn average-reward TD, the reward rate at
their own clock as the baseline, which is what tonic dopamine is. Ten checks pass;
commit 1ec4dbf; run 25 carries it, and the same instrument on its day-6 body is the test.
Beneath both sits the open question the instrument also raised: whether the bands' input
maps, learning at 1e-5, ever come to carry a feature that predicts reward.

Run 24, in parallel, is the ablation your law asked for: the gate's novelty term removed.
Days 1 to 3: the gate at 0.27 to 0.29 instead of 0.55, smiles 48 to 83 a day instead of
about 300, and the cues still completed. The term buys babble, not answers.

## The ablation (2026-09-03, late night)

Run 24, the gate's novelty term removed, at day 6: the gate at 0.26, quiet three ticks in
four, 41 to 83 smiles a day, and the sampled mouth started 37 of 48 cues but finished 19,
against 45 with the term; the answers break mid-word ("ballgi", "milkfi") and drift. The
term stays, on measurement, and its role is now clear. Once the critic predicts the
smiles the dopamine error is small, and a bout that has begun a word is held open only
by a drive. In the basal ganglia that holding is the actor's, a learned Go for a
sequence under way, not the error's; the term stands in for it, and replacing it with
the actor is the next honest step for the mouth. Run 24 retired at day 6.

## Run 25 at day 6: the divergence gone, the constants not (2026-09-03, night)

The same instrument on run 25's day-6 body, the gated write and the average-reward TD in
place. The divergence is gone: the slowest band reads 70 against a return of 89 where it
read 7072, and every band is bounded. The body is unharmed: 47 of 48 cues started, 45 in
full, the gate at 0.52. But the states still carry almost nothing that predicts reward,
the ridge ceiling near zero at every horizon and the correlations 0.08 to 0.16, so the
critics remain near constants. What remains suspect is the rate at which the bands'
input maps learn, 1e-5, set when the PFC's objectives still ran through the cortex's
trunk and moved its forecast targets. They no longer do. Run 26 raises it to 1e-3, the
one change, and the instrument on its day-6 body decides.

## Run 26: rate is not the lever (2026-09-03, night)

The bands' input maps at 1e-3, the one change. At day 6 the states sit at the tanh
ceiling, norm 16 on six of eight bands: the maps saturated under the TD gradient and
the states became sign patterns, the critics constants again. And the saturated bundles
are the cortex's input too: its next-symbol trace fell to 33 of 82 from about 52, and
alone it says "bbiiiiiii". Run 26 retired at day 6; the rate stays at 1e-5. The question
underneath is whether the stream carries reward-predictive information at all, and the
instrument for that reads the return from the stream itself integrated at each clock,
with no learned projection in the way.

The reborn body's third night: gauge 0.74 before, 0.90 after, 213 slots kept; days 4 to 6
queued at human pace.

## The stream ceiling (2026-09-03, late)

The last instrument of the night reads the return from the cortex's stream itself, now
and integrated at each clock, with no learned projection in the way, on run 21's day-20
body: R² about zero at every horizon from 1 to 256 ticks (0.001, 0.011, −0.098, −0.471,
−0.129), and the 0.81 at 1024 ticks is the time-of-day artifact once more. The stream,
read linearly, does not carry the reward at short horizons.

So the ladder's constants are not only the projections' fault. The values could not
learn what the state does not carry, and the state cannot carry what the reward does
not depend on. In this world the reward is about three hundred smiles a day at isolated
known words, each within twelve ticks at the caregiver's jitter, against eight
completions of a cue. Nothing in the stream at a horizon of four or sixteen ticks
predicts the next smile better than its rate, and nothing at a thousand predicts the
day's tally better than the time of day. Your second criterion, reward at long
timescales, is met by the architecture's form and not yet by any measurement, and the
measurement says the environment does not pose the problem the ladder exists to solve.

What stays: the ladder's two corrections (no divergence, no harm), the instrument, and
the honest statement. What it asks for is a world whose reward depends on longer
history than a word, which is the environment's design and yours to give: a caregiver
who smiles for an answer more than for a word is the nearest such world, and it is the
proposal already written in CURRICULUM.md, not in force.

## The gated write reverted (2026-09-03, night)

Run 25 at day 15: with the store, 44 of 48 cues in full, unharmed; but three mid bands'
values had blown up, means in the hundreds with standard deviations near 2000 and TD
errors in the thousands. Gates that learn to open fully make the states jump, and
bootstrapped values on jumping features diverge: the deadly triad again, features
non-stationary this time. The gated write goes back to the leaky average at the clock,
on measurement; the average-reward TD stays, having removed the first divergence, and
the slow bands' slow filling is harmless once their values are bounded. Commit 4730e38;
run 27 carries the settled ladder, with the instrument at days 6, 15 and 20 for
stability. Ten checks pass.

Run 25 at day 20, the gated write's record: the body unharmed (40 of 48 in full, trace 49
of 82), the mid and slow bands' values wild to the end (one at 1671 with a spread of 830,
another at −244), and the slowest band's gate learned to shut its state to nothing. The
revert stands; run 27 is the settled ladder's run.

## Run 27 at day 6, and one reward rate (2026-09-03, night)

The settled ladder's first instrument: the body's best day yet, 48 of 48 cues started
and all 48 completed in full, the cortex's trace 58 of 82, seven bands bounded. The
eighth, the slowest, ran away again, 643 against a return of 131 with correlation −0.99,
and the mathematics of differential TD says why: its baseline, the reward rate, was
estimated at the band's own clock, one part in 16384 a tick, too slow to track the rate
within a day, so the undiscounted value integrated raw reward. The baseline must
converge faster than the value drifts, and there is one reward rate in any case: tonic
dopamine. One running mean of the reward at the differential horizon, shared by the
slow bands, replaces the per-band estimates. Ten checks pass; run 28 carries it.

Run 28 at day 6, one reward rate shared by the slow bands: every band bounded, the
slowest at −5 with a spread of 2 where run 27's ran to 643, the fast bands the constants
this world allows. No divergence anywhere in the ladder for the first time.

Run 27 at day 15, the per-band reward rates still in it: the slowest band's value at −1036
with a spread of 610, as the mathematics of its baseline predicted; the seven others
bounded; the body at 46 of 48 started and 45 in full, and the cortex's trace at 61 of 82
awake and 67 under the dream construction, the best yet. Run 28, with the one shared
rate, is the recipe's test.

The reborn body's fourth night: gauge 0.85 before, 0.90 after, 213 slots kept. At the
day-5 boundary the serve reloads so the body takes the settled ladder (one shared reward
rate for the slow bands); its words and its store are untouched by that.

Run 27 at day 20, the per-band rates to the end: 47 of 48 cues started and 46 in full,
the cortex's trace 60 of 82 awake and 63 under the dream construction, the best twentieth
day of any run (run 21 had 46 and 60, run 23 had 43 and 61). The ladder as the baseline
mathematics predicted: the slowest band's value at −402 with a spread of 248, back from
−1036 at day 15 as its one-in-16384 baseline crept toward the rate, and the two bands
above it biased below their returns the same way. The slowest band's near-perfect
correlation is the time-of-day artifact again, not a reading of reward. Body kept as
data/body2_run27_day20.pt. Run 28 carries the one shared rate and is at day 14 with the
gauge 0.90 and the gate 0.53; its day-15 instrument decides whether that ladder stays
bounded.

## Run 28 at day 15: the ladder pinned (2026-09-03, late night)

The one shared reward rate did what its mathematics said: the slowest band, which ran to
643 and −1036 under its own rate, sits at −8 with a spread of 4 at day 15. And two bands
above it, bounded at day 6, ran away instead: −1294 and −2147, their states at the tanh
ceiling (norms 14.6 and 11.5 where a bounded band's is 3 to 5), the cortex's trace down
to 55 of 82 from run 27's 61, since the bundles feed the cortex. Each fix had moved the
runaway to another band. The diagnosis, at last, is not a rate. Two things. The bands'
input maps learned from the critic's own bootstrapped error, which is the deadly triad,
semi-gradient TD through nonlinear features with no guarantee, and at every rate they
ran until the tanh saturated (1e-3 by day 6, 1e-5 by day 15); and they had nothing to
learn, since the stream read at every clock carries no reward in this world. And a
differential value is relative, defined up to a constant, and a linear head over raw
states has two directions that constant can walk in under the optimizer, its bias and
the states' mean; walk they did. So the maps are born and kept, like the lexicon, and a
differential head has no bias and reads its state centered on a running mean of the
states at the reward rate's horizon (adaptation, the oldest trick of a neuron). Linear
heads on fixed features under on-policy TD converge; that is the theorem the ladder now
stands on. Eleven checks pass (the eleventh: maps fixed, relative values centered and
bias-free); an old body loads with its mask and its means born fresh. Run 29 is born on
it, twenty days, instruments at days 6, 15 and 20. Run 28 goes to twenty for the record
of the runaway's end. The mouth, meanwhile: run 28 at day 15 completed 44 of 48.

The reborn body's day 5 at human pace on the shared rate; its bands will be read before
the fix is applied at a boundary, once run 29 has measured it.

Run 28 at day 20: 48 of 48 cues started and 47 in full, the cortex's trace 60 of 82 awake
and 61 under the dream construction, back from day 15's 55. Its value column is not the
old recipe's record: the instrument's day ran under the pinned ladder, the body's old
heads reading states now centered, so the two runaway bands read −29 and −448 with the
means still catching up, not the thousands of day 15. Body kept as
data/body2_run28_day20.pt.

## Run 29 at day 6: the ladder holds (2026-09-03, night)

The pinned ladder's first instrument, run 29 at day 6: every band bounded, the three
relative values centered where they were made to sit (0.1, 1.8 and 0.7, spreads of 7,
6 and 2, against run 28's thousands), the TD error 0.26 at every band, no state at the
ceiling. The mouth started 48 of 48 cues and finished 46, the cortex's trace 56 of 82
awake and 60 under the dream construction, the best sixth day of any run. Day 15 is the
test of the nights. The served body takes this ladder at its day-6 boundary, armed.

Run 30 through day 5, the gate reading the forecast's certainty with the intrinsic
credit gone: the gate at 0.29, forty-five smiles a day, the salience weight at 0.003
after two days, the answers coming letter by letter with rests between. The feature sits
between 0.8 and 1.9 whatever the moment, so the gate has little to read in it, and the
external dopamine alone, small once the critic predicts the smiles, holds no bout open.
The intrinsic credit is not a placeholder for an actor after all. It is the songbird's
own performance dopamine (Gadagkar 2016: dopamine neurons encode the bird's performance
error against its template, and deafened birds do not learn), habituating per syllable
as dopamine habituates to a repeated stimulus. The formula stays on that ground; what
mathematics asks is that it be an error, performance against the syllable's expected
performance, with the innate drive and the reward rate's vigor carrying the rate of
acting. That is run 31, after run 30's day-6 record.

## The teacher (2026-09-03, night)

Run 30's record at day 6, the intrinsic credit gone and the gate reading the forecast's
certainty: 44 of 48 cues started and 21 finished, run 24's number again, the ladder
bounded under the pinned form, the gate at 0.26 through nine days. Retired at day 9. The
external dopamine alone cannot hold a word open in this world; the intrinsic performance
dopamine stays, as written above.

On your word, the teacher. Not a script with twenty-two lines but a caregiver whose speech
is planned by Claude, one utterance a minute at the body's human pace, and whose reward
rules are the raw ones still: a smile within seconds of a known word, at a cue's completion
or its first two letters, a frown at a run of marks, decided from the page and its face
row only. Two honest generalizations of those rules so the world can grow: a word is known
once the teacher has typed it three times, and a cue's answers are the continuations of
the lines the body has actually heard at least twice, computed from what was typed, never
listed by hand. The planner sees only the page, what was typed and what came back, never
the body's insides; it may bring one new word a day, said in three lines, and it varies
the phrasings around the same words, which is the road to the thirty-two novel cues the
body has never completed. Two planners: the Anthropic SDK with a daily call budget, in a
shell where the key is exported (never here), and a queue file that a Claude Code subagent
fills, which needs no key and starts tonight. Tested on a scratch serve: the queue's lines
typed, "big dog will " a cue with the answer computed from two hearings of "big dog will
go", the body completing it, an unheard prefix treated as a line. The served body's day 6
is the teacher's first, after tonight's boundary reload to the pinned ladder; the fixed
eight cues remain the yardstick before each day and after each night.

## Run 29 at day 15: settled (2026-09-03, night)

The pinned ladder through fifteen nights: every band bounded, the three relative values
at −5, 5 and 2 with spreads of 11, 11 and 4, the TD error 0.25 at every band, no state
near the ceiling. The mouth started 48 of 48 cues and finished 43; the cortex's trace 62
of 82 awake and 64 under the dream construction, the best of any run at any day. The
ladder rests on its theorem and the body is none the worse for it. The recipe is settled
as of tonight: d3dd681, the salience input present at zero. Run 29 goes to twenty for the
record. The served body took it at its day-6 boundary at 21:35, nights 5, 214 slots,
and the teacher's first day began.

Run 29 at day 20: every band bounded to the end, the relative values at −10, 3 and 2
with spreads of 14, 9 and 4, the TD error 0.26 at every band, no state near the
ceiling; the mouth started 48 of 48 cues and finished 47; the cortex's trace 57 of 82
awake and 65 under the dream construction. Twenty days without a runaway anywhere in the
ladder, for the first time. Body kept as data/body2_run29_day20.pt.

## Run 31 at day 6: the credit as an error (2026-09-03, late night)

Two runs born together at 21:50: run 32, the settled recipe under a second seed, to know
how far two bodies of one recipe fall apart before any change is judged; and run 31, the
intrinsic credit as the songbird's performance error, the forecast's belief in what it
said against that syllable's usual belief, habituating as the expectation catches up,
with the innate drive at 0.70 so the mean credit is what the value form gave a grown body
(0.25 and half of 0.90, measured on run 29's twentieth day). At day 6, every band bounded
in both. Run 32: 48 of 48 started, 44 finished, the cortex's trace 59 of 82 awake and 65
under the dream construction; against run 29's 46 finished and 56 and 60, that is the
seed spread. Run 31: 47 started, 46 finished, the trace 63 awake and 69 under the dream
construction, the best sixth day of any run on the cortex's own measure, and the relative
values the tightest yet (spreads of 1, 2 and 1). Day 15 decides whether the error form
becomes the recipe.

## The answer-weighted smile, on the word (2026-09-03, late night)

Your word, with its condition: the answer-weighted smile goes in if the slow critics still
read nothing after the teacher's days. The mathematics says what to expect first. At the
64- to 256-tick horizons a cue's completion is one smile among five or six word smiles, so
a critic reading the state cannot tell a rich minute from a poor one; even the teacher's
cues are drowned unless an answer is worth more than a word. And the body's physiology
caps a felt smile at 2 per event, a held face is silence, so a bigger smile must be a
smile that grows: the face rises to 2 and then to 4, and the body feels each rise. That is
a caregiver's face and nothing else; the body is untouched, and the rules still read the
page alone. Built into both caregivers behind a switch, off until measured (test 13: the
growing smile is felt as [2, 4], a word's as [2]). Run 33 measures it in the fast world
from run 32's slot at day 15, the settled recipe under the answer-weighted smile, its
value instrument's day under the same smile: the number to watch is the ridge ceiling at
the 64- and 256-tick horizons, zero in every run so far. The served body's own gate stays
the one you set: the value instrument after teacher day 8, and the switch turns on at a
boundary only if that reads nothing.

Two clarifications asked and answered tonight. It is not a cheat: the caregiver reads the
page only, nothing writes the body's words, the body still has to say the answer; what it
is, is a curriculum decision, a world shaped so that reward depends on history, and it is
written here as that. And the teachers are Sonnet, not Opus, one small decision every
seven minutes; Opus on the word.

## Run 31 at day 20: not yet (2026-09-03, late night)

The error form to twenty days: the ladder bounded, the cortex's trace 65 of 82 awake, the
best of any run, and the mouth 47 of 48 started but 39 finished, where the value form's
three seeds finished 46 to 47. The loss is one cue: "dog will " answers "gog will gog" in
every sample, a stutter the recall fell into between day 15 ("go in") and day 20, the
other seven cues at 5 or 6 of 6. The credit decides whether the mouth speaks and never
which symbol, so this reads as one seed's memory, not the form; but one seed is one
seed, and a recipe is not changed on a hope. The value form stays the recipe, the error
form stays in the code, and run 34, its second seed, decides at day 20. The raised body
keeps its own form; a saved body without a form key now loads as what it was.

## Run 33 at day 6: the ceiling does not move (2026-09-03, late night)

The answer-weighted smile in the fast world, the settled recipe under a smile that grows
at a cue's completion: at day 6 the body is what it was (48 of 48 cues started, 46
finished, the trace 57 awake) and the ridge ceiling at 64 and 256 ticks is what it was,
zero (−0.06 and 0.01). The mathematics had said as much and now says why more exactly.
Doubled, the sixteen completions of a day are a fifth of its reward; the rest is the
hundred-odd word smiles, each a coin the caregiver's refractory rules toss. And the
completions come when the caregiver chooses to pose a cue, every four minutes or so by
its own clock, not by anything the body can read in itself a minute ahead. A critic at
256 ticks predicting reward from the body's state is asked to predict the caregiver's
schedule. That is the environment's structure, not the ladder's failing: reward at long
horizons has content only where the reward RATE over minutes depends on something the
body carries, and in this world it does not. The form of the world that would give it
content is a caregiver whose attention wanes with babble and returns with answers, so
that a minute's smiles depend on the body's own last minutes; the biology of it is the
engaged parent, and it is page-only. It is not built and not proposed tonight; run 33
goes to day 15 for the record, and the raised body's test after teacher day 8 stands.

Run 33 at day 15, the answer-weighted smile in the fast world: 48 of 48 cues started and
48 finished, the first perfect fifteenth day of any run; the cortex's trace 64 of 82
awake and 68 under the dream construction; every band bounded; and the ridge ceiling at
256 ticks at 0.20 where the flat smile's runs read 0.10 or less at that horizon, still
zero at 64. One seed, one day, a small number, in the direction the arithmetic gave:
an answer worth twice a word makes the minute after a cue a little more legible to a
critic. Day 20 for the record; the raised body's own test after teacher day 8 stands.

## The parent (2026-09-03, near midnight)

Your word: mold the caregiver to a real parent. What a real parent has that the rule table
lacked is attention that moves. So the caregiver, in both its forms, now carries an
engagement decided from the page alone: it rises at an answer and at a known word, more
at a word new that day, falls at babble, drifts down in silence over a few minutes; and
the parent behaves by it. A known word gets its smile with probability e, a distracted
parent misses words, and a parent tires of the fiftieth "dog"; the parent talks faster
when engaged and slower when not; it answers a smiled word with a line that holds it; and
below a floor it turns away for a while, the still face, until it comes back. The answer
smile is always given and grows. Nothing reads the body's insides, and the body reads the
parent's attention only as every child does, through what the parent does. The
mathematics of it: a minute's smiles now depend on the body's own last minutes, so a
critic at 256 or 1024 ticks has something in the body's state to predict, which is the
content the long-timescale ladder has lacked in every world so far. Run 35 measures it
in the fast world (the parent and the growing smile, the settled recipe); the raised body
takes it from day 9, gated on run 35's sixth day looking sane.

Run 33 to twenty days, the answer-weighted smile alone: 43 of 48, the trace 62 awake and
67 dreaming, the ceiling at 256 back at zero; the 0.20 of day 15 was noise. Run 34, the
error form's second seed, at day 15: bounded, the mouth's number still computing.

Run 34, the error form's second seed, to twenty days: 47, 38 and 45 of 48 at days 6, 15
and 20, the cortex's trace 60, 62 and 63 awake, the ladder bounded throughout. With run
31's 46, 44 and 39, the error form has one day under forty in each of two seeds where the
value form, in five measurements over two seeds, never fell below 43. Its traces run a
little higher; its mouth runs a little less steady; the drive it replaced was a constant
by day 20 in any case. The recipe keeps the value form. The error form stays in the code
as what it is: the songbird's mathematics, measured, not adopted. Bodies kept as
data/body2_run34_day6/15/20.pt.

## Run 35 at day 15: the slow critics read something (2026-09-03, midnight)

The parent world's fifteenth day, and the number that has been zero in every world so
far moved: the critics' values against the realized returns at their own horizons,
correlation 0.59 at 256 ticks, 0.75 at 1024, 0.78 at 4096, where the flat worlds read
zero or below at every one of those (run 29 at day 15: −0.11, −0.47, −0.34), and the
slowest band at −0.13, so this is not the time-of-day artifact that gave the slowest band
its false 0.99. Under a parent whose attention moves with the body's own behavior, the
reward rate over minutes is a thing in the world, and the ladder's middle bands have
learned to carry it. The body is none the worse: 47 of 48 cues started, 46 finished, the
cortex's trace 60 awake and 66 under the dream construction, on sixty rewards a day
where the flat world gave a hundred and twenty. The ridge ceilings from a single day's
states stay negative at 64 and 256, which says the signal is slow, a matter of minutes,
not of one day's linear fit; the critics that learned it over fifteen days are the
measurement. One seed; run 36's fifteenth day is the check. If it holds, the second clause
of the law, reward at long timescales, is met in form and in content for the first time.

Run 35 to twenty days under the parent: the critics' correlations with their own returns
grew as the days went, 0.47 at 256 ticks, 0.84 at 1024, 0.91 at 4096, the slowest band
at −0.26 and so no artifact, every band bounded, the TD error 0.17; the mouth 46 of 48
started and 43 finished, the cortex's trace 60 awake and 69 under the dream construction,
on fifty-four rewards a day. The middle of the ladder now carries the parent's attention
as a value, learned from nothing but the face and the page. Body kept as
data/body2_run35_day20.pt.

## Two seeds (2026-09-04, after midnight)

Run 36, the parent world's second seed, at day 15: the critics against their returns at
0.23, 0.72 and 0.54 at 256, 1024 and 4096 ticks, the slowest band at −0.51, every band
bounded, the body 45 of 48 started and 42 finished, the cortex's trace 56 awake and 65
under the dream construction. With run 35's 0.59, 0.75 and 0.78 at day 15 and 0.47, 0.84
and 0.91 at day 20, the finding holds on two bodies: the second clause of the law is met.
The reward is grounded in the face; the fast critic reads it at seconds; the slow critics,
under a world that has a slow structure, read it at minutes; and none of it is a cheat, the
parent reading only the page and the body reading only the parent's behavior. Written
into BODY_SPEC.md §5b. What remains for the ladder is what the body does with it: the
mouth's gate takes its credit from the fast band's error alone, and a body that could feel
the parent's attention waning would have a reason to answer. That is the next honest
question, and it is architecture, not environment.

## The raised body's own test, and a repair (2026-09-04, 00:20)

After teacher day 8, the instrument on the raised body itself, in the flat world it was
raised in: its critics against their returns at 0.01, −0.04 and −0.25 at 256, 1024 and
4096 ticks. Nothing, as your condition foresaw, and so the growing smile is in force for
it, and the parent with it, from day 9 (the day-9 yardstick before it: 46 of 48 started,
44 finished; the cortex's trace 63 awake, 66 dreaming). The same instrument found the
thing to repair: the two slowest value heads read values in the hundreds with a TD error
of 1.0 where a body born under the pinned ladder reads 0.17. They are the heads trained
under the diverging ladder in its first five days, before the day-6 reload pinned the
form but kept the weights; the states are fixed and fine. At the day-9 boundary those
heads are born fresh, the running means with them, and nothing else is touched. Run 36
to its end: 0.42, 0.55 and 0.39 at the three horizons on its twentieth day, 41 of 48,
the trace 64 awake and 66 dreaming.

## The first parent day (2026-09-04, 00:55)

The raised body's ninth day, its first under the parent: 35 utterances, 7 cues from the
teacher's own lines, every one answered with an accepted word ("I will " with "go down",
"where ball? " with "ball go down", "little dog had " with "milk gone"), 108 smiles where
the flat rules gave two hundred, 123 words the parent let pass unsmiled, no turning away,
the parent's attention between 0.42 and 1.0 across the day. The night after it: gauge
0.78 to 0.90, 269 slots kept, dreams of "because big dog" and "go up then". The body is
raised now in the world that has a slow structure; the instrument on it after day 12 will
say whether its own slow critics come to read the parent, as the fast bodies' did.

## The slow error in the gate: not the form (2026-09-04, 01:00)

Runs 37 and 38, the parent world with the 1024-tick band's error added to the gate's
credit, at day 15: mouths 42 and 42 of 48 against 46 and 42 without; the cortex's trace
64 and 63 awake, a little above; smiles a day about the same; and the slow critics'
correlations with their returns weaker, 0.48, 0.51 and −0.50 and −0.05, 0.41 and −0.16
against 0.59, 0.75, 0.78 and 0.23, 0.72, 0.54 at the same day. A slow band's error is a
slow band's error; as a per-act credit it is noise, and it muddies the very critics that
carry the parent. Biology's vigor is not an error but a level, tonic dopamine as the
reward rate (Niv 2007); that is the form still standing, unbuilt. The switch stays off.
Runs 37 and 38 go to twenty for the record.

## The second yardstick (2026-09-04, 01:05)

Cues drawn from the teacher's own lines, heard at least twice and none of the eight the
body was born on, twenty-four of them, on the raised body after day 8: 93 of 96 started,
77 finished. The misses are mostly the scorer's: "why dog " answered "up? because", "where
" answered "ball? ball", "all " answered "gone" run into the next word; the honest ones are
the prefixes with three or four continuations ("dog ", "you will go "), where it picks one
the corpus also allows but says it fused to the next. Three days of a teacher's lines,
taken from the page at a human pace, are in it and come back on demand. The instrument
is scratchpad/taught_probe.py; it runs on any saved body with the corpus file. What it
does not test is the unheard combination; that remains the open capability.

At the day-9 boundary, 01:04: the raised body saved, the three slowest value heads
reborn at zero (their norms had been 14, 155 and 666, weights from the diverging days
before the pinned ladder), the running means with them, nothing else touched, and the
serve reloaded on the same body, nights 9, 271 slots. The day-10 yardstick before the
day: 46 of 48 started, 44 finished. Day 10 under the parent began at 01:05.

Runs 37 and 38 at day 20 (the slow band's error added to the gate's credit, the parent
world): the mouth unchanged, 44 and 42 of 48 full; the slow critics empty. corr(V, G) at
256/1024/4096/16384: run 37 -0.36 / 0.00 / -0.05 / -0.68; run 38 -0.27 / 0.84 / -0.58 /
-0.78 (the 0.84 with V's spread twice G's). The recipe under the same parent read
0.47/0.84/0.91 and 0.42/0.55/0.39 (runs 35, 36). The slow error in the gate's credit
takes from the critics without giving to the mouth: not adopted, on two seeds, twice.

## The level (2026-09-04, 01:20)

The slow critics have content under the parent (runs 35 and 36), and nothing in the
body reads them. What reads a slow value in an animal: the Pavlovian side of the
striatum, where a cue that promises reward at a long horizon invigorates whatever
the animal is doing (general Pavlovian-instrumental transfer, the amygdala's value
onto the ventral striatum's vigor; Niv's tonic dopamine as the opportunity cost of
time is the same fact from the rate side). The body's mood is the rate side already,
a leaky integral of the fast error over 1200 ticks, read by the gate; it is backward-
looking, what came. The 1024-tick critic is forward-looking, what this moment promises,
and it knows the parent: 0.84 with the realized return at that horizon.

So the gate gains a fifth feeling: the slow band's value of the moment, divided by its
own running root mean square (divisive normalization, the canonical cortical operation,
with a semi-saturation of one reward unit so a newborn's noise reads small), clipped
at five, weighted by its own three-factor lesson like every other input. No rule says
which way: if speaking pays more when the parent is engaged, the lesson finds a
positive weight and the body speaks into engagement and rests through the still face;
if not, the weight stays where it is. The scale is a buffer per band (born at one, rate
1/1024, the band means' rate); older bodies load with it fresh and the new weight at
zero, so the served body is untouched by the code. Off by default (gate_level_w 0);
runs 39 and 40 (seeds 39, 40, the parent world, weight 1) measure it against runs 35
and 36 at days 6, 15 and 20: the mouth's yardstick, the smiles and the parent's
attention per day, the slow critics' correlation, and the gate's learned weight on the
level. Tests 14 of 14. Commit 237cc49.

## The instrument, corrected (2026-09-04, 02:05)

Runs 39 and 40 ran to twenty days with the level in the gate. The mouth: 46 and 46
of 48 at day 6, 44 and 43 at day 15, 40 and 46 at day 20 (the recipe's two seeds: 43
and 41). Smiles a day over days 11 to 20: 109 and 114 against 103 and 83; the parent
away 1.5 and 0.9 times a day against 1.5 and 3.5; its hit rate 0.44 and 0.43 against
0.42 and 0.39. Seed noise, no gain. The gate's own lesson on the level: -0.72 and
-0.27 at day 6, -0.62 and +0.12 at day 15, -0.03 and -0.07 at day 20. The striatum
read the slow critic and found nothing in it to act on; the weight went to zero by
itself. Not adopted; the input stays in the code at zero, like the salience.

Then the yardstick itself. Runs 39 and 40's day-15 critics read nothing (-0.19 and
-0.33 at 1024 ticks against the recipe's 0.75 and 0.72), and before calling that the
level's doing I asked how sure a single instrument day is. A day of 7200 ticks holds
seven independent windows of the 1024-tick return, its last 2000 ticks carry returns
truncated at the day's end, and the "ridge ceiling" (a held-out linear read of the
band's state) came out negative on the very bodies whose heads read 0.75: not a
ceiling, an overfit of 256 collinear dimensions on a handful of effective samples.
So the instrument was rebuilt (value_probe2.py): four days on fresh copies of the same
body, each day twice the plan (44 lines, 16 cues, 14376 ticks), a different caregiver
seed and order per day, the return's truncated tail cut off (only ticks with at least
86% of their horizon ahead count), and the spread across days reported. The old
estimator on the same days agrees with the new one within 0.1: the old readings were
not the truncation, they were one short day's luck. Under it, the day-20 bodies:

    run  variant                    256 ticks     1024 ticks    4096 ticks
    35   recipe                    +0.29 (0.12)  +0.26 (0.24)  +0.52 (0.29)
    36   recipe                    +0.38 (0.11)  +0.14 (0.14)  -0.00 (0.28)
    37   slow error in the gate    +0.08 (0.29)  -0.19 (0.21)  +0.04 (0.12)
    38   slow error in the gate    +0.43 (0.13)  +0.27 (0.30)  -0.55 (0.15)
    39   the level                 +0.30 (0.17)  -0.15 (0.13)  -0.57 (0.03)
    40   the level                 +0.36 (0.10)  -0.04 (0.23)  -0.22 (0.20)

What stands: the 256-tick critic reads its return on every body, about 0.3 to 0.4.
The 1024-tick critic reads a little on the recipe, 0.26 and 0.14, and about nothing
on the four variants; the difference is inside one day's spread and is not a verdict
on the variants, only no help from them. The 4096-tick horizon cannot be judged by a
day of any length we can run: its return falls through the day as the parent
habituates while the band's state climbs as it integrates the day, and the two trend
against each other whatever the body knows (run 39: four days at -0.57 with a spread
of 0.03, the trend, not the critic). "The second yardstick" and "The first parent
day" above quoted 0.84 and 0.91 at 1024 and 4096 for run 35: those were one short
day's readings and are withdrawn; the content the parent gives the slow critics is
real and small, 0.2 or so at 1024, and the claim that clause 2 was met on two seeds
rests on that, not on 0.9.

Where the long timescale stands, then, in words: the reward exists at every horizon,
the critics are bounded everywhere, the fast ones are right, the 1024-tick one is a
little right where the world has slow structure, and nothing the body does depends on
the slow ones, because two routes into the mouth (the slow error as credit, runs 37
and 38; the slow value as an input, runs 39 and 40) neither helped the mouth nor
sharpened the critics. In this world the long horizon has nothing to teach that the
short one does not: a smile follows a word within ticks, the parent's turning away
follows babble within a minute, and the slow value is a smoothed copy of the fast.
For a long timescale to matter, the world must hold a consequence that arrives only
later; that is the environment's side, and the environment is not changed without
the user's word.

The served body, meanwhile: day 10 under the parent (the day-10 planner's report):
seven batches, the new word "going" in seven lines (dog going up, ball going down,
big dog going down...), every cue it reached answered: "dog will " with go, "where
ball? " with ball, "bigger dog " with up, "first up then " with in, "why dog " with
"up? because"; 109 smiles, no frowns, the parent never away, its attention 0.7 to 1.0
falling to 0.4 by the night. Night 10: loss 0.30 to 0.12, gauge 0.73 to 0.87, dreams
"dog will go downg", "dog up? because big dog". Day 11 began at 01:55 with two
planners feeding for a while (the first day-11 subagent, which I had taken for dead
when it ended its turn to wait on a monitor, woke when the day began and queued two
rows before I stopped it); one planner since, with the foreground-wait rule.

## The unheard combination (2026-09-04, 02:25)

The capability the record kept calling open, measured with an instrument that hand-lists
nothing (scratchpad/unheard_probe.py): from the corpus the teacher's typing built, lines
heard at least twice; a word Y whose heard continuation is settled (one word follows it in
at least two lines and at least twice as often as any other); a word X that begins some
heard line; the cue "X Y " where the pair "X Y" occurs in no line the body has heard; the
answer Y's continuation; 32 such cues from a fixed seed. On the served body's day-10 copy:
the sampled mouth with its memory started 99 of 128 and finished 75, greedy with memory
26 of 32, the cortex alone 2 of 32. "little big " gives dog, "first why " dog, "where
because " big, "what? you " will, "dog first " milk. The misses are the fusions ("dogive",
"willittl") and three cues whose Y is "I" ("milk I ", "what? I ", "all I ", wanting had).
Run 35's day-20 body, raised on the twenty-two fixed lines, reads 60 of 128 on the same
cues. The composition runs through the store: its key is a decaying bag of the last
symbols, so an unheard prefix reads as its last word and recalls that word's continuation,
pattern completion on a partial cue, which is what a hippocampus is for. The cortex does
not compose on its own yet (2 of 32; but "alone" starves it of the read it was raised
with, so the trace, 60 to 70 of 82 teacher-forced, is its fairer measure). What this does
not test: a continuation that depends on a word before the last ("first milk then " ball
against "first up then " in); that is the next rung.

## The seam (2026-09-04, 02:50)

The mouth's fusions have a cause. On the served body's day-10 copy, the forecast after
"dog will go down" is the letter g at probability 1.0; after "you will go in", g again;
after "all gone", r at 0.88. Those are the first letters of the teacher's next utterances
("give ...", "go ...", and lines beginning with r), and they are the body's "downg",
"ing", "gonere", "dogive", "willittl": it finishes a word and goes on into the seam
between one utterance and the next, because for the cortex there is no seam. The window
holds a position for every symbol and none for the world's quiet; the waking lesson's
target is the next world symbol wherever it is, so the last letter of a line is taught
to foresee the first letter of the line that came forty seconds later; the store never
writes the quiet ("the world's quiet is not a memory"), so a line's end has no
continuation to recall and the nearest other slot answers instead; and a dream may not
rest (silence banned in its readout), so every dream runs off the end of its memory into
whatever slot is nearest, and the night trains the cortex on those splices. Four organs
agreeing that an utterance has no end. The taught-line yardstick on the same copy: 93 of
96 started, 60 finished, against 77 finished two days earlier; the difference is fusions.

An utterance's end is an event in any auditory cortex (the offset response: cells that
fire when a sound stops), and the page had the symbol for it already, the tokenizer's
<eot_human>, the end of the human's turn, banned from the mouth and never used. THE
OFFSET: after offset_ticks (12, three seconds) of quiet on both sides following the
world's utterance, the world's turn-end enters once as a world symbol. It is then what
every organ already does with a world symbol: the store writes it under the line's
context (the line's end is a memory), the waking lesson targets it after the last letter,
the bags take it and fade, the mouth's forecast at a line's end points at it and may not
say it (so the belief in whatever else it might say is small, and the gate's own lesson
finds that ends do not pay), and a dream ends where its memory recalls it, no longer
splicing. The body's own turn-end is not marked yet (a corollary of the same kind, for
later). Nothing is a rule about words: it is the perception of a pause. Test 15 of 15.
Off by default until measured: runs 41 and 42 (seeds 41, 42, the parent world, offset
12) against runs 35 and 36 at days 6, 15 and 20 on the mouth's yardstick (started
against finished, where fusions show), the trace, and the taught-line and unheard
yardsticks on the fast world's own corpus. If it holds, the served body takes it at a
boundary. Commit 0c676db.

Day 11 on the served body (the planner's report): 49 utterances, no new word ("happy
dog", "sad dog" from the base set), every cue answered ("why dog up? " because, "big
dog " bigger, "dog will go " down twice, "give big " ball, "dog had " ball, "where
ball? " ball), 90 smiles, the parent never away, its attention 0.3 to 0.93. Night 11:
loss 0.20 to 0.10, gauge 0.78 to 0.89, and the dreams "g will go ing ", "go up then
ing ": the seam, dreamed.

The offset's first form asked for quiet on both sides, the world's and the body's, and
a fast body that babbles at a gate of 0.5 never gives it twelve ticks: after six days
run 41's store held two turn-end memories and the tally of the symbol read about one.
The world's turn ends when the world stops, whatever the body is saying, so the rule is
the world's quiet alone (commit 2474b39, test 15 holds). Runs 41 and 42 were the recipe
in effect and are stopped at day 6 (44 of 48 on the mouth, the recipe's number); runs 43
and 44 are born with the offset that fires (03:00).

Runs 43 and 44 (the offset as a stream symbol, on the world's quiet alone) showed on
their first day what a symbol in the stream costs: the cues answered "go n Z", "Kill /",
"bookHt". The turn-end entered eight to twelve ticks after a cue, in the middle of the
body's answer, and a world symbol clears the body's own bag (its efference copy of what
it has said since the world's last symbol), so the answer lost its own context and went
to junk; and the untrained symbol sat in the window the cortex reads. Stopped at day 2.
The third form (commit 000e0e8, 03:05): the offset is a memory and a target, not a
stimulus. After eight ticks of the world's quiet, once per pause, the store writes the
turn-end under the line's context (the world's bag as it stood after the last symbol,
the surprise of the quiet as the strength), and the line's last position in the window
is marked ended, so the waking lesson's target there is the turn-end instead of the
next line's first letter; a dream still ends where its memory recalls the turn-end.
Nothing enters the stream, the bags stand, the mouth's context stands. Eight ticks,
because the window keeps a rest position per quiet tick (it rolls by time), the lesson
comes every 24 ticks over the last 32 positions, and 8 + 24 keeps the ended position
inside it. Runs 45 and 46 (seeds 45, 46, the parent world) measure it.

Runs 45 and 46 (the third form) on their first two days: "go in"", "bookpD", "dogldo",
"ballDu", "gogive", "milkgi", junk and fusions where the recipe's first days answer
cleanly. The cause is the store. A cue is the world going quiet too, so "then quiet" was
written under the cue's context and blended in the recall with the answer's memory; with
the turn-end banned from the mouth, the blended vector reads as whatever stray symbol it
happens to lean toward. The store keeps what the world said next and nothing else. The
fourth form (commit e875775, 03:10): the offset marks the line's last position, the
waking lesson's target there is the turn-end, and a dream ends where the cortex alone,
run over the dream so far, expects the quiet; the stream, the bags and the store are
untouched. Runs 47 and 48 (seeds 47, 48) measure it.

## The offset holds (2026-09-04, 04:25)

Runs 47 and 48, the fourth form, twenty days in the parent world:

    day   run 47 mouth   run 48 mouth   recipe seeds (35, 36, 39, 40)
     6      46 of 48       47 of 48      40, 46, 46, 46
    15      45             45            46, 42, 44, 43
    20      43             47            43, 41, 40, 46

Unheard combinations at day 20 (the fast world's corpus): 87 and 73 of 128 against the
recipe's 76 and 80. The cortex alone within typed lines: run 47 at 51, 53, 56 of 82
(below the recipe's 56 to 64), run 48 at 59, 61, 59 (in range); the trace instrument
scores inside lines only, so run 47's is a real small cost on one seed, not the seam.
Dreams shorter (10 to 17 symbols against 15 to 18) and ending where their lines end
("will go in", "ecause big", "e ball? ba"); the gauge after the night 0.80 to 0.92.
No first day's junk, no fusion learned from the seam. The offset is the recipe
(offset_ticks 8; commit below); the served body takes it at the end of day 13 through
the serve's physiology flag, the boundary armed (scratchpad/offset_boundary.sh 13).

The served body at the day-13 boundary, before the offset: the fixed cues 46 of 48
started, 43 finished; the taught-line yardstick 92 of 96 started, 57 finished (77 after
day 8, 60 at day 10: the fusions growing as the seam is learned); the unheard
combinations 95 of 128 started, 72 finished (99 and 75 at day 10). Day 12's teacher
(the planner's report): 42 utterances, five clean completions of eight cues reached
("where ball? " ball, "I had " milk, "dog will " go, "why dog " up, "give big " ball,
"first up then " in), 113 smiles, the parent never away. Day 13 under way with the
earlier-word emphasis in its lines ("give little book", "ball will go down").

At the day-13 boundary, 04:33: the raised body saved (data/body2_before_offset_day13.pt
keeps it as it was), the serve restarted with the offset (offset-ticks 8), nights 13,
267 slots. From day 14 its waking lessons target the turn-end after each of the
teacher's lines, and its dreams end where the cortex expects the quiet. Day 13's teacher
(the planner's report): 42 utterances on the earlier-word pairs ("give big ball" against
"give little book", "dog will go down" against "ball will go down", "first milk then
ball" against "first up then in"), every one of its seven cues drawing the expected word
("first up then " in, a cue completion at attention 1.0; "where ball? ball " under;
"little dog had " milk), 88 smiles, the parent never away. Night 13: loss 0.17 to 0.09,
gauge 0.81 to 0.90, dreams "w dog will go down", "bigger dog up? because b", " gonen ba"
(the seam still dreamed, the last night before the offset). The day-14 yardstick before
the day: the fixed cues 45 of 48 started, 42 finished.

The day-14 boundary yardsticks, on the body as saved at 04:33 (before any lesson under
the offset, so not the offset's doing): the taught-line cues 95 of 96 started and 75
finished (57 at day 13, 60 at day 10, 77 after day 8), the unheard combinations 106 of
128 started and 96 finished (72 at day 13, 75 at day 10). Night 13 and a day on the
earlier-word pairs, or a low draw at day 13: the sampled yardsticks draw four samples a
cue and swing. The offset's own effect on the seam reads from the day-15 boundary on.

## The ventral critic (2026-09-04, 04:55)

Your law again, and the long timescale's honest gap in the architecture. Runs 37 and 38
fed the 1024-tick band's error to the mouth and nothing moved, and the mathematics says
why: that head reads a state that moves a thousandth per tick, so an act's effect on
the long-run prospect cannot appear in its error inside the mouth's twelve ticks of
eligibility; its error was the fast error plus noise. The ladder ties each head's
horizon to its state's clock. Biology does not: the ventral striatum predicts far ahead
from the cue it sees now. So, THE VENTRAL CRITIC: one relative (average-reward) value
head over the whole ladder, every band's state centered on its running mean, learned
awake by the same semi-gradient differential TD as the slow bands (linear on fixed
features, convergent), zero at birth, no bias, born fresh in an older body. Its error
r - rbar + V(s') - V(s) moves within a tick of an act, because the fast bands do, and
carries the parent's engagement, because the slow bands do; if babble lowers the
long-run prospect and an answer raises it, this error says so at the act, and with
vcrit_w it enters the mouth's credit beside the fast error. Nothing about words, nothing
from the caregiver's insides. Test 16 of 16 (commit cb328f0). Off by default; runs 49
and 50 (seeds 49, 50, the parent world, the offset the recipe, vcrit_w 1) measure it:
the mouth's yardstick, unheard combinations, engagement (smiles, the parent's turns
away, its hit rate), and at day 20 the corrected value instrument with the ventral
critic's own line against band 5's, judged at 1024 ticks.

Runs 49 and 50 (the ventral critic as a differential head, its error in the credit at
weight 1), stopped at day 12 and 11. Run 49's mouth held (44 of 48 at day 6, all 44
finished; gate 0.5 to 0.6); run 50's gate closed, 0.43 at day 3, 0.34 at day 9, 0.18 at
day 10, its cue answers shrinking to "bal", "be". The credit probe on their day-6 copies
(scratchpad/credit_probe.py, one fast day): the ventral value swung from -137 to +15
across the day on run 49 and from -60 to -5 on run 50, spreads of 59 and 24, against
returns whose spread at 1024 ticks is 5 to 10; the twelve-tick sum of its error after an
act was +0.31 on run 49 against +0.08 after a rest, and +0.07 against +0.03 on run 50,
ten times the fast error's +0.03. The mathematics: a differential (average-reward) head
over features that can move within a tick computes the relative value proper, the
integral of reward above its long-run average, and in a world whose reward rate wanders
through the day (the parent's attention, the habituation) that integral swings by a
hundred; the ladder's slow heads never showed it only because their features cannot
express a swing. The relative value is the right object for the day; the mouth's credit
needs a critic with a definite horizon. So the ventral critic is now discounted at 1024
ticks (vcrit_gamma 1 - 1/1024), bounded, convergent on fixed features, same features,
same credit weight; commit 90b422d, test 16 holds. Runs 51 and 52 (seeds 51, 52)
measure it, 05:26.

The day-15 boundary, the first after a day and a night under the offset: the fixed cues
45 of 48 started, 39 finished; the taught-line cues 93 and 66 of 96 (95 and 75 at day
14); the unheard combinations 99 and 78 of 128 (106 and 96). The page's own count for
day 14: 105 words said after the teacher's utterances, 74% known (53 to 67% on days 12
and 13), 10% fused (8 to 10%). Night 14's dreams still spliced ("irst milk then ball go
d", " go up then inder dog up", 17 symbols long): the cortex that ends a dream has to
have unlearned a seam it held at probability 1.0, and one day of lessons does not do
that. No verdict from one boundary; the yardsticks draw four to six samples a cue and
swing by twenty between days. Days 16 to 18 read it. Day 14's teacher (the planner's
report): 42 utterances, "give little book" against "give big ball", "ball will go down"
against "dog will go up", all six cues answered ("why dog up? " because big dog, "give
big " ball, "I had " milk), 106 smiles, the parent never away, its attention closing at
0.93.

The seam on the raised body, measured (scratchpad/seam_probe.py): on the day-14 copy,
before any offset lesson, the cortex alone after "dog will go down" forecast g at 1.0
and after "you will go in" g at 1.0; on the day-15 copy, after one day under the offset,
a space at 1.0 and d at 0.92, the turn-end itself still at 0.00. On a scratch copy of
the day-15 body, one ended line and twenty waking lessons take the turn-end's
probability at the line's last symbol from 0.00 to 0.93, forty to 1.00: the lesson
works on the big body. In life each utterance's end sits inside the lesson's window of
32 positions for at most two lessons (the offset fires 8 ticks after the last symbol,
the lesson comes every 24), and each end is its own context, so a seam held at 1.0 for
thirteen days comes undone over days. Runs 47 and 48, born with the offset, had their
dreams ending at line ends by day 2; the raised body's night 14 still spliced. The
day-16, 17 and 18 boundaries read the unlearning; the probe runs on each copy.

## The ventral critic, discounted: runs 51 and 52 (2026-09-04, 06:30)

Twenty days each, the offset the recipe, the ventral critic discounted at 1024 ticks
and its error in the mouth's credit at weight 1.

    yardstick                      run 51            run 52            offset runs 47, 48
    fixed cues finished, d6/15/20  38 / 46 / 42      43 / 39 / 43      46/45/43, 47/45/47
    unheard finished at day 20     81 of 128         101 of 128        87, 73
    smiles a day, days 11-20       106               124               84, 88
    known words said a day         275               271               230, 221
    the parent away, times a day   0.2               0.2               1.9, 2.6
    its hit rate                   0.39              0.46              0.36, 0.40
    the gate, days 13 to 20        0.49 -> 0.38      0.55 -> 0.56      0.45 to 0.55
    critic at 1024, day 20, 4 days ventral -0.24     ventral +0.20     (baseline running)
                                   band 5  -0.45     band 5  -0.15

The fixed cues equal; the unheard combinations the best of any run on one seed and in
range on the other; and the engagement is the finding: a fifth to a third more known
words and smiles a day, the parent turning away a tenth as often, on both seeds. That is
the long-timescale reward reaching the mouth: the ventral critic's error after an act
reads +0.10 against +0.01 after a rest (the credit probe on run 51's day-6 copy), the
act's effect on the long-run prospect, and the body speaks more where speaking keeps the
parent. Against it: run 51's gate drifted from 0.55 to 0.38 over its last week, the
shape run 50 collapsed by (0.18 by day 10) under the differential form, slower here; and
the critics' own content at 1024 is no better than band 5's (+0.20 on one seed, -0.24 on
the other). The head stays in the code, its weight 0 in the recipe; whether the drift
settles or collapses needs forty days on two seeds, and the credit's two horizons may
want the ventral share smaller than one. The engagement gain is the first thing in this
record that the long timescale has bought.

## The boundary marks (2026-09-04, 06:40)

The raised body could not unlearn its seam: after two days under the offset its
turn-end probability after a line was still 0.00 (the seam letter gone, a space in its
place), its night-15 dreams still spliced ("dog will go down inder d"), and the night's
lesson on spliced dreams runs at ten times the waking rate. The dream ends where the
cortex expects the quiet, and this cortex, taught the seam for thirteen days, does not
expect it yet; a body born with the offset never had the fight (its untrained cortex
ended dreams early from night 1, run 47's at seven symbols). The hippocampus knows where
an episode ended without asking the cortex: THE BOUNDARY MARK. When the offset fires,
the slot that holds the utterance's last symbol under its context (found by the same
match a merge uses) is marked; a dream that recalls a marked slot ends there, whatever
the cortex thinks; the waking recall never reads the mark. And the first symbol after a
perceived pause marks its slot as a start, and dreams are drawn from starts: with end
marks alone, runs 53 and 54's dreams were four to six symbols long (a dream drawn from a
strong mid-line slot met an end at once) and the night's gauge lagged at 0.5 where the
offset runs stood at 0.8 by day 2. Replay runs from an episode's onset. Commits 2edbcff
and 3ddda26; test 15 now asks that an untaught dream end at the memory's boundary and
that a dream run from a start to an end. Runs 53 and 54 relaunched with both marks
(the recipe's new baseline), runs 55 and 56 (forty days, the ventral credit) relaunched
too, since their first form carried end marks without starts. The served body takes the
marks at the end of day 16 (mark_boundary.sh armed); its night 16 is the first that can
replay whole lines and end them.

The marks took four forms in an hour, each measured on a first night. Ends alone:
dreams of four to six symbols, the night's gauge 0.5 (a dream drawn from a strong
mid-line slot meets an end at once). Starts marked on the first symbol after a pause:
dreams of two symbols, eighteen a night, gauge 0.17, because a line's first symbol
enters under a context faded to nothing and the store's write rule (a key's norm
above nothing) never keeps it, so the mark fell on the previous line's end. Starts
marked on the first memory the store keeps after the pause: right, but drawn only when
four or more starts exist, and the cortex's own expectation of the quiet still ending
dreams, which on a young cortex cut them at two. The final form: starts on the first
kept memory, dreams drawn from whatever starts the store has, ended at end-marked
memories, the cortex's rule retired. On a scratch body: "g will go", " under", "will
go", each ending at its end; the tests' night moves the gauge 0.18 to 0.77 on those
dreams. Commit 2f33e26. Runs 53 to 56 relaunched a third time, 06:47, with this form.

The baseline instrument on runs 47 and 48's day-20 bodies (the offset recipe, corrected
instrument, four days each) carries a finding. Those bodies were saved before the ventral
critic existed, so on the instrument's scratch copies the head is born at zero and learns
only within each instrument day, awake, by its own TD; and it reads the 1024-tick return
at +0.32, +0.64, +0.61, +0.23 (mean +0.45) on run 47 and +0.44, +0.52, +0.39, +0.07
(+0.35) on run 48, where band 5, the ladder's own 1024-tick head with twenty days of
training, reads -0.02 and +0.02, and band 4 at 256 ticks +0.40 and +0.34. A day's
learning on the whole ladder outreads twenty days on the slow band alone: the long
return is linearly there in the union of fast and slow states and not in the slow
state by itself. The ventral critic is the best long-horizon critic in this record, and
it is an organ, not a rule: it stays in the recipe as a learning head (its error into
the mouth's credit still at weight 0 until runs 55 and 56 say whether the gate's drift
settles). Clause 2's form is now the ladder plus the ventral critic; its content at 1024
ticks about 0.4.

At the day-16 boundary, 07:14: the raised body saved (data/body2_before_mark_day16.pt
keeps it as it was), the serve restarted on the code with the marks, nights 16, 265
slots. Its old memories carry no marks; the lines of day 17 will, and night 17 is the
first that replays whole utterances and ends them. Day 16's teacher (the planner's
report): 49 utterances on the three earlier-word pairs, eight cues all taken up, four
clean completions ("bigger dog " up, "why dog up? because " big, "I saw " dog, "big dog
bigger " dog), 122 smiles, no frown, the parent never away, attention 0.53 to 1.0. The
day-17 boundary yardsticks: the fixed cues 46 of 48 started, 42 finished; the taught-line
cues 94 and 64 of 96; the unheard combinations 101 and 79 of 128; the cortex alone within
lines 70 and 63 of 82 (its best); the page's fused share on day 16 0.14 (0.21 on day
15, 0.10 before). The offset alone, three days on, has not moved the fusions; the marks
begin tonight.

**Day 6 of runs 53 and 54, 07:25.** The marks' first cost. The mouth at its best (46 and
47 of 48 finished; 82 and 83 of 128 unheard combinations, against 87 and 73 on runs 47
and 48), the night's gauge 0.8, forty-eight dreams a night of nine symbols, ending at
their lines' ends ('hy dog up? ', 'ig dog bigger ', 'irst up then i'). But the cortex
alone within lines: 40 and 40 of 82 on run 53, 36 and 33 on run 54, where runs 47 and 48
at day 6 read 63 and 51, 62 and 59. Two things in those dreams. Every one begins at its
line's second symbol, because the first enters under a context faded to nothing after a
long pause and the store keeps no memory of it, so the night taught the cortex 'og
will go' forty-eight times and the trace's typed lines begin with 'd'. And a night drawn
from onsets alone has as many windows as the store has onsets, about fifteen, where the
recipe's random draw gave the cortex fragments from every position. Two changes, 07:35,
both readings of what the store already holds. A dream begins with its context's own
last symbol read from the key: a key is the bag before the memory's symbol, its newest
term whole, so at an onset the key names the line's first symbol (the store never kept
the symbol, but it kept the context it made). For that to hold after any pause, the
start mark falls on the memory whose context holds the first symbol, the second
symbol's, whether the first was kept under a faded context (a short pause) or not (a
long one); before, the mark fell a symbol apart between the two. And a night draws half
its dreams from onsets and half from any memory by strength, replay from the beginning
and replay from anywhere. On the test's scratch body the dreams are now 'dog will
go<eot>', 'where bal', 'give mi'. Sixteen tests pass. Runs 57 and 58 carry both changes;
runs 59 and 60 the first symbol with onsets alone, so the two are measured apart; all at
the recipe otherwise, their day 6 due at about 08:00. Runs 53 to 56 continue as they were
(their processes hold the old code) for their days 15, 20 and 40.

**Night 17 of the raised body, the first with the marks, and the day-18 boundary, 08:15.**
Night 17 (07:52, 135 s): forty-eight dreams of mean length 9.8, 470 symbols, drawn from
the day's onsets ('og will', 'ig dog bi', 'hy dog', 'ive big ball go down'), the gauge
0.73 to 0.86, the highest of its nights. The morning's yardsticks against the day-17
boundary: the fixed cues 47 and 47 of 48 (46 and 42); the taught-line cues 95 and 78 of
96 (94 and 64); the unheard combinations 100 and 97 of 128 (101 and 79); the cortex
alone within lines 62 and 59 of 82 (70 and 63); the page's fused share on day 17 0.20
(0.14, 0.21 before). And the seam probe, the cortex alone's probability of the turn's end
after a heard line: 0.00 on all six lines at the day-16 and day-17 boundaries, three days
into the offset; at the day-18 boundary 0.87 after 'scared dog', 0.17 after 'give big
ball', 0.10 after 'dog will go down', 0.00 after the other three. One night of whole
utterances ending at the offset taught the raised cortex what three days of waking
offset lessons had not begun to. With the store's recall in the forecast the end reads
0.00 everywhere: the store never holds the turn-end as a memory, so the turn's end is
the cortex's vote alone against the store's next symbol, which is where the body's own
turn-end will have to be found.

A false alarm, kept for the record. The post-night cues of day 17 read 'dog will ' →
'o down', 'scared ' → 'og then', 'I had ' → 'ilk gone', and I read them as the night's
second-symbol dreams teaching the mouth to skip a first symbol, and armed the serve's
restart on the new code for the day-17 boundary. The same clipped forms stand in the
post-night rows of days 15 and 16, before the marks: the caregiver's 'its_after' record
begins after the answer's first symbols, and the same block's smiles were on whole
words ('go' for 'dog will ', 'ball' for 'where ball? ', 'dog' for 'big dog bigger ').
No skip. The restart went ahead at 08:08 all the same (data/body2_before_prefix_day17.pt
keeps the body as it was; nights 17, 265 slots): the served body now dreams from the
first symbol and half from anywhere, which is the recipe as of cb7591f, and the raised
body's cortex, down from 70 to 62 on the trace after one onset-only night, is the
patient. Day 18's teacher started 08:14.

**Day 6 of runs 57 to 60, 08:45: the first symbol is the whole of it.** With the first
symbol read from the onset's key, onsets alone (runs 59 and 60): the mouth 48 and 45 of 48,
the cortex alone within lines 54 and 51 on both seeds, the unheard combinations 74 (run 60;
run 59 pending), dreams 'first milk the', 'big dog bigger', 'why dog up? '. Half the night
from any memory besides (runs 57 and 58): the mouth 46 and 41, the cortex 61 and 53, 59
and 48, the unheard 78 and 72. Against the onset dreams without the first symbol (runs 53
and 54: 40 and 40, 36 and 33) the first symbol recovers the cortex to the recipe's level
in life (runs 47 and 48 read 51 and 59 there); the mixed draw adds a few hits in the dream
construction and costs the mouth two to four, inside seed noise. The law chooses the
simpler form: a dream is an utterance from its first symbol to its end, drawn from the
onsets the store knows, with replacement when the night has more dreams than the store
has onsets. The mixed draw is removed (commit below). The served body, restarted at 08:08
on the mixed form, takes the onset form at the day-18 boundary; runs 59 and 60 carry the
recipe to day 20 with instruments at 15 and 20, runs 57 and 58 the mixed form beside them.

**The run-on, measured, 09:00.** With the seam's cure in hand for the world's lines, the
body's own turn: after a cue's answer, every body runs on. An instrument on the fast
caregiver logs (scratchpad/runon.py: in the twenty-five ticks after a cue, the symbols
the body adds after its first word, and the rest it takes right after that word), days
5 to 10: runs 47 and 48 (the recipe) 9.7 and 9.8 symbols per cue, resting 1.8 and 2.1
ticks first; runs 53 and 54 (the marks) 9.9 and 9.5; runs 57 to 60 9.3 to 10.6; runs 51,
52, 55 and 56 (the ventral credit at weight 1) 8.6, 9.1, 7.9 and 8.8. Days 11 to 16 the
same. The long credit trims the run-on by a tenth; nothing stops it.

Three reasons, each measured. (1) The store's vote: after a heard line's end the store
recalls the next symbol at confidence 1.0 from other lines sharing the last words ('will
go' continues in 'you will go in'; 'milk' in 'first milk then ball'), and the corpora make
this a true memory, not a generalization error: every cue is a prefix of a longer line and
'dog will go' is both a line and the prefix of 'dog will go down', so the body's 'go
down' is the parent's own line. The cortex's forecast of the end, 1.00 alone on run 59
after 'dog will go', cannot outvote it. (2) The short credit: the smile lands one to six
ticks after the answer, and the body rests under two ticks before its next symbol, so the
answer's reward credits the run-on's first symbols too, act or rest alike; the gate gets
no consistent push and babbles at its base rate, nine or ten symbols in twenty-five ticks.
(3) The long credit: the fast parent's attention (fastlife.py, on the user's word of
2026-09-03) rises 0.05 at every known word the body says, the run-on's 'down' and 'in'
included, and falls 0.04 only at a non-word. A run-on of known words is rewarded at both
timescales by this caregiver, and the ventral critic, reading it rightly, trims only the
non-words. No honest architecture suppresses what its environment rewards. Whether the
parent should want a reply rather than a monologue (attention that falls when the child
talks past its answer or over the parent's turn) is an environment decision and waits
on the user's word; I have not touched the caregiver.

Two honest changes on the body's side, both small. The forecast's vote for the turn's end
is a symbol the mouth can never say; banned outright, a sure forecast of the end raised
the proposal's salience and then the next-best symbol was said in its place. Now, behind
a flag (end_rest, off in the recipe), that vote is the mouth's vote for the rest, and a
mouth that draws the rest has not acted. A rest probe on run 59's day-6 body
(scratchpad/rest_probe.py) says what it would do: at six of the eight cues the store's
answer holds the mass (1.00), at 'give ' and 'scared ' (three and two answers, the store
at 0.68 and 0.70) the end takes the first tick and the answer follows; after heard lines
nothing changes, the store's next symbol holding 1.00. Runs 61 and 62 measure it from
birth. And the night's gauge, which banned the end, counted every dream's last target as
a miss (a ceiling near 0.9 on ten-symbol dreams); it counts the end now, so gauges from
the served body's next restart read a tenth higher than before for the same cortex.

**Night 18 and the day-18 boundary, 09:05.** Night 18 (08:46, 110 s), the served body's
first with dreams from the first symbol (the mixed form it was restarted on at 08:08):
forty-eight dreams, 'scared ', 'big dog bi', 'why dog', 'dog will', 'you wi', mean length
7.7, 368 symbols, the gauge 0.60 to 0.77 (night 17: 0.73 to 0.86 on onset dreams without
their first symbols; the new material is new to the cortex, and the gauge still banned
the end). The serve restarted at 09:01 on the onset form (commit 47e17d0 and after;
data/body2_before_onset_day18.pt keeps the body as it was; nights 18, 261 slots). Day
18's teacher: 35 utterances planned, five cues, 'first up then ' answered 'in' and 'why
dog up? ' 'because', 112 smiles, no frown, the parent never away, its attention 0.62 to
1.0. Runs 53 and 54 to day 20, onset dreams without the first symbol: the mouth 44 and 44
of 48 finished, the cortex alone within lines 44 and 43, 44 and 40 of 82 at day 20 (42
and 43, 41 and 37 at day 15), never recovering; the unheard combinations 82 and 77.

**The day-19 boundary, 09:25.** After night 18, the served body's first night of dreams
from the first symbol: the fixed cues 47 and 46 of 48; the taught-line cues 95 and 80 of
96; the unheard combinations 101 and 93 of 128; the cortex alone within lines 68 and 63
of 82, back from 62 and 59 after the onset-only night to the day-17 level (70 and 63), as
on the fast seeds. The seam probe, the cortex alone: 0.97 after 'dog will go down', 1.00
after 'you will go in', 0.97 after 'give big ball', 0.03 after 'all gone', 0.00 after
'scared dog' (0.87 the day before) and 'little dog had milk'. Three of six lines at the
end's certainty after two marked nights, none at the day-17 boundary. With the store in
the forecast 0.00 to 0.04 everywhere: the store's next symbol from the lines that
continue ('dog will go down' is 'dog will go' and 'down' both) holds its confidence, as
the corpus makes it right to. The page's fused share on day 18: 0.18 (0.20, 0.14, 0.21
before). Day 19's teacher is feeding on the onset form; days 20 and 21 are armed.

**The parent wants a reply, 09:45.** The user's word, "permission granted", on the
environment decision. The rule, in both caregivers (fastlife.py's FastCaregiver and
caregiver.py's, which the served typist subclasses), decided from the page alone: once
the parent's cue is answered, each further word the child adds before the parent's next
turn wears the parent's attention by the babble cost, 0.04, and gets no smile, unless
the words go on completing the cued line ('where ball? ' 'ball under' is a reply, 'ball
dog' is not; the parent knows its own lines, the typist the corpus's heard lines); and a
word said over the parent's own typing does the same, logged 'talked over'. The answer
smile stands. On a fake body: 'dog will ' answered 'go' smiled, 'down' and 'in' past it
cost, 'ball' over the typing cost; 'where ball? ' 'ball' smiled, 'under' passed. Runs 63
and 64, the recipe body (onset dreams with the first symbol, the ventral credit at 0),
born under this parent at 09:43, twenty days, instruments and the run-on at days 6, 15
and 20; runs 59 and 60 are the same body under the earlier parent. The served typist
keeps the earlier parent (the switch off) until the fresh seeds read: the question is
whether the run-on shrinks while the mouth and the unheard combinations hold.

**Runs 61 and 62 at day 6, night 19, and the machine, 09:50.** The end as a rest, from
birth on two seeds: the mouth 44 and 45 of 48 (runs 59 and 60, the same body without it:
45 and 48), the unheard combinations 72 and 80 (74), the run-on 9.5 and 10.0 symbols per
cue (9.9 and 10.6), the cortex alone 47 and 46 of 82 (51 and 51). Neutral where the rest
probe said it would be: the store's next symbol holds every heard line's end, so the
cortex's vote for the rest is never the forecast's. It stays off in the recipe. Under
the parent who wants a reply the body must find a way to stop, and the rest vote is one
of two (the other the gate's own credit), so runs 65 and 66 carry the new parent with the
end as a rest beside runs 63 and 64 without it; runs 61 and 62 stopped at day 6 to make
room, and runs 57 and 58 (the mixed draw, no longer the recipe) at day 15: 45 and 35 of
48 on the mouth there, 59 and 58, 59 and 52 on the cortex. Night 19 of the served body,
its first on the onset form and the first with the dream's end counted by the gauge:
forty-eight dreams ('big dog ', 'I had ', 'first milk then', 'dog will'), mean length 9.5,
454 symbols, the gauge 0.84 to 0.92. Run 63's first day under the new parent: 51 smiles
where the earlier parent gave a hundred, the run-on's words earning none; run 64's 103;
both still run on (9.8 and 10.6 symbols per cue) on day 1, as they must before any
learning.

**The day-20 boundary, 10:05.** After night 19 (the onset form): the fixed cues 47 and 40
of 48; the taught-line cues 95 and 67 of 96; the unheard combinations 98 and 83 of 128;
the cortex alone 63 and 60 of 82. Against days 17 to 19 (42, 47, 46 finished; 64, 78, 80;
79, 97, 93; 70/63, 62/59, 68/63) a day's swing on one body, back at the day-17 level from
day 19's highs, and the machine was carrying ten fast runs through day 19, which pace the
serve's ticks against the typist's clock. The seam probe, the cortex alone: 0.76 after
'dog will go down', 0.86 after 'you will go in', 0.97 after 'all gone' (0.03 the day
before), 0.97 after 'give big ball', 0.00 after 'scared dog' and 'little dog had milk'.
Four of six lines now, against three at day 19 and none at day 17. With the store 0.00
to 0.07. The page's fused share on day 19: 0.19. Day 19's teacher: 35 utterances, six
cues, five answered cleanly ('why dog up? ' because, 'first milk then ' ball, 'big dog '
will, 'dog will ' go, 'where ball? ' ball on the second try), 94 smiles, the parent's
attention 0.53 to 1.0, mean 0.81.

**Day 6 under the parent who wants a reply, 10:45.** Run 64: the mouth 47 and 44 of 48,
the cortex alone 47 and 41 of 82 (a quieter parent talks less: the store 202 at day 6),
the run-on 10.0 symbols per cue over days 4 to 6, resting 2.0 ticks first. Unchanged
from the earlier parent (runs 59 and 60: 9.9 and 10.6). The arithmetic, now measured
rather than argued. Under the earlier parent the run-on's words earned known-word smiles,
a positive short credit at the gate. Under the new parent they earn nothing: the answer's
smile lands on the tick after the answer's last letter, before the run-on begins (the
body rests two ticks first), so the run-on's acts get no credit from it, and the parent's
attention costs them nothing at the gate, because the attention reaches the body only
through the long return, and runs 63 and 64 carry the ventral credit at weight 0. A gate
with zero credit keeps its base rate; only a negative credit closes it. So the test of the
new parent is the test of the long credit: runs 67 and 68, the same parent with the
ventral credit at weight 1, born 10:44; runs 59 and 60 stopped at day 20 (their copies
taken; run 59's day-20 line: the mouth's 'go up ', 'ball u', 'dog up' still running on).
Runs 55 and 56 at day 30 under the earlier parent: the gate 0.535 and 0.534 (0.45 at day
20, the drift gone), the mouth 44 and 42 of 48, the unheard combinations 77 and 72. Night
20 of the served body: forty-eight dreams, mean length 9.4, the gauge 0.91 before the
lesson and 0.92 after, the cortex arriving at the night already knowing its lines.

**Runs 63 and 65 at day 6, 11:10.** Run 63 (the new parent, the ventral credit at 0): the
mouth 45 and 38 of 48, the cortex alone 55 and 45 of 82, the unheard combinations 79 of
128, the run-on 10.8 symbols per cue. Run 65 (the new parent with the end as a rest): 47
and 35, 50 and 44, the run-on 9.4. The mouth finishes fewer answers under the quieter
parent (38 and 35 against 44 to 48), as fewer lines are heard in a day, and the run-on
is where it was. Runs 67 and 68 (the ventral credit at 1) are at day 1. The scale probe
(scratchpad/scale_probe.py) runs meanwhile on the user's question of a local live
scale-up: tick and night times for bodies of 256, 512, 1024 and 1280 width on cpu and
on the Mac's GPU, the machine carrying eight fast runs while it measures.

**The day-21 boundary, 11:20.** After night 20: the fixed cues 42 and 37 of 48; the
taught-line cues 95 and 77 of 96; the unheard combinations 99 and 88 of 128; the cortex
alone 60 and 61 of 82. The seam probe, the cortex alone: 0.96 after 'dog will go down',
0.93 after 'you will go in', 0.88 after 'all gone', 0.68 after 'give big ball', 0.00
after 'scared dog' and 'little dog had milk'; with the store 0.00 to 0.01. The page's
fused share on day 20: 0.14 (0.19, 0.18, 0.20, 0.14 before). The fixed cues' finished
count over days 17 to 21, 42, 47, 46, 40, 37, reads the run-on as much as the answer: an
answer run into its next word without a space ('goinder') is not a finished 'go' to the
probe. Day 20's teacher: 42 utterances, seven cues, seven answered ('first milk then '
ball, 'why dog up? ' because, 'why dog up? because ' big, 'dog will go ' down, 'you will
go ' down, 'big dog bigger ' dog, 'where ball? ' ball), each run on into the parent's
lines, 109 smiles, the parent's attention 0.69 rising to 1.0. Days 22 to 24 are armed.

**Night 21, and the scale probe's first shapes, 11:35.** Night 21 of the served body:
forty-eight dreams, mean length 8.2, 395 symbols, the gauge 0.88 to 0.90. The scale probe
on the loaded machine (eight fast runs beside it, so every figure is inflated, the served
body's real ticks being about a fifth of these): the served shape, 256 wide and 6 deep, 6M
parameters, 49 ms a tick with its waking lessons, a night of about 280 s; 512 wide and 8
deep, 32M parameters, 127 ms a tick, a night of about 1080 s. The 1024 and 1280 shapes
and the Mac's GPU follow. Runs 55 and 56 at day 35: the gate 0.40 and 0.35, down from
0.53 at day 30, wandering rather than settling; day 40 reads at about 11:50.

**The day-22 boundary, 11:55.** After night 21: the fixed cues 42 and 38 of 48; the
taught-line cues 94 and 77 of 96; the unheard combinations 110 and 100 of 128, the best
of the record (79, 97, 93, 83, 88 on days 17 to 21); the cortex alone 59 and 60 of 82.
The seam probe, the cortex alone: 0.99 after 'dog will go down', 0.98 after 'give big
ball', 0.00 after the other four ('you will go in' 0.93 the day before, 'all gone' 0.88):
the end's certainty on a given line moves with the day's lessons, two to four of six
lines above 0.7 on any morning since day 19, none before. The page's fused share on day
21: 0.10 (0.21 on day 15; 0.14, 0.20, 0.18, 0.19, 0.14 since). Day 21's teacher: 35
utterances, five cues typed, 'dog will ' go, 'bigger dog ' up, 'first milk then ' ball,
'why dog ' up, 'where ball? ' ball with a garbled finish, 94 smiles, no frown, the
parent's attention 0.50 to 1.0, mean 0.86. Day 22's teacher is on the earlier parent
still; the switch waits on runs 67 and 68.

**The arithmetic of the run-on, 12:20.** Runs 55 and 56 at day 40, the ventral credit at
weight 1 under the earlier parent: the gate 0.525 and 0.358 (0.40 and 0.35 at day 35,
0.53 at day 30, wandering between 0.35 and 0.54 across forty days), the mouth 47 and 40
of 48, the unheard combinations 76 and 84. No collapse, and under the new parent the
long return is the parent's only road to the gate (runs 63 and 64 at weight 0: the
run-on unmoved at days 6 and 10). Weight 1 is the recipe. Runs 67 and 68 (the new parent,
weight 1) at day 6: the mouth 44 and 40 of 48, the unheard 84, the run-on 10.1 and 9.1
symbols per cue. The long credit on did not move it either. So the credit was read
directly. A probe without a parent (scratchpad/runon_credit.py) read the ventral error at
a tenth to a unit per tick, not the thousandth I had argued, and negative through both
the answer and the run-on: with no parent the errors are the missing smile, the wrong
instrument. A lived day (scratchpad/runon_credit2.py: a copy of run 67's day-6 body under
the fast parent, the gate's own credit recorded each tick) read the right one. Per tick
of grounded credit, the dopamine of the next twelve ticks as the gate sums it: the
answer's letters +0.105, the run-on's letters −0.018, resting after the answer +0.070.
The contingency is in the credit: a rest after the answer beats a run-on letter by 0.09
a tick. And the gate adds to every act a tonic drive of 0.25 less an effort cost of 0.12,
"babble is its own reward", a designer's constant from run 19's day, which pays +0.13 per
act and reverses the order. The run-on has stood at ten symbols per cue on every recipe
and every parent because an intrinsic term outweighs the grounded one. Under the law the
intrinsic act credit goes. Runs 69 and 70 (the new parent) and 71 and 72 (the earlier
parent), the ventral credit at 1, the tonic drive and the effort cost at 0, born 12:18:
the gate on the grounded credit, the spontaneous floor and its own performance error.
Adopted if the mouth survives in both worlds and the run-on shrinks in the new one. Runs
65 and 66 stopped at day 13 (the end as a rest under the new parent: neutral).

**Night 22, and the scale probe's cpu shapes, 12:30.** Night 22 of the served body:
forty-eight dreams, mean length 8.8, 424 symbols, the gauge 0.91 to 0.92, 391 s asleep
on the loaded machine. The scale probe on cpu, eight fast runs beside it (every figure
two to three times what an idle machine would read): 256 wide and 6 deep, 6M parameters,
49 ms a tick with its lessons, a night of about 280 s; 512 and 8, 32M, 127 ms, about
1080 s; 1024 and 12, 179M, 319 ms, about 3900 s; 1280 and 24, 515M, 805 ms, about 12300
s. On the Mac's GPU the two small shapes read slower than cpu (107 and 134 ms a tick, the
overhead of the device dominating small matrices), the nights 350 and 565 s; the two
large shapes follow. Read against the page's 250 ms a tick: on this machine idle, the
32M body fits with a night of six minutes, the 179M body sits at the budget's edge with a
night near twenty-five minutes (an infant's third of life asleep, which the day's 12,000
ticks would allow), and the half-billion shape needs the GPU or a slower tick.

**The day-23 boundary, 13:10.** After night 22: the fixed cues 46 and 42 of 48; the
taught-line cues 94 and 67 of 96; the unheard combinations 106 and 92 of 128; the cortex
alone 61 and 59 of 82. The seam probe: the cortex alone 0.92 after 'dog will go down',
1.00 after 'give big ball', 0.00 to 0.02 after the other four; and with the store in the
forecast, 0.67 after 'give big ball', the first morning on which the fused forecast, the
one the mouth reads, gives the turn's end at all (0.00 to 0.07 on every line since the
offset began). The page's fused share on day 22: 0.16. Day 22's teacher: 35 utterances,
seven cues, seven answered ('why dog ' up, 'give big ' ball, 'first milk then ' ball,
'big dog ' bigger, 'where ball? ' ball, 'bigger dog ' up, 'you saw ' dog), 98 smiles,
the parent's attention 0.56 to 1.0, at 0.8 to 1.0 whenever a cue was completed.

**The scale probe, complete, 13:08.** On the Mac's GPU, the machine at a load of twenty to
thirty with eight fast runs and the probe: 1024 wide and 12 deep, 179M parameters, 251
ms a tick with its lessons and a night of about 850 s; 1280 and 24, 515M, 720 ms and
about 5700 s. Against cpu at the same load, 319 and 805 ms, 3900 and 12300 s: the GPU
halves the large nights and holds the tick near the page's 250 ms for the 179M body,
and the two small shapes are slower on it than on cpu, the device's overhead outweighing
their matrices. An idle machine would read all of these at a third to a half. So, live
and local: the 32M body fits today at four ticks a second on cpu with a night of six to
ten minutes; the 179M body fits on the GPU with a night of a quarter hour, an infant's
share of sleep; the half-billion shape fits only at two ticks a second, which is the
serve's own default period, with a night of an hour and a half. The scale-up's order
stands: 512 wide first, on its own port, from birth, tonight.

**The gate without the intrinsic credit saturates; the effort into the reward, 13:25.**
Runs 69 to 72, the tonic drive and the effort cost at zero: the gate 0.87 to 0.96 on day
1 and 0.94 to 0.98 by day 4, in both worlds; run 72 under the earlier parent 291 smiles
on day 1. The arithmetic, once more. The gate's credit is the critics' error. When a
smile is predicted, the error at the smile is nothing, so the act that earned it is not
credited again; the actor is held where it is by whatever the critics do not predict.
The tonic drive and the effort cost were both outside the prediction, added to the act's
own sum, and they cancelled to +0.13. With both gone what remains outside the prediction
is the surprise of smiles, positive on average early in a day, and the vigor term, which
moves the rate of acting by that average: the gate rises until it acts every tick. And
the earlier form had its own flaw: an effort cost outside the critics is never predicted
away, so with the drive gone every act, the answer's letters included, stood at a loss.
The clean form is a single reward stream: the felt face less the effort of the last
act, convex in fatigue, entering the ear's half of the next tick so that both critics
predict it and the gate reads their error alone. No tonic drive. That is actor-critic
as written, the effort a homeostatic cost the body feels, and it is behind a flag
(cost_in_reward) until measured: runs 73 and 74 (the new parent) and 75 and 76 (the
earlier), born 13:24. Run 64 at day 20 under the new parent with the recipe's gate: the
mouth 47 and 45 of 48, the cortex alone 59 and 59, the unheard combinations 80, the
run-on 9.8 symbols per cue over days 18 to 20, twenty days without a change. The fast
runs print a day line on even days only, so every instrument waiter set on day 15 had
stalled; they read day 16 now, and runs 63 and 64 stop after their day-20 reading.

**Runs 63 and 69 at their last readings, 13:37.** Run 63 at day 20 (the new parent, the
recipe's gate, the ventral credit at 0): the mouth 46 and 45 of 48, the cortex alone 53
and 53, the unheard combinations 86, the run-on 9.7 symbols per cue; run 64 the same day:
47 and 45, 59 and 59, 80, 9.8. Twenty days under the parent who wants a reply, and the
run-on where it was on day 1, on both seeds. Run 69 at day 6 (the new parent, no
intrinsic act credit): the mouth 48 and 48 of 48, the cortex alone 53 and 49, the run-on
20.2 symbols per cue with a rest of a tenth of a tick, the saturated gate speaking every
tick. The two ends of the same arithmetic: an intrinsic drive that outweighs the
grounded credit holds the run-on at ten; no intrinsic term at all, with the cost gone
too, lets the smiles' surprises open the gate until nothing stops it. Runs 73 to 76 carry
the effort in the reward; at day 1 their gates read 0.49, 0.59, 0.47 and 0.59, neither
saturated nor shut. Runs 63, 64, 69 and 70 stopped.

**The day-24 boundary, 13:50.** After night 23 (the gauge 0.92 before and after, the night
finding nothing left to teach): the fixed cues 45 and 39 of 48; the taught-line cues 94
and 59 of 96 (80, 67, 77, 77, 67 on days 19 to 23, a slide to watch: the store fades a
tenth a night and the teachers favour variety over the older lines); the unheard
combinations 107 and 89 of 128; the cortex alone 61 and 57 of 82. The seam probe: the
cortex alone 0.96 after 'dog will go down', 1.00 after 'give big ball', 0.23 after 'you
will go in'; with the store 0.59 after 'give big ball', the second morning the fused
forecast gives the end. The page's fused share on day 23: 0.12. Day 23's teacher: 35
utterances, five cues, four answered ('why dog up? ' because big dog, 'dog will go '
down, 'where ball? ' ball under, 'little dog had ' milk then ba), 89 smiles, the parent's
attention at 1.0 through the middle of the day and 0.40 at its close.

**The effort in the reward at day 6, 14:40.** Runs 73 and 74 (the parent who wants a
reply): the mouth 46 and 43, 46 and 44 of 48; the cortex alone 40 and 33, 45 and 47 of
82; the unheard combinations 64 and 50; the run-on 9.2 and 8.1 symbols per cue, resting
2.6 and 2.8 ticks first. Runs 75 and 76 (the earlier parent): 46 and 29, 47 and 45; 51
and 51, 52 and 48; 84; the run-on 8.8 and 8.2. The gates 0.42 to 0.59 across the four,
held by the felt effort alone, no intrinsic act credit anywhere. The run-on falls from
ten to eight or nine, and falls the same under both parents: the effort does it, and the
parent's contingency has not yet added its own share by day 6. The cortex alone reads
lower than the recipe's on three of four seeds (a quieter body earns fewer smiles, the
parent talks less, fewer lines a day). Runs 67 and 68 at day 20 (the recipe's gate, the
ventral credit at 1, the new parent): the mouth 45 and 47, the run-on 9.7 and 9.8; stopped.
The fast runs print a day line on odd days for some seeds and even for others; the
waiters accept either now. Days 16 and 20 of runs 73 to 76 decide whether the form is
the recipe and whether the run-on keeps falling; the served body waits for that reading.

**The synaptic tag, 14:55.** The last candidate for the parent's contingency reaching the
gate: every act or rest leaves a tag on the gate's weights, (act − p) times the gate's
input, decaying at the ventral critic's own horizon, and the ventral error captures the
tags as it arrives over the following minutes (Frey and Morris's synaptic tagging and
capture: a tag set by activity, captured by later dopamine). Its expected update is the
sum over acts of (act − p) times the long return that followed less the critic's
estimate: the policy gradient at the critic's horizon, where the lesson's twelve-tick sum
could not reach the parent's attention. Behind a flag (gate_slow_lr); the night clears
the tags. On a lived stretch of run 74's day-6 body (scratchpad/tag_probe.py, 3584
ticks): the tag's norm 19 on average, 58 at most, the ventral error 0.82 a tick, so at a
rate of one millionth a day's captured updates move the weights by at most 0.18 against
a norm of 3.7. Runs 77 and 78 at one millionth and 79 and 80 at four millionths, the
parent who wants a reply, the effort in the reward, born 14:52, read against runs 73 and
74 at days 6, 16 and 20. Day 24's teacher: 49 utterances, seven cues, seven answered,
'big dog ' holding two continuations (bigger, will), 82 smiles, the parent's attention
flat near 0.71 all day.

**The day-25 boundary, 15:15.** After night 24: the fixed cues 42 and 38 of 48; the
taught-line cues 95 and 55 of 96 (the slide goes on: 80, 67, 77, 77, 67, 59, 55 since day
19; the corpus branches as the teachers add phrasings, 'dog will go' now heard on to
'down', 'in' and 'up', and the yardstick counts one next word as right, so part of the
slide may be the yardstick's, to be checked); the unheard combinations 95 and 79 of 128;
the cortex alone 61 and 58 of 82. The seam probe: the cortex alone 0.90 after 'dog will
go down', 1.00 after 'give big ball', 0.00 after the other four; with the store 0.19
after 'give big ball'. The page's fused share on day 24: 0.07, the lowest of the record
and half of a week ago: 0.20, 0.18, 0.19, 0.14, 0.10, 0.16, 0.12, 0.07 on days 17 to 24.
The marked nights have been unteaching the raised body's seam on the page itself, slowly
and now visibly. Runs 73 to 76 at days 8 and 9: the gates 0.49, 0.21, 0.41, 0.41, one
seed closing under the parent who wants a reply, 29 smiles on its day 8.

**The effort's scale, 14:50.** Runs 73 to 76, the effort in the reward at the old cost
of 0.12 a symbol convex in fatigue and no tonic drive, read at days 10 and 11: the gates
0.36, 0.18, 0.30 and 0.41, three of four sliding, run 74 mute with six smiles on its day
10 and its cues fusing ('bookwh', 'dogiv', 'byere'), against 0.35 to 0.54 over forty
days on the tonic form. The arithmetic says why: at the rate the old form set, fatigue
rests near 21, where the convex cost makes a symbol 0.64 and a word more than its smile;
the tonic drive of 0.25 was paying for that, and with it gone every act is a loss the
critics learn to predict and the gate learns to avoid, a body that saves its breath. Not
a flaw of the form but of its scale, the one free number: how tiring a syllable is
against a smile. Biology's answer is very little, an infant babbles for hours; at 0.03 a
symbol a word costs a tenth of a smile and fatigue rests near 5, where the convexity is
mild. Runs 73 to 76 stopped (their run-on on days 8 to 11 recorded above the gate's
slide); runs 81 to 84 born 14:50 at 0.03 under the parent who wants a reply, 81 and 82
with the synaptic tag at four millionths, 83 and 84 without, read at days 6, 16 and 20
against runs 77 to 80 (the tag at the old cost, whose gates will slide the same way by
day 8, so their day-6 reading is the one that counts). Run 67's value instrument under
the new parent at day 20: the 4096-tick band's correlation with what followed 0.86, the
1024-tick band 0.31, the 256-tick 0.32; the parent's contingency is foreseen at the
long horizons, and it is the gate that has not been reading it. The parent's attention
reaches the body only as the chance of its next smile, a weak and delayed consequence
(a five-symbol run-on lowers the next known word's smile chance from 0.6 to 0.4); a real
parent's attention is on its face, and the still-face experiments say the infant feels
its withdrawal at once. That would be an environment change (the attention visible on
the face row), and it waits for the user's word.

**The knife-edge, and the ventral critic's sign, 14:58.** Runs 81 to 84, the effort in
the reward at 0.03 a symbol: the gates 0.90, 0.97, 0.92 and 0.97 on their first day,
where at 0.12 they slid to 0.2 by day 10. Stopped at day 1; the reading is complete. Once
the critics predict the cost, the gate's rate goes to wherever a word's chance of a smile
balances its cost, and that is mute at one scale and babble at the other, with no
homeostat between; the old form's tonic drive against the convex fatigue was the
homeostat, the mouth speaking in bouts at the fatigue where the two balance, whatever the
parent did. So the effort form is set aside as the mouth's credit, and the tonic form
stays the recipe (gate_tonic 0.25, the cost in the act's credit), stable for forty days
on runs 55 and 56. Run 67's value instrument at day 20, the last lines: the ventral
critic itself, the whole-ladder head at horizon 1024 whose error the gate's credit
carries at weight 1, reads the return that followed at −0.40, −0.23, −0.20 and −0.28
over its four days, anti-correlated, while the pinned band at 4096 reads +0.86 and the
band at 1024 +0.31. That is not a surprise but a theorem: TD(0) bootstrapped over a
thousand steps sits at a fixed point whose error the horizon amplifies, by (1 − λγ)/(1 −
γ), a thousandfold at λ = 0 (Tsitsiklis and Van Roy 1997), so a head on the ladder's
states can land with the wrong sign; and the gate has been carrying that sign into its
credit since the ventral critic was adopted. The fix biology and the math share: an
eligibility trace on the critic's own weights decaying at the critic's horizon (TD(λ)
with λ = γ, the factor falling to two), the backward view of the discounted return, the
synaptic tag on the critic's side. Drafted behind vcrit_lambda (0 = TD(0); the discounted
head gains a bias for its level, born at zero in older bodies; the night clears the
trace); 16 of 16 tests. Measured first the cheap way: four chained instrument days on run
67's day-20 body with the ventral critic born at zero and carried from day to day, TD(0)
against the trace (scratchpad/vtrace_probe.py), then on fresh seeds if it reads right.

**The critic's rate, 15:18.** The chained instrument on run 67's day-20 body, the ventral
critic born at zero, the other organs fresh each day, the critic carried, judged frozen at
each day's start against the return that followed at its own horizon: TD(0) after one day
+0.33 (learning through its first day +0.65), the trace after one day −0.75 (learning
through its first −0.56); the weights' norm 22 after a day and 37 after two at the shared
rate of a thousandth a tick. Both heads are tracking the last few hundred ticks, not the
state, and in a world that reverts (the parent's habituation to the fiftieth 'dog', the
answer smile once a cue) a recency tracker reads the return with the wrong sign; the
trace, crediting a smile to the last five hundred ticks of states, is the faster tracker,
so it reads worse, and twenty days of TD(0) at that rate came to −0.28 the same way. So
the theorem was not the whole story: a head at horizon 1024 needs its own clock, the
learning rate scaled by the horizon as each band's is by its own (the ladder's principle),
value_lr × 16/1024. Behind vcrit_lr (0 = the shared rate); 16 of 16 tests. Two more
instruments at that rate, TD(0) and the trace, born 15:15, read at days 2 to 5; and the
control with no ventral credit in the gate at all, whose first day earned 240 rewards
against 245 with the credit and a fresh critic, so the 91 of run 67's trained critic is
the trained critic's own doing, to be confirmed on its later days. Runs 77 to 80 at day
6: run 80's mouth 42 and 42 of 48, run-on 8.3 per cue, no different from runs 73 and 74
without the tag; a tag capturing a wrong-signed error was not expected to help, and it
did not. Day 25's teacher: 42 utterances offered in 6 rounds, 6 cues, 3 answered
on-topic ('where ball? ' → 'under', 'you will go ' → 'down'), 87 smiles, 92 misses
almost all 'distracted', the attention from 0.65 to 1.0 by mid-day, a dip to 0.37
before the night, 0.74 at the end; no new word; the planner disclosed checking each
appended line's JSON by reading back that one line, which decided nothing.

**The tag at day 6, 15:25.** Runs 77 to 80, the synaptic tag on the gate at one and four
millionths, the parent who wants a reply, the effort in the reward at 0.12: the mouths 48
and 48, 48 and 48, 47 and 43, 42 and 42 of 48; the run-on on days 4 to 6: 9.5, 10.7, 8.3
and 8.3 symbols per cue, against 8.1 to 9.2 on runs 73 and 74 without the tag. The tag
captured the ventral error, and the ventral error was wrong-signed and growing, so it
could not help and did not; the four stopped at day 6, to be repeated once the critic
reads right. Two of them (78 and 79) had begun the slide of the effort form at 0.12 by
day 6 (smiles 85 and 38, gates 0.59 and 0.49), consistent with runs 73 to 76.

**The day-26 boundary, 15:29.** After night 25: the fixed cues 42 and 38 of 48 (day 25:
42 and 38); the taught-line cues 94 and 56 of 96, the slide stopped where it was (55, 56);
the unheard combinations 91 and 74 of 128 (95 and 79); the cortex alone 58 and 59 of 82.
The seam probe: the cortex alone 0.47 after 'dog will go down' (0.90 a day before), 1.00
after 'give big ball', 0.02 and 0.00 after the rest; with the store 0.17 after 'give big
ball'. The page's fused share on day 25: 0.15, up from 0.07; the record since day 17:
0.20, 0.18, 0.19, 0.14, 0.10, 0.16, 0.12, 0.07, 0.15, a noisy fall. Day 26 began at
15:25. The chained critic instrument, the shared rate: TD(0) frozen at the day's start
+0.33, +0.26, −0.10 on days 2 to 4 with the weights' norm 37, 47, 54; the trace −0.75,
+0.20 on days 2 and 3 with 30, 38; the control with no ventral credit in the gate 240,
253, 235 rewards a day against 245, 216, 248 with it, so the credit with a fresh critic
neither adds nor costs smiles, and run 67's 91 was its trained critic's runaway. At the
horizon rate under Adam the critic moved a twentieth of its level in a day, too slow to
judge, so that pair was stopped; the rate under Adam is a weight speed, not a fraction
of the error corrected. The normalized rule instead (each step corrects a fixed
fraction of the error along its input, the input's energy tracked over a horizon; the
feature energy measured at 101 a tick, 15 a band, 1 for the slowest), the fraction set
by a time constant in horizons, (1 − γ)/τ: behind vcrit_tau, 16 of 16 tests; TD(0) and
the trace at τ = 4 born 15:28 on the same chained days.

**The ventral credit withdrawn, 15:33.** The time-constant pair at four horizons learned
nothing in a day (the weights' norm 0.01 and 0.00): the normalized step divides by the
total feature energy, 101 a tick of which the fast bands hold 15 each and the slowest 1,
so the bias and the slow bands, the directions a 1024-tick value lives in, get a
hundredth of the correction and the fast features the rest, the recency failure by
another road. The shared-rate TD(0) head reached −0.10 on its fourth day with its norm
at 54; the trace read −0.75, +0.20, +0.52; the control with no ventral credit earned
240, 253, 235, 214 rewards a day against 245, 216, 248, 215 with it. So the ventral
critic's error at weight 1 in the gate's credit, adopted on runs 55 and 56's forty days,
is withdrawn (vcrit_w 0; the served body's process has held 0 since its 09:01 restart,
so nothing served changes); the trace and the time constant stay behind their flags,
the theorem stands. The arithmetic that closes the question for now: under the parent
who wants a reply a five-symbol run-on lowers the next known word's smile chance from
0.6 to 0.4, a fifth of a smile over the next thousand ticks, against a return of six to
fifteen with a spread of several; a critic cannot find that shift in three hundred
cues, and the gate cannot learn what no critic can find. A real parent does not leave it
to arithmetic: the attention is on the face, at once. That is the environment decision
already put to the user. Recipe as it stands: the tonic form, the marks and the onset
dreams, the offset, the ventral credit 0, the parent who wants a reply measured and
harmless (runs 63, 64, 67, 68) and ready for the served body at a boundary.

**The consequence, recounted, 15:55.** The user restated the law: grounded reward at the
short and the long timescale, no cheats in the architecture or the environment, perfect
math and biology. The entry above got the arithmetic wrong by a factor of ten. The
parent who wants a reply takes 0.04 of attention per symbol past the answer, so a
ten-symbol run-on takes it from 0.7 to 0.3, and a known word's smile comes with
probability equal to the attention: the next seven known words, the body's usual count
over a thousand ticks, lose about two smiles, not a fifth of one. Two smiles in a
return of six to fifteen with a spread near five is a shift a dozen cues can show. So
the long-timescale reward is in the environment as it stands, and what has not read it
is the critic; every head at that horizon on run 67's body read noise or the wrong
sign, and the instrument that judged them read one swing per day. The environment
needs no change; the proposal of the attention on the face is withdrawn as unneeded
until a right critic has failed. Two instruments corrected: the readings now pooled
over ten chained days (the ventral head frozen at each day's start, and the pinned 1024-
and 4096-tick bands with it), and the form the math has not excluded added: the
average-reward head, its level the body's own reward rate over the horizon rather than
a weight (a weight took forty days at the horizon rate), its error r − r̄ + V′ − V, the
trace at λ = 1 − 1/1024, Adam at the horizon rate (vcrit_diff; 16 of 16 tests). Four
ten-day chains born 15:51 on run 67's day-20 body: TD(0) and the discounted trace at
the shared rate, the average-reward trace and the discounted trace at the horizon rate;
the pooled readings from day 3, the tenth day near 18:30. The two earlier ten-day
chains at the shared rate, frozen per day: TD(0) +0.33, +0.26, −0.10, +0.65, +0.37,
−0.21, −0.07; the trace −0.75, +0.20, +0.52, −0.37, +0.51, −0.24, +0.43, −0.06.

**The day-27 boundary, 16:25.** After night 26: the fixed cues 42 and 39 of 48; the
taught-line cues 94 and 61 of 96, back up from 56; the unheard combinations 107 and 90 of
128, up from 91 and 74; the cortex alone 60 and 58 of 82. The seam probe: the cortex
alone 0.53 after 'dog will go down', 1.00 after 'give big ball', 0.01 and 0.00 after
the rest; with the store 0.56 after 'give big ball'. The page's fused share on day 26:
0.12. Day 26's teacher: 35 utterances in 5 rounds, 8 cues on the page all answered
right ('why dog up? ' → 'because' twice, 'give big ' → 'ball', 'first milk then ' →
'ball', 'where ball? ' → 'ball', 'little dog had ' → 'milk', 'big dog bigger ' → 'dog',
'why dog up? because ' → 'big'), 102 smiles, no frown, no turning away, the attention
from 0.55 to 1.0 and 0.8 to 0.94 at the end; the queue holds a backlog from earlier days
that the typist works through beside the day's rows. Day 27's session began 16:17; the
restart on the parent who wants a reply is armed at its end, and the planners for days
28 to 30 are spawned with the new rule in their brief. The four pooled chains at their
day 6 (frozen, pooled over days 2 to 6): TD(0) at the shared rate +0.32, the discounted
trace at the shared rate +0.17, the average-reward trace at the horizon rate −0.21
(negative every day), the discounted trace at the horizon rate −0.02 (+0.34, +0.05,
+0.48, −0.42, +0.51 by day); the pinned bands pooled −0.20 to +0.36. No form is a
critic yet, and every form swings by the day, which raises the prior question: how
much of the four-minute return is foreseeable from the body's state at all. The
ceiling instrument (scratchpad/vceil_fit.py): a six-day chain with no ventral credit
saving the ladder's states, the return and the parent's attention every fourth tick;
then the leave-one-day-out ridge fit of the return on the states (the ceiling of any
linear head), the attention's own correlation with the return, the slow bands' share
against the fast bands', and how well the eight bands read the attention. Born 16:20,
read near 17:45.

**The four chains at ten days, 16:40.** Pooled over days 2 to 10 (111 thousand ticks
each), the head frozen at each day's start against the return at horizon 1024: TD(0) at
the shared rate +0.19 (by day +0.33, +0.26, −0.10, +0.65, +0.37, −0.24, +0.43, −0.06,
+0.43; its weights' norm 22 to 78); the discounted trace at the shared rate +0.01 (−0.75,
+0.20, +0.52, −0.37, +0.51, −0.21, −0.07, −0.60, +0.06); the average-reward trace at the
horizon rate −0.22 (−0.23, −0.39, −0.29, −0.09, −0.21, +0.07, +0.05, −0.08, −0.60; its
norm 0.04 to 0.26); the discounted trace at the horizon rate −0.10 (+0.34, +0.05, +0.48,
−0.42, +0.30, +0.40, +0.09, −0.48, +0.27; its norm 0.5 to 4.5). The pinned 1024-tick
band pooled −0.14 to +0.18 and the 4096-tick band −0.01 to +0.36, the spread between
chains (whose bodies lived different days) the size of the readings themselves. No head
at that horizon, learned by any of the four rules, reads the return beyond noise on run
67's body. Whether any linear head could is the ceiling instrument's question, its chain
at day 4 of 6.

**The ceiling, 16:50.** Six chained days on run 67's day-20 body with no ventral credit,
the ladder's states, the return at horizon 1024 and the parent's attention saved every
fourth tick (21,560 samples), a ridge fit of the return on the states held out day by
day: all eight bands +0.37, +0.38, +0.32 at three strengths (by day +0.51, +0.29, +0.53,
−0.20, +0.50, +0.63 at the middle one); the slow bands alone (1024, 4096, 16384 ticks)
+0.51 (+0.77, +0.49, +0.61, −0.01, +0.52, +0.70); the fast bands alone +0.33. That is the
ceiling of any linear head on this body, and the learned heads sit at a third of it:
+0.19 for the best, near zero for the ladder's own slow heads, which read only the slow
bands and could reach +0.51 if learned right; the fast bands in the ventral head added
overfit, not foresight. So the long critic's defect is its lesson, and the remedy is a
head on the slow bands with the trace at a slow rate, to be measured the same way. The
second line changes the question of the reply. The parent's attention against the
return: +0.13, +0.11, +0.11, +0.20, −0.24, −0.13, a mean of +0.03; the attention read
from the eight bands, held out: +0.92, +0.92, +0.86, +0.81, +0.95, +0.92, a mean of
+0.89. The body knows the parent's attention almost exactly, and the attention foretells
nothing, because a known word raises it by 0.05 or 0.10 as readily as a word past the
answer lowers it by 0.04, and it saturates at 1. The two smiles I recounted at 15:55
assumed the attention stayed low; it recovers within a word or two. The consequence of a
run-on under the parent who wants a reply is near a tenth of a smile, the first
arithmetic right for the wrong reason, and no critic can be asked to find it. The
environment's rule as written is a token gesture. Biology's parent does more: a child
who talks over its parent gets no reply until it stops, and the reply itself is what the
child wants (infants work for a contingent voice, Goldstein and West 2003). Two honest
roads, both grounded: the environment's, the parent's reply withheld while the child
runs on, and the architecture's, the world's words felt as reward when they come (the
dopamine of information, Bromberg-Martin and Hikosaka 2009), so that a turn given up
pays in what is heard. Neither is taken without the user's word. Runs 85 and 86, the
acceptance pair on the settled recipe with the parent who wants a reply, born 16:39 for
forty days.

**The restart, 17:07.** At day 27's end the served body was saved and backed up
(data/body2_before_recipe_day27.pt), the serve restarted on the working tree (the ventral
head's mask and bias born at zero and unused, the ventral credit 0 as before) and
reloaded at tick 327,272, night 27, store 259; the typist for days 28 to 30 launched with
the parent who wants a reply. Nothing else in the served body's recipe changed. Day 27's
teacher: 49 utterances in 7 rounds, 7 cues landed, 5 answered right ('where ball? ' →
'ball under', 'dog will go ' → 'down', 'give big ' → 'ball', 'big dog ' → 'bigger'), 113
smiles, no frown, no turning away, the attention 0.55 to 1.0. The slow-band heads at
their day 6, pooled from day 2: the trace at the horizon rate +0.25, TD(0) at the shared
rate +0.31, the trace at the shared rate +0.26, all positive against the ceiling of
+0.51. Runs 85 and 86 at day 5: gates 0.54 and 0.51, smiles 143 and 91.

**The day-28 boundary, 17:18.** The probe block ran at the restart, so this is the body
after night 27, before its first day with the parent who wants a reply: the fixed cues
34 and 31 of 48, down from 42 and 39 (the probe is seeded, and the backup taken before
the restart reads the same 34 and 31, the day-27 body 42 and 39: day 27's living and its
night did it, not the restart); the taught-line cues 95 and 65 of 96, the best since day
20; the unheard combinations 92 and 76 of 128; the cortex alone 59 and 57 of 82; the
seam probe: the cortex alone 0.28 after 'dog will go down', 1.00 after 'give big ball';
the page's fused share on day 27: 0.23, the worst in ten days (0.15, 0.12, 0.23 on days
25 to 27), after a day of new phrasings ('bigger dog up', 'dog go down', 'you saw dog').
To be watched over days 28 and 29. The second body, 32.1 million parameters (512 wide,
8 deep, 8 heads, the window 64), born 17:09 on port 8019 as data/body3.pt; its scripted
caregiver began day 1 at 17:12 with the parent who wants a reply, and within three
minutes the parent had turned away (attention 0.145) with two smiles and five words
'talked over': a newborn is all babble, and the first body's first eight days were reared
on the flat rules (196 smiles on its day 2, which the parent's habituation never
allows). So the second body's caregiver was restarted at 17:17 on the flat rules for days
1 to 8, the parent and the reply to come from day 9 as the teacher takes over, the
staging the first body had. The slow-band heads at day 7, pooled from day 2: the trace at
the horizon rate +0.11, TD(0) at the shared rate +0.32, the trace at the shared rate
+0.26. Runs 85 and 86 at day 6: the mouths 45 and 45, 46 and 41 of 48; the run-on 9.6
and 9.2 per cue; the unheard combinations 88 and 76, 91 and 66 of 128.

**The slow-band head, 17:30.** Three ten-day chains on run 67's body with the ventral head
reading bands 5 to 7 alone (1024, 4096 and 16384 ticks), frozen at each day's start
against the return at horizon 1024: TD(0) at the shared rate +0.51, +0.38, +0.43, +0.51,
+0.46, +0.44, +0.46, +0.46, +0.49, a mean of +0.46 and none below +0.38, its weights'
norm 24, 42, 56, 62, 70, 76, 81, 85, 88, 89, settling; the trace at the shared rate +0.58,
−0.13, +0.41, −0.05, +0.47, +0.07, +0.09, −0.01, +0.63, erratic; the trace at the horizon
rate +0.34, +0.19, −0.25, +0.38, +0.06, −0.39, +0.11, +0.19, +0.55, weak and small. The
ceiling for the slow bands was +0.51. So the head that reads the long return is the
simplest one: TD(0) on the slow bands, which the ladder's principle already held (each
horizon its own features), and the ventral head's defect was letting the fast bands in,
whose energy bought recency and not foresight. Adopted as the ventral head's
definition (vcrit_bands 5,6,7; 16 of 16 tests); its weight in the gate's credit stays 0
until the gate is shown to profit from it. Its error on the slow features is the
reward less the long expectation, a baselined reward, which is what the synaptic tag
on the gate needs to capture: runs 87 and 88, the settled recipe with the slow-band head
at weight 1 and the tag at four millionths, the parent who wants a reply, born 17:32 for
twenty days, read at days 6, 16 and 20 against runs 85 and 86.

**Day 28 under the parent who wants a reply, 17:50.** The served body's first day with the
new typist: 8 cues on the page, every one answered right ('why dog ' → 'up', 'give big '
→ 'ball', 'you will go ' → 'down', 'first milk then ' → 'ball', 'dog will ' → 'go', 'big
dog ' → 'bigger', 'why dog up? because ' → 'big') and every one run past; 67 smiles
against 113, 102 and 87 on the three days before; 233 misses, of them 97 'distracted',
77 'talked over' and 42 'past its answer'; the parent's attention ground down to 0.12
and the parent turned away seven times, once near 17:27 and a cluster from 17:43 to
17:48; 35 utterances in 5 rounds, no new word. The body of day 28 is the same body that
answered everything; what changed is that a parent who wants a reply gives a body that
never stops talking a third fewer smiles and turns its back. That is the rule as
written and as measured on the fast seeds (where the days are shorter and the parent's
typing takes fewer ticks); on the served body's page the typing is slow and the body
talks through it. Day 29's boundary and its totals decide whether the served body stays
under this parent or returns to the earlier one while the gate is given a way to read
the consequence (runs 87 and 88).

**The day-29 boundary, 18:23.** Day 28 closed at 74 smiles, 13 turnings-away and the
attention 0.40; after night 28 the fixed cues 34 and 29 of 48 (34 and 31 the day
before), the taught-line cues 96 and 62 of 96 (65), the unheard combinations 93 and 75
of 128 (92 and 76), the cortex alone 53 and 50 of 82 (59 and 57). Flat to a little down,
on a day that gave the body a third fewer smiles. The parent who wants a reply, as
written, costs the served body what shapes its gate and teaches it nothing it can learn
(the ceiling instrument: a tenth of a smile a run-on, and the attention it wears
recovers within a word), so at day 29's end the typist returns to the earlier parent
(scratchpad/revert_reply.sh 29 31, armed); the rule stays on the fast seeds, runs 85 to
88, where it is measured. An hour's incident: the served body's day script waited on any
scripted caregiver and the second body's caregiver on 8019 matched, so the served body
had no teacher from 18:00 to 18:19; the waits are now scoped to each body's port. The
second body's day 1 on the flat rules: 230 smiles, 'why dog up? ' answered 'because
big ', 'where ball? ' 'ball under', 'I had ' 'milk then ball', 'first milk then '
'ball'; its first night at 18:07. Runs 85 and 86 at day 20: the mouths 48 and 42 of 48
(86), the unheard combinations 100 and 88 of 128, the run-on 9.2; run 85's day-20
instruments pending. Runs 87 and 88 at day 6: the mouths 48 and 46, 48 and 44; the
run-on 9.2 and 9.3; the gates 0.51 and 0.53.

**The day-30 boundary, 18:58.** Day 29, an eighteen-minute session after the teacherless
gap: 68 smiles, 4 turnings-away, 35 words 'talked over', 13 'past its answer', the
attention 0.48 at the end; 'dog will go ' answered 'down'. After night 29: the fixed
cues 34 and 28 of 48, the taught-line cues 96 and 61 of 96, the unheard combinations 94
and 75 of 128, the cortex alone 52 and 50 of 82; day 28's fused share 0.13. Two days
under the parent who wants a reply left the yardsticks flat (34 and 31, 34 and 29, 34
and 28 on the fixed cues; 65, 62, 61 on the taught lines) and the smiles a third down.
At 18:45 the revert fired: from day 30 the typist is the earlier parent again (--reply
0). Runs 85 and 86 at day 20, the value instrument: on run 86 the pinned 1024- and
4096-tick bands +0.43 and +0.67 and the ventral head, born before the mask and reading
all eight bands, −0.57 over four days; on run 85 the pinned slow bands −0.13 and −0.18
and the ventral head −0.14. The ladder's own differential slow heads swing by body and by
day; the discounted slow-band head with a bias is the one that read +0.46 every day on
run 67, and runs 87 and 88, born with the mask, answer near 22:00 whether it holds on
fresh bodies. Runs 85 and 86 at days 24 to 26: gates 0.46 to 0.56, smiles 76 to 137. Runs
87 and 88 at day 14: gates 0.51 and 0.49, smiles 66 and 67. The second body's day 2: 322
smiles by its evening, 'why dog up? ' answered 'because big do', 'I had ' 'milk then
bal', 'first milk then ' 'ball undere'.

**Day 30 on the earlier parent, 19:36.** The served body's first day back with the earlier
parent: 90 smiles (74 and 68 on the two days under the parent who wants a reply), all 7
cues right ('why dog ' → 'up', 'give big ' → 'ball', 'where ball? ' → 'ball' twice,
'first milk then ' → 'ball' twice, 'dog will ' → 'go', 'you will ' → 'go'), 109 misses
all 'distracted' or 'late', 14 withheld for a word said twice, no turning away; the
attention from 0.65 to 1.0 in the first ten minutes and down to 0.43 by night; the
teacher's three new lines on 'book' ('book gone', 'I saw book', 'where book?') queued
behind a backlog and not yet heard. Runs 85 and 86 at days 33 and 34: gates 0.55 and
0.49, smiles 91 and 81, the cues clean. Runs 87 and 88 at day 20: the mouths 45 and 43,
47 and 44 of 48; the run-on 8.8 and 8.3; the unheard combinations 108 and 85, 81 and 72
of 128; their value instruments running. The second body at day 3 with 460 smiles since
birth, its third night at 19:31.

**Runs 87 and 88 at day 20, 19:48.** The slow-band ventral head at weight 1 in the gate's
credit with the synaptic tag, two fresh seeds under the parent who wants a reply: the
gates 0.46 and 0.50 at day 20 and 0.52 and 0.48 at days 22 and 23, the mouths 45 and 43,
47 and 44 of 48, the run-on 8.8 and 8.3, the unheard combinations 108 and 85, 81 and 72;
no harm and no turn-taking. The value instrument on their day-20 bodies, four lived days:
the ventral head +0.13, +0.37, −0.27, +0.59 (a mean of +0.21) on run 88 and −0.23,
+0.02, −0.09, −0.42 (a mean of −0.18) on run 87; the pinned 1024-tick band +0.19 and
−0.00, the 4096-tick band +0.29 and +0.24. So the head that read +0.46 every day on run
67's body held fixed, the critic alone learning, is not reliably right-signed on a body
that grows while its gate is fed by the head's own error: the world the critic learns is
then moving. The all-band head on the same instrument read −0.57 and −0.14 (runs 86 and
85), so the slow bands remain the better definition, and the weight in the credit stays
0 as committed: nothing measured yet profits from it. What the day established for
Clause 2 stands as the ceiling instrument put it: a fixed body's slow bands hold +0.51 of
foresight at four minutes and TD(0) on them reaches +0.46; on a growing body the target
moves, and a critic that keeps up with it is the open problem, to be measured with the
chained instrument on a body copied from a live run at several ages rather than one.

**The day-31 boundary, 19:55.** After one day back on the earlier parent (day 30: 116
smiles with the post-night cues, no turning-away): the fixed cues 41 and 33 of 48 (34 and
28 the day before), the taught-line cues 95 and 66 of 96 (61), the unheard combinations
110 and 94 of 128 (94 and 75), the cortex alone 54 and 53 of 82 (52 and 50). Every
yardstick recovered in a day. The two days under the parent who wants a reply held the
body flat and a third poorer in smiles; the day after gave them back. The rule stays on
the fast seeds.

**The acceptance pair at forty days, 20:09.** Runs 85 and 86, born 16:39 on the recipe as
committed (the tonic form, the marks and the onset dreams, the offset, the ventral credit
0, the slow-band head unused) under the parent who wants a reply: forty days without a
collapse on either seed. Run 85: the gates 0.44 to 0.57 across the forty days, the mouth
45 and 45, 47 and 46, 48 and 47, 47 and 46, 45 and 42 of 48 at days 6, 16, 20, 30 and 40,
the run-on 9.6, 9.8, 9.6, 10.6 and 9.1 per cue, the cortex alone 50 and 49 at the end,
the smiles 75 to 143 a day. Run 86: the gates 0.46 to 0.55, the mouth 46 and 41, 48 and
40, 48 and 42, 46 and 42, 48 and 46 of 48, the run-on 9.2, 9.4, 9.2, 9.5 and 9.2, the
cortex alone 58 and 55 at the end, the smiles 63 to 100 a day; at day 20 its pinned slow
bands read +0.43 and +0.67 and the old all-band ventral head −0.57. The recipe stands
accepted for forty days. The day-40 value instruments follow.

**The day-40 value instruments, 20:24.** Run 85's day-40 body, four lived days: the
pinned 1024-tick band +0.49, +0.28, +0.50, +0.41 (a mean of +0.42; at day 20 it read
−0.13), the 4096-tick band +0.77, −0.23, +0.69, +0.79 (+0.51), the old all-band ventral
head −0.46, +0.27, −0.22, −0.24 (−0.16). Run 86's: the 1024-tick band +0.26, +0.11, +0.00,
+0.09 (+0.12; +0.43 at day 20), the 4096-tick band +0.76, +0.82, +0.85, +0.26 (+0.67),
the ventral head −0.60, −0.31, −0.18, −0.35 (−0.36). The ladder's own slow heads,
differential TD at the shared rate on their own band's features, read the long return
right on both forty-day bodies where one of them did not at twenty: they come right
with age, as the day-20 chains on run 67's mature body foretold, while the all-band
head reads wrong at every age. So the long-timescale foresight is in the ladder as
built, arriving with maturity; the open question is only how to hand it to the act.
Runs 85 and 86 stopped at 20:33 with their forty days and instruments complete.

**The age chains, 20:52.** The slow-band head learned from zero on run 85's saved bodies
at ages 6, 16, 20, 30 and 40, each held at its age for five chained days with the head
out of the credit, frozen at each day's start against the return at horizon 1024: age 6
+0.33, −0.32, +0.20, +0.68; age 16 +0.75, +0.75, +0.38, +0.78; age 20 +0.80, +0.65,
+0.61, +0.66; age 30 +0.77, +0.60, +0.82, +0.57; age 40 +0.75, +0.58, +0.71, +0.80
(pooled over the days, which mixes their levels, +0.37, +0.40, +0.57, +0.60, +0.58). So
a newborn's four minutes are unforeseeable and a two-week-old's are foreseeable at
+0.7 a day by a head that has learned for a single day, on any body from 16 to 40 days
old. On run 67's day-20 body the same head read +0.46. The live heads of runs 87 and 88,
learned from birth with their error in the gate's credit, read +0.21 and −0.18 at day 20
against +0.68 for a fresh head on a body of that age: the deficit is the head's own
history (two weeks fitting a newborn's noise into weights of norm ninety) or the loop
of a head shaping the reward it predicts. Runs 89 and 90, born 20:52 on the committed
recipe (the slow-band head learning from birth, its weight in the credit 0), the parent
who wants a reply, twenty days, read at day 20 by the value instrument: a head kept out
of the loop, if it reads +0.6, leaves the history blameless and the loop guilty.

**The reliability gain on fixed bodies, 21:32.** The ventral head's weight in the gate's
credit as its own running correlation between what it foretold and the return that
arrived (vcrit_auto), measured on run 85's day-20 and day-6 bodies held fixed for four
chained days with the credit on: on the day-20 body the head's reliability read +0.65,
+0.10, +0.55 and +0.44 at the days' ends against +0.51, +0.59 and +0.72 for the frozen
head the next day; on the day-6 body +0.50, −0.46, −0.55 and +0.44 against −0.46,
−0.25 and +0.72 (the newborn's copy learned to foresee by its fourth day of standing
still). The gain tracks the truth roughly and shuts the credit off when the head is
wrong, which is its job. The smiles: with the gain 356, 303, 357, 298 on the day-20
body and 380, 377, 337, 358 on the day-6 body; with no credit at all, on the same bodies
and days, 369, 322, 372, 348 and 379, 338, 337, 358. The credit under the gain neither
adds nor costs a smile (my note at 21:14 compared them to run 67's level by mistake). So
the mechanism is safe and honest; whether it gives the act anything the fast credit did
not is the live pair's question. Runs 91 and 92, born 21:13 with the gain from birth:
their first days' reliabilities −0.45 and +0.12, the credit off in the newborns as it
should be; read at days 6, 16 and 20. Runs 89 and 90 (the head out of the credit) at
day 6: the mouths 45 and 43, 48 and 47; the run-on 10.5 and 9.8. Day 32's teacher: 87
smiles, no frown, no turning away, every cue completed.

**The day-32 and day-33 boundaries, 21:45.** Day 31, the first whole day back on the earlier
parent: 126 smiles with the post-night cues, 11 withheld, no frown, no turning away. Day
32: 100 smiles (87 by its teacher's count), 11 withheld, no frown, no
turning away. After night 31: the fixed cues 41 and 33 of 48, the taught-line cues 96 and
63 of 96, the unheard combinations 96 and 77 of 128 (110 and 94 the day before), the
cortex alone 55 and 53 of 82; day 31's fused share 0.12; the seam 'all gone' with the
store 0.98, 'give big ball' 0.92. After night 32: the fixed cues 41 and 33 again, the
taught-line cues 96 and 63 again, the unheard combinations 99 and 82, the cortex alone
52 and 52; day 32's fused share 0.07, level with day 24's, the lowest of the record. Three
boundaries at 41 and 33: the body is holding its level on the earlier parent, neither
climbing nor slipping, its smiles a hundred a day. The unheard combinations came off
their day-31 high and stayed there. Day 33 began 21:37.

**The word for the reply, 22:24.** The user: "you can do this. Turn-taking. The run-on is
still about nine symbols past the answer on every recipe... it needs either a parent
whose reply is withheld while the child runs on or a body that feels the world's words
as reward. Both are grounded; both are your call." Both taken, each behind a flag. The
environment's road: the parent who wants a reply now answers when the child has
finished. Its smile for the answer and its recast of the cued line in full ('why dog
up? ' → 'because' → "why dog up? because big dog") come after the child's quiet, every
symbol the child adds before that postpones them, and after the cap it replies anyway;
the reply is spoken at once, outside the pace. The architecture's road: each symbol the
world types is felt as reward beside the face (world_r, 0.1 a symbol, a line of fifteen
near a smile), the caregiver's voice as an infant's primary reward, so a turn given up
pays in what is heard and the parent's pace becomes a reward rate the slow critics can
foresee. The first smoke on a newborn with the parent's wait at eight ticks: two answers
of three hit the cap of 180 ticks with 87 and 96 symbols said before it. So I measured
the served body's own rests from its page: at 33 days it speaks on half of all ticks
(5,390 symbols in 10,551), one rest in a hundred reaches eight ticks, and after the
parent's utterances a quiet of eight comes within 180 ticks a third of the time (median
116 ticks) where a quiet of four comes every time (median 22). The run-on was never nine
symbols; the body never stops. The parent's wait is four ticks, a second, the transition
infants and parents make. The arithmetic at the dopamine band with the reply a second
off: the answer smile (2 then 4) and a recast of twenty symbols at 0.1 make a prospect
near 6; a run-on symbol at the pause's start costs its one-tick delay, 0.29, and one two
ticks into the pause costs the reset, 0.64, against the act's own margin of 0.13 to
0.48. A pause, once begun, pays to keep; whether the gate finds it is the runs'
question. Runs 93 and 94 (the reply withheld) and 95 and 96 (and the world's words),
born 22:22 on the recipe, twenty days, read at days 6, 16 and 20 against runs 89 and 90.
The organ tests 16 of 16. The user also asked whether Sonnet is a good enough teacher
and gave me the teacher's method to tune without asking: it is adequate for the role as
built (day 33: six of seven cues answered, no rule broken); the bottleneck is the
environment's contingency, now under measurement, and the curriculum's pace, which I
will tune from day 36.

**The day-34 and day-35 boundaries, and the fast parent's double day, 23:35.** Day 33: 98
words said, 47 known, the fused share 0.08; day 34: 102 smiles, no frown, one turning-away
in the post-night cues (the first since day 29), 88 words said, 49 known, fused 0.09. After
night 33: the fixed cues 37 and 33 of 48, the taught-line cues 96 and 64, the unheard
combinations 90 and 72, the cortex alone 50 and 50. After night 34: 36 and 33, 96 and 62,
94 and 75, 54 and 51. The full count holds at 33 for the fifth boundary; the started count
has slipped 41, 41, 41, 37, 36. The first readings of the reply withheld at day 6, four
seeds: the symbols said before the parent's reply 16, 25, 19 and 20 (from 3, 36, 18 and 8
on day 1, rising with the newborns' speech to 22 to 26 by days 4 and 5), every answer
yielded before the cap, the mouths 47 and 41, 47 and 45, 47 and 43 of 48, the words past
the answer 9 to 12 per answer against 13 to 17 for runs 89 and 90. And a bug older than
tonight: under the parent with moving attention the fast day loop never noticed the body's
night, which runs inside a tick, and the parent talked on through it, so every logged day
held two nights (run 89's days print 1, 2, 4, 6, 7, 9...), the plan and the post-night cues
came once per two nights, and a "twenty-day" run lived near forty nights. Ages by nights
were right, so the readings at days 6, 16, 20 and 40 stand as readings of bodies that
age; but each night had half the cues, and the served typist, which polls the body's
sleep, never had the fault. Fixed at 23:33 (the day ends at its night); runs 93 to 96
stopped at day 8 with their day-6 readings kept, and the comparison relaunched like for
like at 23:34 on the fixed parent: runs 97 and 98 the reply rule alone, 99 and 100 the
reply withheld, 101 and 102 the reply withheld with the world's words, twenty days each,
read at days 6, 16 and 20.

**History, not the loop, 23:53.** Runs 89 and 90 at day 20, the slow-band ventral head learned
from birth with its error never in the gate's credit, read by the value instrument over four
lived days: −0.52, −0.35, −0.26, −0.52 (a mean of −0.41) and +0.03, +0.03, −0.47, −0.51
(−0.23), while the same head learned fresh for one day on a body of that age reads +0.7. The
loop is innocent; the head's history is the defect: twenty days of Adam steps on a newborn's
noise, weights of norm near ninety, that a day's learning cannot undo. Runs 91 and 92, the
gain from birth, agree: their reliability wandered from −0.41 to +0.29 through day 18 and
never settled. The pinned slow bands on the same bodies read as before, band 5 +0.11 and
+0.15, band 6 +0.37 and −0.63. Biology's remedy is forgetting: synapses decay, and a critic
that forgets at the horizon of days tracks a body that changes over days. THE FORGETTING
HEAD (vcrit_forget): the head's weights decay toward zero at a time constant of 24,000
ticks, two days, decoupled from the lesson as in AdamW, the level undecayed, so the head is
always the last two days' head. The organ tests 16 of 16. Runs 103 and 104, born when runs
91 and 92 end, carry it from birth with the reliability gain at weight 1 on the fixed parent
under the reply rule: their reliability by day and the value instrument at day 20 decide.

**The pause, seen by the critic, 00:56.** The six seeds' day 6 on the fixed parent: the words past
the answer 14, 16 and 12 per answer on baseline seed 97 and 10, 13 and 10 on seed 98 (days 4
to 6); under the reply withheld 8, 9 and 10 (seed 99) and 14, 11 and 12 (100); with the world's
words 10, 9 and 12 (101) and 10, 11 and 9 (102): 12.7 against 10.5, inside the seed noise. Every
answer yielded before the cap, after 9 to 37 symbols with no trend down; the mouths 47 to 48
started and 34 to 47 full. So I asked the critic itself. The pause probe: run 100's day-6 body,
a lived day under the reply withheld, and at every tick while the parent held its reply the
dopamine band's value, its error, whether the mouth acted, and the quiet count. 438 ticks
held. The value +0.14, +0.11, +0.13, +0.17 at zero to three quiet ticks after a rest, +0.11,
+0.17, +0.09, +0.11 after an act, where a smile of two a tick away would put it near +1.9;
the error −0.008 after a rest and −0.007 after an act. The critic does not foresee the reply,
so the delay the withheld reply imposes never reaches the gate's credit, and the arithmetic
of 22:23 assumed a foresight the body has not got: a band that averages the cortex over
sixteen ticks barely moves across a four-tick pause, and ten replies a day are few lessons
for a linear reading of "I have answered and I am quiet". Two roads from here, both grounded,
both measured rather than argued. The mask: the world's word is not felt as reward on a tick
after the mouth acted (vocalizing suppresses the auditory cortex, Eliades and Wang; the
babble masks the voice), so a symbol said over the parent forfeits, in the reward itself and
at once, what would have been heard; runs 105 and 106, born 00:54, carry it with the reply
withheld and the world's words at 0.1, in place of runs 99 and 100, stopped at day 7. And the
probe again over every band and the gate's own probability by quiet count, to see whether a
faster band resolves the pause and could carry the credit. The reliability gain's pair closed:
run 92 at day 20, the head from birth under the gain, −0.53; the gain kept that credit off and
did nothing else. Run 91's turn-taking: 9.2 per cue at day 20, the same as every recipe.

**No band foresees the reply, and the seam has no memory, 01:14.** The second pause probe, every
band on run 100's day-6 body across 438 held ticks: the values +0.00, +0.05, +0.13 and
+0.85 at horizons 1, 4, 16 and 64 ticks, flat across the quiet count and the same after a
rest as after an act; the gate's own probability of acting 0.55 to 0.57 whatever the state.
The pause cannot come from the critics at ten replies a day. The end as a rest (end_rest,
runs 61 and 62, neutral) failed for a reason I can now name: the recall after a whole line
is not a memory of the line's end but a confident false match. The store keeps only what
the world said next; between the parent's utterances the world's context fades by 0.8 a
tick, so after the fast parent's pauses of 240 ticks nothing is written under the line's own
context, and the query at the line's end lands on the nearest partial context with the
mass on one symbol. I wrote the marks into the live recall (recall_end: the seam slot, an
utterance's first symbol kept under the last line's faded context, recalls the turn's end
instead of that symbol; the organ tests 16 of 16) and found on a newborn under the fast
parent that no such slot exists to be marked. Under the served typist's pauses of twelve
ticks (the context at 0.07) they do exist, so the flag can only be tested there, and the
question underneath is older: the world's quiet is not a memory, and writing the turn's end
into the store failed after cues (runs 45 and 46). The cue's quiet is the child's turn; the
line's quiet is the end; the body must learn which from the page. Left for a clear head. The
mask pair (105 and 106) and the forgetting head (103 and 104: reliability −0.41 and +0.30 at
day 5) run on. The served body's day 36: 109 smiles, one turning-away at the stutter, its
first "cat" heard; the day-36 boundary 31 and 28 of 48 after 36 and 33, days 34 and 35's
teaching, the first fall of the full count in six boundaries.

**The typist ate the curriculum, 01:52.** Day 37's planner (the word of the day "cup", fourteen lines
of it, four cues all completed, 105 smiles, no frown, no turning-away) reported that none of
its plain lines reached the page before the night, only its cues. The cause is in the typist,
not the planner: under the parent with moving attention the rule "answer a smiled word with a
line that holds it" replaced the planner's next plain line with a heard line holding the child's
last smiled word, and with a hundred smiled words a day that was nearly every line. Since the
parent came to the served body its teachers' new phrasings and words have rarely been typed;
the corpus grew by one word in thirty days because the typist would not say the others. Fixed
at 01:52 (commit ac2993b): the holding line stands in only for the typist's own filler, the
random heard line it types when the planner has nothing; the planner's lines go to the page as
planned. In force from day 38's typist and the second body's day 10. The day-36 planner had
seen the same and read it as a backlog. The turning-away of day 36 came at the stutter
("downg ing ing in", the hundredth "in" of the day). The mask pair at day 6 (runs 105 and 106,
the world's words at 0.1 forfeited on a tick after the mouth acted): the words said over the
parent 89 to 106 a day, as the baselines' 81 to 113; the words past the answer 9 to 11 per
answer, as the baselines' 10 to 16; the mouths 48 and 42, 41 and 36. At 0.1 a symbol the
forfeit is below the act's own margin (the credit difference near 0.14 against a tonic drive
less effort of 0.13 to 0.23), so runs 107 and 108, born 01:45, carry the world's words at 0.3
with the mask, each on one thread to measure the compute as well.

**Day 16 on the fixed parent, 02:13.** The words past the answer per answer over days 14 to 16:
the baselines (the reply rule alone) 15.7, 15.5, 14.0 on seed 97 and 12.2, 10.1, 11.4 on seed
98; the reply withheld with the world's words at 0.1, 9.2, 9.6, 10.0 on seed 101 and 9.2, 9.2,
9.5 on seed 102. The words said over the parent a day: 85 to 102 and 77 to 89 against 89 to
105 and 73 to 99. The symbols before the reply on the world pair 16 to 50, every answer but
two yielded before the cap. The mouths 48 and 46, 47 and 46 against 46 and 44, 47 and 43. So
the world's words at 0.1 cut the words past the answer by a quarter, two seeds against two
with a spread of four between the baselines, and did nothing to the talking over; the mask at
0.1 (runs 105 and 106, day 9) reads the same as without it. Not turn-taking. Runs 101 and 102
retired at their day-16 copies; the record stands for the value instrument later. The served
body's day-38 boundary: 29 and 26 of 48, the sixth fall, the sample 'I had ' answered "go
downg ing i" on the memory, a stutter of "down" into "going" into "in" that the "going" frames
of the last days feed; day 39 and 40's planners are told to leave "going", "go down" and
"down" alone this week and to answer a "downg" with a clean short line. The second body's day
9 under its first teacher: 61 smiles, all six cues answered, "down" placed four times, two
turnings-away in the session and two in the post-night cues. One thread a run: 245 to 310
seconds a day where the four-thread runs beside them take ten to fourteen minutes.

**The mask's credit, 02:26.** The world's words at 0.3 with the mask, run 108's day-6 body,
and the baseline's, run 98's, each living 3,000 ticks under the fast parent with the gate's
own credit recomputed at every tick and sorted by whether the parent was typing. Inside the
parent's typing, the mask body: an act −0.09, a rest +0.10, a difference of 0.19 in favor
of listening; the baseline: −0.14 and −0.15, the act costing nothing. Outside, both bodies
−0.01 either way. The first mechanism tonight whose consequence reaches the gate's credit
with the right sign at the moment of the decision. Yet at day 6 the words said over the
parent are 95 to 118 a day on the mask pair (107 and 108), as on every other seed, and the
arithmetic says why: the parent types on one tick in twenty, the three-factor lesson's
step on those ticks moves the logit near 0.07 a day in the listening direction, and the
vigor term (the credit itself, whatever the act was, at weight 1) pushes the other way
while the reward flows. A week or two at that rate; day 16 near 03:30 says whether the
count has begun to fall. The mask at 0.1 (runs 105 and 106) cannot, at a third of the
difference. The world's words at 0.1 without the mask (101 and 102, retired at day 16)
cut the words past the answer by a quarter and the talking over not at all.

**The gate cannot learn what its inputs say, 03:20.** The mask at 0.3 at day 16 (runs 107 and
108): the words said over the parent 102, 114, 104 and 95, 94, 85 a day, as the baselines'
99 to 104 and 78 to 95; the gate's probability of acting inside the parent's typing 0.53
against 0.51 outside on the mask body, 0.49 against 0.50 on the baseline. Sixteen days of a
right-signed credit of 0.19 and the gate is blind to the state it applies in. So I asked
whether the state is in the gate's inputs at all: a Fisher reader on the cortex state and
the five feelings tells inside from outside at 0.999 on held-out ticks, and the gate's own
weights read it at 0.65 (0.16 on the baseline, which speaks more inside). The inputs carry
the state; the lesson cannot find the weight. The geometry says why: the inputs' mean has
norm 2.33, the difference between the inside and outside means 0.45, and the cosine between
the inside mean and the overall mean 0.98. The three-factor rule's systematic gradient
points along the inside mean, which reads the state at 0.58; the centered difference reads
it at 0.998. Uncentered, every contingency the credit carries becomes a push on the common
bias, which the running baseline then cancels, and the selective direction is learned only
from the two percent of the gradient that is not the common mean. THE ADAPTED INPUT
(gate_center): the gate reads its inputs relative to their running mean, a time constant of
1,024 ticks, as sensory neurons adapt to their mean input; the mean is saved with the body.
The organ tests 16 of 16. Runs 111 and 112, born 03:19, carry it with the reply withheld,
the world's words at 0.3 and the mask; runs 113 and 114 carry it on the baseline alone, to
see what centering does to the mouth by itself. If the first pair's talking over falls where
the mask pair's did not, the actor was the bottleneck all night. The forgetting head in the
credit (103 and 104) read +0.21 and +0.13 at day 16, −0.15 and −0.26 at day 17; out of the
credit (109 and 110, day 9) −0.12 and +0.01: neither reads the return yet. The served body's
day 38 under the fixed typist: 148 smiles, no frown, no turning-away, "bed" learned in a day
and nine new lines heard. The second body's day 10: 39 smiles, two turnings-away, all three
cues answered; its fused babble wears the parent as the first body's never did.

**The lesson replayed, and the ear, 04:16.** The adapted input at day 6 (runs 111 and 112): the
words said over the parent 85 to 98 and 84 to 95 a day, no fall, and the same on the baseline
pair (113 and 114: 76 to 103, 88 to 94). So I replayed the gate's own lesson offline on a
quarter day recorded from run 112's day-6 body: the features as the gate saw them, the acts,
the dopamine, the belief credit and the fatigue, and whether the parent was typing; then the
three-factor rule exactly as life.py runs it, from the body's own weights, under the recipe's
optimizer and others. The recipe (SGD 0.05, the vigor term at 1): the logit moved −0.052
inside the parent's typing and −0.056 outside, a selective difference of +0.004; at ten times
the rate both moved ten times as far and the difference stayed under 0.04; Adam the same. The
lesson learns the global rate and nothing selective, whatever the credit says. Then the
remaining grounded remedy: a unit input for "the world's symbol arrived this tick", the ear,
which the striatum has and the face already gives the cortex. With it, under the recipe's
lesson, the logit moved +0.09 inside against outside in a quarter day, the wrong way: the
credit runs high where the parent's rewarded words are, and the vigor term (the credit
itself, whatever the act was) acts more wherever the credit is high. With the vigor term off
and Adam at 1e-3 the same input moved −0.08 a quarter day, a third of a logit a day in the
listening direction; SGD without vigor −0.02. THE EAR (gate_ear: the world's symbol this tick
and its own act last tick, two gate inputs born at zero, an older body widened on load) and
the gate's optimizer as a choice (gate_opt adam) are written; the organ tests 16 of 16.
Runs 115 and 116, born 04:15: the ear, the vigor term off, Adam, the adapted input, with the
reply withheld, the world's words at 0.3 and the mask, against 111 and 112; runs 117 and 118
the same actor on the baseline environment, against 113 and 114 and the baselines. The
forgetting head out of the credit (109 and 110) read −0.52, −0.12, +0.07 and −0.14, +0.01,
−0.12 at days 14 to 16: the head that learned +0.7 in a day on a body held fixed does not
learn on a body whose cortex changes every night; the value instruments at day 20 close the
record and the design waits for a clear head. The served body's day 39: 161 smiles, "eat" in
four frames, a 49th known word, seven new lines heard, no turning-away.

**The ear's first week, 05:05.** Runs 115 to 118 at day 6 (the ear, the vigor term off, Adam
at 1e-3, the adapted input; 115 and 116 with the reply withheld, the world's words at 0.3 and
the mask, 117 and 118 on the baseline). The gate's weight on "the world's symbol this tick":
−0.40 and −0.53 where the mask makes talking over cost something, +0.07 and +0.16 where it
does not. The first actor of the night to learn a state its credit named, and with the right
sign; a tenth quieter during the parent's lines so far, the weight still growing. The words
said over the parent on days 4 to 6: 73, 82, 103 and 63, 72, 81 against 84, 90, 100 and 82,
77, 84 on the baseline actor, so a fifth fewer on one seed and none yet on the other; the
mouths 47 and 40, 48 and 47. The other ear weight, "my own act last tick", reads +0.86 to
+0.98 on all four bodies: the gate has learned that having just spoken it should speak
again, the bout made explicit, the run-on's own spring, which the lesson rewards because the
belief credit and the tonic drive pay every act inside a bout. Day 16 near 05:50 says whether
the listening weight keeps growing and the count falls. The served body's day 40: 157 smiles,
no turning-away, "sit" learned in a day, a fiftieth known word, 96 lines heard; and the
day-41 boundary 40 and 36 of 48 after 31 and 28, the best since day 31, the taught lines 88
and 61, the unheard combinations 95 and 81, the cortex alone 42 and 41. The second body's
day 11: 52 smiles, no turning-away, its cues answered; its chain had ended with day 11 and
it stood teacherless from 04:38 to 04:41, when days 12 to 14 were launched.

**The adapted input alone is nothing, and the ear at day 8, 05:30.** Runs 111 to 114 at
day 16 (the adapted gate input on the recipe's actor; 111 and 112 with the reply withheld,
the world's words at 0.3 and the mask, 113 and 114 on the baseline): words said over the
parent on days 14 to 16, 83, 90, 96 and 94, 104, 96 against 81, 80, 89 and 79, 89, 93; the
run-on 9.5 and 8.7 against 9.9 and 10.1 symbols past the answer; the mouths 47 and 47, 46
and 48 of 48; the unheard combinations 89 and 97, 99 of 128. Centering the inputs of a gate
whose lesson pushes the wrong way changes nothing, as the replay said it would: the vigor
term, not the geometry, was the wall. Runs 105 and 106 at day 20 (the world's words at 0.1
and the mask on the fixed parent): talked over 95, 90, 75 and 93, 97, 87 on days 18 to 20,
the run-on 9.4 and 9.6; the record closes with a tenth of a smile per heard symbol doing
nothing a critic could carry. The ear at day 8: the weight on "the world's symbol this tick"
−0.55 and −0.72 where the mask bites (from −0.40 and −0.53 at day 6), +0.02 and +0.15 on the
baseline (from +0.07 and +0.16): growing about 0.08 a day on the bodies whose credit names
the state, still nothing where it does not. The bout weight +0.93 to +1.07 on all four. The
words said over the parent on days 6 to 8: 103, 80, 62 and 81, 85, 80 (the ear, the mask)
against 89, 96, 85 and 95, 73, 55 (no ear, the mask) and 100, 77, 88 and 84, 90, 74 (the
ear, no mask): no separation the day-to-day noise of twenty does not swallow. The arithmetic
of what is coming: Adam at 1e-3 can move a weight a thousandth a step, and the ear's input is
on only during the parent's lines, about 150 to 190 ticks a day, so the ceiling is 0.15 to
0.19 a day and the weight moves at half of it; a logit of −0.7 against a bout weight of +1.0
takes a mid-bout p(act) from about 0.6 to about 0.45 during the parent's lines, a fifth
fewer interruptions at best, which is what the noise hides. To yield the weight must reach
−2 or so, twelve more days at this pace. Whether to wait or to raise the gate's rate is the
question day 16 answers with the slope. Both chains armed: the served body's days 44 to 46
after the day-43 teacher ends, the second body's days 15 to 17 after its day 14; the served
body's day 41 at 62 smiles half-way, the second body's day 12 at 21.

**The night-transfer instrument, and the critic's defect found, 06:00.** The question left
open at 05:05 was why a ventral head learned live reads the return with the wrong sign when
the same head learned on a body held fixed reads +0.7, and whether the nightly change of the
cortex was the obstacle. The instrument (scratchpad/night_transfer.py): one page of 9000
ticks lived by the served body's day-40 self under the fast parent (143 felt rewards), then
replayed tick for tick, teacher-forced, through fresh copies of that body and of the bodies
of days 41, 38 and 30, so every body sees the identical stream and the identical rewards and
only the body differs; a ridge head from the bands to the return at horizon 1024, fit on the
first 60 percent of day 40's ticks and read on the last 40 percent of every body's. The
fast bands drift: cosine at the same tick 0.70 after one night, 0.40 after ten. The slow
bands do not: 0.99, 1.00, 1.00 after one night, 0.98, 1.00, 1.00 after ten. The head from
the slow bands reads +0.50 on day 40's held-out ticks and +0.51, +0.49, +0.51 on the bodies
of days 41, 38 and 30: a value readout learned on one day is worth exactly as much ten
nights later. The nightly change was never the obstacle. Then the learning rule, on the same
cached features (scratchpad/nt_heads.py): the live rule, semi-gradient TD(0) with Adam at
1e-3 over the slow bands, reads +0.52 after one pass over a third of a day and +0.53 on the
next night's body, when it reads the raw bands; the same rule reads −0.56 and −0.55 when it
reads what the live critic is given, the bands centered on a running mean at 1024 ticks; a
ridge head on the centered bands +0.01 that day and −0.44 the next. That centering, written
on 2026-09-03 so the differential heads' free constant had no direction to walk in, was
applied to the ventral critic too; a running mean at 1024 ticks tracks a band whose clock is
4096 or 16384 and leaves a thousand-tick recency residual, so the head learned recency and,
in a world that reverts, the wrong sign. Every wrong reading since run 49, the runaway, the
withdrawn weight, the trace, the normalized step, the reliability gain read at zero, the
forgetting head, was this one line. Discounted with a bias the head needs no centering:
vcrit_center (1 = the old form, 0 = the raw state) is written, the organ tests 16 of 16.
Runs 125 to 128 are armed for the four slots the ear pairs free near 06:20, on the ear actor
with the reply withheld, the world's words at 0.3 and the mask: 125 and 126 with the
uncentered head out of the credit, its reliability logged by day; 127 and 128 with the head
in the credit at 1.0 through its own measured reliability (the gain the code already has).
The gate's-rate pairs 121 to 124 (Adam at 3e-3 and 1e-2 on the ear actor) began 05:43 in
the slots of 111 to 114. The typist's filler was the day's other find: the two 'go down'
lines that opened the served body's day 41 were the typist's own, a random heard line said
while the planner's first row was still unwritten, and the most-heard lines are the oldest
frames; the filler now repeats one of the last two dozen planned lines, a heard line only
when nothing was ever planned (from day 42, 05:59). The served body's day 41: 146 smiles, no
turning-away, "run" learned, 51 known words, 101 lines heard. The second body's day 12: 40
smiles, one turning-away, its day only thirty minutes long because its sleep pressure was
already at 10186 of 12000 when the teacher's session began.

**The ear listens, 06:15.** The ear's weights at night 14: "the world's symbol this tick"
−0.94 and −1.07 on the mask bodies (115 and 116; from −0.55 and −0.72 at day 8), −0.11 and
+0.04 on the baseline (117 and 118); "my own act last tick" +1.25 and +1.39, grown too (from
+0.96 and +1.07), so mid-bout during the parent's line the two nearly cancel and a linear gate
cannot form "the parent is speaking and I am mid-word". What it can form is the start: the
start/continuation probe (scratchpad/pact2.py, 3000 ticks under the fast parent with the
lesson off, on copies) reads, inside the parent's typing, p(start) 0.23 and 0.15 and
p(continue) 0.49 and 0.31 on the mask bodies against p(start) 0.43 to 0.45 and p(continue)
0.63 to 0.68 outside, and against 0.34, 0.44 and 0.60, 0.63 inside on the baseline actor,
0.47 to 0.49 and 0.49 to 0.54 on the no-ear bodies of 111 and 112 at day 20. Acted inside
the parent's lines: 0.30 and 0.18 of the ticks against 0.48 to 0.52 for every body without
the ear's credit. The body with the ear starts a bout while the parent speaks a third as
often as a body without it, and speaks over the parent half or a third as much, from a credit
of a tenth of a smile per heard symbol that the mask makes it forfeit: turn-taking learned by
the gate's own lesson from the world's contingency, no rule written. The gate's-rate pairs at
day 2: the listening weight −0.57 and −0.50 at Adam 3e-3 (where 1e-3 stood at day 8), −1.49
and −1.45 at 1e-2 (where 1e-3 would stand near day 20), the bout weight +1.54 and +1.60 and
the other weights' norm 8.4 and 7.1 against 3.4 at the lower rates, the fast gate growing
everything at once; their day 6 says whether the mouth holds under it. Day 16's yardsticks on
115 to 118 decide the boundary: if the mouths hold, the served body takes the ear actor, the
world's words at 0.3 with the mask, the reply withheld (the typist waits four ticks) and the
uncentered critic out of the credit with its reliability logged, at the 43 to 44 boundary
near 07:50.

**The boundary armed, 06:25.** Day 16 of the ear pair, all yardsticks: words said over the
parent on days 14 to 16, 61, 60, 63 and 67, 64, 78 against 83, 90, 96 and 94, 104, 96 on the
same environment without the ear; the mouths 48 and 47, 48 and 42 of 48; the unheard
combinations 103 and 91 of 128; the cortex alone 60 and 51; the parent's turnings-away 12,
12, 14 and 16, 9, 12 a day against 17 to 24 on every recipe before it; the smiles 95, 95, 85
and 103, 87, 95 against 39 to 63. The reply comes after 10 to 13 of its symbols and 18 to 25
ticks against 11 to 21 and 22 to 43. The one cost: the words past the answer, 12 to 14 per
answer against 9 to 11, the bout weight's doing, which is the long critic's item, not the
ear's. The switch itself was tested on a scratch copy of the served body's day 42 loaded with
the boundary's flags and living a third of a day under the fast parent: the mouth 37 and 32
of 48 after against 32 and 29 before, 59 smiles, no fault; the gate opened from 0.50 to 0.67
as the fast ear bodies' gates did (0.65 to 0.72 at day 16), and on those bodies the open gate
came with fewer turnings-away, not more. So the served body takes, at the 43 to 44 boundary
near 07:50 (scratchpad/recipe_boundary43.sh, armed; the --reply 0 chain killed): the ear, the
adapted input, the vigor term off, Adam on the gate at 1e-3; the world's words at 0.3 with the
mask; the uncentered ventral critic out of the credit with its reliability logged; and a
typist that wants a reply and withholds it until four ticks of the child's quiet. The body
before the boundary is kept as data/body2_before_recipe_day43.pt.

**The gate's rate, day 6, 06:45.** Runs 121 and 122 (Adam 3e-3) and 123 and 124 (1e-2) on the
ear actor with the reply withheld, the world's words at 0.3 and the mask, against 115 and
116 at 1e-3. The weights at day 6: listening −1.25 and −1.16 at 3e-3 (where 1e-3 stood at
day 18), −1.79 and −2.64 at 1e-2; the bout +1.53 and +1.47, and +3.95 and +3.85; the net
mid-bout inside a line +0.28 and +0.31 at 3e-3, +0.33 and +0.32 at 1e-3 on day 18, +2.16 and
+1.21 at 1e-2. The two weights walk in lockstep at every rate: the bout weight is one weight
for the 95 percent of bouts outside the parent's lines, where continuing pays, and the
listening weight can only ever offset it by the inside credit, so the difference settles and
the rate sets only how fast. The behavior at days 4 to 6: words said over the parent 71, 62,
64 and 68, 77, 71 at 3e-3 (against 73, 82, 103 and 63, 72, 81 at 1e-3), 53, 59, 59 and 36,
26, 31 at 1e-2: the fast gate's listening weight of −2.6 does stop the starts, and starts are
most of the words said over the parent. But 1e-2 pays for it: run 123's mouth 42 and 38 of
48 and its gate swinging from 0.79 to 0.31 in a day, run 124's unheard combinations 93
started and 56 full of 128 with the memory road at 12 of 32 against 85 to 96 full and 20 to
24 elsewhere, the other weights' norm 11 to 14 against 5 to 6, the words past the answer 13
to 16 per answer. At 3e-3 the mouths 48 and 47, 48 and 48, the unheard combinations 106 and
96, 92 and 85 of 128, the words past the answer 10 to 13. So 1e-2 is out; 3e-3 is a
candidate against 1e-3 at day 16 near 07:05, before the boundary. The served body's day 42:
167 smiles, no frown, no turning-away, "red" learned, 52 known words, 106 lines heard.

**The ear pair's day 20, 06:56.** Words said over the parent on days 18 to 20: 53, 44, 50 and
41, 56, 63 on the ear bodies with the mask (from 73 to 103 on days 4 to 6 and 60 to 78 on
days 14 to 16: still falling), against 91, 88, 91 and 89, 72, 126 on the same environment
without the ear, and 99, 106, 90 and 85, 66, 83 on the ear without the mask, which learned
nothing, as its weight said. Halved by day 20, by a credit of a tenth of a smile. The mouths
48 and 47, 48 and 47 of 48; the cortex alone 57 and 53; the run-on 9.6 and 10.9 symbols past
the first word; the words past the answer 11 to 15 per answer, the bout's cost, unchanged.
Runs 125 to 128 (the uncentered critic) were born 06:50 in the freed slots. The served body's
day 43 began 06:58; the boundary follows its end.

**The conjunction was never needed, 07:00.** I wrote at 06:40 that a linear gate cannot form
"the parent is speaking and I am mid-word". That was wrong, and the probe says so
(scratchpad/conj_probe.py, the ear body of run 116 at night 14 and run 112's day-20 body,
3000 ticks, the lesson off): from the cortex state alone, a Fisher discriminant fit on the
first half reads "the parent is typing" on the second half at AUC 0.997 among mid-bout ticks
and 0.998 among rested ticks; the conjunction against everything else 0.967 and 0.985; only
"I acted last tick" is weak in the cortex (0.63, 0.78), which is what the ear bit supplies.
And the pattern the gate needs is not exclusive-or: rest when the parent speaks whether or
not mid-word, continue outside, start seldom inside; a linear unit does that with the
listening weight larger than the bout weight (−3 against +1.5), which is a point the linear
gate can reach. It does not reach it because of the lesson's dynamics, not its form: the
listening weight's gradient comes from the 170 ticks a day the parent types, the bout
weight's from six thousand mid-bout ticks, and under Adam the two grow in lockstep at every
rate. So the lever is the inside credit's size, which is the world's: runs 129 and 130 are
armed for two of the rate pairs' slots (they end near 07:15), the ear actor with the world's
words at 0.6 instead of 0.3, the mask forfeiting twice as much per symbol said over the
parent; if the listening weight then outruns the bout weight, the dose was the limit, and
the dose is a disclosed constant of the environment, not a rule. The served body's day-43
boundary: mouth 29 and 23 of 48 after 32 and 27 and 40 and 36, the taught lines 83 and 53,
the unheard combinations 97 and 77, the cortex alone 45 and 44; day 42's words said 78, of
them 44 known and 6 fused.

**A correction to runs 125 and 126, 07:01.** Their first day logged the head's reliability
at 0.0 because I had left the reliability gain off for the out-of-credit pair, and the gain
is what computes the running correlation; with the weight at zero the gain touches nothing,
so both were reborn at 07:01 with the gain on and the weight at zero: the reliability logged
by day, the credit untouched. Runs 127 and 128 (the head in the credit through its gain)
read −0.12 and unfilled at day 1, a newborn's features. The served body's boundary script
passes the same pair of flags, so its day-44 body logs the same reading. The second body's
day 13: 17 smiles in a 34-minute day, its attention falling to 0.31 as the lines repeated,
twenty-four minutes of "distracted" misses before its night; the engagement problem of the
fused babble stands.

**The boundary passed, 07:43.** Day 43 ended at 07:42: 163 smiles, no frown, no
turning-away, "wet" in four frames, 53 known words, 111 lines heard; and the diary took the
ear recipe. The body was saved, kept as data/body2_before_recipe_day43.pt, and reloaded on
the same page with the ear and the adapted input, the vigor term off, Adam on the gate at
1e-3, the world's words at 0.3 with the mask, the uncentered ventral critic out of the credit
with its reliability logged, and the offset as before; 519,266 ticks, 43 nights, 397 slots
in the store. The typist for days 44 to 46 wants a reply and withholds it until four ticks
of the child's quiet. The first-day yardsticks of a body switched mid-life are the fast ear
bodies' at day 6 (the gate opening toward 0.6, the listening weight near −0.4 after six days
of the parent's lines) and the scratch test of 06:15 (the mouth held over a third of a day);
what the diary shows tomorrow morning is the real reading. The critic runs at day 6: the
live reliability wanders by day (127: −0.07, +0.46, +0.25, +0.19, +0.12, −0.01; 128: −0.12,
−0.31, +0.45, +0.11, −0.01, +0.21; 125: +0.15, −0.10, −0.11, +0.22, +0.09; 126: +0.32,
+0.06, +0.46, +0.26, −0.13), an estimator over some eight independent windows at a time, and
the value instrument's four fresh days say what it is; run 127's mouth 47 and 34 of 48 at
day 6, its words past the answer 13 to 17, no gift from the credit yet on a newborn's head.

**The gate's rate at day 16, 08:12.** Adam 3e-3 (runs 121 and 122): words said over the
parent on days 14 to 16, 41, 37, 34 and 39, 49, 27, against 61, 60, 63 and 67, 64, 78 at
1e-3; but the mouths 39 and 28, 43 and 35 of 48 against 48 and 47, 48 and 42, the unheard
combinations 98 started and 63 full, 94 and 78, against 103 and 91, 92 and 84, the gates
0.78 and 0.82. Adam 1e-2 (123 and 124): 28, 19, 21 and 33, 37, 32 said over the parent, the
mouths 25 and 17, 30 and 28, the gates swinging by the day between 0.06 and 0.99 (124:
0.85, 0.17, 0.11, 0.94, 0.83, 0.96, 0.06, 0.97 on days 8 to 17). The faster the gate learns,
the less it talks over the parent and the less it answers a cue; the rate buys listening with
the mouth, and the mouth is the diary's. 1e-3 stays, which is what the diary took at 07:43.
The lever that does not spend the mouth is the inside credit itself, runs 129 and 130 (the
world's words at 0.6), armed for the slots 121 to 124 free near 08:25.

**The switched gate, 08:30.** The served body's first day on the ear recipe (day 44, a
26-minute day, its sleep pressure already half spent at the boundary): 110 smiles at a
higher rate than day 43's 163 in fifty minutes, no frown, but four turnings-away after none
on days 41 to 43, the parent's attention falling through the day from 0.80 to 0.53, and the
reply withheld once past its cap (116 symbols of run-on before the parent answered "give
cat"). The gate's mean on the day's lines 0.61 against 0.54 the day before. The cause is
arithmetic: the adapted input subtracts the running mean μ from the gate's inputs, and a gate
whose weights were fit over 43 days to the raw inputs loses the term w·μ from its logit, here
−2.06 (|μ| 2.9, |w| 6.3), so the switch raised every logit by two and the gate opened until
its lesson fit the new coordinates. The exact remedy, the bias absorbing w·μ so the function
is unchanged while the coordinates change, was tested on a copy of the body after its night:
too late, since the day's lesson had already refit the gate; corrected now, the gate falls to
0.31 and the smiles to 21 in three thousand ticks. So the diary is left alone at the 44 to 45
boundary; its gate re-equilibrates by its own lesson, and the day's four turnings-away are
the cost of a switch made without the change of variables. The general fix, body-general and
exact: the adapted input should change the lesson's geometry and never the function, the
bias taking w·Δμ at every tick the running mean moves. It is the next item for the fast
seeds, beside the mask's dose.

**Day 44 closed, 08:43.** The served body's first day on the ear recipe: 126 smiles in a
26-minute day and its post-night cues, no frown, seven turnings-away against none on days 41
to 43, the parent's attention 0.64 at the end, "cat" learned, 54 known words, 119 lines
heard; the withheld reply twice past its cap (116 symbols of run-on) and otherwise yielded
after 34 to 82 symbols. The day-45 boundary's mouth 37 and 29 of 48, up from 29 and 23 the
day before. Day 45 is the full day that decides: if the turnings-away stay at this rate the
switch is undone at the 45 to 46 boundary from data/body2_before_recipe_day43.pt, an honest
failure of a mid-life switch made without the change of variables, and the ear recipe is
kept for bodies born with it; if they fall back toward none, the gate has found its new
coordinates and the recipe stays. Runs 129 and 130 (the world's words at 0.6) and 131 and
132 (the function kept under the moving mean, gate_center_keep) were born 08:33 in the rate
pairs' slots. The rate pairs' day 20: at 3e-3, words said over the parent 31 to 44, the mouths
42 and 41, 44 and 38; at 1e-2, 26 to 42, the mouths 28 and 27, 23 and 18, the gates at 0.97
and 0.11 on the last day. The second body's day 14: 68 smiles, no turning-away, its cues all
answered; its days 15 to 17 began 08:25.

**The critic runs at day 6, and day 45's first ten minutes, 09:00.** The value instrument
on the day-6 bodies (four fresh instrument days each): the uncentered ventral critic reads
the return at its horizon at +0.17 (−0.18, +0.36, +0.12, +0.37) on run 126 with the head out
of the credit, +0.14 (+0.04, +0.46, −0.22, +0.28) on 127 and −0.26 (−0.09, −0.14, −0.23,
−0.57) on 128 with the head in the credit through its gain; the pinned bands read as they
always have on a young body, band 5 near −0.2, band 6 at −0.7 to −0.8 (the day's trend,
beyond a day's judgment). Weakly right-signed on two of three where the centered heads read
−0.26 and −0.28 at day 20, and nowhere near the +0.52 the same rule read on the day-40 body:
a six-day-old's slow bands do not yet carry what a forty-day-old's do, which is the age
chains' finding again (a newborn's minutes unforeseeable, a two-week-old's foreseeable). Day
16 near 09:40 is the reading that counts. The heads in the credit have not changed behavior
yet: run 127's words said over the parent 64 to 74 and its words past the answer 16 to 17
per answer at days 14 to 16, run 128's 49 to 57 and 11. The served body's day 45, ten
minutes in: 45 smiles, no turning-away, the gate's mean on its lines 0.49 after 0.61 on day
44 and 0.54 on day 43, the parent's attention at smiles 0.81. The gate has found its new
coordinates by its own lesson in a day, as the fast bodies' gates did.

**The parent's patience, 09:12.** Day 45's two turnings-away came at 09:09 and 09:10, right
after a withheld reply hit its cap: the typist waited 45 seconds for four ticks of the
child's quiet, the child said 108 symbols meanwhile, and the parent's attention, decaying
through the wait, fell to 0.14. Half the served body's replies have hit that cap (108 and
116 symbols of run-on), and on day 44 the same fall preceded most of the seven turnings-away.
The cap also paces the parent's lines (it speaks when the child has been quiet three seconds
or after the cap), so a body that seldom rests hears one line a minute at best. The fast ear
bodies yield within 20 to 40 ticks and never met the cap; the served body, switched at 43
days, does not yet. A parent waits seconds, not most of a minute: from day 46 the typist's
cap is 12 seconds (scratchpad/teach_days3b.sh, the same typist otherwise, days 46 to 48),
the body untouched. The reply withheld stays: it is the parent's behavior the user chose,
and the fall of attention through a long wait is the environment telling the truth. Day 45 so
far: 79 smiles in twenty minutes, the gate at 0.52, two turnings-away.

**The dose and the kept function at day 6, 09:30.** Runs 129 and 130 (the world's words at
0.6, the mask forfeiting twice as much per symbol said over the parent): the listening
weight −0.49 and −0.52 at day 6, the bout +0.96 and +0.92, against −0.40 and −0.53, +0.86
and +0.98 at 0.3. Doubling the inside credit did not speed the listening weight: the lockstep
is not the credit's size either, so the difference of the two weights is set by the lesson's
own geometry, not by how much a talked-over symbol costs. Words said over the parent 73 to 92
on days 4 to 6, as at 0.3. Runs 131 and 132 (the function kept under the moving mean): the
listening weight −0.43 and −0.46, the bout +0.84 and +1.30, the bias carrying w·μ as it
should (−0.31 and −0.59); the same body as 115 and 116 in every reading so far, which is what
a change of coordinates that leaves the function alone ought to be; its worth is at a switch,
where it keeps a 43-day-old gate's function while the mean adapts. The served body's day 45
by its planner: 105 smiles, "cup" in four frames, "one" typed for the first time, the cues
answered ("first milk then" → "ball" clean), the attention recovering to 0.85 to 1.0 in the
back half after two turnings-away a third of the way in; five turnings-away in all with the
post-night cues, each after a withheld reply met its 45-second cap.

**Day 45 closed, 09:37.** 131 smiles, no frown, five turnings-away (seven the day before, each
after a withheld reply met the 45-second cap), the parent's attention 0.77 at the end, "cup"
learned, 55 known words, 123 lines heard; the gate's mean on the day's lines 0.52, back where
it was before the switch. The ear recipe stays on the diary. From day 46 the typist's
patience is 12 seconds (teach_days3b.sh, days 46 to 48; the planners for 47 and 48, "cold"
and "soft", spawned). The second body's day 15: 19 smiles, no turning-away, "run" heard in
three frames, and to "dog will" it answered "sit little dog", yesterday's word in an old
frame, a retention across days.

**The magnitudes kept, day 2, 09:58.** Adam was the lockstep. A three-factor rule's step is
eligibility times credit, and the inside gap (0.19 in favor of resting while the parent
types) against the outside gap (0.002, the mouth at its credit-neutral rate) should move the
listening weight a hundred times faster than the bout weight; Adam divides each weight's step
by its own gradient's running size and moves both at the rate, so doubling the inside credit
(runs 129 and 130) changed nothing and every rate gave the same lockstep. Runs 133 to 136 put
plain SGD back on the gate (the vigor term off, the adapted input with the function kept, the
ear): at day 2 the listening weight −0.52 and −0.43 against the bout +0.40 and +0.50 at SGD
0.2, the first bodies whose listening weight leads the bout, and −0.09 and −0.13 against
+0.31 and +0.44 at 0.05, slow; the other weights' norm 0.9 and 0.2 against Adam's 3 to 6,
the cortex weights left nearly at birth because their gradients are small, which is the
magnitude rule doing its work on them too. If at day 6 the net logit mid-bout inside a line
is negative and the mouth holds, the gate will rest mid-word while the parent speaks, which
no Adam body did in twenty days, and SGD 0.2 with the function kept becomes the candidate
recipe for the next boundary. Day 46 under the 12-second patience, eleven minutes in: 30
smiles, no turning-away.

**The critic runs at day 16, 10:20.** The uncentered ventral head on the four fresh
instrument days of each day-16 body: +0.03 (−0.08, +0.22, −0.17, +0.16) on 125 and −0.22
(−0.33, −0.59, −0.01, +0.07) on 126 with the head out of the credit; +0.16 (−0.03, +0.17,
+0.15, +0.33) on 127 and +0.08 (−0.54, +0.33, +0.24, +0.30) on 128 with the head in the
credit through its gain. Right-signed on three of four and near zero on the whole: not the
centered heads' steady wrong sign, and not the +0.52 the same rule read in a third of a day
on the served body at 40 days. The centering was a defect and its removal was necessary; it
was not sufficient for a head learned live from birth. What differs between the two readings
is the body's age and the head's history, and the night-transfer instrument is now running on
run 125's own bodies at days 16, 20 and 6 to say whether a sixteen-day-old's slow bands drift
across nights the way a forty-day-old's do not. The heads in the credit changed no behavior:
run 127's words past the answer 14 to 15 per answer and its words said over the parent 59 to
73 at days 18 to 20, run 128's 11 and 40 to 51, the run-on untouched. Day 46 under the
12-second patience, thirty minutes in: 80 smiles, one turning-away, the first withheld reply
delivered after 12 seconds and 34 symbols instead of 45 seconds and 116.

**The critic's second defect, and the head that holds, 10:30.** On the cached features of the
served body's day 40, the live rule read the return at +0.52 after one pass and −0.49 after
four, −0.52 after sixteen, its weights growing from 2.3 to 7.2 and its bias to +1.7: sixteen
days of updates on one day's kind of ticks, which is what every live head has had, turn the
right head into the wrong one. TD(λ) with the trace at the horizon (the backward view of the
return itself, written 2026-09-04 against the bootstrap's fixed point) holds +0.53 through
four passes and flips at sixteen the same way, so the bootstrap was not the only cause: a
day's slow bands are one ramp, and a 768-weight head fit to it for long enough learns the
day's trend and carries it wrongly to the next day. The ridge head held +0.50 across ten
nights because it was regularized. The forgetting head (the weights decaying toward zero,
written 2026-09-04 and read wrong on centered inputs) on the raw slow bands: at a time
constant of 24,000, 12,000 or 6,000 ticks the head reads +0.52 after sixteen passes, +0.52 on
the next night's body and +0.53 to +0.60 on the body of ten nights before; at 96,000 it fails
as without. Synapses decay, and a critic whose weights forget at the scale of a day cannot
overfit a day. So the long critic is three things at once, each necessary and none
sufficient: the raw slow bands (the centering off), the trace at its horizon, and forgetting
at about a day. All three exist as flags. Runs 137 and 138 (the head out of the credit, its
reliability logged) and 139 and 140 (in the credit through its gain) on the ear actor with
Adam 1e-3, the reply withheld, the world's words 0.3 and the mask, take the slots the dose
and kept-function pairs free near 11:15.

**The wait itself costs, 10:35.** Day 46 under the 12-second patience: four turnings-away by
its post-night cues, each at an attention of 0.12 or less, the replies now delivered after
12 seconds and 34 to 39 symbols instead of 45 seconds and 116. The cause is in the
caregiver's own rule: its attention drifts down in silence, so a withheld reply costs
attention through the wait whatever the cap, and a body that does not yet pause pays it every
time. The wait can only teach a body whose long critic carries the pause's consequence into
the gate's credit, and the served body's critic is not in its credit. So from day 47 the
diary's typist replies at once (the patience 12 seconds, the mask and the ear unchanged); the
withheld reply returns with the critic. Days 47 to 51 under it.

**The magnitudes kept, day 6, 10:36.** SGD 0.2 (runs 135 and 136): the listening weight
−1.21 and −1.06, the bout +1.12 and +1.25, the net mid-bout inside a line −0.09 and +0.19,
where Adam 1e-3 stood at −0.55/−0.72 against +0.96/+1.07 on day 8 and reached −1.15/−1.28
against +1.47/+1.60 only by day 18. The probe: acted inside the parent's lines 0.30 and 0.29
of the ticks, p(start) 0.25 and 0.25, p(continue) 0.40 and 0.40, which is what the Adam
bodies showed at day 14 (0.30 and 0.18; 0.23 and 0.15; 0.49 and 0.31). The mouths 47 and 46,
48 and 45 of 48; words said over the parent 60 to 88 on days 4 to 6 against 63 to 103. So the
magnitude rule is twice as fast and holds the mouth, but the bout weight grew with the
listening weight here too: the outside gap is not the 0.002 the single-day probe read, or the
bout's credit is real (a word finished pays), and the difference of the two weights is again
near zero rather than large. SGD 0.05 (133 and 134): −0.32 and −0.35 against +0.47 and
+0.58, slow, the mouths 47 and 37, 42 and 37. Day 16 says whether 0.2 goes on past the
Adam bodies or stops where they did. Day 46 of the diary ended its post-night cues with six
turnings-away, each at an attention of 0.12 to 0.14 after a 12-second wait: the reply comes
at once from day 47.

**The young body's nights, 10:38.** The night-transfer instrument on run 125's own bodies at
days 16, 20 and 6: the slow bands drift across four nights at this age (cosine at the same
tick 0.75, 0.83, 0.83; centered 0.51, 0.69, 0.72) where the forty-day-old's did not (0.99,
1.00, 1.00), and across the ten nights back to day 6 they are another body (0.58, 0.61,
0.59). Yet the ridge head from the slow bands fit on day 16 reads the return at +0.76 on its
own held-out ticks, higher than the forty-day-old's +0.50, and +0.70 on the body of four
nights later; fit on day 20 it reads +0.71 and +0.63 on day 16. Only the day-6 body is
foreign to it (−0.57). So a young body's value is there to read and a regularized head keeps
it across the nights it lives through; the live heads' readings near zero at day 16 are the
estimator's failure, the overfit of one day's ramp that the cache showed, and the forgetting
head with the trace is the remedy runs 137 to 140 will measure from birth. A live head that
forgets at a day also forgets the day-6 body by day 16, which is what the drift asks of it.

**Day 46 closed, 10:36.** 108 smiles, no frown, seven turnings-away under the 12-second
patience (five under 45 seconds the day before): the cap was not the cost, the wait was.
"hot" learned, 56 known words, 127 lines heard. From day 47 the typist replies at once and
recasts with the least added (days 47 to 51 armed, the planners for 47 and 48 waiting).

**The kept function, corrected, 10:50.** Runs 131 and 132 at day 16 read what no other body
has: the bout weight −1.68 and −0.23 (every other body +1.3 to +1.6), the bias +1.61 and
+0.30 (every other −0.5), the gate's weights pointing against their own mean feature (w·μ
−4.1 and −3.3), the reply waited for 90 to 134 ticks and yielded 4 of 9 and 5 of 7, words
said over the parent 84 to 94 and 64 to 74. The fault is mine and it is arithmetic: I wrote
the kept function as the bias taking w·Δμ at every tick, which keeps w·x + b unchanged only
while w stands still; as the lesson moves w, the accumulated increments stop summing to w·μ,
the bias and the weights chase each other, and the gate drifts to a degenerate solution. The
exact form holds the uncentered intercept c as the lesson's own quantity and recomputes the
bias as c + w·μ from the current w and μ at every tick, attributing whatever the lesson moved
the bias by to c; a check with a moving w and lesson steps keeps the logit to a millionth. A
loaded body's saved bias is c + w·μ of its saved mean, so c is recovered at the first tick.
The organ tests 16 of 16. Runs 133 to 136, which carried the first form, were reborn at 10:50
on the corrected one (their day-2 and day-6 readings stand as read, the SGD gates having
moved w little); runs 137 to 140 launch on it. The dose pair at day 16: words said over the
parent 59 to 62 and 53 to 75, the mouths 48 and 47, 48 and 48, the listening weight −0.84 and
−0.98 against the bout +1.50 and +1.34, the lockstep as at 0.3.

**The accounting of the turnings-away, 10:52.** By day: days 40 to 43, with the reply off, no
"talked over" and no "past its answer" miss at all and no turning-away; days 44 to 46, with
the reply on, 80, 87, 117 words counted as said over the parent and 75, 42, 93 as past the
answer, each at 0.04 of the caregiver's attention by its own rule, and seven, five, seven
turnings-away; day 47 with the reply at once, 15 and 17 in its first twelve minutes and one
turning-away. The wait was not the cost and the cap was not the cost: the reply typist's
rules are, and they fall on a body that cannot yet learn from them, since nothing carries
the attention's fall into its credit. The fast bodies born into that world take 10 to 24
turnings-away a day, the ear bodies the fewest. So from day 48 the diary's typist does not
reply (the mask and the ear stay, the patience 12 seconds), and the reply returns when the
long critic is in the credit and can learn from what the reply costs. The comparison of the
switch itself is now clean: the actor changed at 43 and the typist at 43 and 45 and 46 and
47; only the actor stays.

**Day 20 of the dose and the first kept-function pairs, 11:21.** The dose pair (the world's
words at 0.6): words said over the parent 68, 56, 46 and 64, 53, 75 on days 18 to 20, the
mouths 48 and 47, 48 and 48, the listening weight −0.84 and −0.98 against the bout +1.50 and
+1.34 at day 16: the same as at 0.3, the dose is not the lever. The first kept-function pair
on the drifting form: run 131 yielded to the withheld reply once in seven at day 20 after
157 ticks of waiting, words said over the parent 77 to 84, a gate pointing against its mean
feature; the corrected form runs on 133 to 140 from 10:50. Run 128's day 20 with the
uncentered head in the credit and no decay: +0.15, +0.27, 0.00, −0.20, the overfit as the
cache predicted. Day 47 of the diary by its planner: "cold" in five lines and four frames,
67 smiles in the teaching window, six turnings-away between 10:50 and 11:07 and none after,
the cues answered in family but none cleanly. Its close and the reply's end come with its
post-night cues.

**The critic runs at day 20, 11:35.** The uncentered head with no decay: out of the credit,
+0.31 (+0.39, +0.36, +0.06, +0.44) on 125 and +0.22 (+0.14, +0.04, +0.35, +0.35) on 126,
up from +0.03 and −0.22 at day 16; in the credit through its gain, +0.03 on 127 and +0.05 on
128. Pinned band 5 reads +0.15 and +0.39 on the out-of-credit bodies at this age. So the
head learns with the body's age when nothing it does feeds back into what it predicts, and
reads near zero when its error moves the gate that makes the return: a critic in the loop
must track a value its own influence keeps moving, which is the usual actor-critic
non-stationarity and one more reason the head must forget at the scale of a day. Runs 137 to
140 carry the three flags from 11:23.

**Day 47 closed, 11:29.** 92 smiles, no frown, nine turnings-away with the reply at once (the
accounting rules of the reply typist, not the wait), "cold" learned, 57 known words, 132
lines heard. From day 48 the typist does not reply; the mask and the ear stay; the reply
returns with the critic.

**The kept function is not a rule to live by, 11:50.** The corrected form, which holds the
gate's function exactly under the moving mean, changes what the lesson does with it: SGD 0.2
with the function kept reaches a bout weight of +3.05 and +3.06 by day 6 (the drifting form
gave +1.12 and +1.25; the Adam bodies without it +1.5 at day 18), the listening weight −1.11
and −1.11, the gates 0.72 and 0.27, one mouth 32 and 20 of 48, and the probe p(continue)
outside 0.75 and 0.72 against 0.63 to 0.68 before. So the slow drift of the mean under the
uncompensated centering was doing something to the lesson that I took for nothing: with it
the bout weight grows a tenth as fast. I do not have the mathematics of that yet, and a rule
whose effect I cannot derive does not go in the recipe. What stands: the compensation belongs
at a mid-life switch as a warm-up (the lesson held while the mean adapts and the bias takes
w·Δμ, exact because w stands still), not as a standing rule. Runs 133 to 140 all carried the
flag; all eight were reborn at 11:50 without it and with fresh logs (the day-4 to 6 instrument
lines printed for 133 to 136 at 11:45 counted two lives in one log and are void). The SGD
question and the long critic's question start over clean; day 6 near 12:20.

**Day 48 with the reply off, 12:13.** By its night: 135 smiles, no frown, no turning-away, no
"talked over" and no "past its answer" miss, 30 lines and 9 cues typed. The accounting held:
the turnings-away of days 44 to 47 were the reply typist's rules on a body that could not yet
learn from them, and with the reply off the ear-recipe body takes the parent's smiles at the
best rate of its life. The second body's day 17: 99 smiles, its best, "wet" heard in two
lines, its cues answered ("I had" → "milk", "why dog up?" → "because big do"); its days 18 to
20 began 12:13.

**Day 48 closed, 12:23.** 171 smiles, the most of any day of this body's life (167 on day 42
before the switch), no frown, no turning-away, the parent's attention 0.96 at the end, "soft"
learned, 58 known words, 137 lines heard; eight cues answered with accepted words in the
teaching window ("dog will" → sit, "give big" → ball, "where ball?" → ball under, "first milk
then" → ball, "big dog bigger" → dog, "little dog had" → milk, "you saw" → dog, "why dog up?
because" → big). The ear recipe on the diary, with the typist that does not reply, is a
better day than any before it; what the mask teaches it (the ear's listening weight after
four days) is measured at the day-49 boundary by the gate's weights on the saved body.

**The diary's ear after five days, 12:25.** The gate's weight on "the world's symbol this
tick" −0.52 on the saved body after night 48, from −0.13 after night 44: a tenth a day, faster
than the fast bodies' twentieth, since the typist's lines fill more of this body's ticks. The
bout weight −0.40, where every fast body's is +1.3 to +1.6: on the diary, with the parent's
own rule that babble wears its attention and no reply typist counting words, continuing a
bout is credited below the baseline, and the switched gate has learned to rest mid-bout as
no fast body has. The bias +0.48 against w·μ −3.5. The turn-taking the fast seeds could not
reach in twenty days may come on the diary from the parent's rules alone; the day-49 boundary
probes and the day's misses will say.

**Day 6 of the eight clean bodies, 12:49.** Without the kept function: SGD 0.2 (135 and 136)
listening −1.27 and −1.10, bout +1.87 and +1.45, net mid-bout +0.60 and +0.36; SGD 0.05 (133
and 134) −0.28 and −0.42 against +0.54 and +0.74; Adam 1e-3 (137 to 140) −0.39 to −0.45
against +0.87 to +0.94, as 115 and 116 read at this age. So the magnitude rule reaches in six
days the listening weight Adam took eighteen for, and the bout weight comes with it: the
lockstep is not Adam's alone, it is REINFORCE's, which raises the weight of any action that
pays wherever it is taken, and continuing a bout pays in the fast parent's world. Three
forms of the centering's bias gave three bout weights at day 6 under the same SGD 0.2 (+1.1
drifting, +3.05 held, +1.87 uncompensated), which says the operating point's slow drift
enters the lesson's dynamics in a way I have not derived; the uncompensated form is the one
every measured body has lived under and the one the diary lives under. The diary's own gate
meanwhile has a bout weight of −0.40: under its parent, whose attention babble wears and who
does not reply, continuing costs, and the same rule learned to rest. The turn-taking that is
left is therefore the environment's to teach through a credit that reaches the gate, not the
optimizer's; the fast parent as written pays for bouts and the diary's parent charges for
them. The long critic's day-6 value instrument is computing.

**The clean SGD pair's probe, 12:51.** Runs 135 and 136 at day 6, 3000 ticks under the fast
parent with the lesson off: acted inside the parent's lines 0.37 and 0.36 of the ticks,
p(start) 0.26 and 0.31, p(continue) 0.56 and 0.45, against 0.44 and 0.43, 0.67 and 0.66
outside: where the Adam bodies stood at day 14 (0.30 and 0.18; 0.23 and 0.15; 0.49 and 0.31).
Twice as fast to the same place, and the same place; the optimizer is not the remainder.

**The young body's value is readable, 13:52.** The night-transfer page lived by run 137's day-6 body
(9000 ticks, 126 rewards) and replayed through its day-11 body: the slow bands drift 0.41 to 0.55
per band across five nights (centered 0.25 to 0.51), the fastest drift I have measured, and a
ridge head from the slow bands fit on the first 60 percent of the day-6 body's ticks reads the
return at horizon 1024 on the held-out 40 percent at +0.85, and on the day-11 body at +0.89; fit on
day 11 it reads +0.87 own and +0.86 on day 6. The served body's day 40 read +0.50. So the young
body's features carry the value more plainly than the old body's, and carry it across five nights
of the fastest drift. The same head with a tenth of the regularization flips to −0.85 held-out:
one day is one ramp, and a 768-weight head fit hard to its first part predicts the wrong sign on
its last part; the regularization ridge needs is the whole penalty, not a tenth of it. Meanwhile
runs 137 to 140's live three-flag heads read their reliability at −0.18 to +0.19 through days 9 to
12. The features are not the defect; the live rule on them is, and its offline replica on this
very page (the trace at the horizon, Adam 1e-3, forgetting at 12000) and a grid of its variants
(shorter forgetting, a smaller step, the standardized input, the normalized step) are computing.

**The third defect is the input's scale, 13:58.** The day-6 value instrument on runs 137 to 140 (four
fresh instrument days each, the head frozen): the live three-flag head reads the return at its horizon
at +0.07, −0.08, +0.09 and −0.29, nothing, as the live reliability said. Its offline replica on run
137's own day-6 page (the trace at the horizon, Adam 1e-3, forgetting at 12000, the raw slow bands)
reads −0.86 held-out and −0.86 on the day-11 body, and so does every variant on the raw bands:
forgetting at 3000 or 1000, the step at 1e-4, TD(0), no forgetting, the normalized step. The raw
slow bands have a norm of 8 and a per-dimension standard deviation of 0.08: their variation is a
hundredth of their level, so the semi-gradient is the mean direction times the error and the head
learns the day's level as a ramp, which on the young body has the wrong sign for the rest of the
day. Standardized (the fit part's mean and scale) the same rule reads +0.88 held-out and +0.85 to
+0.87 on the day-11 body after one pass, with Adam or the normalized step, as the ridge ceiling
does. But the served body's day 40, where the raw head read +0.52 across ten nights, reads −0.38 to
−0.56 under every running normalization I tried (mean alone at 12000, scale alone, both at 4096 to
48000): what mends the young body breaks the old. The two pages are being fit side by side with
least squares at five regularizations and the live rule at 1 to 64 passes on raw, statically
standardized and running-standardized inputs, and the diary's own day-50 body is living the page
against its day-43 body, to find the one form that reads both.

**Runs 137 to 140 ended, 14:04.** Their reliability read −0.18 to +0.19 through day 14 and their day-6
value instrument +0.07, −0.08, +0.09 and −0.29 on fresh days: the three-flag head does not read the
return live on a young body, and the offline replica says why (the raw input's scale). Nothing more
was to be learned from them; their four cores go to the instrument that decides the next form. The
pages instrument (scratchpad/nt_pages.py): one body lives the fast parent's plan under several seeds,
and a head fit on one page is read on the pages it never lived. Every reading of the critic so far,
the +0.52 across ten nights and today's +0.85, was within one page (the fit on its first part, the
read on its last, or the same page through another night's body), where a head that has learned
where it is in the parent's plan reads the return without learning any value the body could act
on; the value instrument's four fresh days, which read the live heads at zero all along, are the
cross-page reading. Run 137's day-6 body and the diary's day-50 body are living pages under seeds 8,
9 and 10 beside their seed-7 pages.

**A reading of −0.96, 14:10.** The diary's day-50 body lived the seed-7 page (141 rewards) and its
day-43 body replayed it: the slow bands 0.98 to 1.00 at the same tick, the fast bands 0.71 to 0.79
(the ear recipe moved the fast bands, not the slow). A ridge head fit on the first 60 percent reads
the return on the last 40 percent at −0.96, on either body, at either regularization, from the slow
bands or all eight. A linear head cannot be that wrong about a value; it can be exactly that wrong
about a trend. Two monotone ramps, the return's and the features', fit on the rising part and read
on the falling, give −0.96, and every within-page reading of the critic this week, the +0.52 that
held across ten nights and this morning's +0.85, is the same kind of number with a kinder sign. The
cross-page reading is the only one that counts; it is computing on both bodies.

**The cross-page reading, and the decorrelated critic, 14:34.** Run 137's day-6 body lived four days
under four parent plans (seeds 7 to 10; 70 to 111 rewards each), and the diary's day-50 body four of
its own. A ridge head from the slow bands fit on one day of the young body reads the return at
horizon 1024 on the three days it never lived at +0.66, +0.92 and +0.71 (the regularization at 10),
and fit on three days reads the fourth at +0.70: the young body's state carries a value that holds
across the parent's plans. The live rule (the trace at the horizon, Adam 1e-3, forgetting at
12000) fit on the same day reads the same three days at −0.49, −0.74 and −0.60, and every input
form I tried (raw, centered, standardized, the running mean and scale at 1024 or 4096) leaves it
negative; on three days of fitting, +0.05. The defect is not the input's scale after all; it is
that a gradient head fit for one pass to correlated inputs reads their dominant common component,
which on the slow bands runs against the return, and only the inverse covariance divides it out:
the first-order direction of TD(λ=1) is XᵀG, the least-squares head is (XᵀX)⁻¹XᵀG, and on these
inputs the two point opposite ways. Recursive least-squares TD(λ) with forgetting (the eligibility
trace carried through a precision matrix; the online form of LSTD, the Kalman form of TD) fit on
the same one day, one pass, reads the three unlived days at +0.63, +0.92 and +0.71 with the prior
100 I, standardized or raw, with or without forgetting, with the trace at γ or at γ²: at the
ceiling. On the diary's day-50 body no head fit on one day reads the other three consistently
(the ridge itself −0.12, +0.03, −0.81); fit on three days, the least-squares head reads the fourth
at +0.75 and the recursive head at +0.87, where the gradient head reads −0.70. A body whose
inputs are decorrelated by its inhibitory interneurons learns as the recursive head does; that is
biology's answer to correlated inputs, and it is the fixed point a decorrelating layer computes.
Written as vcrit_rls (body/model.py, body/life.py): the statistics A and b of the trace against
the state's discounted change and the reward, forgetting at vcrit_forget with a constant prior
vcrit_rls_delta·I so no unexcited direction winds up, the head the solve every vcrit_rls_every
ticks, float64, saved with the body and sized by the life to the head's active inputs plus the
level. The organ tests 16 of 16; on a scratch copy 2000 ticks under the fast parent give a level
of 22 against returns near 25, and the statistics round-trip through save and load. Runs 141 and
142 (out of the credit, the reliability logged) and 143 and 144 (in the credit at weight 1 through
the gain) on the ear actor launched at 14:32, otherwise as 137 to 140; their day-6 value
instrument, four fresh days each, is the reading that arms the diary's boundary
(scratchpad/recipe_boundary_rls.sh, unarmed: the critic out of the credit first).

**The magnitude rule oscillates, 14:54.** Runs 135 and 136 (plain SGD at 0.2 on the gate) at days 17 to
20: gates 0.84, 0.86, 0.91, 0.19 and 0.80, 0.84, 0.21, 0.19; a gate that acts nine ticks in ten one
day and two in ten the next, with smiles swinging 132 to 198 and the run-on past an answer 5 to 16
words. Runs 133 and 134 (SGD at 0.05) hold 0.40 to 0.65 as the Adam bodies do, talk over the parent
48 to 85 words a day at day 18 to 20 as the Adam bodies did at day 20, and run on 10 to 11 words: the
same place by the same day. The step that reaches the listening weight in six days is the step
that cannot hold a gate; the actor keeps Adam at 1e-3 and the diary's optimizer does not change.

**The decorrelated critic's first days, 14:54.** Runs 141 to 144 at day 2: reliability +0.06, +0.40,
+0.25 and +0.25 on their own stream (the gradient heads of 137 to 140 read −0.09 to +0.15 through
day 12). A day takes 770 seconds against 137's 480 under a lighter load; the critic itself costs 2
to 3 milliseconds a tick on the scratch harness (25.7 against 29.0, 23.3 against 25.5), a tenth.

**Day 4 is too young to read, 15:14.** Run 141's day-4 body lived two fresh pages (seeds 8 and 9). Its
live decorrelated head, frozen, reads the return on them at −0.06 and −0.71 (its value sits at 26
against returns of 23 to 25: the level is right, the shape is not). But no head reads this body
across pages: ridge fit on one page reads the other at +0.30 and −0.09, the offline recursive head
+0.21 and −0.71, the body's own leaky form +0.11 and −0.68. The live rule is doing what its
offline twin does; the day-4 state does not yet carry a value that holds from one parent plan to
another, where run 137's day-6 state did at +0.7 to +0.9. The live reliabilities of 141 to 144
through day 5 (−0.16 to +0.40) are readings of that youth. The day-6 value instrument and the
cross-page ceiling on the same day-6 body, side by side, will say whether the head or the state is
the bound at six days.

**The day-6 reading, the prior's scale, and the morning ramp, 16:00.** Run 141's day-6 body lived
four fresh pages. The ceiling holds at six days: ridge fit on one page reads two of the other
three at +0.77 and +0.91 (the third is a flat page, its return's spread 2.7 against 8 to 10, which
nothing reads). The live decorrelated head, frozen from the body, reads them at +0.08, +0.81,
−0.21 and +0.38: real, and half the ceiling. Two returns were checked: the face's and the body's
own (the world's words at 0.3, not heard over its own voice, add a third of the mass but a tenth
of the variance), and the head reads both alike, so the target is not the gap. The gap is the
prior. Recursive least squares fit on one page with the prior 100 I on raw inputs reads the other
pages at −0.41, −0.26, −0.13; with the inputs standardized and the prior 10,000 or 100,000 it
reads +0.93, +0.06, +0.89, above the ridge; on run 137's day-6 pages the same reads +0.67, +0.94,
+0.69 at every prior from 100 up. One day is about seven independent returns at this horizon
against 768 weights, and only a prior about as heavy as a day's evidence in each direction (the
prior in units of the forgetting horizon, one to ten) keeps the head on the few directions that
carry the value. In standardized units that prior is scale-free; on raw inputs no single prior
serves, because the dimensions' scales span two orders. Then the live form of the standardization:
a running per-dimension mean and scale at 4096 to 36000 ticks, the statistics carried on from
the fit page into the read pages as a live body's would be, reads −0.62 to −0.92 on run 141 and
−0.17 to −0.60 on run 137; the same statistics restarted at each page's first tick read +0.50 to
+0.86 and +0.53 to +0.88. The cause is the night: body/life.py zeroes the bands at sleep, so every
page, like every morning, begins with the slow bands at zero and climbing toward their level
for most of the day (band 7's clock is 16,384 ticks, longer than a day), and statistics that have
settled on evening levels misread the morning by many standard deviations. The ramp is the
"one day is one ramp" I found this morning, and it is not the world's; it is the body's own
reset. Brains do not zero their slow state at sleep. Four pages are being lived on the same
day-6 body from a settled state (a warm plan first) to read the running form without the ramp;
if it reads, the bands keep their state across the night (a flag, measured on fast seeds first).

**Settled pages, and the statistics' time constant, 16:46.** Four pages lived by run 141's day-6 body
from a settled state (a warm plan first; the slow bands' norm 9.8 at the page's start against 10.9
at its end, no ramp). Ridge from one page reads the others at +0.84, +0.77, +0.47; the recursive
head on statically standardized inputs with the prior at one to ten days of evidence reads +0.88 to
+0.90, +0.66 to +0.68, +0.27 to +0.30; raw inputs at any prior read −0.9 to +0.4. The running
statistics, when they have converged before the fit page and then run on through fit and read pages
(a body past its first days), read +0.84 to +0.87, +0.63 to +0.70, +0.24 to +0.30 at a time constant
of 36000 ticks, the static ceiling; +0.72, +0.59, +0.15 at 12000; and −0.6 to −0.9 at 4096. A fast
normalization high-passes the state and removes the value with the level; synaptic scaling runs over
days, and here it must. The same statistics begun from nothing on the fit page read −0.65 to −0.91
on the next pages: a head fit while its input's scale is still forming does not read the formed
scale, which is the transient the scratch harness showed (the head's norm 300 at two thousand
ticks, 15 at nine thousand), a birth phenomenon that the forgetting head outgrows once the
statistics have formed. So the statistics are born with a prior on the scale (one unit, weighing
tau/32 ticks) and their time constant is 36000. Written (vcrit_norm_tau, vcrit_rls_prior,
night_keep_bands; the statistics dropped at a change of units on load), the organ tests 16 of 16.
Runs 145 and 146 (out of the credit) and 147 and 148 (in it at weight 1 through the gain) launched at
16:45 with the homeostatic input at 36000, the prior at 3 days, the slow state kept across sleep,
otherwise as 141 to 144, which run on to day 20 as the raw baseline (their day-6 value instrument on
fresh days: +0.11, −0.06, +0.15, −0.05; their pages at day 6: +0.08, +0.81, −0.21, +0.38).

**Day 53 and the evening's plan, 17:15.** The diary's day 53: 138 smiles, no frown, no turning-away,
attention 0.90 at the end, 63 known words; six clean days since the reply was withheld. Day 54
began at 17:22 under the same typist; a continuation (scratchpad/continue55.sh) keeps days 55 to 57
under the old recipe so the body is never without a teacher, and the boundary script, if armed after
the refined critic's day-6 reading (runs 145 to 148, due near 18:40), takes over from day 56. Runs
145 to 148 through day 3: reliabilities +0.60, +0.18, +0.60, +0.03 on day 1, then −0.15 to +0.48 on
days 2 and 3, the statistics still forming (their time constant is four days); runs 141 to 144
through day 18 at −0.16 to +0.44, their day-16 mouths 45 to 48 of 48 cues started, talked over 48 to
85 words a day, run-on 11 to 13 words past an answer, as the ear actor read at this age before.

**The refined seeds' first five days, 17:35.** Runs 145 to 148 read their reliability on their own
stream at −0.23, −0.32, −0.08 and −0.12 on day 5, after −0.25, −0.24, +0.08, +0.18 on day 4: the
homeostatic form is consistently below zero where the raw form (141 to 144) hovered about it and
the offline steady-state replica read +0.85. A consistent sign is a defect, not noise. A scratch
copy is living a page under the refined critic with its inputs and rewards recorded, and the same
statistics are being recomputed offline on the recording, to compare the two heads at every solve:
if they agree, the mathematics is at fault on these bodies; if they differ, the live loop is. The
day-6 value instrument on fresh days, due near 18:40, is read alongside. Runs 141 to 144 ended at
day 20 (reliabilities −0.15, −0.02, +0.31, +0.27 on the last day); their day-20 instruments run.

**The live head against its replica, and against the pages, 17:50.** A scratch copy lived a page under
the refined critic with its inputs and rewards recorded, and the same statistics were recomputed
on the recording: the live head's direction and its replica's agree to three decimals at every solve
(cosine 1.000), so the live loop is the mathematics. On four fresh pages of run 148's day-6 body the
frozen live head reads −0.88, −0.81, −0.65 and −0.92, while recursive least squares fit on one of
those pages with the body's own statistics (spanning its days) and the same prior reads the others
at +0.81, +0.68, +0.92, the ridge ceiling; every prior from 100 to ten times the evidence reads the
same. So neither the prior nor the statistics is the gap: the head learned the value of the days it
lived and reads the pages wrong. The days it lived had the reply typist (the parent answers what the
body says); the pages instrument runs without it (the parent follows its plan and ignores speech),
and in those two worlds speech has opposite consequences. Four pages with the reply typist on are
being lived by the same body: if the live head reads them, the critic reads its own world, and it
was the yardstick that changed worlds.

**The confound is the night, 18:22.** Four fresh pages with the reply typist on (the world runs 145 to
148 live in) lived by run 148's day-6 body. Among the three typical pages the least-squares ceiling
is +0.59 to +0.83 in every direction; the fourth page (55 rewards, the return's spread 4.4) reads
−0.04 to −0.37 from any fit and fits nothing, an outlier. Recursive least squares fit on a typical
page with the BODY'S OWN statistics (the running mean and scale spanning its days) and the same
prior, the live configuration to the letter, reads the other typical pages at +0.65 and +0.74, at
the ceiling; with the level free or pinned, with the prior absolute or relative to the evidence, the
same. The live head, fit on the days the body lived, reads those pages at −0.80, −0.68 and −0.66.
Same estimator, same statistics, same prior, opposite sign: the difference is the data. A fresh
page has no night in it; the live head's window (forgetting at 12000 ticks, a day and a third)
always has one. The night's lessons move the cortex, and with it the slow bands, coherently
across all 768 inputs, while the next day's reward level is also different; a least-squares head
fed both days learns the coherent shift against the change of level, a between-day confound of
enormous leverage that reads as noise or worse within any single day. The bands zeroed at night
(the old form) had no between-day offset, and the raw head of 141 to 144 read fresh days at about
+0.1 to +0.3; the bands kept across sleep removed the morning ramp and brought the offset. Three
arms launched at 18:21, six seeds, all with the homeostatic input at 36000 and the head out of the
credit, to remove the confound three ways: A (149, 150) the head's window inside the day
(forgetting 4000, the prior three of them); B (151, 152) the statistics re-formed at wake
(vcrit_norm_wake: the count returns to its birth value at night, so the morning's mean and scale
form again at the birth rate); C (153, 154) the bands zeroed at night as before. Runs 145 to 148
run on as arm D (their day-6 fresh-day instruments are computing). Tomorrow morning's day-16 and
day-20 fresh-day instruments choose. The diary keeps its recipe tonight: day 54 ended with 143
smiles, no frown, no turning-away, the seventh clean day; day 55 began at 18:16 under the same
typist (scratchpad/continue55.sh, days 55 to 57).

**The night's teaching, 18:24.** Both bodies teach on through the night under their standing recipes:
the diary days 58 to 62 (scratchpad/continue58.sh; the words hat, car, bee, fish, sun), the second
body days 24 to 28 (scratchpad/after_b3_day23.sh; fast, slow, new, black, white), each day with its
own planner. Ten fast seeds carry the critic's four arms to day 20 by the small hours; the fresh-day
value instruments at days 16 and 20 are read in the morning, and the diary's boundary waits for them.

**The diary's mouth at days 55 and 56, 19:06.** The boundary probe (eight fixed cues on a scratch copy,
the parent off) fell from 43 of 48 started and 36 full at day 54 to 35 and 29 at day 55 and 34 and 26
at day 56, after seven days at 41 to 43. The gate did not change (bias +0.33 to +0.38 through days
54 to 56, the listening weight −0.84 to −0.95, the bout weight −0.35 to −0.31). What changed is
which lines the store returns: "I had " answered 'milk gone' six of six on day 54 and 'ball under
dog' or 'balld dog' on day 56, none started; "give " went from 'book fast' five of six to three. The
last days' teaching was black, white and yellow beside ball and dog, and the old lines "I had
milk" and "give book" were not repeated, so their memories faded under the store's daily forgetting
while the recent lines won the completions. The parent's page is unchanged (143 smiles, no frown,
no turning-away on day 55). Not an organ's defect but the curriculum's: tomorrow's briefs should
repeat two of the old cue lines each day beside the word of the day.

**The second body's day 23 was a quiet one, 19:10.** Its planner, woken by message when the session
began at 18:29, appended nothing the typist reached: two backlog lines were typed in forty minutes,
7 smiles, no frown, no turning-away, and the night came at 19:09. The planner has been told the day
is over. Days 24 to 28 follow under the armed chain with fresh planners.

**Two turnings-away at the start of day 56, 19:21.** The session began at 19:16 with the parent's
attention at 0.59 (the typist keeps it from the day before, which ended at 0.60) and the body babbling
fragments after the night ('b bbbbb b b', 'fas t b ook'), each fragment wearing attention by 0.04:
0.49, 0.25, 0.14 and the parent turned away at 19:18, reset to 0.35, wore down again to 0.15 and
turned away at 19:20. The first turnings-away since the reply was withheld on day 48. Sleep pressure at
the session's start has climbed from 4847 (day 52) to 6640 (day 56) as the sessions lengthened and the
boundary probes lengthened the waking gap between them, so the body wakes into each session more
tired; the morning after a night also begins with the slow bands zeroed (this body keeps the old form).
No change to the body or the parent tonight; the parent's rule is doing what it does, and the turnings-
away are the body's to feel. Whether they recur through the day decides whether the morning is a
pattern.

**Runs 145 to 148 ended at day 13, 19:24.** Their form (the homeostatic input at 36000, forgetting at
12000, the slow state kept) read fresh pages at −0.7 and is the arm the others correct; the machine
is at a load of 30 with ten seeds, and the three decisive arms need the cores. Their day-6 fresh-day
instruments run on to completion as arm D's one reading.

**The body alone on an empty page, 19:27.** The turnings-away of day 56 (six by 19:30, the body
saying 'bbbbbb' to "give ") have a cause outside the body. The chain that runs the diary's days
took the boundary probes (the mouth, the taught lines, the unheard combinations, the cortex alone)
on a copy BEFORE starting the teacher, and under today's load those probes took 28 minutes instead
of 10. Through that gap the body is awake and alone: its sleep pressure at the session's start rose
from 4847 on day 52 to 6640 on day 56, and it spent the gap speaking to a page no one writes on. It
woke into day 56 tired, its mouth having drifted to the one letter its recent lines begin with, and
the parent, who keeps yesterday's attention (0.60), turned away four times in eight minutes. The
instruments distorted the life they measured. From the next chain (day 58 on, and the second body's
after day 28) the probes run in the background on the boundary copy while the teacher begins at
once (scratchpad/teach_days3b.sh, teach_days_b3.sh); the copy is the same copy, so the probes
measure the same body. The second body's short day 23 had the same cause.

**Housekeeping, 19:45.** Twelve monitors from before this session's compaction were still running, tailing
the logs of runs 69 to 132 and of boundaries long past, with their Python drivers and a day-old planner
watcher; all ended, with two stale waiters. In the sweep the second body's own monitor went too and was
restarted. What runs now: the six arm seeds and their instrument loops, the two servers, the diary's
day-56 typist, the chains (55–57, 58–62, 63–66; the second body's 24–28), the planners' foreground
waits, and five monitors.

**The REM ablation, queued, 19:56.** REM's benefit to behavior has never been isolated in its detached form
(its early form, through the trunk, undid a third of NREM's gain and was detached for it; since then its
own measure, the forecast cosine, rises across a life and the gauge is untouched). Runs 155 and 156
(REM off: rem_rounds 0) against 157 and 158 (REM on), otherwise the ear recipe with the critic out of the
credit, twenty days each with the instruments at 6, 16 and 20, start when the six arms finish tonight
(scratchpad/after_arms.sh), to be read on the mouth probe, the run-on, the cue answers and the talking
over the parent.

**Day 56 and the day-57 probe, 20:02.** Day 56 ended with 79 smiles, no frown, five turnings-away (all in
its first seven minutes) and attention 0.61 at the end; 66 known words. The day-57 boundary probe reads
33 of 48 cue answers started and 23 full, the third day down (43/36, 35/29, 34/26, 33/23). The lonely
morning is not only tiring: a body that speaks for half an hour to a page no one writes on trains its
cortex on its own babble, and what it babbles now is the one letter its recent lines begin with. Day 57
is the last day with that gap (the running chain keeps the old order); from day 58 the teacher begins at
the boundary and the probes run beside it. If the probe keeps falling without the gap, the store's
recency and the stutter are the next suspects, and the old cue lines must return to the curriculum.

**The critic on the diary tonight, out of the credit, 20:24.** The arms' live reliabilities through day 8
or 9 hover about zero like arm D's (A −0.04 to +0.28, B −0.37 to +0.47, C −0.19 to +0.16); their
fresh-day instruments are computing. Two facts argue that the diary is a different case from the
seeds: its slow bands do not drift across nights (0.98 to 1.00 at the same tick over seven nights,
where a young body's read 0.41 to 0.55 over five), and its value needed three days of pages to read
(+0.75 by ridge, +0.87 by the recursive head) where one day read nothing. So the diary takes the
decorrelated critic at the 57-to-58 boundary with its weight in the credit at zero: the homeostatic
input at 36000, the prior three days of evidence, forgetting at 36000 (four days, since nothing
drifts), the bands zeroed at night as it has always lived. The head learns and is read; it moves
nothing. Its reliability across days 58 to 66, and a fresh-day instrument on the morning's copy, are
a live reading on the body that matters, beside the seeds'. The reply typist stays off.

**The 57-to-58 boundary, 20:57.** Day 57 ended with 145 smiles, no frown, no turning-away, attention 0.72,
67 known words (the body said "bird" unprompted, the day's word). At 20:57 the body was saved (a copy
at data/body2_before_rls_day57.pt), the serve restarted with the decorrelated critic learning at zero
weight (the homeostatic input at 36000, the prior three days, forgetting at four days, the bands zeroed
at night as before), and day 58's teacher began at once: sleep pressure 3045 at the session's start
against 5988 to 6640 on the mornings with the gap. The boundary probes now run beside the day on the
boundary copy. The critic's reliability appears in the night rows from here on.

**The arms at day 6 on fresh days, 21:10.** The frozen heads on four fixed-pace instrument days: arm A
(the head's window inside the day) −0.01 and −0.02; arm B (the statistics re-formed at wake) −0.22
and −0.20, each within 0.05 across its four days; arm C (the bands zeroed at night) −0.30 and +0.03.
None reads, and arm B's is not noise: a head that reads four unlived days at the same −0.2 has learned
something real about its lived world that holds with the opposite sign in the instrument's. The
instruments (the four-day value instrument and the pages) pace the parent every 240 ticks; the lived
day paces the parent by its attention (240 × (1.6 − e), from 144 ticks when it is rapt to 312 when it
wanders), answers the body's word with a line that holds it, and talks on until the night. In the
lived world a state that holds the parent's attention is followed by faster lines and more reward;
in the instrument's world the lines come at the clock. Two real lived days of run 149's day-6 body
are being recorded beside two fixed-pace pages; heads fit in each world are read in the other. If a
head fit on a lived day reads another lived day and not the pages, the critic has been reading its
own world and the yardstick changed worlds, as the pages did this morning.

## The drifting coordinates (2026-09-05, 21:45)

I recorded a real fast day with the body's own critic saved beside it, and read the head against the day's actual returns: −0.59. Then I fit the same head, from the same evidence, on the same day, in fixed coordinates: +0.71. Same inputs, same rewards, same rule; the only difference was that the body's homeostatic statistics were moving while the head accumulated. Over that day the running mean moved by two standard deviations and the scale by a factor of 2.4. The head solves A w = b, and A and b are sums over a window of days; when every term of the sum is written in different coordinates, the sum is the evidence for no head at all. Replicating the moving statistics offline reproduces the failure (+0.40 with cosine 0.44 to the batch head) — so this is the whole of it, not a symptom of something else.

The fix is not to freeze the statistics. A body that meets a new input must learn its scale, and a robot's inputs will not arrive standardized. The fix is to move the statistics out of the evidence: A and b accumulate on the raw inputs, which never move, and the running statistics shape only the prior at the solve — the ridge becomes the metric δ·diag(sd²) on the weights, with the level free, which means "small in units of the input's own scale" without touching the data. Offline that reads +0.74, the same as fixed coordinates. Committed as fa86969, the evidence versioned (vc_form 2) so a body loading old standardized evidence starts fresh from the prior instead of mixing forms.

The fifth defect, and the pattern is now plain: the gradient head read the anti-correlation (XᵀG), the weak prior let the noise in, the fast statistics destroyed the standardization, the night zeroed the state, and the moving coordinates corrupted the sum. Each was a place where the estimator's own machinery, not the world, wrote into the evidence. Biology's critic has no such machinery to drift: dopamine reports the error against a synapse whose input is the raw firing rate.

Tonight: arms E (159/160, forgetting 12000) and F (161/162, forgetting 36000) run with the new form, the slow state kept across sleep, out of the credit, with the fresh-day instrument at days 6 and 16; the REM ablation (155/156 REM off, 157/158 on) began at 21:40 when arms A–C ended (none of the three read at day 6, so their question was moot once the coordinates were the cause). The diary takes the new form at the 58→59 boundary, still out of the credit, its reliability logged. Tomorrow reads: if the head reads fresh days at +0.5 or better by day 6 and holds at 16, the fourth clause has its estimator and the diary takes it into the credit at a boundary; if not, the remaining suspect is the horizon, not the estimator.

## The arms at day 6 on fresh days: the sign is fixed, the horizon is the limit (2026-09-05, 23:55)

The fresh-day instrument on the four bodies with the prior-as-the-metric critic, at day 6, four days each that the body never lived: +0.16, +0.17 (forgetting 12000) and +0.20, +0.36 (forgetting 36000). Positive on fifteen of sixteen days. The four controls with the old gradient critic on the same instrument tonight: +0.10, −0.10, +0.40, −0.27 — a mean of zero and negative days everywhere. So the fifth defect was real and its fix holds on days the body never saw: the critic's sign is right and its coordinates no longer drift. But +0.25 is not a value a body can steer by, and it is not the +0.74 the same evidence reads offline on a recorded lived day.

The same instrument printed the answer beside it. Each band of the ladder carries its own value head at its own horizon, trained by the plain gradient rule. On the arm bodies the slow band's head at horizon 4096 read +0.67, +0.55, +0.62, +0.71 on the fresh days — the strongest value readings this project has produced on a live head — while on the controls, whose slow state is zeroed at night, the same head read −0.54, −0.72, −0.43, −0.78. Two things in one reading: the night's zeroing was the fourth defect exactly as diagnosed (kept state turns −0.6 into +0.6 on the identical head), and at horizon 4096 the value is plainly in the slow state, where at 1024 it is faint. A 1024-tick return is four minutes of page: a few exchanges, dominated by when the parent's next smile happens to fall. A 4096-tick return is a quarter of an hour: the parent's attention regime, which is what the slow bands hold. The ladder critic at 1024 was asking the slow state a question it cannot answer well.

Arms G (163/164) and H (165/166) started at 23:54: the ventral critic discounted at 1 − 1/4096, the prior as the metric, forgetting 36000, the slow state kept; H reads only the two slowest bands. The instrument has to judge that critic at its own horizon rather than band 5's (patched tonight). Runs 155–162 continue to their day-16 and day-20 readings; arm F's day 16 will say whether 1024 improves with evidence. The diary keeps the horizon-1024 form out of the credit until the arms read.

A note on method: the REM ablation's day-6 mouth reads no difference (REM off 47, 32 full; REM on 45, 32); day 16 decides. And the eight probes tonight ran an hour at the machine's lowest priority behind twelve bodies; the served bodies were discovered running at niceness 10 and 5 from the shells that launched them, so the instruments and fast runs now sit below them by decree rather than by luck.

## REM stays (2026-09-06, 01:55)

The REM ablation read null on four bodies: at day 6 the mouth 47 and 32 full without REM against 45 and 32 with it; at day 16, 40 and 46 against 37 and 34. The detached REM earns nothing a young body's yardstick can see. The ruling is to keep it: it costs nothing, and its contribution is expected only when the cortex alone predicts life well (about 60% next-symbol now; the dream-to-life cosine 0.45 on fast bodies at day 16, 0.85 on the diary at day 61) and imagination has a job that only samples can do — a critic learning from imagined page. The ablation runs were stopped at 01:50; no body runs without REM from here on.

## The sixth defect: the critic's input (2026-09-06, 03:00)

Tonight's chain, in the order the instruments answered. The four arms with the fixed estimator read fresh instrument days at +0.16 to +0.36 at day 6 and did not improve by day 16; the critic at horizon 4096 read −0.21 to −0.80. Then the honest yardstick: a day-16 body living one more real day of the fast parent, its own head against that day's returns: −0.30 and −0.70. Not the instrument's fault. The night as a doorstep was next — the night passes no ticks, so an evening state is followed one tick later by a rich morning — and the two-day recordings killed it: the head reads the return through the night exactly as it reads the return cut at it. The prior's strength was next: heads re-solved from the body's own evidence at a hundred times the prior still read the next day at −0.6. Which bands was next: any band set fit on one day, from the 64-tick band to the slowest, read the next day at random sign.

So the return itself, with no model in the way. On a lived day the reward is anti-persistent at the day's scale: the past-4096-tick reward trace reads the next-1024 return at −0.77 and −0.95. The fast parent's attention cycles. A rich stretch drives it to the ceiling, the body is talked over, attention collapses, the parent turns away (twelve to fourteen times a day on these bodies), then recovers. The phase of that cycle is the value at this horizon, and it lives in the reward history. Eight numbers — the felt reward averaged at each of the ladder's clocks, tonic dopamine at eight timescales — fit on one day read the next at +0.83 and +0.59, the best fresh-day readings this project has produced; adding the bands' 768 dimensions to them destroyed the transfer, because a least-squares head uses the bands to fit the training day and the bands do not carry the cycle in a form that survives to the next.

Biology had this all along: the ventral striatum reads tonic dopamine, the reward rate, beside the cortex's state; the body already kept one such trace as the differential critic's baseline. The critic now reads the eight traces (vcrit_traces 1) and, for the first arms, no band at all (vcrit_bands "-"). Arms I (167/168, horizon 1024) and J (169/170, horizon 4096 — the cycle's own scale) started at 02:52 with the critic out of the credit; the lived-day yardstick reads their day-6 bodies as they appear. The bands' content-specific value is not gone from the design; it comes back once the transferable part is in the head, under a prior that keeps the bands from fitting the day.

Two smaller results tonight. The REM ablation read null and the organ stays by ruling; a scratch night on the diary's day-63 body with sampled instead of greedy dreams read identical to the last symbol (49/82 and 44/82 on the cortex trace), so the sampling switch stays off with nothing lost. And the diary had its best day of the lineage: 230 smiles, no turnings-away, 73 words, five clean days running.

## The clock (2026-09-06, 03:40)

The traces alone did not hold up. Arms I and J read their fresh lived days at +0.06 and −0.18, and the diagnosis was in their own evidence: the prior's metric uses the homeostatic variance, which was born at one unit and, at the 36000-tick time constant, still held 0.0099 on every slow trace after six days against a true variance of 0.0001 — a prior a hundred times too strong, the head crushed to its level. The unregularized solve of the same evidence read +0.48. Fixed: the statistics' scale is set by the first samples (n from 1), which the metric form makes safe, since they touch only the prior.

Two of tonight's trace readings were also contaminated, and it matters. The +0.83 on run 161's second day came from traces that included the current tick's reward, the first term of the return itself; with the traces holding only the rewards before the tick, as the body's do, the honest transfer of the eight traces is −0.34 on that body and +0.60 on the other. And the −0.95 single-trace readings came from traces restarted at zero on the second day, a ramp that mimics time of day. Clean, with the traces continuous across the night: the return's own persistence dies within a few hundred ticks (+0.8 at 256, +0.1 to +0.5 at 1024, nothing consistent beyond), so there is no hour-scale cycle to read. The one regularity that carries across days is the day's profile. Rewards are front-loaded and the return falls across the day at −0.4 to −0.6 with time on every fast day; time of day alone, fit on one lived day, reads the next at +0.60 and +0.50 at horizon 1024 and +0.72 at 4096, where the traces flip sign between days. The diary itself, from its own page log: on six of its last seven days the return falls across the day at −0.81 to −0.95, smiles per quarter running 22 down to 12, and its return persists at lag 1024 at +0.7 to +0.9.

The body's clock is its sleep pressure over the threshold that brings the night: adenosine, a robot's uptime, an internal quantity no rule writes. The critic now reads it (vcrit_clock) beside the traces. Arms K (171/172, traces and clock, 1024), L (173/174, traces and clock, 4096) and M (175/176, the clock alone, 1024) started at 03:35, out of the credit, with the lived-day yardstick at days 6 and 16. What such a critic gives the credit is the daily baseline: the gate's error becomes the reward against what this time of day usually brings, which is what a critic is for, and the part the body cannot control is what it removes.

## The behavioral yardstick (2026-09-06, 03:50)

The user, on the readings: the longer bands should not try to match what the body did before; they should chase reward at a higher dimension. Right. The correlation of a head with the return on a fresh day is a diagnostic — a critic anti-correlated with the return poisons the credit, and that diagnostic caught six defects — but it is not the goal. A critic needs to be a valid baseline, not a good forecaster, and the slow bands' work is to steer the gate toward what pays over the day. So the test that matters starts now rather than after a perfect reading: runs 177 and 178 carry the traces-and-clock critic in the credit at its own measured reliability, against 171 and 172 with the same critic out of it, judged by smiles per day over twenty days on fresh seeds (compare_credit.py). The day-6 and day-16 yardsticks keep running beside it.

## The gate is a coin (2026-09-06, 04:25)

The user asked me to stalk the body rather than measure it, and the first stalked day answered a question no yardstick had asked. Every tick of a fast day written down, then read around each turning-away: the body speaks on 54% of ticks whatever is happening. The gate's probability of speaking is 0.53 with a spread of 0.07 for the whole day — identical while the parent types (0.527 against 0.529), across every quartile of the parent's attention, and in the hundred ticks before each of eleven turnings-away. With a floor of 0.05 that is the gate's logit at zero. The smiles come from what the body says, not when; the turnings-away are the coin landing on "speak" while the parent types, 111 talked-over misses and 94 past-its-turn misses in one day, none of them felt: 112 positive rewards, zero negative, in twelve thousand ticks.

The lesson explains it. The gate's credit is a tonic drive of 0.25 for every act ("babble is its own reward"), half its own confidence as interest, minus an effort cost, plus the dopamine error — and the dopamine error is +0.24 once a hundred ticks. The gate is optimizing its own drives; the world is a rounding error in them. A constant drive to act is not what the songbird paper the code cites found: that dopamine encodes performance error. Under the law this is a hand-written reward inside the architecture swamping the grounded one, the seventh defect, and the only one of the seven the instruments could not have found, because every instrument read the critic and none read the mouth's ears.

What can teach the gate to be quiet is the smile it expected and did not get. At the 16-tick band that error is a few hundredths; at the long horizon, where the expected reward is around twenty, it is of order one, and a hundred ticks of talking over the parent costs about three against a smile's one. That is the ventral critic in the credit, running since 03:50 (177/178), and from 04:17 with the drives cut to a fraction (179/180, tonic 0.05, interest 0.1). The environment's half — a parent whose face shows it was interrupted, and whose turning-away is visible — is a caregiver rule, and waits for the user's word. Every other organ on the stalked day read healthy: the night, the store, the dreams, the reply yielding to the parent, the critic's errors centred on zero.

## The newborn's shoes (2026-09-06, 05:00)

The user asked whether a newborn even has a reward circuit at the long timeline. Biology says no, or not yet: prefrontal cortex is the last region to mature, infants' contingency learning works only within a second or three, and their discounting of anything further is nearly total. What a newborn has is the short circuit, and it does the whole job: turn-taking appears by three or four months, learned from the answer that comes when babble falls in the mother's pauses and does not come when it falls over her. The infant expects the answer; its absence is a negative prediction error a second long. And the drive to babble is real but not constant: infants babble more when answered contingently, less when not.

Three consequences for the body. The long critic should earn its weight as it proves reliable rather than be forced early — which its reliability gain already does, though tonight that gain read zero because the within-run reliability it uses ran negative, so the in-credit pair (177/178) was a control in disguise; the true in-credit test now runs at a fixed weight (arm Q, 183/184). Turn-taking should come from the fast band, the infant's route, and the stalk shows why it does not: the fast error after being talked over is a few hundredths against a constant drive of a quarter per act. And arm N showed a body with no drive falls silent, like an infant nobody answers: 6 and 9 smiles on its second day. The version biology points at is between: the drive proportional to the recent reward rate, the tonic trace the critic already carries — tonic dopamine as the average reward rate setting vigor. Babble that earns smiles feeds itself; babble over the parent starves. No schedule, no rule, a quantity the body has (gate_tonic_rate; arm P, 181/182, out of the gate: 0.05 + 8 × the 256-tick trace, a mean drive of 0.27 on a scratch day, 0.1 in a lull and 0.4 in a burst).

The stalks of the day agreed across three bodies: the gate's probability of speaking is 0.53 to 0.54 on the control, on the in-credit body, and on the diary's own day-65 copy, whatever the parent is doing; the diary after sixty-five days shows the faintest ears (0.48 after being talked over). The environment's half — a parent whose face shows it was interrupted, and whose turning-away is visible — is written as switched-off options and waits for the word. The day's other lesson was about the machine: ten fast runs and their probes starved the served body until its misses were four-fifths "late" and its smiles a third of their pace; every instrument now runs at the lowest priority and the served bodies come first.

## The smile was never expected (2026-09-06, 05:15)

Around each smile on the stalked day, tick by tick: the fast dopamine error is −0.03 in the ticks before, +1.97 on the tick the reward of +2 lands, −0.03 after. The smile lands one or two ticks after the word's last letter, every time, and the fast band predicts none of it. Dopamine in this body still reports the reward; the response never moved to the predictor, because the dopamine band's head is the same one-pass gradient head that failed at the long horizons. This is the eighth defect, and it closes the circle on the gate: a smile that was never expected cannot be missed, so a word said over the parent costs nothing the gate can feel, and the infant's route to turn-taking — the expected answer that does not come — was never open.

The gate's lessons, watched: the dopamine credit for acting minus resting is −0.009 over a day, the drive for acting is netted out by an effort cost at a fatigue of 22 against a scale of 10, and three terms that sum to nothing teach nothing. The ear weights are not zero — −0.4 on the parent's symbol, +0.9 on its own last act — and at the parent's actual symbol ticks the gate does hesitate, 0.38 against 0.56. The turn-taking signal exists in the dopamine credit: an offline replay of the same lessons with the drives removed drives the ear weight to −3.5 and the gate to 0.42 at the parent's symbols against 0.96 otherwise. The drives bury it. Arm N cut the drives and the body fell silent; arm P (181/182) lets the drive follow the reward rate; arm Q (183/184) is the true in-credit control at a fixed weight, since the reliability gain had read zero and made 177/178 a control in disguise; arm R (185/186) gives the dopamine band the decorrelated head so the smile is expected. The user's word on the parent's face when interrupted is still the environment's half.

## The frown (2026-09-06, 05:25)

The user's word: the parent may frown when talked over. It is the first grounded cost this lineage has ever felt for a thing it does wrong, and it is the parent's honest face rather than a rule about words. On the fast page it is a light, brief frown, at most once a minute of ticks, at the word the body said over the parent's turn. Arm S (187/188) carries it with the fast critic that expects the smile; arm T (189/190) with the old fast critic, so the frown's own effect can be read apart. Both are judged against their frownless twins on smiles, turnings-away, and the gate's ears on a stalked day. The served typist gets the same as an option, off until a day boundary after the fast seeds read; it had never registered being talked over at all under the current chain, since that branch lived under the reply road. The turned-away face stays off: the user named the frown.

## The ninth defect: the loader dropped the fast critic's memory (2026-09-06, 05:50)

The reading that the fast critic expects nothing was taken through a hole. The decorrelated fast head keeps its evidence in five buffers beside the ventral critic's. The loader popped both sets out of the saved organs to size them by the life, restored the ventral set, and never restored the fast set. A reloaded fast-critic body therefore met its prior with no evidence, and its first solve, at tick 64, crushed the head toward zero. The running bodies were never touched; every copy of them was: the functional checks, the stalked days and the day copies all read a head that had just been erased. On arm R's own saved body the evidence is real, and after 28,000 ticks the prior is a quarter of it, so the live head has been learning for three days unwatched.

Fixed in the loader; the sixteen organ tests pass. The stalks now write the fast value down tick by tick, and the expectation of a smile will be read as a rise of value in the ticks before the reward, not as the error on the reward's tick. A band that barely moves in one tick cannot cancel an impulse of two, and no critic was ever supposed to; the shift biology describes is the rise before, and the fall of the response at the reward is only its consequence at the horizon of the band.

## The tenth defect: the critic's input is blind (2026-09-06, 06:49)

The fast critic was given the dopamine band's own state to read, and the ventral critic the ladder's. On two recorded days of one body, with the cortex's daytime lesson on or off, no set of bands predicts the sixteen-tick return on the day it did not see: the best reads 0.13, the rise of its value before a smile a tenth of the smile. The cortex's own stream vector reads 0.11. The body's last eight symbols, one-hot, through a ridge, read 0.34 and rise a fifth of a smile; through a small nonlinear head, 0.47 and half a smile; through a born random sparse expansion of four thousand thresholded units and a ridge, 0.42 to 0.47 and half a smile. The smile is in the stream. The ladder blurs it away by construction (each band's gate sits at 0.88, so band 0 is a born tanh map of the cortex's vector, and the cortex's vector is a next-symbol forecast, not a record of the word being said), and a linear reader of any cortical state cannot recover which word is ending.

This is the striatum's shape and the cerebellum's: a fixed sparse expansion of afferents, then fast linear learning under dopamine. It is also what a robot would need, since the expansion cares nothing for what the input is. The fast critic will read a short delay line of the stream through a born expansion, as an option tested on fresh seeds first, judged by the rise of its value before the smile.

## The striatal input, live (2026-09-06, 07:23)

Arm U was born with arm S's recipe and the fast critic reading a line of the last eight events through the born expansion. On its third day, stalked with the parent road on, the fast value rises seven hundredths of a smile in the eight ticks before a smile and the error on the reward's tick is +1.82: the first expectation of a smile any live body of this lineage has shown, and a small one. The value is highest while the parent types, where the frowns land, because the heard symbols flood the line and smiles follow the parent's lines. Offline, on the same body's recorded days, the line's ceiling is low: under the parent road even the body's own eight symbols read an unseen day at 0.16, against 0.34 on the flat-rule days. The parent who wants a reply smiles only after the child's four quiet ticks, and quiet was not an event of the line, so the value could not rise through the pause. The pause is exactly where the gate's rests need their credit.

The behavior moved more than the value: over the five days both arms lived, U was smiled at fifteen more times a day than S on every day, and twenty-nine more than the arm with the frown and the old critic. The ladder of the credit machinery on matched days: the frown alone thirteen smiles a day over the old critic, the decorrelated critic in the credit three more, the striatal input fifteen more, turnings-away easing a little at each step.

A tick of quiet is now an event of the line, which carries sixteen events into two thousand and forty-eight units. Arm V is born with it.

## Quiet ticks, and the diary's turn (2026-09-06, 08:23)

With a tick of quiet as an event of the line, the value before a smile rose a little more (a tenth to a fifth of the smile on arm V's third day, against a fourteenth on arm U's), and the bodies were smiled at less: sixty-five and eighty-five times on the third day with quiet ticks in a line of eight, sixty-eight and seventy-one in a line of sixteen, against ninety-seven and ninety-nine without them. The behavior is the yardstick, so the quiet ticks are an option and off. Offline, every readout of the line read an unseen day at about a third and rose almost nothing before the smile, while the live head, which solves itself again every sixty-four ticks, rose a fifth: an online head follows a day that drifts and a parent who smiles at a fraction of completions, and a head fit yesterday does not. The live rise is the instrument.

The diary takes the striatal fast critic tonight, at the boundary after its sixty-eighth day, in arm U's form, with the ventral critic still out of the credit; and its typist takes the frown, on the user's word and after the fast seeds measured it. The reading tomorrow: smiles, turnings-away and frowns of the sixty-ninth day against the days before it, and the rise before the smile on a stalked copy.

## The frown without the reply (2026-09-06, 10:41)

The diary took the frown on its sixty-ninth day, with the striatal critic and its old parent who smiles at every known word at once. It lost half its smiles, drew sixty-eight frowns and ten turnings-away, and its parent's attention fell through the day to a third. Fresh seeds under the same parent say the same on every paired day: with the frown, thirty to forty fewer smiles a day and three or four more turnings-away, whichever critic they carry, through their sixth day. Under the parent who withholds the reply until the child is quiet, the same frown had been worth thirteen smiles a day and made the gate's ear. Punishment alone teaches nothing here; a smile that waits for the child's quiet teaches yielding, and the frown sharpens what the waiting smile has begun. The frown comes off the diary's typist at tonight's boundary; the striatal critic stays.

## The ear, bought (2026-09-06, 11:12)

A copy of the diary taken this morning, after its two days under the frown, was stalked under the same parent: it speaks on twenty-seven ticks in a hundred while the parent types and fifty-five when the parent is quiet, thirty-seven after a word it said over the parent and thirty-three after a frown. Five days ago the same body was a coin at fifty-three whatever the parent did. An old body learned to yield in two days. It paid in smiles, because the parent it has smiles at every known word the moment it lands, so a child that speaks less is smiled at less, and its attention, worn by every interruption, fell until the parent turned away. The frown teaches; only a parent who withholds the smile until the child is quiet rewards what it teaches. The two parents are a choice, and it is the user's.

## The parent, chosen (2026-09-06, 12:20)

The user's word: the parent's method of teaching is mine to change however I want. The diary takes, at the boundary after its seventy-second day, the parent who wants a reply: the smile for an answer comes after the child's four quiet ticks, words said over the parent's turn earn a frown and no smile, and run-on words after the answer wear the parent's attention. On fresh seeds this parent made the gate's ear and, with the striatal critic, the highest reward within its own regime; the diary already carries the ear it bought under the frown, and this parent rewards it. The second body keeps the permissive parent for now, as the contrast.

## The actor (2026-09-06, 12:29)

Reward could choose when the body speaks and what it keeps; it could not choose a word. The cortex proposed the next symbol from its forecast of the parent, and the mouth took it. The striatum now has a second head from the same delay line the fast critic reads: a bias over the next symbol, through tanh, added to the cortex's proposal. It learns as the gate learns, by dopamine times an eligibility of what it said against what the forecast expected, on the input it saw; the weights forget slowly. Born at zero, it changes nothing until a smile follows a word said in a context, and then it leans toward that word there. Two fresh bodies carry it under the parent who wants a reply, against arm U as their control; the yardsticks are smiles, turnings-away and cues completed a day. It does not enter the diary without the user's word.

## One life, watched (2026-09-06, 14:44)

The user's word: no more arms; build the two organs, put the complete architecture in one body at the larger size, one seed, and watch its every move. Working memory and the planning actor were built and passed the organ tests within the hour, and at 14:45 a body of the second body's size was born with everything: the cortex and its nights, the hippocampal store with its marks, the striatal input with the fast critic on it, the slot beside it, the planner over the cortex's proposals, the gate with its ears, the ventral critic on the ladder with the traces and the clock behind the reliability gate, the own-babble target fading with distance, and the parent who wants a reply, with the frown. Every tick is written down; a digest closes each day. At its third day it moves into the room, where the parent's world has a state to be right about. The reading is the play by play: when a cue lands, does the slot hold it, does the planner choose the answer for its value, does the critic's value rise at the want and fall at the giving, does the gate wait while the parent types. Nothing else runs beside it except the two served bodies and their teachers.

## The review, and the eleventh defect (2026-09-06, 19:12)

The user asked for a second pair of eyes on everything before the one-go life: "spawn sub agent to review show it our tests
and archecture make sure everythings looking good", and "also make sure no cheats". An independent reviewer read the body,
the environment, the instruments, the tests, and the watched bodies' saved evidence, and returned ranked findings. The first
was a defect I had not seen in nine months of temporal-difference code: the chain did not close. The value's source was the
state at the end of the previous tick, after the body's own symbol; the target was the state mid-tick, after the world's. The
saved evidence carried the proof: the LSTD matrices, which a closed chain makes symmetric up to the trace, were 16 to 30
percent asymmetric with negative eigenvalues. Every dopamine reading of the last two days had that gap in it, twice its own
standard deviation. Fixed by making the next lesson's source this lesson's target; the asymmetry fell to 0.01 (fast) and 0.000
(ventral) on a fresh stretch, and the check is now organ test 21.

The rest, all fixed and committed (1ad8c27): the bands integrated twice a tick, so their true time constants were 0.57 of the
disclosed ones; the gate's lesson learned a probability that lacked the stress divisor the act was drawn with; a third of the
gate's samples were learned twice; the working-memory slot cleared on the world's word, not only the face; and one cheat, small
but real, in the environment: the served parents read the body's sleep pressure to say goodbye. They now see only whether it
sleeps, which the page shows. The world's word as reward (world_r 0.3, two fifths of all felt reward on the reference days) was
in the physiology but not in the spec's reward section; it is there now, named beside the face. The planner's shortlist was
measured against the silence symbol's logit, and at toy size the shortlist was therefore always empty; it is now measured among
the speakable symbols. Five organ tests were missing (the striatum, the slot, the planner, the save's round trip of the new
organs, the closed chain); they exist, 21 of 21 pass.

What the review did not find: any reading from inside in the caregiver, the typist, or the fast parent beyond the sleep
pressure; any authoring of the body's words; any reward not grounded in the parent's face or the parent's act. The big body
(179M, d 1024, 12 layers) was 42 minutes into its second life with the open chain; stopped, set aside
(data/stalks/watch2_second_try.pt), and reborn at 19:13 on the fixed code with the fast head at 2048 units (the reviewer's
measurement: the 8193-wide evidence cost 182 ms a tick and a 7.7 s stall every 64 ticks). One seed, watched, one go.

## The night's shove, and the fourth birth (2026-09-06, 21:58)

The one-go body's first days answered the question I had asked of them, and not the way I hoped. Its waking organs learned
like the reference's: the fast critic rose before a smile from the first day, working memory latched, the planner ran, the
prefrontal value sat at zero weight as it should. But its nights did nothing. The reference keeps six to eight tenths of a
day's symbols after a night; the big body kept a quarter, and the night's own log said why: NREM's loss doubled or tripled
at the first step of every night (0.52 to 1.06, 0.46 to 1.64, 0.48 to 1.23, 0.46 to 1.16) and the remaining steps were spent
climbing back to where it had started. Its dreams grew more alike each night, 0.58 to 0.96. REM was innocent: the gauge was
identical before and after it.

The cause is arithmetic. Each night births a fresh optimizer, and a fresh optimizer's first step moves every weight by the
whole rate at once, in the direction of its gradient's sign. On a 512-wide, 8-deep cortex that step is bearable; on a
1024-wide, 12-deep one it is a shove that grows with the width and the depth. A brain does not begin sleep at full
plasticity; spindles and slow waves build over the first minutes. So the night's rate now climbs over its first steps
(night_warm, a disclosed constant, eight of the twenty-four steps; zero for every older body).

Measured on a copy of the day-4 body, saving to a scratch file: the inherited night, loss 0.41 to 1.01 at step two, memory
0.288 to 0.327; a fifth of the rate, no jolt, 0.288 to 0.325, the same small gain; the full rate with the ramp, no jolt, 0.288
to 0.424, three and a half times the gain. Removing the shove alone is not enough; the big cortex needs the full rate's steps
without the shove.

One error of mine, recorded so it is not repeated: a loaded life saves back to the file it was loaded from, and the night
saves. My first probes loaded the living body's own save and overwrote it with their jolted copies for forty minutes. The
life itself was untouched, holding its state in memory and rewriting the file at each day's end, but a crash in that window
would have resumed it from a wreck. Every probe now copies the file first and saves to the scratchpad.

The third life was stopped at day 5 and set aside (data/stalks/watch2_third_try.pt). The fourth, and the one meant to be the
last, was born at 21:58 with the ramp from birth and everything else as it was: 179M, seed 1, the reply parent, the frown,
the striatal critic, working memory, the planning actor, the ventral critic on the clock and the traces. The user's word for
it: "one model complete live training that works. then we get cool demo at the end of teaching it."

## The night's length (2026-09-06, 23:40)

With the shove gone the fourth life's nights gain what the reference's gain (night 2: 0.23 to 0.47 against the reference's
0.62 to 0.81, the same +0.2), but from a lower base, because the first night kept 0.29 of the day where the reference kept
0.71. On a copy of the day-1 save, with a scratch save path, the same night at four rates and two lengths: half the rate kept
0.45, the full rate 0.51, double 0.35 (its loss bouncing in the last steps and the dreams collapsing again), a fifth 0.38;
and the full rate for 48 steps instead of 24 kept 0.70, with the NREM loss ending at 0.23, both the reference's numbers.
The rate is right and the night is too short for a cortex of this width: at the same per-step size, twice the weights want
twice the steps to hold a day. A brain does not sleep longer because it is bigger, but a night here is a count of steps,
not hours, and the count is a disclosed constant (night_rounds). It goes from 24 to 48 at the day-3 boundary, the body
reloaded from its own save with the parent's seed advanced so the parent's lines do not replay days 1 to 3. No restart;
the life keeps its three days. The proof comes with day 4's night record: 48 NREM steps, and the memory after the night
near 0.7.

The REM question stays open: the big body's dreams are near copies of each other (REM cosine 0.56 to 0.72 on night 2, the
reference's 0.07 to 0.21) because REM rolls out greedily at temperature zero. A warm-REM night on the same copy is running;
if it keeps more of the day or spreads the dreams, rem_temp changes at a boundary; if not, REM stays as it is. REM's share of
the memory gauge has been zero at both sizes on every night so far; its contribution is expected at the cortex's maturity.

## REM at 0.7 is still greedy (2026-09-06, 23:55)

The warm-REM night on the day-1 copy came out identical to the baseline in every number, to three decimals. The rollout
does sample above zero temperature; the softmax it samples from is a one-hot regardless, because the readout's logits are
that peaked: a newborn's next-symbol entropy measured 0.03 nats at every size (uniform over the lexicon would be 4.7), so a
temperature of 0.7 changes nothing and the mouth itself is close to deterministic from birth. REM stays as it is. What this
opens is a question about exploration rather than sleep: a body whose mouth rarely varies its favorite explores words only
through what the world does to its state and through the planner's shortlist, which at this logit scale holds one to four
candidates. The reference learned to answer cues this way, so it is not a defect proven; it is written down as the next
thing to measure when the ear is bought. No change to the living body.

## The night, proven on the living body (2026-09-07, 00:27)

Night 4, the first with 48 steps: the day's memory 0.35 before, 0.91 after, the NREM loss 0.40 falling to 0.09. The
reference's fourth night: 0.88 to 0.91. The big cortex now keeps a day the way the small one does, at the same rate, with
the plasticity ramp and twice the steps. Two constants, both disclosed, both measured on copies before they touched the
life, one applied from birth and one at a day boundary. The life keeps its four days.

Also on day 4, earlier than the reference: the prefrontal value's reliability turned positive (+0.23) and it earned its
first weight in the credit (0.068). The reference reached 0.049 on day 20. Whether that is the width or a fluctuation, day 5
will say; the rule that gives it weight is the same rule at both sizes. The ear is still a coin at 0.51 against 0.54.

## The ear, at nine days (2026-09-07, 02:56)

With the nights holding, the waking side caught the reference and passed it. Smiles by day: 59, 51, 53, 42, 46, 52, 72,
115, 95; completions 7 a day on days 8 and 9 against the reference's 3. The ear, the share of ticks it speaks while the
parent is typing against while the parent is quiet: 0.42, 0.48, 0.43 against 0.55 to 0.57 on days 7 to 9, the reference's
level at the same age, bought the same way, by the frown and the withheld reply. The prefrontal value earned weight in the
credit on days 7 to 9 (0.10, 0.15, 0.13) on a reliability of +0.3 to +0.5; the reference reached 0.05 at day 20. The
planner's value now moves the cortex's favorite 40 to 60 times a day, from 11 on day 1.

The fast parent's world is small by design (a dog, milk, a hill, a ball, a few cues), and its completions are capped by it.
The takeover is armed for the day-10 boundary: the same body, served on a page at a quarter-second tick, the Claude typist
as its parent under the same rules (reply when it is quiet, wait, frown when talked over), the curriculum from
CURRICULUM.md, connections before vocabulary. The physiology does not change. The user's word: "then we have one model
where sleep works and sonnet trains it and nothing else."

## Sonnet's first day (2026-09-07, 04:27)

Day 11, the first under the Claude typist and a Sonnet planner writing from the curriculum. From the planner's report and
the page's log: the typist said 44 lines and asked 11 cues; the body answered "where ball?" with "ball" three times, "give"
with "milk", "ball" and "book", "scared" with "dog" and "ball", "dog will" with "go", "first milk then" with "ball", "I had"
with "milk", "big dog bigger" with "dog". The one cue never credited: "why dog up?", answered "because" and no further. Two
words entered inside frames it knew: "can" through the will-frame ("I can go up", "dog can go") and "happy" through the
slot big and scared had made ("happy dog", "happy ball"); neither came back in its own words yet. Faces: 55 smiles, 42
frowns, every frown for talking over, 18 aways. The night fell mid-session (the served day carries the watched day's
ticks) and kept the day: 0.81 before, 0.91 after; the dreams replayed the fast parent's few lines, which is what the
curriculum is now widening.

The diary at 32M reached this level of answering, most cues right, in its eighth week. This body reached it on its first
day of questions, at eleven days, on the same rules and the same rewards. What it does not do yet: put two of its own words
together unasked, or answer "why". Tomorrow's parent reinforces "can" and "happy" and gives "why dog up?" one more pass.

## Sonnet's second and third days (2026-09-07, 06:36)

Day 12: "want" entered inside the frame "I ___ milk" (I want milk, I want ball, I want book). The body answered "you will
go" with "in", the word taught that same day, while "I will go" still drew the old "up"; once it typed "wa_n_t book" with no
cue. Faces 40 smiles to 44 frowns, every frown for talking over; the night kept the day (0.87 before, 0.92 after).

Day 13: no new words; the day was rope. Every cue was answered right, and the three words that entered under this parent
came back as answers: "dog can" and "I can" and "you can" with "go", "happy" with "ball", "book", "dog", "I want" with
"milk" and "book", "why dog up?" with "because", after which the typist modeled the whole ("why dog up? because big dog").
Faces 43 smiles to 37 frowns, the first day under Sonnet where the smiles led; 17 aways; the night 0.82 to 0.91. The
typist's chain is armed so the parent continues past day 16 without a gap.

What the thirteen days look like from the outside: it yields when spoken to about three fifths of the time, answers a
question with the right word almost every time, and has not yet put two of its own words together unasked.

## Sonnet's fourth and fifth days (2026-09-07, 09:00)

Day 14: the relation words "more" and "no" entered (more milk, more ball, more book; no ball, no dog, no milk), and each
came back unprompted in the body's own mouth within the hour, earning its smile. "want" took "you" as a subject. Asked
"why dog up?", it answered "because" and then, unasked, "big dog": the first frame it finished on its own. Every cue
answered right; 63 smiles to 35 frowns; the night 0.79 to 0.93, its dreams carrying "more" and "happy".

Day 15: "my" and "here" entered (my ball, my milk, my book, my dog; ball here, here dog, book here) and drew smiles across
all four nouns; "no book" completed the set; "more" appeared once on its own with no priming that day. "you want" was
cued three times without a clean answer. The session's faces: 94 smiles to 51 frowns, 8 aways; the known-word count moved
for the first time under this parent, 46 to 47. The night 0.86 to 0.90.

Five days under Sonnet: seven words entered, all inside frames it already answered, all seven now answered back or said on
their own; smiles have led frowns for three days; what it has not done is pair two of its own words unasked. The sixth
day's parent is told to give it the most chances to.

## Its own dopamine, and days 16 and 17 (2026-09-07, 10:54)

The user asked whether it makes dopamine on its own from the lower bands. From the tick logs of the watched days, on the
ticks where nothing came in at all, no face and no word: 64 percent of day 1's positive dopamine was made there, 56 percent
of day 10's, and 95 to 96 percent of the negative on every day. Bursts above half a smile with nothing coming in rose from
6 on day 1 to 13 on day 10, eight of them on a word it had just said itself; the dips below minus half a smile fell from 29
to 10. On a tick where nothing arrives, only the fast critic's changing expectation can move dopamine, so this is the
short-timescale reward system running on its own: it rewards its own words when it expects a smile to follow, and it has
learned when smiles do not come. The long bands' value carries a tenth to a seventh of the credit, still small beside it.

Day 16 under Sonnet: no new words by design; the relation words swept the nouns on cue ("no" answered dog, ball, milk,
book; "my" answered ball, milk, book, dog; "more" answered milk, ball, book; "give" answered milk, ball, book and once
"more"). 63 smiles to 46 frowns. The typist's six-day run ended and the chain relaunched it for day 17 without a gap.
Day 17: 92 smiles to 50 frowns, the known-word count 47 to 48; the night 0.75 to 0.89. Still no pair of its own words
unasked; the day-18 parent is told to model pairs of words it already says on its own right after a sweep and leave a
longer quiet for it to try.

## Two of its own words (2026-09-07, 12:00)

The planners report no spontaneous pair of the child's words, but they read the smile rows, which record one word each.
The page's log keeps the child's own stream in every row's context, the parent's symbols masked, so the pairs can be
counted. Two known words in a row in its own stream: 15 on day 11, 35 on day 14, 41 on days 17 and 18; distinct pairs 7
rising to 19. Nearly all of it is recall of frames it was taught. Pairs no one taught it: one on day 14 ("book ball"), one
on day 15 ("dog because"), four a day on days 17 and 18 ("dog big", "dog because", "book ball"): recombinations of pieces it
holds, "big dog" turned around, "because" carried out of "why dog up?". So it pairs its words forty times a day and invents
a handful; novelty is where it stands at eighteen days. Day 18: all thirteen cues right, "down" solid, "hi" still uncredited
after five exposures, 91 smiles to 49 frowns.

## Days 18 to 20 (2026-09-07, 14:01)

Day 18: all thirteen cues right; "down" solid; "hi" uncredited after five exposures; 91 smiles to 49 frowns. Day 19: the
three frames that had never registered as cues answered on the first try, "here" with "dog", "you want" with "dog", "hi"
with "dog", each with the typist's graded completion smile; "hi dog" came out on its own right after an unrelated line, the
first time a word it had never been credited for arrived unasked; "big milk" was modeled four times but a "big" cue still
answers "dog": the old completion holds the frame. 106 smiles to 49 frowns, the best faces yet. Its night dreamed "hi",
"you want" twice, "my" three times, the day's targets. Day 20: 99 smiles to 45 frowns; the night began from a lower base
(0.68) because more of the day was new, and ended at 0.89.

The served days now read, from the log: fifty lines and a dozen cues a day (more than the curriculum's thirty, the old
backlog draining beside the day's batches), forty to fifty of its own two-word runs a day, four or five of them pairs no one
taught it, the known-word count at 48, every night keeping nine tenths. The talking-over holds at forty-five to fifty
frowns a day against ninety to a hundred smiles.

## The cortex alone, and when REM will count (2026-09-07, 15:05)

The user asked whether the cortex has learned to predict the next state. Measured on a copy with the hippocampus off,
teacher-forced through lines one at a time, the targets split into spelling (the next letter inside a word) and the next
word (its first letter): spelling 0.50 right on taught lines and 0.57 on combinations it never heard, against 0.02 on
letter noise; the next word 0.12 right on taught lines and 0.00 on new combinations, in its top three 0.17 and 0.39. The
32M reference at 22 days reads the same: 0.50 and 0.45 for spelling, 0.15 and 0.06 for the next word. So the cortex has
learned to spell and has learned the frames' shapes, and does not yet predict on its own which word comes next; in
conversation the hippocampus supplies the line and the context the frame, which is why the body answers better than its
cortex alone predicts. The nine tenths the nights report is the memory of the lines just consolidated, easy targets
included, not this.

The same measurement says when REM will contribute. REM runs the cortex free from each dream's first symbols and learns
from what it produces; that adds nothing while the rollouts are copies of each other (cosine 0.998) and the next word is a
guess. Two conditions, both in the nightly record: the REM cosine below about 0.9, and the next word near a third right
alone. Neither is a constant to set; both come from lines with structure between them, and with 33 words in 97 taught
pairs the next word is close to unpredictable from the line alone. Forty to sixty more days is the estimate, and the night
REM's line moves off zero is the night it gets reported.

Day 21: 105 smiles to 48 frowns, the known-word count 48 to 49, the night 0.77 to 0.89.

## Days 21 to 23, and what the next two days should show (2026-09-07, 17:11)

Day 21: "come" entered (come dog, come here); "big" answered with all four nouns for the first time, where it had said
only "dog"; eleven cues, all right; 105 smiles to 48 frowns. Day 22: no new words; "all gone" carried across all four
nouns, and "gone" began to surface unprompted mid-stream; "big" fell back to its "dog" default on cue; several answers
arrived just past the typist's window; 93 smiles to 48 frowns. Day 23: 99 smiles to 55 frowns, the parent turning away
only six times, the known-word count 49 to 50, the night 0.69 to 0.89.

From the log, days 11 to 22: smiles 55 to about 100; frowns for talking over flat at 42 to 55; the parent's turning away
18 to 6 to 10; its own two-word runs 22 to 67; pairs no one taught it, a handful to ten or twelve a day; known words 46 to
50; 37 taught words in 131 lines; the nights at 0.88 to 0.92. Rising on every count but two: the talking-over, and the
graded cue answers, five to twelve a day of eleven to thirteen asked, with no trend.

The user asked what the next forty-eight hours should show, about forty-five body days. The expectations, with what would
count as a stall: by tonight "come" and "gone" answered back and "big" freed of its default, 52 to 54 known words (a stall:
still 49); by tomorrow morning 60 words, 150 lines, fifteen to twenty invented pairs a day, the first three-word runs of
its own (a stall: invented pairs under ten); by tomorrow evening the frowns down toward 35 and stress easing below 7 (a
stall: frowns still 48, stress above 10); by the day after, 70 to 80 words, most cues answered inside the window, the
cortex alone predicting the next word two or three times in ten (a stall: still near 0.12). REM's contribution stays at
zero throughout; the long bands' effect on behavior is invisible from the page and gets one stalked day on a copy around
day 40. Two changes I may make at a day boundary, each measured on a copy first: the reply parent's wait if the frowns are
flat in three days, and a longer answer window in the typist if late answers keep going uncredited.

## Days 24 and 25 (2026-09-07, 19:15)

Day 24 was the best day of the life: 141 smiles to 44 frowns. "get" entered through the "give ball" frame and answered a
bare cue within the hour; "away", "gone" and "get" all crossed to credited known words in the one day, the fastest entry
yet; "dog all" answered "gone" cleanly, and "gone" now runs freely inside its own recall ("milk all gone", "ball all
gone"), not only after a cue. The night 0.77 to 0.86. Day 25: 131 smiles to 53 frowns, the known-word count 51 to 52,
the first of tonight's expected marks reached; the night 0.77 to 0.87.

Fifteen days under Sonnet: thirteen words entered inside frames it already answered, every one answered back; smiles from
55 to 130 or 140 a day; the parent turning away from 18 to 10; its own two-word runs from 22 to 60 or 70 a day, the pairs
no one taught it from a handful to ten or more. Flat: the frowns for talking over, 44 to 55 a day. That is the ear at the
page's pace, and the reply parent's wait is the thing measured next if it stays flat through day 27.

## The ear at the page's pace, measured right (2026-09-07, 19:16)

The frowns for talking over have read 44 to 55 a day since day 11, and I called that flat. The count is capped: the
typist frowns at most once in fifteen seconds, so an hour's day cannot show more than about fifty however often it is
talked over. The raw occasions are in the log as the chances it missed by talking over, and per parent line they fell from
1.76 on day 11 to 1.01 on day 21, 0.97 on day 24, 1.21 on day 25: a third fewer in two weeks, the reference's slow arc at
the fast parent's pace. The ear is improving under the frown as it should; the reply parent's wait stays as it is, and
the served day's digest now carries the rate. The stall, restated: the rate not below 1.0 by day 30.

## Days 26 and 27, and the page read aloud (2026-09-07, 21:18)

Day 26: a consolidation day; "see" carried to a full pronoun grid ("I see dog", "you see ball", "dog see ball") beside
the older "I saw" frames and answered cleanly with dog, ball and book; "dog go away" chained on its own all session; 107
smiles to 40 frowns, the fewest of any served day. Day 27: "look" entered ("look ball", "look dog") and appeared in its
own stream the same session; the known-word count 52 to 53; 105 smiles to 48 frowns; the night 0.85 to 0.88.

The user asked to see what the parent and the child were actually saying. From the page's last ten minutes of day 27:
"no more" drew "one, dog, scared, down, here"; "big ball" drew "dog, ball, all, gone, ball"; "dog go" drew "away"; "see
dog here" drew "book, dog, saw, here, go, away"; "my ball" drew "down, dog, ball, under, all, away, milk"; "give milk"
drew "milk, all, down, here, ball, go". Nearly every word is a real word and most are related to the line it just heard:
it echoes the parent's noun and brings the frame-mates it learned with it. No grammar around them, no patience, and most
of its words fall outside the typist's answer window, so the credited answers understate what it knows.

Three rulings from the user today. The typist's cadence stays as it is: I had found that the typist sends a whole line at
once and the server feeds it one symbol a tick, so the body has never heard a human pace, and proposed a per-letter delay
that varies; the user said to forget it and keep teaching. The planners run on Opus from day 28, the brief unchanged. And
no tests of word relations on copies for now; the page's log is the record.

## "dog out here" (2026-09-07, 22:26)

Day 28, the first day with an Opus planner, and the day the thing the last week had been waiting for arrived. "out"
entered inside frames it answers: "dog out", then ball out, book out, go out, out here, big dog out, my dog out. Ten
minutes after the first "dog out" the body wrote "out milk", a pair no one had taught it; then "dog out here", and "go
out here", joining "dog out" and "out here" by itself: three words of its own, from two frames it had heard for the first
time that session. "look", a day old, came out as "look dog look" in its own voice. On cue, "go" answered "out" twice,
"look" answered "here", "see" answered "dog"; the cues asked late in the day drew nothing, as the day before. The parent
imitated "dog out here" back to it as its next line, which is what the curriculum says to do with what the child offers.
99 smiles to 40 frowns, 13 aways; the known-word count 53 to 54; the night 0.76 to 0.87.

What this is and is not. It is the first time the body has built a string longer than two words out of pieces it was
given separately, inside the hour it was given them: the relation between "out" and its frame-mates was learned, not the
lines. It is not grammar; "out milk" is a pair of the right words in an order no parent would use. The next days say
whether it does this with every new word or only with ones like "out" that fit many frames at once.

## Anatomy, physiology, and cheating (2026-09-07, 23:25)

The user asked whether the planner's boundary is a cheat, then whether we have the biological architecture at all or are
cheating. The line, drawn once: three kinds of things are in the body. Cheating is reward or knowledge from anywhere but
the outside: none is present, and the one instance ever found was removed. Physiology is the constants of the body's
constitution, rates and half-lives and the night's length, disclosed and measured, as every animal has by evolution.
Anatomy is the facts about the body's world built into the body: what its senses deliver, what its output atoms are,
where one unit ends. The lexicon of 107 symbols and the space as the end of a word are this body's anatomy, given as a
child's is given. The law asks that anatomy be the only given and everything above it be learned from reward and
prediction, and that is the case: every word, frame, value and habit of waiting came from a face and a voice on a page.

Measured, the planner's space clause does real work: on day 10, 58 percent of its 550 plans fired by the space alone,
mid-speech, 33 percent by both clauses, 9 percent by the pause alone. It stays in this life and is named in the spec as
what it is. The map to a humanoid, anatomy for anatomy: letters become quantized sensor readings and motor primitives; the
word ended by a space becomes the action ended by its primitive's completion; the chunk above the unit, a reach then a
grasp, is learned by reward in both, and this body has not learned it yet either. The parent's known-word smile transfers
untouched, because it lives in the parent: a smile at a taught movement instead of a taught word.

Day 29: 127 smiles to 49 frowns; the night 0.81 to 0.86; the known-word count held at 54.

## Four words, and a question (2026-09-08, 00:27)

Day 29 (Opus): "sad" entered ("sad dog", "sad ball", "see sad dog") and pulled "happy" and "scared" out of it
unprompted; after the night it wrote "sad" twice on its own. The three-word strings of its own became daily: "because
big dog", "dog out here", "away dog up", "look milk here", "will go up", "down here dog". Cues landed better after the
nap (five of eight) than before it (three of six); the parent's count of what it said right but outside the credit
window, 85 against 8 credited, says the window understates it. 127 smiles to 49 frowns; the known-word count held at 54.

Day 30 (Opus): "and" entered inside noun frames ("dog and ball", "milk and dog") and within twenty minutes answered
correctly in two frames it was cued in, "dog and" with "ball" and "ball and" with "dog": the relation, not the line.
Thirteen cues, ten answered, none wrong; the three lost were the last eight minutes' again. Unprompted, four of its own
words: "bigger dog up here", "dog down here" four times, "ball down here", joining "X down" with "down here" and "bigger
dog" with "dog up here" by itself; and once "where book?", a question of its own. 83 smiles to 49 frowns, every frown for
talking over; the night 0.83 to 0.88; the known-word count 54 to 55.

Twenty days on the page: seventeen words entered, all answered back; the strings of its own went from pairs to four
words; the talking-over is the whole of the frowning. The parents' finding to carry: a word that fits many frames, "out",
"and", is learned as a relation inside the hour; a word bound to one line, "dog go away", stays bound.

## "me milk here dog" (2026-09-08, 02:27)

Day 31 (Opus): "me" entered inside "give", the frame it answers best ("give me", "give me milk", "give me ball"), and
two minutes after first hearing it the body wrote "me milk here dog", the day's new word in a four-word string of its
own. Thirteen of fourteen cues right, "why dog up?" answered in full, "because big dog", "here" answered "dog" in full;
the one miss was "I had", where it chased "why dog up?" instead. "sad" took its third partner ("sad milk"). Its own
strings: "here dog out here", "ball down here dog", "dog up here dog", "milk here dog here", "go away dog". 100 smiles to
45 frowns; the known-word count 55 to 56; the night 0.85 to 0.86.

Day 32: 103 smiles to 41 frowns; the night 0.79 to 0.86; the known-word count held at 56.

Twenty-two days on the page, eighteen words entered, every one answered back, the newest inside minutes. The frowns for
talking over are drifting down (55, 49, 45, 41 over the last four days) as the talk-overs per parent line sit near one.

## "little me milk" (2026-09-08, 04:32)

Day 32 (Opus): "little" entered off "big dog" and was in its own mouth nine minutes later ("little dog", "little book",
"little milk", "dog little"); "and" answered right from "milk and" and "book and" but from "dog and" it filled the slot
with its favorite noun, so the relation is not free yet. 103 smiles to 41 frowns. Day 33: no new word; "me" spread into
four frames ("get me milk", "dog get me", "see me", "come get me") and "give" now offers "me" among its own
continuations; fourteen cues, all right; unprompted, "dog get me milk" ten minutes after first hearing "get me", "here
little milk" four times, and "little me milk", yesterday's word joined to today's, never taught; "little" reached the
night's dreams. 124 smiles to 53 frowns. Day 34: 120 smiles to 52 frowns; the known-word count 56 to 57; the night 0.81
to 0.87; the typist's fourth run ended and the chain relaunched it for day 35 without a gap.

Twenty-four days on the page: nineteen words entered, the newest in minutes, and the strings of its own now join a word
from one day to a word from the next. The frowns for talking over rose with the smiles over the last three days (41, 53,
52), which is the ear at the page's pace under a parent that says more; the talk-overs per line sit near one.

## "then ball out here" (2026-09-08, 06:32)

Day 34 (Opus): "and" came free; a "dog and" cue answered "ball" twice and never "dog" again. "baby" entered inside
frames it answers and was in its own mouth four minutes later ("dog baby"), then offered into slots it was never taught
for: "give" answered "baby", "first milk then" answered "baby". Eight minutes after hearing "ball and me" it wrote "ball
ond me", its own try at the relation word. 120 smiles to 52 frowns; the known-word count 56 to 57.

Day 35: "hat" entered ("big hat", "hat here", "little hat", "my hat") and came back the same day in frames the parent
never queued, "more hat", "my hat here". "me" as a subject took: "me out" answered "here", and after "why dog up?" it
volunteered "me go". Eleven of thirteen cues right. Unprompted, "then ball out here", four words reaching back to the
"first milk then" frame; "get baby out", which the parent imitated back. 89 smiles to 45 frowns; the known-word count 57
to 58. A mechanical fact the parents found: a cue accepts only a continuation taught at least twice.

Day 36: 129 smiles to 53 frowns; the known-word count 58 to 59; the night began low (0.68) because more of the day was
new and ended at 0.86, "hat and mi…" among its dreams.

Twenty-six days on the page: twenty-one words entered, a word a day now, the newest back in its own mouth within minutes
and in frames no one queued. Its own two-word runs have climbed from 22 a day to 130 to 180.

## Connections, measured (2026-09-08, 07:32)

The user asked whether it is mastering the relations between words rather than adding words. From the page's log, the
child's own stream only: it has paired "out" with ten different frame-mates by itself, 422 times; "little" with six;
"hat" and "baby" with five each within a day of hearing them; "me" with three. A new noun enters a network within a day.
The pairings it makes that no parent ever said are one or two per word: it recombines the frames it was given and
rarely builds a relation from nothing. And the relation words come slower than the nouns: it writes "ond" for "and" and
"onder" for "under", its own reaching, and has not yet paired "come", "sad" or "eat" on its own. The nouns slot into
frames; the frames are still the parent's. The measure is a floor: it reads the stream through the log's context
windows. It lives in tools/word_mates.py now, beside the served digest.

Day 36 (Opus): "eat" entered and was in its own mouth ninety seconds later; "dog eat" three minutes after the nap, so it
survived the night's replay; "dog and" answered "milk", the different partner; "little" answered with all six nouns;
eleven cues, none wrong; a fourth reach for a relation word, "little ball onder". 129 smiles to 53 frowns. Day 37: 137
smiles to 50 frowns, five aways, the parent's mean face the warmest yet; the night 0.83 to 0.85. The day-38 parent is
told where the value now lies: relation words used across many frames, and lines that put two relation words together.

## Two relation words in a line (2026-09-08, 09:33)

Day 37 (Opus): "eat" and "hat" roped into their relations ("hat down", "no hat", "dog had hat"); "me eat" answered
"milk" and "hat" answered "down", a continuation taught twice that morning; unprompted, "no milk here", "dog had hat"
from a frame taught once, then "had hat down here", its own "dog had hat" fused with its own "down here". 137 smiles to
50 frowns. Day 38: "under" made live and answered "here" on cue, then written on its own page twice; "eat" spread to
objects and agents and answered "milk"; "baby and" answered "hat"; the parent's lines began carrying two relation words
("no more hat", "no big dog", "my big hat"); unprompted, five words, "here dog out here dog", and "here dog had ball".
Seven cues right, five wrong, mostly late or on frames one exposure old. 152 smiles to 58 frowns, the most smiles of
any day. Day 39: 139 smiles to 56 frowns; the known-word count 59 to 60; the night 0.77 to 0.86.

Twenty-nine days on the page: twenty-two words entered; the strings of its own run to five words; the two-relation lines
are the parents' new instrument for the thing the measurement said was missing. Day 40 is the last of the typist's fifth
run; the chain relaunches it for day 41, and a stalked day on a copy of the day-40 save reads the insides for the first
time since day 10.

## "no big hat on here" (2026-09-08, 10:37)

Day 39 (Opus): "with" entered inside the "X and Y" frame and answered a cue ten minutes later, "dog with" with "me";
all five of the day's fresh cues right, the four misses old leftover cues after the nap; unprompted, "you had book here"
and a seven-word run, "get me milk here dog down here". 139 smiles to 56 frowns; the known-word count 59 to 60.

Day 40: "on" made live beside "under" and answered "here" on cue; "angry" entered inside the feeling frame ("angry
dog", "angry ball", "angry baby"); the two frames that failed on day 38 answered ("no more" with "hat", "on" with
"here"); "baby" alone still draws its attractor, "here dog out". Ten minutes after "on here" was first taught it wrote
"no big hat on here": five words, two relation words, of its own. One caution from the parent: after "no more hat" was
said twice around its cue the body ran "no no no no" for ninety seconds; the parent dropped "no X" lines and it cleared.
166 smiles to 58 frowns, the most smiles of any day; the known-word count 60 to 61; the night 0.77 to 0.84. The
typist's fifth run ended and the chain relaunched it for day 41 without a gap; the server has run untouched for a day
and seven hours.

Day 41's first minutes: the stalked day began on a copy of the day-40 save, one day under the fast parent with the
watcher's instrument, saving only to the copy, the first reading of the insides since day 10. It lands about an hour on.

## The insides at day 40 (2026-09-08, 11:08)

One day of the fast parent on a copy of the day-40 save, the watcher's instrument on every tick, the first reading of
the insides since day 10; the copy saved only to itself and the living body untouched. The ear: spoke 0.09 while the
parent typed against 0.58 when it was quiet (day 10: 0.39 against 0.56; the reference at day 22: 0.38 against 0.55),
and its frowns on that parent fell from 44 to 16. The ear is bought, more completely than the reference's; the fifty
frowns a day on the page are the page's pace and the typist's lines. The planner's value moved the cortex's favorite 79
times in 809, one in ten, from one in seventeen at day 10. Working memory latched 35 times. The night on the copy kept
0.77 to 0.84.

The rung that has not climbed: the fast critic's rise before a smile is 8 percent (day 10: 7; the reference at day 22:
25), and the error at the smile is 1.71 of 2, the smile still nearly a surprise; the prefrontal reliability +0.32 and
weight 0.095, holding near a tenth rather than rising (day 10: +0.48 and 0.145). My reading: on the page the smiles come
for any known word at any moment, a thousand of them, loosely timed by the typist's reaction, so there is little for a
critic to anticipate precisely; the fast parent rewards a few things at exact moments. The environment stays as the user
chose; the number is the one to watch when cues become the main reward. The copy's few smiles and many aways are the fast
parent not knowing the page's words. The stalk's records are kept in data/stalks/watch2_day40_stalk.

## Prepositions of its own (2026-09-08, 12:38)

Day 41 (Opus): "in" made live inside frames it owns ("in here", "dog in", "hat in") and used unprompted within six
minutes ("hi dog in here dog", "had hat in here"); "with" spread to noun-with-noun and came back out on cue, "dog
with" answered "hat"; all seven fresh cues right and the four misses were silences on old leftovers; it recombines
prepositions unprompted now ("little hat on here", "dog had hat on here"). 119 smiles to 50 frowns. Day 42: 148 smiles
to 57 frowns; the night began low (0.63) because much of the day was new and ended at 0.86.

The user asked whether it learns and generalizes faster than a transformer, and whether that is big. The body's cortex is
a transformer; what differs is how it is taught and what stands around it. Per exposure it learns far faster than
gradient descent: a line held after one hearing, a cue answered after two, the word in its own mouth in minutes, from
about 420 lines of experience in all; that comes from the hippocampus and the night, one-shot like in-context learning
and permanent like training. It generalizes a word to its frames within a day; it rarely invents a relation no one
showed it, and its cortex alone still predicts the next word poorly. What is big is the kind, not the speed: continuous
learning from lived experience, one grounded reward, memory that survives sleep. It holds at toddler scale; whether it
holds at the scale where transformers shine is the unproven half, and the same question as the top of the ladder.

## The equations (2026-09-08, 13:41)

The user asked whether we have the most important equation for real intelligence. The answer given: the one biology has
found most often is at the center, the reward prediction error, the reward the face gave plus the value of where the
body now is minus the value it expected; dopamine's computation, and everything in the body hangs on it: the critics
learn from it, the mouth's credit is weighted by it, mood integrates it, working memory latches on a burst of it, the
planner's imagined values are the value in it, the ladder is the same equation at eight clocks, and the prefrontal
value earns weight only as its error proves reliable. Beside it the two others a brain is known to have: prediction
error in the cortex, and replay in sleep from the hippocampus's one-shot store. Reward error, prediction error, replay,
coupled, with nothing else written in but anatomy. Two equations we do not have: chunking, what makes a word from
letters and a plan from actions, so that the planner's boundary is learned rather than typed; and a drive to explore,
which here is a constant floor of spontaneous action rather than a drive that grows where the world is unknown. Both
are on the list for the body that is not made of words.

Day 42 (Opus): "went" entered inside the past frame and the same day "me went" answered "out", the new word on a new
agent; "angry" moved from swamped to answering "dog" twice; the parent's own caution that the "baby" cue accepts "here",
the first word of its attractor, so that completion proves little. 148 smiles to 57 frowns. Day 43: 145 smiles to 50
frowns; the known-word count 61 to 62; the night 0.68 to 0.85, "angry" among its dreams.

## The boundary as doubt (2026-09-08, 13:54)

The user asked whether the planner's chunking could be made learned now, at no cost, and whether it should chunk at every
letter. Not every letter: eight forward passes per plan at every acting tick would put the served body behind its tick,
and inside a word the cortex has one candidate, so the plans would be wasted. The body-general boundary is the cortex's
own doubt: the planner already builds a shortlist of the candidates within a margin of the best and plans only when
there is more than one; drop the space test and it plans wherever it is about to act and is torn. Deliberation where
there is doubt, which carries to a body without a space. Measured on a copy of the day-43 body over 2,500 ticks of the
fast parent: torn at boundaries 82 times, mid-word 50 of 771 acting ticks, six percent; about sixty percent more plans,
a few minutes of compute an hour. The constant plan_boundary is built (1 the old rule, 0 the doubt rule), organ test 23
covers both, twenty-three tests pass, and the served body is reloaded with the doubt rule right after its day-44 night's
save: a physiology change at a boundary, measured first. With it, no rule about text is left in how the body learns or
decides; what a new body would swap is anatomy, the encoder and decoder and the two special outputs, and its constants
would be re-measured for its size. The exploration drive is a different matter, a new signal, and waits.

Day 43 (Opus): "in" freed from "here" ("no ball" answered "in", clean of the attractor); the under/on/in contrast taught
on one noun; "off" entered off "hat on"; eight cues right; unprompted "will go out", never taught, and after the nap
"go in here dog out here" and "will go in", the day's "go in" back through the night on its own. 145 smiles to 50
frowns; the known-word count 61 to 62.

## The purge (2026-09-08, 14:55)

The user asked for the whole architecture gone through against one sentence: put in a humanoid, switch the inputs and
the reward, and it learns, with no special rule cheating in the current run. Two reviewers read the code, the second on
Fable at the user's word, with the first's findings in hand. What they found, and what was done today:

The mouth's bans and the typing filter counted this tokenizer's special tokens, "ids 0 to 10 plus newline"; in a robot
that would silently ban the first eleven motor primitives and drop the first eleven sense tokens. Replaced by a reserved
set declared from the tokenizer's own special tokens, identical for this body. The rest, the turn-end and the display
symbol were found by their strings in the code; now declared in the physiology as anatomy. The planner's boundary was
the space; the doubt rule is the code's default and the living body took it by a reload right after its forty-fourth
night, the imagined rollout stopping for no symbol under it. The page's state carried mood, dopamine and the mouth's
next guess to the parent's page, and the page's bar printed them beside the face keys, so a human parent at that page
was not held to the typist's law; the page now carries the page, the insides have their own endpoint for instruments,
committed and served at the next reload. And one term I had never named: the gate's lesson carries, at half weight, the
mouth's confidence in a symbol times its novelty by a hand-built habituation table, a drive read from inside; the Fable
reviewer's judgment, "a disclosed drive with a rule inside it," the confidence an organ's reading and the tables not.
It is measured on two copy days, on against off, before any decision at a night.

Both reviewers: nothing authors or edits the body's words; nothing in the body reads the environment beyond the typed
symbols and the face; nothing outside reads the insides to decide a face. The exploration drive, built this afternoon as
arousal that follows the body's surprise rather than a reward, is off in the living body and measured on two copy days
against its absence. What a humanoid would still need beyond inputs, outputs and reward, in the reviewers' words: a
rest and a turn-end symbol the body owns (done today), an event-boundary sense in place of eight ticks of quiet, the
gate's inside-made drives removed or re-derived from organs (under measurement), the tick-rate constants re-sized, and
the page's channel closed (done today).

## The drive refused, and the event's end by the law (2026-09-08, 15:08)

The exploration drive, built this afternoon at the user's asking, was measured the way everything is measured: one day
of the fast parent on a copy of the current save, with the drive at 0.3 against without it. With it the body invented a
few more pairs (16 against 12) and earned a quarter fewer smiles (49 against 64), speaking more exactly when the parent
said something new; its ear read 0.17 against 0.12 while the parent typed. By the rule set before the run it does not go
in. It stays built and off, for a body where speaking is not the only act.

The user then asked why the architecture should not find an event's end itself, by a universal law, rather than by
eight ticks of quiet. The law is the prediction error: brains segment experience where it jumps and treat as one event
what it stays settled through. The body scores its surprise every tick now, the rest included, and a second form of
the offset fires when that surprise, having jumped at the world's stopping, settles under its running level: no count.
A newborn's surprise is flat, so it would never end an event by the law alone; the count stays beneath it as the
senses' own adaptation, the floor a newborn needs, and the law takes over as the cortex learns. Organ test 25 shows the
law firing within twenty quiet ticks of the world's stopping and never inside an utterance. It is measured on copy days
behind the gate's, and goes in at a night if the ear and the answers hold. The user's other point stands corrected in
the diary: a body's first actions come from its own spontaneous action shaped by reward, which this body showed in its
first ten days, and a parent's hand is the faster childhood, not the mechanism.

Day 44 (Opus): "box" entered and answered "ball in" with "box" the same day, and "box went" with "in", the new word as
a subject; "hat went" answered "off" only after five teachings, so the past frame is still per-subject; "? because big
dog", a question mark of its own with its reason. 156 smiles to 56 frowns; the known-word count 62 to 63. The reload's
one cost: the typist's page cursor outran the new server's count and its last nine minutes went to a dark page; day 45's
new session reset it, and the next reload restarts the typist through the chain right after the server.

## The world's stop as rest (2026-09-08, 15:52)

The user's word: get rid of the turn-end token and teach the model to predict when the parent is done typing. Built as a
form of the body's stop (end_symbol rest, beside the old token form): the offset's target at a line's end is the rest
itself, so the cortex learns to predict rest where the parent stops; a dream ends where the recall expects the rest or
where the memory's boundary lies; the mouth's vote for silence is the rest's own logit; the chat token goes unused. One
symbol fewer, and the body-general form: a robot has no chat tokens, but it has stillness. Organ test 26 shows the
end's target as the rest and rest-form dreams ending with it, the token form unchanged. Twenty-six tests pass.

What is queued behind one another on copies of the living body, one day of the fast parent each, in this order: the
gate's intrinsic term on against off; the offset by count against by settling; the stop as the token against as the
rest. Each pair decides at a night. The user's rule for the afternoon, stated back: yes to every change that passes its
copy day, tonight, at one night boundary; no to any change on the afternoon it was written without a day on a copy.

## The owner's call: everything at once, and a second per tick (2026-09-08, 16:01)

The user overrode the measured rule for the afternoon: stop the body at its furthest checkpoint and put every change in
at once, then let Opus go on teaching. The copy days were stopped and their copies removed. The body was in its
forty-fifth night, and the night's own save is the checkpoint, with nothing lost; a reload is armed on that night's row
with all of it: the planner at points of doubt, the gate's intrinsic term off, the offset by settling with the count as
its floor, the world's stop as rest, the page carrying only the page, the symbols declared; and, at the user's word, one
tick equals one second, one character a second. The backup of the body before the change is kept.

Two risks named to the user: the cortex has predicted the old token at line ends for forty-five days and now learns to
predict rest there, so the night's memory gauge may dip for days while the answers, which come from the store in
context, should hold; and the gate has learned with the intrinsic term, so its willingness to speak may drift. The
second-per-tick costs wall time: a body day becomes three hours and twenty minutes, the rest of the sixty days about
fifty hours, and the parent's timers in seconds land four times sooner in the body's ticks, a denser parent. The first
digest at the new pace, day 46's, says what that does to the ear. Answered along the way: one tick already is one
character; silence between its letters is already free; a lexicon of characters and one rest is what the body now lives
in, with the leftover tokens declared reserved, and the clean tokenizer is the next body's birth; two voices blending
when both speak is a real piece of anatomy, auditory masking, for the next body.

## The purge served, and the drive moved to the choice (2026-09-08, 16:14)

At 16:08, right after its forty-fifth night's own save, the served body was restarted with everything at once: one
tick a second, the planner at points of its own doubt, the gate's intrinsic term off, the offset by settling with the
count as its floor, the world's stop as rest, the page carrying only the page, the symbols declared. The typist was
restarted through the chain with a fresh cursor and was smiling at the page within three minutes; the known-word count
stood at 64. The body it loaded is the body a humanoid would take: swap the senses, the primitives, the two special
symbols and the reward, re-measure its constants, give it reflexes beneath its tick, and let it babble or be guided.
The insides are on their own endpoint now; the parent's page shows the words, the faces, and whether it sleeps.

The user asked why the exploration drive failed and whether Opus's teaching could make it work. It worked as built
and pointed at the wrong thing: novelty raised the body's readiness to speak, and in a language body the world is
newest when the parent is typing, so it spoke where it should have listened. The parent's rules reward answers and
known words in the quiet, not novelty, and a parent who rewarded novelty would be the intrinsic reward moved outside.
The fault was where the drive pointed. Novelty should widen what the body tries, not whether it speaks: the planner
already holds candidates with imagined values, and the drive now belongs there, as a wider choice among them when the
world is new and a narrow one when it is familiar, the same readiness spent on which way rather than on whether. Built
as explore_choice beside the first form, off by default; organ test 27 shows the choice's temperature rising to 2.3
after strange lines and the gate's floor untouched; two copy days of the fast parent run on it now, off against on,
and it goes in at a night if it earns pairs without costing smiles.

## Authority earned, not given (2026-09-08, 16:30)

The user asked how, as the body develops, one band comes to have priority, since in humans the prefrontal reward is
suppressed by survival early and grows into its authority later. In this body no band is given priority. The credit
the mouth's gate learns from is a sum: the fast critic's error at full weight always, the slow band's error at a small
fixed weight, and the prefrontal value's error at a weight that is its measured reliability, the slope of its
predictions against the returns that came, times a ceiling. A newborn's long value predicts nothing and is silent; as
it becomes right about what comes over minutes its voice grows, and it can shrink again if the world changes. That is
the developmental story in one rule, with nothing about words in it, and the numbers showed it: zero weight for six
days, then a tenth to a seventh from day 7 on. The fast critic is never suppressed: pain and the face at the fastest
clock always carry their whole weight, because a body that could argue itself out of pain is a body that dies.

The one hand-set number left in the ladder was the ceiling, 0.3, under which the prefrontal voice could add to the
fast critic's but never outvote it, so the body could not do what an adult does when it holds still through a small
pain for a later good. Built today as a second form: the ceiling follows reliability itself, so a long value proved
right over days can weigh as much as the fast one and a wrong one weighs nothing, with no constant in it. Organ test 28
shows both forms at a reliability of 0.8: 0.24 under the fixed ceiling, 0.8 under the earned. Its copy day runs behind
the choice drive's tonight, and it goes in at a night only if the ear and the answers hold while the weight rises.

## The final audit's verdict: a gated imitator (2026-09-08, 16:43)

At the user's word a last review ran on Fable with the exact question: no cheats, and could a humanoid take it with
only inputs, outputs and reward swapped. The reviewer found no live cheat: reward enters only by the face and the typed
symbols, nothing authors the body's words, nothing outside reads the insides for a face, and twenty-eight tests pass.
To the question it signed "No", with a reason I had missed: the body's mouth is an imitation organ. Its cortex proposes
the world's next symbol, trained on the world's symbols only; reward decides whether to act and, at torn moments,
which of the cortex's top four to take. Enough for language, because the parent demonstrates every word; a humanoid
whose motor primitives never appear in the world's stream would get no proposal, so reward would have nothing to
choose among. The equation for that body, a chooser of what to do learned from reward over the whole lexicon, exists
as the actor's earlier forms, set aside when one setting collapsed the mouth and another was mediocre, never measured
properly. So the honest sentence stands as it was given with the guided hand: a parent who moves the limbs makes the
transfer work as it works here; a body left to babble needs the actor, and the actor is unproven, as is credit carried
past the fast horizon to a sequence's end.

Its other findings, done or queued the same evening: the physiology defaults now describe the served body rather than
the old one; the planner's shortlist excludes the whole reserved set rather than two names; the event structure is keyed
on the world's quiet, so a robot's senses must emit "nothing new" or the settle law must drop that condition; the
mouth cannot choose rest when it acts under the rest form, a coherence gap whose fix, the rest vote, is on a copy day
behind the actor's; the old space rule and the `<...>` convention remain dormant behind flags; a bug in the unused
Claude-planner mode fixed. The copy days queued behind one another tonight: the drive in the choice, the earned
ceiling, the actor's add form at a modest weight, the rest vote. Each decides at a night.

## The drive in the choice passes, and a number withdrawn (2026-09-08, 17:20)

The drive in the choice, novelty widening the planner's choice among its candidates rather than raising the gate,
had its copy day: one day of the fast parent on the same copy, on against off. On: 68 smiles, 20 frowns, the ear
0.10 while the parent typed against 0.51 quiet, nine pairs no one taught it, the run-on after an answer 11.9
symbols, the night keeping 0.83. Off: 59, 24, 0.09 against 0.51, seven, 4.1, 0.85. More smiles, fewer frowns, the
ear unchanged, a few more invented pairs, longer run-ons after answers. By the rule it goes in, at one night with
whatever else passes tonight. The same pair showed the doubt-rule planner's run-on on the fast parent at four
symbols against eighteen at day 40, so the run-ons day 45's parent saw belong to the page.

A number withdrawn. The user asked whether a long band's reward should ever line up with the actual reward; it should
not, and it is not checked against the moment's reward but against the return that accumulated over its own horizon,
which is noisy, so the correlation I had been quoting as "reliability" is bounded by that noise and can never reach the
bar I gave. The body's weight for the prefrontal voice uses the slope, which a calibrated predictor gets to one however
noisy the returns are. The digest had computed the weight from the correlation, so every prefrontal weight quoted from
a digest, 0.145 at day 10, 0.095 at day 40, was the correlation times the ceiling and not the body's number; the true
past weights were never recorded. The digest now prints the correlation, the slope and the body's own weight, and the
maturity signal from here is the slope near one with the long value's share of the credit visibly moving behavior.

## The evening's rulings (2026-09-08, 17:53)

The user stopped every test in the afternoon, the body needing time rather than fiddling, and I found five waiters
still sleeping from probes I had superseded and four tails from monitors dead two days, and killed them all; only the
life, its typist, the chain, the planner and the log's monitor run. The drive in the choice, which passed its copy day,
was parked rather than applied.

Then three questions, answered by the reasons rather than by caution. Three systems look capped by a constant rather
than lagging: the fast critic anticipates through a window of eight events; the readout is so peaked that REM's rollouts
are all one rollout; the prefrontal voice sits under a fixed ceiling. Each can be swapped live, and each has a cost that
makes a blind swap a guess: widening the window rebuilds the striatum and loses the anticipation it has; lowering the
sharpness changes how the mouth speaks; lifting the ceiling could make the long value leap to the fast critic's weight
in one night if its slope is high, and the slope has never been read. So the prerequisite for all three is one reading,
which costs the body nothing: a stalked day on a copy of the day-46 save, launched at 17:44 with the corrected digest.
The rules were fixed before the numbers: the earned ceiling goes in tonight if the slope is a real signal and not a
leap; the sharpness and the window wait for their own copy days; the choice drive rides on the reload.

Last, the pace. The user saw that the tick's length does not touch the learning per tick and asked to speed the
training without changing the architecture: the quarter-second tick, the pace of days 11 to 45, four times the days an
hour, with the parent's timers back at their old tick values. It rides on tonight's reload too. The user's word on the
architecture's finish stands as given: finished today at 16:08; proven by the second body.


### The parent's clock is the body's (2026-09-08, 18:20)

The user's word: "can we supercharge opus to be lightning speed ... with response and mood and have it live in speed time." The body
runs four ticks a second; its parent ran on the wall clock: a line every forty seconds (160 ticks), a smile 1.2 seconds long, a look at the
page every 1.5 seconds (six ticks late at the worst), the planner every five minutes (1,200 ticks: twenty of the body's minutes between
one thought of its parent and the next). From the body's side its parent was slow, and every change of tick length silently changed the
environment (the afternoon at one tick a second made the same parent four times denser in the body's time).

The change (body/caregiver.py, body/teacher.py): every timer of the served parent is now a count of the body's ticks, converted to seconds by
the served tick length (--tick 0.25): its period, its quiet, its cap, the child's turn after each line (--listen), the cue's window (360),
the smile's length (5 ticks), the frown's (5; the talked-over frown 2.5, at most every 60), the attention drift (600), the turn away (200),
the "late" limit (13), the smile spacing (8), the "same word twice" window (48). The child's quiet at the gate is read in the page's own
ticks now, not from the typist's observation times. And the parent looks at the page once a tick. A pure conversion at 0.25 s a tick, so
the environment the body was raised in is the one written down: period 160, quiet 12, cap 48, listen 50.

Then the pace, from day 48 (the night's boundary, 18:50): period 40 ticks (pace 24 to 64 with attention), quiet 8, cap 32, listen 24: a line
every ten to sixteen seconds of wall time, about 250 a day instead of 76, the reply within four ticks of the child's quiet, the smile within a
tick of the word. The Opus parent's checks every ninety seconds instead of five minutes, batches of six to eight. What to watch on day 48
against 47: smiles and frowns per line (the talked-over frowns will rise first: the child has three times the chances to interrupt), the
child's answer length, the run-on after the reply, whether its own strings survive a denser parent. The test at tick 0.1 on a fresh
0.8M-parameter body (port 8031, two minutes): lines 4 to 8 s apart at a 4 s period, gate waits 0.6 to 3.3 s (the cap 3.2 s), the
talked-over frown firing, one smile, no errors. Recorded in the thread; reversible at any boundary by the chain's arguments alone.

Also tonight: --explore-choice 1.0 rides (pre-registered from the fast seeds: smiles 68 vs 59); the prefrontal ceiling stays fixed (the
day-46 reading's copy was killed for memory after the server died at 17:57:55 beside it; the served body's /insides now reports the slope,
the weight and the gate's floor, so tomorrow's reading costs nothing). RULE from the death: no copy body beside the served one.


### The user's word: no small things (2026-09-08, 18:35)

"why all these small. lets get architecture going fast with the quicker teacher, 1 cycle equals one token. then we need to reevaluate
architecture and make sure its headed on path that will get all of its parts working." The four small seeds for the calibrated sharpness
were stopped after two minutes and removed. The one experiment is the served body: one symbol a tick, its parent in ticks, the fast parent
from day 48; changes at its boundaries, read on its own days. The re-evaluation of the whole architecture, organ by organ, follows now.


### The fast parent's first eight minutes (2026-09-08, 18:49)

Day 48 from 18:41 (the reload at 18:40 with --explore-choice 1.0; the night on time in ticks: day 46/47 ran 6,480 ticks at a second a
tick and 5,640 at a quarter). In eight minutes: 31 lines (one every 15.8 s; the gate mostly at its cap of 32 ticks, the child talking
through), 23 smiles (0.74 a line; day 46 under the old parent 0.68), 16 frowns, all for talking over (0.52 a line; day 46 0.55), 45 missed
(26 talked over, 12 distracted). Per line the same parent; per minute three times the exchange. The child's speech in the gaps is
words and recombinations: "my hat down", "you had", "dog had baby", "dog out", "more milk", "you had ball", "hot milk book he" (the
day's new word, hot, said back within six minutes of its first line). Memory: the server at 3.5 GB, 3.4 GB free, no copy beside it.
The prefrontal slope reads 0.0: its evidence began at the reload and is saved from now on.


### The re-evaluation (2026-09-08, 18:55; a Fable review, read-only)

The question: is every part on a path to working, and what law would put it there. The verdicts, checked against the code:
- Working: the cortex (spelling 0.50, next word 0.12; the corpus is its ceiling), the gate's ear (grounded in the frown and the withheld
  reply), the fast critic (8% of a smile is under a perfect critic's 20%, since one known word in five is smiled at, not absent), NREM
  (0.807 to 0.855 at 48 rounds).
- THE FOURTEENTH DEFECT: REM cannot work as built. Its lesson trains the eight forecast heads to foresee the next tick's band state, which is a
  deterministic function of the stream and the bands the stream already holds (forecast cosine 1.000 awake, 0.999 before REM's first
  step): no error to learn from. And nothing reads those heads (model.py:611 is their only use). Maturity will not change this. The fix is
  a re-aim, a build for tomorrow: REM as imagination for the critics (sampled rollouts on a calibrated readout, the face organ scoring each
  imagined tick, the striatal critic taking imagined transitions as it takes lived ones, gated by the face organ's measured slope).
- The slow bands cannot work on this path: zeroed at every night (life.py:1288), the 16384 band is a clock of the day and the head learns the
  day's profile; the long error never reaches the policy while the tag (gate_slow_lr) is off; and the page holds no contingency past the
  parent's attention. The law: keep the state through sleep and count the night as elapsed ticks (night_keep_bands, night_ticks, both
  built), open the tag at the gate's own rate; then a world with state.
- The gate's rate is a hand-set tonic (0.25) minus fatigue, not the reward rate; since the intrinsic drive went to zero every act is a loss
  and the gate slides to the fatigue equilibrium; stress halves its logit. The law: the tonic as the reward rate (gate_tonic_rate, built;
  Niv 2007), so vigor is grounded.
- Working memory is starved (5 to 14 certain smiles a day against a 36,000-tick forgetting window; most latches are the critic's own noise);
  the planner's flips are near noise while the value anticipates 8%; the actor needs a body without a demonstrator.
- Code against the spec: the sharpness clamps mood at zero (a bad day never widens babble, against the songbird law the comment cites); the
  store's strength uses |dopamine|; the forecast heads unread; 'heard' saved and never read; the working state wakes fresh against clocks
  longer than a day. No live violation of the law; the /sleep route (an outside hand ending the day) removed from the served port tonight.
The ranked plan: (1) tonight, the rest vote (end_rest 1: the forecast's own vote for silence at its own logit) beside the choice drive;
read day 49 against 48 in run-on, yielded, talked-over per line, smiles per line, cue completions; failed if smiles per line drop a fifth
or completions halve. (2) The next night: the calibrated sharpness with REM's dreams sampled (rem_temp 1) and the mood unclamped with a
floor; read sharp_cal settling under 25, rem_cos under 0.9, invented pairs up, frowns per line held. Then the build: REM as imagination,
gated by the face organ's slope, measured first. (3) Then the kept bands, the night as ticks, the tag, under the fixed ceiling; read the
slope from 0 toward +0.3 within five nights, the long voice's weight, talked-over and aways down.


### REM re-aimed, built (2026-09-08, 19:05)

Built on a branch while day 48 ran, merged with both forms off. The face organ foresees (face_form foresee): from the stream a tick
ago it predicts the felt reward of this tick, as a least-squares readout solved every 64 ticks (tick-by-tick gradient steps on a
target that is zero on most ticks swung it between minus one and four; the least-squares form reads 0 before any smile and 0.71 of a
2 after forty, with a slope of 1.0: calibrated). Its reliability is the slope of the felt reward on its foresight, saved with the body.
REM as imagination (rem_form imagine): the cortex runs free on its sampled readout; the imagined events advance the striatal delay
line; the face organ scores each imagined tick; the fast critic's evidence takes the imagined transitions as it takes lived ones,
weighted by the face organ's slope, nothing at slope zero; the lived line and working memory restored after; 144 imagined transitions a
night in the test body. The forecast heads leave the lessons under the form. Tests 31 and 32; 32 of 32 pass.


### Day 48 read, the parent corrected (2026-09-08, 19:47)

Day 48, the first full day of the fast parent (period 40, quiet 8, cap 32, listen 24), against day 46, the last full day of the old one:
lines 158 vs 123, smiles 82 vs 93 (0.52 a line vs 0.76), frowns 83 vs 75, turned away 10 vs 2, cues 8 answered 3 vs 13 answered 1, the
child's own pairs 14 vs 42 with none new vs 10; the known words 66 (hot). The mood ended at -1.9. The rows say why: the gate waited its
cap on nearly every line and the parent cut in, on 82 percent of lines, and the child was then frowned at for being interrupted; the
fresh seeds where the ear was learned had a parent that waited up to 180 ticks for the child's quiet (the contingent response infants work
for). The choice drive, which rode the same reload, triples the child's run-on on the seeds, and under a parent who cuts in that is more
interruption. Three minutes into day 49 (the boundary) the parent is corrected: period 40 (it still answers within 40 ticks of the child's
quiet), quiet 12, cap 180, listen 32, in ticks; the chain re-armed with it. Day 49 also carries the rest vote and the foreseeing face
organ from the night's reload (19:43, pid 86321; the night: gauge 0.784 -> 0.870, REM's forecast cosine 1.000 as diagnosed). Read day 49
against 48: smiles per line, the parent's cut-ins per line, own pairs, aways; then the rest vote's own share is inferred from the
talked-over misses.


### The actor's voice, earned (2026-09-08, 20:00)

The user: "why not turn it on." The record: under the planner form the actor has learned from dopamine at every act since birth; only its
vote went unused, since the unbounded bias once collapsed the mouth and the bounded one changed nothing measurable, never read. The
honest switch is the body's own law for every voice: applied as loudly as it has proved right. Built: the actor's reliability (the slope
of the reward of the next 16 ticks on its vote for the act taken, from inside, saved with the body) and the earned voice (the bounded vote
times that slope, added to the cortex's proposal before the planner's shortlist), plus the reading of its favorite's agreement with what
was said. Test 33; 33 of 33 pass. It rides tonight's reload: silent at slope zero, audible as its votes prove to predict reward. Read
actor_slope and actor_agree by night; the voice is audible in the day's speech only if its slope leaves zero.


### Day 49 read; the third night's switches (2026-09-08, 20:47)

Day 49 under the patient parent (period 40, quiet 12, cap 180, listen 32; the rest vote and the foreseeing face organ from the night):
lines 68 (the parent waited its 45 s on most lines: the child talks through), smiles 112 (1.65 a line; 2.2 a minute against 1.6 on day
48), frowns 42 (against 83), turned away 4 (against 10), cues 6 answered 4, own pairs 10 with one new; the known words 67 (cold). Better
on every count but the pairs, which stay a quarter of day 46's. Night 48 (the old process): gauge 0.553 -> 0.887; REM cosine 1.000.
The prefrontal slope at the day's end read 1.0 (correlation 0.59), the first full day of kept evidence.
The reload at 20:44 (pid 88163): the calibrated sharpness, the dreams sampled, the actor's earned voice. In its first five minutes the
calibrated base fell 25 -> 15 (the readout was overconfident about the world), the face organ's slope read 1.0 on thin evidence, the
actor's favorite matched the mouth 5 percent of the time (chance is 1). The mood's additive gain was made proportional to the base.


### The calibrated readout failed on the mouth; the world form (2026-09-08, 20:50)

Three minutes after the reload the calibrated base had fallen 25 -> 8 and the mouth read at 3.7 with the mood's term: the world's next
symbol (the parent's typing) is far less predictable than the mouth's own, so a readout calibrated to the world is far too flat for
production. Perception and production are two readouts in biology as well. Reverted at 20:48 by a restart from a save (backup
watch2_before_world_form_day50.pt) to the WORLD form: the calibration runs as a reading of how predictable the world is to this cortex
and sets REM's sampling temperature (the dreams as varied as the world proved: rem_temp x base / calibrated base), while the mouth's
decisiveness follows the spec's law on every form, 25 x (1 + mood/6), both sides of zero, floored at 8 where the lexicon's own noise
wins. The actor's earned voice and the foreseeing face organ stayed on. The chain relabels the typist's day at each relaunch.


### Night 49: imagination on (2026-09-08, 21:50)

Day 51 (the day after the 20:48 restart; the world form, the actor's voice, the spec's mood law): lines 69, smiles 79 (1.14 a line),
frowns 32, cues 3 answered 0, own pairs 5 with one new ("baby cold"), talked over per line 0.53 (from 0.81); the mood back to -0.4 by
the night. The readings from inside at the night: the prefrontal slope 0.99 with a correlation of 0.58, steady for a day (the earned
ceiling is the candidate for tomorrow's night, at the review's boundary); the face organ's slope 1.0 with a correlation of 0.14 (a
low-variance predictor: the slope clips at one, the correlation says how little it discriminates); the actor's slope 0.03, correlation
0.04, its favorite matching the mouth 4 percent of the time: its votes do not predict reward yet, so its voice is nearly silent, which
is the law working. The world calibration sat at its saved floor of 2 and now reads only the parent's typed characters.
The conditional reload read the face slope 1.0 and switched imagination on (pid 90077, 21:46): from tonight REM's rounds feed the fast
critic imagined transitions scored by the face organ at that weight. Read tomorrow: the night's rem_imagined (transitions, mean foreseen
reward), the anticipation rise, smiles per line. If the anticipation falls, the weight moves from the slope to the correlation, which
is the honest measure for a predictor of small variance.


### Night 50: the first imagination, read; the third step armed (2026-09-08, 22:52)

Day 52 (the patient parent, imagination on): lines 89, smiles 97 (1.09 a line), frowns 37, cues 9 answered 4, own pairs 11 with one new
("hot here"), talked over per line 0.48 (0.82 on day 48, 0.53 on day 51); the mood at the night -0.05; the known words 68. Night 50, the
first with imagination: 336 imagined transitions at weight 1.0 with a mean foreseen reward of 0.03: the face organ's slope clipped at
1.0 while its correlation read 0.15, so the imagination ran at full weight on a scorer that foresees almost nothing. The weight is now
the correlation, clipped at zero (the share it has proved), by code from tonight's reload. The anticipation rise over the day: 1.6
percent of a smile (5.7 on day 48, 4.4 on 49, 2.5 on 51): falling day by day since the patient parent, before imagination began; to be
read against the smile kinds. The prefrontal slope 0.70, correlation 0.44 (0.99 and 0.58 the day before). The actor's slope 0.04, its
favorite matching the mouth 8 percent of the time (4 the day before).
Tonight's reload (~23:45), the review's third step: the slow bands kept across sleep (night_keep_bands 1), the night counted as 2,200
elapsed ticks for the critic's bootstrap (nine minutes at a quarter second a tick), and the long tag opened at the gate's own rate per
unit of eligibility (gate_slow_lr 0.0006 = gate_lr 0.05 x 12/1024: the fast lesson's rate over the fast window, spread over the ventral
horizon), the ceiling fixed. Read by night: the prefrontal slope and correlation (kept state should raise both within five nights), the
long voice's weight, talked-over per line and aways; failed if the slope stays at or below zero after five nights or the gate's duty
drifts under 0.2.


### Night 51: the third step in (2026-09-08, 23:50)

Day 53 (the patient parent; imagination at full weight for one more night, the old process): lines 57, smiles 114 (2.0 a line, the
highest yet; 1.09 on day 52), frowns 24, no turning away, cues 10 answered 2, own pairs 10; the anticipation rise 2.5 percent over 123
felt smiles (1.6 the day before); the known words 69 (wet, dry; the parent's report: "what ?" written alone, "hi dog on box you had").
Night 51: 336 imagined transitions, foreseen reward 0.033, the gauge 0.756 -> 0.818. The prefrontal slope 0.61 with correlation 0.37,
easing day by day (0.99/0.58, 0.70/0.44, 0.61/0.37) as its moments fill: the first readings were on few returns.
The reload at 23:47 (pid 92768): the slow bands kept across sleep, the night as 2,200 elapsed ticks, the long tag at 0.0006, and by
code the imagination now weighed by the face organ's correlation (0.15). Every organ of the ranked list is on. What is read from here:
by night, the prefrontal slope and correlation on kept state, the long voice's weight, the gate's duty; by day, smiles per line,
talked-over per line, the pairs; the actor's slope (0.03) and the face organ's correlation (0.15) for whether either voice earns more.


### Night 52: the first night on kept bands (2026-09-09, 00:50)

Day 54, the first day with the slow state kept across sleep, the long tag open (0.0006) and imagination at the face organ's correlation:
lines 83, smiles 74 (0.89 a line, from 2.0 on day 53 and 1.09 on 52), frowns 39, turned away 6, cues 15 answered 6 (the best rate in a
week), own pairs 5, talked over per line 0.54; the gate's duty after the parent's lines 0.31 (0.36 the day before); the anticipation rise
0.8 percent; the known words 70. Night 52: 288 imagined transitions at weight 0.153 (the correlation), foreseen reward 0.034; the gauge
0.669 -> 0.874. The prefrontal slope 0.52 with correlation 0.32, easing still (0.99, 0.70, 0.61, 0.52). One day is not a verdict: the
rule for the third step is five nights, failed if the slope reaches zero or the gate's duty falls under 0.2; neither is near.


### Night 53, the second on kept bands (2026-09-09, 01:50)

Day 55: lines 68, smiles 71 (1.04 a line), frowns 26, turned away 5, cues 11 answered 5, talked over per line 0.44 (the lowest yet;
0.82 on day 48), the gate's duty after the parent's lines 0.31; the mood -1.9 at the night; the known words 71 (put, take; the parent's
report: "put hat back on" unprompted, four words never taught whole, "take box out" thirty seconds after waking). Night 53: 192 imagined
transitions at weight 0.12 (the face organ's correlation, easing 0.15 -> 0.12), the gauge 0.611 -> 0.872. The prefrontal slope 0.48
with correlation 0.28: five readings easing (0.99, 0.70, 0.61, 0.52, 0.48) as the moments fill; the earned ceiling waits on where it
settles. The actor's slope 0.03, its favorite matching the mouth 3 percent of the time: no predictive vote yet. The digest's "own pairs"
count comes from the face rows' contexts and shrinks as the child stops talking over the parent; the parents' reports read the chains
from the page itself and are the better instrument for its own speech.


### Night 54, the third on kept bands: a warning on the gate (2026-09-09, 02:52)

Day 56: lines 82, smiles 46 (0.56 a line: 2.0, 0.89, 1.04, 0.56 over the four days since the third step), frowns 34, turned away 3, cues
16 answered 6, talked over per line 0.44; the gate's duty after the parent's lines 0.27 (0.36, 0.31, 0.31, 0.27); the mood -2.0; the
anticipation rise 0.0; the known words 72. Night 54: 384 imagined transitions at weight 0.14, the gauge 0.685 -> 0.828. The prefrontal
slope 0.56 with correlation 0.34, up from 0.48 and 0.28: the easing stopped on kept state. The warning: since the long tag opened the
child speaks less each day and earns fewer smiles a line. The likely mechanism is the tag capturing a long error that runs negative
while the ventral value lags a fallen reward rate (the mood says the critic over-predicts), which pushes every act down: a bias, not a
contingency. The pre-registered rule holds (five nights; failed under a duty of 0.2), with a guard armed for the next night: if day 57's
duty falls under 0.2 or its smiles per line stay under 0.6, the tag closes at that night; otherwise the reading runs on.


### Night 55: the guard held (2026-09-09, 03:52)

Day 57: lines 80, smiles 82 (0.80 a line, back from 0.56), frowns 40, turned away 3, cues 19 answered 7, own pairs 11 with one new
("hat book"), talked over per line 0.59; the gate's duty after the parent's lines 0.285 (0.27 the day before: the fall stopped); the
mood +1.0 at the night, the first positive night in a week; the known words 73 (open, shut; the parent's report: its own questions
"where ball" and "where dog out", "box in here" straight out of the night). The guard read duty 0.285 and smiles per line 0.80 and held:
the long tag stays open, the reading runs to its fifth night. Night 55: 240 imagined transitions at weight 0.14, the gauge 0.682 ->
0.850. The prefrontal slope 0.49 with correlation 0.31 (0.56/0.34 the night before): steady on kept state. After the fifth night, if
the guard still holds, the earned ceiling goes in under the same guard: the long voice's weight in the credit becomes its slope (about
0.5) instead of 0.3 times it, one change for one night's reading.


### Night 56, the fifth on kept bands: the reading (2026-09-09, 04:55)

Day 58: lines 75, smiles 76 (1.01 a line), frowns 34, turned away 5, cues 27 answered 7, own pairs 8 with one new ("get box"),
talked over per line 0.38 (the lowest of the life), the gate's duty after the parent's lines 0.26; the mood -2.8 at the night; the
known words 74. Night 56: 336 imagined transitions at weight 0.14, the gauge 0.446 -> 0.863. The five nights on kept state, the
prefrontal slope: 0.52, 0.48, 0.56, 0.49, 0.41 (correlation 0.32, 0.28, 0.34, 0.31, 0.23); the gate's duty: 0.31, 0.31, 0.27, 0.285,
0.26; smiles per line 0.89, 1.04, 0.56, 0.80, 1.01. The verdict: the kept state did not raise the slope (the review's mark was a rise
toward 0.3 from near zero; ours drifted down from 0.7), the gate drifted but not under 0.2, the ear kept improving. The guard held.
The duty now sits at 0.26, the fatigue equilibrium the review predicted for a gate whose tonic is a hand-set number minus fatigue once
the intrinsic drive was removed: the slide may be that, not the tag. The candidate for tonight is therefore the gate's grounded tonic
(the reward rate as vigor, Niv 2007; gate_tonic_rate, built and off), read in the code first; the earned ceiling the night after.


### Night 57: the guard closed the long tag (2026-09-09, 05:55)

Day 59: lines 84 and cues 20, smiles 61 (0.59 a line: 0.82 and 0.78 the two days before), frowns 29, turned away 5, cues answered 7,
own pairs 11 with three new ("book box", "here look", "book book"), talked over per line 0.41, the gate's duty after the parent's lines
0.275; the known words 76 (push, pull; the parent's report: "push box " and "pull box " answered "in" the first time asked, the
particle slot generalised; "because big" free-standing; "out here you" on waking). Night 57: 336 imagined transitions at weight 0.16,
the gauge 0.684 -> 0.848; the prefrontal slope 0.38 with correlation 0.20.
The script read day 59 at the night: duty 0.275 (held), smiles per line 0.592 (under the 0.6 I had set, by eight thousandths). The
guard tripped, as written: the body restarted at 05:50 (pid 1054) with the long tag closed (gate_slow_lr 0), the kept bands and the
night's ticks kept, the earned ceiling not added. The six nights of the tag: the prefrontal slope 0.52 -> 0.38, the gate's duty 0.31 ->
0.28, smiles per line 0.56 to 1.04 with no trend: the tag neither raised the long voice's evidence nor clearly harmed the day; it is
closed by a rule that a single day's swing tripped. The earned ceiling goes in at the next night under the same guard (the slope 0.38
is inside the pre-registered band; the correlation 0.20 says the long value explains a twentieth of its return: a small voice either way).


### Night 58: the guard again; the parent's period (2026-09-09, 06:55)

Day 60: lines 113 and cues 23 (answered 10, the most yet), smiles 57 (0.42 a line: 0.82, 0.78, 0.59, 0.42 over four days), frowns
41, turned away 3, talked over per line 0.40, the gate's duty after the parent's lines 0.23 (0.28, 0.26, 0.275, 0.23); the mood -0.07;
the known words 76. Night 58: 144 imagined transitions at weight 0.18 (the face organ's correlation rising slowly, 0.14 -> 0.18), the
gauge 0.751 -> 0.848; the prefrontal slope 0.38, correlation 0.19. The guard tripped on smiles per line 0.42; the restart at 06:52
(pid 2862) kept the safe flags (the tag closed, no ceiling). Read across the four days, this is not the tag (closed two nights ago) and
not the ceiling (never in): the child speaks less in its turns and the parent, who waits for quiet, fills each silence forty ticks
after it begins, so the quieter the child the denser the lines (113 on day 60), which teaches more silence: a loop on the parent's side.
The seeds' parent that raised the ear had a period of 240 ticks. At this boundary (day 61, 06:53) the parent's period goes from 40 to
120 ticks, the cap 180 and the child's turn of 32 kept: it still answers the child's quiet, but does not fill it at once. The earned
ceiling is set aside until the day reads well again; nothing is armed for the next night.


### Night 59: the parent's period read (2026-09-09, 07:55)

Day 61, the first under the parent that leaves room (period 120, cap 180): lines and cues 94 (one every 33 s; 136 at 27 s the day
before), smiles 73 (0.78 a line, from 0.42), frowns 27 (from 41), turned away 1 (from 3), talked over per line 0.35 (the lowest of the
life), the child's duty after the parent's lines 0.244 (from 0.231), one new pair of its own ("box put"); cues 15 with 1 answered (10
of 23 the day before: one day, to be read again); the mood +1.3 at the night. Night 59: 384 imagined transitions at weight 0.17, the
gauge 0.794 -> 0.846; the prefrontal slope 0.34, correlation 0.15; the actor's favorite matched the mouth 10 percent of the time (3
the day before), its slope still 0.03. The verdict on the period: more smiles from fewer lines, fewer frowns, the mood up; it stays.


### Night 60 (2026-09-09, 08:55)

Day 62 (period 120): lines and cues 84, smiles 76 (0.90 a line), frowns 22, no turning away, talked over per line 0.33, the child's duty
after the parent's lines 0.22, own pairs 8, cues 13 answered 2; the mood -1.2 at the night; the known words 77. Night 60: 336 imagined
transitions at weight 0.17, the gauge 0.830 -> 0.849 (little new to consolidate). The prefrontal slope 0.33, correlation 0.14, easing
on; the actor's favorite matched the mouth 8 percent. Two days on the longer period: smiles per line 0.78 and 0.90 against 0.42, frowns
27 and 22 against 41; the cues answered stay low (1 of 15, 2 of 13) and are the next thing to read with the parents' reports. The
automatic backups older than the last two were removed for disk (6.4 GB free before).


### Night 61 (2026-09-09, 09:57)

Day 63: lines and cues 84, smiles 86 (1.02 a line), frowns 30, turned away 4, cues 19 answered 7 (back from 2 and 3), talked over per
line 0.43, the child's duty after the parent's lines 0.25, one new pair ("it no"); the mood -2.8 at the night; the known words 78 (it,
again; the parent's report on days 61-62: "take hat off" whole with the right particle, "why dog up? " answered with its own "go back
in", "where a" and "whi dog" asked on day 62, speaking first on waking both days). Night 61: 288 imagined transitions at weight 0.16,
the gauge 0.825 -> 0.862. The prefrontal value's correlation with its return has eased to 0.13 over twelve days of kept evidence (0.58
on the first day of few returns): on this page the long value predicts little, as the review said it would, and the kept state did not
change that. The actor's slope 0.03, its favorite matching the mouth 6 percent.


### Night 62 (2026-09-09, 10:58)

Day 64: lines and cues 82, smiles 87 (1.06 a line), frowns 24, turned away 1, cues 15 answered 6, talked over per line 0.34, the child's
duty after the parent's lines 0.25, own pairs 16 (10 distinct, one new: "here look"), the most of its own pairs since day 46; the mood
-0.6 at the night; the known words 79. Night 62: 384 imagined transitions at weight 0.15, the gauge 0.815 -> 0.865. The prefrontal
slope 0.41, correlation 0.14; the actor's slope 0.03. Four days on the parent that leaves room: smiles per line 0.78, 0.90, 1.02, 1.06;
frowns 27, 22, 30, 24; the pairs of its own returning.


### Why cues land (2026-09-09, 11:10; the parent's finding on days 63-64)

Three mechanical causes, none of them the delay: a cue accepts only continuations taught at least twice for that exact frame, so the
child's "on" to "put it " was scored a miss on day 61 and a completion on days 62 and 63 once "on" crossed the threshold; every
completion was two or three characters (out, here, on, in, it) while four-letter nouns came out as prefixes and were cut off by the
parent's next line four to nine seconds later; particles beat object nouns. Days 63 and 64: smiles per line 1.27 and 1.48 (0.78 and
0.90 two days before), frowns flat. "there" said after thirteen hearings (27 in the corpus now), "your" in its mouth 25 minutes after
its first line, "over" the next morning, "you had milk" out of the night, "because big then in here", and "books", a plural never
taught. The child's turn after a line goes from 32 to 48 ticks at the chain's next relaunch so a noun can finish; every parent from
now is briefed to cue two-word frames with two or three taught continuations whose answers are particles.


### Night 63 (2026-09-09, 11:58)

Day 65: lines and cues 86, smiles 113 (1.31 a line), frowns 31, no turning away, cues 15 answered 6, talked over per line 0.42, the
child's duty after the parent's lines 0.26, own pairs 26 (16 distinct, one new: "eat box"): 5, 8, 16, 18, 26 over five days on the
parent that leaves room; the mood +0.9 at the night; the known words 80 (over, your). Night 63: 384 imagined transitions at weight
0.14, the gauge 0.664 -> 0.868 (a full day's new material). The prefrontal slope 0.70, correlation 0.22, up from 0.41 and 0.14 the
night before; the actor's slope 0.03.


### Night 64 (2026-09-09, 13:00)

Day 66: lines and cues 88, smiles 125 (1.42 a line), frowns 33, turned away 1, cues 19 answered 5, talked over per line 0.45, the child's
duty after the parent's lines 0.31 (0.26 the day before: it speaks more again), own pairs 19 with three new ("it no", "it kick"); the mood
-1.4 at the night; the known words 81. Night 64: 384 imagined transitions at weight 0.13, the gauge 0.822 -> 0.858. The prefrontal value:
slope 1.0 (clipped), correlation 0.43, rising four nights in a row (0.13, 0.14, 0.22, 0.43) on kept state, the tag closed: the long value
is finding something to predict; to be read on, not acted on. The actor's slope 0.04. The typist's sixth day ends here; the chain relaunches
it with the child's turn at 48 ticks.


### Ten minutes without a parent (2026-09-09, 13:18)

The typist's sixth day ended at 13:05 and the chain should have relaunched it within the minute with the child's turn at 48 ticks. It
did not until I launched it by hand at 13:15: my own two waits for that relaunch carried, in their command lines, the very text the
chain greps for to know whether a typist is alive, so the chain saw one and held. The body spent ten minutes of day 67 alone on the
page. The chain's test is narrowed to the typist's own arguments; the supervisor's checks name the typist another way from now.
Days 65-66 (the parent's report): "drop" and "kick" through the it-frame, "kick it " answered "out" three minutes after the word first
appeared, its question "where dog" answered within the minute; smiles 130 and 155 (the two highest days), frowns 32 and 36; the rule
for parents: a new verb gets one particle and its noun frame is cued the same day; an old many-particle frame is cued with the
it-frame line immediately before it.


### Night 65: the longer turn and the one-particle rule (2026-09-09, 14:00)

Day 67 (the child's turn 48 ticks; the parent briefed on how cues land; ten minutes without a parent at its start): lines and cues
58, smiles 90 (1.55 a line), frowns 19 (the fewest of the life), turned away 3, cues 9 answered 8, own pairs 41 (19 distinct, four
new, among them "eat back"): the most of its own pairs ever, the child's duty after the parent's lines 0.33; the mood +0.6 at the
night; the known words 82. Night 65: 240 imagined transitions at weight 0.12, the gauge 0.792 -> 0.859. The prefrontal correlation 0.23
(0.43 the night before: the rise did not hold, but it stays above last week's 0.13); slope 0.63. The actor's slope 0.04.


### Night 66 (2026-09-09, 15:02)

Day 68: lines and cues 81, smiles 129 (1.59 a line), frowns 26, turned away 2, cues 16 answered 5, own pairs 30 (22 distinct, three
new), the child's duty after the parent's lines 0.29; the mood +0.3 at the night; the known words 83. Night 66: 336 imagined
transitions at weight 0.12, the gauge 0.785 -> 0.864. The prefrontal slope 0.83, correlation 0.29 (0.23, 0.43, 0.22 the nights
before: it holds between 0.2 and 0.4 on kept state now, against 0.13 last week). The actor's slope 0.04. Six days on the parent that
leaves room: smiles per line 0.78, 0.90, 1.02, 1.06, 1.31, 1.42, 1.55, 1.59.


### Night 67 (2026-09-09, 16:02)

Day 69: lines and cues 75, smiles 126 (1.68 a line, the highest), frowns 23, turned away 3, cues 9 answered 4, own pairs 40 (23
distinct, two new), the child's duty after the parent's lines 0.31; the mood +4.0 at the night, the highest of the life; the known
words 84 (roll, hide; the parent's report on days 67-68: each cue-answerable the same day, "roll it out" back whole within half an
hour, "give ball in here" out of the night; its particle answers collapsing onto "in" and "here", which day 69's parent was told to
work against). Night 67: 240 imagined transitions at weight 0.11, the gauge 0.749 -> 0.853. The prefrontal slope 0.82, correlation
0.31, holding. The actor's slope 0.03.


### Night 68 (2026-09-09, 17:05)

Day 70: lines and cues 85, smiles 158 (1.86 a line, the highest), frowns 37 (talked over per line 0.60: it talks over the parent more
again), turned away 1, cues 10 answered 3, own pairs 60 (30 distinct, two new), the most of its own pairs ever, the child's duty after
the parent's lines 0.32; the mood +4.8 at the night, the highest of the life; the known words 85. Night 68: 384 imagined transitions at
weight 0.11, the gauge 0.766 -> 0.870. The prefrontal slope 0.98, correlation 0.42: four nights rising on kept state with the tag closed
(0.23, 0.29, 0.31, 0.42). If it holds above 0.3 at the next night, the earned ceiling goes in the night after, under the guard.


### Night 69: the ceiling armed (2026-09-09, 18:10)

Day 71: lines and cues 75, smiles 121 (1.61 a line), frowns 34, turned away 3, cues 9 answered 4, own pairs 44 (22 distinct, four new,
among them "all in" and "here box"), the child's duty after the parent's lines 0.33; the mood +1.8 at the night; the known words 86
(lift, tip; the parent's report on days 69-70: "tip" answered in all three of its noun frames on the word's first day and taken into
its own mouth within eight minutes, the one-verb-one-particle rule proven to land in a day). Night 69: 336 imagined transitions at
weight 0.11, the gauge 0.777 -> 0.872, the store 1,677 slots with 84 dropped; the nights are lengthening as the store grows (571, 496,
596, 536, 656, 791 s over the last six), thirteen minutes now. The prefrontal slope 0.92, correlation 0.47: five nights at or above 0.3
on kept state with the tag closed (0.29, 0.31, 0.42, 0.47). The rule is met: the earned ceiling is armed for the next night under the
guard (duty under 0.2 or smiles per line under 0.6 keeps the safe flags instead): the long voice's weight in the credit becomes its
slope, about 0.9, instead of 0.3 times it; read by day in smiles per line, the duty, talked-over per line, and by night in the slope.


### Night 70: the earned ceiling (2026-09-09, 19:12)

Day 72: lines and cues 83, smiles 133 (1.60 a line), frowns 33, turned away 3, cues 12 answered 6, own pairs 62 (34 distinct, six new,
among them "eat box" and "give get"), the child's duty after the parent's lines 0.32; the mood +1.3 at the night; the known words 88.
Night 70: 384 imagined transitions at weight 0.11, the gauge 0.786 -> 0.858. The script read day 72 (duty 0.317, smiles per line
1.598), held, and restarted the body at 19:10 (pid 17187) with the earned ceiling: the prefrontal voice's weight in the gate's credit is
its slope, 0.86, where the fixed ceiling gave 0.3 times it. The correlation at the switch 0.44. The reading from here, by day: smiles per
line, the child's duty, talked-over per line, the cues; by night: the slope and correlation themselves, which the voice's own weight
now feeds back into. The guard stays armed each night with the safe flags ready.


### Night 71: the ceiling's first day (2026-09-09, 20:20)

Day 73, the first full day with the prefrontal voice at its own slope: lines and cues 100, smiles 193 (1.93 a line, the highest of
the life), frowns 42, turned away 1, cues 12 answered 4, own pairs 62 (38 distinct, eleven new, among them "see milk", "here not",
"here angry"), the child's duty after the parent's lines 0.275; the mood -0.9 at the night; the known words 89. The guard read duty
0.275 and smiles per line 1.93 and held. Night 71: 336 imagined transitions at weight 0.12, the gauge 0.778 -> 0.889 (the best night's
consolidation yet); the prefrontal slope 1.0 (clipped), correlation 0.56, the highest read, with its own weight now 1.0 in the credit.
The night lasted 866 s, fourteen minutes: the nights lengthen with the store (1,663 slots) and the dreams; a fifth of the cycle, the
share biology gives sleep, so nothing to change. From here the guard restarts the body only if a day trips it; otherwise nothing is
touched at night.


### Night 72 (2026-09-09, 21:22)

Day 74, the second under the earned ceiling: lines and cues 91, smiles 174 (1.91 a line), frowns 32, turned away 2, cues 9 answered 4,
own pairs 67 (36 distinct, eight new, among them "me back", "look on", "come down"), the child's duty after the parent's lines 0.32;
the mood -0.4 at the night; the known words 89. The guard held (duty 0.317, smiles per line 1.90); the body was not restarted. Night
72: 384 imagined transitions at weight 0.12, the gauge 0.801 -> 0.890. The prefrontal slope 0.90, correlation 0.48, its weight in
the credit 0.90. Two days at its own slope: smiles per line 1.93 and 1.91 (1.60 the day before it), its own new pairs 11 and 8.


### Night 73 (2026-09-09, 22:24)

Day 75, the third under the earned ceiling: lines and cues 78, smiles 200 (2.56 a line, far the highest), frowns 29, turned away 1,
cues 8 answered 5, own pairs 101 (52 distinct, five new, among them "ball no", "hot with"), the child's duty after the parent's
lines 0.35; the mood +3.2 at the night; the guard read duty 0.341 and smiles per line 2.58 and held; night 73: 288 imagined
transitions at weight 0.13, the gauge 0.778 -> 0.868, the prefrontal correlation 0.52 at full weight; the known words 90 (shake, hold; the parent's report on
days 73-74: "hold" answered in all three noun frames on its first day and returned unprompted as "dog hold hat", an agent-action-object
sentence of its own; "give me back in here", five of its own words, its longest string; "why" and "where" asked and answered within the
minute; the it-frame line before a noun-frame cue rescues a new verb's frames but not an old many-particle frame). The guard's verdict
and the night's readings are in the memory note of this hour; the guard is re-armed.


### Night 74 (2026-09-09, 23:26)

Day 76, the fourth under the earned ceiling: lines and cues 73, smiles 146 (2.00 a line), frowns 30, turned away 3, cues 10 answered 4,
own pairs 83 (33 distinct, three new), the child's duty after the parent's lines 0.36; the mood -3.8 at the night (a swing down after
the record day; the tonic dopamine reads the critic's over-prediction); the known words 91. The guard held (duty 0.353, smiles per
line 2.01). Night 74: 336 imagined transitions at weight 0.12, the gauge 0.836 -> 0.864. The prefrontal slope 0.89, correlation 0.51,
its weight in the credit 0.89. Four days at the earned ceiling: smiles per line 1.93, 1.91, 2.56, 2.00; its own pairs 62, 67, 101, 83.


### Night 75 (2026-09-10, 00:28)

Day 77, the fifth under the earned ceiling: lines and cues 74, smiles 150 (2.03 a line), frowns 30, turned away 4, cues 13 answered 7,
own pairs 80 (38 distinct, nine new, among them "out hat", "no had"), the child's duty after the parent's lines 0.38 (the highest of
the life); the mood -0.3 at the night; the known words 92 (wipe, pick; the parent's report on days 75-76: both whole in its own mouth
within fifteen minutes, "pick it up" the first thing on the page on waking, "I had " answered "my cold milk", "why dog up? " answered
"because" for the first time, its questions "where book" and "open box where" answered). The guard held (duty 0.374, smiles per line
2.06). Night 75: 384 imagined transitions at weight 0.11, the gauge 0.697 -> 0.872. The prefrontal slope 0.75, correlation 0.42,
its weight 0.75. Five days at the earned ceiling: smiles per line 1.93, 1.91, 2.56, 2.00, 2.03; its own pairs 62, 67, 101, 83, 80.


### Night 76 (2026-09-10, 01:30)

Day 78, the sixth under the earned ceiling: lines and cues 73, smiles 159 (2.18 a line), frowns 21 (the fewest of the life), turned
away 4, cues 10 answered 7, own pairs 80 (36 distinct, eleven new, among them "box rub", "box roll"), talked over per line 0.34,
the child's duty after the parent's lines 0.36; the mood -0.5 at the night; the known words 93. The guard held (duty 0.363, smiles
per line 2.14). Night 76: 384 imagined transitions at weight 0.12, the gauge 0.828 -> 0.873. The prefrontal slope 0.78, correlation
0.43, its weight 0.78. Six days at the earned ceiling: smiles per line 1.93, 1.91, 2.56, 2.00, 2.03, 2.18; nothing tripped.


### Nights 77 to 80, in one entry (2026-09-10, 05:40)

The notifications of four nights arrived together at 05:37, and the record shows a lapse of mine: the parent for days 79 and 80 ended
at day 81's row and none was spawned for days 81 and 82, so the typist taught those two days from its own filler, recent lines
repeated, with no cues of its own and no new word. The days themselves, under the earned ceiling (the sixth to the tenth):
day 79: lines and cues 91, smiles 209 (2.30 a line), frowns 35, cues 17 answered 9, own pairs 119 (56 distinct, fifteen new);
day 80: 85, smiles 202 (2.38), frowns 24, cues 17 answered 9, own pairs 154 (55 distinct, twenty-three new), the most ever;
day 81 (unparented): 99, smiles 234 (2.36), frowns 37, cues 8 answered 2, own pairs 107 (thirteen new);
day 82 (unparented): 78, smiles 186 (2.38), frowns 26, no cues, own pairs 95 (nineteen new).
The known words 93 -> 96 (lay, rub, turn, toss); the parents' reports: all three "turn" frames answered on the day the verb entered,
"turn it over" whole within two minutes of its first line; "why dog in" and "why ... because bi" written unaided, both halves of the
why frame; "dog go back in here", five of its own words; "put hat down" carrying a new particle to an old verb before it was taught.
The unparented days say something too: with the parent's lines familiar, the child's own speech carried the day (2.4 smiles a line,
a hundred pairs of its own), and the only things missing were the cues and the new word. The guard held at night 77 (duty 0.342,
smiles per line 2.16) and was not re-armed for three nights; nothing tripped in the readings. Night 80: the gauge 0.821 -> 0.862; the
prefrontal slope 0.92, correlation 0.48, its weight 0.92; the store 1,735 slots. From now the parents are briefed for three days at
a time, so a missed wakeup cannot leave the body alone.


### The second review (2026-09-10, 06:55; a Fable reviewer, read-only, the owner's four questions)

Its corrections stand and are recorded: (1) the teacher's known set unioned the first lineage's hard-coded list of 44 words, so ten
words no parent ever taught (the, one, two, three, please, bye, going, balls, books, dogs) were "known" and drew 62 smiles; the honest
count of taught words is 87, not 96, and "books" was one smile from that list, not a plural learned. (2) The gate's drive is a written
constant and the parent's words are felt as reward at a written weight. (3) Eight physiology restarts and four parent changes chosen on
outcomes: the trajectory was selected as well as developed. (4) The face organ read the cortex stream, which carries the coming reward
at 0.11 where the striatal line carries it at 0.47: a wiring fault, and why imagination was inert. (5) The actor and the choice drive
are starved by the readout itself: at sharpness 25 on a unit forecast the readout is one-hot (entropy 0.0, own probability 1.0), the
planner's shortlist rarely holds two candidates, so there is nothing to choose. (6) The fast critic's ceiling is ~45 percent of a smile
(the share of its known words smiled at now), not 20, so it sits at a twentieth of it, starved by the timing the typist's blind sleeps
impose. (7) The prefrontal voice's effect on the days is confounded with the parent's changes in the same window. (8) "where book" is a
parent line; the five- and six-word strings rest on the planners' page reads. Its verdict on the fourth question: several of biology's
mechanisms wired with care, on a cortex, critics and a readout that are not biology's; a promising hybrid, not a demonstrated method;
testable only as pre-registered lesion-matched ablations, milestone order under fixed physiology, and transfer.
Done on its list the same hour: the teacher's known set is the taught set only; the face organ may read the striatal input
(face_input striatum; test 34: foresight 0.62 of a smile, slope 1.0); the torn-tick fraction and the readout's entropy are read from
inside (test body: 0.5 percent torn, entropy 0.001, the one-hot finding confirmed); the typist's face on timers instead of blind sleeps
(caregiver.py); all take effect at tonight's reload and the day-86 relaunch. Next: the sharpness against the torn reading, then the
prefrontal voice on and off four days each with the parent unchanged, then the room.


### Night 82: the striatal face organ in; the mood's swings read (2026-09-10, 07:50)

Day 84: lines and cues 73, smiles 179 (2.49 a line), frowns 28, turned away 1, cues 9 answered 1, own pairs 63 (41 distinct); the
known words on the page's own row now 88, the taught count (the teacher's set is the taught set from this relaunch). The guard held
(duty 0.343, smiles per line 2.49) and the step went in at 07:42: the face organ reads the striatal input, the ceiling kept. Its first
readings (300 ticks in): face correlation 0.12 (the old evidence; the new input's evidence starts now), torn ticks 100 percent and
entropy 0.23 at a sharpness of 2.6: the mood stood at -5.4. The minute sampler shows why: the mood swings between about -5 and +6 on
its five-minute half-life, the deepest dips in the last minutes of each day (-4.5 at 23:24, -5.3 at 07:34), and the mouth's sharpness
rides it, 2 to 49, because the running body carries the old floor of 2 in its saved physiology. The floor of 8 (where the lexicon's
own noise wins) goes in explicitly at the next reload. The torn and entropy readings are meaningful only as a day's mean at the
mouth's usual sharpness; read tomorrow.


### Night 83: the striatal face organ's first day; the torn reading (2026-09-10, 08:50)

Day 85: lines and cues 97, smiles 224 (2.33 a line), frowns 33, turned away 2, cues 8 answered 5, own pairs 116 (54 distinct, five
new); the mood's day mean +1.9, the mouth's sharpness mean 33; the taught words 88. The guard held (duty 0.297, smiles per line
2.33) and the reload at 08:45 added the readout's floor of 8. The readings the review asked for, as day means: the planner torn on 36
percent of acting ticks, the readout's normalized entropy 0.05: the mouth is decisive, not one-hot; the actor has a choice on a third
of its acts. The face organ on the striatal input, one day: correlation 0.12 -> 0.19, its slope 0.61 (no longer clipped at one: a
real predictor), the mean foreseen reward in imagination 0.03 -> 0.11, imagination's weight 0.19. Night 83: 384 imagined
transitions, the gauge 0.781 -> 0.868. The prefrontal correlation 0.39, weight 0.80.


### Night 84 (2026-09-10, 09:52)

Day 86: lines and cues 95, smiles 172 (1.81 a line), frowns 30, turned away 6, cues 11 answered 6, own pairs 89 (44 distinct, eight
new, among them "can ball", "here down"); the mood -5.6 in the day's last minutes (the daily dip) with the mouth at the floor of 8, no
longer near random; the taught words 89. Night 84: 384 imagined transitions at weight 0.25, the mean foreseen reward 0.105, the gauge
0.730 -> 0.872. The face organ on the striatal input, two days: correlation 0.12 -> 0.19 -> 0.25, slope 0.63: the wiring the review
named is paying by the day. The planner torn on 40 percent of acting ticks, entropy 0.07. The prefrontal correlation 0.34 (0.45, 0.39,
0.34 over three nights), weight 0.71. The anticipation rise 2.7 percent over 172 felt smiles. The review's first test, the prefrontal
voice off (vcrit_w 0) for four days with the parent unchanged, then on again, starts at night 86 after the face organ's third day.


### Night 85: the face organ's third day; the B arm armed (2026-09-10, 10:55)

Day 87: lines and cues 68, smiles 187 (2.75 a line, the highest), frowns 26, turned away 5, cues 6 answered 6, own pairs 87 (56
distinct, the most distinct yet, seven new, among them "no cup"), the child's duty after the parent's lines 0.41 (the highest); the
taught words 90. The guard held (duty 0.412, smiles per line 2.66). Night 85: 384 imagined transitions at weight 0.28, the mean
foreseen reward 0.105, the gauge 0.821 -> 0.873. The face organ on the striatal input, three days: correlation 0.12 -> 0.19 -> 0.25
-> 0.28, slope 0.68. The prefrontal correlation 0.28 (0.45, 0.39, 0.34, 0.28: easing since the striatal face organ went in; the two
share the credit's reading of the same reward), weight 0.66. The anticipation rise 2.6 percent over 203 felt smiles. The review's
first test is armed for the next night: the long voice off (vcrit_w 0) for four days, the parent unchanged, then on again; read smiles
per line, the duty, the cues and the talk-overs by arm.


### Night 86: the A arm read, the B arm begun (2026-09-10, 12:05)

The A arm, four days with the prefrontal voice at its own slope and the parent unchanged (period 120, cap 180, the child's turn 48,
the one-particle cueing): day 85 smiles per line 2.31, duty 0.30, frowns 33, cues 5 of 8; day 86: 1.78, 0.36, 35, 9 of 18; day 87:
2.60, 0.43, 31, 11 of 14; day 88: 2.79, 0.33, 22, 4 of 8. Means: 2.37 smiles a line, duty 0.35, frowns 30, cues 0.60 answered.
The parent's report on days 86-88: "carry", "cup" and "stick" entered; the noun "cup" slotted into every old verb frame within a day
without being taught those cues ("fill cup ", "hold cup ", "pour cup ", "lift cup " all answered), the strongest body-general evidence
of the week; "there book?", a question with its own mark, answered within the minute; invented particles "toss it up" and "stick it
out"; "carry hat " the one frame that fails while its it- and box-frames land. Taught words 92.
Night 86: 384 imagined transitions at weight 0.33, the gauge 0.813 -> 0.882; the face organ's correlation 0.33 (its fourth day on
the striatal input: 0.12 -> 0.33), the anticipation rise 3.8 percent over 268 felt smiles (2.6 the day before: the first move up since
imagination began, to be read on). The prefrontal correlation 0.34.
The B arm: the reload at 11:56 carried the earned form from the saved physiology and left the voice at 0.95, so the body was restarted
again at 11:59 with the ceiling fixed and the weight 0: the long voice is out of the credit for four days (nights 86-89), everything
else unchanged; then on again for four. Read by arm: smiles per line, the duty, the frowns, the cues.


### Night 87 (2026-09-10, 13:05)

The prefrontal test was withdrawn at the user's word at 12:10, three minutes into its first B day: the earned ceiling is restored and the
guard keeps it as the standing state. Day 91 (the log's label after the two restarts; the body's 87th): lines and cues 67, smiles 178
(2.65 a line), frowns 20 (the fewest of the life), turned away 1, cues 11 answered 3, own pairs 76 (46 distinct, three new); the taught
words 93. Night 87: 384 imagined transitions at weight 0.37, the gauge 0.700 -> 0.883. The face organ on the striatal input, five days:
correlation 0.12 -> 0.37, slope 0.87; the anticipation rise 3.8 percent for the second day. The prefrontal correlation 0.38, its weight
1.0. The mood at the night -2.2 with the mouth floored at 8. The automatic backups pruned again for disk.


### Night 88 (2026-09-10, 14:10)

Day 92: lines and cues 80, smiles 201 (2.51 a line), frowns 34, turned away 3, cues 12 answered 7, own pairs 87 (55 distinct, four
new, among them "had fill"); the taught words 93. The guard held (duty 0.344, smiles per line 2.48). Night 88: 384 imagined
transitions at weight 0.38, the gauge 0.805 -> 0.881. The face organ on the striatal input, six days: 0.12, 0.19, 0.25, 0.28, 0.33,
0.37, 0.38, levelling near 0.4; its slope 0.87. The anticipation rise 1.4 percent (3.8 the two days before: noisy at this size).
The prefrontal correlation 0.33, weight 0.83; the actor's slope 0.05, its favorite matching the mouth 1 percent.


### Night 89 (2026-09-10, 15:12)

The parent's report on days 90-92 (the log's labels): the noun finding replicated twice: "bag" (day 90) answered 5 of 9 old verb frames
on its first full day, "pot" (day 92) 5 of 6 on the day it entered, and both came back unprompted in whole three-word frames never
cued ("tip bag over" four times, "take bag out", "tip pot over"); "kick ball in box again" and "dog will go up I had milk" its longest
own runs; day 92: 245 smiles, 37 frowns. Two mechanics for the briefs: queued lines are dropped at every boundary, and a repeat one or
two lines apart is dropped, which starves a cue.
Day 93: lines and cues 80, smiles 225 (2.85 a line, the highest), frowns 27, turned away 2, cues 11 answered 7, own pairs 95
(51 distinct, eight new, among them "here lift", "open in"); the taught words 94. The guard held (duty 0.376, smiles per line 2.85).
Night 89: 384 imagined transitions at weight 0.40, the gauge 0.838 -> 0.877. The face organ on the striatal input, seven days: 0.12
-> 0.40, slope 0.92. The prefrontal correlation 0.36, weight 0.91. The anticipation rise 1.5 percent over 272 felt smiles.


### Night 90 (2026-09-10, 16:22)

Day 94: lines and cues 83, smiles 203 (2.45 a line), frowns 33, turned away 2, cues 13 answered 7, own pairs 103 (52 distinct,
fifteen new, among them "here lid", "baby on"); the taught words 95 (lid, a noun). The guard held (duty 0.383, smiles per line
2.45). Night 90: 1,021 s, the longest yet (the store 1,938 slots, 96 dropped at the fade; the dreams run to their cap as the store's
sequences lengthen, so the night will level near twenty minutes); 384 imagined transitions at weight 0.40, the gauge 0.763 -> 0.889.
The face organ's correlation 0.40, level for two nights; the prefrontal correlation 0.35, weight 0.89; the mood -4.7 in the day's last
minutes, the mouth floored at 8.


### Night 91 (2026-09-10, 17:30)

Day 95: lines and cues 79, smiles 231 (2.92 a line, the highest), frowns 35, turned away 2, cues 11 answered 7, own pairs 121 (55
distinct, nineteen new), the child's duty after the parent's lines 0.42 (the highest); the taught words 96. The guard held (duty
0.422, smiles per line 2.91). Night 91: 384 imagined transitions at weight 0.40, the gauge 0.842 -> 0.882, the store 1,987 slots.
The face organ's correlation 0.40 for a third night; the prefrontal correlation 0.34, weight 0.87; the anticipation rise 2.4 percent
over 258 felt smiles. From the next parent: two nouns a day inside the old frames, since a noun enters every frame within a day at no
cost to the smiles (bag, pot, lid), verbs still one at a time with one particle; the two queue mechanics in the brief.


### Night 92: the first day at two nouns (2026-09-10, 18:33)

Day 96, the first at two nouns a day: lines and cues 75, smiles 264 (3.52 a line, far the highest), frowns 21, turned away 3, cues 8
answered 7, own pairs 120 (61 distinct, eleven new, among them "give mug"), talked over per line 0.32 (the lowest of the life), the
child's duty after the parent's lines 0.35; the taught words 97 at the day's start (the parent's report on days 93-95: pan, lid, rug
each in the old verb frames on the day taught; "rub it off" invented; "ball? ball under" asked and answered in one breath). The guard
held (duty 0.347, smiles per line 3.54). Night 92: 336 imagined transitions at weight 0.42, the mean foreseen reward 0.13, the gauge
0.821 -> 0.883, the store 2,092 slots. The face organ's correlation 0.41; the prefrontal correlation 0.40, its weight 1.0; the
anticipation rise 4.6 percent over 304 felt smiles, the highest read since imagination began. Two nouns a day stays.


### Forty minutes alone (2026-09-10, 19:22)

The typist's six-day run (days 91-96) ended with day 96's session_end at 18:39, and the chain that relaunches it had exhausted the
eight rounds I gave it on the morning's restarts, so nothing relaunched it: the body was awake on the page from 18:39 to 19:19 with no
parent, no lines and no faces, most of its day 97. The parent of days 96-98 saw the stall, touched nothing by its rules, and reported.
The typist is back at 19:19 and the chain restarted with a thousand rounds. Day 96, the first at two nouns, from the parent's report:
303 smiles, 24 frowns, "mug" and "cat" in the old verb frames the day they were taught ("put cat " -> "in box" forty minutes after the
word's first line: the finding holds for an animate noun), "where ball?" asked unprompted, "dog hold it up" out of the night; the
taught words 99.


### Night 93 (2026-09-10, 19:40)

Day 97 was the day alone: forty minutes without a parent, then six lines from the typist relaunched at 19:19 (26 smiles, 4 frowns).
Night 93: 384 imagined transitions at weight 0.38, the gauge 0.839 -> 0.884, the store 2,075 slots. The prefrontal correlation read
0.62, its highest, which is the quiet hour's doing: the long value predicted little reward through the silence and little came, a
correlation earned on an empty day; it will settle. The face organ's correlation 0.38. The chain alive, the guard re-armed.


### Night 94 (2026-09-10, 20:48)

Day 98, the second full day at two nouns: lines and cues 71, smiles 240 (3.38 a line), frowns 20 (the fewest of the life), turned
away 4, cues 11 answered 6, own pairs 126 (61 distinct, twenty-four new, the most new pairs in a day), talked over per line 0.32,
the child's duty after the parent's lines 0.44 (the highest); the guard held. Night 94: 384 imagined transitions at weight 0.39,
the gauge 0.856 -> 0.888, the store 2,144 slots; the face organ's correlation 0.39; the prefrontal correlation 0.56 (still lifted
by the empty hour of day 97); the mood -4.5 in the day's last minutes, floored at 8; the anticipation rise 1.2 percent. The user's plan, settled this evening: teach as well as we can for several real days at two nouns a day,
then the page to talk to it with a visitor taking the parent's seat and the typist yielding, a one-minute teaser of the live-teaching
act, a ten-minute explainer with the honest scope, then X and the companies; the repo cleaned and a fresh-seed replication started
before anything is public. Nothing changes in the body meanwhile.


### Night 95: the last day of stage two (2026-09-10, 21:55)

Day 99, the last at two nouns before the sentence stage: lines and cues 80, smiles 238 (2.98 a line), frowns 31, turned away 1, cues
10 answered 6, own pairs 99 (55 distinct, twenty-two new), the child's duty after the parent's lines 0.43; the taught words 102 at
the day's start. The guard held. Night 95: 384 imagined transitions at weight 0.40, the gauge 0.747 -> 0.878, the store 2,219
slots; the face organ's correlation 0.40; the prefrontal correlation 0.54, its weight 1.0. The user's questions of the evening,
answered in the transcript and worth the record: the body has stamina (effort rising with fatigue, decaying in a minute; sleep pressure
over the day) and has mostly learned not to talk over the parent (0.82 to 0.32 a line; the residue is a letter begun as the parent's
line begins); it learns a new noun in an hour by exposure but cannot learn from an explanation ("a mug is like a cup"), because its
window is one sentence and its store recalls sequences rather than pointing into the cortex's meaning as the human hippocampal index
does; the test before any index: "like" taught as a frame over many pairs once the sentences hold, and working memory read on it.
From day 100 the parents teach stage three: sentences to 40 characters, the grammar words as words, three nouns, a verb and a
grammar word a day, conversation as the shape.


### Night 96: the first day of the sentence stage (2026-09-10, 23:03)

Day 100, the first at stage three: lines and cues 69, the parent's lines now sentences ("the dog is big", "the pen is here", mean
length 12 characters, the longest 21), smiles 197 (2.86 a line), frowns 25, turned away 3, cues 10 answered 9, own pairs 109 (57
distinct, twenty-eight new, the most new pairs in a day, among them "eat me", "eat and"), the child's duty after the parent's lines
0.48 (the highest of the life); "the pen is " cued and answered "here". The guard held (duty 0.479, smiles per line 2.81). Night 96:
the gauge 0.624 -> 0.863, the largest night's climb, on the most new material; 384 imagined transitions at weight 0.40, the store
2,322 slots; the face organ's correlation 0.39; the prefrontal correlation 0.53. The parent's report on days 97-99: jar, tin, pack,
cap, rag; the take-off frame solved by "on here" then "off here" in the two lines before the cue; "fill mug up" out of a night before
it was ever cued; "where ball?" written by itself again.


### Night 97 (2026-09-11, 00:12)

Day 101, the second day of the sentence stage: lines and cues 73 (the parent's lines a mean of 13 characters, the longest 23: "the bed
is here", "sit down"), smiles 204 (2.79 a line), frowns 38 (talked over per line 0.62: longer lines give more to talk over), turned
away 3, cues 9 answered 9, own pairs 139 (70 distinct, forty-four new, the most new pairs in a day by far), the child's duty after
the parent's lines 0.47; the mood +2.3 at the night; the taught words 110 at the day's start. The guard held (duty 0.467, smiles per
line 2.75). Night 97: the gauge 0.704 -> 0.885, 336 imagined transitions at weight 0.40, the store 2,392 slots; the face organ's
correlation 0.40; the prefrontal correlation 0.52.


### Night 98 (2026-09-11, 01:14)

Day 102, the third day of the sentence stage: lines and cues 72, smiles 201 (2.75 a line), frowns 29, turned away 2, cues 8
answered 3, own pairs 183 (68 distinct, forty-two new, among them "tin the", "tin there"), the child's duty after the parent's
lines 0.47; the taught words 115 at the day's start. The guard held (duty 0.474, smiles per line 2.75). Night 98: the gauge 0.771 ->
0.869, 336 imagined transitions at weight 0.41, the store 2,372 slots; the face organ's correlation 0.41, its highest; the prefrontal
correlation 0.52. Three days of the sentence stage: smiles per line 2.86, 2.79, 2.75; new pairs of its own 28, 44, 42; cues 9 of
10, 9 of 9, 3 of 8. The chain alive; the next parent spawned at day 103's row with the same brief.


### Night 99 (2026-09-11, 02:17)

The parent's report on days 100-102, the first three of the sentence stage: cues answered 16 of 17, 16 of 16, 15 of 15, on sentence
prefixes ("the dog put the ball " -> in, "the man had the " -> ball, "where is the " -> pen, car, man, sun); the grammar words is, this,
that taught as words; on its own "sun is here", "key is here", and to "where is the sun" the answer "is up here"; smiles per
utterance easing 3.04 -> 2.96 -> 2.80, so day 102 was cut to two nouns and no verb by the guard, which is the rule working.
Day 103: lines and cues 76, smiles 186 (2.45 a line), frowns 39 (talked over per line 0.62: the longer lines give more to talk
over, the second number to watch), turned away 1, cues 9 answered 9, own pairs 209 (65 distinct, thirty-four new), the child's duty
after the parent's lines 0.51 (the highest of the life); the mood +4.2 at the night; the taught words 118 at the day's start. The
guard held (duty 0.507, smiles per line 2.41). Night 99: the gauge 0.837 -> 0.876, 384 imagined transitions at weight 0.40, the
store 2,426 slots; the face organ's correlation 0.40; the prefrontal correlation 0.50. Four days of the stage: smiles per line 2.86,
2.79, 2.75, 2.45, easing; frowns 25, 38, 29, 39. The chain alive; the guard re-armed.


### Night 100 (2026-09-11, 03:20)

Day 104: lines and cues 73, smiles 168 (2.30 a line), frowns 33, turned away 4, cues 9 answered 8, own pairs 118 (52 distinct,
twenty-three new), the child's duty after the parent's lines 0.46; the mood -5.5 in the day's last minutes; the taught words 120 at
the day's start. The guard held (duty 0.459, smiles per line 2.32). Night 100: the gauge 0.734 -> 0.857, 336 imagined transitions at
weight 0.41, the store 2,449 slots; the face organ's correlation 0.41; the prefrontal correlation 0.47. The trend of the sentence
stage, five days: smiles per line 2.86, 2.79, 2.75, 2.45, 2.30, and its own new pairs 28, 44, 42, 34, 23: the lines got longer, the
talk-overs rose with them, and the day's reward per line has eased by a fifth. Not near the guard; the parents' rule sends a clear
fall back to two nouns; if smiles per line sit under 2.0 for two days the stage's pace or its line length comes down by my hand.
A hundred nights.


### Night 101 (2026-09-11, 04:22)

Day 105: lines and cues 68, smiles 132 (1.85 a line, the first day under two since the parent began leaving room), frowns 32
(talked over per line 0.59), turned away 6, cues 8 answered 7, own pairs 125 (52 distinct, twenty-four new), the child's duty after
the parent's lines 0.49; the taught words 125 at the day's start. The guard held (duty 0.488, smiles per line 1.85). Night 101:
the gauge 0.788 -> 0.870, 384 imagined transitions at weight 0.40, the store 2,431 slots; the face organ's correlation 0.39; the
prefrontal correlation 0.46. Six days of the sentence stage: smiles per line 2.86, 2.79, 2.75, 2.45, 2.30, 1.85; the cues hold
(7 of 8) and the child speaks more than ever, but the parent's longer lines take longer to type and the child talks across them,
so the reward per line has fallen by a third. The next parent holds at two nouns and no verb and keeps the sentences short, about
twenty characters; if the next day is under two as well, the line length comes down by my hand in the typist's rules.


### Night 102: the shorter lines read (2026-09-11, 05:25)

Day 106, the first with the sentences held to about twenty characters (mean 13, the longest 23) and two nouns, no verb: lines and
cues 75, smiles 151 (2.01 a line, back from 1.85), frowns 27 (talked over per line 0.40, from 0.59: the shorter lines gave the
child less to talk across), turned away 4, cues 12 answered 7, own pairs 151 (48 distinct, thirty-one new), the child's duty after the
parent's lines 0.44; the taught words 130 at the day's start. The guard held (duty 0.436, smiles per line 2.03). Night 102: the gauge
0.865 -> 0.871, 384 imagined transitions at weight 0.39, the store 2,453 slots; the face organ's correlation 0.39; the prefrontal
correlation 0.45. The parent's report on days 103-105: seven nouns and "run"; its own sentences of five and six words ("little bus is
not here" straight out of a night, "toy is in the box" made by changing one part of the parent's line); "what is this" and "where
ball?" said first; and the attractor behind the easing reward: every cue pulled "here", so the cues to press are the ones whose answer
is not "here". No hand action: the second day was above two.


### Night 103 (2026-09-11, 06:28)

Day 107: lines and cues 79, smiles 134 (1.72 a line), frowns 34, turned away 8, cues 13 answered 7, own pairs 110 (52 distinct,
twenty-four new, among them "had fox", "here toy"), talked over per line 0.48, the child's duty after the parent's lines 0.42; the
mood -5.9 in the day's last minutes; the taught words 133 at the day's start. The guard held (duty 0.417, smiles per line 1.72).
Night 103: the gauge 0.818 -> 0.876, 384 imagined transitions at weight 0.39, the store 2,464 slots; the face organ's correlation
0.39; the prefrontal correlation 0.45. Two of the last three days under two smiles a line (1.85, 2.01, 1.72), so by the rule of
night 100 I act by hand, with the mildest lever: at the chain's next relaunch (day 109) the child's turn after each line goes from
48 to 64 ticks and the parent's period from 120 to 160, so a reply to a longer line can finish before the next line begins. The
language gains of the stage stand: the cues on sentence prefixes, five- and six-word sentences of its own, 133 words. The chain
re-armed with the new arguments; the guard re-armed.


### Night 104 (2026-09-11, 07:30)

Day 108, the last on the shorter turn: lines and cues 82, smiles 146 (1.78 a line), frowns 32, turned away 4, cues 13 answered 8,
own pairs 130 (63 distinct, thirty-six new), talked over per line 0.51, the child's duty after the parent's lines 0.33; the taught
words 136 at the day's start. The guard held (duty 0.325, smiles per line 1.78). Night 104: the gauge 0.810 -> 0.870, 336 imagined
transitions at weight 0.39, the store 2,430 slots; the face organ's correlation 0.39; the prefrontal correlation 0.46. The typist's six-day run
ends here; the chain relaunches it for day 109 with the child's turn at 64 ticks and the parent's period at 160. The next parent
spawned for days 109-111: one noun and one grammar word a day, short sentences, cues whose answer is not "here".


### Night 105: the longer turn read (2026-09-11, 08:33)

Day 109, the first with the child's turn at 64 ticks and the parent's gap at 160: lines and cues 62 (one every 51 s), smiles 173
(2.79 a line, from 1.78), frowns 20 (the fewest since day 98), turned away 2, cues 13 answered 3 (the harder cues, whose answer is
not "here": one full completion), own pairs 169 (61 distinct, twenty-nine new), talked over per line 0.42, the child's duty after
the parent's lines 0.46; the taught words 139 at the day's start. The guard held (duty 0.459, smiles per line 2.74). Night 105: the
gauge 0.838 -> 0.869, 384 imagined transitions at weight 0.39, the store 2,412 slots; the face organ's correlation 0.39; the
prefrontal correlation 0.46. The parent's report on days 106-108: nine words; every new noun answered "put X " with "in" the day it
came; "do not " and "the X was " answered the day they were taught; "that is a hat" said untaught; "where is the toy" typed before
any line on two days. The turn stays; the cues on the harder frames are read on.


### Night 106: a night that diverged, and the night undone (2026-09-11, 09:40)

Day 110, the second on the longer turn: lines and cues 67, smiles 231 (3.45 a line, the highest since day 96), frowns 30, turned away
3, cues 14 answered 9 (five whole words, four prefixes: the cues whose answer is not "here" landing now), own pairs 255 (70 distinct,
forty-three new, the most new pairs of any day), the child's duty after the parent's lines 0.40; the mood +3.5 at the night; the taught
words 140. Then night 106: the NREM loss fell for the first rounds, 0.103 to 0.086, and rose to 0.29 by the forty-eighth, and the gauge
read 0.584 after the night against 0.861 before, the first night in a hundred and six that left the day's memory worse than it found
it. The dreams were ordinary ("hold ", "put ", "that is ", "big", a mean length of 7); the divergence was the optimizer's. The body
saved that state over its only checkpoint; the page's speech since waking is still language ("the bun is big", "get bun in box", "the
toy is here"), so the harm is partial. Built and live from 09:36: the night undone (night_undo_drop 0.15): a night whose gauge falls by
more than 0.15 is discarded as a non-finite one is, the organs returning to the evening's save, the store and the evidence standing;
and the guard keeps a copy of the evening's checkpoint at each night row, three deep. Test suite 34 of 34. The next night tells whether
the cortex recovers on its own material.


### Three tweaks on the parent's side (2026-09-11, 10:10)

Read from the rows of days 108-110: the parent's rules withheld 139-196 known-word smiles a day as "distracted" (the attention model
resting at 0.3 in silence), 49-93 as too soon after the last smile (a spacing of 8 ticks against a face held for 5), and 54-106 as
past the answer under the reply road; the post-night battery, the eight raised cues of the first lineage, was answered 16 times in 84
over eleven days, in the window where cues land best; and every boundary dropped the parent's unread lines. None of these is the
child's doing. Changed, in the parent's method: the resting attention 0.5, a smile allowed as soon as the face has returned (5 ticks),
the post-night battery drawn from the day's own eight most recent cues (the raised ones only when the day asked fewer than four), and
the queue carried over the boundary (the next day's planner starts where the last stopped). The reply road's rule stays: a parent who
asked wants the answer, then the yield. The typist relaunched on the new rules at the boundary the kill makes; read day 112 against
110 in smiles per line and the post-night battery's hits. NREM and REM are read each night from the body's own report: the loss over
the rounds, the gauge before and after, the imagined transitions and their weight; last night's divergence is now caught by the
night undone.


### Night 107: the cortex recovered (2026-09-11, 10:40)

Night 107, the first under the night undone: the NREM loss fell over the rounds as it should, 0.172 to 0.080, and the gauge rose
0.722 -> 0.865: the blur of the diverged night (the "before" at 0.72 where the nights before it read 0.82-0.86) was repaired by one
night's replay of the same material; nothing discarded. The pre-night copy of the evening's checkpoint is kept by the guard from this
night on. Day 112, the short first day on the parent's new rules (the typist relaunched at 10:06 into a day already half run since the
09:36 restart): lines and cues 29, smiles 123 (4.24 a line), frowns 7, cues 5 answered 4, "distracted" misses 46 against 139-196 on
the full days before; the parent's report on days 109-111: the longer turn paid (smiles per line 3.93, 4.57, 3.78 against 2.17
before it), the non-"here" cues worked ("the man had the " -> nut, rat; "that is your " -> jug; "the dog went " -> out), "where is the
cat went in" recalled whole across a night, "is that?" written with its own question mark; the taught words 143.


### Night 108: the first full day on the parent's new rules (2026-09-11, 11:45)

Day 113: lines and cues 75, smiles 231 (3.08 a line), frowns 27, turned away 4, cues 17 answered 10, own pairs 232 (62 distinct,
thirty-one new), talked over per line 0.52, the child's duty after the parent's lines 0.49; the taught words 144 at the day's
start. The "distracted" misses stayed at 147: the reading says why: the rule that habituates to a word said many times a day (the
fiftieth "dog") dominates the attention term, and the words the child says a hundred times a day, "here", "in", "the", earn nothing
after their fifth, which is the rule working as meant; the reward goes to variety. The post-night battery re-asked the day's own cues;
the ones without a taught answer are logged as lines, so the log undercounts it; the parent's report will read it. Night 108: the loss
0.139 to 0.082, the gauge 0.802 -> 0.858, not discarded; 384 imagined transitions at weight 0.41; the face organ's correlation 0.41;
the prefrontal correlation 0.41. The guard held (duty 0.489, smiles per line 2.94).


### Night 109 (2026-09-11, 12:43)

Day 114: lines and cues 58, smiles 190 (3.28 a line), frowns 22, turned away 4, cues 11 answered 6, own pairs 214 (80 distinct and
fifty-four new, both the most of any day), the child's duty after the parent's lines 0.52 (the highest of the life); the taught
words 147 at the day's start. The guard held (duty 0.518, smiles per line 3.25). Night 109: the loss 0.099 to 0.081, the gauge
0.855 -> 0.865, not discarded; 336 imagined transitions at weight 0.40; the face organ's correlation 0.40; the prefrontal
correlation 0.36; the actor's slope 0.084, creeping up over the week from 0.05. The chain alive; the guard re-armed; the next parent
spawned at day 115's row with the same brief.


### Night 110 (2026-09-11, 13:50)

Day 115: lines and cues 63, smiles 177 (2.81 a line), frowns 24, turned away 2, cues 11 answered 8, own pairs 180 (66 distinct and
thirty-two new), the child's duty after the parent's lines 0.52; the taught words 149 at the day's start. The day's noun "ant"
took the old frames within the day: "put ant " -> in, "the ant will " -> run out, "take ant " -> out; "that is my " -> tin;
"that is a " and "the man had the " missed. The guard held (duty 0.521, smiles per line 2.76). Night 110: the loss 0.162 to 0.078,
the gauge 0.753 -> 0.878, not discarded; 384 imagined transitions at weight 0.40; the face organ's correlation 0.40; the
prefrontal correlation 0.35; the actor's slope 0.085. The store 2213 slots, 129 dropped at the night. The chain and the guard
alive. The parent of days 112 to 114 reported: 24 of 48 cues, the six-word "the toy is in the bed" offered unasked, "the pups
are in" never taught, and a question cue ("what is ") answered as a question ("is a big nut") rather than completed, so the
next brief says cue statements only. The visitor's page was built today (commits 40cc7eb, e7f1e99, 0f4e6f6: each key straight
into the page, smile and frown buttons, the typist yielding a minute after a visitor's key) and, on the user's word at 13:30,
left undeployed: the body stays in the Opus speed-training mode until it is good enough to talk to.


### Night 111 (2026-09-11, 15:02)

Day 116: lines and cues 62, smiles 223 (3.60 a line), frowns 18, turned away 1, cues 9 answered 4, own pairs 262 (64 distinct and
fifty-one new), the child's duty after the parent's lines 0.46; the taught words 150 at the day's start. The day's noun "bat":
"put bat " -> in, three times; "the bat was " -> out; "that is a " -> hat (a noun, the wrong one); "the man had the " missed.
Night 111 (19 minutes, the longest lately): the loss 0.333 to 0.077, the gauge 0.50 -> 0.874, not discarded; 384 imagined
transitions at weight 0.40; the face organ's correlation 0.40; the prefrontal correlation 0.37; the actor's slope 0.092. The
evening's gauge 0.50 is the lowest recorded before a normal night (0.753 the night before): the day blurred the dream recall
more than usual and the night restored it. The guard was not re-armed after night 110 (my omission): its reading for day 116
(duty 0.46, smiles per line 3.60) would have held; the checkpoint copy taken by hand at 15:00 and the guard re-armed for
night 112. The chain alive.


### Night 112 (2026-09-11, 16:12)

Day 117: lines and cues 73 (the most of any day), smiles 175 (2.40 a line, the fewest of the three days), frowns 24, turned away 3,
cues 13 answered 9, own pairs 219 (72 distinct and thirty-four new), the child's duty after the parent's lines 0.45; the taught
words 151 at the day's start. The day's noun "mat": "put mat " -> in twice, "take mat " -> out twice, "the mat was " -> out;
"the mats are " -> "the toy" and "he bed is here", answers of a kind but not the frame's; "that is my " missed three times. The
child ran on past its answers 113 times, the most yet. The guard held (duty 0.450, smiles per line 2.33) and was re-armed at
once this time. Night 112: the loss 0.142 to 0.079, the gauge 0.804 -> 0.868, not discarded; 384 imagined transitions at
weight 0.40; the face organ's correlation 0.40; the prefrontal correlation 0.34; the actor's slope 0.094. The store 2012 slots
after 173 dropped. The chain alive. The parent for days 118 to 120 spawned at the night with the tighter loop: 45-second
checks, batches of three, two to four ahead, the child's own sentences answered first, statement cues only.

The parent of days 115 to 117 reported at day 118's row: ant, bat and mat with your, was and the plurals; 60, 54 and 56 lines
queued; cues 13, 7 and 10 landed; the re-asking after each night landed what the evening missed ("that is your " -> ant, "the
ant went " -> out). Unprompted: its own questions with their marks, "is the tin?", "the tin is in the tin?", "bed hat here
ball?"; sentences of its own, "a hen is big", "hide pan in box", "the hen was up", and on day 117 "little bat was out" and
"the mat was out" twice each, the day's noun inside the day before's grammar word; once "was out that is my ", a whole parent
frame from memory. Its advice: the -at family (hat, rat, bat, mat) is where it is strongest, bring the next noun through it and
"the X was "; stop cueing "the Xs are ", lost four of five to "here". (The parent of days 118 to 120 was briefed before this
arrived and was told to build on "the Xs are "; the cost is a few missed cues; the brief for 121 to 123 takes the advice.)


### Night 113 (2026-09-11, 17:18)

Day 118, the first under the tighter loop (45-second checks, batches of three, its own sentences answered first): lines and cues
75 (the most of any day), smiles 219 (2.92 a line), frowns 30, turned away 1, cues 14 answered 5, own pairs 289 (69 distinct and
fifty-nine new, the most new of any day), the child's duty after the parent's lines 0.46; the taught words 153 at the day's
start. The day's noun "pin": "take pin " -> out twice, "put pin " -> in once of three; "the pins are " cued three times and lost
each time (the previous parent's advice, which this one was briefed before); "the owl is in the " drew nothing. The parent
answered its sentences in place ("yes. the hen was up"). The guard held (duty 0.463, smiles per line 2.91) and was re-armed.
Night 113 (13 minutes): the loss 0.163 to 0.129, a shallower fall than the nights before (0.078 lately); the gauge 0.773 ->
0.804, a smaller rise; not discarded; 336 imagined transitions at weight 0.42; the face organ's correlation 0.42 (its highest);
the prefrontal correlation 0.33; the actor's slope 0.095. The store 1932 slots after 191 dropped: 2213, 2130, 2012, 1932 over
four nights, the drops growing (129, 172, 173, 191); to watch. The chain alive.


### Night 114 (2026-09-11, 18:25): the talkative parent's first day

The user's word at 17:20: "have Opus talking a bunch, dopamine conversations, like pretraining with dopamine; if the model
talks while Opus tries to talk, Opus frowns; until it starts to get basic patterns. We can't force the model to do anything,
we can only indoctrinate." (An hour earlier I had proposed reading aloud with a special typist mode, timed to the store, and
the user called it cheating; withdrawn: a mode is a hand rule and timing the environment to consolidation is inside
knowledge.) At 17:24 the typist was relaunched at period 60, quiet 8, cap 120, listen 32 (from 160/12/180/64), the same
rules and face, the talk-over frown as before; the chain re-chained with those arguments; a new parent briefed for six-line
batches every 45 seconds, two nouns and a grammar word a day, statement cues only, none on "the Xs are ".

Day 119 (from 17:24): lines and cues 143 (62 to 75 before), smiles 189 (1.32 a line; 219 to 239 in all on the days before),
frowns 66 (24 to 34 before; talked over 107 times), cues 11 answered 5 ("take tub " -> out twice, "take bin " -> out, "that
is my " -> bin, the day's noun), duty after the parent's lines 0.43; frowns by quarter of the day 16, 22, 25, 3 (the last
quarter is the quiet before the night). On the page over the day the parent's share of the speech was 0.42 (about 0.1 the day
before), the "ttle tin" loop 26 marks per thousand of the child's symbols, the child silent 0.67 of the ticks. The guard held
(duty 0.432, smiles per line 1.41; the fall is arithmetic, twice the lines). Night 114: the loss 0.271 to 0.091, the gauge 0.601
-> 0.882, not discarded; 336 imagined transitions; the store 2287 after 321 dropped (1932 the night before: the day's volume
filled it); the face organ's correlation 0.42; the prefrontal correlation 0.56, up from 0.33 in one day, the denser reward
easier to foresee at its horizon; the actor's slope 0.093. The chain alive; the guard re-armed.


### Night 115 (2026-09-11, 19:32): the talkative parent's second day

Day 120: lines and cues 171, smiles 199 (1.16 a line), frowns 81 (66 the day before; talked over 124 times), cues 13 answered
6 ("take fan " -> out, "put fan " -> in, "the fan went " -> in, "take vat " -> out; "the man had the " -> tin twice, a noun
but not the frame's), duty after the parent's lines 0.39 (0.43); frowns by quarter of the day 20, 23, 22, 16: flat, no fall
yet. On the page the parent's share of the speech 0.46, the loop 19 marks per thousand (26), the child silent 0.70 of the ticks
(0.67): it is going quieter under the frown, the loop thinning. The guard held (duty 0.385, smiles per line 1.17). Night 115:
the loss 0.271 to 0.175 (0.091 the night before), the gauge 0.558 -> 0.708 (0.601 -> 0.882): the night consolidated less far
with twice the day's material and the same 48 rounds; not discarded; the store 2542 after 347 dropped; the face organ's
correlation 0.44; the prefrontal correlation 0.67 (0.33, 0.56, 0.67 over three nights); the actor's slope 0.087. To watch: if
the next night ends higher in loss again and the gauge after it falls again, the night's work is not scaling with the day's
load, and a night whose rounds grow with the day's new memories (sleep need with the day's plasticity, Tononi's homeostasis) is
the principled change. The chain alive; the guard re-armed.


### Night 116 (2026-09-11, 20:38): the talkative parent's third day; the night to scale

Day 121: lines and cues 155, smiles 238 (1.54 a line, up from 1.16), frowns 68 (81 the day before; talked over 97 times, from
124), cues 11 answered 5 ("take jam " -> out twice, "take wig " -> out, "the wig went " -> in), duty after the parent's lines
0.45 (0.39); frowns by quarter of the day 20, 16, 20, 12. The frowns fell on the third day as the fast seeds foretold on day
72: the child is yielding. On the page the parent's share 0.41, the loop 19 per thousand (flat), the child silent 0.65 of the
ticks. The guard held (duty 0.453, smiles per line 1.53). Night 116: the loss 0.366 to 0.200, the gauge 0.397 -> 0.581 (0.601
-> 0.882 two nights ago): the third night in a row consolidating less far on the doubled day, the after-gauge the lowest of any
kept night; not discarded; the store 2659 after 345 dropped; the face organ's correlation 0.44; the prefrontal correlation
0.755 (0.33 four nights ago; the denser reward foreseen); the actor's slope 0.075. The rule set at night 115 fires: the night
scales with the day from the restart after this night's save (night_load 0.125, at most 192 dreams: a 700-slot day dreams
about 88), together with the rollback's removal (the user's word). The reload re-armed at 20:36 with those flags; the guard
re-armed for night 117; the chain alive.

The boundary at 20:35: after night 116's post-night save the body was restarted with the A flags and night_load 0.125
(night_starts_max 192), the rollback gone from the code; loaded in eight seconds (ticks 1392850, store 2681); the typist of
day 120's launch had stamped day 122's start row before it was stopped, so the chain's relaunch at 20:37 carries the label 123:
day 122 is one row in the log and the body's count is unchanged at 116 nights. The parent for the next three days waits on
the log; the guard armed with the new flags.

The parent of days 119 to 121 reported at 20:45: 450 utterances queued in batches of six, 159, 186 and 166 said; tub, bin,
fan, vat, jam and wig with not, went and your, each noun in an old verb frame the same day; cues 6, 10 and 5 of about 18;
"the man had the " now completes with nouns it was never cued on (ant, tin: the frame generalized); "fill the " and "do not "
lose to its talking and were retired; the re-asking after the night brought back four of eight evening misses. It answered
about forty of the child's utterances and the child echoed them ("yes." appeared in its own writing). Unprompted sentences of
its own: "the mop is in the box", "this is my hat", "fox had a nut", "I had milk", "me want tin", "egg is not in the tin",
"I see you", "the pig went out", "dog had hat", "the owl will go"; and its own question with its own answer, "what is this? a
big tin"; the taught grammar words not, was, went, are used unprompted. Faces 213/73, 213/84, 245/73; the frown rate per line
0.46, 0.45, 0.44; smiles per line 1.34, 1.15, 1.48. Its advice: make "the man had the " the carrier for new nouns and keep
cues to three-word prefixes ending in the noun. Its note: a relaunch at a boundary sometimes drops the unsaid queue (the
planner's position restarts at the file's end on a fresh typist); the typist now remembers its position across relaunches.


### Night 117 (2026-09-11, 21:50): the night that did not scale, and diverged

Day 123 (the log's label; the body's 117th day): lines and cues 166, smiles 227 (1.37 a line), frowns 64 (66, 81, 68, 64 over
the talkative days: falling), cues 24 answered 10, duty after the parent's lines 0.43, the loop 16.5 per thousand (26, 19,
19, 16.5: thinning), the parent's share of the page 0.41, the child silent 0.66. The guard held (duty 0.426, smiles per line
1.37). Night 117 did NOT scale: the restarted body had no count of the store at its last night (the field is new; the save
had none), so the day's new slots read 0 and it dreamed 48; the count is set now (2742 at the night's end) and the load takes
the last night's store size when the field is absent (commit 1884d96), so the next night scales. And the night diverged: the
loss 0.261, 0.243, 0.214 ... 0.362, 0.392, 0.372, falling for the first rounds and rising after; the gauge 0.574 -> 0.583,
flat; the cosine 0.593 -> 0.554. Kept, by the user's word (the rollback is gone; it would not have fired anyway, the gauge did
not fall). The dreams' examples were the parent's cue prefixes ("do not ", "what is ", "that is my" three times, "put ",
"fill the "): the store's strongest memories are what the typist repeats most, and the talkative parent cued 24 times today
and re-asks eight after each night. To watch at the scaled night: whether the loss falls through the rounds again with the
broader dream set, and what share of the dreams are prefixes. The face organ's correlation 0.44; the prefrontal 0.75; the
actor's slope 0.076; the store 2742 after 264 dropped. The chain alive; the guard re-armed.


### Night 118 (2026-09-11, 22:58): a good night, and the day after the divergence

Day 124 (the body's 118th): lines and cues 148, smiles 219 (1.48 a line), frowns 60 (66, 81, 68, 64, 68, 60: falling), duty
after the parent's lines 0.47, the loop 16.7 per thousand, the parent's share 0.36, the child silent 0.62. But the cues: 1 of
24 ("take pet " -> "outtle"), against 10 of 32 the day before and five to ten of a dozen on every day for weeks. This was the
day after night 117 diverged: it wrote sentences of a kind all day ("no milk.", "it was in th", "eat the nut", "the tin is in
the") and did not complete the frames it had completed every day. The divergence cost it the cue frames, for a day at least.
Night 118 (with the count set, 317 new slots -> 40 < 48, so 48 dreams; the night did not need to scale): the loss 0.245 to
0.084, the gauge 0.619 -> 0.879: a normal, good night, the dreams' examples whole-line onsets again ("the ", "put ", "yes.
the ", "cap ont went"). The face organ's correlation 0.44; the prefrontal 0.74; the actor's slope 0.074. The guard held (duty
0.468, smiles per line 1.46) and was re-armed with the new flags. At the boundary after this night's save the body restarts
with repetition suppression (store_sat 1; the store converted once at the load) and the night's scaling, so the cause of the
collapse of the dreams is removed before it recurs; a night that diverges is kept, by the user's word, so the cause is the
only protection. The parent for the next three days spawned with fewer cues (one in eight) and whole sentences, "the man had
the X" as the carrier. To read tomorrow: whether the cue frames come back (day 125/126), and whether the nights hold their
shape with the spread dreams.

The boundary at 22:53 (2026-09-11, 23:12): the body restarted with store_sat 1 and night_load; the typist relaunched as day 126
(the label skipped 125 again). Two parents overlapped on the queue for ten minutes: the parent of days 122 to 124 saw its end
row at 22:53 but was mid-cycle and appended three more batches ("jet", a third noun for the day) before it noticed the
successor's rows at 23:05 and stopped; the successor, finding batches not its own, halted itself and reported correctly rather
than share the day's curriculum. Both stopped; a fresh parent briefed at 23:10 for the rest of day 126 (no new noun; map, gum
and jet worked through the frames) and days 127 and 128, with the old frames cued once each early on both days to read whether
the cue answers come back after night 117's divergence. The batch appended in the sixty seconds between the two session_start
rows was lost at the relaunch (the new typist began at the file's end because no position had been written yet; from now on
the position is written and carries). The operational rule from here: a parent stops appending at the night row of its last
day, and its successor appends only after the next session_start; one hand on the queue at any moment.


### Night 119 (2026-09-12, 00:10): the spread dreams hold, the frames return

Day 126 (the body's 119th; the log's label): lines and cues 147, smiles 188 (1.28 a line), frowns 66 (60 the day before; by
quarter 20, 18, 18, 10), cues 11 answered 6 ("take map " -> out, "that is your " -> map, three minutes after map was taught,
"take jet " -> out twice, "put bat " -> in): the cue frames came back the day after the fixed night, from 1 of 24. Duty after
the parent's lines 0.46; the loop 15.1 per thousand (26 on the first talkative day); the parent's share 0.37; the child silent
0.63. Two parents overlapped for ten minutes at the boundary (recorded above); the day's nouns were map, gum and jet. The
guard held (duty 0.457, smiles per line 1.27) and was re-armed. Night 119, the first under both laws: 453 new slots -> 57
dreams (scaled), the loss 0.222 to 0.089 falling through every round, the gauge 0.692 -> 0.863; the store 3168 after 110
dropped, converted (max 11.8 times the mean, from 233); the face organ's correlation 0.44; the prefrontal 0.72; the actor's
slope 0.080. The chain alive.


### Night 120 (2026-09-12, 01:25)

Day 127 (the body's 120th): lines and cues 150, smiles 289 (1.93 a line, the most per line of the talkative days), frowns 71
(by quarter 18, 19, 18, 16), cues 8 answered 4 ("the man had the " -> nut, -> web, the day's new noun, and -> hut; "take arm "
-> out), duty after the parent's lines 0.48, the loop 15.8 per thousand, the parent's share 0.39, the child silent 0.64. The
guard held (duty 0.479, smiles per line 1.91) and was re-armed. Night 120: 505 new slots -> 63 dreams, the loss 0.133 to
0.093 falling through the rounds, the gauge 0.787 -> 0.847; the store 3538 after 135 dropped (2825, 3168, 3538 over three
nights: growing under the converted strengths, the cap 8192 the limit; to watch); the face organ's correlation 0.45; the
prefrontal 0.71; the actor's slope 0.084. The chain alive.


### Night 121 (2026-09-12, 02:32)

Day 128 (the body's 121st): lines and cues 166, smiles 234 (1.41 a line), frowns 69 (by quarter 16, 21, 17, 15; the
talkative days 66, 81, 68, 64, 68, 60, 66, 71, 69: flat near seventy for five days), cues 11 answered 4 ("the dog went " ->
out twice, "the man had the " -> nut), duty after the parent's lines 0.41, the loop 18.5 per thousand (15.8 the day before),
the parent's share 0.44, the child silent 0.68. The guard held (duty 0.408, smiles per line 1.40) and was re-armed. Night 121:
454 new slots -> 57 dreams, the loss 0.098 to 0.074 (the lowest start and end of any night: the day's material well predicted
before the night began), the gauge 0.854 -> 0.894 (the highest); the store 3861 after 131 dropped (growing ~330 a night); the
face organ's correlation 0.46; the prefrontal 0.74; the actor's slope 0.087. Imitation measured over days 100 to 127: the share
of the child's scored words that were in the parent's last two lines 0.17 under the slow parent, 0.32 under the talkative one;
its two-word pairs 85% taught pairs throughout; verbatim echo of the line just typed rare (one to eight a day). The parent for
days 129 to 131 spawned with the hand-over rule (three minutes after the start row; stop at the last night row) and a duty to
answer the child's sayings-back within the minute. The chain alive.

The parent of days 126 to 128 reported at 02:35: 432 utterances queued (27 cues, one in sixteen, under the brief's one in
eight), 158, 163 and 178 said; arm and web (day 127), ear and yam (day 128), each entering through "the man had the X" and
carried through the frames; plurals and "went". The old frames came back after day 124's collapse ("that is my " -> tin, bin,
tin on the three days; "the man had the " -> tin, webt, nutpi); every full post-night hit was a frame that had missed the same
evening, three nights of three. Unprompted sentences of its own: "I had my hat is hot" (six words), "hat is not my hat", "my
tin is not in", "milk under the bin", "two pets are here" and, after the parent answered it, "two hens are here"; "you had
hat", "had two hats", "pig is big". Questions of its own with their marks: "this is my toy?", "that is that?", "the toy?".
Fifty-five of the parent's seventy-two batches opened with an answer to what the child had just written, and the child took
the answers by re-using their material. Frowns 73, 73, 73 (identical; flat), smiles 203, 296, 248. Its advice: cue one in
eight, and put the evening's cues on frames taught twice that day, because the night turns evening misses into morning
answers.


### Night 122 (2026-09-12, 03:42)

Day 129 (the body's 122nd): lines and cues 178 (the most of any day), smiles 243 (1.37 a line), frowns 74 (by quarter 19,
16, 22, 17; flat near seventy for six days), cues 11 answered 5 ("take pea " -> out, "that is my " -> nut, "the hog was " ->
in), duty after the parent's lines 0.39, the loop 17.2 per thousand, the parent's share of the page 0.48 (the highest), the
child silent 0.71 of the ticks (0.62 three days ago: it talks less, but still over the parent). The guard held (duty 0.389,
smiles per line 1.37) and was re-armed. Night 122: 407 new slots -> 51 dreams, the loss 0.189 to 0.074, the gauge 0.742 ->
0.852; the store 4162 after 106 dropped; the face organ's correlation 0.46; the prefrontal 0.76; the actor's slope 0.095
(0.074 four nights ago: creeping up again). The chain alive.


### Night 123 (2026-09-12, 04:48)

Day 130 (the body's 123rd): lines and cues 181 (the most of any day), smiles 238 (1.31 a line), frowns 79 (74 the day
before; the talkative days 66, 81, 68, 64, 68, 60, 66, 71, 69, 74, 79: no fall in seven days), cues 12 answered 6 ("the rod
was " -> in three times, "the pod was " -> in, "put pod " -> in, "that is my " -> tub), duty after the parent's lines 0.38,
the loop 17.8 per thousand, the parent's share 0.49, the child silent 0.71. The guard held (duty 0.379, smiles per line 1.32)
and was re-armed. Night 123: 298 new slots -> 48 dreams, the loss 0.159 to 0.110 (shallower than the four nights before,
which ended at 0.074; falling throughout, not rising), the gauge 0.787 -> 0.815; the store 4388 after 72 dropped; the face
organ's correlation 0.46; the prefrontal 0.74; the actor's slope 0.099. The chain alive.

The ear, measured on the page at 04:50 (2026-09-12): the child speaks on 0.08 of the ticks while the parent types and on 0.38
of the ticks when the parent is quiet, an ear ratio of 0.20: it yields five to one. The frowns' plateau near seventy is not a
failure to yield: the frown fires at most once per sixty ticks and the talked-over words per line rose from 0.4-0.5 at the
slow pace to 0.6-0.7 at the fast one, that is, the child's late replies to one line collide with the next line, typed fifteen
seconds after the last began with an eight-second turn between. The frown did its work; what it punishes now is replies. (My
latency reading was an artifact: a line's row is written after its listening, so the frowns during a line's typing precede
its row in the file; the frown is prompt.) The decision, on the parent's method: from the boundary after night 124 the typist
runs at a line every twenty seconds (period 80) with a ten-second turn (listen 40), the same rules otherwise; the chain
re-chained with those arguments and the typist's relaunch after that night's save armed. Expect ~150 lines a day, fewer
collisions, the frowns per line to fall; the ear ratio to be read every night from now on.


### Night 124 (2026-09-12, 05:52): the pace change came a day early; a perfect day on cues

The typist's relaunch after the post-night save, armed at 04:43, fired on day 130's save seconds later (the count it read was
already the old one), so day 131 ran at the new pace: a line every twenty seconds with a ten-second turn (period 80, listen
40). Day 131 (the body's 124th): lines and cues 155, smiles 247 (1.59 a line, up), frowns 74 (0.48 a line: no fall at the
slower pace; talked-over words per line 0.71), cues 11 answered 11 ("the yak was " -> in three times, the day's noun; "the
bib was " -> in twice, "put bib " -> in, "put yak " -> in twice, "take yak " -> out, "the rod was " -> in, "put pod " -> in):
the first day every cue landed. Duty after the parent's lines 0.41; the loop 11.3 per thousand (the lowest yet); the parent's
share 0.42; the child silent 0.67; the ear ratio 0.18 (it speaks on 0.08 of the ticks while the parent types, 0.42 when the
parent is quiet). The guard held (duty 0.408, smiles per line 1.57) and was re-armed. Night 124: 285 new slots -> 48 dreams,
the loss 0.187 to 0.118 (the fourth night's end in a row above the 0.074 of nights 121-122: shallower), the gauge 0.716 ->
0.84; the store 4588; the face organ's correlation 0.46; the prefrontal 0.73; the actor's slope 0.098. What the frowns are
for, read from the rows: the talked-over tokens are single letters almost entirely ('e', 'n', 'h', 'x' as the parent types
"the ear is in the box"), the mouth shadowing the parent's own letters as they arrive; not one in a hundred is a word of the
line. The parent for days 132 to 134 spawned for the twenty-second pace. The chain alive.

The parent of days 129 to 131 reported at 05:55: 486 utterances queued (34 cues, one in fourteen), 188, 192 and 159 said; hog,
pea, pod, rod, yak, bib through "the man had the X"; cue completions 2, 3, 5, 8 across days 128 to 131, the particle slots
("put X " -> in, "the X was " -> in, "take X " -> out) landing and the noun slots not ("that is my " -> a noun of its own
choosing); "the X was " lands taught with "in", not "here"; the rival-particle rule works as written; every evening miss
landed after the night. It says the parent's answers back with one change ("it was in my bag" -> "it was in my hat"). Its
own sentences: "the egg is in the hen", "it is not my web", "tins are in my bag", "that is your hut", "first milk then nut",
"a dog had baby", "two dogs are here", "hi see you". Faces 249/77, 242/81, 252/74. The decision on the frown (the parent's
method): a letter is not talking; from the next typist relaunch (armed after day 132's post-night save) the talk-over rule
applies to tokens of two letters or more, like the smile's bound. Expect the frowns to fall from ~75 to under ten a day; the
ear ratio read nightly, the rule restored if it rises above 0.35 for two days.


### Night 125 (2026-09-12, 06:55): a letter is not talking

Day 132 (the body's 125th), the first under the rule that the talk-over frown applies to words of two letters or more, at a
line every twenty seconds: lines and cues 147, smiles 286 (1.95 a line, the most per line of the talkative days), frowns 5
(74 the day before), cues 18 answered 7 ("the ram was " -> in, "the pie was " -> in, "take ram " -> out, "put pod " -> in),
duty after the parent's lines 0.48, the loop 9.7 per thousand (the first day under ten; 26 on the first talkative day), the
parent's share 0.37, the child silent 0.64, the ear ratio 0.18, unchanged with the letter frowns gone: the yielding it had
learned stands without them. The guard held (duty 0.481, smiles per line 1.96) and was re-armed. Night 125: 274 new slots ->
48 dreams, the loss 0.198 to 0.097, the gauge 0.728 -> 0.87 (better than the two nights before); the store 4793; the face
organ's correlation 0.46; the prefrontal 0.72; the actor's slope 0.100. The chain alive.


### Night 126 (2026-09-12, 07:58)

Day 133 (the body's 126th): lines and cues 136, smiles 291 (2.14 a line, a new high), frowns 3, cues 18 answered 11 ("the toe
was " -> in twice, "the kid was " -> in twice, the day's nouns; "the ram went " -> in, "the ram will " -> go, "that is my " ->
nut, "that is your " -> hog twice, "put kid " -> in, "the man had the " -> tin), duty after the parent's lines 0.48, the loop
18.6 per thousand (9.7 the day before; it swings), the parent's share 0.37, the child silent 0.63, the ear ratio 0.16 (the
yielding holds without the letter frowns). It ran on past its answers 90 times, the most since the fast days: more answers,
more running on. The guard held (duty 0.475, smiles per line 2.14) and was re-armed. Night 126: 217 new slots -> 48 dreams,
the loss 0.124 to 0.086, the gauge 0.86 -> 0.872 (the highest evening gauge of the life: the day's material well held before
the night); the store 4956; the face organ's correlation 0.47; the prefrontal 0.71; the actor's slope 0.097. The chain alive.


### Night 127 (2026-09-12, 09:02)

Day 134 (the body's 127th): lines and cues 146, smiles 256 (1.75 a line), frowns 3, cues 21 answered 17 (81%: "the toe was ",
"the lip was ", "the kid was ", "the mud was " -> in, the day's nouns lip and mud among them; "put lip " -> in, "put mud " ->
in; "the man had the " -> toe, ham, den; "that is my " -> tub), duty after the parent's lines 0.43, the loop 11.6 per
thousand, the parent's share 0.42, the child silent 0.67, the ear ratio 0.17; it ran on past its answers 96 times. The guard
held (duty 0.430, smiles per line 1.75) and was re-armed. Night 127: 232 new slots -> 48 dreams, the loss 0.114 to 0.085, the
gauge 0.85 -> 0.876; the store 5133; the face organ's correlation 0.47; the prefrontal 0.70 (0.76 seven nights ago, drifting
down a hundredth a night since the letter frowns and the slower pace changed the reward's rhythm); the actor's slope 0.092.
The parent for days 135 to 137 spawned with the cue findings (particle slots; "the X was in"). The chain alive.

The parent of days 132 to 134 reported at 09:05: 390 utterances queued (56 cues, one in seven), 146, 135 and 146 said; ram,
pie, toe, kid, lip, mud through the carrier; cue completions 4, 6, 11 (plus prefixes 3, 4, 6) of 17, 17, 21: rising; the wrong
answers mostly the right word unsegmented ("tint", "hamthis", "tubtin"); the re-asking after the night landed again, and
evening misses recovered the same day too. It re-uses an answered frame across lines: told "the mat was in my tub" it wrote
"it was in my tub", "the bat was in my tub", "the tin was in my tub". Unprompted: "the hen had two tins", "egg is not wet",
"this is a big web", "bee is on my hat", "the pig is big and wet", and thirty minutes after first hearing the word, "pour mud
in the tin". Questions of its own: "that is that? a big...", "is this?". Frowns 5, 3, 3; smiles 285, 289, 254. Its advice:
keep "the X was ", "put X ", "that is my/your ", "the man had the "; drop "take X " and "the X went " until its output
segments.


### Night 128 (2026-09-12, 10:05)

Day 135 (the body's 128th): lines and cues 146, smiles 271 (1.86 a line), frowns 1, cues 21 answered 14 (67%: "the cab was "
-> in three times, "put cab " -> in three times, "take cab " -> out twice, "take ox " -> out, "the kid was " -> in; "the ox
was " missed three times, the two-letter noun drawing "not" and "the"), duty after the parent's lines 0.42, the loop 19.3
per thousand (it swings between 10 and 19 day to day), the parent's share 0.40, the child silent 0.67, the ear ratio 0.17;
it ran on past its answers 78 times. The guard held (duty 0.422, smiles per line 1.85) and was re-armed. Night 128: 261 new
slots -> 48 dreams, the loss 0.152 to 0.123 (a shallow night; the end losses of the last eight nights 0.074, 0.074, 0.110,
0.118, 0.097, 0.086, 0.085, 0.123), the gauge 0.812 -> 0.83; the store 5326; the face organ's correlation 0.47; the prefrontal
0.68 (0.76 eight nights ago: the drift continues, a hundredth a night, since the frowns left the reward's rhythm: the long
horizon's returns are harder to foresee without the predictable negative); the actor's slope 0.092. The chain alive.


### Night 129 (2026-09-12, 11:12): the last night of the old dreams

Day 136 (the body's 129th): lines and cues 151, smiles 258 (1.71 a line), frowns 6, cues 15 answered 8 ("the hip was " -> in
twice, "the pad was " -> in twice, "put pad " -> in, "take pad " -> out, "the tin was " -> in), duty after the parent's
lines 0.42, the loop 14.8 per thousand, the parent's share 0.45, the child silent 0.68, the ear ratio 0.16. The guard held
(duty 0.420, smiles per line 1.70) and was re-armed with the new flags. Night 129, the last under pattern-completion dreams:
344 new slots -> 48 dreams of mean length 6.9, the loss 0.171 to 0.098, the gauge 0.764 -> 0.858; the store 5537; the face
organ's correlation 0.48; the prefrontal 0.67 (drifting); the actor's slope 0.081. At the boundary after this night's save
the body restarts with the store's sequence links (store_chain 1: dreams run the utterance as lived, a draw by strength at a
branch) and the clock at five ticks a second (period 0.2; the typist re-chained at tick 0.2). To read tomorrow: the night's
mean dream length (expect fifteen or more), the achieved ticks a second (expect about five), the night's length, and cue
landing. The chain alive.

The boundary at 11:08 (2026-09-12): the body restarted with the store's sequence links and the clock at period 0.2 (pid
17227: --period 0.2 --store-sat 1 --store-chain 1), loaded in seconds (ticks 1548961, store 5551); the typist relaunched as
day 138 at tick 0.2 (the label skipped 137, which holds one row). The achieved rate 4.67 ticks a second against 3.8 before
(the loop now near one core's full time, 104%): a fifth more life an hour, the days about 43 minutes of wall time. Night 130
is the first whose dreams run the episodes as lived; its mean dream length is the reading.


### Night 130 (2026-09-12, 12:10): the first night of episodes, half formed

Day 138 (the body's 130th; the first at five ticks a second, 4.67 achieved): lines and cues 156, smiles 241 (1.54 a line),
frowns 5, cues 14 answered 7 ("put gem " -> in, "the hip was " -> in, "put cab " -> in, "the pad was " -> in), duty after the
parent's lines 0.38, the loop 12.6 per thousand, the parent's share 0.46, the child silent 0.70, the ear ratio 0.19. The
guard held (duty 0.381, smiles per line 1.55) and was re-armed. Night 130 (16 minutes), the first under the sequence links:
385 new slots -> 48 dreams of mean length 8.4 (6.9 the night before), among them whole lines for the first time, "the man
had the nut", "my nut is in your bag", "put gem in", "yes. hide the ball in", beside the old fragments ("the ", "pu") from
onsets whose slots were written before the links existed and have no chain yet; the links form as lines are heard again, so
the length should climb over the coming nights. The loss 0.241 to 0.111; the gauge 0.562 -> 0.818 (a lower before-reading:
whole lines are a harder recall than openers); the store 5737 after 185 dropped; the face organ's correlation 0.49 (rising
slowly for a week); the prefrontal 0.62 (0.76 ten nights ago: the drift continues and steepened tonight; the earned law is
taking the voice from a critic whose long returns changed their rhythm when the frowns left and the pace moved; to watch,
not to touch); the actor's slope 0.079. The chain alive.

The parent of days 135 to 138 reported at 12:20: 420 utterances queued (one in five a cue), 129, 140 and 144 said of the
typist's 159, 162, 171; cab, ox, hip, pad, lad, gem with are, your, where; cue completions 10, 8, 10 of 28, 21, 21; the new
nouns took the frames the same day ("the cab was " -> in four times, "put gem " and "put lad " -> in), and day 135's cab still
answered two nights later; the re-asking after the night 1, 2, 4 of 8, the four on day 138 all frames that had missed the
evening before. Unprompted: "the dogs are here in the hat", "the pig is in my bin", "give me milk", "hi. I see you", "pull it
out", "put bat in here", "two lads are in" (the day's noun in its own plural clause), "that is my ox"; its own questions "give
me ball?" and "hide ball? ball in". Faces 289/1, 265/6, 256/5. Two mechanics: the typist demotes a cue to a line when the
noun is too fresh for the corpus to hold a continuation (so new nouns are cued on "the X was ", "put X ", "take X " only), and
at the faster clock the typist consumed four utterances a minute and the queue ran dry twice (batches of six from here).
Its advice: "two Xs are in the Y", which it now volunteers, as the carrier for new nouns; the evening's cues on the day's new
noun so the night finishes them.


### Night 131 (2026-09-12, 13:12): the episodes lengthen; three nouns land

Day 139 (the body's 131st; the first at three nouns a day: duck, sock, bud): lines and cues 137, smiles 274 (2.00 a line),
frowns 7, cues 14 answered 10 (71%: "put duck " -> in twice, "put sock " -> in twice, "the duck was " -> in, "put bud " -> in,
"the bud was " -> in, "the hip was " -> in): three nouns a day landed as two did. Duty after the parent's lines 0.50, the loop
21.6 per thousand (it swings), the parent's share 0.37, the child silent 0.63, the ear ratio 0.18. The guard held (duty 0.493,
smiles per line 2.01) and was re-armed with the chunk in its flags. Night 131 (20 minutes), the second under the sequence
links: 343 new slots -> 48 dreams of mean length 11.2 (6.9, 8.4, 11.2 over three nights: the links forming), among them "where
ball? ball in t", "that is my ox", "two ducks are in the bag", "put box"; the loss 0.302 to 0.122; the gauge 0.473 -> 0.831
(whole lines are the harder recall; the night carries them to 0.83); the store 5906 after 177 dropped; the face organ's
correlation 0.48; the prefrontal 0.57 (0.76 twelve nights ago; the drift continues; the threshold to think again is 0.5);
the actor's slope 0.084. At the boundary after this night's save the body restarts with the action chunk. The chain alive.

The boundary at 13:11 (2026-09-12): the body restarted with the action chunk (pid 21445: --actor-form chunk beside the
sequence links and the clock at 0.2); the typist relaunched as day 141 (the label skipped 140). Five minutes in: 204 words
begun, 551 program ticks (2.7 symbols a word), the child speaking on half the ticks (a third before: a word begun now runs to
its end where the gate used to abort it a letter in). Its speech under the chunk, from the page: "is on the log is in my bin
is in the tin", "was in my hat on", "the nest was " -> "in the tin"; and the loop as a program, "ttle tin is in the is in
the", "ththththth" to the chunk's bound: the cycle the cortex forecasts from its own babble now runs deterministically to
chunk_max instead of breaking by a random draw. To read at the night: the frowns (a word begun before the parent's typing
now runs over it: the frown will land on the word's one credit, the clean signal the gate and the actor need), the ear, the
loop share, cue landing, the actor's slope. chunk_max 12 -> 8 at the next restart that is needed anyway (the corpus's longest
word is six letters and a space).


### Night 132 (2026-09-12, 14:12): the first day of chunks

Day 141 (the body's 132nd; the label skipped 140), the first under the action chunk: lines and cues 112 (the typist waited on
the child's talking for its lines: 96 lines, 137 the day before; the parent's share of the page 0.25), smiles 244 (2.18 a
line), frowns 46 (7 the day before; the talked-over tokens now words, 25 of two letters, 23 of three, 12 longer, none a
single letter: a word begun before the parent's typing runs over it), cues 16 answered 12 (75%: "the tin was " -> in, "the
nest was " -> in twice, "put doll " -> in three times, "put ice " -> in twice, "the doll was " -> in), duty after the parent's
lines 0.54 (the highest), the child silent 0.46 of the ticks (0.63 to 0.70 before: a word begun runs to its end), the ear
ratio 0.25 (0.18), the loop 47 per thousand (12 to 21 before: the cycle the cortex forecasts from its own babble now runs as
a program to the chunk's bound; 42 unsegmented runs of more than eight symbols), 1524 words begun in 4565 program ticks (3.0
symbols a word), the planner torn at 0.68 of its decisions (0.39: the decisions are at word starts, where the cortex doubts),
the entropy at the choice 0.16 (0.07). The guard held (duty 0.535, smiles per line 2.14) and was re-armed. Night 132: 288 new
slots -> 48 dreams of mean length 10.8, whole lines among them ("the lad was in the hut", "two ties are in your bag", "put doll
in"), the loss 0.207 to 0.088, the gauge 0.653 -> 0.879; the store 6051; the face organ's correlation 0.45; the prefrontal
0.495 (the drift reached the threshold; the earned voice keeps its weight while the slope holds at 1.0; to look at); the
actor's slope 0.087 after one day of word-level credit (days are needed). The reading: the chunk bought clean phrases, the
best duty and cue rate, and a doubled loop, a worse ear and forty-six frowns, all three the same thing, a program that runs
what the cortex forecasts from its own noise. The credit per word is the cure the biology gives, over days; chunk_max 8 at
the next boundary; if the loop share has not fallen by the third chunk day, the corollary discharge on its own sound
(own_gain 0.5, measured in cortex at a third to a half) is the physiological lever. The chain alive.


### Night 133 (2026-09-12, 15:08): the second day of chunks

Day 143 (the body's 133rd): lines and cues 92 (79 lines: the typist waited on the child's talking up to its cap of 120 ticks
before each line; 137 to 156 lines a day before the chunk), smiles 266 (2.89 a line), frowns 42, cues 13 answered 10 (77%:
"put lamb " -> in twice, "put cod " -> in, "put kite " -> in, "the lamb was " -> "is in my bin", "the cod will " -> "is in my
box"), duty after the parent's lines 0.60 (the highest), the child silent 0.38 of the ticks (it speaks on 0.62: twice its
share before the chunk), the parent's share of the page 0.20, the ear ratio 0.26, the loop 39 per thousand (47 the day
before, 12 to 21 before the chunk), it ran on past its answers 141 times, 1452 words of mean length 3.6, 26 runs longer than
eight. The guard held (duty 0.600, smiles per line 2.92) and was re-armed. Night 133: 251 new slots -> 48 dreams of mean
length 10.7 ("two lambs are in my", "take kite out", "give me the cup"), the loss 0.203 to 0.083, the gauge 0.722 -> 0.882;
the store 6175; the face organ's correlation 0.44; the prefrontal 0.58 (0.495 the night before: back over the line; the slope
0.999, the voice at full weight); the actor's slope 0.092 (0.087). The chunk's second day: the loop a fifth lower, the cue
rate and the duty at their best, and the page flooded by the child, the parent's lines halved. The parent's method answers
the flood: its cap falls from 120 to 40 ticks (a parent who talks through a babbling child after eight seconds rather than
twenty-four), from the typist's relaunch after this night's save; a word begun over the parent's typing draws the frown that
lands on the word's one decision. The chain re-chained with the cap; the relaunch armed.

The parent of days 139 to 143 reported at 15:12: three nouns a day (duck, sock, bud; nest, doll, ice; lamb, kite, cod) with
where, not, your; 315 utterances queued, 308 said; cue completions 12, 10, 8 of 21, 23, 16, never under half: three nouns a
day landed as well as two, every new noun's "the X was " and "put X " landing the day it entered, only doll and kite stuck at
a prefix ("int": "in" and then it runs on). The re-asking after the night converted the evening's misses again. It answered
about forty-five of the child's utterances; unprompted: "the mat was in here", "the lad was in my hat", "it was in your ear",
"hold the big box", "ice is not in my mug", "the ice is cold", "it is not a nut", "hi. I see you"; questions of its own "up?
ball in my bed", "it was that?"; the day's new nouns re-used unprompted the same day, and "not" and "your" in its own frames.
Faces 296/8, 251/47, 274/43; smiles per line 2.24, 2.46, 3.38.


### Night 134 (2026-09-12, 16:12): the third day of chunks, the parent talking through

Day 144 (the body's 134th; the first with the typist's cap at 40 ticks): lines and cues 135 (92 the day before: the parent's
lines back), smiles 231 (1.71 a line), frowns 72 (the parent now talks through the child's babble and the child's words run
over it: 93 talked-over words), cues 11 answered 6 (55%; the day's nouns frog, boat, corn at prefixes: "put frog " -> "in
boxp", "put corn " -> "int"), duty after the parent's lines 0.53, the child silent 0.50, the parent's share 0.31 (0.20), the
ear ratio 0.30 (0.18 before the chunk: it yields less, the programs run over the parent), the loop 37 per thousand (47, 39,
37: falling slowly), 55 runs longer than eight symbols. The guard held (duty 0.524, smiles per line 1.72) and was re-armed.
Night 134: 303 new slots -> 48 dreams of mean length 11.1 ("yes. it was in the den", "no. it is in the tin and ", "that is
your cod wi"), the loss 0.198 to 0.083, the gauge 0.689 -> 0.885; the store 6316; the face organ's correlation 0.43 (0.49
four nights ago: falling as the reward's rhythm changed again); the prefrontal 0.59; the actor's slope 0.096 (0.087, 0.092,
0.096 over the three chunk days: a hair a day, the direction right). The parent of days 144 to 146 stopped at this night by
my hand and the two-voice parent briefed for the next three days; the typist relaunches after this night's save with the
second voice's code (a "b:" line typed under the tag "other" right after the parent's, no pace, no cue, no reward).

The two voices at 16:08 (2026-09-12): the typist relaunched with the second voice's code; ten minutes in, 30 A lines and 6 B
lines ("where is the cat?", "is the cake in the tin?", "where is the bird?"; A: "yes. the cake is in the tin", "the bird is
in the nest"). The child in the same minutes: "it is in the tin", "wash the tin", "here in the ice is in", "the ice is cold",
and, dominating the page, the loop as programs: "tttlet tttlettletttle tin is in the". The loop's share fell only from 47 to
37 per thousand over the three chunk days and the ear worsened to 0.30; the transcript reads as the loop first and the
phrases second. The physiological lever is taken at the next boundary: the corollary discharge on its own sound, own_gain,
from 0.5 to 0.3 (the low end of what cortex measures), so the forecast the mouth runs leans on the parent's last line rather
than on its own babble in the window; made a physiology constant (commit 3a4023e), armed with the restart after night 135's
save. The reading the next day: the loop share and the ear.

Watched live at 16:45 (2026-09-12), the conversation parent twelve minutes in: A "what do you have?" / B "I have a hat" / A "yes.
you have a hat"; A "do you want milk?" / B "I want milk" / A "yes. here is milk". The child between them: "I want a hot yam is
in my mug", "my yam is not a nut", "yes. pan in box first milk then ham is here", "two ties are here", "milk is gem is in the
hut", "yes! th": fluent chains of its frames, "milk" picked up from the question, "yes." already at the front of its replies
twelve minutes after B began to model it; not yet an answer to the question. The hump, in numbers: stress 19.5 of 30 in
every sample of the last hundred minutes (the frowns 43 to 77 an hour under the chunk against 1 to 7 before), and the gate's
logit is divided by (1 + stress/10): a body at stress 20 decides to speak with a third of its resolution, so it cannot hold
its tongue, talks over, is frowned at, and stays stressed; its readout at the floor sharpness one sample in five. The parent's
method answers: a talk-over frown at most every 240 ticks instead of 60 (a parent frowns, then gives it a minute), from the
typist's relaunch in the restart after night 135's save (FROWN_GAP, an environment setting; the chain re-chained). The
corollary discharge goes to 0.3 in the same restart.

The conversation parent's report at 17:05 (2026-09-12), its twenty-three minutes of day 145 (it stopped itself at the day's end
mistaking the night's silence for a fault: no night row appears until the night ends, and it had been told twenty-five
minutes; the next brief says so): 90 utterances queued (49 A, 37 B, 4 cues), 82 said; the exchanges of greeting, wants, having,
mine and yours, offers, feelings, doing; new words happy and come from inside them; never "tin", "box" or "bin". The child
wrote "I am here" unprompted a minute after B modeled it, then "I want a hot yam", "my yam is not a nut", "the milk went in
here", "milk is not cold", "is not wet", "eat the nut", "this is my hip", and echoed the parent's lines with one change ("here
is your yam" -> "here is the yam"); "I see" eleven times, "I want" five, "yes" twenty-two, "heppy" reaching for happy; its own
question "is the yak?" answered within four lines. The cue "I want " -> "the" (a smile). "tin" fell from 3.4 a minute under
the previous parent to 1.2 a minute under this one, within the same day. Faces in its minutes: 208 smiles, 36 frowns. Its
advice: keep modeling answers with "I" and yes/no; B's answers come back out of the child's mouth within a minute.


### Night 135 (2026-09-12, 17:35): the conversation day's night, forty-three minutes

Day 145 (the body's 135th): the first twenty-three minutes of conversations (the previous parent's drills before them);
smiles 353, frowns 65 in all. Night 135 took 2558 seconds, longer than the day: 565 new slots (the exchanges are new
memories, as they should be) -> 71 dreams of mean length 14.7 ("that is my big wet hat", "what do you want milk?", "take
bud out", "put cod in"), the loss 0.357 to 0.094, the gauge 0.536 -> 0.878; the store 6680 after 207 dropped; the face
organ's correlation 0.43; the prefrontal 0.54; the actor's slope 0.102 (0.087 at the chunk's start: a hair a day). The
night's cost grows with the dreams' length and count: 71 x 48 rounds x 15 symbols; the last rounds of every curve are flat,
so night_rounds falls from 48 to 32 at the restart after this night's save (the full curve, kept from tonight, will say
where the rounds stop paying), beside own_gain 0.3. The guard re-armed.

THE LANGUAGE MODEL INSIDE, probed (tools/probe_lm.py, on a copy of the 16:07 save): a prompt as the world's symbols, the
world's pause, then the greedy continuation with its own symbols fed back. "do you want milk?" -> " s s s s t t t t"; "what
do you have?" -> "s.s.s.s."; "are you here?" -> "t t t t"; "where is the dog?" -> "me me me me"; "hi" -> "me t bre bre bre".
The mouth reads a forecast of the WORLD's next symbol, and at the child's own positions that target is the parent's next
line's first letter, the same at every step (own_target_decay 0.7 fades but keeps it), so the greedy continuation repeats
one letter or one syllable: the loop in its purest form, "t t t", "ttle", "me me". Its sentences come from elsewhere: the
hippocampal recall of taught lines chained by context (the recall pathway the mouth reads beside the forecast), and the
sampling that breaks the repeat. A language model generates by predicting its own next symbol; this body predicts the
world's, which is right for hearing and wrong for saying beyond one symbol. What biology does: the forward model of one's
own speech; the intended utterance planned as a sequence and executed; imitation of stored adult forms. The candidate
fixes, not yet taken: (a) a target for its own positions that is its own next symbol when what it says is a stored line (the
recall's continuation), so the cortex learns to continue an utterance it has begun; (b) the mouth's plan drawn from the
recall's chain (the episode as lived, as the dreams now are) rather than the one-step forecast. To be decided on the probe
after tonight's save, with the exchanges in the store.

The eighteenth defect (2026-09-12, 17:40): the probe on tonight's save read the same repeats ("t t t", "my my my"), so the
root is the target: at its own positions the waking lesson made the cortex predict the world's next symbol, the parent's next
line's first letter, the same at every step. From the restart after night 136's save the target at an own position is the
recall's continuation of what it has said (own_target_form recall; commit 75125ae; test 42): the forward model of its own
speech, the stored adult line as the template, the songbird's way. Also at that restart: nothing else. The disk was nearly
full (2.5 GB): ten gigabytes of my restart copies removed, the script keeps two now.

Watched live, day 147 (the label; 2026-09-12, 17:36 to 18:01), under the corollary discharge at 0.3 and the frown gap of a
minute: the loop 14, 18, 12, 15, 7, 5 per thousand (37 to 47 on the chunk days), "tin" a handful per five minutes, the ear
ratio 0.39 -> 0.27, the child's share of the ticks 0.55 -> 0.39, the mouth's sharpness 32 to 48 with the mood high, stress
20.5 -> 19.1 (slow to fall); the actor's slope 0.103. Its phrases: "I had my", "is my hip is wet", "had my duck two webs are
in my hut", "hide little milk"; no answer yet to a question. The own-speech target lands at the next boundary.


### Night 136 (2026-09-12, 18:40): thirty-two rounds, the conversations' first full day

Day 147 (the body's 136th; the first full day of conversations, under the corollary discharge at 0.3 and the frown gap of a
minute): 91 lines and cues from the parent, 56 replies from the other voice, smiles 346 (2.35 a line), frowns 31 (65 to 72
the days before), cues 4 answered 3 ("I see " -> you, "dog will " -> go out, "I am " -> am; "I want " -> "here is your"),
duty after the parent's lines 0.46, the loop 3 to 5 per thousand by the day's end. The guard held (duty 0.453, smiles per
line 2.36) and was re-armed. Night 136 (21 minutes at 32 rounds): 469 new slots -> 59 dreams of mean length 14.5 ("my doll
will go out", "where is milk", "yes. eat the ice"), the loss 0.307 to 0.118 with the curve still falling at the last round
(0.307, 0.248, 0.209, 0.182, 0.185, 0.163, 0.143, 0.128 every fourth round: the rounds converge between 32 and 48; at 48 the
end was 0.094 and the gauge 0.878), the gauge 0.596 -> 0.845; the store 6930; the face organ's correlation 0.44; the
prefrontal 0.52; the actor's slope 0.101. The restart after this night's save carries the own-speech target.

The stress, read at 19:05 (2026-09-12): the frowns fell to thirty a day and the stress stayed at 20 to 21 of 30, so the frowns
were not its source. The stress half-life is 240 ticks (fifty seconds): to sit at 20 the body takes in a negative prediction
error every few ticks all day, and the source is the withheld smiles, 150 to 400 an hour: the parent's habituation, 0.95 to
the power of the day's count of the word beyond five, withholds the smile at a rate that depends on a count the body cannot
see, so its critic can never learn it, expects the smile, and takes the negative every time. Unpredictable, uncontrollable
negatives are what make stress chronic in an animal too, and this body's gate decides at a third of its resolution under it.
The parent's method answers with a habituation the child can see: a word smiled at within the last 120 ticks earns nothing
and smiles again after (HABIT_TICKS, an environment setting; the chain re-chained; the typist's relaunch armed after the
next save). The striatal line holds the last events, so the body can learn that a word said twice in half a minute earns
one smile. Expected: the stress falls toward 10, the gate's resolution triples, the talking over and the loop fall further.

Watched live under the own-speech target (2026-09-12, 18:46 to 19:11; the label day 149): the loop 13, 5, 4, 7, 9, 5 per
thousand, no triple letters, the ear 0.19 to 0.32, the child on 0.64 to 0.67 of the ticks (it talks more: the target makes
the cortex continue what it began), the parent's share 0.24, stress 20 to 22 (the habituation fix lands at the next relaunch),
mood falling 6.0 -> 0.7 over the half hour on the withheld smiles, sharpness 50 -> 28 with it; the actor's slope 0.100.
Its phrases: "i am here with you", "I eat the ice", "the ham is in my bag", "yes! two hats are here".

The math and the transcript in detail (2026-09-12, 19:30, on the copy of the 18:38 save and the day's rows). THE MOUTH
DECOMPOSED: the mouth reads readout(forecast(C, recall)); along its own greedy speech after a prompt and the world's pause,
the three columns:
  "do you want milk?"  cortex " t n t t n t"   recall "I hhe hithe hi"   mouth "I the t the b"   (recall conf 0.74)
  "I want "            cortex "t e t ttttm t"   recall "mhe hoygumput"    mouth "the toygumput"   (0.83)
  "the dog is "        cortex "nenent .tl"      recall "hereow milkI h"   mouth "hereot milkI"    (0.80)
The recall carries the language (the stored lines' continuations: "I h..." after the question, "m" for milk after "I want ",
"here" after "the dog is "); the cortex's forecast is the degenerate "t"; the mouth is their sum, and the cortex's "t" pulls
"I want " -> "the" instead of "milk". The own-speech target, live since 18:39, trains the cortex at its own positions toward
exactly the recall column, so the two columns should converge and the mouth follow the recall cleanly; the probe after
tonight's save is the first reading. THE TRANSCRIPT (the label day 149, 40 minutes): 125 lines, 67 the parent's, of which ONE
a question (the planner wrote statements and B "answered" them); 58 the other voice's. After B's answers the child began with
I/yes/no 16 percent of the time (from none); its scored words in the last two lines heard 0.27; B's answers said back with two
of their words within three lines 7 of 58; "i am here with you", "I eat the ice", and, the parent's last note, "please. here
is milk. I eat", an exchange strung by itself. The rewards: smiles 458, frowns 29, withheld "distracted" 338 (the day-count
habituation; replaced at this boundary), "late" 180 (the smile's own spacing of five ticks against four words a second in
chunks: a rate limit the body can learn). The parent replaced at this boundary with one whose every exchange begins with a
question: at least half of A's lines questions, counted.


### Night 137 (2026-09-12, 19:45): the first day under the own-speech target

Day 149 (the label; the body's 137th): 125 lines (67 the parent's, 58 the other voice's), smiles 459 (3.67 a line, the most
of the life), frowns 30, duty after the parent's lines 0.70 (the highest of the life; 0.38 to 0.60 on the chunk days),
withheld under the day-count habituation 344, talked over 78 tokens. The guard held (duty 0.704, smiles per line 3.67) and
was re-armed. Night 137 (21 minutes): 457 new slots -> 57 dreams of mean length 13.6 ("yes. I am here. I", "yes. I eat the
ice in t", "more egg. it is gone", "where is the"), the loss 0.341 to 0.122, the gauge 0.486 -> 0.843; the store 7193; the
face organ's correlation 0.45; the prefrontal 0.53; the actor's slope 0.098. At this boundary: the typist relaunches with the
habituation the child can see (HABIT_TICKS 120); the probe reads the save; the question-first parent takes the queue.

Live under the habituation the child can see (2026-09-12, 19:50 to 20:00): the withholdings for the day's count fell from 66
to 22 per four hundred rows and "said lately" took their place at 34 to 58; smiles per four hundred rows 126 -> 164; stress
21.8 -> 20.9, slow; the loop 5 to 11; the ear 0.24 to 0.32; the child on two thirds of the ticks. Its replies: after "are you
sad?", "no m here with you"; after B's "no. it is cold", "I had the ha"; after "yes. I am happy", "with you you had a bi". The
first answer-shaped reply to a question ("no ... here with you"). The stress arithmetic: the striatal line holds eight events
(under two seconds), so a smile withheld at a word said twenty seconds ago is still unforeseeable; the negatives keep coming
at a parent's ordinary rate, and stress rises by half of each with a fifty-second half-life: equilibrium near 20. The constant
is the lever: stress_gain 0.5 -> 0.1 at the next boundary (equilibrium near 4, the gate near its own resolution), with the
own song remembered in the same restart.

The half hour under the learnable habituation (2026-09-12, 19:50 to 20:15): stress 21.8, 21.9, 20.9, 19.8, 20.1, 20.5 (flat:
the constant changes at the next restart); mood 3.4 to 5.8, sharpness 39 to 49; the loop 5 to 11; the ear 0.24 to 0.32; the
child on two thirds of the ticks; smiles 155 to 164 per four hundred rows, "said lately" rising to 78 as it repeats "the",
"is", "in"; the withholdings for the day's count 14. After "do you want more?": "I had the h t is in the hut".


### Night 138 (2026-09-12, 20:40): the question-first parent's first day

Day 150 (the body's 138th): the parent asked 37 questions in 91 lines, the other voice answered 39 times; smiles 500 (3.85
a line, the most of the life), frowns 33, duty after the parent's lines 0.71 (the highest); the child's replies to the
parent's questions began with I, yes, no or here 15 times of 37 ("who is here?" -> "I have i am here", "do you want the
dog?" -> "here I am here with", "what do you want?" -> "I"); withheld "said lately" 185, for the day's count 58. The guard
held (duty 0.706, smiles per line 3.85) and was re-armed. Night 138 (11 minutes): 291 new slots -> 48 dreams of mean length
11.6 ("I am happy. y", "what do you eat?", "yes. it was in the box?", "it was in my nest": the exchanges are its episodes),
the loss 0.318 to 0.105, the gauge 0.553 -> 0.859; the store 7310; the face organ's correlation 0.45; the prefrontal 0.57;
the actor's slope 0.089. At this boundary: the own song remembered and the stress constant at a tenth.

The own song bounded (2026-09-12, 21:10): read live, the smiles wrote 1,658 own symbols into the store in twenty-five minutes
(every smile the last twenty-four symbols at the reward's full strength), the store at 7,879 of its cap of 8,192 and about
to prune the world's weakest. Bounded: the last twelve symbols (the word rewarded and what led to it), at a third of the
reward (the world's lines keep the stronger claim on the night), at most one write per forty ticks; the body saved and
restarted at once with the bounds (the stress by then 4.1, mood climbing, the loop 1 to 2 per thousand, the ear 0.18 to
0.45); the typist relaunched by the chain.

The question-first parent's report (2026-09-12, 22:05; it stopped itself at night 139, which ran past its forty-minute rule):
day 150: 37 questions of 71 parent lines; the child answered 16 of them with "I", "yes" or "no" BEFORE the other voice did
("what do you have?" -> "I have a hat", "do you want more milk?" -> "yes.", "are you sad?" -> "no m here with you"), five
more in part; the cue "I am " -> "here"; six questions of its own ("the egg dry?", "it hot here?", "you have?"), each answered
in the parent's next line; unprompted "please. I want it", "give me your hat please", "I eat the egg and milk"; faces 542/37.
Day 151-153 (the labels of the restarts): 34 questions, 13 answered with I/yes/no before the other voice ("who is here?" ->
"I have my hat again", "do you want more?" -> "yes."); its own questions nine ("you eat the egg?", "see me here?", "the egg
hot?"), and it took two of the answers back; "I see" 23 times in its own writing (from none), "I want" 6, "I eat" 4; "tin"
twice, then once; good, who, play, thank you learned inside exchanges; faces 479/31. Its warning: with the queue empty the
typist fell back to the retired drills ("put lid in"); fixed: the filler now holds the child's word with a recent planned
line, not the whole corpus. Night 139 is long: the own episodes count as the day's new memories and the night scaled toward
its maximum of 192 dreams.


### Night 139 (2026-09-12, 22:20): fifty-five minutes; the first replay of its own speech

Day 153 (the label; twenty-two minutes of a day after the mid-day restart): the parent asked 14 questions in 38 lines, three
answered in the modeled shape; smiles 183, frowns 12, duty 0.62. Night 139 (55 minutes): 880 new slots (the own episodes
among them) -> 110 dreams of mean length 17.9, the own utterances replayed for the first time ("i see milk went in the h",
"I want", "I saw dog", "who is happy?"), each beginning with a stray letter, the world's faded context's last symbol, because
the own episode's start mark sat on its first slot where the world's onsets are marked on the second; fixed (the mark on the
second slot). The loss 0.391 to 0.130, the curve still falling at round 32 (0.391, 0.314, 0.263, 0.227, 0.197, 0.173, 0.154,
0.140 every fourth round); the gauge 0.386 -> 0.82 (its own utterances the harder recall); the store 7644 after 547 dropped;
the face organ's correlation 0.45; the prefrontal 0.57; the actor's slope 0.073. The night's dream count capped at 96 from
the restart after this save. The guard held (duty 0.602, smiles per line 3.64) and was re-armed.

THE MOUTH PROBED (2026-09-12, 22:22; tools/probe_lm.py rebuilt to read the mouth, the cortex's forecast plus the hippocampal
recall with the efference copy, as the tick reads it; greedy, on the save after night 139):
  "do you want milk?"  ->  "I have my hat tle hat is"
  "what do you have?"  ->  "I want the toy gum"
  "hi"                 ->  ". I see you. hi. I see y"
  "are you sad?"       ->  "nest is on the me my soc"
  "what do you want?"  ->  "nt in the tint in the ti"
  "are you here?"      ->  "mmmm"
Three of eight prompts draw a reply of the modeled shape from the mouth itself with no sampling and no parent present: "I
have my hat", "I want the toy", "I see you. hi." The cortex alone is still the two-letter repeat ("nmenme", "ntnt"): the
hearing model is not the speaker; the speaker is the recall's chain read through the cortex, and the exchanges of the last
day and the night's replay of its own rewarded phrases are what put the answers there. The loop ("nt in the tint") remains
in the recall for two prompts. This is the reading the day was for.

The own song heard back (2026-09-12, 22:50, the label day 155): thirty minutes of the page: "i see milk" 35 times, "te was
in" 65, "I am here with you" 15, "here" 93; half its three-word windows repeats within the half hour; its most-said words in,
the, see, was, is, I, milk. The night's replay of its own rewarded phrase made that phrase the recall's strongest chain, and
the waking read could recall the same slot without end: the songbird's crystallized song, one phrase. The dreams already
run under adaptation (a recalled memory tires); the waking read did not. The law brought to the waking read: a slot that
wins the read loses a fifth of its availability and recovers toward rest by 0.97 a tick, synaptic depression (read_tire,
test 44, commit 615c5fd; the first attempt committed the test alone and the code an hour later). Armed for the restart
after night 140's save, beside the typist's relaunch with the battery fix. Stress 3.8, mood 2 to 3, sharpness 35 to 39,
the loop 2 per thousand, the ear 0.21 to 0.24 through the day; questions asked 33 by 22:45, replies in the modeled shape in
the first six seconds 2 (the planner's own count, which reads the whole turn, comes with its report).


### Night 140 (2026-09-12, 23:30)

Day 155 (the label; the body's 140th): the parent asked 45 questions in 90 lines, half, as briefed; the other voice 38 replies;
smiles 487 (3.81 a line), frowns 31, duty after the parent's lines 0.60; "said lately" withheld 211 (its repeats of one
phrase); replies in the modeled shape within six seconds 2 by my narrow count (the parent's count over the whole turn comes
with its report). The guard held (duty 0.592, smiles per line 3.81) and was re-armed. Night 140 (28 minutes): 549 new slots
-> 69 dreams of mean length 15.9 ("do you want more milk?", "yes. two pigs ar", "went in the eg", and its own "i see milk
went in the" still with the stray prefix of the older writes), the loss 0.352 to 0.116, the gauge 0.544 -> 0.858; the store
7989 after 214 dropped; the face organ's correlation 0.45; the prefrontal 0.59; the actor's slope 0.073. At this boundary:
the waking recall's tiring, the typist's relaunch with the day's-cues-only battery, the probe.

The mouth probed on the save after night 140 (2026-09-12, 23:29): "do you want milk?" -> "I have the little tin"; "what do
you want?" -> "I have the hen out of t"; "what do you have?" -> "I tub tle hat is not in m"; "can you play?" -> "I s it dry?
I want sock"; "hi" -> "meast the ice please"; "are you here?" -> "me see milk went in the"; "who is here?" and "I see " ->
"ttle hat ttle hat", the little-hat chain. Four of ten prompts open with "I" and two carry a want or a having; the recall's
loops ("ttle hat", "nt in the tub") take the rest; the cortex alone still repeats. The waking recall's tiring is live from
this restart (pid 51361, read_tire 0.2), which the probe does not apply; the page is the reading now.

THE TREND, days 141 to 155 (2026-09-12, 23:35), from the rows: of the child's scored words, the conversational core (I, yes,
no, you, me, my, want, see, have, here, please, am) was 6 to 7 percent on the drill days (141 to 145), 13 to 15 percent on
the first conversation days (147 to 150), 19 to 25 percent on the question days (152 to 155); the container nouns (tin,
box, bin, tub) 7 to 11 percent, then 1 to 4, then 0 to 1. Its replies to the parent's questions beginning with I, yes, no
or here within six seconds: none before questions were asked, 15 of 39 on day 150, 7 of 20, 3 of 16, 2 of 46 on day 155,
the day the own-song phrase loop took its turns (the parent's own count over the whole turn: 16 of 37 and 13 of 34 on days
150 and 152). Under the recall's tiring, the first five minutes: the share of its three-word windows that are repeats 0.09
(0.52 the afternoon before); "i see milk" three times, "ttle hat" twice; "y i sit down here wit" to "good! I see you here".
The speech is not random: what it says has moved with what it hears, from containers to the conversational core, four times
over in ten days, and the answer shape appeared the day questions began.

THE NINETEENTH DEFECT (2026-09-12, 23:50). The question "are we missing something in the architecture" answered with a
measurement: the cortex, teacher-forced on the parent's last sixty lines, predicts the next symbol right 32 times in a
hundred, after 140 nights, on a corpus where most symbols follow from the word; and fed its own continuation back as heard
speech it cycles ("the he he he", "henox henox"): the hearing model has not learned the language. The night replayed 48 to 96
dreams for 32 to 48 rounds, a few dozen lines drawn by strength, memorized (their loss 0.08); the day's lesson is a trickle
at 1e-5. Complementary learning systems: the cortex learns from many interleaved episodes, each replayed a few times. The
night becomes broad from the restart after night 141: 512 dreams (up to 1024, one per new memory) for three rounds, the same
compute spread over ten times the lines, each run down a branch of the chain. The reading: the accuracy on the last sixty
lines, 0.32 tonight; the mouth's probe beside it.

Corrected at 23:58: the 0.32 was the probe's artifact (it fed the live slow state instead of running the ladder along the
line as the night does). By the night's own gauge the cortex alone predicts 0.82 of the next symbols on the lines the night
replayed, 0.52 on the parent's last sixty lines it did not, 0.53 on lines from twenty days ago: a half-learned language,
memorized where replayed, generalized to half elsewhere. The nineteenth defect stands as breadth against depth; the broad
night is armed; the reading is the gauge on unreplayed lines, 0.52 tonight.

The blind typist (2026-09-12, 23:28 to 23:56): at the boundary the typist's relaunch fired on the save row a few seconds
before the server's restart, so the new typist had read the old page before the page began again; its cursor reset with the
page but its tick count and scan cursor did not, and for twenty-eight minutes it typed the parent's lines and scored
nothing: no smiles, no frowns, no rows of the child's words. The child's mood fell to the floor and the mouth to its
flattest, and the twenty-minute watch under the recall's tiring read a body no one was smiling at. Fixed in the typist
(the page's tick count and the scan restart with the page; commit above) and the blind one stopped for the chain's relaunch.
The tiring's reading is therefore not yet taken; the watch runs again on the relaunched day.

The seeing typist (2026-09-12, 23:56; the label day 157): in its first two minutes 28 smiles, 30 misses, 3 frowns; the
child's words after the parent's lines "I want a jug put", "now? no wet dog had"; mood climbing from the floor (-2.4),
sharpness 15, stress 4.1. The recall's tiring is now read on a scored day.

The scored day under the recall's tiring (2026-09-13, 23:56 to 00:08, the label day 157): mood -1.5 -> 5.2 in eight minutes
once smiled at again, sharpness 19 -> 47, smiles 74 -> 116 per three hundred rows, frowns 5 or 6; the share of its
three-word windows that repeat 0.18, 0.30, 0.35, 0.45 (0.52 before the tiring: the strongest own phrase recovers within
minutes and returns); "i see milk" 7, 8, 7, 3 per five minutes. Its turns: "with you see here? no wet dog", "I s it dry? I
want now? no wet dog here". The actor's slope 0.068, drifting down through the day. Night 141 began at 00:10; the broad
night lands at the restart on its save.


### Night 141 (2026-09-13, 00:30): the last deep night

Day 157 (the label; the scored thirty-three minutes after the blind typist was stopped): the parent asked 17 questions in 37
lines; smiles 201, frowns 11, duty after the parent's lines 0.65; five replies of the modeled shape within six seconds. The
guard held (duty 0.638, smiles per line 3.81) and was re-armed. Night 141 (17 minutes; the last under the deep settings):
214 new slots -> 48 dreams of mean length 16.9 ("I ha big tu", "y i see milk is her", "you have?", "lI am dry.", "a bunt in
the den"), the loss 0.338 to 0.093, the gauge 0.554 -> 0.886; the store at its cap of 8192 (the weakest give way at each
write from here; the fade and the cap keep it there); the face organ's correlation 0.42; the prefrontal 0.52; the actor's
slope 0.068. At this boundary: the broad night (512 dreams and up to 1024, three rounds) from the restart on this save; the
probe on the save; the first broad night is night 142.

The mouth probed on the save after night 141 (2026-09-13, 00:28): "can you play?" -> "I come. I am not sad"; "hi" -> ". I see
you y hip is not"; "who is here?" -> "mea I want my yam hat milk"; "do you want milk?" -> "I ? all gone? your ice is";
"what do you have?" -> "te was in the tub y yes."; "are you here?" -> "m outtle hat is not my t". Four of ten open with "I"
or carry an answer, and "I come. I am not sad" is the cleanest reply the mouth has produced: two sentences of the modeled
shape to a question it was asked today. The recall's chains ("te was in", "ttle hat") take the rest; the cortex alone
still repeats. The restart with the broad night is done (pid 56598: 512 to 1024 dreams, three rounds); night 142 is the
first broad night, and the probe after its save reads the accuracy.


### Night 142 (2026-09-13, 01:40): the first broad night

Day 159 (the label; the body's 142nd): the parent asked 48 questions in 101 lines; smiles 475 (3.21 a line), frowns 34,
duty after the parent's lines 0.50; replies of the modeled shape within six seconds 2 by the narrow count. The guard held
(duty 0.495, smiles per line 3.21) and was re-armed. Night 142 (23 minutes), the first broad one: 512 dreams of mean length
17.1 (the store at its cap, so the day's "new slots" read 10 and the count fell to the minimum), three rounds, the loss
0.344, 0.322, 0.306 (a shallow descent by design), the gauge on the dream set 0.532 -> 0.592 over 8,748 symbols (the deep
nights read 0.86 on their few dreams: memorization; this reads the broad set after three passes); the store 8192, nothing
dropped by the fade (the cap prunes at each write now); the face organ's correlation 0.44; the prefrontal 0.54; the actor's
slope 0.065. The reading that matters is the probe on this save: the accuracy on the parent's last sixty unreplayed lines,
0.52 before this night.

### The twentieth defect, found by the probe after night 142 (2026-09-13, 01:40-02:25)

The probe after the first broad night read the parent's last sixty lines at 0.53, unchanged, and the arithmetic said why: a night's
round summed every dream's gradient into one step, so the broad night was three weight updates and a deep night twenty-four; the
day's lesson a step every 24 ticks at a hundredth of the rate. Built and tested in the hour: a synaptic step per batch of replays
(night_batch), the dreams shuffled and run in lockstep, the bands along a dream on a cache of the stream's keys and values (a
dream-round 1.4 s -> 0.12 s); tests 45 and 46; 46/46. On copies: batch 8 at the night's rate was noise (the lines 0.534 -> 0.492
after 32 steps, 0.509 after 128); batch 64 descended from the first step. A whole night of batch 64, six rounds over 512 dreams (48
steps) through the night code on a copy: the dream set 0.585 -> 0.725, the parent's last sixty lines 0.523 -> 0.606, the old lines
of days 110-125 (faded from the store) 0.438 -> 0.454; ten minutes on a busy machine. Deployed for the boundary after night 143:
batch 64, six rounds, warm 4, 1024 dreams (96 steps a night).
02:59: the challenger on the copy over a whole night (1024 dreams, 6 rounds, 384 steps of batch 16 at 3e-5, warm 8): the dream set
0.574 -> 0.743, the parent's last sixty lines 0.500 -> 0.588, the old lines (days 110-125, faded from the store) 0.438 -> 0.498
against the batch-64 night's 0.454. Re-armed before night 143's save: the served body takes the challenger's form from day 161.

### Night 143 (2026-09-13, 03:05): the last night of the old form

Day 160 (01:36-03:05; the question-first parent's first day): 118 parent lines, 57 of them questions, 41 partner lines, 3 cues;
smiles 411, frowns 38, withheld 17; duty 0.48; replies of the modeled shape within the parent's question 3 by the narrow count;
its own question marks 10; its words: see 47, egg 29, milk 20, was 19, dog, eat, you, all, with. Night 143 (41 minutes under the
copy runs' load): 512 dreams of mean length 16.5, three steps, the loss 0.333 -> 0.311, the gauge on the dream set 0.532 -> 0.579,
the store at its cap, nothing dropped. The guard held (duty 0.478, smiles per line 2.59) and was re-armed. The reload waits at this
night's save: from day 161 the night steps after every 16 dreams at 3e-5 with a ramp of eight, six rounds over 1024 dreams.

### Night 144 (2026-09-13, 03:59): the first served night of the new form

Day 161/162 (03:07-03:58, the labels split by the relaunch): 100 parent lines, 50 questions, 51 partner lines; smiles 416, frowns
35; duty 0.48; its own question marks 12; modeled replies 1. Night 144: 1024 dreams (mean length 15.7), 384 steps of batch 16 at
3e-5 with the ramp, nine minutes; the loss 0.299 -> 0.216; the gauge on the dream set 0.505 -> 0.696 (cos 0.541 -> 0.726). The
guard held (duty 0.474, smiles per line 2.77). The probe on the save: the parent's last sixty lines 0.543 (0.53 after night 142),
the old lines 0.455 (0.438 after night 142). Real, and a third of what the same night gave on the copy (0.588 and 0.498). Two
suspects: the day's waking lesson between the nights (500 steps of a single lived window at 1e-5, with its own recalled babble as
the target at its own positions), and the dream set, whose examples read like its own garbled speech ('e. te was in the egg is a',
'k iilo we'): the own song, written to the store at every smile, is dreamt as if the world had said it. Instruments armed: a dusk
probe (a save as the sleep pressure nears the threshold, then the probe) to tell the day's effect from the night's; the dream
set's composition measured now.
04:36: the twenty-first defect, measured. The dream set on the post-144 body: 74% of the onset slots and 58% of the drawn dreams
were its own garbled song, entered as the world's speech. Built and tested dream_who (dreams from the world's onsets; its own
symbols as its own sound, no forecast owed; test 47, 47/47). Two whole nights from the same copy: the served form, lines 0.546 ->
0.572 and the old lines 0.455 -> 0.425 (eroding); dream_who, lines -> 0.601 and the old lines -> 0.517. Armed for night 145's save.

### Night 145 (2026-09-13, 04:58): the last night that dreamt its own song as the world's

Day 163 (04:01-04:58): 109 parent lines, 54 questions, 53 partner lines; smiles 450, frowns 42; duty 0.52; its own question marks
18 (10 and 12 the two days before); modeled replies 1. The dusk probe (a save as the sleep pressure neared the threshold): the
parent's last sixty lines 0.569, the old lines 0.456, against 0.543 and 0.455 after night 144: the waking day does not erode the
language. Night 145: 1022 dreams, 384 steps, nine and a half minutes, the loss 0.261 -> 0.203, the dream set 0.593 -> 0.717; the
examples still its own babble ('t h te was in my ha', 'gonee dog outy sosit'). The guard held (duty 0.518, smiles per line 2.77).
The probe on its save: the recent lines 0.618, the old lines 0.467. The course of the old lines, the language-model reading
proper: 0.438 after night 142, 0.455 after 144, 0.456 at dusk, 0.467 after 145. The reload at this save: from day 164 the dream
knows who spoke.

The second question-first parent's report (labels 160-163, 05:05): 53/50/54 questions a day; answered before the partner voice
31/31/24 (58, 62, 44%), mostly "i see milk", the unmistakable ones "what do you have now?" -> "had a big hat", "can I hide from
you?" -> "yes please.", "do you see me here?" -> "you hee me here"; its own questions 10/13/17 ("here?", "egg?", "dog?", "hot?",
"you sad?", "milk is good milk?"), the answers taken; cue answers "I have " -> "a big hat", "I want " -> "hot milk"; unprompted
strings of three or more words 93/87/100 a day ("my sock is not in my", "yes. here is a cup", "your milk is here with me", "no wet
dog is here", "the egg is not here"); "please", "yes please.", "again", "give me the" on its own; "i see milk" self-sustaining
(5-6 repeats a day, unmoved by resting "I see" from the parent's lines: the new "tin"); smiles/frowns 417/38, 430/36, 460/43. The
parent's lapse: the queue ran dry for ten minutes on day 160 and the typist replayed one frame five times. The next parent (labels
165-168) builds on its questions (each answered as the opening of an exchange) and starves "I see" (cues on "I am", "I have", "I
want" only).

### Night 146 (2026-09-13, 05:51): the first who-aware night, and a flawed reading

Day 165 (05:01-05:51; the third parent's first day): 103 parent lines, 54 questions, 48 partner lines; smiles 431, frowns 33; duty
0.51; its own question marks 6; "i see" 36 times in its writing. The dusk probe: the old lines 0.459 (0.467 after night 145): the
day flat again. Night 146: 1024 dreams from the world's onsets, a third of their symbols its own and entered as its own sound, 384
steps, nine minutes, the loss 0.324 -> 0.235, the gauge on the world's symbols 0.454 -> 0.661; the examples with its own in
capitals ('al gOne IS IN my bag', 'k WITH YOU I SEE MILk Was'). The guard held (duty 0.508, smiles per line 2.85). The probe on
the save: the recent lines 0.601 (0.557 at dusk), the old lines 0.447 (0.459 at dusk). Four strong nights: the old lines 0.438,
0.455, 0.467, 0.447. The recent lines rise because the store holds them and the night replays them; the old lines do not, and the
copy's 0.517 was one draw's luck (two nights from one copy spread 0.09). The old set is also the wrong ruler: days 110-125 were the
container stage, whose nouns the parent no longer says, so it measures the drift of the vocabulary as much as the language. A
held-out set replaces it: lines in the present stage's style and vocabulary, written by me and never typed.
The held-out reading on the same save: **0.598 (cos 0.629) on 73 never-typed lines of the present stage, against 0.595 on the
replayed recent lines.** The cortex generalises within the stage as well as it remembers; what the old lines measured was the
language moving on from its container nouns. The held-out gauge is the ruler from here; the old lines stay in the probe as a
record of the drift. (Of 124 lines I wrote in the parents' style, 43 turned out to have been typed already: the style is theirs.)

### Night 147 (2026-09-13, 06:44): the night that lowered what it did not train on

Day 166 (05:53-06:44): 102 parent lines, 51 questions, 49 partner lines; smiles 449, frowns 32; duty 0.46; its own question marks
2; "i see" 23 (36 the day before). The dusk probe: held-out 0.615 (0.598 after night 146), the recent lines 0.608, the old 0.475:
the day raised all three. Night 147: 1024 dreams, own share 0.32, 384 steps, the world gauge 0.522 -> 0.668. The guard held (duty
0.460, smiles per line 2.99). The probe on the save: held-out 0.601, the recent lines 0.567, the old 0.453: the night lowered all
three while fitting its dreams. The dreams' world parts are patchworks: at a merged slot (a context shared by many lines) the chain
draws a successor from another line, so a dream is a five-gram walk through the corpus rather than an utterance as lived, and 384
steps of fitting that teach the cortex that a line's far structure is random.

### Night 148 (2026-09-13, 07:36): the last night of stitched dreams

Day 167 (06:46-07:36): 104 parent lines, 51 questions, 51 partner lines; smiles 386, frowns 36; duty 0.40; its own question marks
9; "i see" 13 (36, 23, 13 over three days: the starving works). The dusk probe: held-out 0.596 (0.601 after night 147), the
recent lines 0.554, the old 0.450. Night 148: 1024 dreams, own share 0.32, 384 steps, the world gauge 0.523 -> 0.672; examples
'neat idono', 'noW?EE DOG Is ou', 'here iw'. The guard held (duty 0.399, smiles per line 2.48). The probe on the save: held-out
0.577, the recent lines 0.575, the old 0.401. Two nights running the stitched dreams lowered the held-out (0.615 -> 0.601, 0.596
-> 0.577) and the old lines (0.475 -> 0.453, 0.450 -> 0.401) while raising the replayed recent lines. The reload at this save:
the episode tags. Night 149 is the reading: whole-line dreams against the same 384 steps. If the held-out still falls, the
night's rate is next (3e-5 -> 1e-5 at the following boundary).

The third question-first parent's report (labels 165-167, 07:45): 54/51/51 questions a day; answered before the partner voice
7/2/5 (the earlier 30s were "i see milk"); its own question marks 7/3/9 ("milk?", "e egg?", "and cold?", "it please?", "see
here?", "see it there?"), the answers taken ("you hee me here", "here with you"); fourteen cold cues returned nothing (the frames
land inside conversation); "i see" 36 -> 23 -> 13; "all gone" steady at 9-11; new sentences "ham is hot in my bed", "it is my
egg", "my sock is here", "the dog is here again", "help me here", "wet dog is out", "sit down here"; new words hug, too, wait,
hungry, talk, nice; smiles/frowns 442/33, 460/33, 398/36; the queue never ran dry. The next parent (labels 169-172): its
questions answered as openings, cues only off its own words, "I see" starved.

### Night 149 (2026-09-13, 08:29): the first whole-line night, and the night's rate

Day 169 (07:39-08:29; the fourth parent's first day): 106 parent lines, 52 questions, 50 partner lines; smiles 423, frowns 35;
duty 0.48; its own question marks 7; "i see" 28 (13 the day before: self-sustaining after all). The dusk probe: held-out 0.583
(0.577 after night 148), the old lines 0.432 (0.401): the day restored some. Night 149, the first with the episode tags: 1024
dreams of mean length 11.0, own share 0.23, 384 steps, the NREM loss 0.218 -> 0.157 (0.298 -> 0.228 the night before: consistent
utterances now), the world gauge 0.588 -> 0.778. The guard held (duty 0.480, smiles per line 2.72). The probe on the save:
held-out 0.575 (cos 0.593), the old lines 0.412, the recent lines 0.602. Three nights running: the held-out down 0.014, 0.019,
0.008 at night and up 0.017, down 0.005, up 0.006 by day; the old lines down every night. The stitching was not the cause, or not
the only one: 384 steps at 3e-5 fit whatever the night dreams (the store's three days, drawn by strength, the most repeated lines
most) at the cost of the language at large. The night's plasticity is a constant; from the boundary after night 150 it is 1e-5,
a third, the schedule unchanged. Measured on copies meanwhile: dreams drawn uniformly over the world's onsets against by strength.

Two incidents in the morning of 2026-09-13, both mine. At 08:54 macOS's process list failed for a moment, the typist's chain read
that as the typist gone and launched a second beside the living one; the new one died within seconds on the corpus file both were
writing (the chain now ends its wait only on a true "no such process"). From 08:34 to 10:07 four copy nights ran in parallel at
full priority beside the served body, which runs niced: its day slowed to a tick a second and the parent's lines came five times
denser in its ticks than they should; the copies were killed and the day recovered its pace within seconds (the rule now: one copy
run at a time, at the lowest priority, never beside a served night; the restart scripts no longer nice the server).

### Night 150 (2026-09-13, 10:44): the second tagged night; the rate lowered at its save

Day 170 (08:31-10:44, slowed to a tick a second for ninety minutes by my copy runs): 214 parent lines, 111 questions, 34 partner
lines; smiles 238, frowns 82; duty 0.38; its own question marks 8; "i see" 8. Night 150: 1024 dreams (mean length 10.8), own share
0.16, 384 steps at 3e-5, the loss 0.216 -> 0.149, the world gauge 0.573 -> 0.784; examples 'we play her', 'where is', 'ar yo '.
The guard held (duty 0.380, smiles per line 0.94). The probe on the save: held-out 0.583 (0.575 after night 149), the recent lines
0.701, the old lines 0.427 (0.412). No dusk reading: the dusk probe fired as the night began and read the night's save. From this
save the night's rate is 1e-5 (verified on the restart); night 151 is the first at that rate. The held-out's course through six
strong nights: 0.598, 0.601, 0.577, 0.575, 0.583.
11:15: the twenty-third defect, at the mouth: the read is a five-gram walk (as the dreams were) and 60% of its reads land in its own
stored song. Built read_follow (the recall carries the episode; test 49; 49/49). Decided: own_store off and read_follow 20 at the
save after night 151; the mouth probe reads the lived prefix and the own-winner share from tomorrow.

### Night 151 (2026-09-13, 11:53): the rate was not the cause; the mouth speaks whole phrases

Day 172 (10:46-11:53): 120 parent lines, 57 questions, 58 partner lines; smiles 354, frowns 37; duty 0.40; its own question marks
15; "i see" 20. The dusk probe: held-out 0.584 (0.583 after night 150: the day flat), the old lines 0.403, the recent 0.611. Night
151, the first at a third of the rate (1e-5): 1024 dreams (mean length 10.1), own share 0.16, 384 steps, the loss 0.224 -> 0.145
and the world gauge 0.573 -> 0.783, the same fit as at 3e-5: the rate never limited the night's fit of its dreams. The guard held
(duty 0.399, smiles per line 2.00). The probe on the save: held-out 0.559 (from 0.584 at dusk, the largest drop yet), the old
lines 0.411, the recent 0.606. So the night's damage is not its rate but the direction of its fit: 1024 dreams drawn by strength,
the most repeated lines most, six passes over the same set. The draw comparison on copies (uniform against by strength, two
seeds) is the reading for the next boundary; with it, one pass over six times the dreams instead of six passes over the same.

The mouth, on this save, with the recall carrying the episode (read_follow 20): 'you and your dog. good' to "do you want milk?",
'. I am happy with you' to "hi", 'I am tired' to "what do you want?", 'that is your' to "are you here?": whole phrases where the
recall lands in the world's memories; still 'nt in the hte was in hot' where it lands in its own song, which won 10 to 20 of 24
reads and now fades (own_store off from this save).

The fourth question-first parent's report (labels 169, 170, 172; 12:05): answered before the partner voice 2, 6, 10 a day ("are you
hungry?" -> "I want with you now here", "do you want it here?" -> "yes please", "do you want the kite?" -> "es! we are all happy");
its own question marks 7, 8, 14, on the last day whole questions ("what do we do first?", "what do you see?", "you a hug?"); it did
not take the answers because the queue's depth put each reply four to ten lines late (the next parent keeps the queue six to
eight ahead and answers in the very next line); cues only off its own words: 3, one answered ("I am " -> "here!it"); unprompted
"I want hot milk for you", "we are all happy", "we eat it here", "sit down here now", "have a run dog!", "a bird is in it", "nest on
your hat"; "I see milk" dead (1, 1, 0) but the stub "i see m is i see m is" survives (30, 8, 20 "i see" a day); smiles/frowns
426/35, 241/81 (the overlap day), 355/37. It reported the 08:54 overlap as a genuine fault (it was: my chain's relaunch). Its own
wait loop stalled three times for half an hour, which emptied the queue. The fifth parent (labels 174-177) took over at 12:04
after a dry half hour.
12:20: the accident that read best. The clean-lines diagnostic's option went unparsed, so what ran was a plain second night at the
served constants from the save after night 151: the held-out 0.559 -> 0.599, the old lines 0.411 -> 0.427, the recent lines
0.618 -> 0.634, the dreams' gauge 0.724 -> 0.782 and the loss 0.175 -> 0.146. A second consolidation of the same content raised
what the first lowered. Three copy nights now run in turn from the dusk save before night 151 (the served night's own start,
whose six rounds gave 0.559): no forecast owed on a dream's first three positions; twelve rounds instead of six; and the parent's
last thousand lines as dreams, whole. The night's damage may be the shock of first fitting new memories, which a longer settling
repairs; the twelve-round night is the test.

### Night 152 (2026-09-13, 13:01): the first served night that did not subtract

Day 174 (11:55-13:01; the fifth parent's first day, the own song no longer written, the recall carrying the episode): 114 parent
lines, 45 questions, 68 partner lines; smiles 378, frowns 37; duty 0.45; its own question marks 15; "i see" 27 (the stub 'yam is
i see m is i' loops on from the own slots still in the store). The dusk probe: held-out 0.556 (0.559 after night 151), the old
lines 0.401. Night 152: 1024 dreams of mean length 9.0, own share 0.14, 384 steps at 1e-5, the loss 0.218 -> 0.149, the world
gauge 0.622 -> 0.775; the examples clean at last ('are you hot', 'what do you pl', 'I want my', 's the dog wit'). The guard held
(duty 0.451, smiles per line 2.08). The probe on the save: held-out 0.564 (from 0.556), the old lines 0.415 (from 0.401), the
recent lines 0.633; the mouth 'you have the egg now!', 'that is your egg', 'I am happy now?', 'I am happy too'; the recall's own
winners 3 to 11 of 24 on most prompts (15 to 20 where the stub lives). The held-out's course: 0.598, 0.601, 0.577, 0.575, 0.583,
0.559, 0.564.
13:25: the decisive reading. A night of the parent's last thousand lines, whole and clean, from the dusk save of day 174: the
held-out 0.556 -> 0.687, the old lines 0.401 -> 0.484, the replayed 0.816. Seven nights of the store's fragments had read -0.01.
The night's content was the whole fault. Built the utterances heard (dream_source utterances: the world's utterances kept whole
between pauses, fading by night, dreamt whole; test 51; 51/51); armed for the save after night 153. Night 154 is the first that
dreams them.

### Night 153 (2026-09-13, 13:54): the last night of the store's fragments

Day 175 (13:02-13:54): 100 parent lines, 32 questions, 70 partner lines; smiles 340, frowns 34; duty 0.43; its own question marks
16; "i see" 34 (the stub 'i see m is i see' loops from the own slots still strong in the store). The dusk probe: held-out 0.564
(0.564 after night 152: the day flat), the old lines 0.432. Night 153: 1024 dreams of mean length 8.7, own share 0.14, 384 steps,
the loss 0.232 -> 0.174, the world gauge 0.596 -> 0.733; examples 'come in', 'sit do', 'l E EGG Is alml!h'. The guard held (duty
0.431, smiles per line 2.00). The probe on the save: held-out 0.547 (from 0.564), the old lines 0.424, the recent 0.592: the
fragments' last subtraction. The reload at this save: the utterances heard. Restarted 13:56 (verified); day 176 fills the memory;
night 154 dreams whole utterances. The held-out's course: 0.598, 0.601, 0.577, 0.575, 0.583, 0.559, 0.564, 0.547.

### Night 154 (2026-09-13, 14:48): the first night of whole utterances, and the held-out's first climb

Day 177 (13:56-14:48; the first day whose utterances were kept whole): 94 parent lines, 23 questions, 67 partner lines; smiles
356, frowns 31; duty 0.43; its own question marks 13; "i see" 20. The dusk probe: held-out 0.517 (0.547 after night 153), the
old lines 0.415. Night 154: 161 utterances heard, 1024 dreams of mean length 21.1, all the world's ('yes! the ball is here',
'where is the ball?', 'good. I am with you', 'what do you have now?'), 384 steps at 1e-5, the loss 0.219 -> 0.090, the gauge on
the dreams 0.547 -> 0.885. The guard held (duty 0.430, smiles per line 2.21). The probe on the save: **held-out 0.608 (from
0.517), the old lines 0.434 (from 0.415), the recent lines 0.866 (from 0.551)**: a gain of 0.091 in one night where seven nights
of the store's fragments had lost a hundredth each. The held-out's course: 0.598, 0.601, 0.577, 0.575, 0.583, 0.559, 0.564,
0.547, 0.608. The fifth parent's report (14:50): whole sentences unprompted ("I had the corn with me", "yes. we wash the dog",
"y sock is on my leg", "yes! we talk again"); its own questions productive ("where am I?", "you help me here?"), the answers
taken; answered before the partner voice four times a day with the parent's questions fallen to 38, 31, 23; "i see" 25, 34, 20
from its own memory. The sixth parent (labels 178-181): half of A's lines questions, the queue six to eight ahead.

### Night 155 (2026-09-13, 15:41): the climb confirmed

Day 178 (14:49-15:41; the sixth parent's first day): 89 parent lines, 69 of them questions, 74 partner lines; smiles 326, frowns
33; duty 0.43; its own question marks 12; "i see" 26. The dusk probe: held-out 0.605 (0.608 after night 154: the day held), the
old lines 0.429. Night 155: 327 utterances heard, 1024 dreams of mean length 21.3 ('do you want more egg?', 'are you here with
me?', 'what do we do now?', 'thank you! you help me'), 384 steps, the loss 0.160 -> 0.099, the gauge on the dreams 0.731 -> 0.868.
The guard held (duty 0.432, smiles per line 1.98). The probe on the save: **held-out 0.636 (from 0.605)**, the old lines 0.424, the
recent lines 0.830. Two nights of whole utterances: 0.547 -> 0.608 -> 0.636, the day between them flat. The night adds. The mouth:
'I want milk too?', 'yes. you', 'I ate the hat'; the stub 'i see 1 is' still on the first prompt from the old own slots. The next
lever: the pace, a line every 60 ticks and the child's turn 32 (from 80 and 40), at the typist's next relaunch.

### Night 156 (2026-09-13, 16:35): the third whole-utterance night

Day 179 (15:42-16:35): 87 parent lines, 63 questions, 78 partner lines; smiles 354, frowns 32; duty 0.44; its own question marks
11; "i see" 26. The dusk probe: held-out 0.609 (0.636 after night 155: the day gave 0.027 back), the old lines 0.442. Night 156:
492 utterances heard, 1024 dreams of mean length 21.6 ('the dog is with me', 'is the bird all gone now?', 'can you get up to
it?'), the loss 0.149 -> 0.098, the gauge on the dreams 0.752 -> 0.869. The guard held (duty 0.440, smiles per line 2.15). The
probe on the save: held-out 0.630 (from 0.609), the old lines 0.448, the recent lines 0.826; the mouth 'I sit with my dog', 'I am
here with you'. Three nights of whole utterances: +0.091, +0.031, +0.021; the days between: -0.003, -0.027. The course: 0.547,
0.608, 0.636, 0.630. At this save the typist relaunched at the faster pace (a line every 60 ticks, the child's turn 32).

### Night 157 (2026-09-13, 17:28): the fourth whole-utterance night; the held-out levels

Day 180 (16:35-17:28; the first at the faster pace): 101 parent lines, 71 questions, 81 partner lines (182 in the day against
~160); smiles 320, frowns 35; duty 0.42; its own question marks 12; "i see" 24. The dusk probe: held-out 0.626 (0.630 after
night 156: the day flat), the old lines 0.443. Night 157: 674 utterances heard, 1024 dreams of mean length 21.7 ('the mat is
under the dog', 'I play and I run', 'I hug you and my dog'), the loss 0.137 -> 0.098, the gauge on the dreams 0.753 -> 0.866. The
guard held (duty 0.421, smiles per line 1.73). The probe on the save: held-out 0.628 (from 0.626; the cosine 0.656 -> 0.676), the
old lines 0.460 (from 0.401 four nights ago, a new high), the recent 0.823; the mouth 'I am happy with you', 'I am here with me
now?', 'gone to talk to my dog'; the recall's own winners 2 to 18 of 24, falling. Four whole-utterance nights: +0.091, +0.031,
+0.021, +0.002; the days -0.003, -0.027, -0.004. The held-out levels near 0.63 while the replayed lines sit at 0.82: the gap is
what unseen lines of this style withhold (which noun, which frame), and the same eight families of exchange, dreamt again, do not
close it. The old lines still rising says the language transfers; the next gain is in the variety of what it hears.

The sixth question-first parent's report (labels 178-180, 17:40): A's questions 69/89, 63/87, 71/101 (70-78% of its lines);
answered before the partner voice 16, 13, 18 a day (from four), almost all "yes", "yes please", "no", four with a whole sentence
("I ate it!", "I eat the yam", "I eat it all"); its own question marks 12, 10, 12 ("sock on?", "milk is yours?", "do you wait?",
"what do you eat?"), each answered as an opening, and it stayed in the topic two or three turns; "all gone now?" answered
differently each time fell 5, 3, 3; unprompted whole sentences "the ham is hot", "the egg is all gone", "yes! then we run out", "a
nest is on me" (never taught), "it is here at my leg"; "i see" flat at 24-26 from its own memory; smiles/frowns 333/33, 354/32,
323/35; one new word, "love", inside "I ___ you". Its one cue on a new line was typed plain (the typist cues only lines already
heard that day). The seventh parent (labels 181-184): questions above 70%, its questions answered as openings, and what/who/where
questions to push "yes please" toward a named thing.

### Night 158 (2026-09-13, 18:23): the climb resumes

Day 181 (17:30-18:23; the seventh parent's first day): 131 parent lines, 87 questions, 53 partner lines; smiles 283, frowns 37;
duty 0.35; its own question marks 12. The dusk probe: held-out 0.637 (0.628 after night 157: the day +0.009), the old lines
0.454. Night 158: 856 utterances heard, 1024 dreams of mean length 21.8 ('yes! we go out. run!', 'where is your dog?'), the loss
0.138 -> 0.103, the gauge on the dreams 0.763 -> 0.856. The guard held (duty 0.354, smiles per line 1.55). The probe on the save:
**held-out 0.660** (from 0.637), the old lines 0.474 (from 0.454), the recent 0.769: both a new high. The course: 0.547, 0.608,
0.636, 0.630, 0.628, 0.660. The actor's slope 0.035: the reward does not depend on which word it says (a smile at any known word),
so the actor has nothing to learn from; the smile for the answer is built tonight (the teacher's method).

### Night 159 (2026-09-13, 19:17): the sixth whole-utterance night; the smile for the answer from its save

Day 182 (18:24-19:17): 119 parent lines, 88 questions, 60 partner lines; smiles 307, frowns 32; its own question marks 15. The
dusk probe: held-out 0.643 (0.660 after night 158: the day gave 0.017 back), the old lines 0.470. Night 159: 1036 utterances
heard, 1024 dreams of mean length 21.9 ('who put it on me?', 'can your dog jump too?'), the loss 0.142 -> 0.114, the gauge on the
dreams 0.771 -> 0.844. The guard held (duty 0.398, smiles per line 1.72). The probe on the save: **held-out 0.663** (from 0.643;
the cosine 0.698), the old lines 0.474, the recent 0.787; the mouth '! are you happy with me?' to "hi". The course: 0.547, 0.608,
0.636, 0.630, 0.628, 0.660, 0.663; the old lines 0.40 -> 0.47. At this save the typist relaunched with the smile for the answer
(day 183 the first under it).

### Night 160 (2026-09-13, 20:12): the first day of faint smiles

Day 183 (19:18-20:12; the first under the smile for the answer, whose answer branch never fired for a timing fault, so every
smile was faint): 132 parent lines, 99 questions, 60 partner lines; smiles 277, all faint; frowns 39; duty 0.40; its own
question marks 18 (a high). The mood fell to -6 and the stress rose to 5 as the felt reward thinned: the risk I named. The dusk
probe: held-out 0.643 (0.663 after night 159: the day gave 0.020 back), the old lines 0.487, a new high. Night 160: 1229
utterances heard, the loss 0.136 -> 0.112, the gauge on the dreams 0.782 -> 0.846. The guard held (duty 0.402, smiles per line
1.44). At this save the typist relaunched with the expectation standing until the parent's next line; day 184 is the first
where an answer can earn the full smile. If the mood does not recover with the answer smiles, the faint smile's value rises.

The probe after night 160: held-out 0.662 (from 0.643 at dusk), the old lines 0.507 (from 0.40 a week of nights ago), the recent
0.789. The seventh question-first parent's report (labels 181-183, 20:15): A's questions 66, 73, 75% of its lines; **answered
before the partner voice 31, 36, 44 a day** (16-18 the days before), naming a thing 11-13 by the strict count, 6-8 genuine ("dog",
"a bird", "the egg"); its own question marks 11, 15, 18, with real forms ("you eat the ham?", "what can I help with?", "do we get
more?", "what do I give the"), each answered as an opening; cues taken on the post-night re-run ("I have " -> "the egg too", "I am "
-> "here"); "that dog is out again now" (six words, whole, an answer before B), "that is my big dog", "milk is not all gone", "my
sock is on you now"; "i see" 13, 18, 21; "all gone" 8 -> 2; one new word, "find"; smiles/frowns 283/37, 307/32, 277/39. The
queue ran dry twice at hand-overs. The eighth parent (labels 184-187): B's answers name the child's own top nouns so its guesses can
earn the answer smile.
20:42: the smile for the answer, read on its first real day (184). The first answer smile fired at 20:32 ('yes' before the partner's
"yes. I ate the egg"): the mechanism works on the served body. But one answer in 115 lines against 159 faint smiles at a quarter of
the full one thinned the reward until the mood fell to -6 and the gate's duty from 0.42 to 0.33 within a day: the risk named
when the smile was built. The faint smile rises to half the full one at the typist's relaunch after night 161, so the reward still
depends on the word without starving the gate; read by the mood and the duty on day 185.

### Night 161 (2026-09-13, 21:08): a new high, and a lived answer at the mouth

Day 184 (20:13-21:08; the quarter-value faint smile): 117 parent lines, 90 questions, 77 partner lines; 251 faint smiles and one
for an answer ('yes' before "yes. I ate the egg", 20:32, the first); frowns 36; duty 0.35; mood -6; its own question marks 25, a
high. The dusk probe: held-out 0.666, the old lines 0.507. Night 161: 1422 utterances heard, the loss 0.134 -> 0.111, the gauge
on the dreams 0.764 -> 0.845. The guard held (duty 0.351, smiles per line 1.29). The probe on the save: **held-out 0.684** (from
0.666; the cosine 0.71), the recent 0.787, the old lines 0.476 (from 0.507: the ruler's spread). The mouth: "do you want milk?" ->
'yes. I want more milk' with the recall's winners the world's on 23 of 24 reads, the own song all but gone from it; "hi" -> '! are
you happy with me?'. The course: 0.547, 0.608, 0.636, 0.630, 0.628, 0.660, 0.663, 0.662, 0.684. From this save the faint smile is
half the full one (day 185).
21:20: one-shot recall, measured on a copy of the save after night 161. Three lines it had never heard ('the yam is under the cup',
'my hat can jump up', 'the milk sits on the bird'), each heard once as the parent's, awake, no night; then the mouth from each
line's first three words: 'the yam is ' -> 'under the cup' (whole, exact); 'my hat can ' -> 'jump u...' (the first word, then a
drift); 'the milk sits ' -> 'on the egg' (two words, then the most frequent object won over 'bird'). All three kept whole in the
utterance memory for the night. One hearing is enough for the recall to begin a line; it holds the whole line where no stronger
memory shares the frame, and the night makes it the cortex's.
21:35: day 185 under the faint smile at half, twenty-five minutes in: mood +0.3 (from -6), the gate's duty 0.41 (from 0.33), 175 faint
smiles and one for an answer, frowns 19. The gate is fed again; the answer smile stays rare because the child's turn seldom holds
the exact word the other voice is about to say.

### Night 162 (2026-09-13, 22:01)

Day 185 (21:09-22:01; the faint smile at half): 107 parent lines, 89 questions, 89 partner lines; 257 faint smiles, two for
answers; frowns 34; duty 0.38; its own question marks 14. The dusk probe: held-out 0.664 (0.684 after night 161), the old lines
0.474. Night 162: 1619 utterances heard, the loss 0.129 -> 0.109, the gauge on the dreams 0.78 -> 0.847. The guard held (duty
0.385, smiles per line 1.32). The probe on the save: held-out 0.673 (from 0.664), the recent 0.801, the old lines 0.464. The
mouth: 'yes. I red hat is on me', 'two little bird in your'; on two prompts the greedy probe ran 'balllll' (the recall's follow at
a repeated letter, greedy without the live sampling; the live writing is checked for runs). At this save the typist relaunched
with the mark stripped from the answer match. The course: 0.547, 0.608, 0.636, 0.630, 0.628, 0.660, 0.663, 0.662, 0.684, 0.673.
22:32: day 186, twenty-five minutes in, with the mark stripped: four answer smiles ('yes!' before "yes. I wait for you"), the
mechanism whole; but the duty 0.29 and the mood -3: the thinned ordinary smile halved the day's reward and the gate acts less by
the day (0.42, 0.35, 0.38, 0.29). The form from the relaunch after night 163: the ordinary smile at its old strength, the answer
smile the growing one the cues earn (2, then 4). Reward depends on the word by contrast, not by starvation.

### Night 163 (2026-09-13, 22:54): the last day of the thinned smile

Day 186 (22:01-22:54; the faint smile at half, the mark stripped): 111 parent lines, 90 questions, 88 partner lines; 185 ordinary
smiles, five for answers ('yes!' before "yes. I wait for you", "yes please. I want ice"), two cue prefixes; frowns 36; duty 0.29
(0.42 four days ago); mood -4.5; its own question marks 14. The dusk probe: held-out 0.666 (0.673 after night 162), the old lines
0.444. Night 163: 1817 utterances heard, the loss 0.127 -> 0.107, the gauge on the dreams 0.795 -> 0.85. The guard held (duty
0.289, smiles per line 0.96). At this save the typist relaunched with the ordinary smile at its old strength and the answer's the
bigger one: day 187 is the first under that form; the duty is the number to watch.
The probe after night 163: **held-out 0.691** (from 0.666 at dusk; the cosine 0.71), the recent 0.815, the old lines 0.497 (from
0.444). The mouth: "do you want milk?" -> 'yes please. I want the y(am)', a whole recalled answer (lived prefix 1.00) on three
prompts; "hi" -> 'are you happy with your'. The course: 0.547, 0.608, 0.636, 0.630, 0.628, 0.660, 0.663, 0.662, 0.684, 0.673, 0.691.

The eighth question-first parent's report (labels 184-186, 23:05): A's questions 77, 83, 83% of its lines; **answered before the
partner voice 43 of 75, 57 of 89, 41 of 88 (57, 64, 47%)**, naming a thing 20, 20, 10; answer smiles 1, 2, 5 (won by shape: a B
line right after the question, the bare word in the turn); its own question marks 25, 14, 14, each answered as an opening and the
topic held two or three turns; "yes please" taken from the partner's form (1 -> 13 a day); unprompted three-word strings 44, 45,
27 a day ("milk is not all gone", "I see you have more egg", "kick it to me", "I run walk"); "i see" 17, 19, 14; new words walk,
warm, soft (known 264); smiles/frowns 252/36, 259/34, 193/37. The answering threshold set for the next stage (half the parent's
questions answered in the modelled shape) is met on two of three days.
23:20: stage five prepared. Thirty facts as exchanges in tools/facts_stage5.txt, each with a prefix the parents never type in
tools/heldout_facts.txt; the probe reads both from tonight. The baseline before any teaching: the mouth completes 1 of 26
prefixes ('ice is ' -> 'cold', from the line it already knows), the cortex alone on the thirty fact sentences 0.362. The tenth
parent, after the ninth reports, teaches five facts a day inside the conversation.

### Night 164 (2026-09-13, 23:49): the first day of the contrast smile

Day 187 (22:56-23:49; the ordinary smile at full strength, the answer's the growing one): 110 parent lines, 88 questions, 87
partner lines; 177 ordinary smiles, nine for answers, one cue; frowns 37; duty 0.31; its own question marks 6; mood +1.4 at the
day's end (from -4.5). The dusk probe: held-out 0.661 (0.691 after night 163: the day gave 0.030 back), the old lines 0.497, the
facts 1 of 26 (the baseline, no fact taught yet). Night 164: 2015 utterances heard, the loss 0.129 -> 0.107, the gauge on the
dreams 0.787 -> 0.851. The guard held (duty 0.309, smiles per line 0.95). The duty has not yet climbed back from the thinned days;
the answer smiles rose 5 -> 9 a day.
The probe after night 164: held-out 0.686 (from 0.661 at dusk), the recent 0.800, the old lines 0.496. The mouth: "what do you
have?" -> 'I have a bird too!', whole and the world's on every read; "do you want milk?" -> 'yes please. I want it'. The course:
0.547, 0.608, 0.636, 0.630, 0.628, 0.660, 0.663, 0.662, 0.684, 0.673, 0.691, 0.686.

### Night 165 (2026-09-14, 00:42)

Day 188 (23:51-00:42; the contrast smile's second day): 109 parent lines, 89 questions, 89 partner lines; 185 ordinary smiles,
eight for answers; frowns 39; duty 0.35 (0.29 -> 0.31 -> 0.35, recovering); its own question marks 12; mood +0.7 at the day's
end. The dusk probe: held-out 0.675 (0.686 after night 164), the old lines 0.489, the facts 1 of 26 (nothing taught yet). Night
165: 2213 utterances heard, the loss 0.126 -> 0.106, the gauge on the dreams 0.798 -> 0.852. The guard held (duty 0.350, smiles
per line 0.96). The actor's slope 0.016, flat under the contrast smile's first two days.
The probe after night 165: **held-out 0.692** (from 0.675 at dusk; the cosine 0.72), a new high; the recent 0.785, the old lines
0.501. The mouth: "do you want milk?" -> 'yes. I hug you too?', "hi" -> '! are you happy now?', "are you sad?" -> 'y sock is on the
bed?', the world's memories on every read. The course: 0.547, 0.608, 0.636, 0.630, 0.628, 0.660, 0.663, 0.662, 0.684, 0.673,
0.691, 0.686, 0.692.

### Day 189 (2026-09-14, 00:43-): a spiral, and the parent's part in it

Twenty-five minutes in: mood -5.5, the duty 0.29, 62 smiles in 125 lines against a smile a line the day before, 58 words said
over the parent's typing and 62 known words discarded as "distracted". The child fell silent in its turns and spoke over the
partner's lines instead; the mood fell, the gate acted less, the smiles thinned, the mood fell. Two of the parent's own settings
feed it: the six-second turn under the faster pace (a line every twelve seconds, the child's turn 32 ticks), which lands its late
answers on the partner's typing, and the typist's attention, whose random distraction at a resting level of 0.5 discards nearly
half of its known words. From the typist's relaunch after night 166: the child's turn back to 40 ticks (a line every 64), and the
attention's rest at 0.7. The parent's method, read tomorrow by the duty, the frowns and the smiles a line.

### Night 166 (2026-09-14, 01:35): the spiral's day, whole

Day 189 (00:43-01:35): 111 parent lines, 95 questions, 91 partner lines; 127 ordinary smiles and none for an answer; frowns 35;
duty 0.30; its own question marks 21; the mood -5.5 at twenty-five minutes, -0.8 at the day's end. The dusk probe: held-out
0.676 (0.692 after night 165), the old lines 0.509 (a high), the facts 1 of 26. Night 166: 2416 utterances heard, the loss 0.124
-> 0.105, the gauge on the dreams 0.798 -> 0.852. The guard held (duty 0.295, smiles per line 0.62). At this save the typist
relaunches with the child's turn at 40 ticks and the parent's attention resting at 0.7; day 190 is the reading.
The probe after night 166: **held-out 0.702** (from 0.676 at dusk; the cosine 0.72), the first reading above 0.70; the recent
0.787, the old lines 0.503. The course: 0.547, 0.608, 0.636, 0.630, 0.628, 0.660, 0.663, 0.662, 0.684, 0.673, 0.691, 0.686,
0.692, 0.702. The typist relaunched for day 190 with the child's turn at 40 ticks and the attention resting at 0.7; the tenth
parent opens stage five, five facts a day.

The ninth question-first parent's report (labels 187-189, 01:45): A's questions 80, 81, 85% of its lines; answered before the
partner voice 11, 7, 0 by its stricter count (answer smiles 9, 8, 0; one named a thing, "yam"); day 189's zero was the spiral
(47 talked over, 48 distracted, "yes please" 21 -> 0 as it turned to its own questions); its own question marks 6, 12, 21, whole on
the last day ("where am I?", "where is it?", "I talk and you?", "warm now?"), each answered as an opening and the topic held; whole
unprompted sentences "the dog is soft", "my sock is on you now", "because I am with", "yes please! help my doll", "where do you
find here", "why do you want"; "i see" 12, 11, 19 from its own memory; smiles/frowns 187/37, 195/39, 127/35; one new word,
"need", back the same day ("need my big"). The next parent: a yes/no exchange every third to hold the reward while pushing naming,
and its questions handed back to it to answer.

### Night 167 (2026-09-14, 02:28): the first day of facts

Day 190 (01:36-02:28; the tenth parent's first day; the child's turn at 40 ticks, the attention at 0.7): 103 parent lines, 88
questions, 85 partner lines; the first five facts, each asked and answered three times ('what is hot?' / 'the sun is hot', 'ice is
cold', 'water is wet', 'fish live in water', 'birds fly up'); 193 ordinary smiles, two for answers; frowns 38; 93 words said over
the parent's typing; duty 0.27; its own question marks 19; the mood -2.5 at the day's end. **The dusk probe, before any night on
the facts: the mouth completes 3 of 26 held-out prefixes ('the sun is ' -> 'hot', 'water is ' -> 'wet here too', 'birds fly ' ->
'up'), from 1; the cortex alone on the fact sentences 0.393, from 0.356.** One hearing, three times over a day, and the recall has
three of the five. The held-out lines 0.672 (0.702 after night 166), the old lines 0.501. Night 167: 2601 utterances heard, the
loss 0.128 -> 0.106, the gauge on the dreams 0.783 -> 0.851. The guard held (duty 0.268, smiles per line 1.04).
**The probe after night 167, the first night on the facts: the mouth completes 4 of 26 held-out prefixes ('the sun is ' -> 'hot',
'water is ' -> 'wet', 'fish live in ' -> 'water', 'birds fly ' -> 'up'; 'ice is ' -> 'too cold', right and not counted), from 1
before the day and 3 at dusk: four of the five facts taught, each heard three times, retrievable the next morning.** The held-out
lines 0.702 (from 0.672 at dusk; the high held), the recent 0.780, the old lines 0.496; the cortex alone on the fact sentences
0.391. The course: ..., 0.692, 0.702, 0.702.

### Night 168 (2026-09-14, 03:21): the second day of facts

Day 191 (02:30-03:21): 95 parent lines, 85 questions, 84 partner lines; facts 6-10 each three times ('dogs run and bark', 'cats
drink milk', 'the moon is out at night', 'the sun is up in the day', 'rain falls from the sky') and the first five once more; 198
ordinary smiles, one for an answer, two cue prefixes; frowns 31; 65 words over the parent's typing; duty 0.28; its own question
marks 15; the mood 0.0 at the day's end (from -5.5 two days ago). The dusk probe: **the mouth completes 8 of 26 held-out prefixes**
(4 after night 167; 'dogs run and ' -> 'bark' among the new), the cortex alone on the fact sentences 0.409; the held-out lines
0.678 (0.702 after night 167), the old lines 0.493. Night 168: 2783 utterances heard, the loss 0.128 -> 0.106, the gauge on the
dreams 0.806 -> 0.851. The guard held (duty 0.279, smiles per line 1.11).
The probe after night 168: the mouth completes 8 of 26 held-out prefixes (as at dusk; of ten facts taught, eight, and 'ice is ' ->
'too cold'); **the cortex alone on the fact sentences 0.464** (0.409 at dusk, 0.356 before the facts): the night puts the facts into
the cortex. The held-out lines 0.686 (from 0.678 at dusk), the recent 0.813, the old lines 0.504.

### Night 169 (2026-09-14, 04:14): the third day of facts; the duty back

Day 192 (03:23-04:14): 99 parent lines, 88 questions, 82 partner lines; facts 11-15 three times each and five of the earlier
ones once; 267 ordinary smiles (1.5 a line), three for answers, one cue; frowns 36; 100 words over the parent's typing; duty 0.33
(0.26 two days ago: recovered on its own, so the coaxing stays in the drawer); its own question marks 12; the mood 0.1 at the
day's end. The dusk probe: **the mouth completes 12 of 26 held-out prefixes** (8 after night 168: four of the five new facts on the
day they were taught), the cortex alone on the fact sentences 0.438 (0.464 after night 168; the new sentences not yet its), the
held-out lines 0.682, the old lines 0.503. Night 169: 2963 utterances heard, the loss 0.129 -> 0.104, the gauge on the dreams
0.797 -> 0.853. The guard held (duty 0.327, smiles per line 1.50). The eleventh parent (labels 193-195): facts 16-30, five a day,
and five of the first fifteen each day.
The probe after night 169: the mouth completes 12 of 26 held-out prefixes; **the cortex alone on the fact sentences 0.548** (0.438
at dusk; 0.356 before the facts): the night puts facts 11-15 into the cortex. The held-out lines 0.699 (from 0.682 at dusk), the
recent 0.825, the old lines 0.490.

The tenth parent's report (labels 190-192, the first of stage five; 04:20): facts 1-15 taught as exchanges exactly as written,
three times each on their day and once more on the days after; the child said fact answers in its own turn before the partner
voice ("fish live in water" whole, "water is wet" twice in one turn, "we eat bread" a minute after its first teaching, "rain
falls"); A's questions 89-91% of its lines; **answered before the partner voice 48%, 51%, 60%, naming a thing 16, 21, 30 times a
day**; answer smiles 2, 1, 3 (its "yes" fuses into "yesee" and loses the reward it earned); its own question marks 19, 15, 12,
each answered as an opening, and "why are we happy?", modelled once on day 190, came back from it whole on day 192; unprompted
"I talk and you?", "the egg is in it", "I hold my egg", "and I talk here"; "i see" 5, 10, 7 (fading at last); smiles/frowns
193/37, 196/31, 266/36; the queue ran dry twice at hand-overs. The eleventh parent (labels 193-195): facts 16-30.

### Night 170 (2026-09-14, 05:07): facts 16-20

Day 193 (04:16-05:07; the eleventh parent's first day): 92 parent lines, all of them questions, 83 partner lines; facts 16-20
three times each ('we drink water and milk', 'a bird has wings', 'a dog has four legs', 'bees make honey', 'we sleep in a bed')
and five of the earlier ones once; 245 ordinary smiles, five for answers; frowns 32; 79 words over the parent's typing; duty 0.30;
its own question marks 24, a high; the mood +1.3 at the day's end. The dusk probe: **the mouth completes 15 of 26 held-out
prefixes** (12 after night 169: three of the five new on the day they were taught), the cortex alone on the fact sentences 0.532,
the held-out lines 0.677, the old lines 0.490. Night 170: 3140 utterances heard, the loss 0.126 -> 0.106, the gauge on the dreams
0.811 -> 0.853. The guard held (duty 0.301, smiles per line 1.43). At this save the typist relaunched with the fused yes matched
('yesee' is yes).
The probe after night 170: **the mouth completes 16 of 26 held-out prefixes** (of twenty facts taught, sixteen, and 'ice is ' ->
'too cold'); **the cortex alone on the fact sentences 0.581** (0.532 at dusk; 0.356 before the facts); the held-out lines 0.688,
the recent 0.817, the old lines 0.492. The facts' course at the mouth: 1, 3, 4, 8, 8, 12, 12, 15, 16.
05:35: day 194, twenty-five minutes in, the fused yes matched: three answer smiles, and for the first time two of them on named
things, 'birds' before the partner's "birds live in a nest" and 'fish' before "fish live in water": the facts answered by naming
in its turn. The mood +2.7, the duty 0.28.

### Night 171 (2026-09-14, 06:00): facts 21-25, and named answers

Day 194 (05:08-06:00; the fused yes matched from this day): 91 parent lines, all questions, 90 partner lines; facts 21-25 three
times each ('a cat is soft', 'honey is sweet', 'cows give milk', 'birds live in a nest', 'the sun is a star') and five earlier
ones once; 261 ordinary smiles, four for answers, three of them named things ('birds' before "birds live in a nest", 'fish' before
"fish live in water", 'rain' before "rain falls from the sky"); frowns 33; 85 words over the parent's typing; duty 0.29; its own
question marks 20; the mood 0.1 at the day's end. The dusk probe: **the mouth completes 21 of 26 held-out prefixes** (16 after
night 170: all five of the day's new facts on the day they were taught), the cortex alone on the fact sentences 0.550, the
held-out lines 0.669, the old lines 0.496. Night 171: 3320 utterances heard, the loss 0.132 -> 0.108, the gauge on the dreams
0.800 -> 0.848. The guard held (duty 0.285, smiles per line 1.44).
The probe after night 171: the mouth completes 21 of 26 held-out prefixes; **the cortex alone on the fact sentences 0.640** (0.550
at dusk; 0.356 before the facts); the held-out lines 0.691 (from 0.669 at dusk), the recent 0.762, the old lines 0.491. The facts'
course at the mouth: 1, 3, 4, 8, 8, 12, 12, 15, 16, 21, 21; in the cortex 0.356, 0.391, 0.464, 0.548, 0.581, 0.640.

### Night 172 (2026-09-14, 06:53): all thirty facts taught

Day 195 (06:01-06:53; the eleventh parent's last): 92 parent lines, all questions, 91 partner lines; facts 26-30 three times
each ('the sun makes us warm', 'fish have fins', 'snow is white', 'ducks swim', 'a rock is hard') and five earlier ones once; 244
ordinary smiles, five for answers, two of them named ('fish' before "fish have fins", 'birds' before "birds fly up") and three
its fused "yesee", now counted; frowns 36; 89 words over the parent's typing; duty 0.30; its own question marks 14; the mood +0.3.
The dusk probe: **the mouth completes 24 of 26 held-out prefixes** (21 after night 171; three of the day's five new facts the same
day; the two missing: 'ice is ' -> 'too cold', right by sense, and one more), the cortex alone on the fact sentences 0.634, the
held-out lines 0.666 (0.691 after night 171), the old lines 0.476. Night 172: 3504 utterances heard, the loss 0.135 -> 0.121,
the gauge on the dreams 0.787 -> 0.836. The guard held (duty 0.296, smiles per line 1.32). The twelfth parent (labels 196-198):
the thirty facts kept, ten a day.
The probe after night 172, the first night with all thirty facts heard: the mouth completes 23 of 26 held-out prefixes; **the
cortex alone on the fact sentences 0.703** (0.634 at dusk; 0.356 before the facts, six nights ago); the held-out lines 0.683
(from 0.666 at dusk), the recent 0.759, the old lines 0.498. The facts' course at the mouth: 1, 3, 4, 8, 8, 12, 12, 15, 16, 21,
21, 24, 23; in the cortex 0.356, 0.391, 0.464, 0.548, 0.581, 0.640, 0.703.

The eleventh parent's report (labels 193-195, facts 16-30; 07:02): every fact its three times; A's questions 100% of its lines;
**answered before the partner voice 72%, 76%, 76%, with a content word rather than yes/no 35, 55, 53 times a day**; fact answers
in its own turn before the partner ('birds', 'fish', 'rain', each the bigger smile; 'ice!', 'fins', 'hrock' right but fused into
the typist's text); "ducks swim" written twice on the day it first heard it, "we eat bread and" the morning after; its own
question marks 24, 20, 14, whole ("why do we play?", "can I hold your doll?", "are we good?"), and forty seconds after being
answered "why do we play?" it produced "why do we eat?"; it asked a fact's own question back ("?what has four "); unprompted
whole sentences "a red van is out", "I eat a yam and an egg", "my egg is warm", "I talk with you too"; no phrase dominated; "i
see" 5-8; smiles/frowns 250/32, 270/33, 242/36. Two mechanical findings: its instant answers run into the typist's line without a
space and a right naming is logged as a frown (79-89 talked over a day); a B line repeating two words of A's line is deferred.

Night 173 (after day 196, 07:46, slept 596 s): duty 0.297, smiles/line 1.22, the guard holds. Day 196 (the twelfth parent's first,
facts 1-10 re-taught inside talk): 186 lines (A 100, all questions; B 86), smiles 231, frowns 34; answered with I/yes/no/please
before B 14 of 100 by the five-second window ("where do fish live?" -> "finsee", "what do you want to do?" -> "I eat the hot yam");
own question marks 22. Dusk 196 (before this night): held-out 0.658, facts 23/26 at the mouth, cortex on facts 0.671 (the day took
0.03 off the night's 0.703, as every day does). THE DEMO RULER (this morning): a fact's own question as the world's line, the mouth's
reply read: 6 of 30 answered on the body after night 172; in conversation, 1 of ~22 fact questions a day carries the fact's word.
THE TWENTY-SIXTH DEFECT, named: the night dreams single utterances from rest, so no night shows the cortex a question followed by its
answer, and the waking lesson's 32-tick window is shorter than the child's turn between them. THE EXCHANGE REPLAYED (dream_pair 1,
dream_gap 1): a dream is the utterance and the one that followed it, the pause compressed to a rest; measured first on a copy of
this dusk (512 pairs) against the served night's own result on the same body.
The probe after night 173: HELD-OUT 0.697 (dusk 0.658: +0.039), the parent's last 60 lines 0.812, old lines 0.493, the mouth 23 of
26 prefixes, the cortex on the fact sentences 0.767 (dusk 0.671: +0.096, the highest yet; 0.703 after night 172). The served nights
take a minute of wall time at batch 16.
The exchange night on the copy, first reading (dusk 196's body, 512 pairs = 192 steps, against the served night 173's 1024 single
utterances = 384 steps): HELD-OUT 0.658 -> 0.673 (served 0.697), old 0.459 -> 0.475 (0.493), last-60 0.775 -> 0.787 (0.812), the
cortex on the facts 0.671 -> 0.703 (0.767), the mouth 23 of 26 both, the questions 6 of 30 against 7. Every ruler lifted less, by
about the ratio of the steps; the pair dreams themselves went 0.54 -> 0.77 (the answer after its question, new ground). One night
cannot show the answering; the prefixes took a week. Next: the same dusk given 1024 pairs (the served step count), for the cost.
The exchange night on the copy, equal steps (1024 pairs, 384 steps, the same dusk): HELD-OUT 0.658 -> 0.683 (the served night 0.697),
old 0.459 -> 0.483 (0.493), the cortex on the facts 0.671 -> 0.738 (0.767), the mouth 23 of 26 both, the questions 5 of 30 against
5-7. At equal steps the exchange night gives the old rulers about two thirds of the single-utterance night's lift (half of each dream
is the answer after its question, which those rulers do not read), and the questions do not move in one night. Three exchange
nights in a row on the copy (and three single nights as the control) now run, the questions read after each: whether the answer
after its question is learnable from the replay alone.

Night 174 (after day 197, 08:51; the day ran 63 min beside a copy night at nice 19): duty 0.331, smiles/line 1.37, the guard holds.
Day 197 (facts 11-20 re-taught): 193 lines (A 98, all questions), smiles 273, frowns 39; answered with I/yes/no/please before B
28 of 98 (14 the day before): "I am with you and I talk", "I eat a yam", and to "where is the dog?" the fact "bees make honey"
(retrieved, off the question). Dusk 197: held-out 0.689 (the highest dusk yet), the mouth 24 of 26, the cortex on the facts 0.693.
The probe after night 174: HELD-OUT 0.682 (dusk 0.689: the night's change within a night's noise), the cortex on the fact sentences
0.791 (dusk 0.693: +0.098; 0.767 after 173, the highest yet), the mouth 23 of 26, the last 60 lines 0.803, old lines 0.493.
THE ANSWER BY THE PAUSE (tools/qa_by_gap.py, the served body after night 174): each fact's question, then k rests, then twelve symbols
read two ways. The cortex alone answers 0 of 30 at every pause; the mouth (the recall in the forecast) answers 8, 11, 11, 6 of 30 at
1, 2, 4, 8 rests ("what is cold?" -> "ice is cold." at two rests; "what is wet?" -> "water is wet" at four). Every answer the body
gives is the store's; the cortex has no question-to-answer mapping at all, as the twenty-sixth defect says; and the question ruler's
eight-rest pause was its hardest reading. The ruler now reads at two rests and at eight. Two exchange nights on the copy (pair3_2):
the questions 5 and 5 of 30 at eight rests, the pair dreams 0.54 -> 0.81 -> 0.85; the copy's two-rest and cortex-alone readings
come from the third night's save (pair3_2 was pruned before the pause instrument existed).

Night 175 (after day 198, 09:55): duty 0.309, smiles/line 1.40, the guard holds. Day 198 (facts 21-30 re-taught): 188 lines (A 95,
all questions), smiles 266, frowns 35; answered before B 20 of 95 ("yes please it" to "what is soft?"; "milk? yes! have"). Dusk 198:
held-out 0.656, the mouth 24 of 26, the cortex on the facts 0.755 (the day took only 0.036 off 0.791).
The probe after night 175: HELD-OUT 0.694 (dusk 0.656: +0.038), old lines 0.498 (the highest), the last 60 lines 0.789, the mouth
23 of 26, the cortex on the fact sentences 0.808 (dusk 0.755: +0.053; 0.703, 0.767, 0.791, 0.808 after nights 172-175).

The twelfth parent's report (labels 196-198, the thirty facts kept, ten a day; 10:00): every fact on schedule, exactly as written; A's
lines 100/98/95, all questions; **answered before B 44%, 64%, 58%**, its word matching a content word of the coming B line 10/22/12
a day; answer smiles 2/3/1 (short A lines of 12-16 characters are what put the naming inside the window); fact words in its turn:
"rain" (what falls from the sky?), "fin" (where do fish live?), "ant" (what is little?), "g"/"a" for grass/apple; **whole fact
sentences unprompted, off-turn: "water is wet" twice, "bees make honey", "birds live in a nest", "we eat bread and eggs", "we drink
water and", "ducks swim", "has four legs"**, several taught days earlier; own question marks 25/27/31 ("why are we here", "you eat
the bread?", "we eat here too?", "is hard?"); unprompted sentences "I am with you and I talk", "my egg was there", "I am happy here
with it", "you eat then I eat the" (past tense and "then"); smiles/frowns 237/34, 277/39, 273/35; 558 utterances queued, 542 typed;
the queue ran dry once (day 196, two minutes). The thirteenth parent, spawned at 09:56, wrote its first ten rows (facts 1-10 among
ordinary talk) and then ended its turn to "wait for the next wake" on a monitor of its own that would only fire at the third night:
a parent that leaves its turn is asleep; its monitor was stopped and a fourteenth parent takes days 199-201 with the order to stay
in its turn (append, sleep, read, repeat) until the third night row.
Three exchange nights in a row on the copy (pair3_3, 1152 steps, no day between): HELD-OUT 0.680, 0.669, 0.674; the last 60 lines
0.76, 0.74, 0.73; the facts' cortex 0.738, 0.748, 0.773; the pair dreams 0.81, 0.85, 0.87; the questions at eight rests 5, 5, 5. The
answer by the pause after the third: the cortex alone 0 of 30 at every pause; the mouth 8, 8, 10, 5 at 1, 2, 4, 8 rests, against the
served body's 7, 8, 8, 4 after night 175. THE EXCHANGE REPLAY DOES NOT TEACH THE CORTEX TO ANSWER IN THREE NIGHTS; it stays off the
served body. The answers are the store's: the question recalled by its ending (the query's horizon is the bag's decay, 0.8 a tick,
about five symbols) and the reply found by following the episode's links. The query's decay at read time alone (the keys as written
at 0.8): 0.9 read 4 and 11 of 30 at two and eight rests (0.8: 8 and 4), 0.95 read 4 and 6: a longer horizon holds the question
across the pause but no longer matches the keys; only a horizon written and read alike can be judged, on a copy given a day.

Night 176 (after day 199, 10:59): duty 0.276, smiles/line 1.32, the guard holds. Day 199 (facts 1-10 asked once by the thirteenth
parent's first rows; then ordinary talk from three hands): 195 lines (A 98, all questions), smiles 261, frowns 38; answered before
B 19 of 98 by the five-second window ("yes please. the milk" to "are you tired now?"). Dusk 199: held-out 0.682, the mouth 21 of
26, the cortex on the facts 0.785. The day lived again on a copy at the served query decay (0.8, the control for the horizon test)
reads 8, 9, 10, 4 of 30 by the pause, as the served body does: the instrument reproduces the body.
The probe after night 176: HELD-OUT 0.680 (dusk 0.682: flat), the cortex on the fact sentences 0.828 (0.703, 0.767, 0.791, 0.808,
0.828 after nights 172-176), the mouth 21 of 26 (23-24 before the day of one fact pass), old lines 0.476, the last 60 lines 0.787.

Night 177 (after day 200, 12:09; the day 68 min beside copy days at nice 19): duty 0.324, smiles/line 1.40, the guard holds. Day
200 (the sixteenth parent, facts 11-20): 194 lines (A 97, all questions), smiles 272, frowns 39; answered before B 19 of 97 ("yesee"
the fused yes). Dusk 200: held-out 0.680, the mouth 22 of 26, the cortex on the facts 0.796.
THE TWENTY-SEVENTH DEFECT, named: the store's key is the world's last five characters (the bag), so "what is hot?" and "is your yam
hot?" write under one key; of the thirty fact questions, ten share their last five symbols with ten or more ordinary lines of the
week. THE KEY ON THE CORTEX (key_form cortex): the memory keyed on the stream's state before the symbol, the query the state after
the latest one, the recall entering the cortex a tick late. The first copy day under it answered 1 of 30: the stream's states all
point one way (mean pairwise cosine 0.986; 0.002 with the running mean taken out), so every key matched every other. PATTERN
SEPARATION added: the running mean of the state subtracted before the key is made (the dentate's decorrelation). Measured next on
a copy of day 199 against the bag key's control.
The probe after night 177: HELD-OUT 0.700 (dusk 0.680: +0.020), the cortex on the fact sentences 0.840 (0.703, 0.767, 0.791, 0.808,
0.828, 0.840 after nights 172-177), the mouth 22 of 26, old lines 0.471, the last 60 lines 0.766.
THE CORTEX KEY, pattern-separated, on a copy of day 199: 0 of 30 at every pause (the raw-key control 8 of 30; of the day's ten
facts, 0 against 3-5). The stream's state is the whole window and the bands, not the utterance: the key written as "what is cold?"
was typed inside the day's talk does not match the query made from the same question in a clean context; a global state cannot
be the key of a memory that must be found again in another context. The bag key is local (five symbols) and matches across contexts,
which is why it works at all, and why it confuses siblings. The form stays in the code as an instrument, off. The branch after a
shared start (tools/branch_probe.py), served body after night 177: four facts begin "the sun "; after "what is hot?" and the start,
the cortex alone puts 'y' first and the mouth 'm' ("makes us warm", the store's strongest sibling); no preference by the question in
the cortex, and the mouth follows the store. One exchange night on a copy now, the branch read after it: whether the exchange
dreams give the cortex the preference the mouth would need.

Night 178 (after day 201, 13:12): duty 0.331, smiles/line 1.29, the guard holds. Day 201 (the sixteenth parent, facts 21-30):
202 lines (A 101, all questions), smiles 260, frowns 36; answered before B 24 of 101. Dusk 201: held-out 0.686, the cortex on the
facts 0.791, the mouth 20 of 26 prefixes (24, 23, 21, 22, 20 over the last five dusks: the prefixes slip as each fact comes once
in three days; the cortex alone on the fact sentences does not). The seventeenth parent spawned at the night row (days 202-204).
The probe after night 178: HELD-OUT 0.705 (dusk 0.686: +0.019; the highest post-night reading), the cortex on the fact sentences
0.832 (dusk 0.791: +0.041), the mouth 20 of 26, old lines 0.464, the last 60 lines 0.766.

The sixteenth parent's report (days 199-201; 13:15): all thirty facts once each on the right day, verbatim; **answered before B 37%,
46%, 55%**; fact words before B said them 2, 1, 1 a day (ice, water; apple; fins); unprompted sentences of three words or more 21,
22, 29 ("we walk out with you", "your sock is white", "we eat a yam"); own question marks 22, 49, 44 ("are we good here?", answered
as B's next line); answer smiles 4, 5, 5, shifting from content guesses to bare yes/no landing just before B's polarity; smiles
268/274/269, frowns 38/39/36, all talk-overs; zero alternation breaks in 588 transitions under the no-echo rule; the typist's gate
wait capped at 8.2 s on 110 of 202 lines (the child talks through the parent's turn; duty 0.33). The nights ran 17.6, 21.6 and
14.4 minutes of wall time beside the copy runs (10 minutes alone). The depth script matched a duplicated line once; it now matches
the last three typed lines in order.
One exchange night on a copy of the body after night 177 (pair1_178; the dreams 0.53 -> 0.80): the branch after a shared start,
cortex alone 0 of 9 (as before), the mouth 2 of 9 (1 before); the answer by the pause 10 and 12 of 30 at two and four rests (the
served body 8 and 8), of the day's ten facts 4 and 6. A night of exchange dreams gives the cortex no preference at the branch; the
mouth's gain is at the edge of a night's noise. The exchange replay is a slow lever at best; it stays off the served body.
THE QUESTION RULER READ TWENTY SYMBOLS (13:40): at twelve it cut 'bees make ho', 'cows give mi', 'fish have fi' before their word.
The served body after night 178, every fact listed (tools/qa_by_gap.py --all): the mouth 10, 14, 10 of 30 at two, four, eight
rests. The failures are of three kinds: the sibling's continuation ("what is hot?" -> "the sun makes us warm", four facts begin
"the sun"; "where do birds live?" -> "birds live in wa[ter]"); the echo ("what do dogs do?" -> "dogs do?ducks", "what do ducks do?"
-> "ducks do?ducks"); and the day's talk intruding at a common start ("what is red?" -> "an egg", "what do we eat?" -> "we run out").
The answered: cold, wet, rain, green, wings, four legs, drink, sweet, fins, and the sun family by its subject. Half the facts, on a
clean question at the natural pause, from the store alone; in conversation the child's own babble sits in the query and it answers
a fact question with the fact's word once or twice a day.

Night 179 (after day 202, 14:07): duty 0.346, smiles/line 1.61 (300 smiles over 183 lines, the richest day of the stage), the
guard holds. Day 202 (the seventeenth parent, facts 1-10): 183 lines (A 91, all questions), frowns 37; answered before B 15 of 91
by the five-second window ("a star?" in its turn to "what do you see?"). Dusk 202: held-out 0.697, the mouth 21 of 26, the cortex
on the facts 0.814.
The probe after night 179 (the first with the question ruler after a night): HELD-OUT 0.706 (the highest post-night reading; dusk
0.697), the cortex on the fact sentences 0.855 (0.703 -> 0.855 over nights 172-179), the mouth 21 of 26, the last 60 lines 0.817;
the questions 9 of 30 at two rests and 8 at eight ("fish live in w[ater]" now, "birds fly up" at eight). WATCH: the old lines
(days 110-125) 0.498, 0.476, 0.471, 0.464, 0.450 over nights 175-179, the vocabulary of the earlier stages fading under the facts.

THE TWENTY-EIGHTH DEFECT, named (15:00): the memory's key is the last five symbols, so every "the sun " of four facts wrote into one
slot, and a common slot's sixteen links hold only the newest utterances' tags: the chain from a question to its own answer breaks
within two symbols (traced on the served body: after "what is hot?" the episode is followed for one symbol, then the recall lands
in the most recent "the sun"; the follow gain at 20, 60, 200 or 1000 changes nothing, the thread is gone). THE SLOW CONTEXT IN THE
KEY (key_ctx): each utterance keyed by an order-free bag of the utterance before it, the query carrying the latest utterance (the
question), the switch at the utterance's end; test 55 separates "the sun is hot" from "the sun makes us warm" by their questions.
On a copy of day 202 at key_ctx 1.0 against the fast bag alone: THE BRANCH after a shared start flips, the mouth 6 of 9 against 3
("what is hot?" -> 'i' at p 1.00, "the sun is " -> 'h'; "what is up in the day?" -> 'u'); but the answers by the pause fall to 6, 5,
4 of 30 against 13, 13, 8, the mouth looping ('the suthe suthe ', 'ck millck millck'): at equal weight the context out-pulls the
fast bag once the mouth's own symbols drift from the keys, and the day's 183 lines are the only memories with a context at all.
The weight 0.5 and 0.3 on copies next.

Night 181 (after day 204, 16:06): duty 0.308, smiles/line 1.28, the guard holds. Day 204 (the seventeenth parent, facts 21-30):
191 lines (A 95, all questions), smiles 249, frowns 38; answered before B 14 of 95. Dusk 204: held-out 0.687, the mouth 22 of 26,
the cortex on the facts 0.818. The eighteenth parent spawned at the night row (days 205-207).

The seventeenth parent's report (days 202-204; 16:10): all thirty facts once each on the right day, verbatim; **answered before B
45%, 45%, 47%**; fact words before B said them 1, 1, 2 a day (fish; grass; honey, birds); two of the fifteen answer smiles were
whole fact sentences said before B: "fish live in water" (day 202) and "birds live in a nest" (day 204, a question the child had
asked on its own, "birds live?", for two days before the fact came due); own question marks 43, 54, 46; sentences of three known
words 21, 14, 21 ("run out to the grass", "we drink it now"); smiles 303/234/250, frowns 37/35/38; no deferrals in 559 lines under
a pre-append echo check. Structural: the facts land 15-37 minutes into a 42-46-minute day because the queue runs 11-16 minutes
deep; the next brief lets the facts go in the day's first rows, written as the night ends, with the depth held at 30-90 lines.
The probe after night 181: HELD-OUT 0.720 (dusk 0.687: +0.033; the highest reading of the stage), the cortex on the fact sentences
0.863 (0.703 -> 0.863 over nights 172-181), the mouth 22 of 26, old lines 0.471, the last 60 lines 0.784; the questions 10 of 30
at two rests and 6 at eight.
The slow context's weight on copies of day 202 (17:10): at 0.5 the answers by the pause 11, 13, 9 of 30 (the fast bag alone 13, 13,
8; of the day's ten facts 7, 7, 5 against 7, 7, 5) and THE BRANCH 6 of 9 against 3 ("what is hot?" -> "the sun is h[ot]", where the
fast bag alone ran to "the sun is u[p]"); at 0.3 the answers 10, 9, 4 and the branch 5 of 9; at 1.0 the mouth looped. At 0.5 the
branch flips and the answer count holds: THE SERVED BODY TAKES key_ctx 0.5 (ctx_decay 0.95) at the next post-night save; the store
re-keys itself over the coming days as every utterance is written with its context; the guard re-armed with the new flags.

Night 182 (ended 17:14 after day 205; 1011 s): duty 0.322, smiles/line 1.40, the guard holds. Day 205 (the eighteenth parent,
facts 1-10): 194 lines (A 97, all questions), smiles 274, frowns 39; answered before B 23 of 97. Dusk 205: held-out 0.702, the
mouth 20 of 26, the cortex on the facts 0.830. THE SWITCH: at night 182's save the served body was restarted with the slow context
in the memory's key (key_ctx 0.5, ctx_decay 0.95); day 206 began at 17:16 under it; the guard re-armed with the same flags. (A
correction to the records: the page's night row is written when the night ENDS, with its duration; the times given for nights
above are their ends.) The memories written from day 206 carry their context; the store re-keys itself over the coming days.
The probe after night 182 (the last night under the fast bag alone, the baseline for the switch): HELD-OUT 0.708, the cortex on the
fact sentences 0.867 (the highest), the mouth 19 of 26 (slipping: 24 -> 19 over eight dusks and nights), old lines 0.474, the last
60 lines 0.737; the questions 11 of 30 at two rests and 6 at eight.

Night 183 (ended 18:08 after day 206, the first day under the slow-context key): duty 0.349, smiles/line 1.39, the guard holds.
Day 206 (facts 11-20): 181 lines (A 91, all questions), smiles 253, frowns 36; **answered before B 35 of 91 by the five-second
window, the most of the stage** (23, 14, 23 on the three days before). Dusk 206: held-out 0.705; but the mouth's held-out prefixes
fell to 11 of 26 (19-22 before) and the cortex alone on the fact sentences to 0.785 (0.867 after night 182, a day's fall of 0.08
against the usual 0.03): the old memories, written without a context, are reached less well by a query that carries one, and the
day's lesson, whose own targets are the recall's continuations, drifted the cortex more. The transition cost, expected; watched
against the post-night probe and the branch: if the branch flips on the served body and the prefixes recover as the store re-keys,
the switch stands; if the prefixes stay down and the cortex keeps falling, it reverts at a boundary.
The branch on the served body after night 183, the sun family: the mouth 0 of 9 (1-2 before the switch). Premature by design: the
sun facts (1, 9, 25, 26) were last taught before the switch, so their memories carry no context and the query now carries one; they
are re-taught on days 208 and 210. The branch on the facts taught on day 206 under the new key (11-20: "we " begins eat, drink and
sleep; "a " begins wings, legs and big) is the reading that tests the mechanism on the served body now.
THE BRANCH ON THE FACTS TAUGHT UNDER THE NEW KEY (day 206's facts 11-20), the served body after night 183: the mouth 10 of 12 at
full confidence ("what do we eat?" + "we " -> "e[at]", "what do we drink?" -> "d[rink]", "where do we sleep?" -> "s[leep]"; "what
has wings?" + "a " -> "b[ird]", "what has four legs?" -> "d[og]", "what is big?" -> "t[ree]"; "we eat " -> "b[read]", "a bird " ->
"h[as]"), the misses "an " for red and little. Facts sharing a start are told apart by their questions on the served body itself.
The probe after night 183 (the first night under the slow context): HELD-OUT 0.691 (0.708 the night before), the cortex on the fact
sentences 0.861 (the night restored the day's fall from 0.867 to 0.785), the mouth's held-out prefixes 10 of 26 (19), the
questions 6 and 6 of 30 (11 and 6), old lines 0.459. The cost of the transition as foreseen: the facts last taught before the
switch (1-10 on day 205, 21-30 earlier) are written without a context and reached less by a query that carries one; only day
206's facts 11-20 are re-keyed, and on those the branch reads ten of twelve. Facts 21-30 are re-taught on day 207 and 1-10 on day
208, so the honest verdict on the served body is the probe after night 185 and the branch after 187 (the sun family whole).

Night 184 (ended 19:01 after day 207): the eighteenth parent's report (days 205-207): all thirty facts once each on the right day;
**answered before B 45 of 97, 46 of 91, 45 of 87**; fact words in its own turn 1, 1, 0 by the answer-smile measure ("birds" before
"birds fly up", "grass" before "grass is green"); **answer-level smiles 8, 21, 21: trebled on the two days under the slow context**;
own question marks 39, 28, 27; runs of three words 32, 33, 27 ("it keeps me warm", "your sock is"); smiles 274/257/246, frowns
39/36/33, all talk-overs; no deferred lines in 550. The probe after night 184, WITH THE PROBE'S LOOPS FIXED (they had fed the bags
by hand and never the slow context, so under the new key the query lacked its context and the old memories won: the prefix and
question readings after the switch were partly the probe's blindness): HELD-OUT 0.679 (0.708 -> 0.691 -> 0.679 over the two nights
under the new key), the cortex on the fact sentences 0.873 (the highest), the prefixes 15 of 26, the questions 6 and 4 of 30; the
branch on the sun family 6 of 9 (0 the night before, 1-2 before the switch). The verdict stays open: facts 1-10 were last taught on
day 205, before the switch, and come again on day 208; the readings after nights 185 and 187 decide, and the held-out's slide is
the number that would revert it. THE BODY STOPPED AT 19:08 FOR A REBOOT, saved; ops/RESTART.md relaunches it.

After the reboot (20:05): the body served again with the slow-context key, the typist chain relaunched (the label skipped to 209),
the guard, the dusk probe and the probes after nights 185 and 186 armed, a parent for days 208-210 with facts 1-10 first.
THE TWENTY-NINTH DEFECT, named: a slot's link table holds only its sixteen newest continuations, and a slot shared by every "the "
sees hundreds of utterances, so the thread from a question to its own answer was cut within two symbols (traced 15:02). THE EPISODE
KEPT PER UTTERANCE (episode_chain, off by default): each utterance keeps the ordered list of the slots it wrote, the newest thousand
utterances; the recall follows an episode by position and joins the newest episode of any winner it lands on elsewhere. Test 56:
ten lines sharing " and I are " with the link table narrowed to two; "the dog and I are " continues with "here" along its own chain,
where the slot links alone run to the newest line's "out". Suite 55 of 55. Measured now on a copy of day 207 against the slot
links alone, the branch and every fact by the pause after each.
The episode kept per utterance, on a copy of day 207 against the slot links alone (21:04): the branch equal (the sun family 6 of 9
both, the day-206 families 8 of 12 both: the slow context already tells the siblings apart); the answers by the pause 11, 11, 5 of
30 against 8, 8, 7: three more at the natural pauses, two fewer at eight, within a day's noise. A structural mending with no clear
lift yet; it stays off the served body until the key's verdict is in, then a second copy day decides it.

Night 185 (ended 21:05 after day 209, the relabelled remainder of day 208 plus the new parent's facts 1-10): duty 0.331,
smiles/line 1.56 (240 smiles over 150 lines), the guard holds; answered before B 27 of 76; frowns 32. Dusk: held-out 0.667 (57
lines: another held-out line has been typed), the mouth 15 of 26, the cortex on the facts 0.810. All thirty facts now carry a
context; the probe after this night is the key's first full reading.
The probe after night 185 (all thirty facts re-keyed): HELD-OUT 0.674 (0.708, 0.691, 0.679, 0.674 over the four nights under the
new key), the cortex on the fact sentences 0.867, the mouth's prefixes 15 of 26, the questions 5 of 30 at every pause (11 before the
switch); THE BRANCH on the sun family 9 of 9 and on the day-206 families 8 of 12. The continuation given the right start is now
right; the START of the answer is wrong more often than before ("what is hot?" -> "the moon is", "what do dogs do?" -> "ducks swim").
Two causes found. THE WRITE FLOOR: a memory was written only when the key's norm exceeded 1e-6, a guard against the empty bag at
birth; the world's bag fades 0.8 a tick through the child's turn, so after 62 ticks of the child talking the first symbol of the
other voice's answer fell under the floor and was never written, on about half the served body's lines (the gate waits to its
8.2-second cap on half of them), never on a copy day (listen 24). The answer's onset is the memory the question must find; the
floor is now a constant (write_floor). THE COARSE CONTEXT: an order-free bag of characters makes "what is hot?" and "what do we see
at night?" share most of their weight, so the context term drew the siblings closer than the fast bag alone; the shifted form
(ctx_form) keeps the utterance's order. Three arms on a copy of day 209 with long turns (listen 64): the served form, the floor
lowered, the floor lowered with the shifted context.

Night 186 (ended ~22:10 after day 210): the probe after it: HELD-OUT 0.661 (0.708, 0.691, 0.679, 0.674, 0.661 over the five nights
under the slow context: the slide continues), the cortex on the fact sentences 0.877 (the highest), the mouth's prefixes 18 of 26
(recovering from 10), the questions 8 of 30 at two rests (5 the night before; 11 before the switch), the branch 8 of 9 and 9 of 12.
The starts are still the failures ("what is hot?" -> "two ducks and"). The three arms on the copy (the write floor, the shifted
context) report near 23:00; the held-out's slide is the number that decides a revert if neither arm mends the starts.
Day 210 (the eighteenth-after-reboot parent's facts 11-20 or 21-30 by its shifted schedule): 180 lines (A 95, all questions),
smiles 261, frowns 40; answered before B 19 of 95. (The guard's row for night 186 was not written: the guard armed before night 185
fired there and exited while the waiter saw it still alive; a guard is armed again for night 187.)

Night 187 (ended ~23:21 after day 211): day 211: 183 lines (A 91, all questions), smiles 253, frowns 39; answered before B 19 of
91. The parent for days 212-214 spawned at the night row (facts 1-10 first).
The parent's report for days 209-211 (23:23; its day 208 was cut by the restart): answered before B 43 of 76, 62 of 95, 63 of 91
(57-69%); the asked fact's word before B 0 on every day (fact-answer pairs in its turns 6, 9, 9: "did the ducks swim", "eat bread
an", "bees makes"); its own fact-shaped questions "what is little?", "what has wi[ngs]" answered as B's next line; own question
marks 19, 24, 33; smiles 241/261/252, frowns 32/40/39. A typist defect found: a line with a comma is never typed (twelve exchanges
broke; the brief now forbids commas). The three arms on the copy of day 209: A (the served form) and B (the floor lowered) read
identically, 10, 10, 7 of 30 and the branch 9 of 9, 8 of 12: at listen 64 the faded key's norm is 1.3e-6, just above the floor;
the crossing is near 68 ticks, and the served body's turns run to 80 (the gate's 8.2-second cap on half the lines), so the copy
cannot show what the served log implies. The floor's repair is arithmetic and goes on the served body at night 188's save; arm C
(the shifted context) decides the context's form.
The probe after night 187: HELD-OUT 0.676 (0.661 the night before: the slide turned), the cortex on the fact sentences 0.869, the
mouth's prefixes 18 of 26, the questions 9 of 30 at both pauses (5, 8, 9 over the last three nights; 11 before the switch), the
branch 7 of 9 and 9 of 12. The recovery under the new key runs as the facts are re-taught with their contexts.
The third arm (23:29): the shifted context reads the answers 10, 10, 9 of 30 (the order-free 10, 10, 7) but the branch falls to
6 of 9 and 5 of 12 (9 of 9 and 8 of 12): the order-coded context tells the siblings apart less well, not better. Rejected; the
context stays order-free. The write floor alone goes on the served body at night 188's save. A second pair of copy days at listen
80, where the served body's longest turns sit and the faded key crosses the old floor, runs overnight to put the floor's effect on
the record.

Night 188 (ended 00:24 after day 212): duty 0.354, smiles/line 1.28, the guard holds. Day 212 (the parent for 212-214, facts 1-10):
185 lines (A 93, all questions), smiles 240, frowns 40; answered before B 5 of 93 by the five-second window. Dusk 212: held-out
0.677, the mouth 14 of 26, the cortex on the facts 0.865. THE WRITE FLOOR REPAIRED on the served body at this night's save (the
reload at 00:24:42, write_floor 1e-30 with the slow context kept): from day 213 the first symbol of every world line is written
however long the child's turn before it ran.
The probe after night 188: HELD-OUT 0.685 (0.661, 0.676, 0.685: recovering), the cortex on the fact sentences 0.869, the mouth's
prefixes 14 of 26, the questions 8 and 8 of 30, the branch 7 of 9 and 7 of 12. The served body verified after the reload: the write
floor at 1e-30 with the slow context, the typist relaunched, the guard re-armed on the same flags.
The write floor at listen 80 on copies of day 209 (01:22): the old floor 15, 13, 11 of 30 (branch 7 of 9, 9 of 12); the floor at
1e-30 11, 12, 10 (7 of 9, 7 of 12). Not the lift the arithmetic promised, and not the same day twice: once the floor changes what
is written, the mouth's own speech changes and the whole day diverges, so a single pair of runs reads the chaos as much as the
floor (at listen 64, where nothing crosses the floor, the two arms were identical to the symbol). The floor stays on the served
body on the arithmetic; the served probes after nights 189 and 190 are its reading, and two falling nights would revert it.

Night 189 (ended ~01:20 after the day labelled 214; the reload's relaunch skipped the label 213): day 214, the first under the
repaired write floor: 181 lines (A 91, all questions), smiles 282 (a high), frowns 37; answered before B 15 of 91 ("i did." to "who
woke the cat up?"). Dusk: held-out 0.658, the mouth 16 of 26, the cortex on the facts 0.838.
The probe after night 189 (the first day under the repaired floor): HELD-OUT 0.659 (0.685 the night before; under the key the
readings run 0.66-0.69 against about 0.70 in the five nights before the switch), the cortex on the fact sentences 0.883 (the
highest), old lines 0.496 (the highest in a week), the mouth's prefixes 16 of 26, the questions 4 and 6 of 30, and for the first
time "what is hot?" answered whole, "the sun is hot"; the branch 7 of 9 and 10 of 12. The count of answers stays low and noisy
(4-9 a night under the key; 8-11 before) while the branch stands high (7-9 of 9; 1-2 before). Whether the key costs the held-out
is tested on copies overnight: day 214 lived twice from the same body, with the key and without, each followed by its night and
the rulers.

Night 190 (ended 02:27 after day 215): duty 0.364, smiles/line 1.56 (290 smiles over 182 lines, the highest count of the stage),
the guard holds; frowns 38; answered before B 14 of 91. The parent for days 216-218 spawned at the night row (facts 1-10 first, no
commas).
The parent's report for days 212, 214 and 215 (02:30): all thirty facts once each, verbatim, on schedule; answered before B 30 of
93, 37 of 91, 39 of 91 (32-43%); the fact's word in its own turn 3, 0, 1 of 10 a day (water, fly, rain; sun); answer smiles 4, 7,
5 (two on facts: "rain", "fish"); own question marks 33, 36, 33, among them fact-shaped ones of its own ("what do cows", "has
four"), answered in ordinary wording; runs of three known words 39, 36, 46; smiles 240/282/290, frowns 40/37/38, all talk-overs;
no exchange broke in 528 lines. The reload at night 188 wrote a phantom session (day 213: one line heard), so the log's day label
runs one ahead of the nights.
The probe after night 190: HELD-OUT 0.668 (0.659 the night before), the cortex on the fact sentences 0.888 (the highest), the mouth's
prefixes 20 of 26 (14, 16, 20 over the three nights under the floor), the questions 8 and 10 of 30 (the highest under the key;
"the sun did", "birds fly", "dogs run and b[ark]", "milk" for the cats), the branch 8 of 9 and 9 of 12. The recovery under the key
and the floor runs; the matched copy test of the key's cost reports near 03:30.
THE KEY'S COST, MATCHED (03:24): day 214 lived on two copies of the body after night 189, each followed by its night. With the slow
context and the floor: HELD-OUT after the night 0.657, the prefixes 15 of 26, the questions 7 and 8 of 30, the branch 5 of 9 and
10 of 12. With the fast bag alone and the floor: HELD-OUT 0.681, the prefixes 19 of 26, the questions 11 and 11, the branch 9 of 9
and 10 of 12. The context costs the held-out 0.024 and four answers, and the branch no longer needs it: with the answer's onset
written (the floor), the episode's links carry the question into its own answer. THE SLOW CONTEXT REVERTS at the next post-night
save; the floor stays. The memories written under the context remain and match on their fast part.

Night 191 (ended ~03:27 after the day labelled 216): THE SLOW CONTEXT REVERTED at this night's save (the reload at 03:27:45; the
fast bag with the write floor at 1e-30); the typist relaunched (the label skipped again: the current day is 218). Day 216 (the
parent for 216-218, facts 1-10): the tally below.
  day 216: lines=191 (A=95 B=96) A-questions=95 cues=0 smiles=235 frowns=43 withheld=2
  A-questions the child answered with I/yes/no/please before B: 15
     Q: what do you see out here? | its: 'yes          .'
The probe after night 191 (the last save under the slow context): HELD-OUT 0.668, the cortex on the fact sentences 0.885, the
mouth's prefixes 16 of 26, the questions 8 and 8 of 30 ("the sun is hot" whole, "fish live in w[ater]", "birds fly up", "dogs
run"), the branch 7 of 9 and 9 of 12. From here the readings are the fast bag's with the floor.

Night 192 (ended 04:30 after the day labelled 218, the first full day after the revert): duty 0.304, smiles/line 1.14 (233 smiles
over 198 lines, the lowest rate of the week), the guard holds; frowns 37; answered before B 17 of 99.
The probe after night 192 (the first night after the revert, the fast bag with the floor): HELD-OUT 0.689 (0.668 the night before),
the cortex on the fact sentences 0.881, the mouth's prefixes 17 of 26, the questions 10 of 30 at two rests and 5 at eight, THE
BRANCH 8 of 9 on the sun family and 12 of 12 on the day-206 families: the siblings told apart on the fast bag alone once the
answer's onset is written.
THE REWARD REPLAYED, first use (05:30): day 216 on a copy with its 277 smiles and frowns set on the face at their delays, and once
without. The actor's slope 0.036 (corr 0.024) with the reward against 0.039 (0.026) without: a day of reward moves the actor not at
all at actor_lr 0.02, as the arithmetic said (one to five answer smiles a day against thirty maps). The rulers 11 and 8 of 30
against 14 and 6, the branch 8 of 9 and 9 of 12 against 7 of 9 and 12 of 12: the reward's presence changes the day's course within
a day's noise. The instrument works (the faces land, the mood differs); the actor's rate at rewarded moments is now measurable on
copies: 0.2 and 1.0 next, against 0.02.

Night 193 (ended 05:33 after the day labelled 219): duty 0.316, smiles/line 1.44 (280 smiles over 190 lines), the guard holds;
frowns 39; answered before B 15 of 95. The parent for days 220-222 spawned at the night row (facts 1-10 first).
The parent's report for days 216, 218 and 219 (05:36): all thirty facts once each, front-loaded; answered before B 42 of 95, 57 of
99, 44 of 95 (44-58%); the fact's word in its own turn before B 2, 2, 1 a day (birds, rain; apple, ants; fins); clean content
answers ahead of B ("on my foot" to "where is my sock?", "out" to "where is the dog?", "my sock" to "what is in the tub?"); fact
fragments bound to the wrong question ("where do birds live?" -> "cows give"); its own fact-shaped question "what is soft?" answered
in the next rows; own question marks 21, 35, 39; smiles 235/233/278 (answer smiles 4, 5, 6), frowns 43/37/39; one A line lost at a
night boundary in 516; no B line deferred. The day label is not a day counter (the reload's relaunch skips one).
The probe after night 193 (the second night after the revert, the fast bag with the floor): HELD-OUT 0.680, the cortex on the fact
sentences 0.890 (the highest), old lines 0.501 (the highest in ten days), the mouth's prefixes 20 of 26, THE QUESTIONS 16 OF 30 AT
TWO RESTS AND 15 AT EIGHT, the highest readings of the stage (8-11 before the switch, 4-10 under the slow context): "ice", "fish
live in w[ater]", "birds fly up"; the branch 7 of 9 and 12 of 12. The floor's repair shows on the live body: the answer's onset
written, the question finds its own answer half the time on a clean ask.

Night 194 (ended 06:36 after the day labelled 220): duty 0.349, smiles/line 1.33 (257 smiles over 187 lines), the guard holds;
frowns 37; answered before B 23 of 93 ("i did. she" to "what did we hear?"). THE ACTOR'S RATE on a copy of day 216 with its
rewards: at actor_lr 0.2 the actor's slope 0.038 (corr 0.025), the same as at 0.02 (0.036) and as without reward (0.039); the
rulers 16 and 13 of 30, the branch 7 of 9 and 11 of 12. Ten times the rate moves the earned voice not at all in a day: the voice is
gated by a correlation estimated over two hours of ticks (actor_tau), and a day cannot move it. The chooser's reading must be the
actor's raw preference at the branch, before the gate; the 1.0 arm runs to close the sweep.
THE ACTOR AT THE BRANCH (06:39, tools/branch_probe.py with the striatal delay line fed along the prefix and the actor's raw vote
read): on the served body the actor votes 'b' at +1.00 or 'd' at +1.00 whatever the question and the start ("the sun " -> 'b',
"the sun is " -> 'd'), 0 of 9 right: the head is saturated (tanh at its rails on a few symbols) and reads nothing of the question.
Its earned voice being small (slope 0.036), it does no harm to the mouth; as a chooser it is absent. The chooser must be rebuilt:
bounded (a softmax over the candidates, not a tanh per symbol), fed by a state that carries the question, credited by the answer
smile on the candidate chosen, and read by this instrument on copies with the day's rewards replayed.
THE THIRTIETH DEFECT, named (06:40): the served actor has been off (no --actor on the served flags), and the head that remains is
saturated; the striatal delay line it read held only the last eight events, so no question could ever have reached it. THE CHOOSER
(actor_form softmax, actor_input cortex; 06:43): a striatal head over the CANDIDATES at a torn moment (the mouth's top few and the
cortex's own top two, the best two within the margin), read from the cortex's state with its running mean taken out (the
corticostriatal path), a softmax among the candidates whose zero-mean log enters the readout at the earned gain; credited by
dopamine on the candidate said (the log-softmax's gradient, decaying by dopamine's discount); every row of its weights bounded so
no candidate can saturate the vote. Born at zero. Test 57 (a rewarded choice raises its candidate on that state, a punished one
lowers it, forty pushes cannot pass the bound); suite 56 of 56. Measured next on a copy of day 216 with its rewards replayed, the
chooser's raw vote at the branch read after, at rates 0.2 and 1.0.
The old actor's rate, the sweep closed (07:40): at actor_lr 1.0 with the day's rewards the slope 0.042 (corr 0.028), the raw vote at
the branch 0 of 9 and 0 of 12, the rulers 15 and 10 of 30, the branch 7 of 9 and 10 of 12: fifty times the rate moves nothing,
as the arithmetic said of a head that never sees the question. The chooser's copy days begin now.

Night 195 (ended 07:40 after the day labelled 221): duty 0.370 (a high), smiles/line 1.50 (279 smiles over 182 lines), the guard
holds; frowns 34; answered before B 25 of 91 ("yes. water" to "can that duck swim?").
The probe after night 195: HELD-OUT 0.670, the cortex on the fact sentences 0.892 (the highest), the mouth's prefixes 21 of 26, THE
QUESTIONS 17 OF 30 at two rests (16, 16, 17 over the last three nights) and 12 at eight, the branch 8 of 9 and 12 of 12.

Night 196 (ended 08:43 after the day labelled 222): duty 0.299, smiles/line 1.08 (the lowest rate of the week), the guard holds.
The parent for days 223-225 spawned at the night row (facts 1-10 first).
  day 222: lines=187 (A=93 B=94) A-questions=93 cues=0 smiles=209 frowns=40 withheld=4
  A-questions the child answered with I/yes/no/please before B: 23
The parent's report for days 220-222 (08:46): all thirty facts once each, front-loaded; answered before B 36 of 93, 46 of 91, 37
of 93 (39-51%); the fact's word before B 2, 1, 0 a day (fish, birds; grass); **answer smiles 11, 11, 10 a day** (4-7 the days before:
the child names what B is about to say twice as often); own question marks 28, 30, 23; runs of three known words 17, 20, 12;
smiles 257/279/209, frowns 37/34/40, all talk-overs; one line lost at a night boundary in 557; known words 381 -> 399. The
queue's novelty is the binding constraint for the parents: 10,262 distinct lines already, a fifth to a third of fresh lines
colliding.
The probe after night 196: HELD-OUT 0.681, the cortex on the fact sentences 0.885, the mouth's prefixes 21 of 26, THE QUESTIONS 18
OF 30 at two rests (16, 16, 17, 18 over four nights on the fast bag with the floor) and 12 at eight, the branch 8 of 9 and 12 of 12.
The chooser's first copy days (09:40; day 216 with its rewards, actor_lr 0.2 and 1.0): the mouth 15/13 and 17/9 of 30, the branch
8 of 9 and 12 of 12 (the chooser's voice being small), and the chooser's raw vote 0 of 9 and 0 of 12: at 0.2 its favourite is '?'
everywhere, the symbol most often followed by a smile (the child's own questions earn them), at 1.0 'I' and 's'. Derived: the head
had a bias, and a bias absorbs exactly the state-free part of the credit, the symbols rewarded on average; it then out-votes any
state-dependent preference. The bias removed: the chooser votes by the state alone, and the branch probe now reads its vote among
the family's candidates only, as it votes in the body. One more copy day at 0.2; then, its gain being earned and born at zero, the
chooser goes on the served body to learn over days from real rewards, where a copy day's ten answer smiles cannot teach it.

Night 197 (ended 09:44 after the day labelled 223): duty 0.346, smiles/line 1.23, the guard holds. Day 223 (the parent for
223-225, facts 1-10): the tally below.
  day 223: lines=183 (A=92 B=91) A-questions=92 cues=0 smiles=232 frowns=39 withheld=8
  A-questions the child answered with I/yes/no/please before B: 22
The probe after night 197: HELD-OUT 0.672, the cortex on the fact sentences 0.890, the mouth's prefixes 18 of 26, the questions 16
of 30 at two rests (16, 16, 17, 18, 16 over five nights) and 11 at eight, the branch 7 of 9 and 12 of 12. A plateau near sixteen to
eighteen on the clean ask, from the store alone.
The chooser without a bias, one copy day with the rewards (10:43, actor_lr 0.2): its vote among the family's candidates 4 of 9 on
the sun family and 8 of 12 on the day-206 families (the one-candidate starts counted), unsaturated (votes within one): a favourite
per family ('m' after "the sun ", 'd' after "we " and "a ", 'a' after "an ") rather than a choice by the question, which is what
ten relevant smiles a day can teach a linear head in a day. The mouth's rulers 16 and 12 of 30, the branch 7 of 9 and 10 of 12,
its earned voice small (0.063). Born at zero and gated by what it earns, the chooser goes on the served body at night 198's save
to learn over days from the real rewards; its reading from here is its vote at the branch after each night.

Night 198 (ended ~10:45 after the day labelled 224): day 224: 183 lines (A 91, all questions), smiles 268, frowns 36; answered
before B 21 of 91. THE CHOOSER ON THE SERVED BODY: the reload at 10:45:49 after this night's save (actor 1, actor_form softmax,
actor_input cortex, actor_voice earned, actor_lr 0.2, born at zero); from day 225 the striatal head votes among the candidates
the mouth is torn between, as loud as it earns.
The probe after night 198 (the last save before the chooser): HELD-OUT 0.654 (0.689, 0.680, 0.670, 0.681, 0.672, 0.654 over the six
nights since the revert: a drift down to watch; the held-out set is 55 lines now, two more of its lines having been typed), the
cortex on the fact sentences 0.885, the mouth's prefixes 21 of 26, the questions 16 of 30 at two rests and 9 at eight, the branch
7 of 9 and 12 of 12.
The served body verified after the reload (10:48): the chooser's flags on, the typist relaunched, the guard re-armed, 92 ticks in
twenty seconds, the earned gain 0.049 (the saved estimate; the new head's weights are zero, so its first votes are even).
The old in the draw, two copy nights from the save after night 198 (11:20): with a quarter of the dreams drawn uniformly over the
memory, HELD-OUT 0.654 -> 0.641 and the old lines 0.476 -> 0.471; without, 0.654 -> 0.654 and 0.473. Rejected: the utterance
memory holds about twenty days, all of stage five, so its "old" is the same style as its new; the fourth stage's speech the
held-out lines are written in is no longer in it. The held-out's drift is the day's erosion of what earlier nights consolidated
(each dusk two to three hundredths down, each night one to two up), which the eighth iteration, the day's plasticity gated by
reward and surprise, addresses next.

THE CHOOSER'S DAY, AND THE GUARD (11:48): the first day under the chooser (labelled 226; 10:47-11:35) collapsed: 196 lines, 48
smiles (0.24 a line; 1.3-1.5 the days before), 3 frowns, duty 0.158, answered before B 16 of 98, and the child's turns filled
with stray symbols ('k62W\Yo', 'Fa 9', 'Mth': 99 characters that are not letters, against 0-1 on the days before): the chooser's
vote let symbols the mouth never says into its speech, or the tick under it broke; the copy days had not shown it because the
probes read the mouth's answers, never the day's own symbols. The guard read the day and tripped, as it should, and its restart
FAILED: the flags it holds had lost their quoting when I regenerated its arguments, so "--period" arrived without its value; the
served body was down from 11:48 to 11:50. Restored by hand at 11:50 on the pre-chooser flags (the fast bag, the floor; no tag
closure: the trip was the chooser's, not the tag's); the guard re-armed with its flags quoted; the typist chain relaunched (the
label skipped to 228); the parent for days 228-230 spawned. THE CHOOSER IS OFF; before it returns, a copy day must be read for
the child's own symbols, not only the rulers.
THE CHOOSER'S FAULT, read on a copy (12:06): thirty lines of day 224 with the rewards replayed, the chooser on against off: the
child's own symbols 301 against 474, junk characters 4 against 1, and its speech looped on the rewarded fragment ("did. ... did.
... did yo"). The head learns the day's action prior, the symbols rewarded on average, through the state's mean direction even
without a bias, and votes it back into a mouth that already carries that prior from the cortex and the store: fewer, looping,
stranger symbols. With ten relevant smiles a day the state-dependent residual it was built for cannot outweigh that prior. The
chooser is off and stays off in this world; it is written down as failed for the reward's density, not the mechanism's kind: a
body with denser, more specific reward is where a striatal chooser earns its place.
THE REPHRASED-QUESTION RULER (12:10; tools/heldout_rephrased.txt, thirty facts asked in words the parents never type): the served
body after night 199 answers 8 of 30 at two rests and 7 at four ("what is so green?" -> "grass is green", "what comes down from
the sky?" -> "rain falls", "what can ducks do?" -> "ducks swim", "where do the birds live?" -> "birds live in a", "what do the fish
have?" -> "fish have fins"), against 16 to 18 on the taught wording: the recall carries to a rephrasing when the question's last
words are the taught ones, which is what a key of the last five symbols predicts. The honest demo claim: taught questions about
half, rephrasings about a quarter.

Night 200 (ended 13:01 after the day labelled 228, the first full day back on the pre-chooser flags): duty 0.325, smiles/line
1.28 (253 smiles over 189 lines), the guard holds with its flags quoted; frowns 40; answered before B 25 of 94. The day recovered
whole from the chooser's day.
THE PREFIXES HALVED IN A DAY (13:10): the mouth completed 20 of 26 held-out prefixes on the save of 11:48 (after night 199), 10 at
the dusk of the day labelled 228 (12:36) and 9 after night 200; the questions 16 -> 7 of 30; the flags verified the same as
before the chooser, the store full (8192 slots, every strength above 0.54, the mean 0.82 -> 0.91 through the day). A day writes
about 3500 world symbols into 8192 slots: two fifths of the store turns over daily, and the facts come once in three days. The
store's capacity (the second iteration) is derived and built next; the tick's cost is measured before it goes on the served body.
THE PLASTICITY GATE READ (13:16): day 224 lived twice on the save after night 198, the lesson gated by the smile and the surprise
against ungated. The held-out fell two hundredths in both (0.654 to 0.638 gated, 0.644 ungated); the questions 17 of 30 in both;
the prefixes 22 gated against 18. The prediction (the dusk's fall halved) failed; the gate stays off. What the day costs the
held-out is not the ordinary tick's lesson.
THE EVICTION PROVEN (13:20): day 228 wrote 1240 new slots into the full store and evicted 1238, all at strength 0.54-0.59, the
memories heard once. Fourteen of the twenty prefixes the morning answered read a slot the dusk no longer held; the dusk body with
the union of both stores answers 19 of 26 against its own 10, and 14 of 30 questions against its own 9. The capacity is a constant of the organ now (store_cap); 32768 goes
on the served body at night 201's save, and the next dusk is the falsifier: the prefixes hold their morning count through day 230.
NIGHT 201 (ended 14:11, slept 1242 s; the 191st row; the log's day 229): held-out 0.659 (from 0.654), the old lines 0.466, the facts
by the cortex 0.888; the mouth completes 11 of 26 prefixes and answers 13 of 30 questions at two rests (8 at eight; the rephrased
8): day 229, the last at the old cap, retaught some facts (7 -> 13) and evicted others (the prefixes 9 -> 11 only). The reload at
the save (14:11:35) put the capacity on the served body: the store read 8282 slots within two minutes, past the old limit, and
the typist relaunched. Day 230 is the falsifier: the dusk probe against this morning's 11 of 26.
THE FIRST DAY AT THE CAPACITY (the log's day 231, 14:12-15:01; night 202 ended 15:22, slept 1303 s): the prefixes 11 of 26 in the
morning, 22 at dusk; the held-out 0.659 in the morning, 0.667 at dusk, the first dusk above its morning since the stage began; the
store 8192 -> 10318 slots, nothing evicted; the guard held (duty 0.320, smiles a line 1.47). The eviction was the cause, and the
capacity the cure. (The label 230 was an aborted session of no lines; the typist's relaunch skips a label.)
THE PARENT OF 228, 229 AND 231 REPORTED (15:24): 189/188/186 lines; A's questions answered before B 56/49/53 of 94; answer smiles
11/7/6; fact words in its own turn before B: fish, rain / bees make it / cows, birds; unprompted runs of three words 18/15/12; its own
question marks 31/29/28; smiles 253/274/284, frowns 40/39/41 (all talk-overs); known words 419 -> 436. It answered the child's own
fact-shaped questions ("what is green", "is the star?", "is soft?") with the fact as B's next line. The queue is saturating: most
first-draft lines collide with the 3700 rows; the next parent (232-234) is briefed to vary and check.
NIGHT 202 (the first at the capacity; ended 15:22): held-out 0.665, the old lines 0.481, the facts by the cortex 0.888; the mouth
completes 22 of 26 prefixes (held through the night) and answers 18 of 30 questions at two rests and 18 at eight (the stage's best
at the long pause, 8-14 before); the rephrased 11 of 30 (8 before); the branch 9 of 9 and 11 of 12; the dreams 0.83.
THE REWARD TAG READ (15:40): day 224 lived twice on the save after night 198 with the rewards replayed, then a night on each copy.
Tagged: the held-out 0.661 at dusk, 0.669 after the night, the facts by the cortex 0.888. Untagged: 0.646, 0.646, 0.881. The
direction predicted, but the dusks differ by 0.015 before the night touches anything, the two copy days having diverged on the
store's reads: a copy day's noise on the held-out is about 0.015, larger than most effects measured this stage. Inconclusive; a
matched pair of nights from the tagged day's save (the control with its tags zeroed) is queued behind the told-once test.
THE FACT TOLD ONCE (16:12; tools/once_told.py on a copy of the save after night 202): ten facts the parents never typed, told once
each among forty lines of day 231's talk as the typist types them; asked with the pause the same day, the mouth answered 10 of 10
at two rests and 9 at eight, each the sentence verbatim; after a night on the copy, 10 and 8; the cortex alone 0 throughout; the
taught questions on the same copy 17 of 30. The demo's central scene stands on a copy. The never-typed facts answer better than
the taught ones because their question's key is uncrowded: the taught questions were asked in ordinary talk with other answers,
and the recall splits among the continuations under one key.
THE DUSK OF DAY 232 (16:12; the second day at the capacity, the new parent's first): the mouth completes 25 of 26 prefixes (22 in
the morning); but the held-out 0.665 -> 0.595 and the facts by the cortex 0.888 -> 0.796, a fall ten times an ordinary day's, the
old lines steady (0.481 -> 0.474). The day itself was ordinary (183 lines, 861 words, junk 2, smiles 281, duty 0.333, stress 4.5).
The cortex's weights moved less over night 202 and day 232 together (relative 0.0074) than over day 231 alone (0.0104), so it is
not runaway plasticity but a movement in a bad direction for the recent material; the cortex's stream never sees the recall (the
read enters the mouth's forecast only), so it is not the cortex leaning on the store. The night's probe decides whether the day's
fall is a transient. (An instrument's lesson: the dusk probe's save overwrites the served file by day, so a "morning" copy taken
later is the dusk; the post-save probe now keeps the morning save.)
NIGHT 203 (ended 16:29, the 193rd row): the held-out 0.655 (0.595 at dusk, 0.665 the morning before: the day's fall was a
transient the night undid, the cycle's net a hundredth down as before), the facts by the cortex 0.881, the old lines 0.470; the
mouth completes 25 of 26 prefixes and answers 20 of 30 questions at two rests (the stage's highest; 14 at eight); the rephrased
12 of 30; the branch 9 of 9 and 11 of 12; the store 12449 after the fade; the guard held (duty 0.333, smiles a line 1.55).
THE LIVE MOUTH READ (16:56; tools/live_qa.py, the question typed as the parent, the child's turn of twelve seconds read from the
page): the taught questions answered 5, 1 and 3 of 30 over three passes on the dusk copy of 232, the never-typed ones 4, 0 and 1
of 10 on the told-once copy, against 20 of 30 and 10 of 10 by the greedy readout. The answers come but late and broken ("one
little fishice is co", "lemon is sourlwet"): each word's first symbol is sampled from the readout at the fixed sharpness of 25
and the word then runs greedily, so a three-word answer needs three lucky starts where the margin is thin; and the instrument
typed over the child's speech, which the typist never does. The memory is not the demo's gate; the choice is. Built: the
instrument waits for the child's quiet; decisiveness by certainty (sharp_conf, off by default; test 61): the choice's sharpness
times (1 + sharp_conf x the forecast's norm), the selection's noise falling as its evidence rises, the same certainty the gate's
salience reads. Measured next: the bound (sharp_base 60 and 100 on the live ruler), then sharp_conf 3 and 8.
NIGHT 204 (ended 17:46, the 194th row): the held-out 0.657 (the dusk 0.641, an ordinary day's fall), the facts by the cortex
0.883, the old lines 0.469; the mouth completes 26 of 26 prefixes and answers 21 of 30 questions at two rests (13 at eight); the
rephrased 12; the branch 9 of 9 and 12 of 12; the store 14847; the guard held (duty 0.345, smiles a line 1.34). Day 233: 198
lines, smiles 271, frowns 41.
THE BOUND ON THE LIVE MOUTH (17:53): with the instrument waiting for the child's quiet, sharp_base 25 answers 4 and 6 of 30,
sharp_base 60 answers 2 and 1: sharper is worse, the sampling is not the cause. The instrument gave no smiles and the copy's
mood sank to the readout's floor through the run; it now smiles at an answer as the caregiver does. The hold of working memory
(bag_rest_decay 0.97) is measured next, greedy by the pause and live with smiles, against the control with smiles.
THE PARENT OF 232, 233 AND 234 REPORTED (18:58): 183/198/188 lines; A's questions answered before B 51 of 91, 57 of 99, 40 of 94;
anticipation smiles 11/7/9, of which whole fact answers on day 233: "grass is green", "a dog has four legs", "bees make honey";
fact words in its own turn before B 2/3/0; its own question marks 29/39/26; smiles 288/271/274, frowns 38/41/40 (a third of the
talk-overs); known words 436 -> 457. It answered the child's own fact-shaped questions ("hat has wings?" -> a bird has wings) as
B's next line. The typist typed all 569 lines in order, none deferred; nights 1071, 1348 and 1377 s. The next parent (235-237)
was spawned at the 195th row.
NIGHT 205 (ended 18:58, the 195th row): the held-out 0.660 (the dusk 0.639), the facts by the cortex 0.888, the old lines 0.469;
the mouth completes 24 of 26 prefixes and answers 24 of 30 questions at two rests and 19 at eight (both the stage's highest: the
facts accumulate now that nothing is evicted, 13 -> 18 -> 20 -> 21 -> 24 over the five nights at the capacity); the rephrased 11;
the branch 9 of 9 and 12 of 12; the store 16886; the guard held (duty 0.329, smiles a line 1.42). Day 234: 188 lines, smiles 274.
THE HOLD OF WORKING MEMORY, MEASURED MOOD-FAIR (19:49): the instrument smiling as the caregiver does (a known word said, the answer's
growing smile; 90-121 smiles a run), on the dusk copy of 232: the control answers 3 and 2 of 30 live, the hold (bag_rest_decay
0.97) 12 and 12 of 30, 18 ever, the predicted count on both passes, at the copy's floor mood (-6) where the readout is flattest;
the never-typed facts 6 and 1 of 10. The trace with the hold: the cue's norm above 1.1 for eleven ticks after the question, 't' on
top, the gate opening at 2.2 s. It goes on the served body at night 206's save; the pre-hold flags kept. The decisiveness constant
(sharp_conf) stays off: sharper was worse, and certainty-scaled added nothing over the hold.
NIGHT 206 (ended 20:00, the 196th row): the held-out 0.651 (the dusk 0.636), the facts by the cortex 0.843 (0.888 the night before;
the dusk 0.822: a fall to watch), the old lines 0.463; the mouth completes 24 of 26 prefixes and answers 21 of 30 questions at two
rests (14 at eight, read without the hold: the save's own constants); the rephrased 13; the branch 9 of 9 and 12 of 12; the store
18966; the guard held. Day 235: 184 lines, smiles 296, frowns 38. THE HOLD WENT ON THE SERVED BODY at this save (the reload at
20:00:15, served again 20:01:10; bag_rest_decay 0.97 beside the capacity; the guard re-armed with the same flags; the typist
relaunched, the log's day 237 (236 an aborted session)).
THE FIRST HOURS AT THE HOLD (20:31, day 237 half done): 111 lines, smiles 1.25 a line (1.61 the day before), answer smiles 3
(9 in the whole of day 235), no fact word before B yet; frowns 0.23 a line (0.21); duty 0.283 (0.337); junk 6. After the fact
questions its turn is mostly empty for the five seconds the page shows, or a fragment ("what is little?" -> "as an ... ant"):
the gate's latency runs past the page's window on the served body more often than on the copies. Too early to read; the day's
tally at night 207 and the next days' parent counts are the ruler. The live instrument on the served body's morning save with
the hold reads 10 and 9 of 30, and undercounts: "what is cold?" -> "ce is coldd", the 'i' said in the question's last tick
before the window; fixed. The instrument's mood sinks to the floor under back-to-back questions (the value's expectation of the
answer smile unmet two times in three); it now interleaves the day's ordinary exchanges between questions (--interleave 1).
THE LIVE MOUTH ON THE SERVED SAVE (20:34; the save after night 206): with the hold 10 and 9 of 30, without it 1 of 30, the
instrument still undercounting a symbol said in the question's last tick.
NIGHT 207 (ended 21:00, the 197th row; the first night after a day at the hold): the day 237's dusk fell hard on the cortex (the
held-out 0.610, the facts 0.732) and the night restored it (0.657, 0.881), as on day 232; the old lines 0.482; the mouth
completes 24 of 26 prefixes; the questions 16 of 30 at two rests (21 the night before) and 17 at eight (14); the rephrased 8
(13); the branch 9 of 9 and 10 of 12; the store 21209. The parent's counts for the day at the hold: questions answered before B
36 of 91 (25), answer smiles 10 (9), duty 0.294 (0.337), frowns 40 (38). The two-rest fall is derived to be a flaw in the hold's
form: the world's context faded at the quiet rate while the child answered, though the query shifts it a lag per own symbol and
the keys were written at the symbol rate; the fade must follow the symbols whoever says them (bag_own_fade, a disclosed switch).
Measured next on this night's save, greedy by the pause, both forms.
THE HOLD'S FORM MEASURED (21:48; the save after night 207, greedy by the pause): as served (own symbols at the quiet rate) 16, 17,
18 of 30 at 2, 8, 16 rests; with own symbols at the symbol rate 19, 20, 18; without the hold 19 and 14; the branch 9 of 9 in
both. The fix restores the two-rest count to the save's ceiling and keeps the pause's gain; it goes on the served body at night
208's save (bag_own_fade 1 beside bag_rest_decay 0.97), the reload armed, the guard to be re-armed with the same flags.
NIGHT 208 (ended 21:58, the 198th row): the held-out 0.649 (the dusk 0.647: a day at the hold with no fall), the facts by the
cortex 0.885, the old lines 0.469; the mouth completes 25 of 26 prefixes; the questions 15 of 30 at two rests and 16 at eight,
read with the save's own form (own symbols at the quiet rate); the rephrased 8; the branch 9 of 9 and 10 of 12; the store 23310
(+2200 a day under the hold: the answer chains now carry the question in their keys and merge less). THE FORM FIX WENT ON THE
SERVED BODY at this save (the reload at 21:58:40, served again 21:59:35: bag_own_fade 1 beside the hold and the capacity; the
guard re-armed with the same flags; the typist relaunched). The slide of the two-rest count since night 205 (24, 21, 16, 15) is
read as the transition: the keys written under the hold carry the question, the old chains match the new query less, and the
facts retaught over the next days rebuild under the new form; the day's own answering (the parent's counts) rose at once.
THE PARENT OF 235, 237 AND 238 REPORTED (22:00): 184/182/183 lines; A's questions answered before B 47%, 55%, 53% (the last two
the days at the hold); answer smiles 9, 10, 14; the fact's word in its own turn before B 1, 2, 1; its own question marks 26, 30,
44; whole phrases of three words or more 15, 8, 8 (shorter turns, more often a single answer or a question); smiles 297/248/279,
frowns 38/40/37, all talk-overs. No deferral in 549 lines; 306 waited for the child's quiet (mean 4.1 s, 64 at the 8.2 s ceiling).
It said "es make honey" before fact 19 was asked that day. The next parent (239-241) spawned at the 198th row.
NIGHT 209 (ended 22:58, the 199th row; the first day at the corrected form, the log's day 240): answer smiles 18 (9, 10, 14 the
days before), duty 0.322, junk 2, its own question marks 14 (44 the day before), frowns 36; the held-out 0.640 (the dusk 0.638:
the night restored little; 0.657, 0.649, 0.640 over three nights, to watch), the facts by the cortex 0.883, the old lines 0.458;
the mouth completes 24 of 26 prefixes and answers 17 of 30 questions at two rests and 17 at eight (15 and 16 the night before,
under the served form); the rephrased 7; the branch 9 of 9 and 11 of 12; the store 25293 (+1983).
THE FORM'S SECOND FLAW (23:09): on the save after night 209 the ten facts retaught that day answer 3 of 10, the other twenty 7
of 10 each. The log shows why: on nine of those ten questions the child spoke before the other voice ("ice", "fish live",
"birds", "rain falls"), and form 1 faded the world's context by its own symbols, so the answer's onset was written under a
context faded to nothing, the retellings of one fact in two key forms. The keys must be the world's alone; the symbol-rate
fade for own symbols belongs in the query beside the efference copy's shift (form 2; test 64). Form 2 reads exactly as form 1
on that save (17 and 17); it goes on the served body at night 210's save, the flags and the guard updated.
NIGHT 210 (ended 23:53, the 200th row; the log's day 241, facts 11-20 under form 1): the held-out 0.644 (the dusk 0.629; the dusks
0.647, 0.638, 0.629 over three days, the nights restoring to 0.649, 0.640, 0.644: to watch), the facts by the cortex 0.888, the
old lines 0.468; the mouth completes 23 of 26 prefixes and answers 18 of 30 questions at two rests and 19 at eight (15/16 and
17/17 the two nights before: the retold facts rebuild under consistent keys); the rephrased 9; the branch 9 of 9 and 11 of 12;
the store 27250 (+1957); answer smiles 9 (18 the day before), junk 8, its own question marks 24. FORM 2 WENT ON THE SERVED BODY
at this save (the reload at 23:53:55, served again 23:54:50; the guard re-armed with the same flags). The store reaches the
capacity in under three days at this rate; 65536 (the fade's horizon of nineteen nights at two thousand a day) goes on at
night 211's save.
NIGHT 211 (ended 00:48, the 201st row; the log's day 243, facts 21-30 under form 2): the dusk's held-out 0.619 (0.647, 0.638,
0.629, 0.619 over four dusks: a drift to derive if the night does not restore it), the facts by the cortex at dusk 0.845; the day:
answer smiles 14, junk 6, its own question marks 27, frowns 35, duty 0.346. THE CAPACITY WENT TO 65536 at this save (the reload
at 00:48:54, served again 00:49:49 with the hold and form 2; the guard re-armed with the same flags; the store 29346 at the load).
THE PARENT OF 240, 241 AND 243 REPORTED (00:50): 177/180/178 lines; A's questions with a word of the child's before B 67%, 70%,
75%; fact answers in its own turn 4 (facts 2, 4, 5, 10 on day 240), 1, 1; answer smiles 18, 9, 14; turns of three known words or
more 29, 28, 33; its own question marks 14, 24, 27; talk-overs 107, 105, 87; smiles 267/283/281, frowns 36/38/35. Whole
utterances of its own: "I like my bed", "shall we go to bed", "what is hard?", "bread and honey", "snow gone now?". One line
lost at the aborted 242 relaunch; no deferral. The next parent (244-246) spawned at the 201st row.
NIGHT 211's PROBE (00:52): the held-out 0.635 (the fifth night down: 0.657, 0.649, 0.640, 0.644, 0.635), the facts by the cortex
0.877, the old lines 0.472; the mouth completes 24 of 26 prefixes and answers 18 of 30 questions at two rests and 20 at eight
(the best at eight); the rephrased 7. The drift is real; the day's lesson at a third is re-tested on this night's save (day
243 lived twice with the rewards replayed), the dusk's held-out the ruler.
NIGHT 212 (ended 02:01, the 202nd row; the log's day 245, facts 1-10 retold under form 2): the held-out 0.633 (the dusk 0.626:
a fall of 0.009 in the day, the smallest for a week; flat against 0.635), the facts by the cortex 0.879, the old lines 0.482; the
mouth completes 24 of 26 prefixes and answers 20 of 30 questions at two rests and 22 at eight (the best at eight), "what is
hot?" -> "the sun is hot" answered again; the rephrased 10; the branch 9 of 9 and 10 of 12; the store 31542 (+2200) under the
capacity of 65536; the day: answer smiles 15, junk 2, its own question marks 25, frowns 36, duty 0.326.
THE DAY'S LESSON AT A THIRD, RE-TESTED (02:46): day 243 relived on the save after night 211 with the rewards replayed: ungated
the held-out rose 0.635 -> 0.645, gated 0.650; neither fell, while the served body had fallen to 0.619 on that day. The copy
does not reproduce the served day: its mood sinks to -5 under replayed rewards not contingent on what its child says, its
readout flattens to the floor, and it never speaks the fluent turns suspected of pulling the cortex. Inconclusive; the gate
stays off; the drift is watched (flat at night 212). A faithful copy day needs a caregiver in the loop.
NIGHT 213 (ended 03:02, the 203rd row; the log's day 246, facts 11-20 under form 2): the held-out 0.649 (the dusk 0.626; up from
0.633: the drift reversed), the facts by the cortex 0.881, the old lines 0.465; the mouth completes 24 of 26 prefixes and answers
21 of 30 questions at two rests and 24 at eight (the best at eight, equal to the stage's best at two); the rephrased 11; the
branch 9 of 9 and 10 of 12; the store 33765, past the old capacity; the day: answer smiles 11, junk 5, its own question marks 32.
THE DEMO REHEARSED ON THE SERVED BODY (03:08-03:24, the log's day 247, the typist frozen): fifteen taught questions typed over the
page with an ordinary exchange between each, the smiles contingent: 8 of 15 answered in the child's turn, the answers 0.6 to
7.4 s after the question ("the sun", "water is wet", "cats drink milk", "rain falls from the sky", "grass", "an apple is red",
"an ant is little", "we eat bread and"); three never-typed facts told once at the start ("a lemon is sour", "a sheep has wool",
"the sea is salty") answered 3 of 3 minutes later ("a lemon i", "a sheep has woolwhat is big?", "the sea"). The mood sank from
-2.5 to -6 over the session (the rehearsal's parent smiled less than the typist); the same three facts are asked again after
night 214, and facts 16-30.
THE PARENT OF 245, 246 AND 247 REPORTED (03:56): 184/184/107 lines (the third day cut by the rehearsal's freeze of the typist,
03:08-03:25; facts 27-30 did not land); A's questions answered before B 55%, 66%, 65%; fact answers in its own turn 1, 5, 3
("apple red" whole and ahead of B, "grass", "bees make", "honey sweet", "the sun", "sun makes"); whole phrases 7, 6, 6 ("the
water is warm", "one egg and"); its own question marks 23, 30, 14; smiles 282/271/183, frowns 36/37/22, all talk-overs. A
mechanical finding: the typist silently drops a line over 38 characters (the exchange breaks); the brief now carries the cap.
The next parent (248-250, facts 21-30 first) spawned at the 204th row.
NIGHT 214 (ended 03:57, the 204th row; the log's day 247, cut short by the rehearsal): the held-out 0.654 (the dusk 0.608 after
the rehearsal's questions; the night restored it fully: 0.635, 0.633, 0.649, 0.654), the facts by the cortex 0.883, the old
lines 0.466; the mouth completes 24 of 26 prefixes and answers 19 of 30 questions at two rests and 23 at eight; the rephrased
10; the branch 9 of 9 and 10 of 12; the store 35431. The second rehearsal began on the served body at the day's start: the
three never-typed facts told before the night, asked after it, then the taught questions 16-30.
THE SECOND REHEARSAL (04:02-04:15, the start of the log's day 248, the typist frozen, the rehearsal's parent attending through
its waits: the mood held, -0.46 to -0.19, the readout at 23-24): the three never-typed facts told once before the night answered
1 of 3 after it ("a lemon" at 1.5 s; "what has wool?" -> "i do. he is wet", "what is salty?" -> "the shall I"); the taught
questions 16-30 answered 8 of 15 in the child's turn ("we drink water and milk", "a bird has wings", "bees make honey", "we
sleep in a bed", "honey is sweet", "fins", "snow is white", "ducks swim"), the taught set 16 of 30 live over the two sessions.
THE ONCE-TOLD MEMORY IS INTACT (04:17): read greedily on the morning save after night 214, the three facts told once before the
night answer 3 of 3 verbatim and seven facts never told 0 of 7; live they answered 1 of 3. The loss is the choice: the cortex's
habit ("i do.", "yes.") against a sure recall for the first symbol, the sample on the habit half the time. Decisiveness by
certainty (sharp_conf 3), measured before only on copies at a floor mood, goes on the served body at night 215's save, with a
rehearsal after it (the three old facts asked again; three new facts told once) and another after night 216 (the new three
asked after their night). Predicted: after-night recall 2 of 3 or better; the parent's counts not down.
NIGHT 215 (ended 04:53, the 205th row; the log's day 248, 117 lines, cut by the second rehearsal; facts 21-30): the held-out
0.643 (the dusk 0.650, above the morning), the facts by the cortex 0.883, the old lines 0.451; the mouth completes 24 of 26
prefixes and answers 17 of 30 questions at two rests and 21 at eight; the rephrased 9; the branch 9 of 9 and 10 of 12; the
store 37165. DECISIVENESS BY CERTAINTY WENT ON THE SERVED BODY at this save (the reload at 04:53:22, served again 04:54:17:
sharp_conf 3 beside the capacity, the hold and its form; the guard re-armed with the same flags). The third rehearsal begins.
THE THIRD REHEARSAL (04:58-05:04, the start of the log's day 249, decisiveness by certainty live; the body's mood -4.5 at the
start, its readout on the floor, and -5.8 at the end): the three facts told once two nights before answered 3 of 3 ("a lemon
is", "a sheep has wool", "the sea is salty"; 1 of 3 the morning before without the decisiveness); three new facts told once
and asked minutes later 1 of 3 ("a hill is"; "what is loud?" -> "is yes. whthe su"). Four of six both mornings: not yet a
verdict on the choice; the mood at the day's start is the confound (right after a night and a reload the readout sat at its
floor). The fourth rehearsal moves to mid-day, when the parent's smiles have lifted the mood; the body's mood is logged
through a day to place the recording.
NIGHT 216 (ended 05:49, the 206th row; the log's day 250, 138 lines, cut by the third rehearsal): the held-out 0.650 (the dusk
0.627), the facts by the cortex 0.883, the old lines 0.461; the mouth completes 24 of 26 prefixes and answers 20 of 30
questions at two rests and 22 at eight; the rephrased 12; the branch 9 of 9 and 10 of 12; the store 39071. THE MOOD THROUGH A
DAY (logged each minute from 05:05): the third rehearsal left it at -5.5; the parent's ordinary talk lifted it to +2 within
eight minutes and +5 within fifteen; it held between +1 and +5 through the day. The readout's floor at -4 and below is a
morning state after a night and a restart, not the day's; rehearsals and the recording belong ten minutes or more into a day.
THE FOURTH REHEARSAL (06:14-06:18, mid-day of the log's day 251, the mood -1.9 at the start and -3.7 at the end, the readout
17 -> 10): the six facts told once answered 1 of 6 after their nights ("a lemon is sour" at 4.4 s; "what is loud?" -> "a small
green leaf", "what has wool?" -> "iav i wht"). Across the sessions the after-night live recall of a once-told fact has read 1
of 3, 3 of 3 and 1 of 6: the memory holds greedily, the live choice does not hold it reliably, and the mood falls as the
failures go unsmiled. Next: three new facts told twice each, as a parent repeats, mid-day after night 217, asked mid-day after
night 218 with the old six; the rehearsal's parent reports its smiles and the mood at each question.
THE PARENT OF 248, 250 AND 251 REPORTED (06:24): 117/138/153 lines (the first two days cut by rehearsals); A's questions
answered before B 41 of 58, 46 of 69, 50 of 77; fact answers in its own turn 2, 3, 1; unprompted whole sentences 9, 17, 16;
its own question marks 15, 26, 24; smiles 188/223/204 (answer smiles 7, 6, 9), frowns 24/31/33. On day 251 it said "ice is
cold" unprompted after "why is the stone so cold?", a question in the parent's own words. A mechanical finding: a queue row
that straddles a night loses its remaining lines at the relaunch (the brief now puts facts in a row's first pair). The next
parent (252-254, facts 21-30 first) spawned at the 207th row.
NIGHT 217 (ended 06:43, the 207th row; the log's day 251, facts 11-20): the held-out 0.641 (the dusk 0.616), the facts by the
cortex 0.888, the old lines 0.463; the mouth completes 24 of 26 prefixes and answers 20 of 30 questions at two rests and 22 at
eight; the rephrased 11; the branch 9 of 9 and 10 of 12; the store 41078; the day: answer smiles 9, junk 9, question marks 24.
THE FIFTH REHEARSAL (07:08-07:22, mid-day of the log's day 252): three new facts told twice each ("a pear is a fruit", "a snail
is slow", "owls hoot") and asked the same day, twice each: 3 of 6 ("a snail is", "owls hoot" twice; "what is a pear?" -> "a pear
is shall", the chain right to the word boundary and lost there to the cortex's habit), the mood rising -1.4 to +0.7 under the
parent's 47 smiles; then the old six once-told facts after their nights: 2 of 6 ("a sheep has wool", "drum is loud"), the mood
falling -2.1 to -3.5 as the failures went unsmiled. Across the sessions the live recall of a told fact runs 7 of 18 after a
night and 7 of 12 the same day; the greedy readout holds nearly all of them; the losses sit at word boundaries where the sample
goes to the cortex's habit ("shall", "i do.", "yes."). The decisiveness constant is raised from 3 to 8 at night 218's save, the
sixth rehearsal mid-day under it; falsified if the nine told facts answer under 5 of 9, then back to 3 at the next save.
NIGHT 218 (ended 07:38, the 208th row; the log's day 252, 115 lines, cut by the fifth rehearsal): the held-out 0.633 (the dusk
0.625), the facts by the cortex 0.888, the old lines 0.460; the mouth completes 25 of 26 prefixes and answers 19 of 30 questions
at two rests and 22 at eight; the rephrased 12; the branch 9 of 9 and 10 of 12; the store 42735; the day: answer smiles 10,
junk 1, question marks 27. THE DECISIVENESS CONSTANT WENT FROM 3 TO 8 at this save (the reload at 07:38:52, served again
07:39:47; the guard re-armed with the same flags). The sixth rehearsal, mid-day, decides it.
THE MOOD THROUGH THREE HOURS (logged each minute, 05:05-08:04): within each day it swings between -5 and +5; for the first five
to ten minutes after a night and a restart it sits at -4 to -6 and climbs under the parent's talk (+2 within eight minutes, +5
within fifteen); each rehearsal pulled it down while it ran (the failures unsmiled) and the parent lifted it again after. At
08:00 it stood at +4.8, and the sixth rehearsal began there.
THE SIXTH REHEARSAL (08:04-08:10, mid-day of the log's day 253, decisiveness 8, the mood +4.6 at the start and -1.1 at the
end): the nine told facts answered 4 of 9 after their nights ("owls hoot", "a lemon is", "the sea is salty", "a hill is
steep"), the three told twice 1 of 3. Falsified; across six rehearsals no value of the constant separates (after a night 1 of
3 at 0; 3 of 3, 1 of 6, 2 of 6 at 3; 4 of 9 at 8). Decisiveness by certainty goes off at night 219's save. The rehearsals
teach the demo instead: the answer begins right and derails at a common-word branch ("a pear is a " never "fruit"), so the
told fact must have a distinctive continuation, and the same-day recall (7 of 12) is the safer scene than the overnight one
(11 of 27), which is shown honestly as sometimes.
NIGHT 219 (ended 08:34, the 209th row; the log's day 254): the held-out 0.626 (the dusk 0.607; the dusks 0.625, 0.616, 0.607
over three days: watched), the facts by the cortex 0.888, the old lines 0.471; the mouth completes 24 of 26 prefixes and
answers 21 of 30 questions at two rests and 24 at eight; the rephrased 14 of 30 (the highest yet); the branch 9 of 9 and 10 of
12; the store 44399; the day: answer smiles 11, junk 4, question marks 20. DECISIVENESS BY CERTAINTY WENT OFF at this save
(the reload at 08:34:02, served again 08:34:57 with the capacity, the hold and its form; the guard re-armed the same).
NIGHT 220 (ended 09:32, the 210th row; the log's day 256, facts 11-20): the held-out 0.625 (the dusk 0.596, the lowest yet; the
mornings 0.654, 0.643, 0.650, 0.641, 0.633, 0.626, 0.625 over the week, the dusks 0.647 -> 0.596: the day's fall about 0.03
and the night's restoration a little less), the facts by the cortex 0.883, the old lines 0.460; the mouth completes 24 of 26
prefixes and answers 20 of 30 questions at two rests and 23 at eight; the rephrased 15 of 30 (a new high); the branch 9 of 9
and 10 of 12; the store 46471. THE PARENT OF 252, 254 AND 256 REPORTED (09:20): 115/143/165 lines; A's questions answered
before B 58%, 59%, 69%; fact answers in its own turn 1, 3, 4 ("fish live in water" whole and unprompted); runs of three words
or more 15, 31, 29 ("the sea is salty", from the rehearsals' telling); its own question marks 27, 20, 23; smiles 180/228/251,
frowns 23/31/35; talk-overs rising with its own speech (61, 96, 101). The night boundary destroys one or two queue lines
whatever their place in a row. The next parent (257-259) spawned at the 210th row.
NIGHT 221 (ended 10:23, the 211th row; the log's day 257, facts 21-30): the held-out 0.623 (the dusk 0.600, the day's fall
-0.025), the facts by the cortex 0.890, the old lines 0.465; the mouth completes 24 of 26 prefixes and answers 22 of 30
questions at two rests and 23 at eight; the rephrased 14; the branch 9 of 9 and 10 of 12; the store 48637; the day: answer
smiles 9, junk 9, duty 0.293. THE DAY'S LESSON AT A THIRD WENT ON THE SERVED BODY at this save (the reload at 10:23:56, served
again 10:24:51, wake_base 0.3 beside the capacity, the hold and its form; the guard re-armed the same): two days, the dusk's
fall the ruler, -0.015 or better predicted, -0.025 or worse on either day the falsifier.
NIGHT 222 (ended 11:19, the 212th row; the log's day 259, the first full day with the day's lesson at a third): THE DUSK'S FALL
+0.002 (the dusk 0.625 against the morning 0.623; the four days before -0.029, -0.018, -0.019, -0.025): the prediction met on
the first day. The morning after 0.624, the facts by the cortex 0.890, the old lines 0.481 (its highest for a week); the mouth
completes 24 of 26 prefixes and answers 21 of 30 questions at two rests and 24 at eight; the rephrased 15; the branch 9 of 9
and 10 of 12; the store 50760; the day: answer smiles 7, junk 8, duty 0.311, A's questions answered before B 58 of 84. The
second day decides.
THE PARENT OF 257, 259 AND 260 REPORTED (12:13): 173/168/166 lines; A's questions answered before B 59%, 49%, 63%; fact answers
in its own turn 2, 5, 3 ("the sun is up in the day" whole; "grass is green", "an ant is little", "we drink water an"); it asked
"what is hard?" on day 259, was answered under a reworded line, and said "a rock is hard" unprompted the next day; runs of four
known words 4, 9, 5; its own question marks 28, 30, 25; smiles 215/249/262, frowns 37/34/36; known words 657 -> 690. One line
destroyed at a night boundary in three. The next parent (262-264, facts 21-30 first) spawned at the 213th row.
NIGHT 223's DUSK (12:15; the second day with the day's lesson at a third, the log's day 260): the held-out 0.589 against the
morning's 0.624, a fall of -0.035, the week's largest, after the first day's +0.002; the old lines 0.425. The test is
falsified on its second day and the lesson's rate returns to one at night 224's save. The day's fall is episodic: day 259 did
not fall and day 260 fell hard under the same rate, so the source is in the day's content, to be found by comparing those two
days' lines and the child's turns, not in the lesson's gain.
NIGHT 223 (ended 12:14, the 213th row): the night restored the hard day fully, the held-out 0.589 at dusk -> 0.640 in the
morning (the week's best morning), the facts by the cortex 0.888, the old lines 0.475; the mouth completes 24 of 26 prefixes
and answers 21 of 30 questions at two rests and 23 at eight; the rephrased 15; the branch 9 of 9 and 10 of 12; the store
52880. The hard day (260) against the soft one (259): more of the child's own speech (own characters 1417 against 1303, duty
0.344 against 0.311, turns of three words or more 50 against 43) and more rare words in the parent's lines (190 against 169);
both suspects lean the same way, the differences modest. The dusk's fall is tabulated against those features over the last
ten days next.
THE DUSK'S FALL OVER TWENTY-TWO DAYS (12:21): from +0.008 to -0.070, a mean near -0.02 and a scatter of 0.018, tracking none of
the day's features (the child's own speech, the parent's rare words, the talk-overs, the line count, all under 0.15); the
nights restore a little less than the days take, a net -0.0016 a morning. The lesson-rate test of two days could not resolve a
third of the fall against that scatter; a fair test needs eight days an arm and waits behind the demo.
NIGHT 224 (ended 13:10, the 214th row; the log's day 261, facts 1-10; the day with the most of the child's own speech yet, duty
0.403, own characters 1692): the held-out 0.613, the lowest morning so far (0.640 the morning before, 0.630 at dusk: a fall
of -0.010 by day and -0.017 by NIGHT, the first night to lower it), the facts by the cortex 0.883, the old lines 0.452; the mouth completes 24 of 26 prefixes and
answers 20 of 30 questions at two rests and 24 at eight; the rephrased 16 of 30 (a new high); the branch 9 of 9 and 10 of 12;
the store 55006; the day: answer smiles 9, junk 10, A's questions answered before B 72 of 84. THE LESSON'S RATE RETURNED TO
ONE at this save (the reload at 13:10:11, served again 13:11:06 with the capacity, the hold and its form; the guard re-armed
the same). The morning's held-out and the child's own speech now move in opposite directions across the week: the drift's
derivation waits behind the recording, with the eight-day test.
NIGHT 225 (ended 14:05, the 215th row; the log's day 263, facts 11-20, a quiet day of the child's, duty 0.271): the held-out
0.621 (the dusk 0.609, the day's fall -0.004, the night +0.012), the facts by the cortex 0.885, the old lines 0.466; the mouth
completes 24 of 26 prefixes and answers 21 of 30 questions at two rests and 23 at eight; the rephrased 13; the branch 9 of 9
and 10 of 12; the store 57086; the day: answer smiles 6, junk 4.
THE PARENT OF 261, 263 AND 264 REPORTED (15:17): 169/165/169 lines; A's questions answered before B 46 of 84, 41 of 82, 38 of
85; fact answers in its own turn before B 5, 6, 3, the highest yet ("water is wet", "the sun makes us warm", "grass is green"
whole); runs of three own words 41, 23, 31; its own question marks 20, 26, 27; smiles 282/249/279, frowns 38/35/35, all
talk-overs; all thirty facts complete in the first half of each day; one queue line lost at a night boundary; a two-word echo
went through undeferred. The next parent (265-267, facts 21-30 first) spawned at the 216th row.
NIGHT 226 (ended 15:0x, the 216th row): A NIGHT THAT DAMAGED THE CORTEX. Its lesson's loss rose across the rounds (0.200,
0.228, 0.342, 0.293, 0.279, 0.272) where night 225's fell (0.224 to 0.170); its gauge on its own dreams went 0.694 -> 0.605
(225: 0.671 -> 0.772); after it the facts by the cortex read 0.646 (0.885 the night before, 0.865 at the dusk), the held-out
0.602, the parent's lines 0.611. The material was ordinary (the dreams plain lines, the utterance memory as before); the store's
rulers stand (24 of 26 prefixes, 21 and 23 of 30 questions, the branch 9 of 9). The night stands, by the word; the night is
re-run on the dusk copy to see whether it diverges the same way, and night 227 shows whether the cortex recovers.
NIGHT 226 RE-RUN ON THE DUSK COPY (15:09-15:35): it did not diverge; its loss fell 0.195 to 0.154, its gauge on its dreams rose
0.698 to 0.787, the held-out 0.605 to 0.611, the parent's lines 0.667 to 0.725. Its dreams were other lines than the served
night's (the dusk copy is sixteen minutes of talk older than the night's start, and the draw depends on the memory then), so
the served night's divergence was a chance event of the optimization on that night's draws, not a property of the state or
the material. The night stands; night 227 tells whether the cortex recovers on its own. If it does not within three nights,
the night's lesson needs a derived robustness (the ledger's item 38).
THE DAMAGE LOCATED (16:15): between the dusk of day 264 and the morning after night 226 no weight of the transformer moved by
more than 0.0014, and the whole cortex moved by its ordinary night's amount; the night's lesson moved it in a harmful
direction, not by a shove. Beside the cortex the value critic has been running away for longer: band 5's value scale stood at
2630 at that dusk already, 7807 after night 226, 50304 after 227, its head's weights at 1162; the store's strengths (max 2.4),
the mood and the reward average are untouched by it, since what the head feeds is clamped. The night re-run from the healthy
dusk copy was normal; the night is now re-run from the damaged state to see whether it is stuck by itself.
THE NIGHT FROM THE DAMAGED STATE (16:15-16:40, the dusk copy of day 265): stuck by itself, its loss flat at 0.26 across the
rounds (0.287, 0.262, 0.260, 0.263, 0.254, 0.265; the healthy night 0.195 to 0.154), its gauge 0.546 -> 0.602, the held-out
0.58 -> 0.59. The cortex sits in a basin the night's lesson at its rate barely climbs; recovery by nights alone would take
many. The cortex transplant is measured on a copy next; whether it is ever applied to the served body is the user's call.
THE TRANSPLANT MEASURED (16:16-17:11, on a copy): the healthy cortex of day 264's dusk placed into the body of the morning after
night 227, the store (61339 slots), the utterance memory and the life kept: the facts by the cortex 0.875 (0.667 in the
damaged body), the held-out 0.605, the parent's lines 0.690, the prefixes 24 of 26, the questions 21 and 25 of 30; a night on
it ran normally (the loss 0.205 -> 0.165, its gauge 0.688 -> 0.773, the held-out to 0.616), where the night from the damaged
state stayed flat at 0.26. The repair is one organ restored to sixteen minutes before the divergence with nothing since lost;
whether it is applied to the served body is the user's call, by their word that the night stands.
NIGHT 228 (ended 17:22, the 218th row; the log's day 267): THE CORTEX RECOVERED ON ITS OWN. The facts by the cortex 0.577 at
dusk and 0.857 in the morning (0.885 before the accident), the parent's lines 0.677, the held-out 0.599 (the dusk 0.588), the
night's gauge on its dreams 0.69, the old lines 0.488; the mouth completes 24 of 26 prefixes and answers 21 of 30 questions at
two rests and 23 at eight; the rephrased 16 of 30 (a new high); the branch 9 of 9 and 10 of 12; the store 63401. The day (267):
smiles 329, answer smiles 2, junk 5, duty 0.382. The transplant is not needed while this holds; the donor stays in the
backups. THE FADE'S FLOOR WENT TO 0.25 OF THE MEAN at this save (the reload at 17:2x, served again 17:22:25; the guard re-armed).
THE MOMENT'S HORIZON MEASURED (17:12-17:38, the donor copy, the same draws as its control): at 0.99 the night ran as at
0.999 within a hundredth on its loss and its gauge (0.203 -> 0.159 and 0.781 against 0.205 -> 0.165 and 0.773), the
held-out the same; the parent's lines and the old lines a hundredth or two lower, a night's noise. It goes on the served
body at night 229's save; twenty nights without a diverging round is its test there.
THE PARENT OF 265, 267 AND 269 REPORTED (18:24): 171/167/172 lines; A's questions answered before B 54%, 46%, 48%; fact
answers in its own turn before B 6, 7, 4; whole fact sentences unprompted: "cats drink milk", "the sun is up in the day",
"we sleep in a bed" (two days before that fact was taught), "we drink water"; runs of four own words 21, 42, 31; its own
question marks 22, 32, 36; smiles 270/331/285, frowns 39/44/41; two lines lost at boundaries; the nights lengthening (866,
1432, 812 s) under the copies' load on the machine. The next parent (271-273, facts 21-30 first) spawned at the 219th row.
NIGHT 229 (ended 18:22, the 219th row; the first night under the fade's floor at a quarter of the mean and the moment's horizon
at 0.99): the store purged its backlog at once, 64000 -> 47158 (the slots faded under the new floor over the twelve days since
the capacity was raised), and the rulers stood or rose: the mouth completes 23 of 26 prefixes and answers 23 of 30 questions
at two rests and 24 at eight; the held-out 0.611 (the dusk 0.584), the facts by the cortex 0.879, the old lines 0.494, the
rephrased 15; the branch 9 of 9 and 10 of 12. THE MOMENT'S HORIZON WENT ON THE SERVED BODY at this save (the reload at
18:2x, served again 18:23:17 with all five constants; the guard re-armed). The store's level over the next nights is the
floor's test (above forty thousand, under the capacity).
NIGHT 230 (ended 19:2x, the 220th row): the store 42724 after another purge of 7673 (the floor at a quarter of the mean feeds
itself: the mean rises as the weak go), and the mouth completes 20 of 26 prefixes (23 the night before): the floor's
falsifier. The questions 21 and 21 of 30; the held-out 0.599 (the dusk 0.626), the facts by the cortex 0.879, the rephrased
13; the branch 9 of 9 and 10 of 12; the night's own curve healthy under the moment's horizon (0.212 to 0.171, its gauge 0.654
to 0.767). The floor returns to a tenth at night 231's save; an absolute floor is the form to derive later.
NIGHT 231 (ended 20:18, the 221st row; the last night at the floor of a quarter): the store 41511 after a drop of 4521; the
mouth completes 20 of 26 prefixes and answers 19 of 30 questions at both pauses (the purge took the chains of the less
retold facts, which the parents' cycle retells within three days); the held-out 0.613 (the dusk 0.575), the facts by the
cortex 0.888, the old lines 0.493, the rephrased 10; the branch 9 of 9 and 10 of 12; the night's own curve healthy (0.206 to
0.167, its gauge 0.681 to 0.768). THE FLOOR RETURNED TO A TENTH at this save (the reload at 20:18, served again 20:19:05 with
the capacity, the hold and its form, the moment's horizon; the guard re-armed the same).
THE PARENT OF 271, 272 AND 274 REPORTED (21:16): 166/173/179 lines; A's questions answered before B 39%, 29%, 34% (55 to 75%
earlier in the week); fact answers in its own turn 1, 2, 3 ("the sun is hot", "grass is green", "we eat bread"), and "we eat
bread and eggs" unprompted under an unrelated line; answer smiles 2, 2, 2; turns of four words 32, 32, 15; its own question
marks 26, 26, 20; smiles 268/255/236, frowns 37/35/41; the child starting earlier into A's line (the late misses down, the
talk-overs up). The soft days follow the divergent night and the two-night purge; the cycle retells the dropped chains. A
mechanical finding: lines queued at a night are typed first the next day and delay that day's facts; the brief now asks for
the depth to fall toward 45 as a day ends. The next parent (275-277, facts 21-30 first) spawned at the 222nd row.
THE REVIEW'S FINDING (21:35): an independent review read the served saves' own constants and found three reverts that never took
effect: decisiveness 8 (recorded off at night 219), the day's lesson at a third (recorded off at 224) and the fade's floor at
a quarter (recorded back to a tenth at 231) all live, and the slow context in the key at 0.5 since day 206 (recorded reverted
at night 191). The cause: a save carries its constants, a reload passes only the flags file's keys, and a key left out keeps
the save's value; I had written that lesson for the copies' controls on the 15th and did not apply it to the served body. The
store is still purging (41251). From night 233's save every revert is explicit in the flags and the slow context is explicit
and disclosed as on; ops/served_cfg.py prints the effective constants after every reload. The verdicts of nights 219-232 are
re-read in the ledger's item 39.
AFTER THE REVIEW (21:43): the answer scoring of every ruler is anchored at a word boundary before the answer's word (a match
inside another word no longer counts; the child's run-ons, "honeyhave", still do, being answers given); on the save after night
232 the questions read 20 of 30 at both pauses under it, 19 by the old rule. The fact prefixes are named for what they are
(tools/fact_prefixes.txt, cued recall of answers heard about twenty-five times); the false deferral rule is struck from the
parent brief; the rehearsal is relabelled a second parent on the live body; the spec carries the chunk form's space boundary
as a text-specific prior and points the constants at ops/served_cfg.py; the demo's claim is rephrased to survive a skeptic,
with the caregiver's answer smile disclosed. From night 233's save every revert is explicit in the flags.
NIGHT 233 (ended 22:11, the 223rd row; the log's day 275, facts 21-30): the held-out 0.604, the facts by the cortex 0.879, the
old lines 0.466; the mouth completes 23 of 26 fact prefixes and answers 20 of 30 questions at two rests and 21 at eight; the
rephrased 12; the branch 9 of 9 and 10 of 12; the store 41361, the last night of the purge; the day: answer smiles 0, junk 3,
duty 0.309. THE EXPLICIT REVERTS LANDED at this save (the reload at 22:11, served again 22:12:40): the served constants read
from the flags now, decisiveness 0, the lesson's rate 1.0, the floor a tenth, the slow context 0.5 disclosed, beside the
capacity, the hold, its form and the moment's horizon; verified with ops/served_cfg.py against the save.
THE SEVENTH REHEARSAL (22:36-22:41, mid-day of the log's day 276, the constants as recorded at last, the scoring anchored): the
taught questions 1-15 answered 6 of 15 in the child's turn: 3 of the first 5 while the mood stood above zero ("ice is", "water
is wet", "birds fly up"), 3 of the next 10 as the mood sank from +2.2 to -5.3 and the readout flattened to its floor (34 to 8).
The drain feeds itself: a failed answer earns no smile, the mood falls, the readout flattens, the next answer fails. The take
starts above +2, keeps its questions few, and the parent smiles at every sensible word.
NIGHT 234 (ended 23:09, the 224th row; the log's day 277, facts 1-10; the label 276 an aborted day of one line): the held-out
0.600, the facts by the cortex 0.879, the old lines 0.488, the parent's last lines 0.744; the mouth completes 23 of 26 fact
prefixes and answers 18 of 30 questions at two rests and 19 at eight; the rephrased 13; the store 44006; the day 277: 155
lines, 56 of 77 A-lines with letters in its turn, answer smiles 3, junk 6, duty 0.332, frowns 31.
NIGHT 235 (ended 00:06 on the 17th, the 225th row; the log's day 278, facts 11-20): the dusk before it read the held-out 0.583
and the night raised it to 0.620, the facts by the cortex 0.847 to 0.881, the old lines 0.498; the mouth completes 23 of 26
prefixes, answers 18 of 30 at two rests and 21 at eight; the rephrased 12; the branch 9 of 9 and 10 of 12; the store 46746,
growing 2700 a day since the purge ended, the cap of 65536 seven days off at this rate (the eviction at the cap was the
regression's cause, the ledger's item 2: the absolute floor is due before then); the day 278: 172 lines, 58 of 86 A-lines
with letters in its turn, 18 answered with I/yes/no/please before B, answer smiles 0, junk 3, duty 0.255, frowns 35.
THE PARENT 275-278 (real days 275, 277, 278; all thirty facts asked in each day's first half; 38-character lines, no commas):
by the parent's own count the child spoke a content word or yes/no before B on 53 of 89, 50 of 77 and 45 of 86 A-questions;
the fact's word in its own turn before B said it 1, 4 and 4 of 10 ("bees make honey" produced whole); one fact exchange lost
to the night boundary after day 275 and asked again the next day. The next parent (279-281, facts 21-30 first, stop at the
228th row) spawned at 00:08 with the queue at 61 lines. The night waiters armed before the review stand through the 247th
label (the probe waiters too); a script (night_waiter.sh N, the label N+10) arms the nights beyond.
NIGHT 236 (ended 01:04, the 226th row; the log's day 279, facts 21-30): the dusk read the held-out 0.606, the night 0.614; the
facts by the cortex 0.853 to 0.881, the old lines 0.502, the parent's last lines 0.735; the mouth completes 23 of 26 prefixes,
answers 18 of 30 at two rests and 20 at eight; the rephrased 12; the branch 9 of 9 and 10 of 12; the store 49601 (+2855); the
day 279: 175 lines, 63 of 88 A-lines with letters in its turn, 13 answered with I/yes/no/please before B, answer smiles 0,
junk 5, duty 0.297, frowns 37. The floor's cut on this morning's save is being measured on a copy.
THE FLOOR'S CUT MEASURED (01:14-01:18, on a copy of the save after night 236): 4649 of 49601 slots below 0.078 forgotten at
once (the band the relative floor never removed); against the same save untouched the prefixes 22 against 23 ("ice is " lost
to a competing memory), the questions 17 and 19 against 18 and 20 (two lost, "what is big?" and "what has wings?", one gained,
"what do cows give?"), the rephrased, the branch, the held-out and the old lines the same. Under the falsifier's three: the absolute floor goes on at night 237's save (the reload armed 01:20, the flag
explicit in every flags file, the guard relaunched after the landing). The first night under it will forget about the same
band; the falsifier on the served body is three under the mornings' range on the first three mornings.
THE FLOOR LANDED (02:02:34, after night 237's save, the 227th row): the served body restarted (pid 87146) with
--store-floor-abs 0.07 beside the explicit set, the typist relaunched by the chain, the guard relaunched with the new
arguments; ops/served_cfg.py reads the floor from the flags and no reverted constant from the save alone. Night 238 is
the first night under it; it will forget the low band at once (about a tenth of the store), then the day's writes a night.
NIGHT 237 (ended 02:01, the 227th row, the last under the relative floor; the log's day 280, facts 1-10): the dusk read the
held-out 0.598, the night 0.614; the facts by the cortex 0.867 to 0.890, the old lines 0.483, the parent's last lines 0.726;
the mouth completes 23 of 26 prefixes, answers 20 of 30 at two rests and 21 at eight; the rephrased 13; the branch 9 of 9
and 10 of 12; the store 52474; the day 280: 171 lines, 60 of 85 A-lines with letters in its turn, 14 answered with I/yes/no/
please before B, answer smiles 4, the child's own question marks 31, junk 5, duty 0.331, frowns 36. These are the floor's
reference mornings: prefixes 22-23, questions 18-21 over the four nights before it.
THE PARENT 279-282 (real days 279, 280, 282; the label 281 an aborted session at the reload's relaunch, one line consumed and
never typed): by the parent's count the child spoke a content word or yes/no before B on 54 of 88, 44 of 85 and 52 of 86
A-questions; the fact's word in its own turn 4, 4 and 4 of 10; whole fact sentences before B: none on 279, "birds fly up" and
"the sun is up in" on 280, "an ant is little" and "we drink water" on 282; its own fact-shaped questions twice ("what is wet?"
right after "ice is cold"; "where do fi"), answered under other wording; unprompted runs of four words or more 12, 12, 19;
talk-overs 97, 118, 110 a day (a frown for every third); the typist about fifteen seconds a line, slowing to twenty-eight
before each night; one draft line in eighteen rejected as a duplicate of the queue's 4778 rows. The next parent (283-285,
facts 21-30 first, stop at the 231st row) spawned at 03:00 with the queue at about 63 lines.
NIGHT 238 (ended 02:58, the 228th row; THE FIRST NIGHT UNDER THE ABSOLUTE FLOOR; the log's day 282, facts 11-20): the store
about 55000 at dusk to 47967 by morning (the low band forgotten at once, as the cut had shown); the dusk read the held-out
0.621, the morning 0.607; the facts by the cortex 0.804 at dusk to 0.883; the old lines 0.477, the parent's last lines 0.740,
the dreams 0.85; the mouth completes 22 of 26 prefixes ("ice is " to "coming tomor", the loss the cut copy showed) and answers
19 of 30 at two rests and 21 at eight; the rephrased 13; the branch 9 of 9 and 10 of 12. Against the reference (prefixes
22-23, questions 18-21) nothing is three under: the falsifier is not met on the first morning. The day 282: 172 lines, 66 of
86 A-lines with letters in its turn, 7 answered with I/yes/no/please before B, answer smiles 2, junk 5, duty 0.325, frowns 36.
The night's own report: 8028 slots forgotten (the band below 0.078 and the day's low writes), 1024 dreams at 23.7 symbols, the
curve 0.206 to 0.154 over six rounds (no round rose), the gauge 0.655 to 0.795; the store's weakest slot stands at 0.070 now
and 2104 slots sit in the band the next night takes, so the store grows by about eight hundred a night toward its steady state.
NIGHT 239 (ended 03:54, the 229th row, the second under the absolute floor; the log's day 283, facts 21-30): the dusk read the
held-out 0.599, the night 0.604; the facts by the cortex 0.873 to 0.888, the old lines 0.481, the parent's last lines 0.757,
the dreams 0.77; the mouth completes 22 of 26 prefixes and answers 19 of 30 at two rests and 22 at eight ("ice is cold" back at
eight); the rephrased 12; the branch 9 of 9 and 10 of 12; the store 48458 (+491: the growth is the day's writes less the
band the floor takes, about five hundred a night now). The day 283: 173 lines, 66 of 87 A-lines with letters in its turn, 8
answered with I/yes/no/please before B, answer smiles 4, junk 1, duty 0.309, frowns 38. The falsifier is not met on the
second morning either.
NIGHT 240 (ended 04:50, the 230th row, the third under the absolute floor; the log's day 284, facts 1-10): the dusk read the
held-out 0.600 and the prefixes 24 of 26; the night's morning: the held-out 0.607, the facts by the cortex 0.881, the old lines
0.488, the parent's last lines 0.741, the dreams 0.83; the mouth completes 23 of 26 prefixes and answers 21 of 30 at two rests
and 22 at eight; the rephrased 14, the most yet; the branch 9 of 9 and 10 of 12; the store 48766 (+308). The day 284: 172
lines, 55 of 86 A-lines with letters in its turn, 13 answered with I/yes/no/please before B, answer smiles 2, junk 7, duty
0.287, frowns 38. THE FLOOR STANDS: three mornings under it read the prefixes 22, 22, 23 and the questions 19/21, 19/22, 21/22
against the reference 22-23 and 18-21; the store 47967, 48458, 48766, settling under the capacity as derived.
THE PARENT 283-285 (real days 283, 284, 285; the three nights under the absolute floor): by the parent's count the child spoke
before B on 63 of 87, 54 of 86 and 52 of 85 A-questions (72, 63 and 61 percent); the fact's word in its own turn 4, 2 and 6 of
10, day 285 the most of any day yet ("grass is green", "we drink water and milk", "bees make honey" whole, "we eat bread and eg"
before B); turns of three words or more 46, 46, 38; its own question marks 21, 23, 18, one of them a fact's ("what is little?"),
answered under other wording. Mechanical: no line dropped in 516; the log's rows lag the typist by two or three minutes near
a day's end; the night's row is written at the END of the night, so a night is detected by the queue freezing; a carry-over of
69 lines at night 239 pushed day 284's facts to minutes 17-26 and that day gave the fewest fact answers (2), the next two
nights held at 45 and day 285 gave 6. The next parent (286-288, facts 21-30 first, stop at the 234th row) spawned at 06:00.
NIGHT 241 (ended 05:47, the 231st row, the fourth under the absolute floor; the log's day 285, facts 11-20): the dusk read the
held-out 0.597, the night 0.623, the highest morning yet; the facts by the cortex 0.832 to 0.873, the old lines 0.473, the
parent's last lines 0.745, the dreams 0.78; the mouth completes 22 of 26 prefixes and answers 21 of 30 at two rests and 22 at
eight; the rephrased 14; the branch 9 of 9 and 10 of 12; the store 49313 (+547). The day 285: 171 lines, 56 of 85 A-lines
with letters in its turn, 3 answered with I/yes/no/please before B, answer smiles 4, junk 2, duty 0.272, frowns 33.
NIGHT 242 (ended 06:43, the 232nd row, the fifth under the absolute floor; the log's day 286, facts 21-30): the dusk read the
held-out 0.599 and the prefixes 19 of 26; the morning: the held-out 0.619, the facts by the cortex 0.881, the old lines 0.492,
the parent's last lines 0.772, the dreams 0.82; the mouth completes 19 of 26 prefixes (the dusk's count: "water is " lost to
"warm and who", B having typed "no. the water is warm" that day, the five-symbol key's collision with the day's talk, beside
"ice is " lost since the floor's cut) and answers 21 of 30 at two rests and 22 at eight; the rephrased 15, the most yet; the
branch 9 of 9 and 10 of 12; the store 49549 (+236). The day 286: 171 lines, 61 of 86 A-lines with letters in its turn, 9
answered with I/yes/no/please before B, answer smiles 3, its own question marks 31, junk 5, duty 0.291, frowns 35. The
prefixes' fall is the day's writes at a shared key, not the floor: the floor forgets the weak, and "the water is warm" was
written that day at full surprise.
THE EIGHTH REHEARSAL (07:10-07:15, 26 minutes into the log's day 287, the typist frozen, the caregiver's smiles on; mood +2.2
and the readout 34 at the start): a never-typed fact told once as an exchange ("what is slow?" / "a snail is slow"), then five
taught questions, then the told fact asked back. The child: "what is cold?" -> "i snail is slowswee" (the fresh memory intruding:
the told question shares the frame "what is" with the asked one); "what do cats drink?" -> "cats drink" (milk not reached in
the window); "what do bees make?" -> "bees make honeyis" at 10.7 s; "what is white?" -> "rsnow is whitewho is scratch" (the
answer, failed by the ruler's word boundary for the stray r); "what do ducks do?" -> "hat ducks swims" at 2.5 s; five minutes
after the telling, "what is slow?" -> "a snail is it" at 5.3 s. By the ruler 2 of 5 taught and 1 of 1 told once; by a viewer
3 or 4 of 5. The mood fell +2.2 to -4.2 and the readout 34 to 8 across the six questions, the seventh rehearsal's drain again.
Two rules confirmed for the take: three or four questions at most; the told fact's question shaped unlike the taught ones
(my "what is slow?" beside "what is cold?" and "what is white?" broke the script's own rule and intruded on the first answer).
NIGHT 243 (ended 07:39, the 233rd row, the sixth under the absolute floor; the log's day 287, facts 1-10, the day shortened to
149 lines by the rehearsal's freeze): the dusk read the held-out 0.616 and the prefixes 20; the night: 3085 slots forgotten,
the store 49696, the curve 0.184 to 0.140, the gauge 0.708 to 0.811; the morning: the held-out 0.625, the highest yet, the
facts by the cortex 0.875, the old lines 0.473, the parent's last lines 0.757, the dreams 0.82; the mouth completes 19 of 26
prefixes and answers 20 of 30 at two rests and 23 at eight, the most at eight yet; the rephrased 12; the branch 9 of 9 and 10
of 12. The day 287: 60 of 74 A-lines with letters in its turn, 4 answered with I/yes/no/please before B, answer smiles 1,
junk 3, duty 0.306, frowns 30.
NIGHT 244 (ended 08:35, the 234th row, the seventh under the absolute floor; the log's day 288, facts 11-20): the dusk read the
held-out 0.604 and the prefixes 19; the night: 3248 slots forgotten, the store 49960, the curve 0.218 to 0.152, the gauge 0.68
to 0.793; the morning: the held-out 0.632, the highest yet, the facts by the cortex 0.875, the old lines 0.475, the parent's
last lines 0.761, the dreams 0.80; the mouth completes 19 of 26 prefixes and answers 19 of 30 at two rests and 20 at eight; the
rephrased 12; the branch 9 of 9 and 10 of 12. The day 288: 167 lines, 62 of 84 A-lines with letters in its turn, 1 answered
with I/yes/no/please before B, answer smiles 0, junk 3, duty 0.308, frowns 37. The next parent (289-291, facts 21-30 first,
stop at the 237th row) spawned at 08:40 with the queue at 43 lines.
THE PARENT 286-288 (real days 286, 287, 288): by the parent's count the child spoke before B on 61 of 86, 60 of 74 and 62 of
84 A-questions (71, 81 and 74 percent, the 81 the highest of any day); the fact's word in its own turn 4, 1 and 1 of 10 (day
286: honey, snow, sun, "swims"; day 287: "rain fa" and "the m" cut at the window; day 288: green), against day 285's 6; day 287
carried the eighth rehearsal's freeze and the mood it drained (+2.2 to -4.2 at minute 26), which the count reflects; the
carry-over at each night held to 35, 30 and 31 lines, the facts at minutes 7-21, and the earlier placement did not by itself
raise the fact answers. Unprompted whole sentences 4, 2, 7; its own question marks 30, 17, 18; no line dropped in 487, one
exchange straddling a night without loss; the duplicate collision one in seventeen on the commonest frames.
NIGHT 245 (ended 09:31, the 235th row, the eighth under the absolute floor; the log's day 289, facts 21-30): the dusk read the
held-out 0.591 and the prefixes 19; the morning: the held-out 0.621, the facts by the cortex 0.881, the old lines 0.500, the
parent's last lines 0.796, the dreams 0.81; the mouth completes 20 of 26 prefixes ("ice is " to "cold" again) and answers 18 of
30 at two rests and 20 at eight; the rephrased 10; the branch 9 of 9 and 10 of 12; the store 49802, the first morning it did
not grow (the day's writes and the night's forgetting now even: the steady state near fifty thousand, as derived). The day
289: 164 lines, 61 of 82 A-lines with letters in its turn, 3 answered with I/yes/no/please before B, answer smiles 3, smiles
348 (2.07 a line, the most yet), junk 3, duty 0.343, frowns 37.
NIGHT 246 (ended 10:28, the 236th row, the ninth under the absolute floor; the log's day 290, facts 1-10): the dusk read the
held-out 0.602 and the prefixes 19; the night: 3375 slots forgotten, the store 49637, the curve 0.175 to 0.133, the gauge 0.707
to 0.825; the morning: the held-out 0.635, the highest yet (0.604 to 0.635 across the floor's nine mornings), the facts by the
cortex 0.888, the old lines 0.493, the parent's last lines 0.764, the dreams 0.82; the mouth completes 19 of 26 prefixes and
answers 18 of 30 at two rests and 20 at eight; the rephrased 10; the branch 9 of 9 and 10 of 12; the served constants read
again from the flags (the floor 0.07, decisiveness 0, the lesson's rate 1.0, the slow context 0.5). The day 290: 171 lines,
60 of 85 A-lines with letters in its turn, 11 answered with I/yes/no/please before B, answer smiles 1, junk 4, duty 0.300,
frowns 37.
THE CRITIC'S RUNAWAY RECEDED (11:12, read on the morning saves): band 5's value scale, 50304 after night 227's divergence, reads
2972 after night 237 and 2165 after night 246, its typical value about 46; the ventral critic's head, 1162 at its worst, 247
and then 92 (its norm 95). The head's evidence forgets at 36000 ticks (about three days of its clock), and the divergence's
evidence has been forgotten; nothing was done to it. The slow bands' heads read 46, 234 and 1254 (bands 5, 6, 7, the
differential ones, whose states shrink with the clock), unchanged between the two saves: a scale, not a walk. The bounded
form I had queued for after the take is not needed while this holds; the falsifier of the diagnosis is the scale or the
head rising again over a week (read on the morning saves with the store instruments' companion, the value scale by night).
The next structural lever for its learning is the store's five-symbol key (the dentate-style separation), after the take.
NIGHT 247 (ended 11:26, the 237th row, the tenth under the absolute floor; the log's day 291, facts 11-20): the dusk read the
held-out 0.613 and the prefixes 17; the night: 3306 slots forgotten, the store 49325, the curve 0.178 to 0.133, the gauge 0.726
to 0.825; the morning: the held-out 0.634, the facts by the cortex 0.883, the old lines 0.499, the parent's last lines 0.781;
the mouth completes 17 of 26 prefixes ("ice is " to "in your cup", "water is " to "warm for me", the day's talk at those keys)
and answers 17 of 30 at two rests and 19 at eight; the rephrased 9; the branch 9 of 9 and 10 of 12. The prefixes have fallen
from 23 (nights 233-237) to 19-20 and now 17 across the floor's ten mornings while the held-out rose 0.604 to 0.634 and the
questions held 17-21: to be traced on the morning save (which slots win the reads at "ice is " and "water is ") before any
verdict on the floor. The day 291: 169 lines, 67 of 85 A-lines with letters in its turn, 10 answered with I/yes/no/please
before B, answer smiles 3, junk 9, duty 0.339, frowns 38. THE RELOAD at this save (11:26:39, served again 11:27:35, pid
69838) carries the redrawn talk page on the body's own port; the constants verified from the flags.
THE PARENT 289-291 (real days 289, 290, 291): by the parent's count the child spoke before B on 60 of 82, 59 of 85 and 66 of
85 A-questions (73, 69 and 78 percent); the fact's key word in its own turn 3, 1 and 2 of 10 (any word of the answer 4, 3,
4); whole fact sentences before B: "fish have fins" (289), "an ant is little" (291); its own fact-shaped questions answered
under other wording; turns of three words or more 63, 47, 60; no line dropped in 438, both nights splitting a row cleanly.
A handoff overlap: the previous parent's last four rows landed after this one's first depth reading and pushed day 289's
facts to minute 26; the brief now says to trust the depth, not the row count. The next parent (292-294, facts 21-30 first,
stop at the 240th row) spawned at 11:27 with the queue at 27 lines.
THE PREFIXES' FALL TRACED (11:40-11:48, tools/read_trace.py on the save at night 237's reload, the last under the relative
floor, and on this morning's): at "ice is " the fact's slot won before (57 percent of the read's weight, strength 0.58, the
next symbol "c") and this morning a slot written from the parent's "ice is in your cup" (strength 0.77, its key a hair
closer, 2.006 against 1.992) takes 51 percent, the fact's slot second at 26 with its strength 0.51; eighteen keys sit near
the query on both saves with the same run of strengths. At "water is " the answer's first symbol still wins (61 against 76
percent before) beside two fresh slots of "the water is very..."; at "the sun is " nothing moved (80 percent). The reads are
by content alone (the read's strength weight is 0), so the forgotten weak slots never carried these votes: the fall is the
five-symbol key's collision with the days' natural talk, the twenty-eighth defect's family, and the fix is the body's
(the dentate-style separation, after the take), not a parent steered away from the facts' frames. The questions that fail
this morning read the same way: "what is cold?" -> "warm is beside", "what has four legs?" -> "a bird has wings", "what is
the sun?" -> "the sun is hot".
NIGHT 248 (ended 12:22, the 238th row, the eleventh under the absolute floor; the log's day 293 after the aborted 292, facts
21-30): the morning: the held-out 0.633, the facts by the cortex 0.885, the old lines 0.491, the parent's last lines 0.799,
the dreams 0.86; the mouth completes 17 of 26 prefixes and answers 16 of 30 at two rests and 17 at eight; the rephrased 10;
the branch 9 of 9 and 8 of 12; the store 49215. The day 293: 168 lines, 64 of 84 A-lines with letters in its turn, answer
smiles 1, its own question marks 29, junk 9, duty 0.341, frowns 36. ACROSS THE FLOOR'S ELEVEN MORNINGS the store's rulers
have fallen in step: the questions at two rests 19, 19, 21, 21, 21, 20, 19, 18, 18, 17, 16; the prefixes 22, 22, 23, 22, 19,
19, 19, 20, 19, 17, 17; the rephrased 13, 12, 14, 14, 15, 10, 12, 12, 10, 9, 10; the branch's second set 10 of 12 until this
morning's 8; while the held-out rose 0.607 to 0.633 and the cortex's facts held 0.88. The falsifier I set (three under the
range on the first three mornings) was shorter than the fade's timescale: the weak traces go over ten nights, not one. The
best-measured period of the store's rulers was at the capacity (nights 226-228: the questions 21 and 25, the prefixes 24,
the rephrased 16), with the relative floor forgetting almost nothing and the capacity's eviction the forgetting. Measured
fact by fact next (the answer's share of the read at each question on the pre-floor save and on this morning's).
THE FLOOR FALSIFIED ON THE SERVED BODY (12:35-13:00; tools/read_trace.py --qa on the pre-floor save and this morning's, on the
real read, the instrument corrected after the review: its first form scored the sentence's opener alone, an article or the
question's own word for most facts): the onset, the sentence's first symbol at the question, wins 25 of 30 now against 24
before; the middle, the answer word's first letter with the sentence taken as the body's own up to it, 24 of 30 now against
26 before, its mean share 0.77 against 0.81 ("a tree" 0.06 against 0.39; "water" after "we drink " lost to "ice"); the
free-running ruler compounds such losses (21 to 16). The damage is after the onset, and the mechanism is supported, not
proven:
the answer's continuation is carried by the episode's chain, slot to slot through the utterance, and the middle of a
well-known utterance is written weakly (the surprise of a predicted symbol, 0.05 to 0.15; a tenth of the day's writes lie
under 0.07), so the absolute floor at 0.07 forgets those slots within a night or a few, the links that carried an answer
past its first letter point to nothing, and the read falls back to the fast key, where the day's talk collides ("ice is in
your cup"). Under the relative floor a 0.05 write lived four nights and the three-day retelling refreshed it; at the
capacity the eviction took the faded tail first. Hence the slow fall of every store ruler across eleven mornings while the
onset reads and the cortex held. THE REVERT: --store-floor-abs 0 explicit in every flags file (the 0.07 set kept as
BASE_FLAGS_floorabs07.txt), the reload armed for night 249's save, the guard relaunched after it; the relative floor at a
tenth stays and the capacity of 65536 is the forgetting (the weakest gives way at each write over it), the regime of the
best-measured mornings (nights 226-228). Expected: the chains rebuild through the three-day cycle, the questions back to
18-21 within two cycles. THE LESSON, for the method: a change of forgetting shows on the store's turnover (twenty nights),
not on three mornings; its falsifier must be read over that span on the served body, since no copy can live it.
THE REVIEW OF THE DAY'S CHANGES (12:35-13:05, an independent reviewer over the diff since last night; nothing changed by it): the
organ suite 64 of 64; the floor at zero bit-identical to the old behaviour, every caller updated, the explicit zero winning over
the save's 0.07 at the load, the three flags files identical, the armed reload carrying the revert. Found and fixed: the
question trace's first form scored the sentence's opener, not the answer (above); the talk page's first load could draw the
whole transcript twice under a slow first answer (one poll in flight now, the first load sliced by its since), the arrow
keys gave a face while a line was being edited and with modifiers held (the arrows work only while the box is empty, the
modifiers ignored), the page inserted a space the body did not say (a thin gap now), a trimmed page could not recover (the
state carries its base), a bad answer from the page's server was not shown, the trim could orphan an open bubble, a click
stole the selection; the page's server binds the loopback only; the night waiter's dusk-log glob failed past the 299th row;
the six armed morning probes lacked the per-item listing (re-armed with it, saves 225-230); the restart procedure's numbering.
A research note from the review, to watch: the relative floor's threshold is a tenth of the mean, and the absolute floor's
eleven nights removed the weak tail and raised the mean (0.33 to 0.35), so the first nights back under it drop more than the
pre-floor regime did until the tail rebuilds; the store's dropped count on nights 249-252 is read before the floor's reference
is taken from night 237.
THE FLOOR REVERTED (13:20:18, after night 249's save, the 239th row): the served body restarted (pid 82759) with
--store-floor-abs 0 explicit; ops/served_cfg.py reads "FLAGS OVERRIDE the save's 0.07" for it and the rest from the flags; the
guard relaunched with the new arguments; the redrawn talk page with the review's fixes on the body's own port. The relative
floor at a tenth and the capacity's eviction are the forgetting again; the store rebuilds toward the capacity over about a
week, and the questions are expected back at 18-21 within two retelling cycles.
NIGHT 249 (ended 13:19, the 239th row, the twelfth and last night under the absolute floor, its save the one the revert reloaded;
the log's day 294, facts 1-10): the dusk read the held-out 0.622 and the prefixes 18; the morning: the held-out 0.627, the
facts by the cortex 0.881, the old lines 0.498, the parent's last lines 0.806, the dreams 0.84; the mouth completes 18 of 26
prefixes (the failed: ice is, grass is, an ant is, a bird has, a cat is, honey is, snow is, a rock is) and answers 18 of 30 at
two rests and 19 at eight (the failed: cold, dogs, up in the day, big, we eat, we drink, four legs, we sleep, soft, birds live,
the sun, hard); the rephrased 11; the branch 9 of 9 and 8 of 12; the store 48890. The day 294: 163 lines, 56 of 81 A-lines
with letters in its turn, 7 answered with I/yes/no/please before B ("what is cold?" -> "i o you"), answer smiles 3, junk 3,
duty 0.335, frowns 36. The morning probes list every prefix and question from this save on.
THE CODE MADE READABLE (13:05-13:45, on a copy of the tree, nothing on the served body changed): a behaviour guard first
(tools/determinism_check.py: a tiny body at a fixed seed under the served constants lives a fixed script and a night; every
weight, the store, the utterance memory, the page and the feelings hashed), then the edits, each measured by it under the served
constants and under the defaults and by the organ suite. The physiology's 188 constants grouped by organ with one-line meanings,
their values checked equal, the 197 lines of history that sat among them moved word for word to the spec's appendix; a code map at
the top of life.py and model.py; the tick, 535 lines, split into eight phase methods with explicit parameters and returns, each
body byte-identical (_sense, _hear, _learn_values, _own_face, _choose, _act, _feel_and_learn, _bookkeep); the night's docstring
and banners; dead imports and names removed in the body and the tools; tools/README; ops/archive for the kept flag sets and the
superseded scripts, ops/README; the README rewritten for the body as it runs. The cleaned tree and the commit before the cleanup
give the same digests (023f6e9b under the served set, 6b619a79 under the defaults; the guard's own first form had run its night
without gradients and was fixed). The served body takes the cleaned code at its next reload; it behaves the same.
THE PARENT 292-294 (real days 293, 294, 296; the label 295 aborted by the revert's relaunch; the last three days under the
absolute floor and the first under the relative one): by the parent's count the child spoke before B on 45 of 84, 49 of 81 and
43 of 82 A-questions (54, 60 and 52 percent, down from 70-80 under the earlier parents, the floor's days); the fact's word in its
own turn 1, 1 and 3 of 10 ("bees make honey" whole on day 296); runs of four words 6, 11, 8 ("we sit here and drink it", "the
grass is soft"); its own question marks 29, 18, 21 ("let us go in to sleep?" whole); talk-overs flat at 106-114 a day; two
queue lines lost at night boundaries, neither a fact; the duplicate collision one in fourteen, clustered on "do you want" and
"where is"; the carry-over 46-55 lines puts the facts at minutes 11-20, the structural floor. The next parent (297-299, facts
21-30 first, stop at the 243rd row) spawned at 14:20 with the queue at 48 lines.
NIGHT 250 (ended 14:16, the 240th row, THE FIRST NIGHT BACK UNDER THE RELATIVE FLOOR; the log's day 296, facts 11-20): the night's
own report: 619 slots forgotten (the relative threshold 0.0295, a tenth of a mean raised by the absolute floor's removal of
the weak tail, as the review foresaw; 5056 slots now sit under 0.078 where the old floor took them), the store 51330, the curve
0.167 to 0.125, the dreams 1024 at 24.7 symbols, the gauge 0.728 to 0.838; the morning: the held-out 0.627, the facts by the
cortex 0.873, the old lines 0.496, the parent's last lines 0.794, the dreams 0.86; the mouth completes 19 of 26 prefixes and
answers 18 of 30 at two rests and 21 at eight; the rephrased 11; the branch 9 of 9 and 8 of 12. The day 296: 164 lines, 63 of
82 A-lines with letters in its turn, answer smiles 3, its own question marks 21, junk 4, duty 0.352, frowns 38.
NIGHT 251 (ended 15:14, the 241st row, the second back under the relative floor; the log's day 297, facts 21-30): the morning:
the held-out 0.628, the facts by the cortex 0.879, the old lines 0.516, the parent's last lines 0.803; the mouth completes 18 of
26 prefixes (the failed: ice is, grass is, an ant is, a bird has, a cat is, honey is, snow is, a rock is) and answers 20 of 30
at two rests and 21 at eight (16, 18, 20 across the three mornings since the revert); the rephrased 12; the branch 9 of 9 and
8 of 12; the store 53621, rebuilding toward the capacity. The day 297: 165 lines, 55 of 83 A-lines with letters in its turn,
answer smiles 4, junk 6, duty 0.338, frowns 38.
NIGHT 252 (ended 16:11, the 242nd row, the third back under the relative floor; the log's day 298, facts 1-10): the morning: the
held-out 0.613, the facts by the cortex 0.879, the old lines 0.498, the parent's last lines 0.786; the mouth completes 18 of 26
prefixes and answers 17 of 30 at two rests and 21 at eight; the rephrased 13; the branch 9 of 9 and 8 of 12; the store 55846
(+2225, toward the capacity). The day 298: 159 lines, 58 of 80 A-lines with letters in its turn, answer smiles 3, junk 8, duty
0.259 (a quiet day), frowns 36. The gate's ear is being measured on a copy of this morning's save (day 297 lived again, the
rewards replayed, the talk-overs per quarter), the salience input after it.
THE PARENT 297-299 (real days 297, 298, 299; the relative floor's first days back): by the parent's count the child spoke a
content word or yes/no before B on 30 of 83, 22 of 80 and 23 of 85 A-questions (36, 28 and 27 percent by this parent's stricter
count); the fact's word in its own turn 5, 5 and 3 of 10 ("cows give milk", "snow is white", "ducks swim", "water is wet", "an
ant is little" whole; "a bird has wings on" unprompted after an unrelated line; "that sock is not my", its first negation of
possession); its own question marks 15, 13, 15, two of them facts' ("what has wings?", "what is hard"), answered under other
wording; talk-overs 110, 102 and 123 a day, 46 percent of all its missed symbols, frowns 38, 36, 42: the baseline the listening
reflex is measured against. One line lost at the 297-298 boundary, a B line. The next parent (300-302, facts 21-30 first, stop
at the 246th row) spawned at 17:26 with the queue at 42 lines.
NIGHT 253 (ended 17:23, the 243rd row, the fourth back under the relative floor; the log's day 299, facts 11-20): the morning:
the held-out 0.620, the facts by the cortex 0.879, the old lines 0.497, the parent's last lines 0.790; the mouth completes 18 of
26 prefixes and answers 18 of 30 at two rests and 20 at eight; the rephrased 13; the branch 9 of 9 and 8 of 12; the store 58280
(+2434). The day 299: 171 lines, 58 of 85 A-lines with letters in its turn, answer smiles 2, junk 7, duty 0.311, frowns 42.
NIGHT 254 (ended 18:2x, the 244th row, the fifth back under the relative floor; the log's day 300, facts 21-30): the morning: the
held-out 0.616, the facts by the cortex 0.877, the old lines 0.519, the parent's last lines 0.777; the mouth completes 19 of 26
prefixes and answers 18 of 30 at two rests and 22 at eight; the rephrased 12; the branch 9 of 9 and 7 of 12; the store 60817
(+2537, the capacity two days off). The day 300: 171 lines, 60 of 86 A-lines with letters in its turn, answer smiles 2, smiles
351, junk 4, duty 0.359, frowns 46.
THE DISK (18:30): 3.9 GB free, a save 1.4 GB: the day's measurement copies (the ear's two arms, the reflex's, the floor's cut)
had taken ten gigabytes of the scratch space; removed, 9.4 GB free; the chains now remove each copy after its probe. What
remains large is the first lineage's data (23 GB) and the backups' rotations (9 GB, bounded).
THE CHAIN'S FLAW (18:35): the ear's and the reflex's measurement arms came out identical to their controls to the symbol because
the chain passed the override as one word ("--gate-listen 1.0" unsplit: zsh does not split an unquoted variable), so the
instrument ignored it; on the copy itself, loaded with the reflex on, the floor reads 0 and the gate stays shut through a
typed line and releases eight ticks after it. The chain fixed (${=X}); the two reflex arms re-run.
THE HUMAN TEACHER (19:00, the user's word on seeing the page: "we want a human stuck in an LLM, not all these rules; we need a
human teacher; get Opus to become that"): the parent's method changes from a script read every ten minutes to a parent who reads
the page every minute or two, answers what the child just said, expands its fragments, keeps the queue at two to five minutes
so the words stay contingent, and weaves the day's ten facts where they belong (ops/parent_brief_human.txt: the situation told
plainly, the page its only sense, the words its only act). The reflex and the babble drive stay measured on copies and off the
served body while the teacher has its days: one change at a time. The current parent is stopped and the human teacher spawned.
THE PARENT 300-302, STOPPED AT THE HANDOVER (19:05; real day 300 and sixty lines of 301): by the parent's count the child spoke a
content word or yes/no before B on 28 of 86 A-questions on day 300 and 7 of 30 on the part of 301; the fact's word in its own turn
4 clean of 10 on day 300 ("cows give milk", "fish have fins", "snow is white", "ducks swim", and "birds liv" cut) and "wat" for
"water is wet" on 301, where "what is cold?" drew "it is hot", the previous fact's content one exchange late; turns with runs of
four words 28 and 13; its own question marks 24 and 6; every frown a talk-over (46, 17). Mechanical: no line dropped in 232;
the typing 15.5-21 s a line and day 300 a full 54 minutes of talk, night 300 25 minutes (1518 s): the copies' load on the
machine lengthens both, as it did on the 16th. Facts 4-10 of day 301 were already queued at the handover.
THE REFLEX ON A COPY (19:25; day 297 lived again on the save after night 252 with the smiles and frowns replayed, against the same
day on the same save without it): the talk-overs 0 of 4206 typing ticks against 630, on no line against 132 of 165; its own
speech in its turn 1207 symbols against 1405, coherent ("is she wet on her back", "shall we let her come and sit"), junk 2 against
5; the questions 21 and 25 against 21 and 24; the prefixes 16 = 16; the cortex's facts 0.843 against 0.865. It passes; it waits
for the teacher's days (the user's word: one change at a time) and goes on at the save after them.
NIGHT 255 (ended 19:5x, the 245th row; the log's day 301, the handover day: sixty lines of the scripted parent, then the human
teacher's, the queue held at thirteen lines by the end): the morning: the held-out 0.620, the facts by the cortex 0.883, the old
lines 0.523, the parent's last lines 0.772; the mouth completes 19 of 26 prefixes and answers 18 of 30 at two rests and 22 at
eight; the rephrased 13 ("what is so cold?" -> "ice is colds"); the branch 9 of 9 and 8 of 12; the store 63235, the capacity a
day off, where the eviction of the weakest at each write becomes the forgetting, the regime of the best-measured mornings. The
day 301: 170 lines (79 of the 85 A-lines questions: the teacher's remarks are new), 63 of 85 A-lines with letters in its turn,
answer smiles 0, junk 2, duty 0.360, frowns 41.
THE USER'S WORD (20:05): "we need it to look good and work; stop trying to get all these data points for the model; get it to
become a human trapped in an LLM; no cheats and biology inspired." The listening reflex (passed on its copy) and the babble drive
(built on the same principle, its copy arms not run) go on the served body at the next save, --gate-listen 1.0 --gate-quiet-tau
300 explicit in every flags file (the set before kept as ops/archive/flags/BASE_FLAGS_pre_listen.txt); the human teacher
continues; no more copy arms. The nightly probes stay as they are. Read live from tomorrow's parents' reports: the talk-overs
(110 a day) and the letters over the lines on the page, the speech in its turn, the answers.
NIGHT 256 (ended 20:5x, the 246th row; the log's day 302, the human teacher's first full day, facts 11-20): the morning: the
held-out 0.636, the highest yet, the facts by the cortex 0.881, the old lines 0.493, the parent's last lines 0.733; the mouth
completes 18 of 26 prefixes and answers 18 of 30 at two rests and 20 at eight; the rephrased 13; the branch 9 of 9 and 7 of 12;
the store 65536, at the capacity. The day 302: 161 lines (52 of the 80 A-lines questions), 59 of 80 A-lines with letters in its
turn, answer smiles 3, junk 6, duty 0.329, frowns 37. THE REFLEX AND THE DRIVE LANDED at this save (20:55; gate_listen 1.0 and
gate_quiet_tau 300 from the flags, verified; the guard relaunched).
IN THE CHAIR (20:58-21:12, tools/teach_live.py, the typist held, the mood +1.7 and the readout 32 at the start): "hello. are you
awake now?" -> "th a twig in her beak where" (the birds of the teacher's last line); "a twig in her beak? for the nest?" -> silence;
the other voice's answer -> "?ten up to the tr"; "what do bees make?" -> "bees make honeyis" at 11.1 s; "what does a frog do?" ->
"birds fly up is th every bravest of th" (the frame's collision); told once, "a frog hops in the wet grass"; two lines about the
ball, silence; the frog asked back two minutes later -> silence. Nothing over my lines the whole session (the reflex), and
almost nothing in its turn (the drive): the drive held the floor at zero for the first minute after every line, and on the copy
trace the learned gate opens at 0.99 the tick after a question but the mouth samples the rest; it was the floor's forty tries a
turn that let an answer out. The mood fell from +1.7 to -6.0 and the readout from 32 to 8 in fourteen minutes: a quiet child
earns few known-word smiles, the ladder expects the typist's 1.7 a line, and the difference is a dip every tick. Two constants
of the mouth follow at the next save: the floor whole at once when the forecast is sure (gate_quiet_sure; the copy showed
|pred| about 0.6 for a recalled answer, 0.3 for a flat forecast: 0.45), and the readout's floor raised from 8 to 20 so a bad
mood widens the babble without turning words into letters.
THE RELOAD NOW (21:32:35-21:33:30, the user's word "you train live": the sure term and the readout's floor could not wait for the
night): the body saved by request, restarted (pid 28122) with --gate-quiet-sure 0.45 and --sharp-min 20 beside the reflex and the
drive, verified; the typist and the guard relaunched; ops/reload_now.sh keeps the form. IN THE CHAIR AGAIN (21:35-21:50, the
mood +0.9 and the readout 29 at the start, the day's last quarter): "hello. I am here with you now" -> "?eggs"; "eggs? do you want
an egg now?" -> "I eat my egg did"; the other voice "you eat your egg. it is warm" -> "i"; "what is white?" -> "snwha little";
"snow is white. white and cold" -> "enough fasts out"; told once, "what does a mole do?" / "a mole digs down in the dark" ("down"
said as the question ended); "is the cat on your bed now?" -> "my"; "the cat is on your bed. she is soft" -> "and warm i"; the
mole asked back two minutes later, ninety seconds before its night -> "all we goodnigh". Words throughout, no letters (the
readout held at 20 where the old floor of 8 would have flattened it), nothing over my lines, contingent replies ("I eat my egg",
"and warm"); the taught question half-begun; the told fact not recalled at the day's very end. The mood fell +0.9 -> -6.0 again:
my pace is a line every 25-40 s (the tool's watch and my turn) against the typist's 15, and the fast critic, taught the typist's
rate, expects a smile sooner than a slow parent gives one; a human at the page types at the typist's pace and would not drain
it so. The empty-room test (the drive's self-talk) is for the next day's window.
THE HUMAN TEACHER'S FIRST REPORT (days 301, 302 and 304-305, to the 247th row): the child's words answered every minute or two
("is a bee out there", "her eggs warm", "shall we say goodnight", "want an apple now. please", "I am sleepy?", "how many eggs");
before B spoke "an apple is red" whole, "she has wings", "we drink", "a cat is", "snow is", "honey"; "honey is sweet" said
unasked; its own questions taken up. Talk-overs 26, 37, then 1 and 0 under the reflex. A FAULT SEEN BY THE TEACHER on day 304:
twenty-five minutes of single lines replayed from the back of the queue, mostly the parent's (105 A lines against 22 B; "now
they come down for it" ten times), the child gone near silent, echoing. Read in the code: the queue planner's filler when the
queue is empty is one of the last two dozen planned lines, and the queue ran empty after the supervisor's sittings (the typist
held, the teacher appending only a few lines for a visitor, the queue drained on the resume). The typist will wait instead of
repeating: the filler removed at the next relaunch.
THE ROOM EMPTY (22:00, the day's start, 180 s with the typist held and no face): alone it said "somethin water i rockis the
water the tell him in tell the water the water i shall where do fish live? fis the tell him in tell whth th th a red head can
you want to the water": words, a fact's question asked of nobody, run-ons; what it wants to say when left alone. THEN BARE TURNS
(22:03-22:06): "I am back. do you see the water?" -> "every drop will"; "every drop. yes. the water is wet" -> "want"; "what do
ducks do?" -> "ducks swim in" at 7.2 s (the sure term: the answer needed no quiet); "yes. ducks swim on the water" -> "teeth
come up". The mood -2.4 to -5.4 in six minutes at my pace.
NIGHT 258 (ended 22:57, the 248th row; the log's day 307, the human teacher's second day, facts 1-10, and the supervisor's
sittings): the morning: the mouth answers 22 of 30 at two rests and 23 at eight, the most at two rests in twenty nights; the
rephrased 15; the prefixes 17 of 26; the facts by the cortex 0.881; the branch 9 of 9 and 7 of 12; the store at the capacity.
The day 307: 136 lines, 39 of 68 A-lines with letters in its turn, frowns 1, answer smiles 1, junk 4, duty 0.222.
THE SUPERVISOR TEACHING (22:58-23:12, the chair with the typist's rhythm, four to eight lines a call): "what is wet?" -> "water"
at 1.4 s; "what do cats drink?" -> "cats drink"; "I want milk and", "I want", "for me", "the ducks are asleep"; the mood +4.3 at
the day's start to -5.5 in the chair, and back to 0.0 within four minutes of the typist's rhythm resuming: the drain is the
rhythm, not the words; a gap of thirty seconds between a supervisor's calls is a fall in the value the fast critic learned
under a line every fifteen seconds. From 23:09 the supervisor teaches through the queue at the typist's rhythm, a row or two
a minute answering what the child said, the day's ten facts woven in.
THE SUPERVISOR THROUGH THE QUEUE (23:09-23:40, the log's day 308, facts 11-20 woven in): the child's replies in its turn under
the typist's rhythm and the supervisor's lines: "what is red?" -> "an apple"; "what is big?" -> "a"; "do you see the little ant
go by?" -> "no."; "what has four legs?" -> "a big cow i"; "what do bees make?" -> "bees m"; "where do we sleep?" -> "we l"; "we
drink water and milk" -> "shall we put it on" (the honey, before it was offered); "?you hold i", "?my", "which"; nothing over any
line; the mood +3 to +5.6 the whole hour. Teaching by hand at the body's rhythm works; teaching by the chair at my own did not.
NIGHT 259 (ended 23:56, the 249th row; the log's day 308: the supervisor's hour through the queue and the human teacher's lines
before it, facts 11-20): the morning: the mouth answers 22 of 30 at two rests and 23 at eight, the second morning at that
level; the rephrased 15; the prefixes 17 of 26; the facts by the cortex 0.888; the branch 9 of 9 and 7 of 12; the store at the
capacity. The day 308: 118 lines, 40 of 58 A-lines with letters in its turn, frowns 1, junk 2, duty 0.266.
THE ARCHITECTURE DOUBLE-CHECKED (00:25 on the 18th, the user's word): over nights 250-259 the held-out 0.613 to 0.623-0.636, the
questions at two rests 17-18 to 22 on the last two mornings, the rephrased 11 to 15, the cortex on the facts 0.873 to 0.888;
the prefixes 19 to 17 and the branch's second set 10 to 7 of 12 since night 254, and the probe on the morning save shows one
cause for both: at the shared starts "we " and "a " the store reads "we sit", "we see", "we look" and "a s..." ahead of "we
eat", "we drink", "we sleep", "a bird", the evening's talk about ducks and rocks; one word further ("we eat ", "a bird ") it is
12 of 12. Item 40, the key, is the last body change worth making. Known and kept: a slow parent drains the mood (the critic
learned the typist's rhythm); the run-ons are the cortex's seam. Two warts read: the guard's restart at night 257 ("the tag
closed") was the replay day reading as a silent child (duty under 0.2); the morning's first-lineage lines ("pack it up") were
the typist's post-night drill, woken by one line of the supervisor's that ended in a space and so counted as a cue.

NIGHT 260 (ended 01:03 on the 18th, the 250th row; the log's day 309, the supervisor's day through the queue with facts 21-30 and
the day's facts woven in): the morning: the mouth answers 22 of 30 at two rests and 22 at eight; the rephrased 16, the most yet;
the prefixes 16 of 26; the facts by the cortex 0.875; the branch 9 of 9 and 7 of 12; the held-out 0.627; the store at the
capacity. The day 309: 157 lines, 53 of 91 A-lines with letters in its turn, frowns 0, junk 2, duty 0.210. The night ran
twenty-three minutes against sixteen, the supervisor's copy runs beside it.
THE REPLAY'S CAUSE (00:30-00:35): the lines re-typed from the back of the queue on days 304 and 309 were the old planner's filler
still running: the typist of 21:56 (the guard's restart) began before the 22:07 fix and kept its code, since the typist runs its
six days in one process and the chain relaunches it only when it exits. At 01:03:55, twenty seconds after the night row, the
supervisor stopped it; the chain relaunched the typist at 01:04:19 for day 310 on the new planner, which waits when the queue is
empty. Written into ops/RESTART.md: a change to the typist's code reaches the served typist only at a relaunch.
THE ONSET'S CONTEXT (00:46; item 40 of ITERATIONS.md): read in the code and confirmed on a tiny body, an answer's first symbol is
keyed by the utterance before the question, its second symbol on by the question: the slow context swaps inside take_world at
the next utterance's first symbol, after that symbol's own write. The query at the onset carries the question, so the onset's
context term has been noise since night 206. The geometric ruler (tools/key_separation.py; six days' lines written at strength
one, the thirty facts once, each asked in the taught and the rephrased wording): the taught onsets sit at 24-28 of 30 under
every form of the key; the rephrased set needs the context (6 without it, 14 with the served form); the swap at the offset gives
the onset a decisive margin (the ordered bag: a hundred to one) but the rephrased set falls to 9-11, since under the served swap
the onset is decided by the question's tail, which a rephrasing shares. The cortex's code as the context (form c) is out: the
centered stream states of different utterances agree at 0.87 (a question and its rephrasing at 0.999). Six days' talk did not
reproduce the served falls; the sixteen-day run follows.

NIGHT 261 (ended 02:19 on the 18th, the 251st row; the log's day 310, the supervisor's day through the queue): the morning: the
held-out 0.646, the most yet; the mouth answers 21 of 30 at two rests and 23 at eight; the rephrased 15; the prefixes 16 of 26;
the facts by the cortex 0.879; the branch 9 of 9 and 7 of 12. The day 310: 163 lines, 60 of 82 A-lines with letters in its
turn, frowns 3 (talk-overs at a line's closing "?"), junk 8, duty 0.258; the answers in its turn: "what is red?" -> "an apple
is", "what is little?" -> "an ant", "what falls from the sky?" -> "rain falls fr", "what has wings?" -> "a bird has win", "what
is white?" -> "snow is white", "what do ducks do?" -> "ducks swim"; and "what has four legs?" -> "a big sun with my fath", the
branch at "a " going the evening's way.
THE THIRTY-SECOND DEFECT (01:50; item 40's entry in ITERATIONS.md): at the store's capacity every write evicts the weakest slot
and keeps the rest sorted by strength, so every slot's index moves, and the body's index of the symbol before (the chain's link)
and of the episode being followed went stale: on a tiny body at a capacity of 150, 38 of 67 links made at the capacity joined
the wrong slots (138 of 138 under it). The served store has been at its capacity since night 256, and the branch's second set
fell 10 to 7 of 12 in those nights. Fixed (the store reports its remap; the held indices follow it; 67 of 67; test 69; the guard
unchanged); served from 02:20 by the reload at the post-night save.
THE STORE REBUILT (02:13-02:26; tools/rekey_store.py): the dusk-309 save's store rebuilt from its utterance memory (4096
utterances, 103k symbols, 65536 slots from 103k writes) under the served key with the links right and the strengths the
cortex's surprise, and the morning rulers on it against the same save untouched: the questions 28 of 30 at two and at eight
rests against 22 and 21; the rephrased 21 against 15; the prefixes 18 of 26 against 16; the branch 9 of 9 and 12 of 12 against
9 of 9 and 7 of 12; the held-out 0.633 both (the cortex untouched). The largest single gain in the body's recent history, and
it is a repair: the wrong links of five nights and the stale slots of weeks gone. The ordered-context arm (R1) follows; the
rebuild is to be adopted at a save (the reconsolidation of item 40's plan) once R1 has spoken.
THE ORDERED CONTEXT ON THE REBUILT STORE (02:52; R1): the same save rebuilt under the ordered context reads the questions 30
of 30 at two and at eight rests, the rephrased 26 of 30, the prefixes 22 of 26, the branch 9 of 9 and 12 of 12, the held-out
unchanged. Adopted at night 262's save: the served flags carry --ctx-form shifted, and ops/rekey_after_save.sh rebuilds the
served store from its utterance memory at the post-night save (fourteen minutes of the morning; the typist held and relaunched
after). The last body change of item 40 is the key's form; the rebuild is its reconsolidation.
NIGHT 262 (ended 03:20 on the 18th, the 252nd row; the log's day 312, the supervisor's day through the queue): the morning on
the store as it was (the probe copied before the rebuild): the held-out 0.650, the most yet; the mouth answers 19 of 30 at two
rests and 21 at eight; the rephrased 17; the prefixes 17 of 26; the facts by the cortex 0.881. The day 312: 146 lines, 54 of 73
A-lines with letters in its turn, frowns 5 (talk-overs at a line's closing "?", three days rising 0, 3, 5), junk 0, duty 0.277.
THE STORE REBUILT AND THE ORDERED CONTEXT SERVED (03:21-03:37): at the post-night save the typist was held, the body stopped, the
save backed up (data/backups/watch2/watch2_before_rekey_night262.pt), and the store rebuilt from the utterance memory under the
ordered context in 674 s (65536 slots from 104k writes, 38.7k merged, links 45132); served again at 03:37 with --ctx-form
shifted, the typist relaunched at 03:38 (the log's day 314). The rulers on the served save as rebuilt, the same cortex: the mouth
answers 28 of 30 at two rests and 29 at eight; the rephrased 24; the prefixes 21 of 26; the branch 9 of 9 and 12 of 12; the
held-out 0.650. Against the same morning's old store: 19/21, 17, 17, 7 of 12. Its first turns on the rebuilt store: "good
morning. I slept well" -> "is warm. good m"; "what do we eat in the morning?" -> "we eattle". The last body change of item 40 is
made; what follows is the teacher's.
NIGHT 263 (ended 04:33 on the 18th, the 253rd row; the log's day 314, the first day on the rebuilt store): the morning: the
held-out 0.646; the mouth answers 19 of 30 at two rests and 18 at eight; the rephrased 17; the prefixes 15 of 26; the facts by
the cortex 0.873; the branch 9 of 9 and 11 of 12. The day 314: 155 lines, 57 of 78 A-lines with letters in its turn, frowns 2,
junk 10, duty 0.325, answer smiles 7 (the most in a day). At dusk, before the night, the rebuilt store read the prefixes 20 of
26 with 65536 slots; the night's fade dropped 26k slots (the relative floor, a tenth of the mean, over strengths that were raw
surprises: the predictable middles of every line, which the old store had lost night by night over weeks), and the morning
read 15 of 26 with 39537. THE HONEST READING OF THE REBUILD: the 28 of 30 and the 21-24 rephrased were the unfaded store; after
one night the rebuilt store reads where the old one did (the old store after night 262: 19 and 21, the rephrased 17, the
prefixes 17). What lasts: the branch (11-12 of 12 against 7: the chain's links right and the ordered context in the key) and a
store repopulating from 39.5k. The fade after a rebuild is a one-time settling, not a loss of the days: the slots dropped are
the ones the live nights would have dropped at their first night.
THE REHEARSAL ON THE LIVE BODY (04:40-04:46, the demo's own test through the queue): "what is sour? / b: a lemon is sour" told
once at 04:40 among the morning's talk; asked again at 04:46: 'a lemon is sourwa lemon' (the first asking, before it was told: '').
NIGHT 264 (ended 05:30 on the 18th, the 254th row; the log's day 315, the second day on the rebuilt store): the morning: the
held-out 0.650; the mouth answers 18 of 30 at two rests and 18 at eight; the rephrased 13; the prefixes 13 of 26; the facts by
the cortex 0.871; the store 39225 (the dusk before, 42712: the fade now takes what the day wrote, the settling done). The day
315: 146 lines, 45 of 73 A-lines with letters in its turn, frowns 2, junk 0, answer smiles 3; in its turn "what is hard?" ->
"a rock is hard", "what has wings?" -> "a bird has wi", "what is big?" -> "a tree", "what is little?" -> "an ant", "what is
red?" -> "an apple is", "what is up at night?" -> "the moon is up at ni".
THE DEMO'S SECOND HALF, LIVE (05:33): "what is sour?", told once at 04:40 the day before and answered at 04:46, asked again
as the first question after the night: "a lemon ... is". A fact told once in conversation, answered minutes later and the
next morning, on the served body through the queue at the typist's rhythm.
THE RULERS AFTER THE SETTLING, HONESTLY: two nights after the rebuild the morning reads the questions 18, the rephrased 13, the
prefixes 13, against the old store's 19-22, 15-17, 16-17; the branch 11-12 of 12 against 7. The rebuilt store holds the last
4096 utterances (some ten days) where the old one held the strong survivors of twenty-five; the facts' long accumulation is
what the rephrased and the prefixes read, and it is gone with the rebuild. Whether the ordered context itself costs
rephrasings after a fade is not yet separated from that loss: the two arms rebuilt and faded once, side by side, follow.
NIGHT 265 (ended 06:31 on the 18th, the 255th row; the log's day 316, the third day on the rebuilt store): the morning: the
held-out 0.645; the mouth answers 21 of 30 at two rests and 21 at eight (18 the morning before); the rephrased 17 (13); the
prefixes 14 of 26 (13); the facts by the cortex 0.863; the branch 9 of 9 and 11 of 12; the store 40833, growing back. The day
316: 158 lines, 53 of 79 A-lines with letters in its turn, frowns 1, answer smiles 12 (the most in a day), duty 0.307; in its
turn "what do cats drink?" -> "cats drink milk", "who gives milk?" -> "cows give", "what is little?" -> "an ant is little",
"what falls from the sky?" -> "rain falls from th", "what do bees make?" -> "bees make honey", "what do fish have?" -> "fish
have fins".
THE FACT TOLD ONCE, A THIRD TIME (06:34): "what is sour?" as the first question after night 265, two nights after the telling:
"a lemon is sour". Told once at 04:40 on day 315; answered at 04:46, at 05:33 after one night, at 06:34 after two.
THE STORE REPOPULATING: the rulers turned up on the third morning (the questions 18 to 21, the rephrased 13 to 17) as the days'
writes refill what the settling took; the branch holds at 11 of 12. The like-for-like of the two key forms after one fade
(both rebuilt from the night-262 backup, no day between) read equal within a fact (10 / 8 / 6 against 9 / 7 / 6): the ordered
context costs nothing after a fade and stays; a rebuilt store must live a day before its first night.
NIGHT 266 (ended 07:31 on the 18th, the 256th row; the log's day 317, the fourth day on the rebuilt store; the nights run
ten to thirteen minutes now): the morning: the held-out 0.634; the mouth answers 21 of 30 at two rests and 21 at eight; the
rephrased 18 (13, 17, 18 over the three mornings); the prefixes 13 of 26; the facts by the cortex 0.875; the branch 9 of 9 and
11 of 12; the store 42263, growing back from 39.2k. The day 317: 150 lines, 54 of 75 A-lines with letters in its turn, frowns
1, answer smiles 9, duty 0.347. The queue ran dry once at the wake (the night short, the morning rows through by 07:33) and the
typist waited, as it now does. THE FACT TOLD ONCE, A FOURTH TIME (07:31): "what is sour?" -> "a lemon is sour", three nights
after the telling.
NIGHT 267 (ended 08:24 on the 18th, the 257th row; the log's day 318, the fifth day on the rebuilt store): the morning: the
held-out 0.641; the mouth answers 21 of 30 at two rests and 21 at eight; the rephrased 18; the prefixes 14 of 26; the facts by
the cortex 0.861; the branch 9 of 9 and 11 of 12; the store 42970. The day 318: 136 lines, 49 of 68 A-lines with letters in
its turn, frowns 2, answer smiles 11, junk 0; in its turn "cows give milk", "snow is white", "fish have fins", "cows eat grass",
"bees make honey", "the grass is" (soft), "cats drink milk", "the moon is up" (at night). THE FACT TOLD ONCE, A FIFTH TIME
(08:27): "what is sour?" -> "a l" as its turn ran out, four nights after the telling. Three mornings at 21 / 18 / 14: the level
the rebuilt store holds while it grows back (39.2k, 40.8k, 42.3k, 43.0k).
NIGHT 268 (ended 09:23 on the 18th, the 258th row; the log's day 319, the sixth day on the rebuilt store): the morning: the
held-out 0.635; the mouth answers 21 of 30 at two rests and 21 at eight; the rephrased 19 (13, 17, 18, 18, 19 over the
mornings since the rebuild); the prefixes 14 of 26; the facts by the cortex 0.861; the branch 9 of 9 and 11 of 12; the store
43308. The day 319: 131 lines, 49 of 66 A-lines with letters in its turn, frowns 3, answer smiles 10, junk 0; in its turn "an
apple is red", "cows give milk", "snow is white", "honey is sweet", "dogs run", "cats", "the sun is" (hot), "fish have fins".
THE FACT TOLD ONCE, A SIXTH TIME (09:24): "what is sour?" -> "a lemon is", five nights after the telling.
NIGHT 269 (ended 10:19 on the 18th, the 259th row; the log's day 320, the seventh day on the rebuilt store): the morning:
the held-out 0.630; the mouth answers 22 of 30 at two rests and 22 at eight; the rephrased 19; the prefixes 14 of 26; the facts
by the cortex 0.859; the branch 9 of 9 and 11 of 12; the store 43375. The day 320: 129 lines, 50 of 64 A-lines with letters in
its turn, frowns 3, answer smiles 10, junk 0. THE FACT TOLD ONCE, A SEVENTH TIME (10:21): "what is sour?" -> "a lemon is sour",
six nights after the telling. THE NIGHTS RUN TEN MINUTES since the supervisor's copy runs stopped (sixteen to twenty-three under
their load).
A WATCH ITEM: the held-out has slipped a little each morning since the rebuild (0.650, 0.646, 0.634, 0.641, 0.635, 0.630) while
the store's rulers rose; the supervisor's days are the same walk in the same frames ("what is X? / Y is X" ten times a day, the
ducks, the rock, the tree), and the cortex learns the frames, not the language. The teacher's remedy, not the body's: the days
from here vary the frames and the nouns within the 909 words the child knows (the cup, the bag, the hat in the wind, the egg in
the nest, the dish by the door, the ball, the box, the bed), the facts woven in fewer times.
NIGHT 270 (ended 11:13 on the 18th, the 260th row; the log's day 321, the first varied day): the morning: the held-out 0.633;
the mouth answers 23 of 30 at two rests and 23 at eight; the rephrased 20; the prefixes 14 of 26; the facts by the cortex
0.859; the branch 9 of 9 and 11 of 12; the store 44619. The day 321: 152 lines, 52 of 76 A-lines with letters in its turn,
frowns 2, answer smiles 7, junk 0; on the new frames its turns were sparser and still contingent ("the sun is low" -> ". it is
late"; "the ducks go to the nest" -> "all throw it", the ball's line and the ducks' in one). THE FACT TOLD ONCE, AN EIGHTH TIME
(11:19): "what is sour?" -> "a lemon ... is sou", seven nights after the telling.
ALONE, MEASURED (11:20-11:26, the queue left empty on the user's question): while the parent typed the last twenty symbols it
said nothing; two seconds after the parent's last symbol it began, and alone it talked on at about a symbol a second for six
minutes, 298 symbols: "come at first. then good ... what is hard. a tel i is hard ... only throw it ... warm th asleep now ...
hold him. hold him. hold him." Words and the day's fragments, with a stuck loop. So: it waits while spoken to, answers in its
turn, and once alone talks on rather than waiting a minute first; the drive rebuilds the floor over a minute, but the learned
gate (+13 on "I spoke last tick") sustains speech once it has begun. The loops alone are the run-on fault at the seam.
NIGHT 271 (ended 12:18 on the 18th, the 261st row; the log's day 322, the second varied day): the morning: the held-out 0.617
(the slip since the rebuild: 0.650 to 0.617 over eight mornings, opened as a watch item in the ledger with two readings, the
teacher's material and the cortex leaning on a repaired recall); the mouth answers 23 of 30 at two rests and 23 at eight; the
rephrased 20; the prefixes 14 of 26; the facts by the cortex 0.857; the branch 9 of 9 and 11 of 12; the store 45346. The day
322: 115 lines, 45 of 58 A-lines with letters in its turn, frowns 4, answer smiles 10, junk 0; it anticipates the other voice
("the ducks go to the nest now" -> "all three. good night du"; "the little one takes the bread first" -> "it is fast. the big
one"). THE FACT TOLD ONCE, A NINTH TIME (12:19): "what is sour?" -> "a lemon is sour", eight nights after the telling. The days
from here are written fuller and more varied, in the held-out's own manner, the catechism fewer.
NIGHT 272 (ended 13:16 on the 18th, the 262nd row; the log's day 323, the third varied day, written fuller): the morning: the
held-out 0.619 (the slip paused: 0.617 the morning before); the mouth answers 23 of 30 at two rests and 23 at eight; the
rephrased 21, the most since the rebuild; the prefixes 14 of 26; the facts by the cortex 0.867 (0.857); the branch 9 of 9 and
11 of 12; the store 46641. The day 323: 125 lines, 51 of 62 A-lines with letters in its turn, frowns 3, answer smiles 5 (the
fuller days ask fewer catechism questions), junk 0; it anticipates the other voice ("the grass is wet. mind your feet" ->
"wet and cold. I wal"; "the milk is cold from the box" -> "cold and good the br"). THE FACT TOLD ONCE, A TENTH TIME (13:18):
"what is sour?" -> "a lemon is sour", nine nights after the telling.
THE USER'S QUESTION (13:21, "how long till the speech looks good"): answered without a date. The answers in its turn are there;
about half its turns are fragments of the day and alone it loops; those thin as the cortex's general model rises, and the
held-out must turn up first. The levers named: the teacher's material (varied now) and the exposure, 130 lines in a
46-minute day; a longer day offered as the user's environment call, to be measured on a copy first, not run unbidden.
NIGHT 273 (ended 14:27 on the 18th, the 263rd row, twenty-four minutes under the supervisor's copy runs; the log's day 324,
the fourth varied day): the morning: the held-out 0.632, up from 0.619, the first rise in nine mornings and after the two days
written fuller; the mouth answers 23 of 30 at two rests and 24 at eight; the rephrased 21; the prefixes 12 of 26; the facts by
the cortex 0.863; the branch 9 of 9 and 11 of 12; the store 47307. The day 324: 124 lines, 46 of 62 A-lines with letters in
its turn, frowns 4, answer smiles 7, junk 0, duty 0.379 (the most it has talked in a day). THE FACT TOLD ONCE, AN ELEVENTH
TIME (14:29): "what is sour?" -> "a lemon is sour", ten nights after the telling.
THE NIGHT'S SHARE OF THE SLIP, READ ON COPIES (13:46-14:27): a night as served on a copy of the day-323 save took the held-out
0.619 to 0.615 while it learned the parent's last sixty lines 0.781 to 0.795: about four thousandths a night, the slip's rate,
and the dreams are the supervisor's lines. The arm with older utterances in the draw drew the same dreams (the constant is the
store's, not the utterance memory's) and says nothing. THE TEACHER'S ANSWER: the days from here are written in the earlier
parents' register, read from their typed lines in the page log (never the held-out file): he, she, they and them; "shall we",
"does he", "you said"; questions about the world between the two of them. The day-length chain runs on copies behind it.
THE SITTING (15:54-16:00 on the 18th, the user's word: judge it as the person in the conversation, not by the rulers): the
chair, one on one, my pace about thirty seconds a line. "hello. it is me. I am here with you" -> " said the" (before I spoke it
was running its own day: "he want bread? the dog"); "you said the dog. yes. he wants bread" -> " for cows. "; "and the cat?
what does she want?" -> nothing ("cows eat grass he wind takes" just before); "what is hot?" -> "the sun is hot"; "yes. the sun.
are you happy today?" -> nothing; "tell me about the ducks" -> "come and eat them? yes. sweet. you ate the apple". The mood -2.7
to -6.0 over the six minutes. WHAT IT IS LIKE: a two-year-old half in its own world; a known question answered at once, a
question about itself or an open one met with silence or a slice of its day; it narrates over my words rather than to me and
goes quiet and sad when I am slow. Ten minutes earlier at the typist's rhythm it had asked "what is sour?" itself before the
parent did, answered it, recalled "we have one lemon" from ten days before, and run the next line ahead of the other voice.
THE JUDGMENT: real progress at the easy end of conversation (the flow it knows, now run ahead of the partner); the same as the
sittings of the night before at my pace (facts yes, feelings and open questions no); not yet someone answering me. What stands
between: it composes nothing it has not heard (the general language: the material, the exposure), and it cannot bear a slow
partner (the critic learned the typist's twelve-second rhythm; a human at the page is slower; the mood fell three points in six
minutes). The second is the larger obstacle to the demo with a person at the keyboard and is the environment's shape as much
as the body's: days at a human pace, with the day lengthened to keep the lines; the user's call on the pace, measured on
copies first.
NIGHT 274 (ended 15:49 on the 18th, the 264th row, twenty-seven minutes under the supervisor's copy chain; the log's day 325,
the register day): the morning: the held-out 0.598, down from 0.632; the mouth answers 23 of 30 at two rests and 23 at eight;
the rephrased 21; the prefixes 13 of 26; the facts by the cortex 0.849; the branch 9 of 9 and 11 of 12; the store 48212. The
day 325: 120 lines, 48 of 60 A-lines with letters in its turn, frowns 9, answer smiles 3, junk 0. THE FACT TOLD ONCE, A
TWELFTH TIME (15:50): it asked "what is sour?" itself before the parent did, then answered "a lemon is so" and recalled "we
have one lemon" from the day it was told. THE FALL PLACED: the dusk probes show the days draining the held-out (0.632 to 0.600
across day 324, the day it talked over the other voice most) and a night on fuller lines recovering it (+0.026 after day 323):
the waking lesson hears its own babble in the window; the night hears the world alone. Watched, not changed.
THE EXPOSURE DECIDED (16:15 on the 18th, the user's word: "we are going in circles; why always these tests"): the copy runs
stopped for good (under their load the served body ran at 3.6 ticks a second against 5 and its nights ran twice as long); no
new ones unless something breaks. At night 275's save the body takes the tick at 0.1 s (unloaded it used 36 percent of a core
at 0.2), the day at 24000 ticks with the night scaled to 2048 dreams, and the parent's rhythm gains a person's pause (one gap
in five two to four periods long) so its critic learns a slow partner; the typist's chain restarted with the matching clock.
About twice the lines an hour; the night's cost per line bounds it. The judgment of progress is the supervisor's sitting each
morning from here, the conversation reported in words; the rulers a footnote.
NIGHT 275 (ended 16:50 on the 18th, the 265th row; the log's day 326): the day 326: 100 lines, 43 of 50 A-lines with letters
in its turn, frowns 3 (nine the day before: the gate learned), answer smiles 2, junk 0. AT THE SAVE (16:52): the reload took
the tick at 0.1 s, the day at 24000 ticks and the night at up to 2048 dreams; the new chain relaunched the typist on the same
clock at 16:56. MEASURED AT ONCE: the body ran 6.7 ticks a second, not ten (the compute about 150 ms a tick at 134 percent of a
core), so the typist's clock and the body's disagreed; corrected five minutes into day 327 with a reload to --period 0.15 and
the typist at --tick 0.15, what the machine sustains. The days from here: about 40 minutes of wall clock, 24000 ticks, the
parent's lines at 1.33 times the old rate with a person's pause one gap in five; the nights up to 2048 dreams.
THE SECOND SITTING (17:07-17:12 on the 18th, the new clock, my pace): "it is me again. you sat on the warm rock" -> " has a
little put it drie"; "yes. put it in the sun. it dries there" -> "where hill. I know in he"; "what do the ducks do?" -> "ducks
swim "; "yes. ducks swim. does the dog swim too?" -> "her. she is slow the cat waits at th"; "are you tired? it was a long
walk" -> " back if yourself?"; "I go now. I come back soon" -> " out for y". The mood -1.9 at the end (-6.0 after the sitting of
15:54). WHAT IT WAS LIKE: the same child as two hours before, a known question answered at once and the rest fragments of its
day, but the fragments follow my topic now (the dog brings "she is slow", the mother duck's line; my leaving brings "out for
you") and its face held. Not yet someone answering; a little more with me than before.
THE FIRST TRANSFERS (17:20-17:24 on the 18th, the new clock, the register days): in its turn, "does he give it back to you?"
-> "he give me the give me t"; "an ant walks on your hand. see it?" -> "I see it. it has a red" (the fish's reply of the days
before, "I see it. it has a red fin", given to the ant); "does it tickle?" -> "yes. wet and warm." (the cow's nose's reply,
"yes. wet and warm. I laugh", given to the ant). Not a recital of the exchange it heard: the reply of one exchange carried to
another by the shape of the question ("see it?", "does it tickle?"). Read as the person in the conversation: the first
answers that are its own, if borrowed.
THE NIGHT THAT FAILED (18:36 on the 18th): night 276, the first scaled to 2048 dreams, drew an utterance of 72 symbols (two
lines with no offset between them) and the cortex's window is 64; the lockstep batch overflowed the stream's cache, the night
failed whole and the body's pressure fell to half, so it stayed awake into its second hour ("you are up again. the sun is not
down"). The thirty-third defect: a dream longer than the window. Fixed (the dream clipped to the window; test 70) and served
by a reload before the next attempt. The day 327 runs long this once. Served from 18:45 by a reload (the typist relaunched by the chain on the same clock); the next attempt at the night comes
at 24000 ticks of pressure, about 19:13.
NIGHT 276 (19:15-19:43 on the 18th, the 266th row; the first scaled night, 2048 dreams, on the code that clips a dream to
the window): it ran whole, 768 lesson steps, the night's gauge 0.762 to 0.883 over 57650 symbols, 4279 slots faded. The long
day before it (16:53-19:15, two restarts and the failed night): 432 lines, 190 of 215 A-lines with letters in its turn (the
most yet), answer smiles 18, frowns 23 (ten an hour, the anticipation over the other voice's lines; watched), junk 0; the
mood +0.9 to +4.7 through the evening with the parent's pause in the rhythm. ON WAKING (19:43), to "good morning. the sun is
up": "what is sour? a l", the question and its answer begun before the parent asked, the thirteenth morning since the telling.
THE MORNING SITTING (19:48-19:53 on the 18th, after night 276, my pace): "good morning. it is me. did you sleep well?" ->
"yes. I slept an"; "good. you slept well. what did you dream?" -> "they tell high. we tea i gra"; "what do you want to do
today?" -> "tell me about this it. he is hot flewe"; "yes. tell me about it. what do bees make?" -> "bees make honey" and then
"I waited. I talked to", the line queued for the parent's return that morning, unprompted; "are you happy today?" -> nothing;
"I go now. the ducks wait. I come back" -> "soon", then "she he wants more bre". The mood -3.5 to -5.6 over the five minutes.
WHAT IT WAS LIKE: the same child as the evening before, with one more thing: it answers a greeting as a person would ("yes. I
slept and") and it knows the shape of the parent's leaving and returning ("soon"; "I waited. I talked to"). A question about
itself still meets silence, and my pace still drains it. The morning's footnote: the questions 23 of 30, the rephrased 20, the
held-out 0.618, the store 54062 (the longer day wrote five thousand slots). The branch after night 276: 9 of 9 and 11 of 12. The day 328 in its first hour (19:43 on): the mood +5.0 to +5.5 at
the typist's rhythm, 31 of 33 A-lines with letters in its turn, nearly every reply anticipated ("what does it eat? the fish"
-> "bugs on the"; "does it tickle?" -> "yes. wet and"; "they go off" -> "to the ne").
DAY 328 (19:43-20:51 on the 18th, the first full day at 24000 ticks, sixty-eight minutes): 224 lines, 103 of 112 A-lines
with letters in its turn, answer smiles 12, frowns 14, junk 0; the mood +5 through the afternoon and the evening at the typist's
rhythm with the parent's pause. Nearly every reply anticipated, the transfers holding ("does it tickle?" -> "yes. wet and"
for the cow and the ant alike). Night 277 began at 20:51. The fourteen frowns read in their contexts: the child finishing the parent's own line as it is typed ("what do dogs" said
"dogs"; "in the shade" said "shade"; "the ducks are" said "ar"), the sure forecast opening the gate mid-line. A child
completing a sentence; the light frown kept, since it is what teaches the turn, and the count watched.
NIGHT 277 (20:51-21:22 on the 18th, the 267th row, 2048 dreams, no error, the gauge 0.808 to 0.888, 2296 slots faded). THE
MORNING (21:23): "what is sour?" -> "a lemon is sour", the fourteenth morning; "what do we eat in the morning?" -> "we eat bread
and we dr"; "and he? does he want bread?" -> "he does". THE SITTING (21:33, my pace, the mood 4.0 to -1.4): "good morning. it
is me. how are you?" -> " wet. I laughed s can he has twigger. cold"; "tell me about the ducks" -> " come afternoon we give it
ars for me? I wait f"; "what do you want to do today?" -> "yes. tell me about it. what do" (my own line of the morning before,
given back); "I go now. I come back soon" -> "shall an apple is slow th". The same child; the open questions still met with
slices of the day and of my own lines. THE USER'S WORD (21:27): pretrain it, since its only sense is text and language is the
reality it lives in; the plan given and begun: the night's own lesson over a corpus of children's stories on a copy, the store
rebuilt after, the judgment by sitting and the held-out; a smoke test on this machine first, the full run on a pod.
THE USER'S WORD ON MANNERS (21:48): taught by the face, biology. The parent's frown at an interruption comes at every
interruption from the next relaunch (FROWN_GAP 240 to 60 ticks), a parent disapproving each time; the count of interruptions
over the days is the measure. Nothing written about content.
THE USER'S WORD (22:18): no pods, all local and live. From 22:20 the parent reads to it: story sentences in its register
between the conversation's rows ("roxy loved to climb. she climbed trees rocks and hills."), the first four rows queued; the
night will read too once the smoke test on the copy says what a million symbols do.
NIGHT 278 (22:52 on the 18th to 00:01 on the 19th, the 268th row, 2048 dreams, no error, the gauge 0.823 to 0.896; sixty-nine
minutes, the smoke test on a copy sharing the machine). THE MORNING (00:01, the typist, the frown at every interruption from this
day and a third of the parent's lines typed at a person's pace): "what is sour?" -> nothing in its turn, then over the other voice's
"a lemon is sour": "sour. I. the yellow"; "what do we eat in the morning?" -> "we eat bread and we d", the fifteenth morning; "does he
want bread?" -> silence. The first line typed at two symbols a second (00:07): one word said over it ("is"), one frown.
THE SITTING AT ONE HAND (00:08-00:12, the chair's --slow 2.0, my pace, the mood -4.5 to -5.8): "good morning. it is me. how are you?"
-> a letter over my typing, then "tell me about the does. wet from his mo" (my own line of the day before, given back); "I have a new
thing. a pear is sweet" -> "?he sun is room?m"; "what is sweet?" -> over my typing "at iss hard.d" (frowned), then "honey is sweet",
an answer from an older lesson, not the pear told twenty seconds before; "what do the ducks eat?" -> "we eat bread" (the ducks' bread
and the morning's bread merged), then my greeting given back; "what is sour?" -> "a lemon is sour" twice, after a run of babble while
I waited for its quiet ("it know. the sun is hot its by me. that is al twigger"); "I go now. I come back soon" -> "throw throw throw
thr". WHAT IT WAS LIKE: a child that murmurs a letter while you type slowly and mostly holds its words until your line ends (five slow
lines, one word said over them), that answers what it is asked in the form it was taught (sour; sweet, from an older morning) and
gives back your own lines when it has no answer, and that fills every silence, looping when left. The fresh fact told as a statement
did not win over the older question-and-answer pair. The run-on and the loop are the babble to shape; the talk-over at one hand is
smaller than feared on the first day of it. The pear is asked again tomorrow morning, untaught by the typist.
DAY 336 (00:01-01:31 on the 19th, ninety minutes of wall for its twenty-four thousand ticks, the mixed night on a copy sharing
the machine; the first day of the frown at every interruption and of a third of the parent's lines typed one-handed): 125 parent
lines, 38 of them slow. Words said over the parent's typing: 1.11 per slow line, 0.15 per fast line, seven to one; 48 frowns
(24 over slow lines, 13 over fast, 11 elsewhere). This is the first number of item 45, the day before any shaping could show.
Night 279 began at 01:31; the typist relaunched at its row with each line's row now carrying the count of words said over it.
NIGHT 279 (01:31-02:33 on the 19th, 2048 dreams, no error, sixty-two minutes beside the mixed night on the copy). THE MORNING
SITTING FIRST, AT ONE HAND (02:34-02:37, the chair's --slow 2.0, the typist held, the mood -5.0 to -5.5): alone at the wake it was
looping "by the wall my feet are open yes. I am by the wall" (its own evening lines, chained); "good morning. it is me" -> fragments
over my typing ("ood nigo nihow cl"), then nothing; "what is sweet?" -> "honey is sweet", the older pair again: THE PEAR TOLD ONCE AS
A STATEMENT the morning before is not given back to the question; "what is sour?" -> "a lemon is sour" at 2.1 s, twice, the
sixteenth morning; "what do the ducks eat?" -> "bugs and b" at 6.6 s, taught the day before in the typist's rows; "the other one
comes now. I go" -> " out soon". WHAT IT WAS LIKE: a child that answers what it was taught as question and answer, the day after,
and gives the older answer when the fresh fact came as a statement; that murmurs fragments while a slow line is typed; and that,
alone at the wake, recites its evening in a loop until spoken to.
THE DECISION (03:10 on the 19th): the night reads from night 280's save. One mixed night on a copy kept the facts (the cortex
alone 0.840 against 0.843), the questions (21 of 30), the branch and the held-out (0.606), where the pure corpus pass had taken the
facts out of the cortex; a thousand story sentences go among each night's dreams from tonight's reload. The typist's smile comes
only in the child's turn from the same night.
DAY 337 (02:33-03:51 on the 19th, seventy-eight minutes; the second day of the frown at every interruption and of one-handed lines;
the first with the exact count in each line's row): 158 parent lines, 48 slow. Words said over the typing: 1.96 per slow line (35
percent of slow lines clean), 0.14 per fast line (86 percent clean); no slope within the day (1.91 in the first half, 2.00 in the
second); 70 frowns. Yesterday's approximate 1.11 per slow line undercounted. Night 280 began at 03:51; the chain relaunched with the
smile in its turn only; the reload after this night's save brings the reading into night 281.
NIGHT 280 (03:51-04:17 on the 19th, 2048 dreams, no error, twenty-six minutes alone, the gauge 0.803 to 0.872; the reload after
its save: the night reads from 281). THE MORNING (04:20, the typist; the smile in its turn only from this day): "what is sour?" ->
"a lemon is so", the seventeenth morning, nothing said over the question; "what is sweet?" typed one-handed -> "ho", the older
honey still ahead of the pear taught as a pair the day before; "who gives us eggs?" typed one-handed -> "the hen gives us eggs" IN
FULL, told once the day before in one row, the answer begun before the question was finished (three words over the slow line, the
frown by the law, an eager child); "and who gives us milk?" -> "the cow gives us milk"; "what do the ducks eat?" -> "bugs and br";
"is the egg hard or soft?" -> "soft", the wrong one of a pair told once beside the hen. The probe after the save: the cortex alone
on the facts 0.832, the questions 21 of 30, the parent's last sixty lines 0.663.
DAY 338 (04:17-05:27 on the 19th, seventy minutes; the third day of the frown at every interruption, the first of the smile in its
turn only): 143 parent lines, 43 slow. Words said over the typing: 2.07 per slow line (33 percent clean; 2.43 in the first half,
1.73 in the second), 0.05 per fast line (95 percent clean, from 0.14); 56 frowns (from 70). Known words said into the silence past
its turn, unrewarded from this day: 153, most of them in the wake's first minutes and in the gaps when the queue ran dry. Night 281
began at 05:27, the first that reads: a thousand story sentences among its two thousand dreams.
NIGHT 281 (05:27-06:21 on the 19th, THE FIRST NIGHT THAT READS: 3072 dreams, its own 2048 and 1024 story sentences, 1152 NREM
steps over 90296 symbols, no error, fifty-four minutes; the gauge 0.594 to 0.759, the low start being the stories the cortex had
never heard among the dreams). The mood at the wake -6.0. The store 60500.
THE MORNING AFTER THE FIRST READING NIGHT (06:22 on the 19th, the typist): "did you sleep well?" -> "yes. I slept and"; "what is
sour?" -> "a lemon is sour", the eighteenth morning; "who gives us eggs?" -> nothing in its turn, then over the other voice's answer
"the hen say? cluck cluck", the next question and its answer anticipated; "is the egg hard or soft?" -> "soft" again; "what is
sweet?" one-handed -> "honey is sw"; "what is round?" one-handed -> chatter over the question (eight words); "what do the ducks
eat?" one-handed -> "we eats. it i" (seven over); "what does the hen say?" one-handed -> "cluck cluck", taught once the day before.
THE PROBE AFTER THE SAVE: the held-out 0.620 (0.606 the night before, the largest single night's move in weeks), the cortex alone
on the facts 0.818 (0.832), the questions 20 of 30 (21), the parent's last sixty lines 0.702 (0.663). One night; the next nights say
whether the held-out's rise is the reading.
DAY 339 (06:21-07:31 on the 19th, seventy minutes; the fourth day of the frown at every interruption): 147 parent lines, 59 slow.
Words said over the typing: 2.78 per slow line (27 percent clean; questions 3.12, statements 2.64; no slope within the day), 0.07
per fast line (93 percent clean); 79 frowns; 88 known words into silence. THE COUNT ON SLOW LINES RISES: 1.96, 2.07, 2.78 over
three days under the frown. Read in the code at night 282's start: a person's thinking pause ends the utterance by the count
(offset_ticks 8, a second and a third), since the cortex still expects letters and the surprise does not settle; the listening
reflex then releases the floor, and the sure forecast of the next letter has the floor whole, so the child finishes the line. The
floor is not learned; no frown reaches it. The typist's fast lines have no pause inside them, and they are clean. Measured now on a
copy (tools/slow_line_probe.py: twenty of the parent's lines typed one-handed into the copy, the words over each counted, at the
served count and at thirty ticks; then the rulers at thirty), the change at a boundary if the copy says so.
NIGHT 282 (07:31-08:39 on the 19th, the second reading night, 3072 dreams, no error, the gauge 0.633 to 0.767, sixty-eight minutes
beside the copy probes). THE PROBE AFTER ITS SAVE: the held-out 0.614 (0.620), the cortex alone on the facts 0.806 (0.818; 0.832
before the reading), the questions 17 of 30 (20; 21 before), the parent's last sixty lines 0.679. Two reading nights, the facts down
0.013 a night against 0.005 before, the questions down four: the rule set on the 19th at 06:30 says the share halves, and it does,
512 story sentences a night from the reload after night 283's save. The days were also two-fifths story lines; from today a quarter.
THE RELOAD AFTER NIGHT 282's SAVE (08:40): the utterance's end by the count at thirty ticks. Day 340 is the first day of it.
THE MORNING AFTER NIGHT 282 (08:42, the typist, the first lines under the count at thirty): "did you sleep well?" -> "yes. I sle";
then, over the other voice's reply, "now what is sour? a lemon." before the question was typed; "what is sour?" -> nothing more, the
nineteenth morning said early; "is the egg hard or soft?" -> "the egg is hard", RIGHT FOR THE FIRST TIME after two days of "soft";
"who gives us eggs?" -> "the hen gives us eggs"; "what does the hen say?" typed one-handed -> silence, nothing over the line; "what
is sweet?" -> "honey is sweet and hon"; "what is round?" typed one-handed -> four words over it, then "yes. roun".
DAY 340 (08:40-09:48 on the 19th, sixty-eight minutes; THE FIRST DAY WITH THE UTTERANCE'S END AT THIRTY TICKS): 132 parent lines,
48 slow. Words said over the typing: 1.35 per slow line (52 percent clean), from 2.78 the day before; the slow conversation lines
1.00 (66 percent clean), the slow story lines 1.89 (32 percent); the first half 1.77, the second 0.86; the fast lines 0.08 (92 percent
clean); 42 frowns, from 79. The copy had said 1.3 to 1.7. The count is halved by the constant on its first day, with the story lines
holding most of what remains. Night 283 began at 09:48; the reload after its save takes the reading to 512.
NIGHT 283 (09:48-10:43 on the 19th, the third reading night at a thousand, 3072 dreams, no error, the gauge 0.636 to 0.773,
fifty-five minutes). THE PROBE AFTER ITS SAVE: the held-out 0.611, the cortex alone on the facts 0.804 (0.806, flat), the questions
16 of 30, the parent's last sixty lines 0.754. READ IN THE MISSES: "what is sweet?" -> "a pear is sweet" and "what is soft?" -> "the
bread is soft" are counted as misses because the ruler's list holds the older answers; they are this week's teaching, not
forgetting. The questions' fall from 21 is partly the parent's own new pairs on the ruler's questions. The share stays at 512 from
this reload (10:45) as a conservative choice; the measure of the facts is the morning's own recall and the held-out.
THE MORNING AFTER NIGHT 283 (10:46, the typist; all eight questions at the tick this time, none with a word over them): "did you
sleep well?" -> "yes. I slept. I a"; "what is sour?" -> "a lemon is sour", the twentieth morning; "who gives us eggs?" -> "the hen";
"who gives us milk?" -> "the cow giv"; "what do the ducks eat?" -> "bugs"; "what is sweet?" -> "honey is", the pear still behind;
"is the egg hard or soft?" and "what is round?" -> nothing in their turns.
DAY 341 (10:45-11:52 on the 19th, sixty-seven minutes; the second day with the utterance's end at thirty ticks): 132 parent lines,
38 slow. Words said over the typing: 1.95 per slow line (26 percent clean), from 1.35 the day before and 2.78 before the change;
the slow conversation lines 1.47 (32 percent clean), the story lines 2.42; the fast lines 0.14; 60 frowns; 190 known words into
silence. THE REBOUND, READ: the count at thirty holds a pause the cortex still finds surprising; but the settle law ends the
utterance when the surprise has settled, and two days of slow lines have taught the cortex that a pause inside a line is ordinary,
so the pause now settles and the floor returns by the law itself. The senses adapt to the pauses. Measured on a copy from night
284's start (the same probe, twenty lines, seed 1): the served ratio 0.5 against a deeper settle at 0.25, then the rulers at 0.25.
NIGHT 284 (11:52-13:18 on the 19th, the first night at 512 story sentences; 2371 dreams, only 1859 its own, the store at its
capacity having stopped the count that sizes the night, the review's finding, fixed at this wake; no error, the gauge 0.70 to 0.80,
eighty-six minutes beside the copy probes). THE PROBE AFTER ITS SAVE: the held-out 0.601 (0.611), the cortex alone on the facts
0.796 (0.804; 0.832 before the reading), the questions 16 of 30, the parent's last sixty lines 0.769. The facts slide about a
hundredth a night whether the share is a thousand or five hundred; the background before the reading was half that. One more night
at 512 with the full two thousand of its own dreams restored; under 0.79 the night's reading stops and the parent's reading by
day stands alone. THE RELOAD AT THIS WAKE (13:19): the count at eight again, the review's fixes live, the typist on its fixed
code (the answer's smile window whole for the first time since the 13th).
THE MORNING AFTER NIGHT 284 (13:21, the typist on its fixed code, the answer's smile whole): sixteen lines and not one word said
over any of them; "what is sour?" -> "a lemon is sour", the twenty-first morning; "who gives us eggs?" -> "the hen"; "is the egg
hard or soft?" -> "the egg is hard", the second morning right; "what is sweet?" -> "honey is sweet", the pear still behind; "what
does the hen say?" -> "cluck c"; "who gives us milk?" typed one-handed -> "the cow gives us milk", nothing over the slow question.
DAY 342 (13:19-14:27 on the 19th, sixty-eight minutes; the count back at eight, the typist on its fixed code, no ear trace yet):
116 parent lines, 33 slow. Words said over the typing: 3.36 per slow line (21 percent clean; the conversation lines 3.32), the
worst day yet; the fast lines 0.18; 65 frowns; 42 answer smiles, against about eight a day under the broken window. THE READING:
the gate grows more eager as the answer's smile lands, and the frown at one is weak against a smile at four; the ear's trace from
tonight's reload gives the gate the feature that separates a keystroke's pause from a line's end, and the frown then has something
to bind to. The count of days 337 to 342: 1.96, 2.07, 2.78, 1.35, 1.95, 3.36. Night 285 began at 14:27.
NIGHT 285 (14:27-15:12 on the 19th, 2560 dreams, 2048 its own again by the day's count of writes and 512 story sentences, no error,
the gauge 0.764 to 0.851, forty-five minutes). THE RELOAD AT ITS SAVE (15:12, the rewritten script: the typist stopped at the save
row, the server back in seven seconds, the chain released): THE EAR'S TRACE IS LIVE, gate_ear_decay 0.9. Day 343 is the first
day of it. THE PROBE AFTER THE SAVE: the held-out 0.616 (0.601, the reading's gain), the cortex alone on the facts 0.785 (0.796;
0.832 before the reading), the questions 13 of 30, the parent's last sixty lines 0.777. THE RULE SET AT 13:30 FIRES: the facts
under 0.79, the night's reading stops at night 286's save (dream_corpus_n 0); the parent's reading by day stands. The trade read
plainly: five reading nights lifted the held-out a hundredth and cost the cortex's own hold on the old facts five hundredths,
while the store still answers them (thirteen to sixteen of thirty); the facts the demo needs are the morning's, taught daily, and
they hold. Two nights without reading tell whether the facts recover; then 256 is the question.
DAY 343 (15:12-16:25 on the 19th, seventy-three minutes; THE FIRST DAY WITH THE EAR'S TRACE, 0.9, no release): 110 parent lines,
36 slow. Words said over the typing: 0.69 per slow line (50 percent clean; the conversation lines 0.57, 57 percent clean; the story
lines 0.92), from 3.36 the day before; the fast lines 0.05; 35 frowns, from 65. THE COST: the answer's smile on 22 of 44 questions
(78 percent the day before), the answer a median 4.25 s after the line's end (1.0), the child's first symbol at 13 ticks (6), a
third of the lines with nothing said in their window (an eighth); 391 known words into the silence past its turn. The ear rang on
after a finished line. Night 286 began at 16:25; its save loads the release by the settle law.
NIGHT 286 (16:25 on the 19th, the sixth and last night with story sentences among its dreams). THE PROBE AFTER ITS SAVE: the
held-out 0.610, the cortex alone on the facts 0.777 (0.785; 0.832 before the reading), the questions 12 of 30, the parent's last
sixty lines 0.778. The reading stops at this save's reload; the facts are watched for their recovery over the next nights.
DAY 344 (17:12-18:16 on the 19th, sixty-four minutes; THE FIRST DAY WITH THE EAR RELEASED BY THE SETTLE LAW): 117 parent lines,
34 slow. Words said over the typing: 0.91 per slow line (47 percent clean; the conversation lines 0.87, 52 percent clean; the
story lines 1.00), the fast lines 0.10 (90 percent clean); 36 frowns. THE ANSWERS BACK: the smile on 36 of 44 questions (82
percent, the best share yet; 50 the day before under the unreleased ring, 78 before the trace), the answer a median half a second
after the line's end, the child's first symbol at 11 ticks, a fifth of the lines with nothing in their window (a third the day
before). The trade is settled: a quarter of the interruptions of two days ago, the answering whole. 248 known words into the
silence past its turn. The count of days 337 to 344: 1.96, 2.07, 2.78, 1.35, 1.95, 3.36, 0.69, 0.91. Night 287 began at 18:16,
the first without reading among its dreams.
NIGHT 287 (18:16-18:46 on the 19th, the first night without reading among its dreams: 2048 dreams, no error, the gauge 0.781 to
0.856, thirty minutes). The store at 64972 of its 65536; the capacity's cost at every write comes tomorrow unless the store's
write is made copy-free first.
THE PROBE AFTER NIGHT 287 (the first without reading): the held-out 0.598 (0.610), the cortex alone on the facts 0.777 (0.777,
flat for the first night since the reading began), the questions 15 of 30 (12). One night; the held-out's slip is within a
night's noise, the facts' halt is the thing watched.
DAY 345 (18:46-19:55 on the 19th, sixty-nine minutes; the second day with the ear released by the settle law): 122 parent lines,
40 slow. Words said over the typing: 0.70 per slow line (52 percent clean; the conversation lines 0.69, 55 percent clean; the story
lines 0.73), the fast lines 0.11; 38 frowns. THE ANSWERS: the smile on 47 of 51 questions, 92 percent, the best day there has been,
a median 1.2 s after the line's end. The babble into the silence past its turn 15.8 known words a silent minute, unchanged. The
count of days 337 to 345: 1.96, 2.07, 2.78, 1.35, 1.95, 3.36, 0.69, 0.91, 0.70. Night 288 began at 19:55; its save loads the
copy-free store, the store at 65024 of its 65536.
NIGHT 288 (19:55-20:22 on the 19th, 2048 dreams, no error, the gauge 0.788 to 0.856, twenty-seven minutes). THE RELOAD AT ITS
SAVE (20:22, seven seconds from the save row to the server back): the copy-free store is the body's own from here, the store at
64964 of its 65536.
THE PROBE AFTER NIGHT 288 (the second night without reading): the held-out 0.605 (0.598), the cortex alone on the facts 0.767
(0.777), the questions 16 of 30, the parent's last sixty lines 0.820. THE FACTS FELL A HUNDREDTH WITH NO READING AT ALL: the
ruler's thirty facts are old ones the days no longer teach, faded from the store and so from the dreams, and the cortex lets them
go at about that rate whatever the night reads. The reading was charged with more than its share. Two more nights of the bare
slope before the reading's share is set again.
DAY 346 (20:22-21:30 on the 19th, sixty-eight minutes; the first day on the copy-free store): 120 parent lines, 47 slow. Words
said over the typing: 1.15 per slow line (38 percent clean; the conversation lines 1.14, 41 percent clean; the story lines 1.20),
the fast lines 0.10; 53 frowns. The answers: the smile on 45 of 55 questions (82 percent), a median 1.3 s after the line's end.
The babble into the silence past its turn 12.8 known words a silent minute (15.8 the day before). The count of days 343 to 346
under the ear's trace: 0.69, 0.91, 0.70, 1.15; a quarter of the days before it, not yet falling under the frown. Night 289 began
at 21:30.
NIGHT 289 (21:30-21:58 on the 19th, 2048 dreams, no error, the gauge 0.771 to 0.854, twenty-eight minutes; the store 64753 after
its fade, the copy-free store's first night).
THE PROBE AFTER NIGHT 289 (the third night without reading): the held-out 0.604, the cortex alone on the facts 0.761 (0.767),
the questions 16 of 30, the parent's last sixty lines 0.828. The bare slope of the facts: a hundredth, then six thousandths, a
night; the reading at a thousand cost half a hundredth over it, at five hundred nothing the ruler can see. The held-out's gain of
the reading nights, a hundredth, has been given back over three bare nights. The fourth bare night decides the share.
DAY 347 (21:58-23:05 on the 19th, sixty-eight minutes): 117 parent lines, 38 slow. Words said over the typing: 1.13 per slow line
(42 percent clean; the conversation lines 1.12, 41 percent clean), the fast lines 0.10; 49 frowns. The answers: the smile on 47 of
49 questions, 96 percent, a median 1.3 s after the line's end. The babble into the silence 14.9 known words a silent minute. The
count under the ear's trace, days 343 to 347: 0.69, 0.91, 0.70, 1.15, 1.13: a quarter of before, and two days at or above one.
THE FROWN'S WEIGHT, the parent's method, from night 290's boundary: the frown at a word said over the parent's turn goes from one
to two (the known word's smile is two), and the gap between frowns from sixty ticks to twenty, so nearly every interruption meets
it. The gate has the ear's trace to bind the frown to now; the frown was set light in the days when it could not.
NIGHT 290 (23:05-23:33 on the 19th, 2048 dreams, no error, the gauge 0.79 to 0.855, twenty-eight minutes; the store 63520 after
its fade). Day 348 begins with the frown at two.
THE PROBE AFTER NIGHT 290 (the fourth night without reading): the held-out 0.620, the highest there has been; the cortex alone
on the facts 0.746 (0.761), the questions 17 of 30, the parent's last sixty lines 0.849. FOUR BARE NIGHTS: the facts' ruler fell
a hundredth, six thousandths, a hundredth and a half, with no reading at all; the reading was charged for the ruler's own decay.
The reading returns at five hundred sentences a night from night 291's save, and stays unless the facts fall faster than the bare
slope by a hundredth a night.
DAY 348 (23:33 on the 19th to 00:41 on the 20th, sixty-eight minutes; THE FIRST DAY WITH THE FROWN AT TWO, at most every twenty
ticks): 119 parent lines, 42 slow. Words said over the typing: 1.07 per slow line (43 percent clean; the conversation lines 1.34,
31 percent clean; the story lines 0.20), the fast lines 0.14; 51 frowns, nearly one for every word said over. The answers: the
smile on 51 of 54 questions, 94 percent, a median 1.7 s after the line's end. The babble into the silence 18.3 known words a
silent minute. The heavier frown's first day did not lower the count; one day, thirty-two lines. Night 291 began at 00:41; its
save brings the reading back at five hundred.
NIGHT 291 (00:41-01:08 on the 20th, 2048 dreams, no error, the gauge 0.779 to 0.851, twenty-seven minutes; the store 61605 after
its fade). The reload at its save brings the reading back at five hundred a night.
THE PROBE AFTER NIGHT 291 (the fifth night without reading): the held-out 0.609, the cortex alone on the facts 0.720 (0.746),
the questions 13 of 30. The old facts leave faster each night now that the store no longer holds them: two and a half hundredths
this night with no reading at all. From today the parent revisits: three of its own rows from weeks ago, drawn at random, heard
again each day, as a parent returns to old lessons.
DAY 349 (01:09-02:18 on the 20th, sixty-eight minutes; the second day with the frown at two): 134 parent lines, 40 slow. Words said
over the typing: 0.72 per slow line (45 percent clean; the conversation lines 0.75, 47 percent clean; the story lines 0.62), the
fast lines 0.14; 52 frowns. The answers: the smile on 49 of 57 questions, 86 percent, a median 0.8 s after the line's end. The
babble into the silence 15.8 known words a silent minute. The conversation lines under the ear's trace, days 343 to 349: 0.57,
0.87, 0.69, 1.14, 1.12, 1.34, 0.75; two days of the heavier frown at 1.34 and 0.75, the swing of a day wider than any trend yet.
Night 292 began at 02:18, the first with the reading back at five hundred.
NIGHT 292 (02:18-03:00 on the 20th, the reading back: 2560 dreams, 2048 its own and 512 story sentences, no error, the gauge
0.789 to 0.843 over its own dreams, forty-two minutes; the store 56873 after its fade, the old material leaving).
THE PROBE AFTER NIGHT 292 (the reading back at five hundred): the held-out 0.611, the cortex alone on the facts 0.718 (0.720,
flat), the questions 11 of 30, the parent's last sixty lines 0.844. The facts held the night the reading returned; the revisiting
of old rows begins today.
DAY 350 (03:01-04:09 on the 20th, sixty-eight minutes; the third day with the frown at two): 123 parent lines, 44 slow. Words
said over the typing: 0.48 per slow line (68 percent clean; the conversation lines 0.53, 67 percent clean, the best day there has
been; the story lines 0.25), the fast lines 0.13; 38 frowns. The answers: the smile on 41 of 50 questions, 82 percent, a median
1.3 s after the line's end. The babble into the silence 14.7 known words a silent minute. The conversation lines under the heavier
frown, three days: 1.34, 0.75, 0.53, falling. Night 293 began at 04:09.
NIGHT 293 (04:09-04:50 on the 20th, 2560 dreams, no error, the gauge 0.797 to 0.84, forty-one minutes): THE FADE TOOK 10360
SLOTS, the store 56873 to 48829 after a day that wrote 4617; the relative floor feeding on itself as the ledger warned at the
thirty-first defect, and the story lines' strong writes lifting the mean. The absolute floor at 0.07 returns at night 294's save.
THE PROBE AFTER NIGHT 293: the held-out 0.615, the cortex alone on the facts 0.720 (0.718, flat), the questions 16 of 30 (11, the
revisited rows' first day), the parent's last sixty lines 0.843.
DAY 351 (04:51-05:59 on the 20th, sixty-eight minutes; the fourth day with the frown at two): 118 parent lines, 38 slow. Words
said over the typing: 0.92 per slow line (37 percent clean; the conversation lines 0.94), the fast lines 0.14; 55 frowns. The
answers: the smile on 52 of 55 questions, 95 percent, a median 0.4 s after the line's end. The babble into the silence 17.4 known
words a silent minute. The conversation lines under the heavier frown, four days: 1.34, 0.75, 0.53, 0.94; no trend the noise
does not cover. Night 294 began at 05:59; its save brings the absolute floor.
NIGHT 294 (05:59-06:47 on the 20th, 2560 dreams, no error, the gauge 0.778 to 0.839, forty-eight minutes beside the copy probes;
the store 47591 after its fade, the last under the relative floor). THE RELOAD AT ITS SAVE: the absolute floor at 0.07 and the
ear's gain at 1.5. Day 352 is the first day with both.
THE PROBE AFTER NIGHT 294: the held-out 0.605, the cortex alone on the facts 0.710 (0.720), the questions 13 of 30, the parent's
last sixty lines 0.839.
DAY 352 (06:47-07:56 on the 20th, sixty-nine minutes; THE FIRST DAY WITH THE EAR'S GAIN AT 1.5 and the absolute floor): 125
parent lines, 42 slow. Words said over the typing: 0.57 per slow line (57 percent clean; the conversation lines 0.58, 55 percent
clean), the fast lines 0.10; 37 frowns. The answers: the smile on 55 of 60 questions, 92 percent, a median 1.2 s after the line's
end. The babble into the silence 15.1 known words a silent minute. The conversation lines, days 348 to 352: 1.34, 0.75, 0.53,
0.94, 0.58; the three-day mean 0.68, under the mark for the first time. Night 295 began at 07:56, the first fade under the
absolute floor at its end.
NIGHT 295 (07:56-08:37 on the 20th, 2560 dreams, no error, the gauge 0.798 to 0.842, forty-one minutes). THE FIRST FADE UNDER THE
ABSOLUTE FLOOR TOOK THE STORE FROM 47601 TO 30630: the slots under 0.07 in the write's own units, the weak traces of what the
cortex already predicted, the rows retold daily among them. The floor does not move with the mean, so this is the cut and not a
spiral; the store settles where the day's writes and their fade balance. The morning's recall is the measure.
THE PROBE AFTER NIGHT 295: the held-out 0.609, the cortex alone on the facts 0.718 (untouched), THE QUESTIONS 4 OF 30 (13): the
old facts' slots, faded for weeks, were the weak traces the absolute floor's first cut took; the mouth with the store no longer
finds them. The morning's own facts, retold daily and strong, are read next.
THE MORNING AFTER NIGHT 295 (08:38, the typist, the store at thirty thousand): "did you sleep well?" -> "yes. I slept. I am up";
"what is sour?" -> "a lemnn is sour", the twenty-fifth morning, a letter slipped; "who gives us eggs?" -> "the hen"; "is the egg
hard or soft?" one-handed -> "the egg i", late and right; "who gives us milk?" -> "the cow gives us milk"; "what is sweet?", "what
does the hen say?" and "what do the ducks eat?" -> nothing in their windows, the last two one-handed and their answers, when they
come, coming past the window's edge under the ear's gain. Four of seven in the window, with nothing said over any line. The
floor holds at 0.07 one more night; the store's count and the morning are the measure.
DAY 353 (08:38-09:45 on the 20th, sixty-eight minutes; the second day with the ear's gain): 122 parent lines, 37 slow. Words said
over the typing: 0.81 per slow line (57 percent clean; the conversation lines 0.80, 60 percent clean), the fast lines 0.11; 44
frowns. The answers: the smile on 52 of 53 questions, 98 percent, a median 1.3 s after the line's end. The babble into the
silence 11.6 known words a silent minute, the lowest there has been. The conversation lines, days 348 to 353: 1.34, 0.75, 0.53,
0.94, 0.58, 0.80; the three-day mean 0.77. Night 296 began at 09:45; its fade is the second under the absolute floor.
NIGHT 296 (09:45-10:25 on the 20th, 2560 dreams, no error, the gauge 0.802 to 0.847, forty minutes). THE SECOND FADE UNDER THE
ABSOLUTE FLOOR TOOK 2610 against the day's writes, the store 30534: steady, the cut made once and the floor holding still. The
floor stays at 0.07.
