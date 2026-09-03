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
