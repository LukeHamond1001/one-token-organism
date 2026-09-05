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
