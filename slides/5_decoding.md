---
layout: center
---

<center>

# Decoding

# Greedy search, beam search, and the bottleneck
</center>

---
zoom: 0.88
---

# The Most Likely Word ≠ the Most Likely Sentence

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

### Greedy search: take the best token, every step
<figure>
  <img src="/s2s-prob1.svg" style="width: 300px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/beam-search.html">d2l.ai Fig. 10.8.1 At each time step, greedy search selects the token with the highest conditional probability</a>
  </figcaption>
</figure>

<br>

$0.5 \times 0.4 \times 0.4 \times 0.6 = 0.048$

</div>
<div>

### Second best at step 2 — and better overall
<figure>
  <img src="/s2s-prob2.svg" style="width: 300px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/beam-search.html">d2l.ai Fig. 10.8.2 At time step 2, the token "C", which has the second highest conditional probability, is selected</a>
  </figcaption>
</figure>

<br>

$0.5 \times 0.3 \times 0.6 \times 0.6 = 0.054$

</div>
</div>

<v-click>

#### Trying every sequence costs $|\mathcal{Y}|^{T'}$: for 10,000 words and 10 steps, $10^{40}$. Greedy costs $|\mathcal{Y}|\,T'$ — and can miss the best.

</v-click>

<span class="refs">Read: [d2l.ai 10.8 Beam Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html) · Watch: [A. Ng, Sequence Models, Week 3: "Picking the Most Likely Sentence"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
d2l's example, four tokens A, B, C, <eos>; the columns are the decoder's distributions at steps 1-4. Left: greedy
search picks the most likely token at every step - A, B, C, <eos> - and the sequence has probability 0.048.
Right: take C, the second most likely token, at step 2. Everything after it changes, because the decoder is conditioned
on its own previous outputs: now B at step 3 has probability 0.6 and <eos> at step 4 has 0.6. The whole sequence
A, C, B, <eos> has probability 0.054 - more likely than greedy's.

What we want (d2l): "the most likely sequence, not the sequence of (greedily selected) most likely tokens". The
probability of a sequence is the product of its conditional probabilities - the chain rule of Lecture 6.

Exhaustive search finds it, at a cost of |Y|^T' sequences: d2l's numbers, |Y| = 10,000 and T' = 10, give 10^40. Our
French vocabulary has 16,716 tokens. Greedy search evaluates |Y| T' = 10^5 - "miraculously cheap but far from optimal".

Ng's example of why greedy fails: "Jane is visiting Africa in September" against "Jane is going to be visiting Africa
in September" - "going" is a more frequent word after "Jane is" than "visiting", so greedy search commits to the longer,
worse sentence.
-->

---
zoom: 0.8
---

# Beam Search

<div class="grid grid-cols-[11fr_10fr] gap-8">
<div>

<figure>
  <img src="/beam-search.svg" style="width: 420px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/beam-search.html">d2l.ai Fig. 10.8.3 The process of beam search (beam size = 2; maximum length of an output sequence = 3)</a>
  </figcaption>
</figure>

</div>
<div>

### Keep the $k$ best partial sequences
1. Step 1: the $k$ most likely first tokens
2. Each step: extend each of the $k$ by every token, keep the $k$ best of the $k\,|\mathcal{Y}|$
3. A sequence that emits `<eos>` is finished

<v-click>

* Cost $k\,|\mathcal{Y}|\,T'$; $k = 1$ **is** greedy search
* Work with $\log P$: a sum, no underflow

</v-click>
<v-click>

### Normalize for length
* Each extra token multiplies by $P < 1$: short outputs win
* Pick the finished sequence with the best $\frac{1}{L^{\alpha}} \log P(y_1, \ldots, y_L \mid \mathbf{c})$, $\alpha \approx 0.75$

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.8.3 Beam Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html#beam-search-1) · [Sutskever, Vinyals & Le (2014)](https://arxiv.org/abs/1409.3215) · Watch: [A. Ng, Sequence Models, Week 3: "Beam Search" and "Refinements to Beam Search"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
d2l's figure: vocabulary A-E, beam size 2, at most 3 steps. Step 1: the two most likely tokens are A and C. Step 2:
extend both by all five tokens - ten candidates - and keep the best two: AB and CE. Step 3: again ten candidates, keep
ABD and CED. The answer is chosen among the candidates by the score at the bottom of the slide.

Our implementation (scripts/lecture7/experiments.py beam_search) is the usual variant: candidates that end in <eos> are
set aside as finished; the search stops when k sequences are finished; the winner is the finished sequence with the
best length-normalized log-probability.

Cost: k times greedy - in practice a batch of k decoder rows per step, so on a GPU almost free for small k. Beam size 1
is greedy. Sutskever et al. (2014): "our system performs well even with a beam size of 1, and a beam of size 2 provides
most of the benefits of beam search".

Log-probabilities: products of 20 probabilities underflow and are awkward; sums of logs are what the decoder's
log_softmax gives anyway.

Length normalization (d2l's formula; Ng's "Refinements to Beam Search" uses the same with alpha = 0.7): log P only
decreases with every token, so without normalization the search prefers short outputs, and with a large beam it finds
a short one that stops too early. Dividing by L^alpha (L the length including <eos>) compensates; alpha = 0 is no
normalization, alpha = 1 the average log-probability per token. alpha is a hyperparameter - choose it on validation
data, not on the test set.

Beam search is not only for translation: the same procedure decodes speech recognizers and any encoder-decoder; for
open-ended text generation, sampling with a temperature (Lecture 6) is used instead, because the most likely text is
dull and repetitive.
-->

---
zoom: 0.84
---

# Beam Search — Measured

<figure>
  <img src="/beam.svg" style="width: 700px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; text-align: center">
    Our encoder–decoder (epoch 10) on the 9,315 test sentences, all their translations as references.
  </figcaption>
</figure>

<div class="grid grid-cols-[11fr_10fr] gap-8">
<div>

<div class="bx">

| English | greedy, log P | beam 5, log P |
|---|---|---|
| a horse can run very fast . | une porte de porte peut courir très vite . −6.1 | **un cheval** peut courir très vite . −2.9 |
| i don't really know why . | je ne sais pas pourquoi je le fais . −4.0 | je ne sais pas **vraiment** pourquoi . −1.5 |

</div>
</div>
<div style="font-size: 0.85em">

* Greedy **31.6** → beam 5 **34.1**; most of the gain at $k = 2$–3
* Beam changes 43 % of the outputs; 39 % become **more likely**

<v-click>

* Without length normalization the outputs get **shorter** as $k$ grows, and BLEU stalls at 33.6

</v-click>
</div>
</div>

<style>
.bx table { font-size: 0.66em; }
.bx td, .bx th { padding-top: 0.15em; padding-bottom: 0.15em; }
</style>

<span class="refs">Read: [d2l.ai 10.8 Beam Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html) · [Koehn & Knowles, Six Challenges for NMT (2017)](https://aclanthology.org/W17-3204/) · Watch: [A. Ng, Sequence Models, Week 3: "Error Analysis in Beam Search"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
Measured for this lecture (DEVICE=cuda scripts/lecture7/experiments.py decode gru2 0): the model of the training slide
(epoch 10), the 9,315 test sentences, BLEU-4 against all their reference translations.

Test BLEU, greedy: 31.6. Beam search with length normalization (alpha = 0.75): k = 2: 33.5, 3: 33.9, 5: 34.1,
10: 34.2. Without normalization (alpha = 0): 33.3, 33.6, 33.6, 33.6. Two thirds of the gain is at k = 2 - Sutskever et
al.: "a beam of size 2 provides most of the benefits of beam search".

What the search does. With k = 5 and alpha = 0.75 the output changes for 43 % of the sentences, and for 39 % the new
output is more likely under the model than greedy's (for the rest the length normalization preferred a longer, slightly
less likely one). The average log-probability of the outputs rises from -4.02 (greedy) to -3.38. So greedy search
misses the most likely translation often - d2l's toy example of the previous slide happens on real data.

The right panel: log P of every extra token is negative, so without normalization a larger beam finds shorter outputs:
7.55 tokens on average with greedy search, 6.95 with k = 10. BLEU punishes that with the brevity penalty (0.99 for
greedy, 0.91 for k = 10, alpha = 0), so BLEU stops improving at 33.6. With alpha = 0.75: 7.25 tokens, penalty 0.95.
This is the "beam search curse" in mild form; Koehn & Knowles (2017) found that translation quality drops for large
beams: "Beam search decoding only improves translation quality for narrow beams and deteriorates when exposed to a
larger search space."

The examples are test sentences (k = 5, alpha = 0.75), with the model's log-probabilities. Greedy search commits to
"une porte" after "une" (door, after which every continuation is unlikely); the beam keeps "un cheval" alive. In the
second, greedy search drops "vraiment" and has to invent an ending.

Cost: one sentence at a time, 7 ms per sentence on the GPU for k = 1 to 10 alike - each step is one decoder call on k
rows; batched greedy search takes 0.1 ms per sentence. For a deployed system, k = 4-5 with normalization is typical.

Ng's "Error analysis in beam search" asks the right question for a bad translation: is the human translation more
likely under the model than the output? If yes, the search failed (larger beam); if no, the model is wrong (more data,
better model).
-->

---
zoom: 0.92
---

# Try It: Beam Search, Step by Step

<BeamSearchDemo />

<div class="mt-2" style="font-size: 0.7em">

#### Pick a sentence, press **step ▶**: three partial translations survive each step; a finished one waits in the right column &nbsp;·&nbsp; at the end, compare with greedy search

</div>

<!--
Live demo, 3 min. The traces are real: our model with beam size 3 and alpha = 0.75 on six English sentences (four from
the test set; "the train was late ..." and "do you know where she lives ?" written for the lecture and in none of the
data), recorded by scripts/lecture7/experiments.py
beam_traces; the page replays them, nothing is computed in the browser.

Suggested order:
1. "the train was late because of the snow ." Step: the first words are certain ("le", log P -0.07). At step 2 the
   beam keeps "le train", "le retard", "la neige" - three different ways to start. Step on until "en retard": greedy
   search continues with "pour la matinée", the beam also keeps "à cause de" alive, and finishes with "le train était
   en retard à cause de la neige ." (log P -3.31 against greedy's -4.12). The right column shows how finished
   candidates collect, with their length-normalized score.
2. "a horse can run very fast .": greedy writes "une porte de porte ..." (log P -6.09) - after "une" the most likely
   noun was "porte"; the beam also keeps "un", and "un cheval peut courir très vite ." wins with -2.90.
3. "do you know where she lives ?": the beam agrees with greedy - this happens for more than half of the test sentences.

Point to make: every number on the screen is a log-probability from the decoder; beam search only decides which
partial translations to keep.
-->

---
zoom: 0.84
---

# The Bottleneck: Long Sentences — Measured

<div class="grid grid-cols-[10fr_11fr] gap-8">
<div>

<figure>
  <img src="/bleu_by_length.svg" style="width: 470px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 6px">
    Test BLEU of our encoder–decoder by the length of the English sentence.
  </figcaption>
</figure>

<v-click>

<div class="lx">

| she owes him a lot of money , but she probably won't be able to pay it back . |
|---|
| greedy: elle lui doit être très en mesure de parler mais il ne peut le faire , mais elle ne le mit plus . |

</div>
</v-click>
</div>
<div>

* 4–5 words: BLEU **43**; 16–20 words: **7** — and beam search does not rescue it
* The whole sentence must fit into **one vector** of 256 numbers

> *"we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder–decoder architecture"* <small>— Bahdanau, Cho & Bengio (2014)</small>

<v-click>

* Honest caveat: long sentences are also **rare** in training — 0.7 % of the pairs have 16–20 words
* Reading the source **backwards** (Sutskever et al.: 25.9 → 30.6) did not help here: 30.4 greedy, 33.7 with beam 5

</v-click>
<v-click>

#### The fix: let the decoder **look back** at every encoder state, not just the last — *attention* ([d2l.ai, Ch. 11](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html))

</v-click>
</div>
</div>

<style>
.lx table { font-size: 0.66em; }
.lx td, .lx th { padding-top: 0.15em; padding-bottom: 0.15em; }
</style>

<span class="refs">Read: [Cho et al., On the Properties of Neural Machine Translation (2014)](https://arxiv.org/abs/1409.1259) · [Bahdanau, Cho & Bengio (2014)](https://arxiv.org/abs/1409.0473) · [Sutskever, Vinyals & Le (2014)](https://arxiv.org/abs/1409.3215) · Watch: [S. Raschka, L19.3 RNNs with an Attention Mechanism](https://www.youtube.com/watch?v=mDZil99CtSU)</span>

<!--
Measured for this lecture (decode gru2 0): test BLEU per length of the English sentence, greedy / beam 5:
1-3 tokens 36.7 / 42.5 (155 sentences); 4-5 43.3 / 46.3 (2,083); 6-7 38.8 / 41.3 (3,432); 8-9 30.5 / 33.3 (2,180);
10-12 20.4 / 22.9 (1,076); 13-15 13.2 / 14.9 (301); 16-20 7.0 / 7.7 (88). (Very short sentences score lower than 4-5
words because one wrong word is a large share of a 3-word output, and "go ." has many valid translations.)

The example is the 17-token test sentence on the slide, greedy: the first words are right ("elle lui doit"), then the
output drifts into fluent French about something else - the decoder has run out of information and falls back on
being a language model. Its reference: "elle lui doit beaucoup d'argent mais elle ne sera probablement pas en mesure
de le rembourser ."

Why: every sentence, of 3 words or of 20, must pass through the same 256 numbers (per layer) of the encoder's final
state. Cho et al. (2014, "On the Properties of Neural Machine Translation") measured the same shape on English-French:
"its performance degrades rapidly as the length of the sentence and the number of unknown words increase". Bahdanau,
Cho & Bengio (2014) - the quote - then let the decoder look at all encoder states.

Honest caveat: in our data, long sentences are also rare - of the training pairs, 2.9 % have 13-15 English tokens and
0.7 % have 16-20 (36.7 % have 6-7). So part of the drop is lack of training examples; Cho et al. had the same confound.

Reversing the source (Sutskever, Vinyals & Le, 2014): "we found it extremely valuable to reverse the order of the words
of the input sentence", which brought the first source words close to the first target words; their test BLEU rose
from 25.9 to 30.6 with a 4-layer LSTM. Measured here (config gru2_rev, same everything else): test BLEU 30.4 greedy and
33.7 with beam 5, against 31.6 and 34.1 without reversing - no gain. Our sentences are short (7 tokens on average) and
our decoder sees the context at every step, so the first words do not have a long way to travel.

Next in d2l (Ch. 11, attention): instead of one context vector, the decoder computes, at every step, a weighted
average of ALL encoder states, with weights that it learns to put on the relevant source words.
-->
