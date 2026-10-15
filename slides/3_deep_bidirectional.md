---
layout: center
---

<center>

# Deep and Bidirectional RNNs

# Stacking layers; reading in both directions
</center>

---
zoom: 0.78
---

# Deep RNNs: Stack the Layers

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

<figure>
  <img src="/deep-rnn.svg" style="width: 330px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/deep-rnn.html">d2l.ai Fig. 10.3.1 Architecture of a deep RNN</a>
  </figcaption>
</figure>

</div>
<div>

* Layer $l$ reads the **outputs of layer** $l-1$ at the same step, and **its own state** from the previous step
* Deep in two directions: $T$ steps in time, $L$ layers from input to output

```python
lstm = nn.LSTM(28, 256, num_layers=2, dropout=0.2)
H, (h_n, c_n) = lstm(torch.randn(32, 2, 28))
H.shape        # [32, 2, 256]: the last layer, every step
h_n.shape      # [2, 2, 256]:  every layer, the last step
```

<v-click>

* `dropout` acts **between** layers, not along time <small>*(Lecture 3)*</small>
* Typical: 1–4 layers <small>(d2l: depths 1–8)</small>
* Deeper needs Lecture 5's trick: Google's 2016 translator stacked 8 LSTM layers **with residual connections**

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.3 Deep Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-modern/deep-rnn.html) · [Deep Learning, §10.5 Deep Recurrent Networks](https://www.deeplearningbook.org/contents/rnn.html) · [Wu et al., Google's Neural Machine Translation System (2016)](https://arxiv.org/abs/1609.08144) · Watch: [A. Ng, Sequence Models, Week 1: "Deep RNNs"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
d2l's observation: even a one-layer RNN is deep - an input at step 1 passes through T applications of the recurrent
layer before it reaches the output at step T. But between the input at step t and the output at the same step t there
is only one layer. Stacking adds depth in that direction: "we stack the RNNs on top of each other. Given a sequence of
length T, the first RNN produces a sequence of outputs, also of length T. These, in turn, constitute the inputs to
the next RNN layer."

Code (printed shapes are real, scripts/lecture7/slide_code.py): num_layers stacks the layers inside one module. H holds
the top layer's states at every step - what an output layer needs; h_n (and c_n) hold every layer's state at the last
step - what a decoder or the next window needs. The first dimension of h_n is the number of layers (times the number
of directions, next slide).

dropout=0.2 applies Lecture 3's dropout to the outputs of every layer except the last, i.e. between layers. It is not
applied to the recurrent connection; dropping the state at every step would disturb the memory (variational dropout,
Gal & Ghahramani 2016, uses one mask for all steps - not in nn.LSTM).

Depth in practice. d2l: "Common RNN layer widths (h) are in the range (64, 2056), and common depths (L) are in the range
(1, 8)." (2056 is d2l's number; 2048 is presumably meant.) Sutskever et al. (2014) used four layers of 1,000 LSTM units;
Wu et al. (2016), Google's production translator (GNMT), used eight LSTM layers in the encoder and eight in the decoder,
with residual connections between them, because "simple stacked LSTM layers work well up to 4 layers, barely with 6
layers, and very poorly beyond 8 layers": Lecture 5's lesson, again.

Measured on The Time Machine (scripts/lecture7/experiments.py charlm gru2_h256 0, one seed, the recipe of the comparison
table): two stacked GRU layers of 256 units, 621,596 parameters. Best validation perplexity 4.16 (at step 4,600), against
4.37 for one layer (seed 0); then it overfits - training perplexity 2.58, validation 4.31 at step 6,000. Depth helps a
little and overfits sooner on a small book. Its 4.16 is the lowest perplexity of any character model today, but the
best step was picked on the same validation text it is reported on, which flatters it; still behind the 8-gram
table's 4.0.
-->

---
zoom: 0.86
---

# Bidirectional RNNs: Read Both Ways

<div class="grid grid-cols-[10fr_11fr] gap-8">
<div>

### The right context matters too
* I am `___` .
* I am `___` hungry .
* I am `___` hungry, and I can eat half a pig .

<figure>
  <img src="/birnn.svg" style="width: 380px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/bi-rnn.html">d2l.ai Fig. 10.4.1 Architecture of a bidirectional RNN</a>
  </figcaption>
</figure>

</div>
<div>

### Two RNNs, one input
* One reads $\mathbf{x}_1 \to \mathbf{x}_T$, the other $\mathbf{x}_T \to \mathbf{x}_1$, with separate weights
* At each step: $\mathbf{H}_t = [\overrightarrow{\mathbf{H}}_t,\ \overleftarrow{\mathbf{H}}_t]$, $2h$ numbers

```python
bi = nn.LSTM(28, 256, bidirectional=True)
H, (h_n, c_n) = bi(torch.randn(32, 2, 28))
H.shape        # [32, 2, 512]
h_n.shape      # [2, 2, 256]: forward, backward
```

<v-click>

### When to use it
* **Yes**: the whole sequence is known — tagging every word, encoding a sentence (Section 4)
* **No**: predicting the next token. The future is not there yet

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.4 Bidirectional Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-modern/bi-rnn.html) · [Schuster & Paliwal (1997)](https://doi.org/10.1109/78.650093) · [Graves & Schmidhuber (2005)](https://doi.org/10.1016/j.neunet.2005.06.042) · Watch: [A. Ng, Sequence Models, Week 1: "Bidirectional RNN"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
d2l's motivation, verbatim examples: what fills the blank depends on what comes after it - "happy" for the first
sentence, "not" or "very" for the second, and "not" is incompatible with the third. Ng's example is named-entity
recognition: in "He said, Teddy Roosevelt was a great President" against "He said, Teddy bears are on sale", the
first three words are the same, and whether "Teddy" is a name is decided by the words after it.

The construction (Schuster & Paliwal, 1997): two independent RNN layers on the same input, one reading forwards, one
backwards, and their hidden states concatenated step by step. d2l: "a simple technique transforms any unidirectional
RNN into a bidirectional RNN". The output layer, or the next layer, sees 2h numbers that summarize the left and the
right context of position t.

Code (printed shapes are real): bidirectional=True doubles the feature dimension of H; h_n has one entry per direction:
h_n[0] is the forward direction after x_T, h_n[1] the backward direction after x_1. With num_layers=2 it would be 4
entries, ordered by layer, then direction.

When: d2l - "Bidirectional RNNs are mostly useful for sequence encoding and the estimation of observations given
bidirectional context." Graves & Schmidhuber (2005) combined both of today's ideas (bidirectional LSTM) for phoneme
classification. They are a standard choice for tagging and as encoders; BERT (later in the course) is the transformer
version of "read both ways, fill in the blank".

Cost: twice the parameters and computation of one direction, and the output for x_1 needs the whole sequence - no
streaming.
-->

---
zoom: 0.84
---

# Fill in the Blank — Measured

<div class="grid grid-cols-[12fr_10fr] gap-8">
<div>

### Guess the letter between 15 characters on each side

<div class="compact-table">

| LSTM | sees | Parameters | Accuracy |
|---|---|---|---|
| one direction, answer at the blank | left only | 84,508 | 46 % |
| both directions, 90 units each | left and right | 91,468 | **70 %** |
| both directions, 128 units each | left and right | 168,988 | **73 %** |

</div>

<div class="fb mt-4">

| *The Time Machine*, validation text | true | left only | both |
|---|---|---|---|
| `able apprehe_sion still k` | n | d | **n** |
| `phrase be wa_dering on so` | n | s | **n** |
| `intensely br_ght and seem` | i | o | **i** |
| `traveller hes_tated then` | i | t | **i** |

</div>
</div>
<div>
<v-click>

### The trap: a bidirectional language model
* Train a bidirectional LSTM to predict the **next** character, as in Lecture 6
* Validation perplexity after 2,000 steps: **1.004** — "perfect"
* Generated text: `stststststststststst`

</v-click>
<v-click>

* At step $t$ the backward direction has already **read** $x_{t+1}$: the answer is in the input
* When generating, there is no $x_{t+1}$ yet

</v-click>
</div>
</div>

<style>
.compact-table table { font-size: 0.78em; }
.compact-table td, .compact-table th { padding-top: 0.24em; padding-bottom: 0.24em; }
.fb table { font-size: 0.74em; }
.fb td, .fb th { padding-top: 0.18em; padding-bottom: 0.18em; }
</style>

<span class="refs">Read: [d2l.ai 10.4 Bidirectional RNNs](https://d2l.ai/chapter_recurrent-modern/bi-rnn.html) · [Graves & Schmidhuber (2005)](https://doi.org/10.1016/j.neunet.2005.06.042) · Watch: [S. Raschka, L19.5.2.3 BERT](https://www.youtube.com/watch?v=_BFp4kjSB-I) (fill in the blank, at scale)</span>

<!--
Measured for this lecture (scripts/lecture7/experiments.py fillblank <model> <seed>, seeds 0-2). Windows of 31
characters from The Time Machine; the middle one is replaced by <unk> (a token that never occurs in the book) and the
model must name it. Blanks on spaces are skipped, so it is always a letter. nn.LSTM on one-hot characters; the output
layer reads the state at the blank. Adam (learning rate 0.002), clipping at 1, 4,000 steps of 128 windows; accuracy on
16,094 windows of the validation text.

Accuracy per seed: one direction 46.4 / 46.6 / 45.7 %; both directions, 90 units each (about the same number of
parameters) 69.9 / 69.9 / 69.7 %; both directions, 128 units each 72.7 / 73.0 / 72.0 %. Perplexity at the blank:
5.6-5.8 against 2.6-2.8. With only the left context the task is next-character prediction - the language model of
Lecture 6, whose perplexity was 4.4-5.1. The right context is worth more than doubling the network.

The examples: among the first 30 validation windows, the one-direction model is wrong and the bidirectional one right
in nine; four of them are on the slide (seed 0 of both models, 128 units). "apprehe_sion": the left context
allows "apprehend", the right context forces "n".

A fourth model, not on the slide: one direction, reading all 31 characters and answering after the last one - all
the information, but it must carry the letter 16 steps. In 4,000 steps it learned nothing: 12.6 % in all three runs,
exactly the share of "e" among the letters - it always answers "e". The memory problem of Section 2, with PyTorch's
default gates.

The trap (scripts/lecture7/experiments.py charlm bilstm_h128 0): Lecture 6's language-model loop with
nn.LSTM(..., bidirectional=True). Validation perplexity 4.7 after 200 steps, 1.18 after 400, 1.004 after 2,000: the
output at step t sees the backward state, which has read the target x_{t+1}. Generated, one character at a time, the
backward direction sees only the current character: greedy decoding of "it has" gives " i i i i i i i i i i", sampling
"stststststst". This is why masked-token models (BERT) replace the token they predict with a mask - exactly what our
fill-in-the-blank task does with <unk>.
-->
