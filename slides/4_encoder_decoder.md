---
layout: center
---

<center>

# Encoder–Decoder

# Machine translation: from one sequence to another
</center>

---
zoom: 0.8
---

# Machine Translation: Unaligned Sequences

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

### Different length, different order
<div class="mt-table">

| English | French |
|---|---|
| they are watching . | ils regardent . |
| i have a red car . | j'ai une voiture rouge . |
| he gave it to me . | il me l'a donné . |
| i love you . | je t'aime . / je t'adore . |

</div>

* Lecture 6: **one output per input**. Here the whole input comes first
* Often several correct answers

</div>
<div>

### d2l's dataset <small>(Tatoeba, English–French)</small>
* **167,130** sentence pairs; tokens: words and punctuation
* Tokens seen at least twice in training: **9,450** English, **16,716** French; the rest is `<unk>`
* Special: `<bos>`, `<eos>`, `<pad>`, `<unk>`

<v-click>

### Our split: by English sentence
* **146,576** training pairs; **9,315** test sentences with all **13,324** of their translations
* No test sentence was seen in training with another translation <small>*(data snooping, Lecture 2)*</small>

</v-click>
</div>
</div>

<style>
.mt-table table { font-size: 0.85em; }
.mt-table td, .mt-table th { padding-top: 0.2em; padding-bottom: 0.2em; }
</style>

<span class="refs">Read: [d2l.ai 10.5 Machine Translation and the Dataset](https://d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html) · Data: [Tatoeba](https://tatoeba.org/) via [manythings.org/anki](https://www.manythings.org/anki/) · Watch: [A. Ng, Sequence Models, Week 3: "Basic Models"](https://www.coursera.org/learn/nlp-sequence-models) · [Y. Abu-Mostafa, Lecture 17: Three Learning Principles](https://www.youtube.com/watch?v=EZBUDG12Nr0)</span>

<!--
All four pairs are lines of the dataset, preprocessed as in d2l 10.5 (lower case, a space before punctuation). The
first is d2l's own example of Section 10.6.

The new problem: in Lecture 6's language model, input t and output t were aligned - one prediction per character.
Here the French sentence has a different number of tokens ("they are watching ." has 4, "ils regardent ." 3), the
words come in a different order ("red car" - "voiture rouge"; "gave it to me" - "me l'a donné"), and the first French
word may depend on the last English one. d2l calls such problems "unaligned"; also: dialog prompts and replies,
questions and answers. "Sequence-to-sequence" (seq2seq) is the name of the whole family.

Data: Tatoeba's crowd-sourced sentence pairs in the version d2l ships (fra-eng.zip). Lengths: 90 % of the English
sentences have at most 10 tokens; we keep pairs with at most 20 tokens on each side (166,658 pairs, 99.7 %). Word
tokens instead of characters (d2l: "today's state-of-the-art models use more complex tokenization techniques" -
subwords, the byte-pair encoding of Lecture 6's tokens slide). Vocabulary as in d2l: a token that occurs only once in the training sentences becomes
<unk>; 1.1 % of the English and 1.9 % of the French test tokens are unknown.

The split. Tatoeba often has several translations of one English sentence ("i love you ." has three). A random split
of PAIRS would put "i love you . -> je t'aime ." into training and "i love you . -> je t'adore ." into the test set: the
test would measure memory, not translation. We split the 116,444 distinct English sentences instead (88 % training, 4 % validation,
8 % test), and keep every translation of a test sentence as a reference for BLEU. Abu-Mostafa's principle (Lecture 17): "If a data set has affected any step in the learning process, its
ability to assess the outcome has been compromised."
-->

---
zoom: 0.88
---

# Batches of Unequal Length: Pad, Mask, Pack

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

### Pad to the longest sentence in the batch
```text
i   have  a   red  car  .   <eos>
he  gave  it  to   me   .   <eos>
go  .     <eos> <pad> <pad> <pad> <pad>
```
`valid_len = [7, 7, 3]`

<v-click>

### Mask: padding does not count in the loss
```python
l = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1),
                    reduction="none")
mask = (Y.reshape(-1) != PAD).float()
loss = (l * mask).sum() / mask.sum()
# the same: F.cross_entropy(..., ignore_index=PAD)
```

</v-click>
</div>
<div>
<v-click>

### Bucket: batch sentences of similar length
* Random batches of 128 pairs: **53 %** of all tokens are `<pad>`
* Sorted into buckets by length: **6.5 %**

</v-click>
<v-click>

### Pack: let the RNN stop at the real end
```python
packed = nn.utils.rnn.pack_padded_sequence(
    embs, valid_len, enforce_sorted=False)
H, h_n = encoder_rnn(packed)    # h_n: after "car .", not after <pad>
```

* Without it, the final state has also read the padding. **Measured**: such a model scores BLEU 32 one sentence at a time, **20** in mixed batches

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.5.3 Loading Sequences of Fixed Length](https://d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html#loading-sequences-of-fixed-length) · [d2l.ai 10.7.5 Loss Function with Masking](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#loss-function-with-masking) · Code: [S. Raschka, a packed LSTM](https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L15/2_packed-lstm.ipynb) · Docs: [pack_padded_sequence](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)</span>

<!--
Lecture 6 trained on windows of exactly 32 characters, so every batch was a rectangle. Sentences are not. Four tools,
all of them standard:

1. Pad (d2l 10.5): append <pad> until all sequences in the batch have the same length, and remember the valid lengths.
   d2l pads or truncates every sentence to a fixed num_steps = 9; we pad each batch to its own longest sentence.
2. Mask (d2l 10.7): the loss at a padding position is meaningless and must not be averaged in. d2l multiplies the
   per-token losses by a 0/1 mask and divides by the number of real tokens; PyTorch's ignore_index does the same.
   Without the mask, a batch with much padding would teach the model to predict <pad> after <pad> - easy and useless -
   and the average would be diluted (d2l's exercise 2 of 10.7).
3. Bucket (d2l: "Modern implementations often bucket sequences with similar lengths to avoid wasting excessive
   computation on padding"). Measured on our training set (scripts/lecture7/experiments.py mt_data): with random
   batches of 128 pairs, 53 % of the token positions in source and target are padding - more than half of the
   computation wasted, because one long sentence stretches the whole batch. Sorting chunks of 100 batches by length
   and then shuffling the batches: 6.5 %.
4. Pack: pack_padded_sequence tells the RNN where each sequence ends; it then processes only the valid steps, and the
   returned final state of each sequence is the state after its own last token. Without it (d2l's code), the encoder
   keeps running through <pad> tokens and its "final" state is the state after the padding - the network can learn to
   carry the information across the pads, but it has to. enforce_sorted=False lets the batch come in any order.

Measured (config gru2_nopack: the same model trained without packing, everything else equal; decode gru2_nopack 0
light): on its own, one sentence at a time, it translates as well as the packed model (test BLEU 32.1 greedy, 34.5 with
beam 5, against 31.6 and 34.1). But greedy decoding in batches of 256 test sentences of mixed length - as anyone would
evaluate - gives 19.9: 8,492 of the 9,315 translations change. In training, the bucketed batches had little padding,
so the model never learned to carry the sentence through many <pad> steps; in a mixed batch a 5-word sentence is
followed by up to 15 of them. A model whose output depends on what else is in its batch is a bug that no loss curve
shows: its validation loss was even slightly lower (4.77 against 4.90 in perplexity). Masking the loss, by contrast,
changed nothing measurable here (config gru2_nomask: 31.5 / 34.2), because bucketing leaves only 6.5 % of padding.

This closes the gap Lecture 6 left open: "variable length" was the first property of sequences on its slide 7, and
every Lecture 6 experiment used windows of equal length.
-->

---
zoom: 0.8
---

# The Encoder–Decoder Architecture

<div class="grid grid-cols-[11fr_10fr] gap-8">
<div>

<figure>
  <img src="/seq2seq.svg" style="width: 470px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/seq2seq.html">d2l.ai Fig. 10.7.1 Sequence-to-sequence learning with an RNN encoder and an RNN decoder</a>
  </figcaption>
</figure>

<br>

> *"This thought vector can then be used as the initial hidden state of (or as extra input to) a jointly trained French ‘decoder’ network"* <small>— LeCun, Bengio & Hinton, [Deep learning](https://www.nature.com/articles/nature14539), *Nature* (2015)</small>

</div>
<div>

### Two RNNs
* **Encoder**: reads the source; its final state is the **context** $\mathbf{c}$ — one vector, any length
* **Decoder**: a language model that also sees $\mathbf{c}$:

$$P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$$

<v-click>

### Training: teacher forcing <small>*(Lecture 6)*</small>
* Decoder input: `<bos> ils regardent .`
* Targets: `ils regardent . <eos>` — shifted by one
* One loss trains both RNNs, end to end

</v-click>
<v-click>

* **Using it**: start with `<bos>`, feed back each output, stop at `<eos>`

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.6 The Encoder–Decoder Architecture](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html) · [Sutskever, Vinyals & Le (2014)](https://arxiv.org/abs/1409.3215) · [Cho et al. (2014)](https://arxiv.org/abs/1406.1078) · Watch: [J. Starmer, StatQuest: Seq2Seq, Clearly Explained](https://www.youtube.com/watch?v=L8HKweZIOmg) · [S. Raschka, L19.1 Sequence Generation with Word and Character RNNs](https://www.youtube.com/watch?v=fSBw6TrePPg)</span>

<!--
The architecture of 2014 (Sutskever, Vinyals & Le; Cho et al.), and the one d2l builds in 10.6-10.7.

Read the figure left to right: the encoder RNN reads "They are watching ." plus <eos> and produces nothing but its final
state. That state - the context - is handed to the decoder. The decoder is a language model exactly like Lecture 6's,
except that it is conditioned on the context: it predicts "Ils" after <bos>, "regardent" after "Ils", and so on, and
the target sequence ends with <eos> so that the model can learn when to stop.

Teacher forcing (Lecture 6): during training the decoder's input at step t' is the TRUE previous French token, not its
own prediction - the decoder input is the target shifted right by one, with <bos> in front. All target positions are
then trained in parallel with one cross-entropy, and the gradient flows from the decoder's loss back through the
context into the encoder: both are trained together.

Using the model (d2l Fig. 10.7.3, backup): the decoder starts from <bos>, and from then on its own output is its next
input - the free-running generation of Lecture 6 - until it emits <eos>.

How the decoder gets the context. Sutskever et al. use the encoder's final state as the decoder's initial state. Cho et
al. also feed the context as an extra input at every decoding step. d2l (and our model) does both. The Nature quote
describes the same two options and coins the name: "After reading an English sentence one word at a time, an English
'encoder' network can be trained so that the final state vector of its hidden units is a good representation of the
thought expressed by the sentence. This thought vector can then be used as the initial hidden state of (or as extra
input to) a jointly trained French 'decoder' network".

"Whatever the length": a 3-word sentence and a 20-word sentence are both squeezed into the same 256 numbers. Keep that
in mind for the last slide of the lecture.
-->

---
zoom: 0.82
---

# Seq2seq in PyTorch

<div class="grid grid-cols-[14fr_9fr] gap-6">
<div>

```python {all|1-9|11-21|23-25|all}
class Encoder(nn.Module):
    def __init__(self, vocab, embed=256, hidden=256, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed)  # token id -> vector
        self.rnn = nn.GRU(embed, hidden, layers, dropout=0.2)
    def forward(self, X, X_len):                     # X: (batch, steps)
        embs = pack_padded_sequence(self.embedding(X.T), X_len, enforce_sorted=False)
        _, state = self.rnn(embs)
        return state                                 # (layers, batch, hidden)

class Decoder(nn.Module):
    def __init__(self, vocab, embed=256, hidden=256, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed)
        self.rnn = nn.GRU(embed + hidden, hidden, layers, dropout=0.2)
        self.dense = nn.Linear(hidden, vocab)
    def forward(self, Y_in, state, context):         # context: (batch, hidden)
        embs = self.embedding(Y_in.T)                # (steps, batch, embed)
        ctx = context.expand(embs.shape[0], -1, -1)  # the same context at every step
        out, state = self.rnn(torch.cat([embs, ctx], -1), state)
        return self.dense(out).swapaxes(0, 1), state # (batch, steps, vocab)

state = encoder(X, X_len)                            # one training step:
logits, _ = decoder(Y_in, state, state[-1])          # Y_in = <bos> + target[:-1]
loss = masked_loss(logits, Y)                        # Y = target, ends in <eos>
```

</div>
<div>

### The embedding layer
* `nn.Embedding(V, 256)`: a learned table, row $i$ = the vector of token $i$
* = one-hot vector × matrix, without the multiplication
* Trained with the rest: similar words get similar vectors

<v-click>

### 12.8 M parameters
* Embeddings 2.4 M + 4.3 M, output layer 4.3 M
* Both GRUs: only 1.8 M

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.7 Sequence-to-Sequence Learning](https://d2l.ai/chapter_recurrent-modern/seq2seq.html) · Docs: [nn.Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) · Watch: [J. Starmer, StatQuest: Word Embedding](https://www.youtube.com/watch?v=viZrOnJclY0) · [A. Ng, Sequence Models, Week 2: "Word Representation"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
d2l 10.7's Seq2SeqEncoder and Seq2SeqDecoder, with packing added and d2l's base classes left out; the code runs
(scripts/lecture7/slide_code.py), and it is the model trained on the next slide (scripts/lecture7/experiments.py,
config gru2: d2l's hyperparameters - embeddings of 256, two GRU layers of 256 units, dropout 0.2).

Walk through the highlights:
1. Encoder: tokens -> embeddings -> a 2-layer GRU over the packed batch. Only the final state is returned: one
   256-vector per layer and sentence.
2. Decoder: the same pattern, plus two things. Its GRU input is the embedding of the previous French token
   concatenated with the context (the encoder's last-layer final state), at every step; and its initial state is the
   encoder's final state - hence the same number of layers and units in both. A linear layer maps each state to logits
   over the 16,716 French tokens.
3. Training: one call to each, teacher forcing, the masked loss of the padding slide. Adam with learning rate 0.001
   (d2l uses 0.005 on its 512-pair subset), batches of 128 pairs, the gradient clipped at 1 as in d2l.

The embedding layer is the second new building block of today. Lecture 6 fed one-hot vectors of length 28; with 16,716
French words that would be a 16,716-long input with a single 1. Multiplying a one-hot vector by a weight matrix just
selects one row, so nn.Embedding stores the matrix and does the selection. Its rows are trained by backpropagation like
any weight, and words used in similar contexts end up with similar rows - which answers Lecture 6's open problem that
counts and one-hot vectors know nothing about similarity. (Pretrained word vectors, word2vec and GloVe: StatQuest's
video; d2l Ch. 15.)

Parameters (counted on the trained model): English embedding 9,450 x 256 = 2.4 M, French embedding 16,716 x 256 = 4.3 M,
output layer 256 x 16,716 + 16,716 = 4.3 M, the two GRUs together 1.8 M. The recurrent part - the subject of this
lecture - is 14 % of the model; the vocabulary is the rest.
-->

---
zoom: 0.82
---

# Training on 146,576 Sentence Pairs — Measured

<figure>
  <img src="/seq2seq_training.svg" style="width: 640px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; text-align: center">
    The model of the previous slide; Adam, learning rate 0.001, batches of 128 pairs, 15 epochs of 17 s each on a GPU. BLEU: next slide.
  </figcaption>
</figure>

<div class="tr">

| Greedy translations of sentences **not in the data** | after epoch 1 | epoch 2 | epoch 10 |
|---|---|---|---|
| he's calm . | il est mort . | il est très intelligent . | il est calme . |
| do you know where she lives ? | savez-vous où elle l'a fait ? | savez-vous où elle est ? | sais-tu où elle vit ? |
| the train was late because of the snow . | le train était la maison à la maison . | le train était en retard à la dernière fois . | le train était en retard pour la matinée . |

</div>

<v-click>

#### Fluent first, faithful later — and never quite: *"pour la matinée"* is good French and the wrong meaning. From epoch 10 on the validation loss rises: we keep epoch 10 *(early stopping, Lecture 3)*.

</v-click>

<style>
.tr table { font-size: 0.72em; }
.tr td, .tr th { padding-top: 0.18em; padding-bottom: 0.18em; }
</style>

<span class="refs">Read: [d2l.ai 10.7.6 Training](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#training) · [Sutskever, Vinyals & Le (2014)](https://arxiv.org/abs/1409.3215) · Watch: [A. Ng, Sequence Models, Week 3: "Basic Models"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
Measured for this lecture (DEVICE=cuda scripts/lecture7/experiments.py seq2seq gru2 0): d2l 10.7's model and loss, on our
split of the full dataset instead of d2l's 512 pairs. 15 epochs; after every epoch, the validation loss (per French
token, padding masked) and the BLEU of greedy translations of 1,000 validation sentences.

Validation perplexity per French token: 16.1 after epoch 1, 8.6 after 2, 5.3 after 5, 4.90 at epoch 10 (the best);
then it rises to 5.11 at epoch 15 - the model starts to memorize its training pairs (Lecture 3's two curves). Dev
BLEU: 9.3, 17.6, 27.4, 31.4 at epoch 10, 31.7 at 15 - BLEU keeps creeping up while the loss already rises: the two
measure different things (the loss rewards probability on the reference words, BLEU the n-gram overlap of the single
greedy output). We keep the checkpoint with the lowest validation loss, epoch 10; the test set is used only once, on
the next slides.

Time: 17-19 s per epoch on an NVIDIA RTX 5060 Ti. The same code on this machine's CPU (12 threads, shared with other
programs) took 16 minutes for epoch 1 and 34 for epoch 2, so the CPU run was stopped (logs kept in scripts/). The
cost is in the output layer: 1,000 French tokens per batch, each scored against all 16,716 words.

The table: three sentences that occur nowhere in the dataset, greedy decoding after epochs 1, 2 and 10. After one
epoch the model produces French-looking sentences with the right beginning ("savez-vous où elle ...") and an
invented rest ("il est mort" for "he's calm"); after two the structure is right and the content drifts; after ten,
two of three are correct. "the train was late because of the snow" never gets its second half: "à cause de la neige"
is replaced by a fluent, frequent ending. The decoder is a strong French language model; the context vector is the
weak part - the bottleneck of the last slide.

d2l's own four test sentences ("go .", "i lost .", "he's calm .", "i'm home .") are a weak test here: three of them are
training pairs of the full dataset. After epoch 10 the model translates them as "va y aller !", "j'ai perdu .",
"il est calme .", "je suis chez moi ." (at epoch 15: "va !").
-->

---
zoom: 0.8
---

# Measuring a Translation: BLEU

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

### Count matching $n$-grams
* $p_n$: the share of the output's $n$-grams that also occur in the reference <small>(each reference $n$-gram used once)</small>
* d2l's example — reference `A B C D E F`, output `A B B C D`:

$$p_1 = \tfrac{4}{5},\quad p_2 = \tfrac{3}{4},\quad p_3 = \tfrac{1}{3},\quad p_4 = 0$$

<v-click>

### Combine, and punish short outputs
$$\textrm{BLEU} = \underbrace{\min\!\Big(1, e^{\,1 - \textrm{len}_\textrm{ref}/\textrm{len}_\textrm{out}}\Big)}_{\textrm{brevity penalty}} \cdot \big(p_1 p_2 p_3 p_4\big)^{1/4}$$

* Without the penalty, the output `.` would score perfectly on $p_1$

</v-click>
</div>
<div>
<v-click>

### In practice
* Computed over the **whole test set**, not per sentence, and reported ×100
* Several references help: a correct translation that is not *the* reference still scores
* A proxy for human judgment — compare systems on the **same** test set and tokenization, e.g. with [sacreBLEU](https://github.com/mjpost/sacrebleu)

</v-click>
<v-click>

### Our model, 9,315 test sentences
* Greedy: BLEU **31.6** — $p_1 = 0.62$, $p_4 = 0.18$, brevity penalty 0.99

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.7.8 Evaluation of Predicted Sequences](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#evaluation-of-predicted-sequences) · [Papineni et al., BLEU (2002)](https://aclanthology.org/P02-1040/) · [M. Post, A Call for Clarity in Reporting BLEU Scores (2018)](https://aclanthology.org/W18-6319/) · Watch: [A. Ng, Sequence Models, Week 3: "Bleu Score"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
BLEU (Papineni, Roukos, Ward & Zhu, 2002) answers "how many of the output's word groups also appear in a human
translation?".

Left: d2l's worked example. Reference A B C D E F, output A B B C D. Unigrams: A, B, B, C, D - four of five occur in the
reference, but B only once, so the second B does not count ("clipping": each reference n-gram can be matched once):
p1 = 4/5. Bigrams AB, BB, BC, CD: three match, p2 = 3/4. Trigrams ABB, BBC, BCD: one, p3 = 1/3. The only 4-gram does
not match: p4 = 0, and the geometric mean - hence BLEU-4 of this sentence - is 0. That is why BLEU is computed over a
whole test set: all matches and all n-grams are summed over the corpus first, and only then divided.

The formula on the slide is the standard one (uniform weights 1/4, n up to 4, brevity penalty with the reference
length). d2l uses a variant with weights p_n^(1/2^n) that favours long n-grams, and its sentence-level function with
k = 2; scripts/lecture7/experiments.py contains both (corpus_bleu, and bleu_d2l verbatim). On d2l's example the d2l
version with k = 2 gives 0.68.

The brevity penalty: precision alone rewards saying little. The output "." has p1 = 1 against almost any reference
of ours; the penalty exp(1 - len_ref / len_out) is 1 for outputs at least as long as the reference and drops fast for
shorter ones.

Several references: Tatoeba has up to 18 translations of one English sentence in our test set; an n-gram counts as
matched if it occurs in any of them (clipped by its maximum count in one reference), and the brevity penalty uses the
reference closest in length. 29 % of our test sentences have two or more references.

Practice: BLEU depends on tokenization, lower-casing and the number of references, so numbers from different setups
cannot be compared. Post (2018) proposed sacreBLEU to standardize it. Ours is tokenized, lower-cased BLEU on our own
test split - comparable only within this lecture. BLEU is cheap and correlates with human judgment on average over a
test set, not for a single sentence; Ng's video makes the same points.

Our model (decode gru2 0, greedy): BLEU 31.6 on the test set; p1 = 0.62, p2 = 0.37, p3 = 0.26, p4 = 0.18; brevity
penalty 0.989 (7.55 output tokens per sentence on average). On the 1,000 validation sentences used during training:
31.4 at the kept epoch. With an LSTM instead of the GRU (config lstm2): 31.2. Published English-French systems of 2014
reported BLEU in the 30s on WMT'14 news text (Sutskever et al.: 34.8) - a different test set, tokenization and
difficulty, so the numbers are NOT comparable; that is the point of the "In practice" column.
-->
