---
zoom: 0.9
---

# Essentials of Artificial Neural Networks

### Building blocks (<span style="color:#FA9370">new</span>):
<div class="grid grid-cols-[4fr_4fr_2fr] gap-3">
<div>

* Neuron
* Fully-connected (a.k.a. *Linear*) layer
* Activation function
* Loss function
* Convolution, pooling *(Lecture 4)*
</div>

<div>

* Batch normalization, residual connection *(Lecture 5)*
* Recurrent layer *(Lecture 6)*
<v-clicks>

* <span style="color:#FA9370">**Gated recurrent cell: LSTM, GRU**</span>
* <span style="color:#FA9370">**Embedding layer**</span>
</v-clicks>
</div>

<div>
  <figure>
    <img src="/lego_A.jpg" style="width: 190px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image source:
      <a href="http://sgaguilarmjargueso.blogspot.com/2014/08/de-lego.html">http://sgaguilarmjargueso.blogspot.com</a>
    </figcaption>
  </figure>   
</div>
</div>

<br>

### Key concepts (review from previous lectures | <span style="color:#FA9370">new</span>):
<div class="grid grid-cols-[1fr_1fr_1fr_1fr] gap-3">
<div>

* Weights & Biases
* Backpropagation
* Gradient descent

</div>
<div>

* Learning rate
* MiniBatch
* Regularization
</div>

<div>

* Weight sharing
* Residual learning
* Hidden state, BPTT
</div>

<div>
<v-clicks>

* <span style="color:#FA9370">**Gating**</span>
* <span style="color:#FA9370">**Encoder–decoder**</span>
* <span style="color:#FA9370">**Beam search**</span>
</v-clicks>
</div>

</div>

<!--
Lecture 6 added the recurrent layer. Today replaces its cell with a better one and puts two recurrent networks
together.

Key points:
- Gated recurrent cell: an LSTM or a GRU is a drop-in replacement for the recurrent layer of Lecture 6 - same inputs,
  same outputs, same training loop - with multiplicative gates inside that decide what to keep.
- Embedding layer: a learned table with one vector per token. It replaces the one-hot vectors of Lecture 6 as soon as
  the vocabulary has thousands of words instead of 28 characters.
- "Gating": multiply a signal by a number between 0 and 1 that the network computes itself. The idea that makes
  memory last, and it returns in transformers.
- "Encoder-decoder": one network reads the input, another writes the output - for tasks where input and output have
  different lengths, like translation.
- "Beam search": how to pick a whole output sequence, not just one token at a time.

BPTT = backpropagation through time (Lecture 6).

Transition: Lecture 6 ended on two measurements that said a vanilla RNN forgets. Let us see what we have.
-->

---
zoom: 0.8
---

# From Lecture 6 to Today

<div class="grid grid-cols-2 gap-12">
<div>

### We already have *(Lectures 1–6)*
* A recurrent layer, a language model, teacher forcing *(Lecture 6)*
* Clipping against **exploding** gradients *(Lecture 6)*
* Residual connections: make the identity easy *(Lecture 5)*

### We are still missing
* A memory that **lasts** *(Lecture 6: 0.8 per step back)*
* Outputs whose **length and order** differ from the input's

</div>
<div>

### Plan for today
1. **Gated memory**<br><small>why a vanilla RNN forgets, and the fix we already know</small>
2. **LSTM and GRU**<br><small>memory cells and gates; which one to use</small>
3. **Deep and bidirectional RNNs**<br><small>stacking layers; reading in both directions</small>
4. **Encoder–decoder**<br><small>machine translation; padding, masking, teacher forcing</small>
5. **Decoding**<br><small>greedy and beam search; BLEU; the bottleneck</small>

</div>
</div>

<span class="refs">Main text: [d2l.ai, Ch. 10 Modern Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-modern/index.html) · Refresher: [J. Starmer, StatQuest: LSTM, Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) · Watch: [A. Ng, Sequence Models, Weeks 1 and 3](https://www.coursera.org/learn/nlp-sequence-models) · Read: [C. Olah, Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)</span>

<!--
Say, don't show: the training loop of Lecture 6 does not change today. The cell changes (Sections 1-3), and in
Section 4 the data become pairs of sentences.

Two running examples:
- the same as last week: "The Time Machine", read character by character - to compare the new cells with Lecture 6's;
- new: about 150,000 English-French sentence pairs (d2l's dataset), for translation.
Everything "measured" today was computed for this lecture (scripts/lecture7/): the translation model on a GPU in
minutes, everything else on a CPU.

Timing plan (80 min, of which 10-15 min is the quiz):
- Title, building blocks, this slide:                  3 min
- 1. Gated memory (3 slides):                          7 min
- 2. LSTM and GRU (6 slides):                         16 min
- 3. Deep and bidirectional RNNs (3 slides):           7 min
- 4. Encoder-decoder (6 slides):                      13 min
- 5. Decoding (5 slides, incl. the demo):             11 min
- Conclusions:                                         3 min
                                              total   60 min  + quiz, ~5-10 min slack for questions

If running late, in this order:
  (a) "LSTM in PyTorch": show only the nn.LSTM half,
  (b) "Deep RNNs": one sentence (num_layers=2 stacks two layers; the second reads the first's outputs),
  (c) the demo: only "the train was late because of the snow .",
  (d) "Measuring a Translation: BLEU": one sentence (the share of the output's word n-grams found in a reference,
      with a penalty for short outputs),
  (e) "Why: The Gradient Along the Memory": panel (a) only, one sentence.
-->
