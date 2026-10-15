---
zoom: 0.8
---

# Conclusions

<div class="grid grid-cols-2 gap-10">
<div>

### Ideas you will reuse
* A **gate** is a sigmoid layer used as a multiplier: the network decides, at every step, what to keep
* **LSTM, GRU**: a state that is copied by default, so the gradient survives — start the forget gate **open**
* **Stack** layers for depth; read **both ways** when the whole input is known — never to generate
* **Encoder–decoder**: input → one context vector → a conditional language model; teacher forcing
* Unequal lengths: **pad, mask, bucket, pack**
* **Beam search** with length normalization; **BLEU** to measure

</div>
<div>

### Measured today
* Memory task: with PyTorch's gates, 20–30 steps — worse than Lecture 6's RNN; gate open at the start: **200 steps** (GRU)
* The gradient along the memory: **0.6** per step → **0.98**
* Same size: GRU 4.43, LSTM 4.64, RNN 5.09 perplexity
* Fill in the blank: one direction 46 %, both **70–73 %**
* Translation: BLEU 31.6 greedy, **34.1** beam; 43 for 4–5 words, **7** for 16–20
* No packing: the same model scores 32 alone, **20** in mixed batches

</div>
</div>

<br>

#### The one idea to take away: **make the identity easy — in time, and start there.** An LSTM is to an RNN what a ResNet is to a plain deep network. What remains: a whole sentence squeezed into one vector.

<!--
One minute. The left column is what carries over: gates reappear in transformers (gated feed-forward layers) and in
modern recurrent models; encoder-decoder, teacher forcing, padding masks and beam search are exactly how transformer
translation models are trained and decoded.

The right column is today's evidence, all of it computed for this lecture with scripts/lecture7/ (the translation model
on a GPU in five minutes, everything else on a CPU) - a ready-made seminar: reproduce one number, then change one
thing (the forget-gate bias, alpha, the beam size, packing on or off).

The take-away links three lectures. Lecture 3: gradients vanish through many layers. Lecture 5: residual connections
make "do nothing" the default, and depth trains. Today: the LSTM's memory makes "copy" the default, and time trains -
but only if the network starts there: with PyTorch's default gates the LSTM was worse than Lecture 6's RNN. And the
last measurement is the next problem: one vector for a whole sentence; d2l's next chapter (attention) removes it.
-->

---
zoom: 0.62
---

# Learn More from the Experts

<div class="experts">

| Expert | Watch / read | Today's topics |
|---|---|---|
| Andrew Ng | [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models), Week 1: *Gated Recurrent Unit*, *Long Short Term Memory*, *Bidirectional RNN*, *Deep RNNs*; Week 3: *Basic Models*, *Picking the Most Likely Sentence*, *Beam Search*, *Refinements to Beam Search*, *Bleu Score* | the whole lecture, one idea per video |
| Yaser Abu-Mostafa | [Learning From Data, Lecture 13: Validation](https://www.youtube.com/watch?v=o7zzaKd0Lkk) · [Lecture 17: Three Learning Principles](https://www.youtube.com/watch?v=EZBUDG12Nr0) | choosing the epoch and the beam on validation data; no test sentence in training |
| Yann LeCun | [NYU Deep Learning, Week 6: RNNs, GRUs, LSTMs, Attention, Seq2Seq](https://atcold.github.io/NYU-DLSP20/en/week06/06-2/) · [practicum: RNN and LSTM](https://atcold.github.io/NYU-DLSP20/en/week06/06-3/) · [Deep learning, *Nature* (2015)](https://www.nature.com/articles/nature14539) | gated cells, encoder–decoder, "thought vectors" |
| Yoshua Bengio | [Cho et al. (2014): GRU and RNN encoder–decoder](https://arxiv.org/abs/1406.1078) · [Chung et al. (2014)](https://arxiv.org/abs/1412.3555) · [Cho et al. (2014): On the Properties of NMT](https://arxiv.org/abs/1409.1259) · [Bahdanau, Cho & Bengio (2014)](https://arxiv.org/abs/1409.0473) · [Deep Learning, Ch. 10](https://www.deeplearningbook.org/contents/rnn.html) | GRU, seq2seq, the fixed-length bottleneck — and its fix |
| Andrej Karpathy | [CS231n 2016, Lecture 10](https://www.youtube.com/watch?v=yCC09vCHzF8) · [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) · [Visualizing and Understanding Recurrent Networks (2015)](https://arxiv.org/abs/1506.02078) | LSTM gradient flow; what LSTM cells learn |
| Sebastian Raschka | [L15.5 Long Short-Term Memory](https://www.youtube.com/watch?v=k6fSgUaWUF8) · [a packed LSTM in PyTorch](https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L15/2_packed-lstm.ipynb) · [L19.1 Sequence Generation with Word and Character RNNs](https://www.youtube.com/watch?v=fSBw6TrePPg) | LSTMs and padding in PyTorch |
| Josh Starmer | [StatQuest: LSTM](https://www.youtube.com/watch?v=YCzL96nL7j0) · [Seq2Seq](https://www.youtube.com/watch?v=L8HKweZIOmg) · [Word Embedding](https://www.youtube.com/watch?v=viZrOnJclY0) | the cell, the encoder–decoder and embeddings, visually |

</div>

<style>
.experts td, .experts th { padding-top: 0.3em; padding-bottom: 0.3em; }
</style>

#### Main text: [d2l.ai, Chapter 10 — Modern Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-modern/index.html) &nbsp;·&nbsp; Read: [C. Olah, Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

#### Next in d2l.ai: attention — let the decoder look back at every encoder state ([Ch. 11](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html))

<!--
For self-study, in this order: StatQuest's LSTM and Seq2Seq videos (the two pictures of today), then Ng's Week 1 videos on
GRU and LSTM and his Week 3 videos on beam search - the length normalization and his "error analysis in beam search" are
worth the time - then Olah's post. Karpathy's CS231n lecture for the gradient-flow argument; his paper with Johnson and
Fei-Fei for what LSTM cells actually learn. Raschka's packed-LSTM notebook is the practical complement of the padding
slide. Bengio's group wrote the primary sources of half of today: the GRU and the RNN encoder-decoder (Cho et al.), the
GRU-LSTM comparison (Chung et al.), the analysis of the bottleneck (Cho et al., "On the Properties") and its fix,
attention (Bahdanau, Cho & Bengio) - d2l's next chapter.
-->
