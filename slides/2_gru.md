---
zoom: 0.82
---

# The GRU: Two Gates, One State

<div class="grid grid-cols-[10fr_11fr] gap-8">
<div>

<figure>
  <img src="/gru-3.svg" style="width: 420px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/gru.html">d2l.ai Fig. 10.2.3 Computing the hidden state in a GRU model</a>
  </figcaption>
</figure>

<br>

<v-click at="1">

* **Update gate** $\mathbf{Z}_t$: keep the old state ($\approx 1$) or take the candidate ($\approx 0$) — forget and input gate in one
* **Reset gate** $\mathbf{R}_t$: how much of the past the candidate may see

</v-click>
</div>
<div>

$$
\begin{aligned}
\mathbf{R}_t, \mathbf{Z}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{x}\cdot} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{h}\cdot} + \mathbf{b}_{\cdot}) \\
\tilde{\mathbf{H}}_t &= \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + (\mathbf{R}_t \odot \mathbf{H}_{t-1}) \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}) \\
\mathbf{H}_t &= \mathbf{Z}_t \odot \mathbf{H}_{t-1} + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t
\end{aligned}
$$

<v-click at="2">

* The last line is a weighted **average** of old and new: $\mathbf{Z}_t = 1$ copies the state — the identity path again
* $\mathbf{R}_t = 1,\ \mathbf{Z}_t = 0$: Lecture 6's RNN
* **3×** the parameters of an RNN, ¾ of an LSTM's; `nn.GRU` has the interface of `nn.RNN`

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.2 Gated Recurrent Units](https://d2l.ai/chapter_recurrent-modern/gru.html) · [Cho et al., Learning Phrase Representations using RNN Encoder–Decoder (2014)](https://arxiv.org/abs/1406.1078) · Watch: [A. Ng, Sequence Models, Week 1: "Gated Recurrent Unit (GRU)"](https://www.coursera.org/learn/nlp-sequence-models) · Docs: [torch.nn.GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)</span>

<!--
As on the LSTM slide, the first line stands for two gate equations with their own weights (W_xr, W_hr, b_r and
W_xz, W_hz, b_z).

The GRU (Cho et al., 2014 - the same paper that introduced the RNN encoder-decoder of Section 4) keeps the LSTM's key
idea, a state that is copied by default, with less machinery: no separate memory cell, no output gate.

Read the last equation first. H_t is an elementwise weighted average of the old state and a candidate: with Z_t = 1
the unit copies its state and ignores the input - d2l: "effectively skipping time step t in the dependency chain";
with Z_t = 0 it takes the candidate. In LSTM terms the update gate plays the forget gate (Z) and the input gate (1 - Z)
at once: what is written is exactly what is forgotten.

The candidate is Lecture 6's recurrent layer, except that the old state is first multiplied by the reset gate. R_t = 0
makes the candidate depend on the current input only; d2l: "the candidate hidden state is the result of an MLP with
X_t as input. Any pre-existing hidden state is thus reset to defaults."
d2l's summary: reset gates help capture short-term dependencies, update gates long-term ones.

Extreme case: R = 1, Z = 0 gives H_t = tanh(X_t W_xh + H_{t-1} W_hh + b_h), the vanilla RNN - so a GRU can be at
least as good as an RNN, in principle.

Conventions differ: Ng writes c<t> = Gamma_u * c~<t> + (1 - Gamma_u) * c<t-1>, so his update gate means "write"; d2l,
Cho et al. and PyTorch use Z for "keep": PyTorch's documentation has h_t = (1 - z_t) * n_t + z_t * h_{t-1}. In PyTorch's
weight matrices the order is reset, update, new.

Parameters: three blocks instead of four. For 28 inputs and 256 units: 219,648 against the LSTM's 292,864.
-->

---
zoom: 0.84
---

# RNN, GRU or LSTM? — Measured

<div class="grid grid-cols-[13fr_10fr] gap-8">
<div>

<div class="compact-table">

| *The Time Machine*, Lecture 6's loop | Parameters | ms per step | Validation perplexity |
|---|---|---|---|
| RNN, 256 units *(Lecture 6)* | 80,412 | 26 | 5.09 <small>(5.05–5.14)</small> |
| GRU, 128 units | 64,284 | 30 | 4.43 <small>(4.41–4.44)</small> |
| LSTM, 128 units | 84,508 | 24 | 4.64 <small>(4.62–4.66)</small> |
| GRU, 256 units | 226,844 | 77 | **4.36** <small>(4.34–4.39)</small> |
| LSTM, 256 units | 300,060 | 62 | 4.64 <small>(4.63–4.65)</small> |
| 8-gram counts *(Lecture 6)* | 96,165 counts | — | **4.0** |

</div>

<small>3 seeds each; SGD, learning rate 1, clipping at 1, windows of 32 characters, 6,000 steps. Time: one training step on a CPU, 2 threads.</small>

</div>
<div>

### Reading it honestly
* At the **same size**, a gated cell beats the RNN by 0.5–0.7
* No run of a gated cell ever needed clipping: largest gradient norm 0.7 (RNN: up to 6.3)

<v-click>

* GRU ≥ LSTM here; across many tasks, *"comparable"*
* 2–3× the time per step — and on this CPU the GRU is the **slower** one
* Counting still wins: 32-character windows hold no long dependency to exploit

</v-click>
<v-click>

#### Practice: an LSTM or a GRU, gate open; tune size and regularization, not the cell

</v-click>
</div>
</div>

<style>
.compact-table table { font-size: 0.78em; }
.compact-table td, .compact-table th { padding-top: 0.26em; padding-bottom: 0.26em; }
</style>

<span class="refs">Read: [Chung, Gulcehre, Cho & Bengio (2014)](https://arxiv.org/abs/1412.3555) · [Jozefowicz, Zaremba & Sutskever (2015)](https://proceedings.mlr.press/v37/jozefowicz15.html) · [d2l.ai 10.1–10.2, exercises](https://d2l.ai/chapter_recurrent-modern/gru.html#exercises) · backup: "LSTM Variants: Does Any of Them Win?"</span>

<!--
Measured for this lecture (scripts/lecture7/experiments.py charlm <config> <seed>, seeds 0-2): Lecture 6's CharRNN with
nn.RNN replaced by nn.GRU or nn.LSTM, nothing else changed - one-hot inputs, SGD with learning rate 1, clipping at 1,
128 random windows of 32 characters per step, 6,000 steps (d2l 10.1-10.2 use the same recipe with lr 4). Perplexity:
the model reads the last 10 % of the book once, left to right.

Per seed, validation perplexity after 6,000 steps:
- RNN 256: 5.05, 5.09, 5.14 (Lecture 6's runs, reproduced bit for bit with today's script);
- GRU 128: 4.43, 4.44, 4.41; LSTM 128: 4.64, 4.62, 4.66;
- GRU 256: 4.39, 4.34, 4.36; LSTM 256: 4.65, 4.63, 4.64 (the run of Lecture 6's review, reproduced).
Training perplexity: GRU 256 3.10-3.13, LSTM 256 3.34-3.43, RNN 256 3.19-3.27: the GRU fits the training text best
and still generalizes best.

Size: GRU 128 has fewer parameters than RNN 256 and is better by 0.66; LSTM 128 has 5 % more and is better by 0.45. Twice
the units (and 3-4 times the parameters) buy the GRU 0.07 and the LSTM nothing in 6,000 steps.

Gradients: the largest gradient norm of any step in the 12 gated runs is 0.70 (LSTM 128); the clipping line never acted.
The RNN runs exceeded 1 in one or two steps each (up to 6.27) - Lecture 6's "rare, fatal" events. Bounded gates and an
additive memory make the gradient tamer in both directions.

Time (scripts/lecture7/experiments.py timing; one forward-backward-update step on a 128 x 32 batch, 2 threads, idle
machine): RNN 12 / 26 ms, GRU 30 / 77 ms, LSTM 24 / 62 ms for 128 / 256 units. A GRU does 3/4 of an LSTM's multiply-adds,
but PyTorch's CPU kernels for nn.LSTM are better optimized; on a GPU with cuDNN both are fast. d2l says GRUs are "faster
to compute" - in principle; measure on your hardware. d2l's exercise 3 of 10.1 asks exactly this.

The honest comparison with Lecture 6: the ReLU RNN of Lecture 6's backup reached 4.46 with 256 units, between the two
gated cells, and the 8-gram table 4.0. On a 156,000-character book with 32-character windows there is little long-range
structure for a memory to exploit; the gated cells' advantage is the memory task of the previous section.

"Comparable": Chung et al. (2014), "we found GRU to be comparable to LSTM". Jozefowicz et al. (2015): an LSTM with
forget-gate bias 1 closes the gap to the GRU. Our LSTMs use PyTorch's default (bias about 0), which may be why the GRU
leads here.
-->
