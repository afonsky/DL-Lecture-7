---
layout: center
---

# Backup Slides

---
zoom: 0.78
---

# The LSTM's Gradient in Detail

<div class="grid grid-cols-[1fr_1fr] gap-12">
<div>

### Along the memory
$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t$$

* The direct path from $\mathbf{C}_{t-1}$ to $\mathbf{C}_t$: multiply by $\mathbf{F}_t$, elementwise
* Over $k$ steps: $\mathbf{F}_t \odot \mathbf{F}_{t-1} \odot \cdots \odot \mathbf{F}_{t-k+1}$ — **no weight matrix, no** $\tanh'$
* Each factor is chosen by the network, step by step: $\approx 1$ to keep, $\approx 0$ to erase

<v-click>

### The other paths
* $\mathbf{C}_{t-1}$ also reaches $\mathbf{C}_t$ through $\mathbf{H}_{t-1}$ and the gates — these contain $\mathbf{W}_{\textrm{h}\cdot}$ and can vanish like an RNN's
* The 1997 LSTM cut them off in the backward pass; today's LSTM keeps the full gradient

</v-click>
</div>
<div>

### Lecture 6's RNN, for comparison
$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \mathrm{diag}\big(\tanh'(\cdot)\big)\, \mathbf{W}_{\textrm{hh}}$$

* Over $k$ steps: a product of $k$ such matrices — the same $\mathbf{W}_{\textrm{hh}}$ every time

<v-click>

### The GRU
$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1} + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t$$

* Direct path: multiply by $\mathbf{Z}_t$ — the same idea, in the state itself

</v-click>
</div>
</div>

<span class="refs">Read: [Hochreiter & Schmidhuber (1997), Sec. 3–4](https://doi.org/10.1162/neco.1997.9.8.1735) · [Deep Learning, §10.10 The Long Short-Term Memory and Other Gated RNNs](https://www.deeplearningbook.org/contents/rnn.html) · [A. Karpathy, CS231n 2016, Lecture 10](https://www.youtube.com/watch?v=yCC09vCHzF8) · [CS231n 2017, Lecture 10 slides](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf) ("Uninterrupted gradient flow")</span>

<!--
The derivative of C_t with respect to C_{t-1} holding the gates fixed is diag(F_t): an elementwise scaling, which is
what "the gradient survives" means. Compared with Lecture 6's recurrent layer there is no multiplication by W_hh and no
tanh' on this path; the factors are numbers in (0, 1) that the network chooses at every step. If it keeps F close to 1
for the units that store something, the gradient reaches back as far as the memory does.

The gates themselves depend on H_{t-1}, which depends on C_{t-1}, so there are further paths from C_{t-1} to C_t, and
they do contain the recurrent weight matrices. Hochreiter & Schmidhuber (1997) truncated those in the backward pass
(only the error through the memory cell was propagated back in time), which made the "constant error carousel"
exact. Modern implementations compute the full gradient with autograd; the direct path is still there, so the
gradient never has to rely on the matrix products alone.

CS231n's 2017 slides draw the cell state as a highway on top of the cell with the caption "Uninterrupted gradient flow!"
and compare it to ResNet - the comparison of Section 1.

GRU: H_t is an interpolation between H_{t-1} and the candidate; holding Z fixed, the derivative along the direct path
is diag(Z_t), the same mechanism without a separate memory.
-->

---
zoom: 0.86
---

# Initializing the Gates

<div class="grid grid-cols-[11fr_10fr] gap-10">
<div>

<div class="compact-table">

| Forget-gate bias $b_\textrm{f}$ | 0 | 1 | 3 |
|---|---|---|---|
| Gate at the start, $\sigma(b_\textrm{f})$ | 0.5 | 0.73 | 0.95 |
| Memory left after 30 steps | $10^{-9}$ | $10^{-4}$ | 0.24 |
| after 100 steps | $10^{-30}$ | $10^{-14}$ | 0.008 |
| Who uses it | PyTorch (≈ 0) | Keras, Jozefowicz et al. | this lecture |

</div>

<br>

```python
h = lstm.hidden_size        # gate order: input, forget, cell, output
with torch.no_grad():
    lstm.bias_ih_l0[h:2*h] = 3.0
    lstm.bias_hh_l0[h:2*h] = 0.0
```

</div>
<div>

### What the table says
* At the start the gates read noise, so $\mathbf{F}_t \approx \sigma(b_\textrm{f})$ at every step
* The memory — and the gradient along it — is multiplied by $\sigma(b_\textrm{f})$ per step

<v-click>

### Going further
* GRU: the same for the update gate (PyTorch's order: reset, **update**, new)
* *Chrono* initialization: draw $b_\textrm{f}$ from $\log \mathcal{U}(1, T_{\max} - 1)$, set $b_\textrm{i} = -b_\textrm{f}$ — a spread of time scales up to the longest dependency you expect

</v-click>
</div>
</div>

<style>
.compact-table table { font-size: 0.82em; }
.compact-table td, .compact-table th { padding-top: 0.3em; padding-bottom: 0.3em; }
</style>

<span class="refs">Read: [Jozefowicz, Zaremba & Sutskever (2015), §2.2](https://proceedings.mlr.press/v37/jozefowicz15.html) · [Tallec & Ollivier, Can Recurrent Neural Networks Warp Time? (2018)](https://arxiv.org/abs/1804.11188) · Docs: [Keras LSTM, unit_forget_bias](https://keras.io/api/layers/recurrent_layers/lstm/) · [torch.nn.LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)</span>

<!--
The numbers are powers of sigma(b_f): 0.5^30 = 9.3e-10, 0.731^30 = 8.3e-5, 0.953^30 = 0.24; 0.5^100 = 7.9e-31,
0.731^100 = 2.5e-14, 0.953^100 = 0.0081 (scripts/lecture7/slide_code.py prints the first ones). They describe the
network at initialization, when the gates do not yet depend on the input in any useful way; training then moves them.

Jozefowicz, Zaremba & Sutskever (2015), "An Empirical Exploration of Recurrent Network Architectures", Section 2.2:
"Most applications of LSTMs simply initialize the LSTMs with small random weights which works well on many problems.
But this initialization effectively sets the forget gate to 0.5. This introduces a vanishing gradient with a factor of
0.5 per timestep, which can cause problems whenever the long term dependencies are particularly severe ... This problem
is addressed by simply initializing the forget gates bf to a large value such as 1 or 2." Their headline finding:
"adding a bias of 1 to the LSTM's forget gate closes the gap between the LSTM and the GRU."

Keras: unit_forget_bias defaults to True - "add 1 to the bias of the forget gate at initialization ... This is
recommended in Jozefowicz et al." PyTorch draws every weight and bias of nn.LSTM from U(-1/sqrt(h), 1/sqrt(h)), so the
forget gate starts near sigmoid(0) = 0.5. The code on the slide sets it, in the layout of the "LSTM in PyTorch" slide;
PyTorch adds bias_ih and bias_hh, so one of them carries the value.

Our memory task (Section 2) tried 0, 1 and 3: 0 and 1 failed beyond 20-30 steps, 3 learned up to 200 steps in most
runs.

Chrono initialization (Tallec & Ollivier, 2018) derives the gate bias from the time scale the network should remember:
a forget gate of f keeps information for about 1/(1 - f) steps, so b_f = log(T - 1) gives a time scale of T. Drawing
b_f per unit from log U(1, T_max - 1) spreads the units over all time scales up to T_max; the input-gate bias is set to
-b_f. In their words, it "is shown to greatly improve learning of long term dependencies, with minimal implementation
effort."
-->

---
zoom: 0.86
---

# LSTM Variants: Does Any of Them Win?

<div class="grid grid-cols-[1fr_1fr] gap-12">
<div>

### Proposed changes
* **Peepholes**: the gates also see the memory $\mathbf{C}_{t-1}$ <small>(Gers & Schmidhuber, 2000)</small>
* **Coupled gates**: write exactly what is forgotten, $\mathbf{I}_t = 1 - \mathbf{F}_t$ — a step towards the GRU
* **GRU**: no separate memory, no output gate <small>(Cho et al., 2014)</small>
* 10,000 architectures found by **automatic search** <small>(Jozefowicz et al., 2015)</small>

</div>
<div>

### What large comparisons found
> *"none of the variants can improve upon the standard LSTM architecture significantly, and demonstrate the forget gate and the output activation function to be its most critical components"* <small>— Greff et al. (2017), 5,400 runs</small>

> *"we found GRU to be comparable to LSTM"* <small>— Chung et al. (2014)</small>

<v-click>

* Practice: LSTM or GRU, open forget gate; tune the size, the depth and the regularization — not the cell

</v-click>
</div>
</div>

<span class="refs">Read: [Greff et al., LSTM: A Search Space Odyssey (2017)](https://arxiv.org/abs/1503.04069) · [Chung, Gulcehre, Cho & Bengio (2014)](https://arxiv.org/abs/1412.3555) · [Jozefowicz, Zaremba & Sutskever (2015)](https://proceedings.mlr.press/v37/jozefowicz15.html) · [C. Olah, Understanding LSTM Networks: "Variants on Long Short Term Memory"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)</span>

<!--
Olah's post draws the variants: peephole connections (the gates look at the cell state), coupled forget and input gates
("we only forget when we're going to input something in its place"), and the GRU.

Greff, Srivastava, Koutnik, Steunebrink & Schmidhuber, "LSTM: A Search Space Odyssey" (IEEE TNNLS 2017): eight variants
on three tasks (speech recognition, handwriting recognition, polyphonic music), with random hyperparameter search -
"5400 experimental runs" in their abstract. The quote is from the abstract. Removing the forget gate or the output
activation hurt the most; coupling the input and forget gates or removing peepholes did not hurt.

Chung, Gulcehre, Cho & Bengio (2014): GRU and LSTM on polyphonic music and speech signals; both clearly better than tanh
units, and "we found GRU to be comparable to LSTM".

Jozefowicz, Zaremba & Sutskever (2015): an evolutionary search over more than 10,000 architectures; the best ones were
close to the GRU, some beat LSTM and GRU on some tasks, none on all - and an LSTM with forget bias 1 closed the gap to
the GRU.

Our own comparison table (Section 2) agrees with the practical bullet.
-->

---
zoom: 0.86
---

# Prediction with an Encoder–Decoder

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

<figure>
  <img src="/seq2seq-predict.svg" style="width: 430px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/seq2seq.html">d2l.ai Fig. 10.7.3 Predicting the output sequence token by token using an RNN encoder–decoder</a>
  </figcaption>
</figure>

<br>

* At step $t'$ the decoder's input is its **own** output of step $t'-1$
* Stop at `<eos>`, or after a maximum length

</div>
<div>

<figure>
  <img src="/seq2seq-details.svg" style="width: 420px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/seq2seq.html">d2l.ai Fig. 10.7.2 Layers in an RNN encoder–decoder model</a>
  </figcaption>
</figure>

<br>

* Embedding → recurrent layers → (decoder only) a dense layer over the vocabulary
* The context enters the decoder twice: as its initial state, and concatenated to every input

</div>
</div>

<span class="refs">Read: [d2l.ai 10.7.7 Prediction](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#prediction) · Watch: [J. Starmer, StatQuest: Seq2Seq](https://www.youtube.com/watch?v=L8HKweZIOmg)</span>

<!--
d2l's two remaining figures of 10.7. Left: prediction is Lecture 6's free-running generation - <bos> first, then each
predicted token is fed back, until <eos>. Training used teacher forcing (the true previous token), so at test time the
decoder sees inputs it never saw in training whenever it makes a mistake: exposure bias (Lecture 6).

Right: the layers of the model of the code slide - embedding, two recurrent layers in the encoder; embedding, two
recurrent layers and a dense output layer in the decoder; the encoder's final state initializes the decoder's state,
and its last layer is also concatenated to the decoder's inputs.
-->
