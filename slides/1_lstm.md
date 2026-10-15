---
layout: center
---

<center>

# LSTM and GRU

# Memory cells and gates; which one to use
</center>

---
zoom: 0.82
---

# The LSTM Cell

<div class="grid grid-cols-[10fr_11fr] gap-8">
<div>

<figure>
  <img src="/lstm-3.svg" style="width: 420px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/lstm.html">d2l.ai Fig. 10.1.4 Computing the hidden state in an LSTM model</a>
  </figcaption>
</figure>

<br>

<v-click at="2">

* **Two states**: the memory $\mathbf{C}_t$ stays inside; only $\mathbf{H}_t$ goes out — to the output layer and to the next step's gates
* Four fully connected layers instead of one: **4×** the parameters of an RNN

</v-click>
</div>
<div>

$$
\begin{aligned}
\mathbf{I}_t, \mathbf{F}_t, \mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{x}\cdot} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{h}\cdot} + \mathbf{b}_{\cdot}) \\
\tilde{\mathbf{C}}_t &= \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xc}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hc}} + \mathbf{b}_\textrm{c}) \\
\mathbf{C}_t &= \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t \\
\mathbf{H}_t &= \mathbf{O}_t \odot \tanh(\mathbf{C}_t)
\end{aligned}
$$

<v-click at="1">

* **Input node** $\tilde{\mathbf{C}}_t$: Lecture 6's recurrent layer — the new content
* **Forget gate** $\mathbf{F}_t$: how much of the old memory stays
* **Input gate** $\mathbf{I}_t$: how much of the new content is written
* **Output gate** $\mathbf{O}_t$: how much of the memory is shown

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.1 Long Short-Term Memory](https://d2l.ai/chapter_recurrent-modern/lstm.html) · [C. Olah, Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) · Watch: [J. Starmer, StatQuest: LSTM](https://www.youtube.com/watch?v=YCzL96nL7j0) · [A. Ng, Sequence Models, Week 1: "Long Short Term Memory (LSTM)"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
d2l's equations; the first line is shorthand for d2l's three gate equations - three fully connected layers with their
own weights (W_xi, W_hi, b_i for the input gate, and so on), all with a sigmoid. The figure is the last of d2l's four
LSTM figures and contains all of them. Read it from left to right:
- three gates (sigma boxes) and the input node (tanh box), each a fully connected layer on X_t and H_{t-1};
- the memory C_{t-1} enters at the top left, is multiplied by the forget gate, and the input gate times the input node
  is added: that is C_t, which leaves at the top right;
- the hidden state H_t is tanh(C_t), scaled by the output gate.

The name (d2l): a simple RNN has long-term memory in its weights, which change slowly during training, and short-term
memory in its activations, which change at every step; "The LSTM model introduces an intermediate type of storage via
the memory cell."

Extreme cases worth saying out loud:
- F = 1, I = 0: the memory is copied unchanged - the identity path of the previous slide.
- F = 0, I = 1: the memory is replaced by the new content - close to a vanilla RNN.
- O = 0: the cell keeps its content but shows nothing; d2l: "a memory cell can accrue information across many time
  steps without impacting the rest of the network ... and then suddenly impact the network at a subsequent time step as
  soon as the output gate flips from values close to 0 to values close to 1."

Parameters: four blocks of (d x h input weights, h x h recurrent weights, h biases). For our 28-character vocabulary and
256 units: 4 x (7,168 + 65,536 + 512) = 292,864 in nn.LSTM (which keeps two bias vectors, like nn.RNN), against
73,216 for nn.RNN.

Ng and StatQuest use other letters (Ng: Gamma_u, Gamma_f, Gamma_o and c<t>, a<t>; Olah: f_t, i_t, o_t, C_t, h_t); it
is always the same cell. d2l's exercise 4 is a good question for the class: the input node is already in (-1, 1), so
why is there another tanh in H_t? (Because C_t is a sum over many steps and can grow beyond 1.)
-->

---
zoom: 0.92
---

# LSTM in PyTorch

<div class="grid grid-cols-[1fr_1fr] gap-8">
<div>

### From scratch <small>(d2l 10.1.2)</small>
```python {all|3-6|7|8|all}
H, C = zeros(batch, h), zeros(batch, h)
for X in inputs:                       # one time step after another
    I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
    F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
    O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
    C_tilde = torch.tanh(X @ W_xc + H @ W_hc + b_c)
    C = F * C + I * C_tilde            # keep, then add
    H = O * torch.tanh(C)
    outputs.append(H)
```

* The loop of Lecture 6's `RNNScratch`, with the new state update
* `*` is the elementwise product $\odot$

</div>
<div>

### The library layer
```python {all|3-4|5-6|all}
lstm = nn.LSTM(input_size=28, hidden_size=256)
X = torch.randn(32, 2, 28)          # (steps, batch, features)
H, (h_n, c_n) = lstm(X)             # the state is a pair
H.shape, h_n.shape, c_n.shape  # [32, 2, 256], [1, 2, 256] ×2
lstm.weight_hh_l0.shape        # [1024, 256]: i, f, g, o stacked
sum(p.numel() for p in lstm.parameters())       # 292864
```

<v-click>

* A drop-in replacement: in Lecture 6's `CharRNN`, change `nn.RNN` to `nn.LSTM` — the loop stays
* Carrying the state between windows: `detach()` **both** `h_n` and `c_n`
* `nn.GRU`: one state, 3 blocks of weights

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.1.2 Implementation from Scratch](https://d2l.ai/chapter_recurrent-modern/lstm.html#implementation-from-scratch) · Docs: [torch.nn.LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) · Watch: [S. Raschka, L15.5 Long Short-Term Memory](https://www.youtube.com/watch?v=k6fSgUaWUF8) · [A. Karpathy, CS231n 2016, Lecture 10](https://www.youtube.com/watch?v=yCC09vCHzF8)</span>

<!--
Left: d2l's forward method, shortened (no device handling; the 12 parameters are created as in d2l's LSTMScratch,
normal with sigma 0.01, zero biases). Compare with Lecture 6's RNNScratch: there, the loop body was one line,
state = tanh(X @ W_xh + state @ W_hh + b_h). Here it is four layers that read the same X and H, and two lines that
combine them. The line C = F * C + I * C_tilde is the reason for the lecture.

Right: the printed values are real (scripts/lecture7/slide_code.py, PyTorch 2.14).
- The state is a pair (h, c). H holds the hidden state after every step, as for nn.RNN; c is never an output of the
  layer except as the final c_n.
- PyTorch stores the four blocks stacked in one matrix: weight_ih_l0 is (4h, d), weight_hh_l0 is (4h, h), rows in the
  order input gate, forget gate, cell candidate ("g"), output gate. We need this order on the slide after next, to set
  the forget gate's bias.
- 292,864 = 4 x 73,216, the nn.RNN layer of the same size.

Drop-in: Lecture 6's CharRNN with nn.LSTM is exactly the lstm_h256 run of today's comparison table; the training loop,
clipping included, is unchanged. With a carried state (Lecture 6's truncated BPTT slide), the state is a tuple:
state = tuple(s.detach() for s in state).

Karpathy's CS231n lecture writes the four gates as one matrix product followed by a split (in the order i, f, o, g) -
the same idea as PyTorch's stacked layout (i, f, g, o).
-->

---
zoom: 0.84
---

# How Far Back Can It Remember? — Measured Again

<figure>
  <img src="/memory_gated.svg" style="width: 860px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; text-align: center">
    Lecture 6's task: name the first of <i>T</i> tokens after the last one. 64 units, Adam, clipping, at most 10,000 steps; "learned" = 99 % accuracy. RNN, LSTM with bias 0 and 1: Lecture 6's runs.
  </figcaption>
</figure>

<div class="grid grid-cols-3 gap-6" style="font-size: 0.8em">
<div>

**PyTorch's defaults: worse than the RNN.** Every gate starts near $\sigma(0) = 0.5$, so the memory halves at every step: the LSTM learns 20 steps, the GRU 30

</div>
<div>
<v-click>

**Open the gate.** Forget-gate bias 3: 80 steps in 6 of 6 runs, 100 in 5 of 6. GRU with update-gate bias 3: **200 steps** in every run, in under 1,000 training steps

</v-click>
</div>
<div>
<v-click>

**Lecture 5's lesson, once more**: the identity must be the **starting point**, not something to learn. Two lines of code: backup "Initializing the Gates"

</v-click>
</div>
</div>

<span class="refs">Read: [Jozefowicz, Zaremba & Sutskever (2015)](https://proceedings.mlr.press/v37/jozefowicz15.html) · [Tallec & Ollivier (2018)](https://arxiv.org/abs/1804.11188) · [Hochreiter & Schmidhuber (1997)](https://doi.org/10.1162/neco.1997.9.8.1735) (long-delay experiments) · Watch: [J. Starmer, StatQuest: LSTM](https://www.youtube.com/watch?v=YCzL96nL7j0)</span>

<!--
Measured for this lecture (scripts/lecture7/experiments.py memory <cell> <seed>), with Lecture 6's code, task and budget:
8 symbols, filler from 8 other symbols, nn.LSTM / nn.GRU(16, 64) + nn.Linear(64, 8) on the last step, Adam with learning
rate 0.001, clipping at 1, batches of 64 fresh sequences, at most 10,000 steps; "learned" = 99 % accuracy on 2,000
held-out sequences. The RNN line and the LSTM lines with bias 0 and 1 are Lecture 6's runs (6, 3 and 3 seeds,
results/lecture6/); new today: LSTM with forget-gate bias 3 (6 seeds), GRU with PyTorch's initialization and with
update-gate bias 3 (3 seeds each), lengths up to 200 for the two "open" cells.

Runs that learn it (median training steps of those that do):
- RNN (Lecture 6): T = 20-40: 5 of 6 (300 / 4,200 / 3,800 steps); 50-60: 2 of 6; 80-100: 1 of 6.
- LSTM, bias 0: up to T = 20 (3 of 3, 1,900-5,000 steps); none from 30 on.
- LSTM, forget bias 1: up to T = 30 (3 of 3, 3,900-8,500 steps); none from 40 on.
- GRU, bias 0: up to T = 30 (3 of 3, 5,000-6,400 steps); none from 40 on.
- LSTM, forget bias 3: 6 of 6 up to T = 80, within 100-400 steps; T = 100: 5 of 6; T = 150: 3 of 6; T = 200: 0 of 6.
- GRU, update bias 3: 3 of 3 at every length up to 200; T = 200 within 700-900 steps.

Why the defaults fail: PyTorch draws all weights and biases from U(-1/sqrt(64), 1/sqrt(64)), so at the start every gate
is about 0.5. The forget gate (LSTM) or the update gate (GRU) then multiplies the memory by 0.5 per step: 0.5^30 = 10^-9.
Jozefowicz et al. (2015): "this initialization effectively sets the forget gate to 0.5. This introduces a vanishing
gradient with a factor of 0.5 per timestep". The gated cells are then worse than the vanilla RNN, whose gradient
shrinks by about 0.6 per step at this initialization (gradient slide). Bias 1 (Keras' default) gives 0.73 per step:
30 steps. Bias 3 gives 0.95.

Why the GRU does better than the LSTM at 150-200 steps: in the GRU, keeping and writing are coupled - an update gate of
0.95 also means that only 5 % of the new input is written. The LSTM's input gate still starts at 0.5, so 200 filler
tokens keep being written into the memory and bury the one symbol. Checked (config lstm_f3i, 3 seeds): forget-gate
bias 3 AND input-gate bias -3, as chrono initialization does. T = 150: 3 of 3 runs (3,300-4,100 steps), against 3 of 6
with the input gate at 0.5; T = 200: 1 of 3 (7,900 steps), against 0 of 6. It helps, but the GRU, whose single gate
makes keeping and not-writing the same decision, is still far ahead.

Practical rule: with nn.LSTM, set the forget-gate bias to 1 or more (two lines, backup); for long dependencies, more.
Hochreiter & Schmidhuber (1997), abstract: LSTM "can learn to bridge minimal time lags in excess of 1000 discrete-time
steps" - with their own training procedure (and a truncated gradient) on tasks of this kind.
-->

---
zoom: 0.84
---

# Why: The Gradient Along the Memory — Measured

<figure>
  <img src="/gradflow.svg" style="width: 860px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; text-align: center">
    Size of the gradient of a loss at the last step with respect to the memory <b>c</b> (LSTM) or the state <b>h</b> (RNN, GRU) <i>k</i> steps earlier; geometric mean over 512 sequences.
  </figcaption>
</figure>

<div class="grid grid-cols-3 gap-6" style="font-size: 0.8em">
<div>

**(a) The gate decides.** Gates at 0.5: about 0.6 per step, like the RNN. Bias 3: **0.98** along the LSTM's memory — after 100 steps the gradient is still there

</div>
<div>
<v-click>

**(b) And it is learned.** Trained on 32-character windows, the language models **forget fast**: 0.83 per step for the LSTM's memory, 0.75 for the GRU

</v-click>
</div>
<div>
<v-click>

**A gate is a choice, not a guarantee**: the network keeps what its task rewards. Lecture 6's 0.8 was the RNN's only option

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 10.1.1.4 Memory Cell Internal State](https://d2l.ai/chapter_recurrent-modern/lstm.html#memory-cell-internal-state) · [Karpathy, Johnson & Fei-Fei (2015)](https://arxiv.org/abs/1506.02078) (gate saturation in trained LSTMs) · [CS231n 2017, Lecture 10 slides](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf)</span>

<!--
Measured for this lecture (scripts/lecture7/experiments.py gradflow), with Lecture 6's method: unroll the cell step by
step (nn.LSTMCell / nn.GRUCell / nn.RNNCell with the weights of the library layer - checked to give the same states),
put a cross-entropy loss on the last of 100 steps only, and record the norm of the gradient with respect to every
earlier memory c_t (LSTM) or state h_t (RNN, GRU), relative to the last step; geometric mean over 512 sequences. "Per
step" is the average factor over the first 20 steps back.

(a) The networks of the memory task at initialization (16 inputs, 64 units, the biases of the memory slide), on its
sequences: RNN 0.57 per step; LSTM memory with PyTorch's gates 0.61, with forget bias 1 0.83, with forget bias 3 0.98;
GRU 0.63, with update bias 3 0.955. After 100 steps that is 10^-20 for the RNN, 10^-6 for bias 1, and about 1 for bias
3 (the LSTM line even rises slightly: some units' memory grows). This is the memory slide's ranking, in the same order:
the cells whose gradient reaches step 1 at the start are the ones that learn long delays in a few hundred steps. One
exception: the RNN starts at 0.57 and still learned 40 steps in 5 of 6 runs - with Adam a tiny gradient is not a tiny
step, and Lecture 6's review found that the runs that succeed raise this factor to about 1.0 during training.

(b) The trained language models of the comparison table (256 units, seed 0), on the validation text: RNN 0.87, LSTM
memory 0.83 (hidden state 0.82), GRU 0.75 per step. The gated models forget FASTER than the RNN - because they can:
their training windows were 32 characters long, so nothing older than 32 characters was ever rewarded, and the forget
and update gates learned to clear the memory. The RNN has no such switch; its 0.87 is a property of its weights, not a
decision. (Lecture 6's 128-unit RNN: 0.80.)

So the honest statement is not "LSTMs do not forget" but "LSTMs can keep a gradient alive when the task needs it -
and they need to start in that position". Karpathy, Johnson & Fei-Fei (2015) looked inside trained LSTMs and found
gates that are almost always open or almost always closed ("left- or right-saturated") for some cells, and in between
for others: the cells specialize in long and short time scales.
-->
