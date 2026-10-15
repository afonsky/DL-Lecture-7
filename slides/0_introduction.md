---
layout: center
---

<center>

# Gated Memory

# Why a vanilla RNN forgets, and the fix we already know
</center>

---
zoom: 0.84
---

# Where Lecture 6 Stopped

<figure>
  <img src="/memory_lecture6.svg" style="width: 760px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; text-align: center">
    Lecture 6: the first of <i>T</i> tokens must be named after the last one. <code>nn.RNN</code> with 64 units, 6 runs per length.
  </figcaption>
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

* "Remember the first symbol": learned up to 40 steps in 5–6 runs of 6; at 100 steps in **1 of 6**
* The trained character RNN: **0.8 per step** back — 1 % of the gradient left after 20 steps

</div>
<div>
<v-click>

* Each step back multiplies the gradient by the **same** $\mathbf{W}_{\textrm{hh}}^\top$ and by $\tanh' \le 1$
* Clipping repairs **exploding** gradients — nothing so far repairs **vanishing** ones

</v-click>
</div>
</div>

<span class="refs">Read: [Bengio, Simard & Frasconi (1994)](https://doi.org/10.1109/72.279181) · [Deep Learning, §10.7 The Challenge of Long-Term Dependencies](https://www.deeplearningbook.org/contents/rnn.html) · Watch: [A. Ng, Sequence Models, Week 1: "Vanishing Gradients with RNNs"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
Two minutes of recap; the figure is Lecture 6's own (scripts/lecture6/, memory task, seeds 0-5).

The task: the first input is one of 8 symbols, then T-1 random filler tokens, and only after the last one the network
must name the first symbol. It is the smallest possible long-term dependency: the loss sits at step T, the information
at step 1. Ng's version: "The cat, which already ate ..., was full" - the verb depends on a noun many words back.

The numbers: Lecture 6 measured the gradient of a last-step loss with respect to earlier hidden states of the trained
128-unit character model - about 0.8 per step back, 1 % at 20 steps, 0.00002 at 50. The memory-task network itself
loses even more at initialization: about 0.6 per step (measured today with the same method, see the gradient slide of
Section 2).

d2l opens the chapter with the same diagnosis: "Recall that in Chapter 9 we dealt with exploding
gradients by applying a blunt gradient clipping heuristic. Despite the efficacy of this hack, it leaves open the problem
of vanishing gradients."

Transition: we have met vanishing gradients before - through depth, in Lecture 3 - and Lecture 5 had an answer.
-->

---
zoom: 0.9
---

# The Fix We Already Know: Make the Identity Easy

<div class="grid grid-cols-2 gap-12">
<div>

### Lecture 5: through depth
$$\mathbf{x}_{l+1} = \mathbf{x}_l + F(\mathbf{x}_l)$$

* A residual block that does nothing is the **default**
* The gradient passes the "+" **unchanged**

### Lecture 6: through time
$$\mathbf{h}_t = \tanh(\mathbf{W}_{\textrm{hh}}\mathbf{h}_{t-1} + \mathbf{W}_{\textrm{hx}}\mathbf{x}_t + \mathbf{b})$$

* Every step **rewrites** the whole state

</div>
<div>
<v-click>

### LSTM, 1997: a memory that is *added to*
$$\mathbf{c}_t = \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

* Copying the memory is the default: $\partial \mathbf{c}_t / \partial \mathbf{c}_{t-1} = \mathbf{I}$ — the *"constant error carousel"*

</v-click>
<v-click>

### LSTM, 2000: … that can also forget
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

* $\mathbf{f}_t \approx 1$: keep, $\mathbf{f}_t \approx 0$: erase — learned, at every step, for every unit

</v-click>
</div>
</div>

<v-click>

#### Highway networks (2015) moved the LSTM's gates into deep feedforward networks; ResNet dropped the gates and kept the identity.

</v-click>

<span class="refs">Read: [Hochreiter & Schmidhuber (1997)](https://doi.org/10.1162/neco.1997.9.8.1735) · [Gers, Schmidhuber & Cummins, Learning to Forget (2000)](https://doi.org/10.1162/089976600300015015) · [Srivastava, Greff & Schmidhuber, Training Very Deep Networks (2015)](https://arxiv.org/abs/1507.06228) · [d2l.ai 10.1](https://d2l.ai/chapter_recurrent-modern/lstm.html)</span>

<!--
The whole lecture in one slide. Write the three equations on the board if you can.

Left: Lecture 5's take-away was "make the identity easy": a residual block computes x + F(x), so with F = 0 it passes
its input through, and in the backward pass the "+" passes the gradient through as well. Lecture 6's recurrent layer
is the opposite: every step pushes the whole state through a matrix and a tanh, and the backward pass multiplies by
W_hh^T and tanh' at every step.

Right, first click: Hochreiter & Schmidhuber (1997) gave each unit a memory cell c that is ADDED to. With nothing
written (i = 0), the cell is copied unchanged, and the derivative of c_t with respect to c_{t-1} is the identity. They
called it the constant error carousel: the error signal circulates through the self-connection of weight 1 without
shrinking or growing. d2l: "the memory cell avoids the vanishing gradient problem by keeping values in each memory
cell's internal state cascading along a recurrent edge with weight 1 across many successive time steps."

Second click: a memory that only accumulates fills up. Gers, Schmidhuber & Cummins (2000) added the forget gate f, a
number between 0 and 1 per unit, computed at every step: multiply the old memory by f before adding the new content.
This is the LSTM everybody uses today, and the 1997 cell is the special case f = 1.

Third click, the history of the idea: Highway networks (Srivastava, Greff & Schmidhuber, 2015) were explicitly
"inspired by Long Short-Term Memory recurrent networks" - gated skip connections in a feedforward network, which
trained 100-layer networks. ResNet, the same year, used the plain identity. So the idea went from time (1997) to depth
(2015) - and today we bring it back to time.

Notation: these slides use column vectors here (W h) like Lecture 6's backup and Olah; from the next slide on, d2l's
row form (X W) with capital letters for minibatches.
-->

---
zoom: 0.82
---

# Gates: Learned Soft Switches

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

### A gate is a layer with a sigmoid
$$\mathbf{G}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xg}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hg}} + \mathbf{b}_\textrm{g}) \in (0, 1)^{n \times h}$$

* Used by **elementwise** multiplication: $\mathbf{G}_t \odot \mathbf{V}$
* $0$ blocks, $1$ passes, in between scales:

$$\begin{bmatrix} 0.99 \\ 0.02 \\ 0.5 \end{bmatrix} \odot \begin{bmatrix} 3 \\ -1 \\ 4 \end{bmatrix} = \begin{bmatrix} 2.97 \\ -0.02 \\ 2 \end{bmatrix}$$

<v-click>

### Not dropout <small>*(Lecture 3)*</small>
* Dropout: a **random** 0/1 mask, training only
* A gate: **computed** from the input and the state, between 0 and 1, always on — and **differentiable**, so it is learned by backpropagation

</v-click>
</div>
<div>

<figure>
  <img src="/lstm-0.svg" style="width: 390px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
    <a href="https://d2l.ai/chapter_recurrent-modern/lstm.html">d2l.ai Fig. 10.1.1 Computing the input gate, the forget gate, and the output gate in an LSTM model</a>
  </figcaption>
</figure>

<br>

* An LSTM computes **three** gates at every step, a GRU two
* Each is a fully connected layer with its own weights, on the same inputs: $\mathbf{X}_t$ and $\mathbf{H}_{t-1}$

</div>
</div>

<span class="refs">Read: [d2l.ai 10.1.1 Gated Memory Cell](https://d2l.ai/chapter_recurrent-modern/lstm.html#gated-memory-cell) · Watch: [J. Starmer, StatQuest: LSTM, Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) · [A. Ng, Sequence Models, Week 1: "Gated Recurrent Unit (GRU)"](https://www.coursera.org/learn/nlp-sequence-models)</span>

<!--
The one new operation of today. A gate is Lecture 3's fully connected layer with a sigmoid, computed from the same two
inputs as the recurrent layer - the current input X_t and the previous hidden state H_{t-1} - and it has as many
outputs as there are hidden units. Its values are used as multipliers, element by element (the Hadamard product,
written with a circled dot).

The example: three units. The first passes almost everything (0.99), the second blocks (0.02), the third halves.
Because the gate is recomputed at every step, it can be open at one step and closed at the next: the network decides,
from what it reads, what to keep.

Why a sigmoid: its values lie in (0, 1), and it is differentiable, so the gates are trained by ordinary backpropagation
like every other weight. A hard 0/1 switch would have zero gradient almost everywhere. StatQuest's LSTM video reads the
sigmoid's output as the percentage of a value that is kept.

Dropout (Lecture 3) also multiplies activations by a mask; the differences are on the slide. A useful mental picture
nevertheless: a gate is a mask that the network computes for itself.

Figure: d2l's first LSTM figure - the three gates of an LSTM, each a fully connected layer with sigmoid activation (FC
box with sigma), all reading X_t and H_{t-1}. The next slide shows what the gates are multiplied with.
-->
