# Steps in a Language Model

<div>
  <figure>
    <img src="/step_1.png" style="width: 550px !important;">
  </figure>
</div>
<br>
<br>
<br>

<div style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br><a href="https://c.d2l.ai/odsc2019/slides/Part-4.pdf">Based on slides by Alex Smola</a></div>

---

# Steps in a Language Model

<div>
  <figure>
    <img src="/step_2.png" style="width: 550px !important;">
  </figure>
</div>
<br>
<br>
<br>

<div style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br><a href="https://c.d2l.ai/odsc2019/slides/Part-4.pdf">Based on slides by Alex Smola</a></div>

---

# Steps in a Language Model

<div>
  <figure>
    <img src="/step_3.png" style="width: 550px !important;">
  </figure>
</div>
<br>
<br>
<br>

<div style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br><a href="https://c.d2l.ai/odsc2019/slides/Part-4.pdf">Based on slides by Alex Smola</a></div>

---

# Steps in a Language Model

<div>
  <figure>
    <img src="/step_4.png" style="width: 550px !important;">
  </figure>
</div>
<br>
<br>
<br>

<div style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br><a href="https://c.d2l.ai/odsc2019/slides/Part-4.pdf">Based on slides by Alex Smola</a></div>


---

# Recurrent Neural Networks with Hidden States

<div>
  <figure>
    <img src="/rnn.svg" style="width: 500px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://d2l.ai/chapter_recurrent-neural-networks/rnn.html">d2l.ai fig. 9.4.1</a>
    </figcaption>
  </figure>
</div>
<br>
<br>

* Hidden state update<br>
$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}  + \mathbf{b}_\textrm{h})$

* Observation update<br>
$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}$

---

# Recurrent Neural Networks with Hidden States

<div>
  <figure>
    <img src="/rnn.svg" style="width: 500px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://d2l.ai/chapter_recurrent-neural-networks/rnn.html">d2l.ai fig. 9.4.1</a>
    </figcaption>
  </figure>
</div>
<br>
<br>

<div class="grid grid-cols-[1fr_1fr] gap-16">
<div>

* Hidden state update<br>
$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}  + \mathbf{b}_\textrm{h})$

* Observation update<br>
$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}$
</div>

<div>

#### Compare to MLP:<br>
* $\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{\textrm{xh}} + \mathbf{b}_\textrm{h})$

* $\mathbf{O} = \mathbf{H} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}$
</div>
</div>


---

# Recurrent Neural Networks with Hidden States

* The calculation of $\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ for the hidden state is equivalent to
matrix multiplication of the
concatenation of $\mathbf{X}_t$ and $\mathbf{H}_{t-1}$
and the
concatenation of $\mathbf{W}_{\textrm{xh}}$ and $\mathbf{W}_{\textrm{hh}}$.
* Though this can be proven mathematically,
in the following we just use a simple code snippet as a demonstration.
* To begin with,
we define matrices `X`, `W_xh`, `H`, and `W_hh`, whose shapes are (3, 1), (1, 4), (3, 4), and (4, 4), respectively.
Multiplying `X` by `W_xh`, and `H` by `W_hh`, and then adding these two products,
we obtain a matrix of shape (3, 4).

---

# Recurrent Neural Networks with Hidden States

```python {all}
X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
tensor([[ 1.2526,  0.0580, -3.3460, -0.2519],
        [-1.3064,  1.4132, -0.1435,  0.3482],
        [ 3.1495,  0.8172,  1.5167, -0.9038]])
```

* Now we concatenate the matrices `X` and `H`
along columns (axis 1),
and the matrices
`W_xh` and `W_hh` along rows (axis 0).
* These two concatenations
result in
matrices of shape (3, 5)
and of shape (5, 4), respectively.
* Multiplying these two concatenated matrices,
we obtain the same output matrix of shape (3, 4)
as above.

```python {all}
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
tensor([[ 1.2526,  0.0580, -3.3460, -0.2519],
        [-1.3064,  1.4132, -0.1435,  0.3482],
        [ 3.1495,  0.8172,  1.5167, -0.9038]])
```

---

# Ex. RNN-Based Character-Level Language Model

* Let's illustrate how RNNs can be used to build a language model
* Let the minibatch size be one, and the sequence of the text be `machine`

<div>
  <figure>
    <img src="/rnn-train.svg" style="width: 500px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://d2l.ai/chapter_recurrent-neural-networks/rnn.html">d2l.ai fig. 9.4.2</a>
    </figcaption>
  </figure>
</div>
<br>

* During the training process, we run a softmax operation on the output from the output layer for each time step, and then use the cross-entropy loss to compute the error between the model output and the target

---

# Gradients

* Long chain of dependencies for backprop
	* Need to keep a lot of intermediate values in memory
	* Butterfly effect style dependencies
	* Gradients can vanish or diverge

* Clipping to prevent divergence<br>

$\mathbf{g}\leftarrow \mathrm{min}\big(1, \frac{\theta}{\lVert \mathbf{g} \rVert} \big)\mathbf{g}$

rescales to gradient of size at most $\theta$

---

# Pros and Cons of a typical RNN architecture

<br>
<div class="grid grid-cols-[4fr_3fr] gap-4">
<div>

### Advantages

* Possibility of processing input of any length
* Model size not increasing with size of input
* Computation takes into account historical information
* Weights are shared across time
</div>
<div>

### Drawbacks
* Computation being slow
* Difficulty of accessing information from a long time ago
* Cannot consider any future input for the current state
</div>
</div>
<br>
<br>
<br>
<br>
<br>

<div style="color:#b3b3b3ff; font-size: 11px; position: absolute;">Source:<a href="https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks"> Recurrent Neural Networks cheatsheet</a></div>

---

# Applications of RNNs


| Type of RNN                    | Illustration | Example                    |
|--------------------------------|--------------|----------------------------|
| One-to-one<br> $T_x = T_y = 1$     |   <img src="/rnn-one-to-one-ltr.png" style="width: 220px !important;">       | Traditional neural network |
| One-to-many<br> $T_x = 1, T_y > 1$ | <img src="/rnn-one-to-many-ltr.png" style="width: 220px !important;">           | Music generation           |
| Many-to-one<br> $T_x > 1, T_y = 1$ | <img src="/rnn-many-to-one-ltr.png" style="width: 220px !important;">           | Sentiment classification           |

<div style="color:#b3b3b3ff; font-size: 11px; position: absolute; right:80px; bottom:5px;">Source:<a href="https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks">Recurrent Neural Networks cheatsheet</a></div>

---

# Applications of RNNs


| Type of RNN                    | Illustration | Example                    |
|--------------------------------|--------------|----------------------------|
| Many-to-many<br> $T_x = T_y$     |   <img src="/rnn-many-to-many-same-ltr.png" style="width: 220px !important;">       | Name entity recognition |
| Many-to-many<br> $T_x \neq T_y$ | <img src="/rnn-many-to-many-different-ltr.png" style="width: 220px !important;">           | Machine translation           |

<div style="color:#b3b3b3ff; font-size: 11px; position: absolute; right:80px; bottom:5px;">Source:<a href="https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks">Recurrent Neural Networks cheatsheet</a></div>