# Working with Sequences
* We focus on an **ordered list of feature vectors** $\mathbf{x}_1, \dots, \mathbf{x}_T$,
where each feature vector $\mathbf{x}_t$ is
indexed by a time step $t \in \mathbb{Z}^+$
lying in $\mathbb{R}^d$
* Sequence can be:
  * Entire long stream of collected data
    * Sensor readings
      * Climate data, earthquake data, ...
  * (More often) List of subsequences of some predetermined length
      * Sensor outputs information in specific portions

---

# Working with Sequences
#### Consider Financial Times Stock Exchange Index (FTSE 100) Index
<br>
<div>
<center>
  <figure>
    <img src="/ftse100.png" style="width: 500px !important;">
  </figure>
</center>   
</div>
<br>

#### At each **time step** $t \in \mathbb{Z}^+$, we observe the price, $x_t$, of the index at that time

---

# Autoregressive Models
<div>
  <figure>
    <img src="/ftse100.png" style="width: 250px !important; position: absolute; right:50px; top:50px;">
  </figure>  
</div>
<br>
<br>
<br>
<br>

* Suppose that a trader would like to make short-term trades using only the history of prices
* Thus the trader is interested in knowing the probability distribution $P(x_t \mid x_{t-1}, \ldots, x_1)$ over prices that the index might take in the subsequent time step
* It is more realistic to estimate the conditional expectation $\mathbb{E}[(x_t \mid x_{t-1}, \ldots, x_1)]$<br> would be to apply a linear regression model
  * Such models that regress the value of a signal
on the previous values of that same signal
are naturally called **autoregressive models**

---

# Autoregressive Models

* The problem with autoregressive models is that the number of inputs,
$x_{t-1}, \ldots, x_1$ depends on $t$
  * Or the number of inputs increases with the amount of data that we encounter
  * Each example has a different number of features
* A few strategies to solve this problem:
  1. Consider some window of length $\tau$ and only use $x_{t-1}, \ldots, x_{t-\tau}$ observations
      * This allows us to train any linear model that requires fixed-length vectors as inputs
  2. Develop models that maintain a summary $h_t$ of the past observations and at the same time update $h_t$ in addition to the prediction $\hat{x}_t$
      * These models estimate not only $x_t$ with $\hat{x}_t = P(x_t \mid h_{t})$ but also updates of the form $h_t = g(h_{t-1}, x_{t-1})$
      * Since $h_t$ is never observed, these models are called **latent autoregressive models**

---

# Latent autoregressive model
<br>

<div>
  <figure>
    <img src="/sequence-model.svg" style="width: 350px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://d2l.ai/chapter_recurrent-neural-networks/sequence.html">d2l.ai fig. 9.1.2</a>
    </figcaption>
  </figure>
</div>
<br>
<br>

#### To construct training data from historical data, one typically creates examples by sampling windows randomly. In general, we do not expect time to stand still. However, we often assume that while the specific values of might change, the dynamics according to which each subsequent observation is generated given the previous observations do not. Statisticians call dynamics that do not change **[stationary](https://en.wikipedia.org/wiki/Stationary_process)**.

---

# Sequence Models

#### Mostly used by **language models**
* Tokenization - a way to sample sequences
* Ability to estimate joint probability of an entire sequence:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1)$$

#### Thus, sequence model can decompose the joint density  of a sequence $p(x_1, \ldots, x_T)$ into the product of conditional densities

---

# Markov Models

<div>
  <figure>
    <img src="/Markov_portrait.jpg" style="width: 150px !important; position: absolute; right:50px; top:10px;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right:50px; top:190px;"><br>Andrey Markov (Sr)<br> Image credit:<br>
      <a href="https://bigenc.ru/c/markov-andrei-andreevich-f6b638">St. Petersburg Branch of the Archive of the RAS</a>
  </figcaption>
  </figure>  
</div>
<br>
<br>
<br>
<br>

* Suppose, the model used only $\tau$ previous time steps
* **[Markov condition](https://en.wikipedia.org/wiki/Causal_Markov_condition)**: the future is conditionally independent of the past,<br>
given the recent history
* When $\tau = 1$, we say that the data is characterized by a first-order Markov model:
$P(x_1, \ldots, x_T) = P(x_1) \prod\limits_{t=2}^T P(x_t \mid x_{t-1})$
* When $\tau = k$, we say that the data is characterized
by a $k^{\textrm{th}}$-order Markov model

---

# The Order of Decoding

* Left-to-right (standard):<br> $P(x_1, \ldots, x_T) = P(x_1) \prod\limits_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1)$

* Right-to-left:<br> $P(x_1, \ldots, x_T) = P(x_T) \prod\limits_{t=T-1}^1 P(x_t \mid x_{t+1}, \ldots, x_T)$
<br>
<br>

#### Which order to choose?
* Choose based on the task:
  * (Language models) Consider read direction
  * Mind the causality
    * Future events cannot influence the past