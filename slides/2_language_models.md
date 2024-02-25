# Language Modeling

* Goal: predict the probability of a sentence, e.g.<br>
$p(\mathrm{Deep},\mathrm{learning}, \mathrm{is}, \mathrm{fun}, \mathrm{.})$

* A fundamental task in Natural Language Processing
  * Typing
    * predict the next word
  * Machine translation
    *  <font color="green">dog bites man</font> vs <font color="orange">man bites dog</font>
  * Speech recognition
    * <font color="green">to recognize speech</font> vs <font color="orange">to wreck a nice beach</font>
<br>
<br>
<div style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br><a href="https://c.d2l.ai/odsc2019/slides/Part-4.pdf">Slide by Alex Smola</a></div>

---

# Text Preprocessing

* Sequence data has **long dependency** (very costly)
* Truncate into shorter fragments
* Transform examples into mini-batches with ndarrays
<br>
<br>

<div class="grid grid-cols-[1fr_1fr] gap-3">
<div>
  <figure>
    <img src="/fashion_mnist_dataset_piece.png" style="width: 90px !important;">
  </figure>
</div>
<div>

$\Rightarrow~~$ batch size, width, height, channel
</div>
</div>

<br>
<br>

<div class="grid grid-cols-[1fr_1fr] gap-3">
<div>
 Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
</div>
<div>

$\Rightarrow~~$ batch size, sentence length
</div>
</div>

<div style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br><a href="https://c.d2l.ai/odsc2019/slides/Part-4.pdf">Based on slides by Alex Smola</a></div>

---

# Tokenization

* Basic Idea - map text into sequence of tokens
  * `“Deep learning is fun”` $~~\Rightarrow~~$ `[“Deep”, “learning”, “is”, “fun”, “.”]`
* **Character Encoding** (each character as a token)
  * Small vocabulary
  * Doesn’t work so well (needs to learn spelling)
* **Word Encoding** (each word as a token)
  * Accurate spelling
  * Doesn’t work so well (huge vocabulary = costly multinomial)
* **Byte Pair Encoding** (Goldilocks zone)
  * Frequent subsequences (like syllables)

  <div style="color:#b3b3b3ff; font-size: 11px; position: absolute; right:80px; bottom:5px;"><a href="https://c.d2l.ai/odsc2019/slides/Part-4.pdf">Based on slides by Alex Smola</a></div>

---

# Vocabulary

* Find unique tokens, map each one into a numerical index
  * `“Deep” : 1, “learning” : 2, “is” : 3, “fun” : 4, “.” : 5`
<br>
<br>

<div class="grid grid-cols-[1fr_1fr] gap-16">
<div>

* The frequency of words often obeys a power law distribution
  * Map the tailing tokens, e.g. appears<br> < 5 times, into a special “unknown” token
</div>
<div>
  <figure>
    <img src="/nlp_vocabulary.png" style="width: 400px !important;">
  </figure>
</div>
</div>


<div style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br><a href="https://c.d2l.ai/odsc2019/slides/Part-4.pdf">Based on slides by Alex Smola</a></div>

---

# Minibatch Generation

<div>
  <figure>
    <img src="/text_example.png" style="width: 450px !important;">
  </figure>
</div>
<br>
<br>
<br>
<br>
<br>

<div style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br><a href="https://c.d2l.ai/odsc2019/slides/Part-4.pdf">Based on slides by Alex Smola</a></div>

---

# Learning Language Models

* Suppose that we tokenize text data at the word level
  * Let’s start by applying basic probability rules:<br>
$P(x_1, x_2, \ldots, x_T) = \prod\limits_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1})$

  * For example, 
the probability of a text sequence containing four words would be given as:<br>

$\begin{aligned}&P(\textrm{deep}, \textrm{learning}, \textrm{is}, \textrm{fun}) = \\
=&P(\textrm{deep}) P(\textrm{learning}  \mid  \textrm{deep}) P(\textrm{is}  \mid  \textrm{deep}, \textrm{learning}) P(\textrm{fun}  \mid  \textrm{deep}, \textrm{learning}, \textrm{is})\end{aligned}$

---

# Markov Models and $n$-grams

* A distribution over sequences satisfies the Markov property of first order if<br> $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$
  * Higher orders correspond to longer dependencies. This leads to a number of approximations that we could apply to model a sequence:

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

The probability formulae that involve one, two, and three variables are typically referred to as **unigram**, **bigram**, and **trigram** models, respectively.

---

# Word Frequency

* The probability of words can be calculated from the relative word frequency of a given word in the training dataset
  * For example, the estimate $\hat{P}(\textrm{deep})$ can be calculated as the
probability of any sentence starting with the word "deep"
  * Moving on, we could attempt to estimate

$$\hat{P}(\textrm{learning} \mid \textrm{deep}) = \frac{n(\textrm{deep, learning})}{n(\textrm{deep})},$$

where $n(x)$ and $n(x, x')$ are the number of occurrences of singletons
and consecutive word pairs, respectively

* The problem is that for most of $\geq 2$ words combination, the frequency will be negligible w.r.t. the frequency of single words

---

# Laplace Smoothing

#### A common strategy is to perform some form of **Laplace smoothing**.
The solution is to
add a small constant to all counts. 
Denote by $n$ the total number of words in
the training set
and $m$ the number of unique words.
This solution helps with singletons, e.g., via<br>

$\hat{P}(x) = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1},$<br>
$\hat{P}(x' \mid x) = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2},$<br>
$\hat{P}(x'' \mid x,x') = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}$

  * Here $\epsilon_1,\epsilon_2$, and $\epsilon_3$ are hyperparameters.
    * Take $\epsilon_1$ as an example:
when $\epsilon_1 = 0$, no smoothing is applied;
when $\epsilon_1$ approaches positive infinity,
$\hat{P}(x)$ approaches the uniform probability $1/m$.

---

# Perplexity

#### How to measure the quality of the language model?
* A good language model is able to predict, with high accuracy, the tokens that come next.
* Consider the following continuations of the phrase "It is raining", as proposed by different language models:
  1. "It is raining outside"
  1. "It is raining banana tree"
  1. "It is raining piouw;kcj pwepoiut"

* We can measure model quality by the cross-entropy loss averaged
over all the $n$ tokens of a sequence:<br>
$\frac{1}{n} \sum\limits_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1)$
  * **Perplexity** is just $\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right)$