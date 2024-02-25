# Tabular data

##### The previously discussed models work with tabular (or fixed-length) data
<br>
<div>
  <figure>
    <img src="/iris_flower_dataset.png" style="width: 680px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Source:
      <a href="https://doi.org/10.1111/j.1469-1809.1936.tb02137.x">R. A. Fisher, The Use of Multiple Measurements in Taxonomic Problem (1936)</a>
    </figcaption>
  </figure>
</div>

<br>
<br>

<div>
  <figure>
    <img src="/titanic_dataset.png" style="width: 680px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Source:
      <a href="https://www.openml.org/search?type=data&id=40945">Titanic dataset</a>
    </figcaption>
  </figure>
</div>

---

# Tabular data
#### Even images are tabular data - every image can be represented as row of features
<br>
<div>
  <center>
  <figure>
    <img src="/mnist_dataset.png" style="width: 700px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right:80px;"><br>Source:
      <a href="https://doi.org/10.3390/app9153169">MNIST dataset from https://doi.org/10.3390/app9153169</a>
    </figcaption>
  </figure>
  </center>
</div>

---

# Tabular data
#### Some models are agnostic to the permutation of the features (fully-connected NN),<br> some doesn't (convolutional NN)
<br>
<div class="grid grid-cols-[2fr_1fr] gap-3">
<div>
  <figure>
    <img src="/mnist_raw.png" style="width: 600px !important;">
  </figure>
</div>
<div>

```python {all}
# raw image data
FC (CNN) test loss: 0.42 (0.17)
FC (CNN) accuracy: 0.88 (0.95)
```
</div>
</div>

<div class="grid grid-cols-[2fr_1fr] gap-3">
<div>
  <figure>
    <img src="/mnist_permuted.png" style="width: 600px !important;">
  </figure>
</div>
<div>

```python {all}
# permuted pixels
FC (CNN) test loss: 0.46 (0.54)
FC (CNN) accuracy: 0.86 (0.83)
```
</div>
</div>
<br>

#### We require that all the models be agnostic to the permutation of the observations!
* This is one of the purpose of the resampling methods

---

# Tabular data

* So far, collect observations $(x_i, y_i)$ for training
* Assume observations are independent and identically distributed (i.i.d.)
* **The order of the data does not matter**

<br>
<br>
<div>
  <center>
  <figure>
    <img src="/fashion_mnist_dataset.png" style="width: 600px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right:190px;"><br>Examples from Fashion-MNIST dataset <a href="https://github.com/zalandoresearch/fashion-mnist">https://github.com/zalandoresearch/fashion-mnist</a>
    </figcaption>
  </figure>
  </center>
</div>

---

# Time matters
<div>
  <center>
  <figure>
    <img src="/meteogram_history.png" style="width: 480px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right:190px;"><br>Meteogram for Moscow from <a href="https://www.meteoblue.com/en/weather/archive/yearcomparison/moscow_russia_524901">https://meteoblue.com</a>
    </figcaption>
  </figure>
  </center>
</div>

---

# Time matters
<div>
  <figure>
    <img src="/eur_rub_exchange_rate.png" style="width: 800px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source: <a href="https://en.wikipedia.org/wiki/File:EUR-RUB_exchange_rate.webp">https://en.wikipedia.org/wiki/File:EUR-RUB_exchange_rate.webp</a>
    </figcaption>
  </figure>
</div>
<br>
<br>
<div>
  <figure>
    <img src="/audio_signal.png" style="width: 800px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right:110px;"><br>Example of audio signal</figcaption>
  </figure>
</div>