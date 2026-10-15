<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import W from './charRnnWeights.json'

// The character RNN trained for this lecture (scripts/lecture6/experiments.py charrnn sgd_h128 0), run in the
// browser: h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h), o_t = h_t W_hq + b_q. One-hot inputs, so x_t W_xh is one
// row of W_xh. 23,708 numbers; nothing is fetched from the network.
const H: number = W.hidden
const vocab: string[] = W.vocab
const V = vocab.length
const stoi: Record<string, number> = Object.fromEntries(vocab.map((ch, i) => [ch, i]))
const shown = vocab.map((ch, i) => ({ ch, i })).filter(t => t.ch !== '<unk>')   // 27 tokens that occur in the book

function stepState(h: Float32Array, tok: number) {
  const out = new Float32Array(H)
  const wx = W.W_xh[tok] as number[]
  for (let j = 0; j < H; j++) {
    let s = wx[j] + (W.b_h as number[])[j]
    for (let i = 0; i < H; i++)
      s += h[i] * (W.W_hh as number[][])[i][j]
    out[j] = Math.tanh(s)
  }
  return out
}
function outputLogits(h: Float32Array) {
  const o = new Float32Array(V)
  for (let k = 0; k < V; k++) {
    let s = (W.b_q as number[])[k]
    for (let i = 0; i < H; i++)
      s += h[i] * (W.W_hq as number[][])[i][k]
    o[k] = s
  }
  return o
}
function softmax(o: Float32Array, temperature: number) {
  const p = new Float32Array(V)
  let best = 0
  for (let k = 1; k < V; k++) {
    if (o[k] > o[best])
      best = k
  }
  if (temperature === 0) {                       // greedy decoding: all the mass on the most likely token
    p[best] = 1
    return p
  }
  let z = 0
  for (let k = 0; k < V; k++) {
    p[k] = Math.exp((o[k] - o[best]) / temperature)
    z += p[k]
  }
  for (let k = 0; k < V; k++)
    p[k] /= z
  return p
}
const clean = (s: string) => s.toLowerCase().replace(/[^a-z]+/g, ' ')   // the preprocessing of the training text

const temperatures = [0, 0.5, 0.8, 1, 1.5]
const prefix = ref('the time traveller ')
const generated = ref('')
const temperature = ref(0.8)
const maxGenerated = 200
let rng = 1

function random() {                              // mulberry32: the same clicks give the same text every time
  rng = (rng + 0x6D2B79F5) | 0
  let t = Math.imul(rng ^ (rng >>> 15), 1 | rng)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

// read the whole text once, left to right: the final hidden state, the next-character logits and the perplexity
const run = computed(() => {
  const text = clean(prefix.value) + generated.value
  let h = new Float32Array(H)
  let nll = 0
  let n = 0
  for (let t = 0; t < text.length; t++) {
    const tok = stoi[text[t]]
    if (t > 0) {
      const p = softmax(outputLogits(h), 1)
      nll -= Math.log(Math.max(p[tok], 1e-12))
      n++
    }
    h = stepState(h, tok)
  }
  return { text, h, logits: outputLogits(h), perplexity: n > 0 ? Math.exp(nll / n) : Number.NaN, n }
})
const probs = computed(() => softmax(run.value.logits, temperature.value))
const top = computed(() => shown.map(t => ({ ...t, p: probs.value[t.i] })).sort((a, b) => b.p - a.p).slice(0, 3))
const maxP = computed(() => Math.max(...shown.map(t => probs.value[t.i])))

function sample() {
  const p = probs.value
  let u = random()
  for (let k = 0; k < V; k++) {
    u -= p[k]
    if (u <= 0)
      return k
  }
  return top.value[0].i
}
function generate(count: number) {
  for (let c = 0; c < count && generated.value.length < maxGenerated; c++)
    generated.value += vocab[sample()]            // run and probs are recomputed after every character
}
function reset() {
  generated.value = ''
  rng = 1
}
watch(prefix, reset)

const label = (ch: string) => (ch === ' ' ? '␣' : ch)
const pct = (p: number) => (p >= 0.0995 ? `${Math.round(100 * p)} %` : `${(100 * p).toFixed(1)} %`)
function cellColor(v: number) {                  // diverging blue - gray - red, as in the lecture's figures
  const mid = [240, 239, 236]
  const end = v < 0 ? [42, 120, 214] : [227, 73, 72]
  const a = Math.min(1, Math.abs(v))
  return `rgb(${mid.map((m, i) => Math.round(m + a * (end[i] - m))).join(',')})`
}
</script>

<template>
  <div class="cr">
    <div class="cr-controls">
      <span class="cr-label">prefix</span>
      <input v-model="prefix" class="cr-input" maxlength="60" spellcheck="false" aria-label="prefix">
      <span class="cr-label cr-gap">temperature</span>
      <button v-for="t in temperatures" :key="t" :class="{ 'cr-sel': temperature === t }" @click="temperature = t">{{ t === 0 ? '0 (greedy)' : t }}</button>
      <button class="cr-gap cr-act" @click="generate(1)">+1 character</button>
      <button class="cr-act" @click="generate(60)">+60</button>
      <button @click="reset">reset</button>
    </div>

    <div class="cr-text">
      <span>{{ run.text.slice(0, run.text.length - generated.length) }}</span><span class="cr-gen">{{ generated }}</span><span class="cr-caret">▏</span>
    </div>
    <div class="cr-meta">
      <span v-if="run.n > 0">perplexity of this text under the model: <b>{{ run.perplexity.toFixed(1) }}</b> <span class="cr-muted">(28 = uniform guessing; 4.9 on the held-out part of the book)</span></span>
      <span v-else class="cr-muted">type at least two characters</span>
    </div>

    <div class="cr-panels">
      <div>
        <div class="cr-head">
          P(next character | text so far) at temperature {{ temperature }} &nbsp;·&nbsp;
          <span v-for="(t, i) in top" :key="t.ch"><b>{{ label(t.ch) }}</b> {{ pct(t.p) }}<span v-if="i < 2">, </span></span>
        </div>
        <div class="cr-bars">
          <div v-for="t in shown" :key="t.ch" class="cr-col" :title="`${label(t.ch)}: ${pct(probs[t.i])}`">
            <div class="cr-track">
              <div class="cr-bar" :style="{ height: `${Math.max(1, 100 * probs[t.i] / maxP)}%` }" />
            </div>
            <div class="cr-tick">{{ label(t.ch) }}</div>
          </div>
        </div>
      </div>
      <div>
        <div class="cr-head">hidden state <b>h</b><sub>t</sub>: {{ H }} numbers in [−1, 1] — all the model remembers</div>
        <div class="cr-grid">
          <div v-for="(v, i) in run.h" :key="i" class="cr-cell" :style="{ background: cellColor(v) }" :title="`unit ${i}: ${v.toFixed(2)}`" />
        </div>
        <div class="cr-legend">
          <span class="cr-swatch" :style="{ background: cellColor(-1) }" /> −1
          <span class="cr-swatch" :style="{ background: cellColor(0) }" /> 0
          <span class="cr-swatch" :style="{ background: cellColor(1) }" /> +1
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cr {
  font-size: 13px;
  color: #0b0b0b;
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.cr-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-bottom: 8px; }
.cr-label { color: #52514e; margin-right: 2px; }
.cr-muted { color: #898781; }
.cr-gap { margin-left: 12px; }
.cr-input {
  width: 190px;
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace;
  font-size: 13px;
  border: 1px solid #c3c2b7;
  border-radius: 4px;
  padding: 1px 6px;
  background: #ffffff;
}
.cr-controls button {
  font-size: 12.5px;
  border: 1px solid #c3c2b7;
  border-radius: 4px;
  padding: 1px 7px;
  color: #52514e;
  background: #ffffff;
}
.cr-controls button.cr-sel { background: #2a78d6; border-color: #2a78d6; color: #ffffff; }
.cr-controls button.cr-act { border-color: #2a78d6; color: #0b0b0b; }
.cr-text {
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace;
  font-size: 14.5px;
  line-height: 1.45;
  min-height: 4.4em;
  border: 1px solid #e1e0d9;
  border-radius: 4px;
  padding: 5px 8px;
  white-space: pre-wrap;
  word-break: break-all;
}
.cr-gen { background: #cde2fb; }
.cr-caret { color: #898781; }
.cr-meta { margin: 5px 0 10px; font-variant-numeric: tabular-nums; }
.cr-panels { display: grid; grid-template-columns: 5fr 3fr; gap: 28px; }
.cr-head { color: #52514e; margin-bottom: 5px; font-variant-numeric: tabular-nums; }
.cr-head b { color: #0b0b0b; }
.cr-bars { display: flex; align-items: flex-end; gap: 2px; }
.cr-col { flex: 1; text-align: center; }
.cr-track { height: 96px; display: flex; align-items: flex-end; border-bottom: 1px solid #c3c2b7; }
.cr-bar { width: 100%; background: #2a78d6; border-radius: 3px 3px 0 0; }
.cr-tick { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; color: #52514e; }
.cr-grid { display: grid; grid-template-columns: repeat(16, 1fr); gap: 2px; width: 100%; }
.cr-cell { aspect-ratio: 1.35; border-radius: 2px; }
.cr-legend { margin-top: 5px; color: #52514e; font-size: 12px; display: flex; align-items: center; gap: 4px; }
.cr-swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-left: 8px; }
.cr-swatch:first-child { margin-left: 0; }
</style>
