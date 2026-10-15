<script setup lang="ts">
import { computed, ref } from 'vue'
import traces from './beamTraces.json'

// Beam search (k = 3) on English sentences of the test set, step by step, as run by the encoder-decoder trained for
// this lecture (scripts/lecture7/experiments.py beam_traces gru2 0). The traces are precomputed in PyTorch; the
// page only replays them. Scores: log-probabilities; the winner maximizes log P / L^0.75 (L counts <eos>).
interface Step { live: [string, number][]; finished: [string, number][] }
interface Trace { src: string; unk: string[]; greedy: string; greedy_logp: number; beam: string; beam_logp: number; steps: Step[] }
const T = traces as Trace[]
const alpha = 0.75

const sel = ref(0)
const step = ref(0)
const tr = computed(() => T[sel.value])
const n = computed(() => tr.value.steps.length)
const cur = computed(() => tr.value.steps[Math.min(step.value, n.value - 1)])
const done = computed(() => step.value >= n.value - 1)
const len = (s: string) => (s ? s.split(' ').length : 0) + 1        // + <eos>
const score = (s: string, lp: number) => lp / len(s) ** alpha
const finished = computed(() => cur.value.finished
  .map(([s, lp]) => ({ s, lp, sc: score(s, lp) }))
  .sort((a, b) => b.sc - a.sc))
const best = computed(() => (done.value && finished.value.length ? finished.value[0].s : null))

function pick(i: number) { sel.value = i; step.value = 0 }
function next() { if (step.value < n.value - 1) step.value++ }
function prev() { if (step.value > 0) step.value-- }
const fmt = (x: number) => x.toFixed(2).replace('-', '−')
const short = (s: string) => (s.length > 34 ? `${s.slice(0, 32)}…` : s)
</script>

<template>
  <div class="bs">
    <div class="bs-controls">
      <span class="bs-label">English:</span>
      <button v-for="(t, i) in T" :key="i" :class="{ 'bs-sel': i === sel }" @click="pick(i)">{{ short(t.src) }}</button>
    </div>
    <div class="bs-controls">
      <button class="bs-act" @click="prev">◀ step</button>
      <button class="bs-act" @click="next">step ▶</button>
      <button @click="step = n - 1">to the end</button>
      <button @click="step = 0">reset</button>
      <span class="bs-meta">step {{ Math.min(step, n - 1) + 1 }} of {{ n }} · beam size 3 · the decoder has read <b>{{ tr.src }}</b></span>
    </div>
    <div class="bs-panels">
      <div>
        <div class="bs-head">the 3 best unfinished translations</div>
        <div v-for="([s, lp], i) in cur.live" :key="`l${i}`" class="bs-row">
          <span class="bs-text">{{ s }} …</span><span class="bs-num">log P = {{ fmt(lp) }}</span>
        </div>
        <div v-if="!cur.live.length" class="bs-muted">none</div>
      </div>
      <div>
        <div class="bs-head">finished (ended with &lt;eos&gt;), best score first</div>
        <div v-for="f in finished.slice(0, 4)" :key="`f${f.s}`" class="bs-row" :class="{ 'bs-best': f.s === best }">
          <span class="bs-text">{{ f.s }}</span>
          <span class="bs-num">log P = {{ fmt(f.lp) }} · L = {{ len(f.s) }} · score {{ fmt(f.sc) }}</span>
        </div>
        <div v-if="!finished.length" class="bs-muted">none yet</div>
        <div v-if="finished.length > 4" class="bs-muted">… and {{ finished.length - 4 }} more</div>
      </div>
    </div>
    <div class="bs-foot" :style="{ visibility: done ? 'visible' : 'hidden' }">
      greedy search: <span class="bs-text">{{ tr.greedy }}</span> <span class="bs-num">(log P = {{ fmt(tr.greedy_logp) }})</span>
      &nbsp;·&nbsp; beam search: <span class="bs-text">{{ tr.beam }}</span> <span class="bs-num">(log P = {{ fmt(tr.beam_logp) }})</span>
    </div>
  </div>
</template>

<style scoped>
.bs { font-size: 13px; color: #0b0b0b; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; }
.bs-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-bottom: 7px; }
.bs-label { color: #52514e; margin-right: 2px; }
.bs-controls button {
  font-size: 12.5px;
  border: 1px solid #c3c2b7;
  border-radius: 4px;
  padding: 1px 7px;
  color: #52514e;
  background: #ffffff;
}
.bs-controls button.bs-sel { background: #2a78d6; border-color: #2a78d6; color: #ffffff; }
.bs-controls button.bs-act { border-color: #2a78d6; color: #0b0b0b; }
.bs-meta { margin-left: 10px; color: #52514e; }
.bs-meta b { color: #0b0b0b; font-weight: 600; }
.bs-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 4px; }
.bs-head { color: #52514e; margin-bottom: 4px; }
.bs-row {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  border: 1px solid #e1e0d9;
  border-radius: 4px;
  padding: 2px 7px;
  margin-bottom: 3px;
}
.bs-best { background: #cde2fb; border-color: #2a78d6; }
.bs-text { font-family: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace; font-size: 12.5px; }
.bs-num { color: #52514e; font-variant-numeric: tabular-nums; white-space: nowrap; font-size: 12px; }
.bs-muted { color: #898781; }
.bs-foot { margin-top: 8px; color: #52514e; }
.bs-foot .bs-text { color: #0b0b0b; }
</style>
