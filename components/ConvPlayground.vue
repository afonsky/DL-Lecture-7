<script setup lang="ts">
import { computed, ref } from 'vue'

// A 7x7 input, a 3x3 kernel, padding and stride: the whole of d2l 7.2-7.3 on one screen.
const N = 7
const K = 3

type Grid = number[][]
const zeros = (n: number): Grid => Array.from({ length: n }, () => Array(n).fill(0))

const patterns: Record<string, () => Grid> = {
  'bar': () => zeros(N).map(row => row.map((_, j) => (j >= 2 && j <= 4 ? 1 : 0))),
  'square': () => zeros(N).map((row, i) => row.map((_, j) => (i >= 2 && i <= 4 && j >= 2 && j <= 4 ? 1 : 0))),
  'diagonal': () => zeros(N).map((row, i) => row.map((_, j) => (i === j ? 1 : 0))),
}

const kernels: Record<string, { k: Grid, label?: string }> = {
  'vertical edges': { k: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
  'horizontal edges': { k: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
  'blur': { k: zeros(K).map(r => r.map(() => 1 / 9)), label: '1/9' },
  'sharpen': { k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
  'identity': { k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
}

const pattern = ref('bar')
const input = ref<Grid>(patterns.bar())
const kname = ref('vertical edges')
const pad = ref(0)
const stride = ref(1)
const hover = ref<[number, number]>([0, 0])
const shifted = ref(0)

const kernel = computed(() => kernels[kname.value].k)
const P = computed(() => N + 2 * pad.value)                    // padded size
const out = computed(() => Math.floor((N + 2 * pad.value - K) / stride.value) + 1)

// padded input: value, or null for a padding cell
const padded = computed(() => Array.from({ length: P.value }, (_, i) => Array.from({ length: P.value }, (_, j) => {
  const r = i - pad.value
  const c = j - pad.value
  return r >= 0 && r < N && c >= 0 && c < N ? input.value[r][c] : null
})))

function products(oi: number, oj: number): Grid {
  return kernel.value.map((row, a) => row.map((kv, b) => (padded.value[oi * stride.value + a][oj * stride.value + b] ?? 0) * kv))
}
const output = computed(() => Array.from({ length: out.value }, (_, i) => Array.from({ length: out.value }, (_, j) =>
  products(i, j).flat().reduce((s, v) => s + v, 0))))

const h = computed<[number, number]>(() => [Math.min(hover.value[0], out.value - 1), Math.min(hover.value[1], out.value - 1)])
const prod = computed(() => products(h.value[0], h.value[1]))
const hval = computed(() => output.value[h.value[0]][h.value[1]])
const inWindow = (i: number, j: number) => {
  const [oi, oj] = h.value
  const r0 = oi * stride.value
  const c0 = oj * stride.value
  return i >= r0 && i < r0 + K && j >= c0 && j < c0 + K
}

function fmt(v: number) {
  if (Math.abs(v) < 1e-9)
    return '0'
  const r = Math.round(v * 100) / 100
  return Number.isInteger(r) ? String(r) : r.toFixed(2).replace(/0$/, '').replace(/^(-?)0\./, '$1.')
}

// diverging fill: blue positive, red negative, neutral gray at zero (dataviz reference pair)
function mix(a: number[], b: number[], t: number) {
  return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(',')})`
}
const NEUTRAL = [240, 239, 236]
const BLUE = [42, 120, 214]
const RED = [227, 73, 72]
function fill(v: number, scale: number) {
  const t = Math.min(1, Math.abs(v) / (scale || 1))
  return { background: mix(NEUTRAL, v >= 0 ? BLUE : RED, t), color: t > 0.55 ? '#ffffff' : '#0b0b0b' }
}
const outScale = computed(() => Math.max(1e-9, ...output.value.flat().map(Math.abs)))
const kScale = computed(() => Math.max(...kernel.value.flat().map(Math.abs)))
const pScale = computed(() => Math.max(1e-9, ...prod.value.flat().map(Math.abs)))

function toggle(i: number, j: number) {
  const r = i - pad.value
  const c = j - pad.value
  if (r < 0 || r >= N || c < 0 || c >= N)
    return
  input.value[r][c] = input.value[r][c] ? 0 : 1
}
function setPattern(name: string) {
  pattern.value = name
  input.value = patterns[name]()
  shifted.value = 0
}
function shiftRight() {
  input.value = input.value.map(row => [0, ...row.slice(0, N - 1)])
  shifted.value += 1
}
</script>

<template>
  <div class="cp">
    <div class="cp-row">
      <div class="cp-block">
        <div class="cp-cap">input {{ N }}×{{ N }}<span v-if="pad"> + padding {{ pad }}</span></div>
        <div class="cp-grid" :style="{ gridTemplateColumns: `repeat(${P}, var(--cell))` }">
          <template v-for="(row, i) in padded" :key="`r${i}`">
            <div
              v-for="(v, j) in row" :key="`c${i}-${j}`"
              class="cp-cell cp-click"
              :class="{ 'cp-pad': v === null, 'cp-on': v === 1, 'cp-win': inWindow(i, j) }"
              @click="toggle(i, j)"
            >
              {{ v === null ? '0' : v }}
            </div>
          </template>
        </div>
      </div>

      <div class="cp-op">⊙</div>

      <div class="cp-block">
        <div class="cp-cap">kernel {{ K }}×{{ K }}</div>
        <div class="cp-grid" :style="{ gridTemplateColumns: `repeat(${K}, var(--cell))` }">
          <template v-for="(row, a) in kernel" :key="`k${a}`">
            <div v-for="(v, b) in row" :key="`k${a}-${b}`" class="cp-cell" :style="fill(v, kScale)">
              {{ kernels[kname].label ?? fmt(v) }}
            </div>
          </template>
        </div>
        <div class="cp-cap cp-mt">products</div>
        <div class="cp-grid" :style="{ gridTemplateColumns: `repeat(${K}, var(--cell))` }">
          <template v-for="(row, a) in prod" :key="`p${a}`">
            <div v-for="(v, b) in row" :key="`p${a}-${b}`" class="cp-cell" :style="fill(v, pScale)">
              {{ fmt(v) }}
            </div>
          </template>
        </div>
        <div class="cp-sum">Σ = <b>{{ fmt(hval) }}</b></div>
      </div>

      <div class="cp-op">→</div>

      <div class="cp-block">
        <div class="cp-cap">output {{ out }}×{{ out }} &nbsp;<span class="cp-muted">(hover a cell)</span></div>
        <div class="cp-grid" :style="{ gridTemplateColumns: `repeat(${out}, var(--cell))` }">
          <template v-for="(row, i) in output" :key="`o${i}`">
            <div
              v-for="(v, j) in row" :key="`o${i}-${j}`"
              class="cp-cell cp-click" :class="{ 'cp-win': i === h[0] && j === h[1] }"
              :style="fill(v, outScale)"
              @mouseenter="hover = [i, j]" @click="hover = [i, j]"
            >
              {{ fmt(v) }}
            </div>
          </template>
        </div>
        <div class="cp-formula">
          ⌊({{ N }} + 2·{{ pad }} − {{ K }}) / {{ stride }}⌋ + 1 = <b>{{ out }}</b>
        </div>
      </div>
    </div>

    <div class="cp-controls">
      <span class="cp-label">kernel</span>
      <button v-for="(_, name) in kernels" :key="name" :class="{ 'cp-sel': kname === name }" @click="kname = name">{{ name }}</button>
      <span class="cp-label cp-gap">padding</span>
      <button v-for="p in [0, 1, 2]" :key="`p${p}`" :class="{ 'cp-sel': pad === p }" @click="pad = p">{{ p }}</button>
      <span class="cp-label cp-gap">stride</span>
      <button v-for="s in [1, 2, 3]" :key="`s${s}`" :class="{ 'cp-sel': stride === s }" @click="stride = s">{{ s }}</button>
    </div>
    <div class="cp-controls">
      <span class="cp-label">input</span>
      <button v-for="(_, name) in patterns" :key="name" :class="{ 'cp-sel': pattern === name && !shifted }" @click="setPattern(name)">{{ name }}</button>
      <button class="cp-gap" @click="shiftRight">shift input → <span v-if="shifted">({{ shifted }})</span></button>
      <span class="cp-muted cp-gap">click input cells to toggle them</span>
    </div>
  </div>
</template>

<style scoped>
.cp {
  --cell: 30px;
  font-size: 13px;
  color: #0b0b0b;
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.cp-row { display: flex; align-items: flex-start; gap: 14px; }
.cp-block { display: flex; flex-direction: column; align-items: flex-start; }
.cp-cap { color: #52514e; margin-bottom: 4px; font-size: 13px; white-space: nowrap; }
.cp-mt { margin-top: 10px; }
.cp-muted { color: #898781; }
.cp-grid { display: grid; gap: 2px; }
.cp-cell {
  width: var(--cell);
  height: var(--cell);
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 3px;
  background: #f0efec;
  color: #898781;
  font-variant-numeric: tabular-nums;
  font-size: 12px;
  line-height: 1;
  box-sizing: border-box;
  user-select: none;
}
.cp-click { cursor: pointer; }
.cp-on { background: #2a78d6; color: #ffffff; }
.cp-pad { background: #ffffff; border: 1px dashed #c3c2b7; color: #c3c2b7; }
.cp-win { outline: 2.5px solid #eb6834; outline-offset: -1px; z-index: 1; }
.cp-op { align-self: center; font-size: 24px; color: #52514e; margin-top: 18px; }
.cp-sum { margin-top: 6px; font-size: 15px; }
.cp-formula { margin-top: 8px; font-size: 15px; color: #0b0b0b; white-space: nowrap; }
.cp-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-top: 10px; }
.cp-label { color: #52514e; margin-right: 2px; }
.cp-gap { margin-left: 12px; }
.cp-controls button {
  font-size: 12.5px;
  border: 1px solid #c3c2b7;
  border-radius: 4px;
  padding: 1px 7px;
  color: #52514e;
  background: #ffffff;
}
.cp-controls button.cp-sel { background: #2a78d6; border-color: #2a78d6; color: #ffffff; }
</style>
