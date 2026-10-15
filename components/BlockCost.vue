<script setup lang="ts">
import { computed, ref } from 'vue'

// Weights and multiply-adds of the blocks met today, on one input map (stride 1, "same" padding, biases ignored).
// Per layer: weights = k*k*c_in*c_out (depthwise: k*k*c_in), multiply-adds = h*w*weights - Lecture 4's formula.
type Layer = { k: number, cin: number, cout: number, dw?: boolean }
type Block = { name: string, tag: string, layers: Layer[], rf: number, relus: number }

const presets = [
  { label: '28×28×192 → 32', note: 'A. Ng\'s example: the input of GoogLeNet\'s first Inception block', h: 28, c: 192, d: 32, b: 16 },
  { label: '56×56×256 → 256', note: 'ResNet-50, first stage', h: 56, c: 256, d: 256, b: 64 },
  { label: '14×14×512 → 512', note: 'VGG-16, last block', h: 14, c: 512, d: 512, b: 128 },
]
const preset = ref(0)
const h = ref(28)
const c = ref(192)
const d = ref(32)
const b = ref(16)
const hover = ref(3)

function setPreset(i: number) {
  preset.value = i
  const p = presets[i]
  h.value = p.h
  c.value = p.c
  d.value = p.d
  b.value = p.b
}

const blocks = computed<Block[]>(() => [
  { name: '5×5 conv', tag: 'AlexNet-style', layers: [{ k: 5, cin: c.value, cout: d.value }], rf: 5, relus: 1 },
  { name: '3×3 conv', tag: 'one layer', layers: [{ k: 3, cin: c.value, cout: d.value }], rf: 3, relus: 1 },
  { name: '3×3 → 3×3', tag: 'VGG', layers: [{ k: 3, cin: c.value, cout: d.value }, { k: 3, cin: d.value, cout: d.value }], rf: 5, relus: 2 },
  { name: '1×1 → 5×5', tag: 'Inception branch', layers: [{ k: 1, cin: c.value, cout: b.value }, { k: 5, cin: b.value, cout: d.value }], rf: 5, relus: 2 },
  { name: '1×1 → 3×3 → 1×1', tag: 'ResNet bottleneck', layers: [{ k: 1, cin: c.value, cout: b.value }, { k: 3, cin: b.value, cout: b.value }, { k: 1, cin: b.value, cout: d.value }], rf: 3, relus: 3 },
  { name: '3×3 depthwise → 1×1', tag: 'MobileNet', layers: [{ k: 3, cin: c.value, cout: c.value, dw: true }, { k: 1, cin: c.value, cout: d.value }], rf: 3, relus: 2 },
])

const weights = (l: Layer) => (l.dw ? l.k * l.k * l.cin : l.k * l.k * l.cin * l.cout)
const macs = (l: Layer) => h.value * h.value * weights(l)
const total = (bl: Block, f: (l: Layer) => number) => bl.layers.reduce((s, l) => s + f(l), 0)
const maxMacs = computed(() => Math.max(...blocks.value.map(bl => total(bl, macs))))
const base = computed(() => total(blocks.value[0], macs))

function big(v: number) {
  if (v >= 1e9)
    return `${(v / 1e9).toFixed(2)} G`
  if (v >= 1e6)
    return `${(v / 1e6).toFixed(1)} M`
  if (v >= 1e3)
    return `${(v / 1e3).toFixed(1)} K`
  return String(v)
}
const comma = (v: number) => v.toLocaleString('en-US')
const layerText = (l: Layer) => (l.dw ? `${l.k}×${l.k} dw, ${l.cin}` : `${l.k}×${l.k}: ${l.cin}→${l.cout}`)
const breakdown = (bl: Block) => (bl.layers.length > 1 ? bl.layers.map(l => big(macs(l))).join(' + ') : '')
</script>

<template>
  <div class="bc">
    <div class="bc-controls">
      <span class="bc-label">input → output</span>
      <button v-for="(p, i) in presets" :key="p.label" :class="{ 'bc-sel': preset === i && d === p.d && b === p.b }" @click="setPreset(i)">{{ p.label }}</button>
      <span class="bc-muted bc-gap">{{ presets[preset].note }}</span>
    </div>
    <div class="bc-controls">
      <span class="bc-label">output channels</span>
      <button v-for="v in [16, 32, 64, 128, 256, 512]" :key="`d${v}`" :class="{ 'bc-sel': d === v }" @click="d = v">{{ v }}</button>
      <span class="bc-label bc-gap">bottleneck width b</span>
      <button v-for="v in [8, 16, 32, 64, 128]" :key="`b${v}`" :class="{ 'bc-sel': b === v }" @click="b = v">{{ v }}</button>
    </div>

    <table class="bc-table">
      <thead>
        <tr>
          <th>block</th>
          <th>layers</th>
          <th class="bc-num">sees</th>
          <th class="bc-num">ReLUs</th>
          <th class="bc-num">weights</th>
          <th>multiply-adds on {{ h }}×{{ h }}×{{ c }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(bl, i) in blocks" :key="bl.name" :class="{ 'bc-hover': hover === i }" @mouseenter="hover = i" @click="hover = i">
          <td><b>{{ bl.name }}</b> <span class="bc-muted">{{ bl.tag }}</span></td>
          <td class="bc-layers">{{ bl.layers.map(layerText).join(', ') }}</td>
          <td class="bc-num">{{ bl.rf }}×{{ bl.rf }}</td>
          <td class="bc-num">{{ bl.relus }}</td>
          <td class="bc-num">{{ comma(total(bl, weights)) }}</td>
          <td>
            <div class="bc-barrow">
              <div class="bc-bar" :style="{ width: `${Math.max(0.6, 100 * total(bl, macs) / maxMacs) * 0.62}%` }" />
              <span class="bc-val">{{ big(total(bl, macs)) }}</span>
              <span v-if="i > 0" class="bc-muted">&nbsp;{{ (base / total(bl, macs)).toFixed(1) }}× cheaper</span>
            </div>
          </td>
        </tr>
      </tbody>
    </table>

    <div class="bc-foot">
      <span v-if="breakdown(blocks[hover])"><b>{{ blocks[hover].name }}</b>: {{ breakdown(blocks[hover]) }} = {{ big(total(blocks[hover], macs)) }} multiply-adds &nbsp;·&nbsp; </span>
      <span class="bc-muted">per layer: {{ h }}·{{ h }} · k² · c<sub>in</sub> · c<sub>out</sub> (Lecture 4); stride 1, "same" padding, biases ignored</span>
    </div>
  </div>
</template>

<style scoped>
.bc {
  font-size: 13px;
  color: #0b0b0b;
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.bc-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-bottom: 7px; }
.bc-label { color: #52514e; margin-right: 2px; }
.bc-muted { color: #898781; }
.bc-gap { margin-left: 12px; }
.bc-controls button {
  font-size: 12.5px;
  border: 1px solid #c3c2b7;
  border-radius: 4px;
  padding: 1px 7px;
  color: #52514e;
  background: #ffffff;
}
.bc-controls button.bc-sel { background: #2a78d6; border-color: #2a78d6; color: #ffffff; }
.bc-table { width: 100%; border-collapse: collapse; margin-top: 6px; font-variant-numeric: tabular-nums; }
.bc-table th {
  text-align: left;
  font-weight: 500;
  color: #52514e;
  border-bottom: 1px solid #c3c2b7;
  padding: 3px 8px 3px 0;
  white-space: nowrap;
}
.bc-table td { padding: 4px 8px 4px 0; border-bottom: 1px solid #e1e0d9; white-space: nowrap; }
.bc-table tr { cursor: pointer; }
.bc-table tr.bc-hover td { background: #f0efec; }
.bc-num { text-align: right; }
.bc-layers { color: #52514e; font-size: 12px; }
.bc-barrow { display: flex; align-items: center; width: 330px; }
.bc-bar { height: 12px; background: #2a78d6; border-radius: 0 4px 4px 0; margin-right: 6px; }
.bc-val { font-weight: 600; }
.bc-foot { margin-top: 8px; font-size: 12.5px; }
</style>
