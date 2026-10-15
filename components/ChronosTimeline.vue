<template>
  <div ref="containerRef" class="chronos-timeline-wrapper"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { ChronosTimeline, attachChronosStyles } from 'chronos-timeline-md'
import type { ChronosPluginSettings } from 'chronos-timeline-md'

const props = withDefaults(defineProps<{
  source: string
  settings?: Partial<ChronosPluginSettings>
}>(), {
  source: '',
  settings: () => ({})
})

const containerRef = ref<HTMLElement | null>(null)
let timeline: ChronosTimeline | null = null

const defaultSettings: ChronosPluginSettings = {
  selectedLocale: 'en',
  align: 'left',
  clickToUse: false,
  roundRanges: false,
  useUtc: false,
  useAI: false,
}

const renderTimeline = () => {
  if (!containerRef.value) return
  
  // Destroy existing timeline if any
  if (timeline) {
    timeline.destroy()
    timeline = null
  }
  
  // Clear container
  containerRef.value.innerHTML = ''
  
  if (props.source.trim()) {
    timeline = ChronosTimeline.render(
      containerRef.value,
      props.source,
      { ...defaultSettings, ...props.settings }
    )
  }
}

onMounted(() => {
  attachChronosStyles(document)
  nextTick(renderTimeline)
})

onBeforeUnmount(() => {
  if (timeline) {
    timeline.destroy()
    timeline = null
  }
})

watch(() => props.source, () => nextTick(renderTimeline))
watch(() => props.settings, () => nextTick(renderTimeline), { deep: true })
</script>

<style scoped>
.chronos-timeline-wrapper {
  width: 100%;
  min-height: 200px;
}
</style>

<!-- Unscoped: vis-timeline builds its DOM imperatively, so it carries no scope attribute. -->
<style>
/*
  Era regions ("@ [1940~1960] Foundations") are vis background items. They are anchored to
  the bottom of the item area and carry their label at the *top* of that area - which is
  exactly where the first stacked row of flag events ("- [1969] ...") sits, so the two
  collide. Extend each era region upwards by a fixed header strip and let it overflow the
  panel, so the era label gets a row of its own above the events.
  Using top/height:auto rather than a fixed height keeps this independent of the timeline
  height (chronos "> HEIGHT n" flag) and of how many rows the events stack into.
*/
.chronos-timeline-container {
  --chronos-era-header: 64px;
  padding-top: var(--chronos-era-header);
}

.chronos-timeline-container .vis-timeline,
.chronos-timeline-container .vis-panel.vis-center {
  overflow: visible;
}

.chronos-timeline-container .vis-item.vis-background {
  top: calc(-1 * var(--chronos-era-header)) !important;
  height: auto !important;
}
</style>
