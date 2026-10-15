<script setup lang="ts">
import { computed } from 'vue'
import { handleBackground } from '../layoutHelper'

const props = defineProps({
  image: {
    type: String,
  },
  class: {
    type: String,
  },
  backgroundSize: {
    type: String,
    default: 'cover',
  },
})

const style = computed(() => handleBackground(props.image, false, props.backgroundSize))
</script>

<template>
  <div class="chapter-title-root grid grid-cols-2 w-full h-full auto-rows-fr">
    <div class="w-full w-full" :style="style" />
    <div class="slidev-layout default chapter-title-content" :class="props.class">
      <slot />
    </div>
  </div>
</template>

<style scoped>
/* Plain CSS, not the "bg-black" utility: in this project UnoCSS does not
   generate that class (verified: no .bg-black rule ends up in the compiled
   stylesheet), so the utility silently did nothing and the grid fell through
   to the slide's actual white background. The right panel has no background
   of its own, so it needs this element to truly be black. */
.chapter-title-root {
  background-color: #000;
}

/* Against that black background, the theme's normal (dark) heading color is
   invisible. Force the heading white so the chapter title can be read.
   The heading comes from the slide's markdown, i.e. slotted content, so it
   must be reached with :slotted(), not :deep() (which targets a child
   component's own internal DOM, not content passed into this component). */
.chapter-title-content :slotted(h1),
.chapter-title-content :slotted(h2) {
  color: white;
}
</style>