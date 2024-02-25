<script setup lang="ts">
import { computed } from 'vue'
import { handleBackground } from '../layoutHelper'
const props = defineProps({
  imageTop: {
    type: String,
  },
  imageBottom: {
    type: String,
  },
})
const top = computed(() => handleBackground(props.imageTop))
const bottom = computed(() => handleBackground(props.imageBottom))
</script>

<template>
  <div class="grid grid-cols-3 w-full h-full gap-x-8 shadercode">
    <div class="ml-16 my-16 col-span-2">
      <slot />
    </div>
    <div class="grid grid-rows-2 w-full h-full">
      <div class="" :style="top"></div>
      <div class="" :style="bottom"></div>
    </div>
  </div>
</template>