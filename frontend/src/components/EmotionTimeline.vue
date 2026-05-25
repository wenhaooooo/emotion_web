<template>
  <div v-if="segments.length" class="timeline-container">
    <h3 class="timeline-title">情感时间轴</h3>
    <div class="timeline-bar">
      <div
        v-for="seg in segments"
        :key="seg.index"
        class="timeline-segment"
        :style="segmentStyle(seg)"
        :title="segmentTooltip(seg)"
        @mouseenter="hoveredSegment = seg"
        @mouseleave="hoveredSegment = null"
      ></div>
    </div>
    <div class="timeline-labels">
      <span>{{ formatTime(0) }}</span>
      <span>{{ formatTime(totalDuration) }}</span>
    </div>
    <div class="timeline-legend">
      <span v-for="(name, key) in names" :key="key" class="legend-item">
        <span class="legend-dot" :style="{ background: colors[key] }"></span>
        {{ name }}
      </span>
    </div>
    <div v-if="hoveredSegment" class="tooltip-detail">
      <strong>{{ formatTime(hoveredSegment.start_time) }} - {{ formatTime(hoveredSegment.end_time) }}</strong>
      : {{ getLabelName(hoveredSegment, labelMode) }} ({{ (hoveredSegment.confidence * 100).toFixed(0) }}%)
      <br />
      <span class="tooltip-text">{{ hoveredSegment.text }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { getColors, getNames, getLabel, getLabelName } from '../emotionConfig.js'

const props = defineProps({
  segments: { type: Array, default: () => [] },
  totalDuration: { type: Number, default: 0 },
  labelMode: { type: String, default: 'e2v' },
})

const hoveredSegment = ref(null)

const colors = computed(() => getColors(props.labelMode))
const names = computed(() => getNames(props.labelMode))

function segmentStyle(seg) {
  const total = props.totalDuration || 1
  const left = (seg.start_time / total) * 100
  const width = (seg.duration / total) * 100
  const label = getLabel(seg, props.labelMode)
  return {
    left: left + '%',
    width: Math.max(width, 0.5) + '%',
    background: colors.value[label] || '#ccc',
  }
}

function segmentTooltip(seg) {
  return `${formatTime(seg.start_time)}-${formatTime(seg.end_time)}: ${getLabelName(seg, props.labelMode)} (${(seg.confidence * 100).toFixed(0)}%)`
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}
</script>

<style scoped>
.timeline-container { background: #fff; border-radius: 12px; padding: 20px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.timeline-title { font-size: 15px; font-weight: 600; color: #333; margin: 0 0 12px; }
.timeline-bar { position: relative; height: 32px; background: #f0f0f0; border-radius: 6px; overflow: hidden; }
.timeline-segment {
  position: absolute; top: 0; height: 100%; min-width: 2px;
  cursor: pointer; transition: opacity 0.15s; border-right: 1px solid rgba(255,255,255,0.3);
}
.timeline-segment:hover { opacity: 0.8; }
.timeline-labels { display: flex; justify-content: space-between; font-size: 12px; color: #999; margin-top: 4px; }
.timeline-legend { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 13px; color: #666; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }
.tooltip-detail { margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 8px; font-size: 13px; color: #333; }
.tooltip-text { color: #666; font-size: 12px; }
</style>
