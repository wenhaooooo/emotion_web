<template>
  <div v-if="segments.length" class="result-container">
    <div class="result-header">
      <h3 class="result-title">识别结果</h3>
      <div class="filter-buttons">
        <button
          v-for="(name, key) in names"
          :key="key"
          class="filter-btn"
          :class="{ active: activeFilter === null || activeFilter === Number(key) }"
          :style="activeFilter === Number(key) ? { background: colors[key], color: '#fff' } : {}"
          @click="toggleFilter(Number(key))"
        >
          {{ name }}
        </button>
      </div>
    </div>
    <table class="result-table">
      <thead>
        <tr>
          <th>#</th>
          <th>时间段</th>
          <th>情感标签</th>
          <th>置信度</th>
          <th>文本</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="seg in filteredSegments" :key="seg.index">
          <td>{{ seg.index + 1 }}</td>
          <td class="time-cell">{{ formatTime(seg.start_time) }} - {{ formatTime(seg.end_time) }}</td>
          <td>
            <span class="emotion-badge" :style="{ background: colors[getLabel(seg, labelMode)] }">
              {{ getLabelName(seg, labelMode) }}
            </span>
          </td>
          <td class="confidence-cell">
            <div class="confidence-bar-bg">
              <div class="confidence-bar-fill" :style="{ width: (seg.confidence * 100) + '%', background: colors[getLabel(seg, labelMode)] }"></div>
            </div>
            <span class="confidence-text">{{ (seg.confidence * 100).toFixed(0) }}%</span>
          </td>
          <td class="text-cell" :title="seg.text">{{ seg.text }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { getColors, getNames, getLabel, getLabelName } from '../emotionConfig.js'

const props = defineProps({
  segments: { type: Array, default: () => [] },
  labelMode: { type: String, default: 'e2v' },
})

const activeFilter = ref(null)

const colors = computed(() => getColors(props.labelMode))
const names = computed(() => getNames(props.labelMode))

const filteredSegments = computed(() => {
  if (activeFilter.value === null) return props.segments
  return props.segments.filter(s => getLabel(s, props.labelMode) === activeFilter.value)
})

function toggleFilter(label) {
  activeFilter.value = activeFilter.value === label ? null : label
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}
</script>

<style scoped>
.result-container { background: #fff; border-radius: 12px; padding: 20px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }
.result-title { font-size: 15px; font-weight: 600; color: #333; margin: 0; }
.filter-buttons { display: flex; gap: 6px; flex-wrap: wrap; }
.filter-btn {
  padding: 4px 12px; border: 1px solid #ddd; border-radius: 16px;
  background: #fff; font-size: 12px; cursor: pointer; transition: all 0.15s;
}
.filter-btn.active { border-color: transparent; }
.result-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.result-table th { text-align: left; padding: 10px 8px; border-bottom: 2px solid #eee; color: #666; font-weight: 600; }
.result-table td { padding: 10px 8px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }
.time-cell { white-space: nowrap; color: #555; }
.emotion-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; color: #fff; font-size: 12px; font-weight: 500; }
.confidence-cell { min-width: 100px; }
.confidence-bar-bg { display: inline-block; width: 50px; height: 6px; background: #eee; border-radius: 3px; vertical-align: middle; margin-right: 6px; }
.confidence-bar-fill { height: 100%; border-radius: 3px; }
.confidence-text { font-size: 12px; color: #666; }
.text-cell { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #555; }
</style>
