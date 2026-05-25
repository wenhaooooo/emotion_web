<template>
  <div v-if="videoUrl" class="player-container">
    <h3 class="player-title">原始视频</h3>
    <video
      ref="videoEl"
      :src="videoUrl"
      controls
      class="video-player"
      @timeupdate="onTimeUpdate"
      @loadedmetadata="onLoaded"
    ></video>
    <div v-if="segments.length" class="player-timeline">
      <div class="timeline-track">
        <div
          v-for="seg in segments"
          :key="seg.index"
          class="track-segment"
          :style="trackStyle(seg)"
          :title="`${formatTime(seg.start_time)}-${formatTime(seg.end_time)}: ${getLabelName(seg, labelMode)}`"
          @click="seekTo(seg.start_time)"
        ></div>
        <div
          class="playhead"
          :style="{ left: playheadPos + '%' }"
        ></div>
      </div>
      <div class="track-labels">
        <span>{{ formatTime(currentTime) }} / {{ formatTime(duration) }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { getOriginalVideoUrl } from '../api.js'
import { getColors, getLabel, getLabelName } from '../emotionConfig.js'

const props = defineProps({
  jobId: { type: String, default: '' },
  segments: { type: Array, default: () => [] },
  labelMode: { type: String, default: 'e2v' },
})

const videoEl = ref(null)
const currentTime = ref(0)
const duration = ref(0)

const videoUrl = computed(() => props.jobId ? getOriginalVideoUrl(props.jobId) : '')

const playheadPos = computed(() => {
  if (!duration.value) return 0
  return (currentTime.value / duration.value) * 100
})

const colors = computed(() => getColors(props.labelMode))

function trackStyle(seg) {
  const total = duration.value || 1
  const left = (seg.start_time / total) * 100
  const width = (seg.duration / total) * 100
  const label = getLabel(seg, props.labelMode)
  return {
    left: left + '%',
    width: Math.max(width, 0.3) + '%',
    background: colors.value[label] || '#ccc',
  }
}

function seekTo(time) {
  if (videoEl.value) {
    videoEl.value.currentTime = time
    videoEl.value.play()
  }
}

function onTimeUpdate() {
  if (videoEl.value) {
    currentTime.value = videoEl.value.currentTime
  }
}

function onLoaded() {
  if (videoEl.value) {
    duration.value = videoEl.value.duration
  }
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

defineExpose({ seekTo })
</script>

<style scoped>
.player-container { background: #fff; border-radius: 12px; padding: 20px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.player-title { font-size: 15px; font-weight: 600; color: #333; margin: 0 0 12px; }
.video-player { width: 100%; border-radius: 8px; background: #000; }
.player-timeline { margin-top: 10px; }
.timeline-track { position: relative; height: 24px; background: #f0f0f0; border-radius: 4px; overflow: hidden; cursor: pointer; }
.track-segment {
  position: absolute; top: 0; height: 100%; min-width: 1px;
  opacity: 0.7; transition: opacity 0.15s; border-right: 1px solid rgba(255,255,255,0.3);
}
.track-segment:hover { opacity: 1; }
.playhead {
  position: absolute; top: 0; width: 2px; height: 100%;
  background: #333; pointer-events: none; transition: left 0.1s linear;
}
.track-labels { display: flex; justify-content: flex-end; font-size: 12px; color: #999; margin-top: 4px; }
</style>
