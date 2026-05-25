<template>
  <div v-if="visible" class="progress-panel">
    <div class="progress-header">
      <span class="progress-title">处理进度</span>
      <span class="progress-percent">{{ Math.round(progress * 100) }}%</span>
    </div>
    <div class="progress-bar-bg">
      <div class="progress-bar-fill" :style="{ width: (progress * 100) + '%' }"></div>
    </div>
    <p class="progress-message">{{ message }}</p>
    <p v-if="status === 'failed'" class="progress-error">处理失败: {{ message }}</p>
  </div>
</template>

<script setup>
defineProps({
  visible: Boolean,
  progress: { type: Number, default: 0 },
  message: { type: String, default: '' },
  status: { type: String, default: '' },
})
</script>

<style scoped>
.progress-panel {
  background: #fff;
  border-radius: 12px;
  padding: 20px 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.progress-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
.progress-title { font-size: 14px; font-weight: 600; color: #333; }
.progress-percent { font-size: 14px; color: #007AFF; font-weight: 600; }
.progress-bar-bg {
  height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden;
}
.progress-bar-fill {
  height: 100%; background: linear-gradient(90deg, #007AFF, #5ac8fa);
  border-radius: 4px; transition: width 0.3s ease;
}
.progress-message { font-size: 13px; color: #666; margin: 8px 0 0; }
.progress-error { font-size: 13px; color: #FF3B30; margin: 4px 0 0; }
</style>
