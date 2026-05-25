<template>
  <div
    class="upload-area"
    :class="{ dragging: isDragging, disabled: disabled }"
    @dragover.prevent="isDragging = true"
    @dragleave="isDragging = false"
    @drop.prevent="onDrop"
    @click="triggerFileInput"
  >
    <input
      ref="fileInput"
      type="file"
      :accept="acceptTypes"
      style="display: none"
      @change="onFileSelect"
    />
    <div v-if="!uploading" class="upload-content">
      <div class="upload-icon">+</div>
      <p class="upload-text">拖拽视频到此处 或 点击选择文件</p>
      <p class="upload-hint">支持 mp4, avi, mov, mkv 格式</p>
    </div>
    <div v-else class="upload-content">
      <p class="upload-text">正在上传: {{ fileName }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { uploadVideo } from '../api.js'

const emit = defineEmits(['upload-start', 'upload-success', 'upload-error'])
const props = defineProps({ disabled: Boolean })

const fileInput = ref(null)
const isDragging = ref(false)
const uploading = ref(false)
const fileName = ref('')
const acceptTypes = '.mp4,.avi,.mov,.mkv,.flv,.wmv'

function triggerFileInput() {
  if (!props.disabled && !uploading.value) {
    fileInput.value.click()
  }
}

function onDrop(e) {
  isDragging.value = false
  if (props.disabled) return
  const file = e.dataTransfer.files[0]
  if (file) processFile(file)
}

function onFileSelect(e) {
  const file = e.target.files[0]
  if (file) processFile(file)
  e.target.value = ''
}

async function processFile(file) {
  const ext = file.name.split('.').pop().toLowerCase()
  const allowed = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']
  if (!allowed.includes(ext)) {
    emit('upload-error', `不支持的文件格式: .${ext}`)
    return
  }

  uploading.value = true
  fileName.value = file.name
  emit('upload-start', file.name)

  try {
    const res = await uploadVideo(file)
    emit('upload-success', res.data)
  } catch (err) {
    const msg = err.response?.data?.detail || err.message || 'Upload failed'
    emit('upload-error', msg)
  } finally {
    uploading.value = false
  }
}
</script>

<style scoped>
.upload-area {
  border: 2px dashed #ccc;
  border-radius: 12px;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: #fafafa;
}
.upload-area:hover { border-color: #007AFF; background: #f0f7ff; }
.upload-area.dragging { border-color: #007AFF; background: #e8f4ff; }
.upload-area.disabled { opacity: 0.5; cursor: not-allowed; }
.upload-icon { font-size: 48px; color: #999; margin-bottom: 8px; }
.upload-text { font-size: 16px; color: #333; margin: 0; }
.upload-hint { font-size: 13px; color: #999; margin: 8px 0 0; }
</style>
