<template>
  <div class="realtime-container">
    <!-- 上半部分：画面 + 结果面板 -->
    <div class="realtime-main">
      <!-- 左侧：摄像头画面 -->
      <div class="camera-section">
        <div class="camera-wrapper">
          <video ref="videoRef" autoplay muted playsinline></video>
          <canvas ref="canvasRef" class="face-overlay"></canvas>
          <div v-if="!isRunning" class="camera-placeholder">
            <span class="placeholder-icon">📷</span>
            <span class="placeholder-text">点击"开始识别"开启摄像头</span>
          </div>
        </div>
        <div class="camera-status">
          <span v-if="isRunning" class="status-active">● 摄像头已开启</span>
          <span v-else class="status-inactive">○ 摄像头未开启</span>
        </div>
      </div>

      <!-- 右侧：识别结果 -->
      <div class="result-panel">
        <!-- 当前情感大字 -->
        <div class="current-emotion" :style="{ borderColor: currentEmotionColor }">
          <!-- 面部：未检测到人脸 -->
          <template v-if="displaySource === 'face' && isRunning && !faceResult.faceDetected">
            <span class="emotion-icon">👤</span>
            <div class="emotion-info">
              <span class="emotion-label status-hint">未检测到人脸</span>
              <span class="emotion-confidence">请面向摄像头</span>
            </div>
          </template>
          <!-- 语音：安静状态 -->
          <template v-else-if="displaySource === 'voice' && isRunning && voiceResult.isSilence">
            <span class="emotion-icon">🔇</span>
            <div class="emotion-info">
              <span class="emotion-label status-hint">安静中</span>
              <span class="emotion-confidence">等待语音输入</span>
            </div>
          </template>
          <!-- 正常显示识别结果 -->
          <template v-else>
            <span class="emotion-icon">{{ currentEmotionIcon }}</span>
            <div class="emotion-info">
              <span class="emotion-label">{{ currentEmotionName }}</span>
              <span class="emotion-confidence">{{ currentConfidence }}%</span>
            </div>
          </template>
        </div>

        <!-- 数据源切换 -->
        <div class="source-tabs">
          <button
            :class="['source-tab', { active: displaySource === 'face' }]"
            @click="displaySource = 'face'"
          >
            😊 面部表情
          </button>
          <button
            :class="['source-tab', { active: displaySource === 'voice' }]"
            @click="displaySource = 'voice'"
          >
            🎤 语音情感
          </button>
        </div>

        <!-- 7类概率柱状图 -->
        <div class="probability-bars">
          <div
            v-for="emotion in REALTIME_EMOTIONS"
            :key="emotion.key"
            class="bar-row"
          >
            <span class="bar-icon">{{ emotion.icon }}</span>
            <span class="bar-label">{{ emotion.name }}</span>
            <div class="bar-track">
              <div
                class="bar-fill"
                :style="{
                  width: (getCurrentProb(emotion.key) * 100) + '%',
                  backgroundColor: emotion.color
                }"
              ></div>
            </div>
            <span class="bar-value">{{ (getCurrentProb(emotion.key) * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 下半部分：情感时间线 -->
    <div class="timeline-section" v-if="timeline.length > 0">
      <h3 class="timeline-title">情感变化时间线</h3>
      <div class="timeline-scroll" ref="timelineRef">
        <div
          v-for="(entry, index) in timeline"
          :key="index"
          class="timeline-entry"
          :style="{ backgroundColor: getEmotionColor(entry.emotion) + '33' }"
          :title="`${entry.time} - ${getEmotionName(entry.emotion)} (${(entry.confidence * 100).toFixed(0)}%) [${entry.source}]`"
        >
          <span class="timeline-emoji">{{ getEmotionIcon(entry.emotion) }}</span>
        </div>
      </div>
      <div class="timeline-legend">
        <span
          v-for="emotion in REALTIME_EMOTIONS"
          :key="emotion.key"
          class="legend-item"
        >
          <span class="legend-dot" :style="{ backgroundColor: emotion.color }"></span>
          {{ emotion.name }}
        </span>
      </div>
    </div>

    <!-- 控制按钮 -->
    <div class="controls">
      <button
        v-if="!isRunning"
        class="btn-start"
        @click="startRecognition"
      >
        ▶ 开始识别
      </button>
      <button
        v-if="isRunning"
        class="btn-stop"
        @click="stopRecognition"
      >
        ⏹ 停止识别
      </button>
      <button
        v-if="hasResults && !isRunning && currentJobId"
        class="btn-export"
        @click="exportResults"
      >
        📥 导出 Excel
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onUnmounted, nextTick } from 'vue'
import { createRealtimeSocket } from '../api.js'
import {
  REALTIME_EMOTIONS,
  getEmotionByKey,
  getEmotionColor,
  getEmotionName,
  getEmotionIcon,
} from '../emotionConfig.js'

// ===== 状态 =====
const isRunning = ref(false)
const hasResults = ref(false)
const currentJobId = ref('')
const videoRef = ref(null)
const canvasRef = ref(null)
const timelineRef = ref(null)
const displaySource = ref('face')

// 当前面部/语音结果
const faceResult = reactive({
  emotion: 'neutral',
  confidence: 0,
  probabilities: {},
  faceDetected: false,
})

const voiceResult = reactive({
  emotion: 'neutral',
  confidence: 0,
  probabilities: {},
  isSilence: true,
})

// 时间线
const timeline = ref([])

// ===== 计算属性 =====
const currentResult = computed(() => {
  return displaySource.value === 'face' ? faceResult : voiceResult
})

const currentEmotionIcon = computed(() => getEmotionIcon(currentResult.value.emotion))
const currentEmotionName = computed(() => getEmotionName(currentResult.value.emotion))
const currentEmotionColor = computed(() => getEmotionColor(currentResult.value.emotion))
const currentConfidence = computed(() => (currentResult.value.confidence * 100).toFixed(1))

function getCurrentProb(key) {
  return currentResult.value.probabilities?.[key] || 0
}

// ===== 媒体流和 WebSocket =====
let mediaStream = null
let ws = null
let videoInterval = null
let audioInterval = null
let audioContext = null
let audioWorkletNode = null

// ===== 开始识别 =====
async function startRecognition() {
  try {
    // 获取摄像头 + 麦克风
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
    })

    // 显示摄像头预览
    if (videoRef.value) {
      videoRef.value.srcObject = mediaStream
    }

    // 建立 WebSocket 连接
    ws = createRealtimeSocket({
      onOpen: () => {
        isRunning.value = true
        startVideoCapture()
        startAudioCapture()
      },
      onMessage: handleWsMessage,
      onClose: () => {
        if (isRunning.value) {
          stopRecognition()
        }
      },
      onError: (err) => {
        console.error('WebSocket 错误:', err)
      },
    })
  } catch (err) {
    console.error('获取摄像头/麦克风失败:', err)
    if (err.name === 'NotAllowedError') {
      alert('无法访问摄像头/麦克风。\n请在浏览器设置中允许权限后重试。')
    } else if (err.name === 'NotFoundError') {
      alert('未检测到摄像头或麦克风设备。')
    } else {
      alert(`获取设备失败: ${err.message}`)
    }
  }
}

// ===== 视频帧截取（每 500ms） =====
function startVideoCapture() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const video = videoRef.value

  videoInterval = setInterval(() => {
    if (!video || video.readyState < 2 || !ws || ws.readyState !== WebSocket.OPEN) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)

    // JPEG 压缩，质量 0.6
    const dataUrl = canvas.toDataURL('image/jpeg', 0.6)
    const base64 = dataUrl.split(',')[1]

    if (base64) {
      ws.send(JSON.stringify({
        type: 'video_frame',
        data: base64,
      }))
    }
  }, 500)
}

// ===== 音频录制（每 1.5s 发送 3 秒窗口） =====
async function startAudioCapture() {
  try {
    audioContext = new AudioContext({ sampleRate: 16000 })
    const source = audioContext.createMediaStreamSource(mediaStream)

    await audioContext.audioWorklet.addModule('/audio-processor.js')
    audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-recorder')
    source.connect(audioWorkletNode)

    // 每 1.5 秒获取缓冲数据并发送
    audioInterval = setInterval(() => {
      if (audioWorkletNode && ws && ws.readyState === WebSocket.OPEN) {
        audioWorkletNode.port.postMessage('flush')
      }
    }, 1500)

    audioWorkletNode.port.onmessage = (event) => {
      const pcmData = event.data
      if (!pcmData || pcmData.length === 0) return
      if (!ws || ws.readyState !== WebSocket.OPEN) return

      // Float32Array → base64
      const buffer = pcmData.buffer
      const bytes = new Uint8Array(buffer)
      let binary = ''
      for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i])
      }
      const base64 = btoa(binary)

      ws.send(JSON.stringify({
        type: 'audio_chunk',
        data: base64,
      }))
    }
  } catch (err) {
    console.error('音频录制初始化失败:', err)
    // 降级：不录音频，只做面部识别
  }
}

// ===== WebSocket 消息处理 =====
function handleWsMessage(msg) {
  if (msg.type === 'connected') {
    currentJobId.value = msg.job_id
    return
  }

  if (msg.type === 'face_emotion') {
    if (msg.face_detected) {
      faceResult.emotion = msg.emotion
      faceResult.confidence = msg.confidence
      faceResult.probabilities = { ...msg.probabilities }
      faceResult.faceDetected = true

      // 更新人脸框叠加
      drawFaceBox(msg.face_box, msg.emotion, msg.confidence)

      // 添加到时间线
      addToTimeline('face', msg.emotion, msg.confidence)
    } else {
      faceResult.faceDetected = false
      // 未检测到人脸，清除叠加
      clearFaceBox()
    }
  }

  if (msg.type === 'voice_emotion') {
    if (msg.silence) {
      voiceResult.isSilence = true
    } else {
      voiceResult.isSilence = false
      voiceResult.emotion = msg.emotion
      voiceResult.confidence = msg.confidence
      voiceResult.probabilities = { ...msg.probabilities }
      addToTimeline('voice', msg.emotion, msg.confidence)
    }
  }

  hasResults.value = true
}

// ===== 时间线 =====
function addToTimeline(source, emotion, confidence) {
  timeline.value.push({
    time: new Date().toLocaleTimeString(),
    source,
    emotion,
    confidence,
  })

  // 自动滚动到底部
  nextTick(() => {
    if (timelineRef.value) {
      timelineRef.value.scrollLeft = timelineRef.value.scrollWidth
    }
  })
}

// ===== Canvas 人脸框叠加 =====
function drawFaceBox(faceBox, emotion, confidence) {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')

  const video = videoRef.value
  if (!video) return

  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  if (!faceBox || faceBox.length !== 4) return

  const [x, y, w, h] = faceBox
  const color = getEmotionColor(emotion)

  // 绘制人脸框
  ctx.strokeStyle = color
  ctx.lineWidth = 3
  ctx.strokeRect(x, y, w, h)

  // 绘制标签背景
  const labelHeight = 28
  ctx.fillStyle = color + 'CC'
  ctx.fillRect(x, y - labelHeight, w, labelHeight)

  // 绘制标签文字
  ctx.fillStyle = '#FFFFFF'
  ctx.font = 'bold 14px -apple-system, BlinkMacSystemFont, sans-serif'
  ctx.textBaseline = 'middle'
  const label = `${getEmotionName(emotion)} ${(confidence * 100).toFixed(0)}%`
  ctx.fillText(label, x + 8, y - labelHeight / 2)
}

function clearFaceBox() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
}

// ===== 停止识别 =====
function stopRecognition() {
  // 停止定时器
  if (videoInterval) {
    clearInterval(videoInterval)
    videoInterval = null
  }
  if (audioInterval) {
    clearInterval(audioInterval)
    audioInterval = null
  }

  // 停止 AudioWorklet
  if (audioWorkletNode) {
    audioWorkletNode.port.postMessage('stop')
    audioWorkletNode.disconnect()
    audioWorkletNode = null
  }

  // 关闭 AudioContext
  if (audioContext) {
    audioContext.close()
    audioContext = null
  }

  // 关闭摄像头/麦克风
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop())
    mediaStream = null
  }

  // 清除视频预览
  if (videoRef.value) {
    videoRef.value.srcObject = null
  }

  // 关闭 WebSocket
  if (ws) {
    ws.close()
    ws = null
  }

  isRunning.value = false
  clearFaceBox()
}

// ===== 导出结果 =====
function exportResults() {
  if (!currentJobId.value) return
  const url = `/api/jobs/${currentJobId.value}/export`
  const a = document.createElement('a')
  a.href = url
  a.download = `realtime_emotions_${currentJobId.value}.xlsx`
  a.click()
}

// ===== 清理 =====
onUnmounted(() => {
  if (isRunning.value) {
    stopRecognition()
  }
})
</script>

<style scoped>
.realtime-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* 上半部分 */
.realtime-main {
  display: flex;
  gap: 16px;
}

/* 摄像头区域 */
.camera-section {
  flex: 1;
  min-width: 0;
}

.camera-wrapper {
  position: relative;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  aspect-ratio: 4 / 3;
}

.camera-wrapper video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.face-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.camera-placeholder {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: #8e8e93;
}

.placeholder-icon {
  font-size: 48px;
}

.placeholder-text {
  font-size: 14px;
}

.camera-status {
  margin-top: 8px;
  font-size: 12px;
  color: #8e8e93;
}

.status-active {
  color: #34c759;
}

.status-inactive {
  color: #8e8e93;
}

/* 结果面板 */
.result-panel {
  width: 320px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.current-emotion {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  border-left: 4px solid #8e8e93;
}

.emotion-icon {
  font-size: 48px;
  line-height: 1;
}

.emotion-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.emotion-label {
  font-size: 20px;
  font-weight: 600;
  color: #333;
}

.emotion-confidence {
  font-size: 14px;
  color: #8e8e93;
}

.status-hint {
  color: #8e8e93;
  font-weight: 500;
}

/* 数据源切换 */
.source-tabs {
  display: flex;
  gap: 8px;
}

.source-tab {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: #fff;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
  text-align: center;
}

.source-tab:hover {
  border-color: #007aff;
}

.source-tab.active {
  background: #007aff;
  color: #fff;
  border-color: #007aff;
}

/* 概率柱状图 */
.probability-bars {
  background: #fff;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.bar-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.bar-icon {
  font-size: 16px;
  width: 20px;
  text-align: center;
}

.bar-label {
  font-size: 12px;
  color: #666;
  width: 32px;
  flex-shrink: 0;
}

.bar-track {
  flex: 1;
  height: 12px;
  background: #f0f0f0;
  border-radius: 6px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 6px;
  transition: width 0.3s ease;
  min-width: 0;
}

.bar-value {
  font-size: 11px;
  color: #8e8e93;
  width: 42px;
  text-align: right;
  flex-shrink: 0;
}

/* 时间线 */
.timeline-section {
  background: #fff;
  border-radius: 12px;
  padding: 16px 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.timeline-title {
  font-size: 14px;
  font-weight: 600;
  color: #333;
  margin-bottom: 12px;
}

.timeline-scroll {
  display: flex;
  gap: 4px;
  overflow-x: auto;
  padding: 8px 0;
  scrollbar-width: thin;
}

.timeline-scroll::-webkit-scrollbar {
  height: 4px;
}

.timeline-scroll::-webkit-scrollbar-thumb {
  background: #c7c7cc;
  border-radius: 2px;
}

.timeline-entry {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  cursor: default;
}

.timeline-emoji {
  font-size: 16px;
}

.timeline-legend {
  display: flex;
  gap: 12px;
  margin-top: 10px;
  flex-wrap: wrap;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: #666;
}

.legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

/* 控制按钮 */
.controls {
  display: flex;
  gap: 12px;
  justify-content: center;
}

.btn-start,
.btn-stop,
.btn-export {
  padding: 12px 32px;
  border: none;
  border-radius: 10px;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-start {
  background: #34c759;
  color: #fff;
}

.btn-start:hover {
  background: #2da44e;
}

.btn-stop {
  background: #ff3b30;
  color: #fff;
}

.btn-stop:hover {
  background: #d63029;
}

.btn-export {
  background: #007aff;
  color: #fff;
}

.btn-export:hover {
  background: #0066d6;
}

/* 响应式 */
@media (max-width: 768px) {
  .realtime-main {
    flex-direction: column;
  }

  .result-panel {
    width: 100%;
  }
}
</style>
