<template>
  <div class="app">
    <header class="app-header">
      <h1>教师情感识别系统</h1>
      <nav class="mode-tabs">
        <button
          :class="['tab', { active: mode === 'upload' }]"
          @click="switchMode('upload')"
        >
          📹 视频分析
        </button>
        <button
          :class="['tab', { active: mode === 'realtime' }]"
          @click="switchMode('realtime')"
        >
          📷 实时识别
        </button>
      </nav>
    </header>

    <main class="app-main">
      <!-- 视频分析模式 -->
      <template v-if="mode === 'upload'">
        <VideoUpload
          :disabled="processing"
          @upload-start="onUploadStart"
          @upload-success="onUploadSuccess"
          @upload-error="onUploadError"
        />

        <ProgressPanel
          :visible="processing"
          :progress="progress"
          :message="progressMessage"
          :status="jobStatus"
        />

        <div v-if="error" class="error-banner">{{ error }}</div>

        <VideoPlayer
          v-if="jobId"
          :job-id="jobId"
          :segments="result ? result.segments : []"
          :label-mode="labelMode"
        />

        <div v-if="result" class="mode-toggle">
          <span class="mode-label">标签模式:</span>
          <button
            class="mode-btn"
            :class="{ active: labelMode === 'e2v' }"
            @click="labelMode = 'e2v'"
          >emotion2vec (9类)</button>
          <button
            class="mode-btn"
            :class="{ active: labelMode === 'teacher' }"
            @click="labelMode = 'teacher'"
          >教师情感 (4类)</button>
          <a :href="getExportExcelUrl(jobId)" class="export-btn" download>导出 Excel</a>
        </div>

        <div class="mapping-rules">
          <h3 class="mapping-title">9类→4类映射规则</h3>
          <table class="mapping-table">
            <thead>
              <tr><th>emotion2vec 原始标签</th><th>→</th><th>教师情感标签</th></tr>
            </thead>
            <tbody>
              <tr><td><span class="e2v-tag" :style="{ background: '#AF52DE' }">恐惧(2)</span> <span class="e2v-tag" :style="{ background: '#FF3B30' }">愤怒(0)</span></td><td>→</td><td><span class="teacher-tag" :style="{ background: '#FF3B30' }">紧张焦虑(3)</span></td></tr>
              <tr><td><span class="e2v-tag" :style="{ background: '#FF9500' }">开心(3)</span> <span class="e2v-tag" :style="{ background: '#FFCC00' }">惊讶(7)</span></td><td>→</td><td><span class="teacher-tag" :style="{ background: '#FF9500' }">热情投入(0)</span></td></tr>
              <tr><td><span class="e2v-tag" :style="{ background: '#007AFF' }">中性(4)</span> <span class="e2v-tag" :style="{ background: '#8E8E93' }">其他(5)</span> <span class="e2v-tag" :style="{ background: '#C7C7CC' }">未知(8)</span></td><td>→</td><td><span class="teacher-tag" :style="{ background: '#007AFF' }">平稳中性(1)</span></td></tr>
              <tr><td><span class="e2v-tag" :style="{ background: '#A2845E' }">厌恶(1)</span> <span class="e2v-tag" :style="{ background: '#5856D6' }">悲伤(6)</span></td><td>→</td><td><span class="teacher-tag" :style="{ background: '#8E8E93' }">消极低落(2)</span></td></tr>
            </tbody>
          </table>
        </div>

        <EmotionTimeline
          v-if="result"
          :segments="result.segments"
          :total-duration="result.total_duration"
          :label-mode="labelMode"
        />

        <ResultTable
          v-if="result"
          :segments="result.segments"
          :label-mode="labelMode"
        />
      </template>

      <!-- 实时识别模式 -->
      <template v-if="mode === 'realtime'">
        <RealtimeRecognition />
      </template>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import VideoUpload from './components/VideoUpload.vue'
import VideoPlayer from './components/VideoPlayer.vue'
import ProgressPanel from './components/ProgressPanel.vue'
import EmotionTimeline from './components/EmotionTimeline.vue'
import ResultTable from './components/ResultTable.vue'
import RealtimeRecognition from './components/RealtimeRecognition.vue'
import { connectSSE, getJobResult, getExportExcelUrl } from './api.js'

const mode = ref('upload')  // 'upload' | 'realtime'

const processing = ref(false)
const progress = ref(0)
const progressMessage = ref('')
const jobStatus = ref('')
const error = ref('')
const result = ref(null)
const jobId = ref('')
const labelMode = ref('e2v')

function switchMode(newMode) {
  mode.value = newMode
}

function onUploadStart() {
  error.value = ''
  result.value = null
  jobId.value = ''
  processing.value = true
  progress.value = 0
  progressMessage.value = 'Uploading...'
}

function onUploadSuccess(jobData) {
  jobId.value = jobData.job_id
  progressMessage.value = 'Processing...'

  connectSSE(
    jobData.job_id,
    (data) => {
      progress.value = data.progress
      progressMessage.value = data.message
      jobStatus.value = data.status

      if (data.status === 'done') {
        fetchResult(jobData.job_id)
      } else if (data.status === 'failed') {
        processing.value = false
        error.value = data.message || 'Processing failed'
      }
    },
    () => {
      fetchResult(jobData.job_id)
    },
  )
}

async function fetchResult(jobId) {
  try {
    const res = await getJobResult(jobId)
    result.value = res.data
  } catch (e) {
    error.value = 'Failed to fetch results'
  } finally {
    processing.value = false
  }
}

function onUploadError(msg) {
  processing.value = false
  error.value = msg
}
</script>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f7; color: #333; }
.app { min-height: 100vh; }
.app-header {
  background: #fff; padding: 16px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  position: sticky; top: 0; z-index: 10;
  display: flex; align-items: center; gap: 24px;
}
.app-header h1 { font-size: 18px; font-weight: 600; }
.mode-tabs { display: flex; gap: 4px; }
.tab {
  padding: 6px 16px; border: 1px solid #ddd; border-radius: 8px;
  background: #fff; font-size: 13px; cursor: pointer; transition: all 0.15s;
}
.tab:hover { border-color: #007AFF; color: #007AFF; }
.tab.active { background: #007AFF; color: #fff; border-color: #007AFF; }
.app-main { max-width: 900px; margin: 24px auto; padding: 0 16px; display: flex; flex-direction: column; gap: 16px; }
.error-banner {
  background: #fff0f0; border: 1px solid #ffcdd2; border-radius: 8px;
  padding: 12px 16px; color: #c62828; font-size: 14px;
}
.mode-toggle {
  display: flex; align-items: center; gap: 8px;
  background: #fff; border-radius: 12px; padding: 12px 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.mode-label { font-size: 13px; color: #666; }
.mode-btn {
  padding: 4px 14px; border: 1px solid #ddd; border-radius: 16px;
  background: #fff; font-size: 12px; cursor: pointer; transition: all 0.15s;
}
.mode-btn.active { background: #007AFF; color: #fff; border-color: #007AFF; }
.export-btn {
  margin-left: auto; padding: 4px 14px; border: 1px solid #34C759; border-radius: 16px;
  background: #34C759; color: #fff; font-size: 12px; text-decoration: none; transition: all 0.15s;
}
.export-btn:hover { background: #2DA44E; }
.mapping-rules {
  background: #fff; border-radius: 12px; padding: 20px 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.mapping-title { font-size: 15px; font-weight: 600; color: #333; margin: 0 0 12px; }
.mapping-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.mapping-table th { text-align: left; padding: 8px; border-bottom: 2px solid #eee; color: #666; font-weight: 600; }
.mapping-table td { padding: 8px; border-bottom: 1px solid #f0f0f0; }
.e2v-tag, .teacher-tag {
  display: inline-block; padding: 1px 8px; border-radius: 10px;
  color: #fff; font-size: 11px; margin-right: 4px;
}
</style>
