import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
})

export function uploadVideo(file) {
  const formData = new FormData()
  formData.append('file', file)
  return api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export function getJobResult(jobId) {
  return api.get(`/jobs/${jobId}/result`)
}

export function connectSSE(jobId, onMessage, onError) {
  const eventSource = new EventSource(`/api/jobs/${jobId}/progress`)
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
      if (data.status === 'done' || data.status === 'failed') {
        eventSource.close()
      }
    } catch (e) {
      console.error('SSE parse error:', e)
    }
  }
  eventSource.onerror = (err) => {
    eventSource.close()
    if (onError) onError(err)
  }
  return eventSource
}

export function getSegmentVideoUrl(jobId, segmentIndex) {
  return `/api/jobs/${jobId}/video/${segmentIndex}`
}

export function getOriginalVideoUrl(jobId) {
  return `/api/jobs/${jobId}/original-video`
}

export function getExportExcelUrl(jobId) {
  return `/api/jobs/${jobId}/export`
}
