/**
 * AudioWorklet 处理器
 * 用于实时采集麦克风 PCM 数据（float32, 单声道）
 * 主线程通过 postMessage('flush') 获取缓冲的音频数据
 */
class AudioRecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this._buffer = new Float32Array(0)
    this._isRecording = true

    this.port.onmessage = (event) => {
      if (event.data === 'flush') {
        const data = this._buffer
        this._buffer = new Float32Array(0)
        this.port.postMessage(data)
      } else if (event.data === 'stop') {
        this._isRecording = false
      }
    }
  }

  process(inputs) {
    if (!this._isRecording) return false

    const input = inputs[0]
    if (input && input[0]) {
      const newData = input[0] // Float32Array, 单声道
      const newBuffer = new Float32Array(this._buffer.length + newData.length)
      newBuffer.set(this._buffer)
      newBuffer.set(newData, this._buffer.length)
      this._buffer = newBuffer
    }
    return true
  }
}

registerProcessor('audio-recorder', AudioRecorderProcessor)
