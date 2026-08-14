/* Microphone capture: Float32 -> 16kHz mono PCM16, framed for streaming STT. */

const FRAME_SAMPLES = 1600; // 100ms at 16kHz

class PCMCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Int16Array(FRAME_SAMPLES);
    this._n = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      const s = Math.max(-1, Math.min(1, channel[i]));
      this._buf[this._n++] = s < 0 ? s * 0x8000 : s * 0x7fff;

      if (this._n === FRAME_SAMPLES) {
        const out = this._buf.buffer;
        this.port.postMessage(out, [out]);
        this._buf = new Int16Array(FRAME_SAMPLES);
        this._n = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PCMCapture);
