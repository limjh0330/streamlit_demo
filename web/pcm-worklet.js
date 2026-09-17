/**
 * pcm-worklet.js — AudioWorkletProcessor
 *
 * 마이크 → 128 프레임 단위 float32 → (필요시) 16 kHz 리샘플 → PCM16
 * → 지정한 블록 크기(기본 100 ms)로 모아 메인 스레드에 postMessage.
 *
 * 오디오 렌더 스레드에서 도는 코드이므로 할당과 연산을 최소로 유지한다.
 */
class PCMWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || 16000;
    this.blockMs = opts.blockMs || 100;

    // sampleRate 는 AudioWorkletGlobalScope 전역 (컨텍스트의 실제 샘플레이트)
    this.ratio = sampleRate / this.targetRate;
    this.needsResample = Math.abs(this.ratio - 1) > 1e-6;

    this.blockSize = Math.round((this.targetRate * this.blockMs) / 1000);
    this.out = new Int16Array(this.blockSize);
    this.outLen = 0;

    // 리샘플 상태: 소스 상의 소수점 위치와 직전 샘플(블록 경계 보간용)
    this.pos = 0;
    this.last = 0;

    this.muted = false;
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === 'mute') this.muted = !!e.data.value;
    };
  }

  /** 선형 보간 리샘플. 블록 경계를 넘어가는 위치는 this.pos 로 이어간다. */
  pushResampled(input) {
    const n = input.length;
    while (this.pos < n) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const a = i === 0 ? this.last : input[i - 1];
      const b = input[i];
      this.emit(a + (b - a) * frac);
      this.pos += this.ratio;
    }
    this.pos -= n;
    this.last = input[n - 1];
  }

  pushDirect(input) {
    for (let i = 0; i < input.length; i++) this.emit(input[i]);
  }

  emit(sample) {
    const s = sample > 1 ? 1 : sample < -1 ? -1 : sample;
    this.out[this.outLen++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.outLen === this.blockSize) {
      const buf = this.out.buffer;
      this.port.postMessage({ type: 'audio', payload: buf }, [buf]);
      this.out = new Int16Array(this.blockSize);
      this.outLen = 0;
    }
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || this.muted) return true;

    // 레벨 미터용 RMS (100 ms 마다 보내면 충분하지만 계산이 싸서 매 블록 전송)
    let sum = 0;
    for (let i = 0; i < channel.length; i++) sum += channel[i] * channel[i];
    this.port.postMessage({ type: 'level', rms: Math.sqrt(sum / channel.length) });

    if (this.needsResample) this.pushResampled(channel);
    else this.pushDirect(channel);
    return true;
  }
}

registerProcessor('pcm-worklet', PCMWorklet);
