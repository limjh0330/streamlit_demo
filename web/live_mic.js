/**
 * live_mic.js — 브라우저 마이크 → WebSocket → RunPod STT 백엔드 (CCv2 컴포넌트)
 *
 *   getUserMedia()  →  AudioWorklet  →  PCM16 / 16 kHz / mono, 100~500 ms chunk
 *      →  WebSocket(binary)  →  RunPod: streaming ASR
 *      ←  JSON(partial / final / metrics)  →  이 화면
 *
 * CCv2 컴포넌트는 iframe 이 아니라 앱 문서 안에서 실행된다. 덕분에 마이크 권한
 * 위임 없이 getUserMedia / AudioWorklet / WebSocket 을 그대로 쓸 수 있다.
 *
 * 부분 전사(partial)는 초당 여러 번 갱신되므로 Python 으로 올리지 않고 여기서
 * 직접 그린다. 세션이 끝날 때만 setStateValue 로 결과를 한 번 넘긴다.
 */

const WORKLET_NAME = "er-pcm-worklet";

/** AudioWorklet 은 URL 로만 로드할 수 있어 소스를 Blob URL 로 만들어 넘긴다. */
const WORKLET_SOURCE = `
class PCMWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.targetRate = opts.targetRate || 16000;
    this.blockSize = Math.round((this.targetRate * (opts.blockMs || 100)) / 1000);

    // sampleRate 는 AudioWorkletGlobalScope 전역(컨텍스트의 실제 샘플레이트)
    this.ratio = sampleRate / this.targetRate;
    this.needsResample = Math.abs(this.ratio - 1) > 1e-6;

    this.out = new Int16Array(this.blockSize);
    this.outLen = 0;
    this.pos = 0;     // 리샘플 시 소스 상의 소수점 위치 (블록 경계를 넘어 이어짐)
    this.last = 0;    // 직전 블록의 마지막 샘플 (경계 보간용)

    this.levelSum = 0;
    this.levelCount = 0;
    this.muted = false;

    this.port.onmessage = (e) => {
      if (e.data && e.data.type === "mute") this.muted = !!e.data.value;
    };
  }

  emit(sample) {
    const s = sample > 1 ? 1 : sample < -1 ? -1 : sample;
    this.out[this.outLen++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.outLen === this.blockSize) {
      const buf = this.out.buffer;
      this.port.postMessage({ type: "audio", payload: buf }, [buf]);
      this.out = new Int16Array(this.blockSize);
      this.outLen = 0;
    }
  }

  /** 선형 보간 리샘플. 대개 AudioContext 가 이미 16 kHz 라 쓰이지 않는다. */
  pushResampled(input) {
    const n = input.length;
    while (this.pos < n) {
      const i = Math.floor(this.pos);
      const a = i === 0 ? this.last : input[i - 1];
      this.emit(a + (input[i] - a) * (this.pos - i));
      this.pos += this.ratio;
    }
    this.pos -= n;
    this.last = input[n - 1];
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || this.muted) return true;

    // 레벨 미터: 렌더 블록(128 프레임)마다 보내면 너무 잦아 50 ms 로 묶는다.
    for (let i = 0; i < channel.length; i++) this.levelSum += channel[i] * channel[i];
    this.levelCount += channel.length;
    if (this.levelCount >= sampleRate / 20) {
      this.port.postMessage({ type: "level", rms: Math.sqrt(this.levelSum / this.levelCount) });
      this.levelSum = 0;
      this.levelCount = 0;
    }

    if (this.needsResample) this.pushResampled(channel);
    else for (let i = 0; i < channel.length; i++) this.emit(channel[i]);
    return true;
  }
}
registerProcessor("${WORKLET_NAME}", PCMWorklet);
`;

const WORKLET_URL = URL.createObjectURL(
  new Blob([WORKLET_SOURCE], { type: "text/javascript" })
);

const METRIC_LABELS = {
  rtf_mean: ["RTF", (v) => v.toFixed(2)],
  first_partial_ms: ["첫 partial", (v) => `${Math.round(v)} ms`],
  final_latency_ms: ["final 지연", (v) => `${Math.round(v)} ms`],
  revision_rate: ["revision", (v) => `${(v * 100).toFixed(0)} %`],
  audio_sec: ["오디오", (v) => `${v.toFixed(0)} s`],
  utterances: ["발화", (v) => `${v}`],
};

/**
 * 백엔드 WebSocket 주소. 설정이 비어 있으면 주소창에서 유도한다.
 * RunPod 은 포트마다 별도 호스트(<podId>-<port>.proxy.runpod.net)를 준다.
 */
function resolveWsUrl(config) {
  if (config.ws_url) return config.ws_url;
  const port = config.backend_port || 8000;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const proxy = location.hostname.match(/^(.+)-\d+\.proxy\.runpod\.net$/);
  if (proxy) return `${proto}//${proxy[1]}-${port}.proxy.runpod.net/ws`;
  return `${proto}//${location.hostname}:${port}/ws`;
}

class LiveMic {
  constructor(root) {
    this.root = root;
    this.config = {};
    this.emitResult = () => {};

    const $ = (id) => root.querySelector(`#${id}`);
    this.ui = {
      start: $("lm-start"),
      stop: $("lm-stop"),
      status: $("lm-status"),
      level: $("lm-level-bar"),
      finals: $("lm-finals"),
      stable: $("lm-stable"),
      partial: $("lm-partial"),
      metrics: $("lm-metrics"),
      empty: $("lm-empty"),
    };

    this.ui.start.onclick = () => this.start();
    this.ui.stop.onclick = () => this.stop();

    this.reset();
  }

  reset() {
    this.ws = null;
    this.ctx = null;
    this.node = null;
    this.stream = null;
    this.running = false;
    this.finals = [];
    this.sessionId = null;
  }

  /** 리런마다 호출된다. 진행 중인 세션은 건드리지 않고 설정만 갈아끼운다. */
  update(config, emitResult) {
    this.config = config || {};
    this.emitResult = emitResult;
    this.ui.start.disabled = this.running;
    this.ui.stop.disabled = !this.running;
  }

  setStatus(text, kind = "idle") {
    this.ui.status.textContent = text;
    this.ui.status.dataset.kind = kind;
  }

  // ---------------------------------------------------------------- 시작
  async start() {
    if (this.running) return;
    this.ui.start.disabled = true;
    this.finals = [];
    this.ui.finals.replaceChildren();
    this.ui.stable.textContent = "";
    this.ui.partial.textContent = "";
    this.ui.metrics.replaceChildren();

    if (!navigator.mediaDevices || !window.isSecureContext) {
      this.setStatus("HTTPS 또는 localhost 에서만 마이크를 쓸 수 있습니다.", "error");
      this.ui.start.disabled = false;
      return;
    }

    this.setStatus("마이크 권한 요청 중…", "busy");
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (err) {
      this.setStatus(`마이크를 열 수 없습니다: ${err.message}`, "error");
      this.ui.start.disabled = false;
      return;
    }

    // 16 kHz 컨텍스트를 요청한다. 브라우저가 거부하면 워클릿이 리샘플한다.
    try {
      this.ctx = new AudioContext({ sampleRate: 16000 });
    } catch {
      this.ctx = new AudioContext();
    }
    await this.ctx.resume();
    await this.ctx.audioWorklet.addModule(WORKLET_URL);

    this.setStatus("백엔드 연결 중…", "busy");
    this.openSocket();
  }

  openSocket() {
    let ws;
    try {
      ws = new WebSocket(resolveWsUrl(this.config));
    } catch (err) {
      this.setStatus(`백엔드에 연결할 수 없습니다: ${err.message}`, "error");
      this.teardownAudio();
      this.ui.start.disabled = false;
      return;
    }
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "start",
          sample_rate: 16000,
          engine: this.config.engine,
          model_size: this.config.model_size,
          language: this.config.language || null,
          window_sec: this.config.window_sec,
          overlap_sec: this.config.overlap_sec,
          silence_sec: this.config.silence_sec,
          vad_threshold_db: this.config.vad_threshold_db,
          medical_correction: this.config.medical_correction,
          save_wav: this.config.save_wav,
        })
      );
      this.setStatus("모델 로드 중… (최초 1회는 수십 초 걸릴 수 있습니다)", "busy");
    };

    ws.onmessage = (event) => this.handle(JSON.parse(event.data));
    ws.onerror = () => this.setStatus("WebSocket 오류 — 백엔드 주소를 확인하세요.", "error");
    ws.onclose = () => {
      if (this.running) this.teardownAudio();
      this.running = false;
      this.ui.start.disabled = false;
      this.ui.stop.disabled = true;
    };
  }

  // ---------------------------------------------------------------- 캡처
  beginCapture() {
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, WORKLET_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: {
        targetRate: 16000,
        blockMs: this.config.chunk_ms || 100,
      },
    });

    this.node.port.onmessage = ({ data }) => {
      if (data.type === "level") {
        const db = 20 * Math.log10(data.rms + 1e-9);
        this.ui.level.style.width = `${Math.max(0, Math.min(100, ((db + 60) / 60) * 100))}%`;
        return;
      }
      if (data.type === "audio" && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(data.payload);
      }
    };

    source.connect(this.node);
    this.running = true;
    this.ui.stop.disabled = false;
    this.ui.empty.hidden = true;
  }

  teardownAudio() {
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
      this.node = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.ctx) {
      this.ctx.close();
      this.ctx = null;
    }
    this.ui.level.style.width = "0%";
  }

  stop() {
    this.ui.stop.disabled = true;
    this.setStatus("마무리 중… (남은 오디오 전사)", "busy");
    // 오디오는 즉시 끊되 소켓은 열어둔다. 백엔드가 잔여분을 전사해 closed 를 보낸다.
    if (this.node) this.node.port.postMessage({ type: "mute", value: true });
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop" }));
    } else {
      this.teardownAudio();
      this.running = false;
      this.ui.start.disabled = false;
    }
  }

  destroy() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop" }));
    }
    if (this.ws) this.ws.close();
    this.teardownAudio();
    this.reset();
  }

  // ---------------------------------------------------------------- 수신
  handle(msg) {
    switch (msg.type) {
      case "ready":
        this.sessionId = msg.session_id;
        this.setStatus(`녹음 중 · ${msg.engine} · ${msg.session_id}`, "live");
        this.beginCapture();
        break;

      case "partial":
        this.ui.stable.textContent = msg.stable || "";
        this.ui.partial.textContent = msg.partial || "";
        break;

      case "final":
        this.finals.push(msg);
        this.renderFinals();
        this.ui.partial.textContent = "";
        break;

      case "metrics":
        this.renderMetrics(msg);
        break;

      case "error":
        this.setStatus(`오류: ${msg.message}`, "error");
        break;

      case "closed":
        this.teardownAudio();
        this.running = false;
        this.ui.start.disabled = false;
        this.ui.stop.disabled = true;
        this.ui.partial.textContent = "";
        this.setStatus(`종료 · 전사 저장됨 (${msg.session_id})`, "idle");
        this.renderMetrics(msg.summary || {});
        // 결과를 Python 으로 한 번만 올린다 → 리런 1회
        this.emitResult({
          session_id: msg.session_id,
          text: msg.text || "",
          utterances: msg.utterances || [],
          metrics: msg.summary || {},
          wav: msg.wav || null,
          transcript: msg.transcript || null,
        });
        break;
    }
  }

  renderFinals() {
    const rows = this.finals.map((f) => {
      const row = document.createElement("div");
      row.className = "lm-utterance";
      const time = document.createElement("span");
      time.className = "lm-time";
      time.textContent = `${f.start.toFixed(1)}s → ${f.end.toFixed(1)}s`;
      const text = document.createElement("span");
      text.textContent = f.text;
      row.append(time, text);
      return row;
    });
    this.ui.finals.replaceChildren(...rows);
    this.ui.finals.scrollTop = this.ui.finals.scrollHeight;
  }

  renderMetrics(m) {
    const cells = [];
    for (const [key, [label, fmt]] of Object.entries(METRIC_LABELS)) {
      if (m[key] === undefined || m[key] === null) continue;
      const cell = document.createElement("div");
      cell.className = "lm-metric";
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = fmt(m[key]);
      cell.append(dt, dd);
      cells.push(cell);
    }
    this.ui.metrics.replaceChildren(...cells);
  }
}

/** parentElement 당 인스턴스 하나. 리런으로 이 함수가 다시 불려도 재생성하지 않는다. */
const INSTANCES = new WeakMap();

export default function (component) {
  const { parentElement, data, setStateValue } = component;

  let app = INSTANCES.get(parentElement);
  if (!app) {
    app = new LiveMic(parentElement);
    INSTANCES.set(parentElement, app);
  }
  app.update(data, (result) => setStateValue("result", result));

  // 언마운트(페이지 이동 등) 시에만 호출된다 — 리런마다 불리지 않는다.
  return () => {
    app.destroy();
    INSTANCES.delete(parentElement);
  };
}
