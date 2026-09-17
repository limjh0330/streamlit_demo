/**
 * recorder.js — 브라우저 마이크 → WebSocket → STT 서버
 *
 *   getUserMedia() → AudioContext(16 kHz) → AudioWorklet(pcm-worklet)
 *   → PCM16 블록 → WebSocket 바이너리 전송
 *   ← JSON(partial / final / metrics) → 화면 갱신
 */
'use strict';

const $ = (id) => document.getElementById(id);

const ui = {
  start: $('start'),
  stop: $('stop'),
  engine: $('engine'),
  modelSize: $('model-size'),
  language: $('language'),
  window: $('window'),
  overlap: $('overlap'),
  silence: $('silence'),
  correction: $('correction'),
  status: $('status'),
  level: $('level-bar'),
  finals: $('finals'),
  stable: $('stable'),
  partial: $('partial'),
  metrics: $('metrics'),
  download: $('download'),
  empty: $('empty'),
};

const state = {
  ws: null,
  ctx: null,
  node: null,
  stream: null,
  sessionId: null,
  running: false,
  finals: [],
  sent: 0,
};

function setStatus(text, kind = 'idle') {
  ui.status.textContent = text;
  ui.status.dataset.kind = kind;
}

function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/ws`;
}

// ------------------------------------------------------------------ 시작
async function start() {
  if (state.running) return;
  ui.start.disabled = true;
  setStatus('마이크 권한 요청 중…', 'busy');

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    setStatus(`마이크를 열 수 없습니다: ${err.message}`, 'error');
    ui.start.disabled = false;
    return;
  }

  // 16 kHz 컨텍스트를 요청한다. 브라우저가 거부하면 워클릿이 리샘플한다.
  try {
    state.ctx = new AudioContext({ sampleRate: 16000 });
  } catch {
    state.ctx = new AudioContext();
  }
  await state.ctx.resume();
  await state.ctx.audioWorklet.addModule('pcm-worklet.js');

  setStatus('서버 연결 중…', 'busy');
  await openSocket();
}

function openSocket() {
  return new Promise((resolve) => {
    const ws = new WebSocket(wsUrl());
    ws.binaryType = 'arraybuffer';
    state.ws = ws;

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: 'start',
          engine: ui.engine.value,
          model_size: ui.modelSize.value,
          language: ui.language.value || null,
          sample_rate: 16000,
          window_sec: parseFloat(ui.window.value),
          overlap_sec: parseFloat(ui.overlap.value),
          silence_sec: parseFloat(ui.silence.value),
          medical_correction: ui.correction.checked,
        })
      );
      setStatus('모델 로드 중… (최초 1회는 수십 초 걸릴 수 있습니다)', 'busy');
      resolve();
    };

    ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
    ws.onerror = () => setStatus('WebSocket 오류', 'error');
    ws.onclose = () => {
      if (state.running) teardownAudio();
      state.running = false;
      ui.start.disabled = false;
      ui.stop.disabled = true;
    };
  });
}

// ------------------------------------------------------------------ 오디오
async function beginCapture() {
  const source = state.ctx.createMediaStreamSource(state.stream);
  state.node = new AudioWorkletNode(state.ctx, 'pcm-worklet', {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    processorOptions: { targetRate: 16000, blockMs: 100 },
  });

  state.node.port.onmessage = ({ data }) => {
    if (data.type === 'level') {
      const db = 20 * Math.log10(data.rms + 1e-9);
      const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
      ui.level.style.width = `${pct}%`;
      return;
    }
    if (data.type === 'audio' && state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(data.payload);
      state.sent += data.payload.byteLength;
    }
  };

  source.connect(state.node);
  state.running = true;
  ui.stop.disabled = false;
  ui.empty.hidden = true;
}

function teardownAudio() {
  if (state.node) {
    state.node.port.onmessage = null;
    state.node.disconnect();
    state.node = null;
  }
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  if (state.ctx) {
    state.ctx.close();
    state.ctx = null;
  }
  ui.level.style.width = '0%';
}

function stop() {
  ui.stop.disabled = true;
  setStatus('마무리 중… (남은 오디오 전사)', 'busy');
  if (state.node) state.node.port.postMessage({ type: 'mute', value: true });
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'stop' }));
  } else {
    teardownAudio();
  }
}

// ------------------------------------------------------------------ 수신
function handleMessage(msg) {
  switch (msg.type) {
    case 'ready':
      state.sessionId = msg.session_id;
      setStatus(`녹음 중 · ${msg.engine} · ${msg.session_id}`, 'live');
      beginCapture();
      break;

    case 'partial':
      ui.stable.textContent = msg.stable || '';
      ui.partial.textContent = msg.partial || '';
      break;

    case 'final':
      state.finals.push(msg);
      renderFinals();
      ui.partial.textContent = '';
      break;

    case 'metrics':
      renderMetrics(msg);
      break;

    case 'error':
      setStatus(`오류: ${msg.message}`, 'error');
      break;

    case 'closed':
      teardownAudio();
      state.running = false;
      ui.start.disabled = false;
      ui.stop.disabled = true;
      ui.partial.textContent = '';
      setStatus(`종료 · 전사 저장됨 (${msg.session_id})`, 'idle');
      renderMetrics(msg.summary || {});
      enableDownload();
      break;
  }
}

function renderFinals() {
  ui.finals.innerHTML = '';
  state.finals.forEach((f) => {
    const row = document.createElement('div');
    row.className = 'utterance';
    const time = document.createElement('span');
    time.className = 'time';
    time.textContent = `${f.start.toFixed(1)}s → ${f.end.toFixed(1)}s`;
    const text = document.createElement('span');
    text.textContent = f.text;
    row.append(time, text);
    ui.finals.append(row);
  });
  ui.finals.scrollTop = ui.finals.scrollHeight;
}

const METRIC_LABELS = {
  rtf_mean: ['RTF', (v) => v.toFixed(2)],
  first_partial_ms: ['첫 partial', (v) => `${Math.round(v)} ms`],
  final_latency_ms: ['final 지연', (v) => `${Math.round(v)} ms`],
  revision_rate: ['revision', (v) => `${(v * 100).toFixed(0)} %`],
  cpu_mean_pct: ['CPU', (v) => `${v.toFixed(0)} %`],
  rss_mean_mb: ['메모리', (v) => `${Math.round(v)} MB`],
  audio_sec: ['오디오', (v) => `${v.toFixed(0)} s`],
  utterances: ['발화', (v) => `${v}`],
};

function renderMetrics(m) {
  ui.metrics.innerHTML = '';
  Object.entries(METRIC_LABELS).forEach(([key, [label, fmt]]) => {
    if (m[key] === undefined || m[key] === null) return;
    const cell = document.createElement('div');
    cell.className = 'metric';
    cell.innerHTML = `<dt>${label}</dt><dd>${fmt(m[key])}</dd>`;
    ui.metrics.append(cell);
  });
}

function enableDownload() {
  const lines = state.finals.map(
    (f) => `[${f.start.toFixed(1)}s → ${f.end.toFixed(1)}s] ${f.text}`
  );
  const body = `${lines.join('\n')}\n\n---\n${state.finals
    .map((f) => f.text)
    .join(' ')}\n`;
  const blob = new Blob([body], { type: 'text/plain;charset=utf-8' });
  ui.download.href = URL.createObjectURL(blob);
  ui.download.download = `${state.sessionId || 'transcript'}.txt`;
  ui.download.hidden = false;
}

// ------------------------------------------------------------------ 바인딩
ui.start.addEventListener('click', start);
ui.stop.addEventListener('click', stop);
ui.engine.addEventListener('change', () => {
  ui.modelSize.disabled = ui.engine.value !== 'whisper';
});

fetch('/api/engines')
  .then((r) => r.json())
  .then((info) => {
    Object.entries(info.engines).forEach(([name, meta]) => {
      const opt = [...ui.engine.options].find((o) => o.value === name);
      if (opt && !meta.ready) {
        opt.disabled = true;
        opt.textContent += ' (미설치)';
      }
    });
  })
  .catch(() => {});
