"""[Real E2E test] 실행 중인 STT 서버에 WAV 를 실시간으로 흘려 보내 전사를 확인한다.

일반 `pytest` 에는 포함되지 않는다. 실제 모델(기본 funasr_mlt_nano)과 GPU 로
추론 경로 전체를 확인할 때 서버를 띄워 둔 상태에서 수동으로 실행한다.

    python -m scripts.stt_client recordings/sample.wav
    python -m scripts.stt_client sample.wav --url wss://<POD_ID>-8000.proxy.runpod.net/ws/v1/stt/stream
    python -m scripts.stt_client sample.wav --fast            # 실시간 대기 없이 전송

순서:
  1. /api/v1/stt/health/ready 가 READY 가 될 때까지 기다린다(--no-wait 로 생략)
  2. 100 ms(3,200 B) PCM16 프레임을 실시간 속도로 전송
  3. {"type":"close"} 전송 → done 까지 수신
  4. partial/final 개수, 소요 시간을 요약. done 과 final 을 받으면 exit 0

입력은 PCM16 WAV 입니다. 16 kHz mono 가 아니면 변환해서 보냅니다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

import numpy as np

from stt.audio.resampler import float32_to_pcm16, pcm16_to_float32, resample
from stt.config import SAMPLE_RATE

FRAME_BYTES = 3200                     # 100 ms = 1600 samples × 2 bytes


def load_pcm(path: str) -> tuple[bytes, float]:
    """WAV → (PCM16 LE 16 kHz mono 바이트, 길이 초)."""
    with wave.open(path) as wf:
        if wf.getsampwidth() != 2:
            raise SystemExit(f"{path}: 16-bit PCM WAV 만 지원합니다")
        rate, channels = wf.getframerate(), wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    if (rate, channels) == (SAMPLE_RATE, 1):
        return raw, len(raw) / 2 / SAMPLE_RATE
    audio = pcm16_to_float32(raw)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = resample(np.ascontiguousarray(audio, dtype=np.float32), rate, SAMPLE_RATE)
    return float32_to_pcm16(audio), audio.size / SAMPLE_RATE


def ready_url(ws_url: str) -> str:
    """ws(s)://host/ws/v1/stt/stream → http(s)://host/api/v1/stt/health/ready"""
    u = urllib.parse.urlparse(ws_url)
    scheme = "https" if u.scheme == "wss" else "http"
    return f"{scheme}://{u.netloc}/api/v1/stt/health/ready"


def wait_ready(url: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        try:
            # RunPod 프록시(Cloudflare)는 Python 기본 User-Agent 를 403 으로 막는다
            req = urllib.request.Request(url, headers={"User-Agent": "er-stt-client/1.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                last = json.load(r)
        except urllib.error.HTTPError as e:           # 503 = LOADING / NOT_READY
            try:
                last = json.loads(e.read() or b"{}")
            except ValueError:                        # 프록시 오류 페이지 등 JSON 이 아닌 응답
                last = {"status": f"HTTP {e.code} (STT 서버가 아닌 응답 — 포트 노출/주소 확인)"}
        except OSError as e:
            last = {"status": f"unreachable ({e})"}
        status = last.get("status")
        print(f"[ready] {status}", file=sys.stderr)
        if status == "READY":
            return last
        if status == "NOT_READY":
            raise SystemExit(f"서버가 NOT_READY 입니다: {last.get('error')}")
        time.sleep(2)
    raise SystemExit(f"{timeout:.0f}s 안에 READY 가 되지 않았습니다: {last}")


async def stream(url: str, pcm: bytes, realtime: bool) -> dict:
    import websockets

    stats = {"partials": 0, "finals": [], "errors": [], "done": False}
    started = time.monotonic()
    async with websockets.connect(url, max_size=None) as ws:

        async def send_audio() -> None:
            for i in range(0, len(pcm), FRAME_BYTES):
                await ws.send(pcm[i:i + FRAME_BYTES])          # binary frame
                if realtime:
                    await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "close"}))       # "close" 문자열도 된다

        sender = asyncio.create_task(send_audio())
        async for raw in ws:
            msg = json.loads(raw)
            t = time.monotonic() - started
            if msg["type"] == "transcript":
                if msg["is_final"]:
                    stats["finals"].append(msg)
                    span = (f"[{msg['start_time']:.1f}→{msg['end_time']:.1f}s]"
                            if "start_time" in msg else "")
                    print(f"{t:6.1f}s [final]   {span} {msg['text']}")
                else:
                    stats["partials"] += 1
                    print(f"{t:6.1f}s [partial] {msg['text']}")
            elif msg["type"] == "error":
                stats["errors"].append(msg["message"])
                print(f"{t:6.1f}s [error]   {msg['message']}")
            elif msg["type"] == "done":
                stats["done"] = True
                print(f"{t:6.1f}s [done]")
                break
        await sender
    stats["elapsed"] = time.monotonic() - started
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav", help="PCM16 WAV 파일")
    parser.add_argument("--url", default="ws://localhost:8000/ws/v1/stt/stream")
    parser.add_argument("--fast", action="store_true", help="실시간 대기 없이 전송")
    parser.add_argument("--no-wait", action="store_true", help="READY 확인을 건너뛴다")
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    args = parser.parse_args()

    pcm, duration = load_pcm(args.wav)
    if not args.no_wait:
        info = wait_ready(ready_url(args.url), args.ready_timeout)
        print(f"[ready] engine={info.get('engine')} device={info.get('device')} "
              f"gpu_available={info.get('gpu_available')} gpu_memory_mb={info.get('gpu_memory_mb')}",
              file=sys.stderr)

    print(f"[send] {args.wav}: {duration:.1f}s, {len(pcm) // FRAME_BYTES + 1} frames → {args.url}",
          file=sys.stderr)
    stats = asyncio.run(stream(args.url, pcm, realtime=not args.fast))

    ok = stats["done"] and bool(stats["finals"])
    print(
        f"\n[summary] audio={duration:.1f}s elapsed={stats['elapsed']:.1f}s "
        f"partials={stats['partials']} finals={len(stats['finals'])} "
        f"errors={len(stats['errors'])} done={stats['done']} → {'PASS' if ok else 'FAIL'}",
        file=sys.stderr,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
