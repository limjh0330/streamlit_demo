"""Zipformer / SenseVoice 모델을 내려받아 models/ 아래에 배치한다.

    python -m scripts.fetch_models              # 둘 다, int8 만 (권장)
    python -m scripts.fetch_models zipformer    # 하나만
    python -m scripts.fetch_models --keep all   # fp32 까지 (GPU 권장)

faster-whisper 는 최초 실행 시 자동으로 받으므로 여기서 다루지 않는다.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from stt.config import MODELS_DIR

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

#: name -> (릴리스 아카이브, int8 만 쓸 때 남길 파일 패턴)
MODELS = {
    # 한국어 전용 streaming transducer (KsponSpeech 학습)
    "zipformer": (
        "sherpa-onnx-streaming-zipformer-korean-2024-06-16",
        ("*.int8.onnx", "tokens.txt", "bpe.model"),
    ),
    # 다국어 non-autoregressive (zh/en/ja/ko/yue)
    "sensevoice": (
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        ("*.int8.onnx", "tokens.txt"),
    ),
}

#: int8 여부와 무관하게 항상 남기는 것
ALWAYS_KEEP = ("tokens.txt", "bpe.model", "LICENSE")


def _download(url: str, dest: Path) -> None:
    """진행률을 보여주며 내려받는다. 파이프로 넘길 때를 위해 5% 단위로만 찍는다."""
    tty = sys.stdout.isatty()
    last = [-1]

    def hook(block: int, size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block * size, total)
        pct = done * 100 // total
        if not tty:
            if pct // 5 == last[0]:
                return
            last[0] = pct // 5
            print(f"  {pct:3d}%  {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
        else:
            print(f"\r  {pct:3d}%  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    if tty:
        print()


def _wanted(path: Path, patterns: tuple[str, ...], keep_all: bool) -> bool:
    if path.name in ALWAYS_KEEP:
        return True
    if keep_all:
        return path.suffix == ".onnx" or path.parent.name == "test_wavs"
    if path.parent.name == "test_wavs":
        return True
    return any(path.match(p) for p in patterns)


def fetch(name: str, keep_all: bool, force: bool) -> None:
    archive, patterns = MODELS[name]
    target = MODELS_DIR / name

    if target.is_dir() and any(target.glob("*.onnx")) and not force:
        print(f"[{name}] 이미 있습니다: {target}  (--force 로 다시 받기)")
        return

    url = f"{BASE}/{archive}.tar.bz2"
    print(f"[{name}] {url}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tarball = tmp / "model.tar.bz2"
        _download(url, tarball)

        print("  압축 푸는 중…")
        with tarfile.open(tarball, "r:bz2") as tar:
            tar.extractall(tmp, filter="data")

        src = tmp / archive
        if not src.is_dir():                      # 아카이브 구조가 다르면 첫 디렉터리
            src = next(d for d in tmp.iterdir() if d.is_dir())

        if target.is_dir():
            shutil.rmtree(target)
        target.mkdir(parents=True)

        kept = 0
        for path in sorted(src.rglob("*")):
            if not path.is_file() or not _wanted(path, patterns, keep_all):
                continue
            out = target / path.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            kept += 1

    total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"  {kept} 개 파일, {total / 1e6:.0f} MB → {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", choices=list(MODELS),
                        help="받을 모델 (기본: 전부)")
    parser.add_argument("--keep", choices=["int8", "all"], default="int8",
                        help="int8 만 남길지, fp32 까지 남길지 (기본: int8)")
    parser.add_argument("--force", action="store_true", help="이미 있어도 다시 받기")
    args = parser.parse_args()

    for name in args.models or list(MODELS):
        try:
            fetch(name, keep_all=args.keep == "all", force=args.force)
        except Exception as e:
            print(f"[{name}] 실패: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
