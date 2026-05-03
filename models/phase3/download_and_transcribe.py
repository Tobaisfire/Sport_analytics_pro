#!/usr/bin/env python3
"""
Phase 3 — Download a YouTube clip → WAV → ASR transcript (for the dashboard).

Default clip is a longer **final highlights**-style upload (~15 min), not ICC’s short “Epic Montage”.
To **watch** the official ICC packages, use the Streamlit links (ICC.tv match / extended highlights).

Requires:
  pip install yt-dlp openai-whisper
  ffmpeg on PATH

On Windows, OpenAI Whisper (tiny/base) is more reliable than faster-whisper here.

Outputs:
  media/p3_<youtube_id>.* and media/p3_<youtube_id>_16k.wav (unique per URL)
  artifacts/transcript_1415755_asr.txt
  artifacts/asr_run_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import re

# Windows: avoid some OpenMP crashes when multiple runtimes load
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import shutil
import subprocess
import sys
from datetime import datetime, timezone

P3_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(P3_DIR, "media")
ART_DIR = os.path.join(P3_DIR, "artifacts")

# Match-style **highlights** commentary (~5 min, ICC-style title). Not the “Epic Montage” (3yiWqnKl7lQ).
# Longer upload (studio + ads first): SbP1CN4rTlo — worse for ASR unless you trim audio.
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=gpLWO9Erl40"
MATCH_ID = 1415755


def youtube_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not m:
        raise ValueError(f"Could not parse YouTube id from URL: {url!r}")
    return m.group(1)


def _run(cmd: list[str], **kw) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        sys.stderr.write("ERROR: ffmpeg not on PATH.\n")
        sys.exit(1)


def download_audio(url: str, out_base: str) -> str:
    os.makedirs(MEDIA_DIR, exist_ok=True)
    pattern = os.path.join(MEDIA_DIR, out_base + ".%(ext)s")
    _run([
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "--no-playlist",
        "--no-continue",
        "--force-overwrites",
        "-o", pattern,
        url,
    ])
    candidates = [
        os.path.join(MEDIA_DIR, f)
        for f in os.listdir(MEDIA_DIR)
        if f.startswith(out_base + ".")
    ]
    if not candidates:
        raise FileNotFoundError(f"No download for base {out_base!r}")
    return max(candidates, key=os.path.getmtime)


def to_wav_16k_mono(src_path: str, wav_path: str) -> None:
    _run([
        "ffmpeg", "-y", "-i", src_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        wav_path,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_openai_whisper(wav_path: str, model_size: str) -> str:
    import whisper

    m = whisper.load_model(model_size)
    r = m.transcribe(wav_path, language="en", fp16=False)
    return (r.get("text") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_VIDEO_URL)
    ap.add_argument(
        "--model",
        default="tiny",
        help="Whisper model: tiny, base, small, medium, large (tiny is fastest)",
    )
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    ensure_ffmpeg()
    os.makedirs(ART_DIR, exist_ok=True)
    vid = youtube_video_id(args.url)
    out_base = f"p3_{vid}"
    wav_path = os.path.join(MEDIA_DIR, f"p3_{vid}_16k.wav")

    if args.skip_download and os.path.isfile(wav_path):
        print("Using existing", wav_path)
    else:
        raw = download_audio(args.url, out_base)
        print("Downloaded:", raw)
        to_wav_16k_mono(raw, wav_path)
        print("Wrote", wav_path)

    print("Transcribing with openai-whisper (%s)..." % args.model)
    text = transcribe_openai_whisper(wav_path, args.model)

    out_txt = os.path.join(ART_DIR, f"transcript_{MATCH_ID}_asr.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print("Wrote", out_txt)

    meta = {
        "match_id": MATCH_ID,
        "youtube_id": vid,
        "source_url": args.url,
        "source_note": (
            "YouTube **match highlights** reel (English commentary; not ICC’s short “Epic Montage” montage 3yiWqnKl7lQ). "
            "For the official ICC video player, use the Streamlit ICC buttons."
        ),
        "whisper_engine": "openai-whisper",
        "whisper_model": args.model,
        "wav_path": wav_path,
        "transcript_file": out_txt,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(ART_DIR, "asr_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Done. Restart Streamlit — Phase 3 uses transcript_%s_asr.txt when present." % MATCH_ID)


if __name__ == "__main__":
    main()
