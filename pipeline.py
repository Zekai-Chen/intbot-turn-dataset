#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import webrtcvad
except Exception as e:  # pragma: no cover - handled at runtime
    webrtcvad = None

try:
    import requests
except Exception as e:  # pragma: no cover - handled at runtime
    requests = None


@dataclass
class SegmentConfig:
    method: str = "webrtcvad"
    vad_mode: int = 2
    min_speech_ms: int = 600
    min_silence_ms: int = 300
    merge_gap_ms: int = 800
    frame_ms: int = 30


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed: {}\nstdout:\n{}\nstderr:\n{}".format(
                " ".join(cmd), proc.stdout, proc.stderr
            )
        )


def ensure_ffmpeg() -> None:
    try:
        run_cmd(["ffmpeg", "-version"])
    except Exception as e:
        raise RuntimeError(
            "ffmpeg not available. Please install ffmpeg and ensure it is on PATH."
        ) from e


def extract_audio(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    run_cmd(cmd)


def read_wave(path: Path) -> Tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        if num_channels != 1:
            raise ValueError("Expected mono audio")
        if sample_width != 2:
            raise ValueError("Expected 16-bit PCM")
        if sample_rate != 16000:
            raise ValueError("Expected 16000 Hz")
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def frame_generator(frame_ms: int, audio: bytes, sample_rate: int):
    frame_size = int(sample_rate * (frame_ms / 1000.0) * 2)
    offset = 0
    timestamp_ms = 0
    duration_ms = frame_ms
    while offset + frame_size <= len(audio):
        yield timestamp_ms, audio[offset : offset + frame_size]
        offset += frame_size
        timestamp_ms += duration_ms


def vad_segments(
    audio: bytes, sample_rate: int, cfg: SegmentConfig
) -> List[Tuple[int, int]]:
    if webrtcvad is None:
        raise RuntimeError("webrtcvad not installed. Please pip install webrtcvad")

    vad = webrtcvad.Vad(cfg.vad_mode)
    in_speech = False
    last_speech_end = 0
    silence_ms = 0
    segments: List[Tuple[int, int]] = []

    for ts_ms, frame in frame_generator(cfg.frame_ms, audio, sample_rate):
        is_speech = vad.is_speech(frame, sample_rate)
        frame_end_ms = ts_ms + cfg.frame_ms
        if is_speech:
            if not in_speech:
                seg_start = ts_ms
                in_speech = True
            last_speech_end = frame_end_ms
            silence_ms = 0
        else:
            if in_speech:
                silence_ms += cfg.frame_ms
                if silence_ms >= cfg.min_silence_ms:
                    segments.append((seg_start, last_speech_end))
                    in_speech = False
                    silence_ms = 0

    if in_speech:
        segments.append((seg_start, last_speech_end))

    # filter short segments
    filtered = [s for s in segments if (s[1] - s[0]) >= cfg.min_speech_ms]

    # merge by gap
    merged: List[Tuple[int, int]] = []
    for start, end in filtered:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start - prev_end < cfg.merge_gap_ms:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def write_segments(path: Path, segments: List[Tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "segments": [
            {"start_ms": int(s), "end_ms": int(e), "duration_ms": int(e - s)}
            for s, e in segments
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def read_segments(path: Path) -> List[Tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = []
    for s in payload.get("segments", []):
        segments.append((int(s["start_ms"]), int(s["end_ms"])))
    return segments


def cut_clip(audio_path: Path, start_ms: int, end_ms: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_sec = start_ms / 1000.0
    duration_sec = (end_ms - start_ms) / 1000.0
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        "{:.3f}".format(start_sec),
        "-t",
        "{:.3f}".format(duration_sec),
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    run_cmd(cmd)


def _load_openai_api_key() -> Tuple[Optional[str], Optional[str]]:
    key_file = os.getenv("OPENAI_API_KEY_FILE")
    if key_file:
        try:
            raw = Path(key_file).read_text(encoding="utf-8")
        except Exception as e:
            return None, f"Failed to read OPENAI_API_KEY_FILE: {e}"
        # Strip all whitespace to avoid header errors from newlines/spaces
        key = "".join(raw.split())
        if any(ord(ch) > 127 for ch in key):
            return None, "OPENAI_API_KEY_FILE contains non-ASCII characters"
        if not key:
            return None, "OPENAI_API_KEY_FILE is empty"
        return key, None

    raw_key = os.getenv("OPENAI_API_KEY", "")
    key = "".join(raw_key.split())
    if any(ord(ch) > 127 for ch in key):
        return None, "OPENAI_API_KEY contains non-ASCII characters"
    if not key:
        return None, "OPENAI_API_KEY not set"
    return key, None


def transcribe_openai(audio_path: Path, model: str) -> Tuple[str, Optional[float], Optional[str]]:
    api_key, key_error = _load_openai_api_key()
    if key_error:
        return "", None, key_error
    if requests is None:
        return "", None, "requests not installed"

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (audio_path.name, open(audio_path, "rb"), "audio/wav")}
    data = {"model": model}
    try:
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    finally:
        files["file"][1].close()

    if resp.status_code != 200:
        return "", None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    payload = resp.json()
    text = payload.get("text", "")
    return text, None, None


def build_dataset(
    video_id: str,
    source_video_path: Path,
    audio_path: Path,
    segments: List[Tuple[int, int]],
    cfg: SegmentConfig,
    clips_dir: Path,
    asr_model: str,
    do_asr: bool,
) -> List[dict]:
    rows = []
    total = len(segments)
    for idx, (start_ms, end_ms) in enumerate(segments):
        print(f"[{video_id}] turn {idx+1}/{total} {start_ms}-{end_ms}ms")
        clip_name = f"{video_id}_{idx:06d}.wav"
        clip_path = clips_dir / clip_name
        if not clip_path.exists():
            cut_clip(audio_path, start_ms, end_ms, clip_path)
        if do_asr:
            transcript, confidence, error = transcribe_openai(clip_path, asr_model)
        else:
            transcript, confidence, error = "", None, "asr_skipped"
        row = {
            "example_id": f"{video_id}_{idx:06d}",
            "video_id": video_id,
            "source_video_path": str(source_video_path),
            "audio_path": str(clip_path),
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "duration_ms": int(end_ms - start_ms),
            "speaker": "unknown",
            "speaker_source": "unknown",
            "transcript": transcript,
            "asr_model": asr_model,
            "asr_confidence": confidence,
            "segmentation": {
                "method": cfg.method,
                "vad_mode": cfg.vad_mode,
                "min_speech_ms": cfg.min_speech_ms,
                "min_silence_ms": cfg.min_silence_ms,
                "merge_gap_ms": cfg.merge_gap_ms,
            },
            "pipeline_version": "v0.1",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            row["errors"] = [error]
            row["transcript"] = ""
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Turn-level dialogue dataset pipeline")
    parser.add_argument(
        "--videos",
        nargs="*",
        default=[
            "/Users/chenzekai/Desktop/Intbot/clip1_short.mp4",
            "/Users/chenzekai/Desktop/Intbot/clip2_long.mp4",
        ],
    )
    parser.add_argument("--out", default="/Users/chenzekai/Desktop/Intbot/data")
    parser.add_argument("--asr-model", default="gpt-4o-mini-transcribe")
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help="Skip transcription and only segment/cut audio.",
    )
    parser.add_argument(
        "--asr-only",
        action="store_true",
        help="Only run ASR using existing segments/audio/clips.",
    )
    args = parser.parse_args()

    ensure_ffmpeg()

    cfg = SegmentConfig()
    out_root = Path(args.out)
    audio_dir = out_root / "audio"
    clips_dir = out_root / "clips"
    segments_dir = out_root / "segments"
    dataset_path = out_root / "dataset" / "turns.jsonl"

    all_rows: List[dict] = []

    if args.skip_asr and args.asr_only:
        raise ValueError("Cannot use --skip-asr and --asr-only together")

    for video in args.videos:
        video_path = Path(video)
        if not video_path.exists():
            raise FileNotFoundError(str(video_path))
        video_id = video_path.stem
        audio_path = audio_dir / f"{video_id}.wav"
        segments_path = segments_dir / f"{video_id}_segments.json"

        if args.asr_only:
            if not audio_path.exists():
                raise FileNotFoundError(str(audio_path))
            if not segments_path.exists():
                raise FileNotFoundError(str(segments_path))
            segments = read_segments(segments_path)
        else:
            extract_audio(video_path, audio_path)
            audio_bytes, sample_rate = read_wave(audio_path)
            segments = vad_segments(audio_bytes, sample_rate, cfg)
            write_segments(segments_path, segments)
        rows = build_dataset(
            video_id,
            video_path,
            audio_path,
            segments,
            cfg,
            clips_dir,
            args.asr_model,
            not args.skip_asr,
        )
        all_rows.extend(rows)

    # sort by video then time
    all_rows.sort(key=lambda r: (r["video_id"], r["start_ms"]))
    write_jsonl(dataset_path, all_rows)

    empty_tx = sum(1 for r in all_rows if not r.get("transcript"))
    print(f"Wrote {len(all_rows)} rows to {dataset_path}")
    print(f"Empty transcripts: {empty_tx}/{len(all_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
