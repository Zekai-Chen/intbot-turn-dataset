# Intbot Turn-Level Dialogue Dataset Pipeline

This repo builds a turn-level dialogue dataset from two robot-interaction videos.

## Outputs
- `data/dataset/turns.jsonl`
- `data/clips/{video_id}_{turn_idx:06d}.wav`
- `data/audio/{video_id}.wav`
- `data/segments/{video_id}_segments.json`
- Schema: `schema/turns_schema.json`

## Run
```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY_FILE="/Users/chenzekai/Desktop/Intbot/.openai_key"
python3 /Users/chenzekai/Desktop/Intbot/pipeline.py
```

## Notes
- VAD uses 30ms frames, `vad_mode=2`, `min_speech_ms=600`, `merge_gap_ms=800`.
- `OPENAI_API_KEY` or `OPENAI_API_KEY_FILE` is required for ASR.
