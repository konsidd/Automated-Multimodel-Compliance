"""
app/ingestion/video_transcriber.py
───────────────────────────────────
Download a YouTube video with yt-dlp and transcribe it using
OpenAI Whisper (runs entirely locally).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from app.telemetry.logging_config import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)


# ── Download ──────────────────────────────────────────────────────────────────

def download_video(url: str, output_dir: Optional[str] = None) -> Path:
    """
    Download a YouTube (or any yt-dlp supported) video to output_dir.
    Returns the path to the downloaded audio/video file.
    """
    cfg = get_config().ingestion
    out_dir = Path(output_dir or cfg.temp_video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download best audio-only format for faster transcription
    output_template = str(out_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestaudio/best",
        "--output", output_template,
        "--no-playlist",
        url,
    ]
    logger.info("downloading_video", url=url)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    # Find the downloaded file
    files = sorted(out_dir.glob("*.webm")) + sorted(out_dir.glob("*.m4a")) + sorted(out_dir.glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No downloaded file found in {out_dir}")

    downloaded = files[-1]
    logger.info("video_downloaded", path=str(downloaded))
    return downloaded


# ── Transcribe ────────────────────────────────────────────────────────────────

def transcribe_audio(audio_path: Path, save_transcript: bool = True) -> str:
    """
    Transcribe an audio/video file using local Whisper model.
    Returns the full transcript text.
    """
    import whisper  # type: ignore

    cfg = get_config().ingestion
    transcript_dir = Path(cfg.transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading_whisper_model", model=cfg.whisper_model)
    model = whisper.load_model(cfg.whisper_model)

    logger.info("transcribing", file=str(audio_path))
    result = model.transcribe(str(audio_path), fp16=False, verbose=False)
    text: str = str(result["text"]).strip()

    if save_transcript:
        transcript_path = transcript_dir / (audio_path.stem + ".txt")
        transcript_path.write_text(text, encoding="utf-8")
        logger.info("transcript_saved", path=str(transcript_path))

    return text


# ── Combined pipeline ─────────────────────────────────────────────────────────

def transcribe_youtube_video(url: str) -> str:
    """Download a YouTube video and return its transcript."""
    audio_path = download_video(url)
    return transcribe_audio(audio_path)


def transcribe_local_file(file_path: str) -> str:
    """Transcribe a local audio/video file."""
    return transcribe_audio(Path(file_path))