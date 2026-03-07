"""
app/ingestion/ocr_processor.py
───────────────────────────────
Extract text from images and video frames using Tesseract OCR (local).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from app.telemetry.logging_config import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)


def ocr_image(image_path: str | Path) -> str:
    """Extract text from a single image file using Tesseract."""
    cfg = get_config().ingestion
    img = Image.open(str(image_path)).convert("RGB")
    text: str = str(pytesseract.image_to_string(img, lang=cfg.ocr_lang))
    logger.info("ocr_complete", path=str(image_path), chars=len(text))
    return text.strip()


def ocr_from_pil(image: Image.Image) -> str:
    """Run OCR on an already-loaded PIL image."""
    cfg = get_config().ingestion
    return str(pytesseract.image_to_string(image.convert("RGB"), lang=cfg.ocr_lang)).strip()


def extract_frames_from_video(
    video_path: str | Path,
    frame_interval_sec: float = 5.0,
    max_frames: int = 50,
) -> list[np.ndarray]:
    """
    Sample frames from a video at the given interval.
    Returns a list of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval_frames = max(1, int(fps * frame_interval_sec))
    frames: list[np.ndarray] = []
    frame_idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval_frames == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    logger.info("frames_extracted", total=len(frames), video=str(video_path))
    return frames


def ocr_video(
    video_path: str | Path,
    frame_interval_sec: float = 5.0,
    max_frames: int = 50,
) -> str:
    """
    Extract text from video frames via OCR.
    Returns concatenated text from all sampled frames (deduped).
    """
    frames = extract_frames_from_video(video_path, frame_interval_sec, max_frames)
    seen: set[str] = set()
    texts: list[str] = []

    for frame in frames:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = ocr_from_pil(pil_img)
        if text and text not in seen:
            seen.add(text)
            texts.append(text)

    combined = "\n\n".join(texts)
    logger.info("ocr_video_complete", unique_segments=len(texts))
    return combined