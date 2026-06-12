"""Speech recognition using Whisper."""

from __future__ import annotations

import re
from typing import Any, Optional, Union, BinaryIO

import numpy as np

from ..config import NUM_CORES, WHISPER_LANGUAGE


# Lazy-loaded Whisper model
_whisper_model: Optional[Any] = None


def get_whisper_model() -> Any:
    """Get or initialize the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            cpu_threads=NUM_CORES // 2,
            num_workers=NUM_CORES // 2,
        )
    return _whisper_model


def wav_to_text(audio_input: Union[str, BinaryIO, np.ndarray]) -> str:
    try:
        whisper_model = get_whisper_model()

        segments, info = whisper_model.transcribe(
            audio_input,
            language=WHISPER_LANGUAGE,
            task="transcribe",
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.8,       
            log_prob_threshold=-1.5,     
            compression_ratio_threshold=2.4, 
            beam_size=5,
            best_of=5,
        )

        result = "".join(segment.text for segment in segments).strip()
        print(f"[Whisper] Detected language: {info.language} | Transcript: {result!r}")
        return result

    except Exception as e:
        print(f"[Whisper] ERROR {type(e).__name__}: {e}")
        return ""


def extract_prompt(transcribed_text: str, wake_word: str = "") -> Optional[str]:
    cleaned = transcribed_text.strip()
    return cleaned if cleaned else None