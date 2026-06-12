"""
Simple audio debug app — record via st.audio_input, play back immediately.
Run with: streamlit run debug_audio.py
"""

import io
import wave
import struct
import numpy as np
import soundfile as sf
import streamlit as st

st.set_page_config(page_title="Audio Debug", page_icon="🎙️", layout="centered")
st.title("🎙️ Audio Record & Playback Debug")
st.markdown("---")


def read_wav_with_wave_module(raw_bytes: bytes):
    """
    Read WAV bytes using Python's built-in wave module.
    More permissive than soundfile — reads raw PCM directly.

    Returns (pcm_float32: np.ndarray, sample_rate: int, n_channels: int)
    or (None, None, None) on failure.
    """
    try:
        buf = io.BytesIO(raw_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels   = wf.getnchannels()
            sampwidth    = wf.getsampwidth()   # bytes per sample: 1=8bit, 2=16bit, 4=32bit
            sample_rate  = wf.getframerate()
            n_frames     = wf.getnframes()
            raw_pcm      = wf.readframes(n_frames)

        print(f"[wave] channels={n_channels}, sampwidth={sampwidth}, "
              f"rate={sample_rate}, frames={n_frames}, pcm_bytes={len(raw_pcm)}")

        # Convert raw PCM bytes to float32 numpy array
        if sampwidth == 2:
            # 16-bit signed PCM — the standard format
            samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
            samples /= 32768.0   # Normalize to [-1.0, 1.0]
        elif sampwidth == 4:
            # 32-bit signed PCM
            samples = np.frombuffer(raw_pcm, dtype=np.int32).astype(np.float32)
            samples /= 2147483648.0
        elif sampwidth == 1:
            # 8-bit unsigned PCM
            samples = np.frombuffer(raw_pcm, dtype=np.uint8).astype(np.float32)
            samples = (samples - 128.0) / 128.0
        else:
            return None, None, None

        # Reshape for multi-channel audio
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)

        return samples, sample_rate, n_channels

    except Exception as e:
        print(f"[wave] ERROR: {e}")
        return None, None, None


def parse_wav_chunks(raw_bytes: bytes) -> dict:
    """
    Manually parse WAV RIFF chunks to inspect the raw structure.
    Returns dict with chunk info — helps detect empty/malformed data chunks.
    """
    result = {}
    try:
        buf = io.BytesIO(raw_bytes)

        # RIFF header: 'RIFF' + file_size (4 bytes LE) + 'WAVE'
        riff_id   = buf.read(4)
        file_size = struct.unpack("<I", buf.read(4))[0]
        wave_id   = buf.read(4)

        result["riff_id"]      = riff_id
        result["file_size"]    = file_size
        result["wave_id"]      = wave_id
        result["chunks"]       = []

        # Read all sub-chunks
        while True:
            chunk_id_bytes = buf.read(4)
            if len(chunk_id_bytes) < 4:
                break

            chunk_size_bytes = buf.read(4)
            if len(chunk_size_bytes) < 4:
                break

            chunk_id   = chunk_id_bytes.decode("ascii", errors="replace")
            chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
            chunk_pos  = buf.tell()

            # Peek at first 16 bytes of chunk data
            preview = buf.read(min(16, chunk_size))
            result["chunks"].append({
                "id":       chunk_id,
                "size":     chunk_size,
                "position": chunk_pos,
                "preview":  preview.hex(),
            })

            # Skip remaining bytes in this chunk (pad to even boundary)
            remaining = chunk_size - len(preview)
            buf.seek(remaining + (chunk_size % 2), 1)

    except Exception as e:
        result["parse_error"] = str(e)

    return result


# ── Step 1: Record ──
st.markdown("### Step 1: Record")
st.caption("Click the microphone button, speak, then click stop.")
audio_value = st.audio_input("Click to record", key="debug_recorder")

if audio_value is not None:
    raw_bytes = audio_value.getvalue()

    # ── Step 2: Raw bytes info ──
    st.markdown("---")
    st.markdown("### Step 2: Raw bytes info")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total size (bytes)", f"{len(raw_bytes):,}")
    col2.metric("Magic header (hex)", raw_bytes[:4].hex())

    if raw_bytes[:4] == b"RIFF":
        fmt = "WAV"
    elif raw_bytes[:4] == b"\x1a\x45\xdf\xa3":
        fmt = "WebM"
    elif raw_bytes[:4] == b"OggS":
        fmt = "OGG"
    else:
        fmt = "Unknown"

    col3.metric("Detected format", fmt)

    # ── Step 3: WAV chunk structure inspection ──
    st.markdown("---")
    st.markdown("### Step 3: WAV internal structure")
    st.caption("Inspects each RIFF chunk to find if the 'data' chunk has actual content.")

    if fmt == "WAV":
        chunks_info = parse_wav_chunks(raw_bytes)

        if "parse_error" in chunks_info:
            st.error(f"RIFF parse error: {chunks_info['parse_error']}")
        else:
            st.write(f"**RIFF header:** `{chunks_info['riff_id']}` | "
                     f"file_size=`{chunks_info['file_size']:,}` | "
                     f"WAVE=`{chunks_info['wave_id']}`")

            for chunk in chunks_info["chunks"]:
                is_data = chunk["id"].strip() == "data"
                color   = "🔴" if (is_data and chunk["size"] == 0) else (
                          "✅" if is_data else "📦")

                st.write(
                    f"{color} Chunk **`{chunk['id']}`** | "
                    f"size=`{chunk['size']:,}` bytes | "
                    f"preview: `{chunk['preview']}`"
                )

                # Critical check: data chunk size
                if is_data:
                    if chunk["size"] == 0:
                        st.error(
                            "❌ **data chunk size is 0** — WAV container is empty. "
                            "The browser sent a valid WAV header but NO audio samples. "
                            "This is a known issue with some Windows Chrome/Edge versions "
                            "where the microphone stream is captured but not written to the WAV body."
                        )
                    else:
                        st.success(
                            f"✅ data chunk has `{chunk['size']:,}` bytes "
                            f"≈ `{chunk['size'] / 2 / 16000:.2f}s` of 16-bit 16kHz audio"
                        )
    else:
        st.info(f"Format is {fmt} — skipping WAV chunk inspection.")

    # ── Step 4: Raw playback ──
    st.markdown("---")
    st.markdown("### Step 4: Raw playback (browser native — no conversion)")
    st.caption("If you can hear audio here, the recording itself is fine.")
    st.audio(raw_bytes)

    # ── Step 5: wave module decode (more permissive than soundfile) ──
    st.markdown("---")
    st.markdown("### Step 5: Decode with Python wave module")
    st.caption("`wave` reads PCM bytes directly — bypasses soundfile's stricter parsing.")

    wave_samples, wave_sr, wave_ch = read_wav_with_wave_module(raw_bytes)

    if wave_samples is not None:
        # Convert to mono if needed
        if wave_samples.ndim == 2:
            mono_wave = wave_samples.mean(axis=1)
            st.info(f"wave module: {wave_ch} channels → averaged to mono")
        else:
            mono_wave = wave_samples

        rms_wave  = float(np.sqrt(np.mean(mono_wave ** 2)))
        peak_wave = float(np.max(np.abs(mono_wave)))

        st.success(
            f"wave decoded OK — rate={wave_sr}Hz | "
            f"samples={len(mono_wave):,} | "
            f"duration={len(mono_wave)/wave_sr:.2f}s"
        )
        st.write(f"**wave RMS:** `{rms_wave:.6f}` | **Peak:** `{peak_wave:.6f}`")

        # Show first 20 raw sample values to spot all-zeros
        st.write("**First 20 sample values** (should NOT all be 0.0 if mic is working):")
        st.code(str(mono_wave[:20].tolist()))

        if rms_wave < 0.001:
            st.error(
                "❌ wave module also sees near-zero RMS. "
                "The PCM data chunk contains silence/zeros. "
                "**Root cause: browser microphone permission denied or wrong device selected.** "
                "Check: browser address bar → lock icon → Microphone → Allow."
            )
        elif rms_wave < 0.01:
            st.warning("⚠️ Very low amplitude — mic volume may be too low.")
        else:
            st.success("✅ wave module sees real audio signal.")

        # Waveform
        step = max(1, len(mono_wave) // 2000)
        st.line_chart(mono_wave[::step])

    else:
        st.error("wave module failed to decode. See console for details.")

    # ── Step 6: soundfile decode (compare with wave) ──
    st.markdown("---")
    st.markdown("### Step 6: Decode with soundfile (compare with Step 5)")

    try:
        sf_buf              = io.BytesIO(raw_bytes)
        sf_data, sf_rate    = sf.read(sf_buf, dtype="float32")

        if sf_data.ndim == 2:
            mono_sf = sf_data.mean(axis=1)
            st.info("soundfile: stereo → averaged to mono")
        else:
            mono_sf = sf_data

        rms_sf  = float(np.sqrt(np.mean(mono_sf ** 2)))
        peak_sf = float(np.max(np.abs(mono_sf)))

        st.success(f"soundfile decoded OK — rate={sf_rate}Hz | samples={len(mono_sf):,}")
        st.write(f"**soundfile RMS:** `{rms_sf:.6f}` | **Peak:** `{peak_sf:.6f}`")

        # Agreement check between wave and soundfile
        if wave_samples is not None:
            rms_diff = abs(rms_wave - rms_sf)
            if rms_diff > 0.01:
                st.warning(
                    f"⚠️ RMS mismatch between wave ({rms_wave:.6f}) "
                    f"and soundfile ({rms_sf:.6f}) — decoding inconsistency."
                )
            else:
                st.success("✅ wave and soundfile agree on amplitude.")

    except Exception as e:
        st.error(f"soundfile failed: {type(e).__name__}: {e}")
        mono_sf = mono_wave if wave_samples is not None else None
        sf_rate = wave_sr   if wave_samples is not None else 16000

    # ── Step 7: Re-export and playback ──
    st.markdown("---")
    st.markdown("### Step 7: Re-export as 16kHz mono WAV and play back")
    st.caption("Uses wave-module decoded data to avoid soundfile normalization artifacts.")

    if wave_samples is not None and rms_wave > 0.0:
        # Use wave-module data (more reliable on Windows)
        export_mono = mono_wave

        # Resample to 16kHz if needed
        if wave_sr != 16000:
            duration   = len(export_mono) / wave_sr
            target_len = int(duration * 16000)
            export_mono = np.interp(
                np.linspace(0, len(export_mono) - 1, target_len),
                np.arange(len(export_mono)),
                export_mono,
            )
            wave_sr = 16000
            st.info("Resampled to 16000 Hz")

        out_buf = io.BytesIO()
        sf.write(out_buf, export_mono, wave_sr, format="WAV", subtype="PCM_16")
        wav_bytes = out_buf.getvalue()

        st.metric("Re-exported WAV size (bytes)", f"{len(wav_bytes):,}")
        st.audio(wav_bytes, format="audio/wav")
    else:
        st.warning(
            "Skipping re-export — source audio has zero amplitude. "
            "Fix the microphone issue first (Step 5/6)."
        )
        wav_bytes = None

    # ── Step 8: Whisper transcription tests ──
    st.markdown("---")
    st.markdown("### Step 8: Whisper transcription tests")

    if wav_bytes is None:
        st.warning("No valid audio to transcribe. Fix microphone first.")
    else:
        if st.button("🔍 Run Whisper transcription"):
            with st.spinner("Transcribing…"):
                try:
                    from faster_whisper import WhisperModel

                    model = WhisperModel("base", device="cpu", compute_type="int8")

                    def run_whisper(label: str, buf: io.BytesIO, **kwargs):
                        """Run one Whisper transcription test and display result."""
                        st.markdown(f"#### {label}")
                        segs, info = model.transcribe(buf, language="en", **kwargs)
                        result = "".join(s.text for s in segs).strip()
                        st.code(
                            f"Language : {info.language}\n"
                            f"Transcript: {result!r}\n"
                            f"Settings  : {kwargs}"
                        )
                        return result

                    # Test A: VAD ON (was causing empty transcripts)
                    run_whisper(
                        "Test A — vad_filter=True",
                        io.BytesIO(wav_bytes),
                        vad_filter=True,
                        condition_on_previous_text=False,
                        no_speech_threshold=0.8,
                    )

                    # Test B: VAD OFF (recommended for browser mic)
                    run_whisper(
                        "Test B — vad_filter=False",
                        io.BytesIO(wav_bytes),
                        vad_filter=False,
                        condition_on_previous_text=False,
                        no_speech_threshold=0.8,
                    )

                    # Test C: Completely raw — no filters at all
                    run_whisper(
                        "Test C — No restrictions (raw Whisper)",
                        io.BytesIO(wav_bytes),
                    )

                except Exception as e:
                    st.error(f"Whisper error: {type(e).__name__}: {e}")

    # ── Step 9: Microphone diagnostics ──
    st.markdown("---")
    st.markdown("### Step 9: Microphone diagnostics checklist")
    st.markdown("""
    If RMS is near zero, work through this checklist:

    | # | Check | How to fix |
    |---|-------|-----------|
    | 1 | **Browser mic permission** | Address bar → 🔒 → Microphone → **Allow** |
    | 2 | **Correct mic device selected** | Address bar → 🔒 → Microphone → choose correct device |
    | 3 | **Windows mic privacy** | Settings → Privacy → Microphone → Allow apps |
    | 4 | **Mic not muted in Windows** | Right-click speaker icon → Sound settings → Input → check volume |
    | 5 | **Wrong default device** | Sound settings → Input → set correct microphone as default |
    | 6 | **Browser version** | Update Chrome/Edge — older versions have WebRTC mic bugs |
    | 7 | **Try different browser** | Test in Firefox vs Chrome — different WebRTC implementations |
    """)

else:
    st.info("👆 Click the record button above, speak, then stop.")