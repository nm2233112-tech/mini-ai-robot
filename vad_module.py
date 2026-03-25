import torch
import numpy as np
import sounddevice as sd
import os

# ==========================================
# 1. Initialization
# ==========================================

print("Loading Silero VAD model...")
model, utils = torch.hub.load(
    'snakers4/silero-vad',
    'silero_vad',
    force_reload=False
)

(get_speech_timestamps, _, _, _, _) = utils

SAMPLE_RATE = 16000
CHUNK_SIZE = 4000  # ~0.25 sec


# ==========================================
# 2. Core Functions
# ==========================================

def is_speech(audio_chunk):
    """
    Detect if audio chunk contains speech
    """
    try:
        audio_tensor = torch.from_numpy(audio_chunk).float()
        segments = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.4
        )
        return len(segments) > 0

    except Exception as e:
        print(f"[ERROR] Speech detection failed: {e}")
        return False


def record_until_silence(max_silence_chunks=10):
    """
    Record audio until silence is detected
    """
    print("Recording... Speak now")

    recorded_audio = []
    silent_chunks = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        ) as stream:

            while True:
                chunk, _ = stream.read(CHUNK_SIZE)
                chunk = chunk.flatten()
                recorded_audio.append(chunk)

                if is_speech(chunk):
                    silent_chunks = 0
                else:
                    silent_chunks += 1

                if silent_chunks >= max_silence_chunks:
                    break

        return np.concatenate(recorded_audio)

    except Exception as e:
        print(f"[ERROR] Recording failed: {e}")
        return None


# ==========================================
# 3. Test Function
# ==========================================

def test_vad():
    """
    Test the VAD module by recording audio
    and detecting if speech exists
    """
    audio = record_until_silence()

    if audio is None:
        print("Test failed: No audio recorded")
        return

    if is_speech(audio):
        print("Test result: Speech detected")
    else:
        print("Test result: No speech detected")


# ==========================================
# 4. Run Test
# ==========================================

if __name__ == "__main__":
    test_vad()