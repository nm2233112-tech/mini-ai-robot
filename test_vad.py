import numpy as np
from vad_module import is_speech, record_until_silence


def test_is_speech_with_silence():
    # صوت صامت (مفروض False)
    silence = np.zeros(16000)
    result = is_speech(silence)
    assert result == False
    print("Test 1 Passed: Silence detected correctly")


def test_is_speech_with_random_noise():
    # ضوضاء عشوائية
    noise = np.random.randn(16000)
    result = is_speech(noise)
    print("Test 2 Done: Noise processed (may vary)")


def test_record_function():
    print("Speak something for test...")
    audio = record_until_silence()

    assert audio is not None
    print("Test 3 Passed: Recording works")


if __name__ == "__main__":
    test_is_speech_with_silence()
    test_is_speech_with_random_noise()
    test_record_function()