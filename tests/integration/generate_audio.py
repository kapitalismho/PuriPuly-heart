import math
import struct
import wave


def generate_sine_wave(filename, duration=3.0, freq=440.0, sample_rate=16000):
    """Generate a sine wave WAV file."""
    n_samples = int(sample_rate * duration)

    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for i in range(n_samples):
            value = int(32767.0 * math.sin(2 * math.pi * freq * i / sample_rate))
            data = struct.pack("<h", value)
            wav_file.writeframes(data)


if __name__ == "__main__":
    generate_sine_wave(".test_audio/test_sine.wav")
    print("Generated .test_audio/test_sine.wav")
