import numpy as np
from scipy.io import wavfile


def save_wav(wav: np.ndarray, output_path: str, sample_rate=24000):
    wav_norm = wav * 32767
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))


def save_wav_norm(wav: np.ndarray, output_path: str, sample_rate=24000):
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))
