import librosa
import numpy as np

def speech_quality(audio_file):
    y, sr = librosa.load(audio_file)
    energy = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    energy_mean = np.mean(energy)
    zcr_mean = np.mean(zcr)
    return energy_mean, zcr_mean

# Example usage
audio_file = 'path_to_audio.wav'
energy, zcr = speech_quality(audio_file)
print(f"Energy: {energy}, Zero Crossing Rate: {zcr}")
