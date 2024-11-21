from pyAudioAnalysis import audioFeatureExtraction

def pacing_and_pause(audio_file):
    [fs, signal] = audioFeatureExtraction.read_audio_file(audio_file)
    features, _ = audioFeatureExtraction.stFeatureExtraction(signal, fs, 0.050, 0.025)
    pause_duration = np.sum(features[2] < 0.02)  # Detecting pauses based on energy threshold
    return pause_duration

# Example usage
audio_file = 'path_to_audio.wav'
pause_duration = pacing_and_pause(audio_file)
print(f"Pause Duration (in frames): {pause_duration}")
