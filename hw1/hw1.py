import librosa
import os
import random

noise_file_names = []
for path, subdirs, files in os.walk("bg_noise"):
    for name in files:
        noise_file_names.append(os.path.join(path, name))


def add_noise(audio_file, k=0.15):
    y, sr = librosa.load(audio_file)
    noise_file = random.choice(noise_file_names)
    n_y, _ = librosa.load(noise_file)
    for i in range(y.shape[0]):
        y[i] = (1 - k) * y[i] + k * n_y[i % n_y.shape[0]]

    return y, sr


files = ["amy.wav", "eric.flac"]
noised_files = ["amy_noised.wav", "eric_noised.flac"]
for i in range(len(files)):
    y, sr = add_noise(files[i])
    librosa.output.write_wav(noised_files[i], y, sr)
