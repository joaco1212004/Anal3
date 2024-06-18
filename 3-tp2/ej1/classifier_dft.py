import os
import soundfile as sf

import numpy as np

from sklearn.metrics import accuracy_score


def dft(signal):
    N = len(signal)
    dft_result = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            dft_result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return dft_result


def load_audio_files(directory):
    audio_files = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            audio, sr = sf.read(filepath)
            label = int(filename[0])
            audio_files.append(audio)
            labels.append(label)
    return np.array(audio_files), np.array(labels)


def calculate_average_representations_dft(audio_files, labels):
    averages = {}
    for digit in range(10):
        digit_audio = [audio_files[i] for i in range(len(audio_files)) if labels[i] == digit]
        digit_audio_dft = [np.abs(dft(audio)) for audio in digit_audio]
        averages[digit] = np.mean(digit_audio_dft, axis=0)
    return averages


def classify_dft(audio, averages):
    audio_dft = np.abs(dft(audio))
    min_distance = float('inf')
    predicted_digit = None
    for digit, avg_rep in averages.items():
        distance = np.linalg.norm(audio_dft - avg_rep)
        if distance < min_distance:
            min_distance = distance
            predicted_digit = digit
    return predicted_digit


audio_files, labels = load_audio_files("../data/train")
test_audio_files, test_labels = load_audio_files("../data/test")

averages = calculate_average_representations_dft(audio_files, labels)

predictions = [(label, classify_dft(audio, averages)) for audio, label in zip(test_audio_files, test_labels)]

accuracy = accuracy_score(test_labels, [p[1] for p in predictions])
print(f"Accuracy: {accuracy * 100:.2f}%")
# for p in predictions:
#     print(f"Real: {p[0]}, Predecido: {p[1]}")