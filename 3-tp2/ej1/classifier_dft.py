import numpy as np
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
            label = int(filename[0])  # Asume que el primer carácter del nombre del archivo es el dígito
            audio_files.append(audio)
            labels.append(label)
    return np.array(audio_files), np.array(labels)

def calculate_average_representations_dft(audio_files, labels):
    averages = {}
    for digit in range(10):
        digit_audio = [audio_files[i] for i in range(len(audio_files)) if labels[i] == digit]
        # Convertir cada señal al dominio de la frecuencia usando DFT
        digit_audio_dft = [np.abs(dft(audio)) for audio in digit_audio]
        # Calcular la representación promedio en el dominio de la frecuencia
        averages[digit] = np.mean(digit_audio_dft, axis=0)
    return averages

def classify_dft(audio, averages):
    # Convertir la señal de entrada al dominio de la frecuencia usando DFT
    audio_dft = np.abs(dft(audio))
    min_distance = float('inf')
    predicted_digit = None
    for digit, avg_rep in averages.items():
        distance = np.linalg.norm(audio_dft - avg_rep)
        if distance < min_distance:
            min_distance = distance
            predicted_digit = digit
    return predicted_digit

# Cargar los archivos de audio
audio_files, labels = load_audio_files("../data/train")
test_audio_files, test_labels = load_audio_files("../data/test")

# Calcular las representaciones promedio con el conjunto de entrenamiento
averages = calculate_average_representations_dft(audio_files, labels)

# Clasificar los audios del conjunto de prueba
predictions = [(label, classify_dft(audio, averages)) for audio, label in zip(test_audio_files, test_labels)]

# Evaluar el rendimiento
accuracy = accuracy_score(test_labels, [p[1] for p in predictions])
print(f"Accuracy: {accuracy * 100:.2f}%")
# for p in predictions:
#     print(f"True digit: {p[0]}, Predicted digit: {p[1]}")
