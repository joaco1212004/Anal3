import numpy as np
import soundfile as sf
import librosa
import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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


def calculate_average_representations(audio_files, labels):
    averages = {}
    for digit in tqdm(range(10)):
        digit_audio = [audio_files[i] for i in range(len(audio_files)) if labels[i] == digit]
        # Convertir cada señal a MFCC
        digit_audio_mfcc = [librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13, n_fft=19000) for audio in digit_audio]
        # Calcular la representación promedio en el dominio MFCC
        averages[digit] = np.mean([np.mean(mfcc, axis=1) for mfcc in digit_audio_mfcc], axis=0)
    return averages


def classify(audio, averages):
    audio_mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13, n_fft=19000), axis=1)
    min_distance = float('inf')
    predicted_digit = None
    for digit, avg_rep in averages.items():
        distance = np.linalg.norm(audio_mfcc - avg_rep)
        if distance < min_distance:
            min_distance = distance
            predicted_digit = digit
    return predicted_digit


# Cargar los archivos de audio
audio_files, labels = load_audio_files("../data/train")
test_audio_files, test_labels = load_audio_files("../data/test")

# Calcular las representaciones promedio con el conjunto de entrenamiento
averages = calculate_average_representations(audio_files, labels)

# Clasificar los audios del conjunto de prueba
predictions = [(label, classify(audio, averages)) for audio, label in tqdm(zip(test_audio_files, test_labels))]

# Evaluar el rendimiento
accuracy = accuracy_score(test_labels, [p[1] for p in predictions])
print(f"Accuracy: {accuracy * 100:.2f}%")
# for p in predictions:
#     print(f"True digit: {p[0]}, Predicted digit: {p[1]}")
