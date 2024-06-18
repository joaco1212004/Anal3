import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score
import scipy.signal as sig
import os

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
    for digit in range(10):
        digit_audio = [audio_files[i] for i in range(len(audio_files)) if labels[i] == digit]
        averages[digit] = np.mean(digit_audio, axis=0)
    return averages


def classify(audio, averages):
    min_distance = float('inf')
    predicted_digit = None
    for digit, avg_rep in averages.items():
        distance = np.linalg.norm(audio - avg_rep)
        if distance < min_distance:
            min_distance = distance
            predicted_digit = digit
    return predicted_digit


def correlation_lags(in1_len, in2_len, mode='full'):
    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid - lag_bound):(mid + lag_bound)]
        else:
            lags = lags[(mid - lag_bound):(mid + lag_bound) + 1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags


def align_samples_global(audio_files):
    ref_audio = audio_files[0]
    aligned_audio_files = []
    max_length = max([len(audio) for audio in audio_files])
    for audio in audio_files:

        corr_s = sig.correlate(audio, ref_audio)
        lags = correlation_lags(len(audio), len(ref_audio))
        lag = lags[np.argmax(corr_s)]
        if lag < 0:
            aligned_audio = audio[:lag]
        else:
            aligned_audio = audio[lag:]

        if len(aligned_audio) < max_length:
            aligned_audio = np.pad(aligned_audio, (0, max_length - len(aligned_audio)))
        aligned_audio_files.append(aligned_audio)
    return np.array(aligned_audio_files)

def align_samples_digitwise(audio_files, labels):

    ref_audios = {}
    aligned_audio_files = []
    max_length = max([len(audio) for audio in audio_files])
    for audio, label in zip(audio_files, labels):
        if label not in ref_audios:
            ref_audios[label] = audio
        ref_audio = ref_audios[label]

        corr_s = sig.correlate(audio, ref_audio)
        lags = correlation_lags(len(audio), len(ref_audio))
        lag = lags[np.argmax(corr_s)]
        if lag < 0:
            aligned_audio = audio[:lag]
        else:
            aligned_audio = audio[lag:]

        if len(aligned_audio) < max_length:
            aligned_audio = np.pad(aligned_audio, (0, max_length - len(aligned_audio)))
        aligned_audio_files.append(aligned_audio)
    return np.array(aligned_audio_files)


train_audio_files, train_labels = load_audio_files("../data/train")
test_audio_files, test_labels = load_audio_files("../data/test")

# train_audio_files = align_samples_global(train_audio_files)
# test_audio_files = align_samples_global(test_audio_files)

# train_audio_files = align_samples_digitwise(train_audio_files, train_labels)
# test_audio_files = align_samples_digitwise(test_audio_files, test_labels)

averages = calculate_average_representations(train_audio_files, train_labels)

predictions = [(label, classify(audio, averages)) for audio, label in zip(test_audio_files, test_labels)]

accuracy = accuracy_score(test_labels, [p[1] for p in predictions])
print(f"Accuracy: {accuracy * 100:.2f}%")
# for p in predictions:
#     print(f"Real: {p[0]}, Predecido: {p[1]}")
