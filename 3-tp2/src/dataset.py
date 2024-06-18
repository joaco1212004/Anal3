import glob
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import soundfile as sf

class AudioMNISTDataset(Dataset):
    def __init__(self, data_path, feature, test=False):
        self.data_path = data_path
        self.feature = feature
        self.test = test

    def __len__(self):
        if not self.test:
            return len(glob.glob(os.path.join(self.data_path, 'train', '*')))
        else:
            return len(glob.glob(os.path.join(self.data_path, 'test', '*')))

    def __getitem__(self, idx):
        # Get audio paths
        if not self.test:
            audio_paths = glob.glob(os.path.join(self.data_path, 'train', '*'))
        else:
            audio_paths = glob.glob(os.path.join(self.data_path, 'test', '*'))

        # Get audio data and labels
        audio, fs = sf.read(audio_paths[idx])
        label = os.path.basename(audio_paths[idx]).split('_')[0]
        # Extract features
        if self.feature == 'raw_waveform':
            feat = torch.tensor(audio, dtype=torch.float32)
        elif self.feature == 'audio_spectrum':
            feat = self.dft(audio, fs)
        elif self.feature == 'mfcc':
            feat = self.mfcc(audio, fs)
        
        feat = feat.type(torch.float)
        label = torch.tensor(int(label), dtype=torch.long)

        return feat, label

    @staticmethod
    def dft(audio: np.ndarray, fs: float) -> torch.Tensor:
        """
        Calculates the discrete Fourier transform of the audio data, normalizes the result and trims it, preserving only positive frequencies.
        Args:
            audio (Numpy array): audio file to process.
            fs (float): sampling frequency of the audio file.
        Returns:
            audio_f (Tensor): spectral representation of the audio data.
        """
        audio_f = np.fft.fft(audio)
        # Preserve only positive frequencies, ensuring the correct length
        audio_f = np.abs(audio_f[:len(audio_f) // 2 + 1])
        audio_f = audio_f / np.max(audio_f)
        return torch.tensor(audio_f, dtype=torch.float32)

    @staticmethod
    def mfcc(audio, fs):
        """
        Calculates the Mel Frequency Cepstral Coefficients (MFCCs) of the audio data.
        Args:
            audio (Numpy array): audio file to process.
            fs (float): sampling frequency of the audio file.
        Returns:
            mfcc (Tensor): MFCC of the input audio file.
        """
        mfcc_features = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13)
        mfcc_features = np.mean(mfcc_features, axis=1)
        mfcc_features = mfcc_features.flatten()

        # Ajustar el vector a la longitud esperada
        expected_length = 320
        if len(mfcc_features) < expected_length:
            mfcc_features = np.pad(mfcc_features, (0, expected_length - len(mfcc_features)), 'constant')
        elif len(mfcc_features) > expected_length:
            mfcc_features = mfcc_features[:expected_length]

        return torch.tensor(mfcc_features, dtype=torch.float32)

