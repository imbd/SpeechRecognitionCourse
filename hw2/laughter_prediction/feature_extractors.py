import os
import tempfile
import pandas as pd
import librosa
import numpy as np


class FeatureExtractor:
    def extract_features(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""

    def __init__(self):
        self.extract_script = "./extract_pyAA_features.py"
        self.py_env_name = "ipykernel_py2"

    def extract_features(self, wav_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            feature_save_path = tmp_file.name
            cmd = "python \"{}\" --wav_path=\"{}\" " \
                  "--feature_save_path=\"{}\"".format(self.extract_script, wav_path, feature_save_path)
            os.system("source activate {}; {}".format(self.py_env_name, cmd))

            feature_df = pd.read_csv(feature_save_path)
        return feature_df


class MyExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""

    def __init__(self, frame_sec):
        self.frame_sec = frame_sec

    def extract_features(self, wav_path):
        print(wav_path)
        y, sr = librosa.load(wav_path)
        frame_len = int(sr * self.frame_sec)
        features = []
        for i in range(0, len(y) - frame_len + 1, frame_len):
            new_features = np.concatenate((np.mean(librosa.feature.mfcc(y[i: i + frame_len], sr).T, axis=0),
                                           np.mean(librosa.feature.melspectrogram(y[i: i + frame_len], sr).T,
                                                   axis=0)))
            features.append(new_features)
        return pd.DataFrame(np.array(features))
