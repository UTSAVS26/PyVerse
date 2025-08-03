import json
import numpy as np
import librosa
from keras.models import load_model
import argparse

class MusicGenreClassifier:
    """A class to load a trained model and predict the genre of an audio file."""

    def __init__(self, model_path, data_path):
        # Load the trained model
        self.model = load_model(model_path)

        # Load genre mappings
        with open(data_path, "r") as fp:
            data = json.load(fp)
        self.mapping = data["mapping"]

        # Constants for audio processing
        self.sample_rate = 22050
        self.duration = 30
        self.samples_per_track = self.sample_rate * self.duration
        self.num_segments = 10

    def _preprocess_audio(self, file_path):
        """Loads an audio file, processes it into segments, and extracts MFCCs."""
        mfccs = []
        num_samples_per_segment = int(self.samples_per_track / self.num_segments)

        try:
            signal, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

        # Process each segment
        for s in range(self.num_segments):
            start_sample = num_samples_per_segment * s
            end_sample = start_sample + num_samples_per_segment

            mfcc = librosa.feature.mfcc(
                y=signal[start_sample:end_sample],
                sr=sr,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            mfcc = mfcc.T

            if len(mfcc) == 130:
                mfccs.append(mfcc)

        return np.array(mfccs)

    def predict(self, file_path):
        """Predicts the genre of an audio file using a majority vote from its segments."""
        mfccs_to_predict = self._preprocess_audio(file_path)

        if mfccs_to_predict is None or len(mfccs_to_predict) == 0:
            print("Could not process audio file.")
            return

        # Add channel dimension for the CNN
        mfccs_to_predict = mfccs_to_predict[..., np.newaxis]

        # Get predictions for each segment
        predictions = self.model.predict(mfccs_to_predict)
        predicted_indices = np.argmax(predictions, axis=1)
        
        # Use a majority vote to find the most common genre
        counts = np.bincount(predicted_indices)
        if len(counts) == 0:
            print("Could not make a prediction.")
            return

        final_prediction_index = np.argmax(counts)
        predicted_genre = self.mapping[final_prediction_index]
        confidence = (np.max(counts) / len(predicted_indices)) * 100

        print(f"\nPredicted Genre: {predicted_genre.upper()}")
        print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify the genre of a music file.")
    parser.add_argument("file", type=str, help="Path to the audio file for classification.")
    args = parser.parse_args()

    classifier = MusicGenreClassifier("music_genre_classifier.keras", "data.json")
    classifier.predict(args.file)