import librosa
import librosa.display
import matplotlib.pyplot as plt
import argparse

def create_spectrogram(file_path):
    """Loads an audio file and displays its spectrogram."""
    try:
        signal, sr = librosa.load(file_path, sr=22050)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Create a spectrogram
    stft = librosa.stft(signal)
    spectrogram = librosa.amplitude_to_db(abs(stft))

    # Plot the spectrogram
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectrogram, sr=sr, hop_length=512)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format='%+2.0f dB').set_label('Decibels')
    plt.title(f"Spectrogram for {file_path.split('/')[-1]}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and display a spectrogram for an audio file.")
    parser.add_argument("file", type=str, help="Path to the audio file.")
    args = parser.parse_args()

    create_spectrogram(args.file)