# Music Genre Classification using CNN

This project uses a Convolutional Neural Network (CNN) built with Keras and TensorFlow to classify music into 10 different genres based on their audio features (MFCCs).

## Demo Video
https://github.com/user-attachments/assets/b972ef58-335c-4851-955c-32258bb2543f

## Features
- Classifies 10 music genres from the GTZAN dataset.
- Extracts Mel-Frequency Cepstral Coefficients (MFCCs) using Librosa.
- Uses a Keras CNN model for classification.
- Provides a command-line script to predict the genre of any audio file.

## Project Structure
- `data_and_training.ipynb`: Jupyter Notebook for data processing, model training, and evaluation.
- `predict.py`: Command-line script to classify a new audio file.
- `visualize.py`: Generate a spectrogram for an audio file
- `requirements.txt`: A list of all required Python packages.

## Setup and Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/debug-soham/Music-Genre-Classification.git
    cd music-genre-classification
    ```

2.  **Download the Dataset:**
    - Download the GTZAN Genre Collection dataset from [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
    - Unzip it and place the `genres_original` folder inside a new `Data` folder in the project root. The final path should be `Data/genres_original/`.
    *(Note: The Data folder is not included in this repository due to its size.)*

3.  **Create a Virtual Environment:**
    ```bash
    # Make sure you have Python 3.11 installed
    py -3.11 -m venv my_env
    .\my_env\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1.  **Train the Model:**
    - Open and run all the cells in the `01_data_and_training.ipynb` notebook.
    - This will create two files: `data.json` and `music_genre_classifier.keras`.

2.  **Predict a Genre:**
    - Use the `predict.py` script to classify an audio file:
    ```bash
    python predict.py "path/to/your/song.wav"
    ```
3. **Visualize a Spectrogram (Optional)**
    - Use the `visualize.py` script to generate a spectrogram for any audio file:
    ```bash
    python visualize.py "path/to/your/song.wav"

## Technologies Used
- Python 3.11
- TensorFlow / Keras
- Librosa
- Scikit-learn
- NumPy
- Matplotlib
