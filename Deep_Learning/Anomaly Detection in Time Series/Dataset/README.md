## DATASET 📊

The dataset used for this project is synthetically generated time series data, designed to simulate scenarios often encountered in real-world applications. It consists of a sine wave pattern with added Gaussian noise to reflect the inherent randomness and variability found in many datasets.

### Key Characteristics:
- **Length**: 1000 time steps, providing a sufficient amount of data for training and testing the models.
- **Features**: 
  - **One-dimensional time series**: The primary feature of the dataset is a single continuous value generated using a sine function.
  - **Noise**: Gaussian noise is added to the sine wave to create a more realistic dataset and challenge the anomaly detection models.
  
### Anomalies:
- **Artificial Anomalies**: Anomalies are artificially introduced into the dataset at randomly selected indices. This allows for testing the models' abilities to detect deviations from the normal pattern.
- **Anomaly Proportion**: Approximately 2% of the dataset consists of anomalies, making the detection task more challenging due to the class imbalance.

### Data Generation Process:
1. **Sine Wave Generation**: A sine function is used to create a periodic pattern.
2. **Noise Addition**: Gaussian noise is added to the sine wave, simulating real-world data irregularities.
3. **Anomaly Insertion**: Random points are selected within the dataset to introduce anomalies, represented as significant deviations from the expected sine wave pattern.


