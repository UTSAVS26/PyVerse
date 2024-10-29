# Anomaly Detection in Time Series 📈

## Table of Contents
- [GOAL](#goal)
- [DATASET](#dataset)
- [DESCRIPTION](#description)
- [WHAT I HAD DONE](#what-i-had-done)
- [🧠 Models Implemented](#-models-implemented)
- [📚 Libraries Needed](#-libraries-needed)
- [📈 Exploratory Data Analysis (EDA) Results](#-exploratory-data-analysis-eda-results)
- [📉 Performance of the Models based on Accuracy Scores](#-performance-of-the-models-based-on-accuracy-scores)
  - [Model Analysis](#model-analysis)
- [📢 Conclusion](#-conclusion)

## GOAL 🎯
The primary goal of this project is to develop an effective anomaly detection system for time series data using various machine learning models, including LSTM (Long Short-Term Memory), Facebook Prophet, and Isolation Forest. The system aims to identify and classify anomalous points in synthetic time series data, enabling better decision-making in applications where anomaly detection is crucial.

## DATASET 📊
The dataset used for this project is synthetically generated time series data, which consists of a sine wave with added Gaussian noise. This simulated dataset mimics real-world scenarios where data might have seasonal patterns and random fluctuations.

### Key Characteristics:
- **Length**: 1000 time steps
- **Features**: One-dimensional time series data generated using a sine function with added noise
- **Anomalies**: Artificially introduced anomalies at randomly selected indices to facilitate evaluation.

## DESCRIPTION 📖
In this project, I developed a pipeline for detecting anomalies in time series data using various models. The process included data generation, preparation, model implementation, evaluation, and comparison based on accuracy scores. The focus was on understanding the strengths and weaknesses of each model in identifying anomalies.

## WHAT I HAD DONE ✔️
1. **Synthetic Data Generation**: Created a time series dataset using a sine wave with noise.
2. **Data Preparation**: Scaled the data and prepared sequences for LSTM modeling.
3. **Model Implementation**: Implemented three models for anomaly detection:
   - LSTM (Long Short-Term Memory)
   - Facebook Prophet
   - Isolation Forest
4. **Evaluation**: Evaluated the performance of each model using accuracy scores and classification reports.
5. **Visualization**: Created plots to visualize original data, predicted values, and detected anomalies.

## 🧠 Models Implemented
1. **LSTM**: A recurrent neural network architecture designed to learn long-term dependencies in sequential data.
2. **Facebook Prophet**: A forecasting tool designed to handle seasonality and trends, suitable for time series data.
3. **Isolation Forest**: An ensemble method specifically designed for anomaly detection in high-dimensional datasets.

## 📚 Libraries Needed
- `numpy`: For numerical operations and data manipulation.
- `pandas`: For data handling and preparation.
- `matplotlib`: For data visualization.
- `scikit-learn`: For model evaluation and metrics.
- `keras`: For building and training the LSTM model.
- `prophet`: For implementing the Facebook Prophet model.
- `statsmodels`: For statistical tests and analysis.

## 📈 Exploratory Data Analysis (EDA) Results
- **Data Visualization**: The initial time series was plotted to observe trends and fluctuations.
- **Statistical Summary**: Key statistics (mean, median, standard deviation) were calculated to understand the data distribution.
- **Autocorrelation Analysis**: The autocorrelation function was analyzed to identify any significant lags in the data, which helps in understanding temporal dependencies.

## 📉 Performance of the Models based on Accuracy Scores
### Model Evaluation Metrics
The models were evaluated based on accuracy, precision, recall, and F1-score. Below are the results for each model:

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| LSTM               | 79%      | 0.02      | 0.15   | 0.03     |
| Facebook Prophet    | 86%      | 0.03      | 0.20   | 0.05     |
| Isolation Forest    | 88%      | 0.02      | 0.10   | 0.03     |

### Model Analysis
- **LSTM**: The LSTM model achieved an accuracy of 79%, with a precision of 0.02 and a recall of 0.15 for detecting anomalies. This suggests that while it identifies the majority class effectively, it struggles to recognize the minority class.
- **Facebook Prophet**: The Prophet model had a better accuracy of 86%, but with a precision of 0.03 and a recall of 0.20 for anomalies. This indicates that it, too, performs well on the majority class but struggles with false positives in anomaly detection.
- **Isolation Forest**: The Isolation Forest model had the highest accuracy at 88%, but with a precision of 0.02 and a recall of 0.10 for anomalies. This highlights the model's tendency to misclassify anomalies, but it effectively identifies the majority class.

### Conclusion on Performance
All models demonstrated a similar pattern where they effectively identified the majority class (0.0) but struggled significantly with the minority class (1.0). Future iterations may involve adjusting thresholds or using more advanced techniques like ensemble methods to improve performance in detecting anomalies.
