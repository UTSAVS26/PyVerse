# Visualizing tools

## 🎯 Goal
The primary objective of this project is to explain your specific project's goal here. It includes implementing, visualizing, and analyzing relevant key features based on the files and project you shared.

## 🧵 Dataset / Input
This project does not rely on a predefined dataset. Instead, it takes user input in the following formats:

- **File Upload**: Users can upload a file containing the input data. Make sure to follow the input file format described below.
- **Manual Entry**: Users can manually input the data.

### File Format Requirements:
- **File1**: Accepts input in format (e.g., CSV), containing fields such as `Country`, `Date_reported`, `New_cases`, etc.

### Example Input: 
- **Manual Input Format Example**:
    ```plaintext
    Country: value
    Date_reported: value
    New_cases: value
    ```

- **File Input Example**: CSV
    ```plaintext
    Country,Date_reported,New_cases
    Country1,2023-01-01,10
    Country2,2023-01-01,20
    ```

## 🧾 Description
This project provides a web-based platform developed using Streamlit that allows users to interact with the system through manual input or file uploads. The tool implements specific project functionalities such as visualizing daily COVID-19 new cases and forecasting using ARIMA.

### Features:
- Real-time visualization of COVID-19 data based on user input.
- Multiple input modes: manual and file-based.
- Dynamic operations that analyze and visualize data trends and predictions.

## 🧮 What I Had Done!
- Developed an interactive web interface using Streamlit.
- Implemented features for file uploading and manual input handling.
- Visualized the operations or data structures in real-time.
- Provided feedback on every step of the process, including analysis and forecasting.

### Sample Output:
- After processing the input, the system generates visualizations and forecasts based on the data provided. For example:
    - **Forecast Visualization**:
    ```plaintext
    Future New Cases Forecast: 
    [Date: value, Forecasted New Cases: value]
    ```

## 📚 Libraries Needed
To run this project, install the following libraries:
- streamlit: for building the web interface.
- plotly: for visualizations.
- statsmodels: for ARIMA modeling.
- scikit-learn: for metrics.
- pandas: for data manipulation.

Install them using:
```bash
pip install -r requirements.txt
