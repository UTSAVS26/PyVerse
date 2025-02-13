# Bulldozer Price Prediction using ML

## Overview
This project aims to predict the auction prices of bulldozers using machine learning techniques. The dataset used for this project comes from the Kaggle competition "Blue Book for Bulldozers," which provides historical data on bulldozer sales.

## Dataset
The dataset includes various features such as machine specifications, sale dates, and operational conditions. The primary objective is to predict the **sale price** of a bulldozer given its attributes.

### Data Sources
- The data is obtained from the [Kaggle Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data)
- The dataset contains multiple CSV files, including:
  - `Train.csv`: Historical data of bulldozers with known sale prices.
  - `Valid.csv`: Validation dataset to test model performance.
  - `Test.csv`: Data for final model predictions.

## Machine Learning Approach
The following steps are followed in building the machine learning model:
1. **Data Preprocessing**
   - Handling missing values
   - Feature engineering
   - Encoding categorical variables
   
2. **Exploratory Data Analysis (EDA)**
   - Identifying trends and relationships
   - Visualizing key insights
   
3. **Model Selection and Training**
   - Random Forest Regressor
   - Hyperparameter tuning using RandomizedSearchCV

4. **Model Evaluation**
   - Root Mean Squared Log Error (RMSLE)
   - R² Score

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Bulldozer-Price-Prediction-using-ML.git
   cd Bulldozer-Price-Prediction-using-ML
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook to explore the data and train the model:
   ```bash
   jupyter notebook
   ```

## Dependencies
The project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Results & Insights
- The best model achieved an RMSE of **X.XX** on the validation set.
- Feature importance analysis showed that `YearMade`, `UsageBand`, and `ProductSize` were key factors influencing bulldozer prices.
- The model performed well on test data, generalizing effectively.

## Future Improvements
- Try deep learning models such as neural networks.
- Incorporate additional data sources for better predictions.
- Deploy the model as a web app using Flask or FastAPI.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`feature-branch`)
3. Commit changes
4. Push to the branch and create a pull request

## License
This project is licensed under the MIT License.
