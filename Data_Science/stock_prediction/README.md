## Stock Prediction Using CAPM and Fama-French Three-Factor Model

### 🎯 **Goal**

The main goal of this project is to predict the stock returns for Mahindra Company using two financial models: the Capital Asset Pricing Model (CAPM) and the Fama-French Three-Factor Model. The project aims to compare the accuracy and performance of both models in determining the expected stock returns based on historical data.

### 🧵 **Dataset**

The historical stock data for Mahindra Company has been sourced from Yahoo Finance.

### 🧾 **Description**

This project leverages two well-known models in finance, CAPM and the Fama-French Three-Factor Model, to predict stock returns. The project uses Mahindra’s historical stock data to evaluate how each model performs in explaining the stock's return based on market factors.

The Capital Asset Pricing Model (CAPM) considers only the market risk (Beta) to determine returns, while the Fama-French model incorporates three factors: market risk, size of the firm (SMB), and value vs. growth (HML).

### 🧮 **What I had done!**

- Data Collection: Retrieved Mahindra’s historical stock data from Yahoo Finance.
- Data Preprocessing: Cleaned the data and calculated necessary variables like excess returns, risk-free rate, and market factors.
- CAPM Implementation:
   - Estimated the beta of Mahindra’s stock.
   - Applied the CAPM formula to predict expected returns.
- Fama-French Three-Factor Model:
   - Retrieved and incorporated additional factors: SMB (Small Minus Big) and HML (High Minus Low).
   - Predicted expected returns using the three-factor model.
- Model Evaluation:
   - Calculated statistical measures like R-squared and p-values to evaluate the models' accuracy.
   - Compared the results of the CAPM and Fama-French models.
- Visualization: Generated visual plots to represent the relationship between predicted and actual returns.

### 🚀 **Models Implemented**

- Capital Asset Pricing Model (CAPM): Chosen for its simplicity in predicting stock returns based on the market risk factor (Beta).
- Fama-French Three-Factor Model: Selected because it extends CAPM by incorporating two additional factors, SMB (size factor) and HML (value factor), making it more comprehensive for predicting stock returns.

- Why these models?

CAPM is a fundamental model that provides a baseline understanding of stock returns based on market volatility.
The Fama-French model was chosen for its greater accuracy in capturing returns by considering multiple factors.

### 📚 **Libraries Needed**

- Pandas: For data manipulation and analysis.
- NumPy: To perform numerical operations on arrays.
- Matplotlib: For visualizing data and results.
- Statsmodels: For performing regression analysis and statistical computations.

### 📊 **Exploratory Data Analysis Results**

![output_12](https://github.com/user-attachments/assets/9966a03d-5a8c-4629-858f-eac88d7c51db)
![output_11](https://github.com/user-attachments/assets/32097575-bf49-4c43-b46a-bfca9242fe19)
![output_10](https://github.com/user-attachments/assets/2a181df5-fbf9-4e89-9077-3e4e9e694358)
![output_9](https://github.com/user-attachments/assets/9b74abb2-9a7f-4148-805d-020f14618f45)
![output_8](https://github.com/user-attachments/assets/9903c057-6d84-42a2-82da-320042380ad0)
![output_7](https://github.com/user-attachments/assets/974b08eb-a69c-4d66-9fca-b124bff53203)
![output_6](https://github.com/user-attachments/assets/720d30ed-53b6-4a56-9cd9-6330f920b472)
![output_5](https://github.com/user-attachments/assets/def28983-ca3b-4de1-84e9-612c24cec65e)
![output_4](https://github.com/user-attachments/assets/bb195ec1-ae23-4bec-9da8-275169dcd41d)
![output_3](https://github.com/user-attachments/assets/6dae166b-725e-41d4-9d74-ee51f56b1384)
![output_2](https://github.com/user-attachments/assets/f0622a64-6b1a-4f73-8a82-c05579623689)
![output_1](https://github.com/user-attachments/assets/befa3864-12c0-4d69-961e-ab11f2e572b1)


### 📈 **Performance of the Models based on the Accuracy Scores**

- CAPM Accuracy: The model provided a decent explanation of stock returns but lacked accuracy in certain periods.
- Fama-French Model Accuracy: Outperformed CAPM with an R-squared value that was 36% higher, showing greater accuracy in predicting returns.

The Fama-French model clearly provided a more accurate prediction due to the inclusion of additional factors beyond just market risk.


### 📢 **Conclusion**

In conclusion, while the CAPM model offers a fundamental approach to predicting stock returns, the Fama-French Three-Factor Model is more effective for Mahindra's stock due to its additional factors, leading to a 36% improvement in accuracy. This project highlights the importance of using more comprehensive models like Fama-French for stock prediction tasks.odels for the particular projects.

### ✒️ **Your Signature**

Sharayu Anuse
