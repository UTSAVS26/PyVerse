## Death Rate Analysis

### 🎯 **Goal**

The main goal of the "Death Rate Analysis" project is to assess mortality statistics across various countries based on different causes of death. The purpose is to aid in improving health policies by analyzing gender-wise death rates and understanding their relationship with factors such as the Human Development Index (HDI) and Gross Domestic Product (GDP).

### 🧵 **Dataset**

- WHO Death Rate Dataset: Gender-wise death rates for 18 causes across 151 countries in 2004, retrieved from the WHO Global Health Observatory (https://www.who.int/data/gho/data/indicators).
- HDI Data: Scraped from CountryEconomy (https://countryeconomy.com/hdi?year=2004) for 172 countries.
- GDP Data: Downloaded from World Bank (https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG) for 181 countries.

### 🧾 **Description**

This project analyzes death rates categorized by gender and examines their relationships with HDI and GDP. Various visualizations were created to identify trends and anomalies. The study aims to provide insights into how socio-economic factors affect mortality rates globally, particularly across developing and developed countries.

### 🧮 **What I had done!**

- Collected datasets on death rates, HDI, and GDP for analysis.
- Cleaned and filtered the datasets for consistency (intersection of countries).
- Visualized relationships between death rates, HDI, and GDP.
- Analyzed gender-wise patterns and biases in the data.
- Identified outliers and formulated insights on socio-economic effects on mortality.

### 🚀 **Models Implemented**

Linear Regression: Used to examine the correlation between death rates and socio-economic factors (HDI, GDP).

### 📚 **Libraries Needed**

- ggplot2: For data visualization.
- dplyr: For data manipulation.
- tidyverse: A suite for data cleaning and visualization.

### 📊 **Exploratory Data Analysis Results**

![plot_6](https://github.com/user-attachments/assets/4b3e41eb-557e-46fa-b625-d0a1168fbf2a)
![plot_5](https://github.com/user-attachments/assets/0b93c6c1-e9cd-4f16-9cd5-50560d999d08)
![plot_4](https://github.com/user-attachments/assets/8d311bce-0f24-4c4d-a3ac-3282b862fdaf)
![plot_3](https://github.com/user-attachments/assets/eabb3342-0111-41d7-b6b1-a99f0c7051b8)
![plot_2](https://github.com/user-attachments/assets/1fe615e4-db1b-49ea-b2ce-8f562e020a3e)
![plot_1](https://github.com/user-attachments/assets/46969721-f78e-419c-ace7-bb7026bb98fa)



### 📈 **Performance of the Models based on the Accuracy Scores**

Linear regression showed:

- Negative correlations between death rates and HDI/GDP for most causes.
- An exception was colon and rectum cancer, which showed a positive correlation.


### 📢 **Conclusion**

The analysis shows that mortality rates are significantly influenced by socio-economic factors, with notable gender-specific trends in developing countries. The findings suggest targeted health interventions in specific regions, like breast cancer awareness in Armenia. Linear regression proved effective in highlighting these correlations.

### ✒️ **Your Signature**

Sharayu Anuse
