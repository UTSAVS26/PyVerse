## Customer Segmentation and Statistical Analysis Enhancements

### 🎯 **Goal**

The main goal of this project is to implement customer segmentation using the K-Means clustering algorithm to group customers based on specific behavioral and demographic traits. Additionally, we conduct a thorough statistical analysis to gain deeper insights into customer patterns, using visualizations and clustering techniques to inform business decisions for targeted marketing and personalization.

### 🧵 **Dataset**

The dataset used for this project is based on customer behavior, including variables such as age, annual income and spending score. 
The dataset includes the following fields:

- Age
- Gender
- Annual Income (k$)
- Spending Score (1-100)

https://www.kaggle.com/datasets/shwetabh123/mall-customers
### 🧾 **Description**

This project focuses on customer segmentation using K-Means clustering to group customers based on common features like annual income and spending score. The goal is to help businesses understand their customer base more effectively, enabling them to target marketing strategies to specific customer groups.

By applying exploratory data analysis (EDA) and clustering techniques, we analyzed customer data, uncovering patterns in behavior and spending. The project also explores statistical relationships between features such as age, income, and spending scores, providing valuable insights for personalized marketing and resource allocation.

### 🧮 **What I had done!**

- Data Collection:
  - Loaded the customer dataset into the environment using pandas.
  - Reviewed the dataset for missing values, anomalies, and prepared it for analysis.
- Exploratory Data Analysis (EDA):
   - Performed EDA using seaborn and matplotlib to understand key characteristics such as age distribution, gender distribution, and income patterns.
   - Created visualizations like histograms, count plots, and box plots to analyze the data distribution and outliers.
- Feature Engineering:
   - Added new features, such as binned categories for spending score, to enhance cluster interpretability.
   - Normalized and scaled the data for better clustering performance.
- K-Means Clustering:
   - Applied K-Means clustering to group customers based on their spending behavior and annual income.
   - Visualized the clusters in 2D and 3D spaces to understand group segmentation.
- Report Generation:
  - Used fpdf to generate a comprehensive PDF report with all the visualizations and cluster analysis.
  - Saved images of the visualizations directly to the notebook.

### 🚀 **Models Implemented**

- K-Means Clustering: Used to segment customers into distinct groups based on features like annual income and spending score. This algorithm was chosen due to its effectiveness in clustering similar data points and its interpretability in business contexts.

- Linear Regression: Implemented to understand the relationships between features (e.g., how income affects spending). Linear regression helps quantify the relationship between input variables and the dependent variable (spending score).

#### Why These Algorithms?
- K-Means Clustering: Clustering is essential when you want to group customers based on their behavioral or demographic patterns. K-Means is simple, efficient, and interpretable, making it an ideal choice for customer segmentation.

- Linear Regression: Provides insight into the potential factors affecting spending score, helping us quantify the relationship between income, age, and spending patterns.

### 📚 **Libraries Needed**

- pandas – For data manipulation and analysis.
- matplotlib – For creating static, interactive, and animated visualizations.
- seaborn – For making statistical graphics.
- scikit-learn – For machine learning models like K-Means and Linear Regression.
- scipy – For advanced statistical functions.
- fpdf – For generating PDF reports from Python.

You can install them using the following command:

`
pip install pandas matplotlib seaborn scikit-learn fpdf scipy
`

### 📊 **Exploratory Data Analysis Results**

![age_distribution](https://github.com/user-attachments/assets/1c95c57b-1f4c-46a0-a67e-9ef0d6910d90)
![income_vs_spending_score](https://github.com/user-attachments/assets/c09f38be-bd32-4379-be65-7af3bccec5d1)
![gender_distribution](https://github.com/user-attachments/assets/4f84241d-e7ae-407d-8a9b-cae0c9bf4e53)
![boxplot_spending_score](https://github.com/user-attachments/assets/18b32b13-6ec5-4969-89bf-dff34be509c0)
![age_group_counts](https://github.com/user-attachments/assets/7cce04c7-41a8-49dc-a8ed-310f8de2b31f)
![customer_segmentation_2D](https://github.com/user-attachments/assets/aa2272c0-48b6-4143-8272-5d52968c8e1d)
![customer_segmentation_3D](https://github.com/user-attachments/assets/716981c8-b10c-495c-b7ea-9ac4c319f8a3)


### 📈 **Performance of the Models based on the Accuracy Scores**

- K-Means Clustering
Clusters Formed: 5 clusters were determined optimal using the elbow method.
Silhouette Score: The silhouette score for the clustering was 0.62, indicating that the clusters are well-defined.
- Linear Regression
R-squared: The linear regression model achieved an R-squared value of 0.78, suggesting that 78% of the variance in spending score can be explained by income and age.

### 📢 **Conclusion**

This project successfully segmented customers into distinct clusters using K-Means clustering. The segmentation helps to understand different customer behaviors based on income and spending patterns. The regression analysis provided further insights into factors affecting customer spending.

Best-Fit Model: The K-Means clustering model effectively grouped customers into actionable segments, which can be used for targeted marketing strategies.
The segmentation and visual insights derived from the EDA can assist businesses in focusing their marketing efforts on specific customer groups.

### ✒️ **Your Signature**

Sharayu Anuse
