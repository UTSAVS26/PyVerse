# 🩺 Understanding Diabetes 🩺

According to the National Institute of Health (NIH), **Diabetes** is a disease that occurs when your blood glucose, also called blood sugar, is too high. Most of the food we eat is broken down into a sugar called glucose, and insulin is the hormone that enables glucose to get into our cells.

Diabetes is caused by the body’s inability to produce enough insulin or to properly utilize the insulin it produces, resulting in excess glucose in the blood, leading to significant health issues. Although there is no cure for diabetes, steps can be taken to preserve health.

## Types of Diabetes 📊

There are three major types of diabetes:

### Type 1 Diabetes 🚼
- **Description:** Your body does not produce insulin.
- **Impact:** Targets and kills the insulin-producing cells in your pancreas.
- **Demographics:** Most commonly diagnosed in children and young adults.

### Type 2 Diabetes 🚻
- **Description:** Your body does not generate or utilize insulin well.
- **Impact:** Most common type; can be developed at any age.
- **Importance:** Critical to get tested as no symptoms may appear.

### Gestational Diabetes 🤰
- **Description:** Occurs in certain women during pregnancy.
- **Impact:** Typically goes away after childbirth.
- **Risk:** Increases the likelihood of acquiring type 2 diabetes later.

## Dataset Description 📊

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases and was downloaded from Kaggle. It consists of 768 instances and eight attributes. The objective is to diagnose diabetes in Pima Indians living in America.

### Attributes 📋:
1. **Pregnancies:** Number of times pregnant
2. **Glucose:** Glucose concentration in the blood (plasma) after eating.
3. **Blood Pressure:** Diastolic blood pressure (mm Hg).
4. **SkinThickness:** Triceps skinfold thickness (mm).
5. **Insulin:** 2-Hour serum insulin (mu U/ml).
6. **BMI:** Body mass index (weight in kg / (height in m²)).
7. **DiabetesPedigreeFunction:** Scores the likelihood of diabetes based on family history.
8. **Outcome (target variable):** 0 — no diabetes, 1 — has diabetes.

Out of 768 instances, 268 are diabetic, and 500 are non-diabetic.

*Note: All patients are females and at least 21 years old.*

## How to Run the Streamlit Web App:- 🚀

1. **Install Streamlit:**
   ```
   pip install streamlit
   ```

2. **Navigate to the Project Directory:**
   ```
   cd Diabetes-Prediction
   ```

3. **Run the Streamlit App:**
   ```
   streamlit run app.py
   ```

4. **Open the App in Your Browser:**
   It will automatically open your web browser and redirect you to the URL provided by Streamlit (usually http://localhost:8501).
🌟🩺