# Student Habits vs Academic Performance

This project analyzes how different lifestyle and behavioral factors influence students' academic performance using regression techniques. The dataset includes features like study hours, screen time, sleep patterns, part-time job status, diet quality, and exercise frequency.

## Objective

To build a regression model that predicts students' academic performance (likely as a continuous variable such as GPA or score) based on their daily habits and lifestyle choices.

## Dataset Features

| Feature                 | Description |
|-------------------------|-------------|
| `student_id`            | Unique identifier for each student |
| `age`                   | Student's age |
| `gender`                | Gender identity (Male, Female, Other) |
| `study_hours_per_day`   | Average daily study hours |
| `social_media_hours`    | Average daily social media usage |
| `netflix_hours`         | Average daily time spent watching Netflix or similar |
| `part_time_job`         | Whether the student has a part-time job |
| `attendance_percentage` | Attendance in class (0–100%) |
| `sleep_hours`           | Average daily sleep duration |
| `diet_quality`          | Categorical: Poor, Fair, Good, Excellent |
| `exercise_frequency`    | Number of workouts per week |
| `academic_performance`  | Target variable (e.g., GPA or final score) |

## Tools & Libraries

- **Python**
- **pandas**, **numpy** — Data manipulation
- **matplotlib**, **seaborn** — Visualization
- **scikit-learn** — Modeling (regression)
- **Jupyter Notebook** — Development environment

## Approach

- Handled missing values and cleaned data
- Performed exploratory data analysis (EDA)
- Encoded categorical features
- Applied regression models (e.g., Linear Regression, Ridge, Lasso, ElasticNet)
- Evaluated model performance using metrics like MAE and R² score

## Results

- **Mean Absolute Error (MAE):** 0.5642305340105693
- **R² Score:** 0.9842993364555513

## Future Improvements

- Try tree-based regressors like RandomForest or XGBoost
- Add SHAP values for interpretability
- Collect a larger or more diverse dataset



---


