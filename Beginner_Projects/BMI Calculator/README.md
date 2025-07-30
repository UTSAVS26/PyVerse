# BMI Calculator with WHO Guidelines
Advanced BMI calculator with age-specific assessment using WHO LMS reference data for children (0-19 years) and standard BMI classification for adults.

# Features
- BMI calculation with multiple unit support (kg/lb, m/cm/in)
- Age-specific BMI assessment using WHO LMS data (0-19 years)
- Adult BMI classification (19+ years)
- Waist-to-Hip Ratio (WHR) calculation and risk assessment
- Z-score and percentile calculation for children
- Clean tkinter GUI

# Requirements
- Python 3.x
- pandas
- numpy
- scipy
- tkinter (included with Python)

# Data Files Required
children_0_5_data.csv - WHO LMS data for 0-5 years (Month, Sex, L, M, S)
children_5_19_data.csv - WHO LMS data for 5-19 years (Age, Sex, L, M, S)

# Usage
python bmi_calculator.py
Enter weight, height, age, sex, waist and hip measurements. The calculator will:

Calculate BMI and classify based on age group
For children: Show Z-score, percentile, and WHO category
For adults: Show standard BMI category
Calculate WHR and cardiovascular risk level

# BMI Categories
- Children (0-19 years):

    Underweight: Z-score < -2
    Normal: Z-score -2 to 1
    Overweight: Z-score 1 to 2
    Obese: Z-score > 2

- Adults (19+ years):

    Underweight: BMI < 18.5
    Normal: BMI 18.5-24.9
    Overweight: BMI 25-29.9
    Obese: BMI ≥ 30

- WHR Risk Levels

    Men: High risk if WHR > 0.90
    Women: High risk if WHR > 0.85

# Authors

* Jyanu Ratna (Current Contributer | GSSOC Contributer)
* Poushali Mitra (Original Contributor | SSOC Contributer)