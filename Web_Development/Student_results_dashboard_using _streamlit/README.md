# Student Results Dashboard

A **Streamlit-based web application** to analyze and visualize student results data for the **Data Science Department**. This dashboard provides section-wise and subject-wise performance insights, pass/fail statistics, top performers, and interactive charts.

---

## Features

- **Batch, Year, and Semester Navigation**
- **Section-wise Subject Analytics**: Pass/Fail counts and percentages
- **Pie and Bar Charts**: Result distribution and performance breakdown
- **Top Performers**: Section-wise and overall toppers
- **Pass/Fail Summary Tables**: Interactive and export-ready

---

## Project Structure

```bash
Directory structure:
└── github username-student-results-dashboard/
  ├── README.md
  ├── app.py
  ├── requirements.txt
  ├── data/
  │    └── 2022/
  │      ├── 1st_year/
  │      │  └── sem2.csv
  │      ├── 2nd_year/
  │      │  ├── sem1.csv
  │      │  └── sem2.csv
  │      └── 3rd_year/
  │         └── sem1.csv
  └── utils/
      └── data_loader.py
```

---

## Demo

**Screenshot-1**
![Screenshot 2025-06-27 002821](https://github.com/user-attachments/assets/d8519164-e182-4020-a847-1b9940346c9f)

**Screenshot-2**
![Screenshot 2025-06-27 002902](https://github.com/user-attachments/assets/05765791-622f-46bd-8470-b69d8a68f6d1)

## Installation (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/student-results-dashboard.git
cd student-results-dashboard
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```
