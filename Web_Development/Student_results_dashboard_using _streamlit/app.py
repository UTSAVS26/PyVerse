import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Student Results Dashboard", layout="wide")
st.title("Student Results Analysis - Data Science Department")

# Sidebar Navigation
Batch = st.sidebar.radio("Select Batch", ["2022", "2021"])
year = st.sidebar.radio("Select Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
sem = st.sidebar.radio("Select Semester", ["2nd Semester", "1st Semester"])

# File path
year_folder = year.lower().replace(" ", "_")
sem_file = "sem1.csv" if "1st" in sem else "sem2.csv"
file_path = f"data/{Batch}/{year_folder}/{sem_file}"

# Load Data
if not os.path.exists(file_path):
    st.error(f"Data not found for Batch {Batch}, {year} {sem}")
    st.stop()

df = pd.read_csv(file_path)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df[df["Name"].str.lower() != "detained"]

# Section Filter
sections = df['Section'].unique().tolist()
selected_sections = st.multiselect("Select Sections", sections, default=sections)
filtered_df = df[df['Section'].isin(selected_sections)]

# Result Column
filtered_df["Result"] = filtered_df["Subjects_due"].apply(lambda x: "Pass" if str(x) == "0" else "Fail")

# Display Data
st.subheader(f"Results for Batch {Batch} - {year} - {sem}")
st.dataframe(filtered_df, use_container_width=True)

# Detect subject and status columns
subject_cols = [
    col for col in filtered_df.columns
    if col not in ["Roll Number", "Name", "Section", "Subjects_due", "SGPA", "CGPA", "Result"]
    and not col.endswith("_Status")
    and pd.api.types.is_numeric_dtype(filtered_df[col])
]

status_cols = [col for col in filtered_df.columns if col.endswith("_Status")]

# 1. Subject-wise Pass/Fail Table (Section-wise)
st.subheader("Subject-wise Pass/Fail Count & Pass Percentage (by Section)")
subject_pass_fail = []

for status_col in status_cols:
    subject = status_col.replace("_Status", "")
    for sec in filtered_df['Section'].unique():
        sec_data = filtered_df[filtered_df['Section'] == sec]
        passed = (sec_data[status_col] == "Pass").sum()
        failed = (sec_data[status_col] == "Fail").sum()
        total = passed + failed
        if total == 0: continue
        pass_percent = round((passed / total) * 100, 2)
        subject_pass_fail.append({
            "Section": sec,
            "Subject": subject,
            "Passed": passed,
            "Failed": failed,
            "Pass%": pass_percent
        })

pass_fail_df = pd.DataFrame(subject_pass_fail).sort_values(by=["Section", "Subject"])
st.dataframe(pass_fail_df, use_container_width=True)

# 2. Section-wise Summary Table
st.subheader("Section-wise Result Summary")
summary_data = []

for sec in filtered_df['Section'].unique():
    total = len(filtered_df[filtered_df['Section'] == sec])
    passed = len(filtered_df[(filtered_df['Section'] == sec) & (filtered_df['Result'] == "Pass")])
    failed = total - passed
    pass_per = round((passed / total) * 100, 2)
    fail_per = 100 - pass_per
    summary_data.append({
        "Section": sec,
        "Total": total,
        "Pass": passed,
        "Fail": failed,
        "Pass%": pass_per,
        "Fail%": fail_per
    })

summary_df = pd.DataFrame(summary_data).sort_values(by="Section")
st.dataframe(summary_df)

# 3. Section-wise Top 5 Students
st.subheader("Top 5 Students per Section")
for sec in filtered_df['Section'].unique():
    st.markdown(f"#### Section {sec}")
    top5 = filtered_df[filtered_df['Section'] == sec].sort_values("CGPA", ascending=False).head(5)
    st.dataframe(top5[["Roll Number", "Name", "SGPA", "CGPA"]], use_container_width=True)

# 4. Section-wise Pass Percentage (Pie Chart)
st.subheader("Section-wise Pass Percentage (Pie Chart)")
try:
    pie_data = summary_df[["Section", "Pass%"]].dropna()
    pie_data = pie_data[pie_data["Pass%"] > 0]
    fig_pie = px.pie(pie_data, names="Section", values="Pass%", title="Section-wise Pass %", hole=0.4)
    fig_pie.update_traces(textinfo='percent+label+value')
    st.plotly_chart(fig_pie, use_container_width=True)
except Exception as e:
    st.warning(f"Could not generate pie chart: {e}")

# Section-wise Pass Count
st.subheader("Section-wise Pass/Fail Count")
fig1 = px.histogram(filtered_df, x="Section", color="Result", barmode="group", title="Pass/Fail Count by Section")
fig1.update_traces(texttemplate='%{y}', textposition='inside')
st.plotly_chart(fig1, use_container_width=True)

# Overall Result Distribution
st.subheader("Overall Result Distribution")
fig2 = px.pie(filtered_df, names="Result", title="Pass vs Fail Ratio", hole=0.3)
fig2.update_traces(textinfo='percent+label+value')
st.plotly_chart(fig2, use_container_width=True)

# Top 10 Performers
st.subheader("Overall Top 10 Performers")
top_students = filtered_df.sort_values("CGPA", ascending=False).head(10)
st.dataframe(top_students[["Roll Number", "Name", "Section", "SGPA", "CGPA"]], use_container_width=True)
