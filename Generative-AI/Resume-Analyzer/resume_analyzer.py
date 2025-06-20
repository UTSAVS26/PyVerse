import os
import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
import json
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space


load_dotenv()  # Load environment variables

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Prompt Template
input_prompt = """
Act as an expert ATS with deep knowledge in tech fields like software engineering, data science, data analysis, and big data engineering. Evaluate the resume based on the job description, considering the competitive job market. Provide the best assistance for improving resumes. Assign a matching percentage and list missing keywords accurately.

Resume: {text}
Job Description: {jd}

Response structure:
{{"JD Match": "%", "MissingKeywords": [], "Profile Summary": ""}}
"""

# Streamlit app
st.set_page_config(page_title='Resume Analyzer',
                   layout='wide',
                   page_icon="📃👩‍💻")

with st.sidebar:
    st.title("Smart ATS!!📃")
    st.subheader("About")
    st.write("""
    📌Welcome to the Smart ATS for Resumes!

    This advanced tool, powered by Gemini Pro and Streamlit, helps you optimize your resume for Applicant Tracking Systems (ATS). Increase your chances of landing your dream job by ensuring your resume includes crucial keywords and meets industry standards.

    **Features:**
    - **Resume Analysis**: Upload your resume for detailed insights.
    - **Keyword Optimization**: Identify missing keywords based on the job description.
    - **Match Percentage**: See how well your resume aligns with the job description.
    - **Profile Summary**: Get personalized suggestions to enhance your resume.

    Stand out in the competitive job market with our Smart ATS for Resumes.
""")

    add_vertical_space(4)
    st.info("📍Please refer to the following websites and videos for ideas on resume making.\n\n")
    st.markdown("""
                - [Website-1](https://resumegenius.com/blog/resume-help/fresher-simple-resume-format)
                - [Website-2](https://www.simplilearn.com/resume-tips-for-freshers-article)
                - [Watch Video-1](https://youtu.be/y3R9e2L8I9E?si=l0_i6AcqLSkRs7KJ)
                - [Watch Video-2](https://youtu.be/ZwP7kv0zHiY?si=rp1FLZOQkUMEHgyh)

                """)

st.title("👩‍💻Smart Resume Analyzer🖇️")
st.text("Find your areas for improvement to become more powerful and unique!")
jd = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the PDF")

submit = st.button("Submit")

if submit:
    if jd and uploaded_file is not None:
        text = input_pdf_text(uploaded_file)
        formatted_prompt = input_prompt.format(text=text, jd=jd)
        response = get_gemini_response(formatted_prompt)
        response_json = json.loads(response)
        st.subheader("Response")
        st.json(response_json)
    else:
        st.error("Please provide both the job description and upload a resume.")

def print_praise():
    praise_quotes = """
    Prerita Saini✨
    """
    title = "**Created By -**\n\n"
    return title + praise_quotes

with st.sidebar:
    add_vertical_space(7)  
    st.sidebar.success(print_praise()) 
