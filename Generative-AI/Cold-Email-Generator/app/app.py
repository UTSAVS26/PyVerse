import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import re


# Load environment variables
load_dotenv()


def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\\s{2,}', ' ', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


# Chain class handling the LLM processing
class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links, user_name, user_about):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are {user_name}. {user_about}
            Your job is to write a cold email to the client regarding the job mentioned above, describing how you can contribute to fulfilling their needs.
            Also, add the most relevant ones from the following links to showcase portfolio: {link_list}
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links, "user_name": user_name, "user_about": user_about})
        return res.content


# Portfolio class using temporary in-memory storage
class Portfolio:
    def __init__(self):
        # Initialize a dictionary to store skills and portfolio links temporarily
        if 'portfolio' not in st.session_state:
            st.session_state['portfolio'] = []

    def add_to_portfolio(self, skills, links):
        """Add the user's skills and portfolio links to temporary storage."""
        if skills and links:
            st.session_state['portfolio'].append({"skills": skills, "links": links})

    def query_links(self, required_skills):
        """Query the temporary storage for relevant links based on provided skills."""
        if not required_skills:
            return []

        # Find relevant portfolio entries based on skills
        matched_links = []
        for entry in st.session_state['portfolio']:
            portfolio_skills = entry['skills']
            if any(skill in portfolio_skills for skill in required_skills):
                matched_links.append(entry['links'])

        return matched_links[:2]  # Return up to 2 matched links


# Function to create the Streamlit app interface
def create_streamlit_app(llm, portfolio, clean_text):
    st.set_page_config(page_title="Cold Email Generator", page_icon="", layout="wide")

    st.markdown("""
    <style>
        .main {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            color: #e0e0e0;
            font-size: 2.5em;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #b0b0b0;
            font-size: 1.2em;
        }
        .container {
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
            background-color: #1e1e1e;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        }
        .input-box {
            margin-bottom: 20px;
        }
        .input-box input, .input-box textarea {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #333;
            background-color: #2c2c2c;
            color: #e0e0e0;
            box-sizing: border-box;
        }
        .button {
            background-color: #007BFF;
            color: #e0e0e0;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }
        .button:hover {
            background-color: #0056b3;
        }
        .code-block {
            background-color: #2c2c2c;
            padding: 10px;
            border-radius: 5px;
            color: #e0e0e0;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stTextInput>div>div>textarea {min-height: 100px;}
        .stButton>button {background-color: #007BFF;}
        .stButton>button:hover {background-color: #0056b3;}
        .footer {
            text-align: center;
            color: #e0e0e0;
            font-size: 1em;
            margin-top: 20px;
            padding: 10px;
            background-color: #333;
            border-radius: 5px;
        }
        .footer a {
            color: #66b3ff;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>Cold Email Generator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Effortlessly craft professional cold emails for job applications based on job postings.</div>", unsafe_allow_html=True)

    st.markdown("<div class='container'>", unsafe_allow_html=True)

    user_name = st.text_input("Enter your name:", value=" ")
    user_about = st.text_area(
        "Enter a brief description about yourself:",
        value=" "
    )

    url_input = st.text_input("Enter a Job Post URL:", value=" ")

    st.subheader("Enter Your Skills and Portfolio Links")
    skills_input = st.text_area("Enter your skills (comma separated):", value="")
    links_input = st.text_area("Enter your portfolio links (comma separated):", value="")

    submit_button = st.button("Submit", key='submit_button', help="Click to generate the cold email")

    if submit_button:
        try:
            skills_list = [skill.strip() for skill in skills_input.split(",")]
            links_list = [link.strip() for link in links_input.split(",")]

            portfolio.add_to_portfolio(skills_list, links_list)

            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            jobs = llm.extract_jobs(data)

            # Initialize an empty email variable for editing
            email_content = ""

            for job in jobs:
                job_skills = job.get('skills', [])
                links = portfolio.query_links(job_skills)
                email_content = llm.write_mail(job, links, user_name, user_about)
                st.markdown(f"<div class='code-block'><pre>{email_content}</pre></div>", unsafe_allow_html=True)

            # Add a text area for manual editing of the generated email
            edited_email = st.text_area("Edit the generated email:", value=email_content, height=200)

            # Options to save as draft, download, or copy to clipboard
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Save as Draft"):
                    st.session_state['draft_email'] = edited_email
                    st.success("Email saved as draft!")

            with col2:
                if st.button("Download Email"):
                    # Save the email as a text file for download
                    st.download_button("Download Email", edited_email, file_name="email.txt", mime="text/plain")

            with col3:
                if st.button("Copy to Clipboard"):
                    st.markdown(f"<script>navigator.clipboard.writeText(`{edited_email}`);</script>")
                    st.success("Email copied to clipboard!")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# Initialize classes and run the Streamlit app
llm = Chain()
portfolio = Portfolio()

if __name__ == "__main__":
    create_streamlit_app(llm, portfolio, clean_text)
