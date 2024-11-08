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
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\\s{2,}', ' ', text)
    text = text.strip()
    text = ' '.join(text.split())
    return text

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

class Portfolio:
    def __init__(self):
        if 'portfolio' not in st.session_state:
            st.session_state['portfolio'] = []

    def add_to_portfolio(self, skills, links):
        if skills and links:
            st.session_state['portfolio'].append({"skills": skills, "links": links})

    def query_links(self, required_skills):
        if not required_skills:
            return []

        matched_links = []
        for entry in st.session_state['portfolio']:
            portfolio_skills = entry['skills']
            if any(skill in portfolio_skills for skill in required_skills):
                matched_links.append(entry['links'])

        return matched_links[:2]

def create_streamlit_app(llm, portfolio, clean_text):
    st.set_page_config(page_title="Cold Email Generator", page_icon="", layout="wide")

    st.markdown("<div class='title'>Cold Email Generator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Effortlessly craft professional cold emails for job applications based on job postings.</div>", unsafe_allow_html=True)

    st.markdown("<div class='container'>", unsafe_allow_html=True)

    user_name = st.text_input("Enter your name:", value=st.session_state.get('user_name', ""))
    user_about = st.text_area("Enter a brief description about yourself:", value=st.session_state.get('user_about', ""))
    url_input = st.text_input("Enter a Job Post URL:", value=st.session_state.get('url_input', ""))

    st.subheader("Enter Your Skills and Portfolio Links")
    skills_input = st.text_area("Enter your skills (comma separated):", value=st.session_state.get('skills_input', ""))
    links_input = st.text_area("Enter your portfolio links (comma separated):", value=st.session_state.get('links_input', ""))

    submit_button = st.button("Submit", key='submit_button', help="Click to generate the cold email")

    if submit_button:
        try:
            st.session_state['user_name'] = user_name
            st.session_state['user_about'] = user_about
            st.session_state['url_input'] = url_input
            st.session_state['skills_input'] = skills_input
            st.session_state['links_input'] = links_input

            skills_list = [skill.strip() for skill in skills_input.split(",")]
            links_list = [link.strip() for link in links_input.split(",")]

            portfolio.add_to_portfolio(skills_list, links_list)

            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            jobs = llm.extract_jobs(data)

            for job in jobs:
                job_skills = job.get('skills', [])
                links = portfolio.query_links(job_skills)
                email = llm.write_mail(job, links, user_name, user_about)

                st.session_state['draft_email'] = email

        except Exception as e:
            st.error(f"An Error Occurred: {e}")

    if 'draft_email' in st.session_state:
        st.text_area("Edit Email", value=st.session_state['draft_email'], key='edit_email', height=300)
        if st.button("Save Draft"):
            st.session_state['final_email'] = st.session_state['edit_email']
            st.success("Draft saved successfully!")

        if 'final_email' in st.session_state:
            st.markdown(f"<div class='code-block'><pre>{st.session_state['final_email']}</pre></div>", unsafe_allow_html=True)

            # Add a copy button using Streamlit built-in feature
            st.code(st.session_state['final_email'], language='text')

            # Download button for the email
            st.download_button(label="Download Email", data=st.session_state['final_email'], file_name="cold_email.txt", mime="text/plain")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)
