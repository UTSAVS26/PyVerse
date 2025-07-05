import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import re
import PyPDF2
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

# Set page config at the very beginning
st.set_page_config(page_title="Cold Email Generator", page_icon="📧", layout="wide")

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\\s{2,}', ' ', text)
    text = text.strip()
    text = ' '.join(text.split())
    return text

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def send_email(sender_email, sender_password, recipient_email, subject, body, sender_name):
    """Send email using Gmail SMTP"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{sender_name} <{sender_email}>"
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Add body to email
        msg.attach(MIMEText(body, 'plain'))
        
        # Gmail SMTP configuration
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Enable security
        server.login(sender_email, sender_password)
        
        # Send email
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        return True, "Email sent successfully!"
    
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Please check your email and app password."
    except smtplib.SMTPRecipientsRefused:
        return False, "Recipient email address was refused. Please check the recipient email."
    except smtplib.SMTPException as e:
        return False, f"SMTP error occurred: {str(e)}"
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

def extract_recipient_email_from_job_description(job_description):
    """Extract email from job description if available"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, str(job_description))
    return emails[0] if emails else ""

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

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

    def extract_user_info_from_pdf(self, pdf_text):
        """Extract user information, skills, and portfolio links from PDF text"""
        prompt_extract_user = PromptTemplate.from_template(
            """
            ### PDF TEXT:
            {pdf_text}
            ### INSTRUCTION:
            Extract user information from the PDF text (likely a resume/CV). Return a JSON format with the following keys:
            - `name`: Person's full name
            - `about`: Brief professional summary/objective (2-3 sentences)
            - `skills`: List of technical skills, tools, and technologies
            - `portfolio_links`: List of any URLs, GitHub links, LinkedIn, portfolio websites, etc.
            - `email`: Email address if found
            - `phone`: Phone number if found
            
            If any information is not found, return empty string for strings or empty list for lists.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract_user = prompt_extract_user | self.llm
        res = chain_extract_user.invoke(input={"pdf_text": pdf_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse user information from PDF.")
        return res

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
    st.markdown("<div class='title'>Cold Email Generator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Effortlessly craft professional cold emails for job applications based on job postings.</div>", unsafe_allow_html=True)

    st.markdown("<div class='container'>", unsafe_allow_html=True)

    # Add PDF upload section
    st.subheader("📄 Upload Your Resume/Portfolio (Optional)")
    uploaded_file = st.file_uploader("Upload PDF Resume/CV", type=['pdf'], help="Upload your resume to auto-populate your information")
    
    # Initialize session state variables
    if 'pdf_processed' not in st.session_state:
        st.session_state['pdf_processed'] = False
    
    # Process PDF if uploaded
    if uploaded_file is not None and not st.session_state['pdf_processed']:
        with st.spinner("Processing PDF..."):
            try:
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    user_info = llm.extract_user_info_from_pdf(pdf_text)
                    
                    # Store extracted information in session state
                    st.session_state['user_name'] = user_info.get('name', '')
                    st.session_state['user_about'] = user_info.get('about', '')
                    st.session_state['skills_input'] = ', '.join(user_info.get('skills', []))
                    st.session_state['links_input'] = ', '.join(user_info.get('portfolio_links', []))
                    st.session_state['pdf_processed'] = True
                    
                    st.success("✅ PDF processed successfully! Your information has been auto-populated below.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    # Reset button for PDF processing
    if st.session_state.get('pdf_processed', False):
        if st.button("🔄 Reset PDF Data", help="Clear auto-populated data from PDF"):
            st.session_state['pdf_processed'] = False
            st.session_state['user_name'] = ""
            st.session_state['user_about'] = ""
            st.session_state['skills_input'] = ""
            st.session_state['links_input'] = ""
            st.rerun()

    st.subheader("👤 Personal Information")
    user_name = st.text_input("Enter your name:", value=st.session_state.get('user_name', ""))
    user_about = st.text_area("Enter a brief description about yourself:", value=st.session_state.get('user_about', ""), height=100)
    
    st.subheader("🔗 Job Posting")
    url_input = st.text_input("Enter a Job Post URL:", value=st.session_state.get('url_input', ""))

    st.subheader("🛠️ Skills & Portfolio")
    skills_input = st.text_area("Enter your skills (comma separated):", value=st.session_state.get('skills_input', ""), height=100)
    links_input = st.text_area("Enter your portfolio links (comma separated):", value=st.session_state.get('links_input', ""), height=100)

    submit_button = st.button("🚀 Generate Cold Email", key='submit_button', help="Click to generate the cold email", type="primary")

    if submit_button:
        # Validation
        if not user_name.strip():
            st.error("Please enter your name.")
            return
        if not user_about.strip():
            st.error("Please enter a description about yourself.")
            return
        if not url_input.strip():
            st.error("Please enter a job post URL.")
            return
        
        try:
            with st.spinner("Generating your cold email..."):
                st.session_state['user_name'] = user_name
                st.session_state['user_about'] = user_about
                st.session_state['url_input'] = url_input
                st.session_state['skills_input'] = skills_input
                st.session_state['links_input'] = links_input

                skills_list = [skill.strip() for skill in skills_input.split(",") if skill.strip()]
                links_list = [link.strip() for link in links_input.split(",") if link.strip()]

                portfolio.add_to_portfolio(skills_list, links_list)

                loader = WebBaseLoader([url_input])
                data = clean_text(loader.load().pop().page_content)
                jobs = llm.extract_jobs(data)

                for job in jobs:
                    job_skills = job.get('skills', [])
                    links = portfolio.query_links(job_skills)
                    
                    email = llm.write_mail(job, links, user_name, user_about)
                    st.session_state['draft_email'] = email
                    st.session_state['current_job'] = job  # Store current job for email extraction
                    break  # Process first job for now

                st.success("✅ Cold email generated successfully!")

        except Exception as e:
            st.error(f"An Error Occurred: {e}")

    # Email editing and preview section
    if 'draft_email' in st.session_state:
        st.subheader("✏️ Edit Your Email")
        st.text_area("Edit Email", value=st.session_state['draft_email'], key='edit_email', height=300)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Draft", type="secondary"):
                st.session_state['final_email'] = st.session_state['edit_email']
                st.success("Draft saved successfully!")

        # Email sending section
        st.subheader("📤 Send Email")
        
        # Auto-extract recipient email from job description if available
        suggested_recipient = ""
        if 'current_job' in st.session_state:
            suggested_recipient = extract_recipient_email_from_job_description(st.session_state['current_job'])
        
        col1, col2 = st.columns(2)
        with col1:
            sender_email = st.text_input("Your Gmail Address:", help="Your Gmail address to send from")
            sender_password = st.text_input("Gmail App Password:", type="password", 
                                          help="Generate an App Password from your Google Account settings")
        
        with col2:
            recipient_email = st.text_input("Recipient Email:", value=suggested_recipient, 
                                          help="Email address to send the cold email to")
            email_subject = st.text_input("Email Subject:", 
                                        value=f"Application for {st.session_state.get('current_job', {}).get('role', 'Position')}")
        
        # Info box about Gmail App Password
        with st.expander("ℹ️ How to get Gmail App Password"):
            st.markdown("""
            **Steps to generate Gmail App Password:**
            1. Go to your Google Account settings
            2. Select "Security" from the left panel
            3. Under "Signing in to Google," select "2-Step Verification" (must be enabled)
            4. At the bottom, select "App passwords"
            5. Select "Mail" and your device
            6. Copy the generated 16-character password
            
            **Note:** Regular Gmail password won't work. You must use an App Password for security.
            """)
        
        # Send email button
        if st.button("📧 Send Email", type="primary", help="Send the cold email"):
            # Validation
            if not sender_email or not validate_email(sender_email):
                st.error("Please enter a valid sender email address.")
            elif not sender_password:
                st.error("Please enter your Gmail App Password.")
            elif not recipient_email or not validate_email(recipient_email):
                st.error("Please enter a valid recipient email address.")
            elif not email_subject:
                st.error("Please enter an email subject.")
            elif 'edit_email' not in st.session_state or not st.session_state['edit_email']:
                st.error("No email content to send. Please generate an email first.")
            else:
                with st.spinner("Sending email..."):
                    success, message = send_email(
                        sender_email=sender_email,
                        sender_password=sender_password,
                        recipient_email=recipient_email,
                        subject=email_subject,
                        body=st.session_state['edit_email'],
                        sender_name=user_name or "Job Applicant"
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")

        # Final email display section
        if 'final_email' in st.session_state:
            st.subheader("📧 Final Email")
            st.markdown(f"<div class='code-block'><pre>{st.session_state['final_email']}</pre></div>", unsafe_allow_html=True)

            # Add a copy button using Streamlit built-in feature
            st.code(st.session_state['final_email'], language='text')

            # Download button for the email
            st.download_button(
                label="📥 Download Email", 
                data=st.session_state['final_email'], 
                file_name="cold_email.txt", 
                mime="text/plain",
                type="primary"
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f4e79;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #666;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .code-block {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)
