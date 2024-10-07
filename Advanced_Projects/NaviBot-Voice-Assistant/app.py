import streamlit as st
import pyttsx3
import speech_recognition as sr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Retrieve Google API key from environment variables
os.getenv("GOOGLE_API_KEY")
# Configure Generative AI API using the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize pyttsx3 for voice output (Text-to-Speech engine)
engine = pyttsx3.init()

# Function to speak the given text using pyttsx3
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to listen for voice input using the microphone
def listen():
    r = sr.Recognizer()
    # Use the microphone as the audio source
    with sr.Microphone() as source:
        st.write("Listening...")  # Notify the user that it's listening
        r.adjust_for_ambient_noise(source)  # Adjust for noise in the environment
        audio = r.listen(source)  # Capture the audio

    try:
        # Use Google Speech Recognition to convert audio to text
        user_input = r.recognize_google(audio)
        st.write(f"You said: {user_input}")
        return user_input
    except sr.UnknownValueError:
        # Handle case where speech is not understood
        st.write("Sorry, I could not understand what you said.")
        return None
    except sr.RequestError as e:
        # Handle request errors with Google Speech Recognition API
        st.write(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Read PDF and extract text from each page
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split large text into smaller chunks using RecursiveCharacterTextSplitter
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)  # Split the text into manageable chunks
    return chunks

# Function to create and save a FAISS vector store from the text chunks
def get_vector_store(text_chunks):
    # Use Google Generative AI embeddings to convert text into vectors
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create a FAISS vector store from the text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save the vector store locally for later use
    vector_store.save_local("faiss_index")

# Function to get the conversational chain for question-answering using a template
def get_conversational_chain():

    # Define a prompt template for generating detailed answers
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize the Google Generative AI model (gemini-pro) with a specific temperature setting
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Set up the prompt template with input variables for context and question
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load the question-answering chain with the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle the user input (both text and voice), perform similarity search, and respond
def user_input(user_question):
    # Initialize Google Generative AI embeddings for vector search
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the local FAISS index with embeddings, allow for dangerous deserialization
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Perform similarity search in the vector store based on the user's question
    docs = new_db.similarity_search(user_question)

    # Get the question-answering conversational chain
    chain = get_conversational_chain()

    # Run the chain to get the answer from the retrieved documents
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Use pyttsx3 to speak the response
    speak(response["output_text"])
    # Display the response on the Streamlit UI
    st.write("Reply: ", response["output_text"])

# Main function for the Streamlit app
def main():
    # Set the page configuration with a title
    st.set_page_config("Beyond GPS Navigation")
    st.header("Beyond GPS Navigator for Blind")  # App header

    # Get user's text input from Streamlit input box
    user_question = st.text_input("Ask your query")
    # Button to trigger voice input
    voice_input_button = st.button("Voice Input")

    # If the voice input button is clicked, listen to the user's voice
    if voice_input_button:
        user_question = listen()  # Capture the user's voice input
        if user_question:
            user_input(user_question)  # Process the input

    # If there's text input from the user, process it
    if user_question:
        user_input(user_question)

    # Sidebar menu for uploading PDF documents
    with st.sidebar:
        st.title("Menu:")
        # Allow users to upload multiple PDF files
        pdf_docs = st.file_uploader("Upload your route data and Click on the Submit & Process Button", accept_multiple_files=True)
        # Button to process the uploaded PDF documents
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):  # Show spinner while processing
                raw_text = get_pdf_text(pdf_docs)  # Extract text from the PDF files
                text_chunks = get_text_chunks(raw_text)  # Split the text into chunks
                get_vector_store(text_chunks)  # Save the text chunks in a vector store
                st.success("Done")  # Notify the user when processing is complete

# Run the Streamlit app
if __name__ == "__main__":
    main()
