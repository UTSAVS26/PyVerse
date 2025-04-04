import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is missing. Please add it to your .env file.")
genai.configure(api_key=api_key)


##Function that reads the pdd goes through each and every page
def get_pdf_text(pdf_docs):
    if not pdf_docs:
        return ""
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
        # Optionally wrap the above block in try-except if needed:
        # try:
        #     pdf_reader = PdfReader(pdf)
        #     for page in pdf_reader.pages:
        #         text += page.extract_text()
        # except Exception as e:
        #     st.error(f"Error reading PDF '{pdf.name}': {str(e)}")
    return  text


##Function that breaks text into chunks
def get_text_chunks(text):
    # Adjust chunk size and overlap as needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


##Function that saves the data we got from conversation to local(here), generally stored in data base
#converting into chunks -> storing data in faiss vector
def get_vector_store(text_chunks, embedding_model="models/embedding-001", store_dir="faiss_index"):
    """Create and save vector store from text chunks.
    
    Args:
        text_chunks: List of text chunks to embed
        embedding_model: Name of the embedding model to use
        store_dir: Directory to save the vector store
    
    Returns:
        None
    """
    if not text_chunks:
        st.warning("No text to process. Please check the PDF content.")
        return
        
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Ensure the directory exists
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        
        # Save the vector store index in the directory
        vector_store.save_local(store_dir)
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")


#Function to give the prompt and ask the bot to act accordingly, giving the gemini model
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


##Function for user input, question
#give the question-> do a similarity search on all the faiss vectors-> go with converstional chain-> response from chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if the faiss_index file exists before loading
    if not os.path.exists("faiss_index/index.faiss"):
        st.error("FAISS index file not found. Please process the PDF files first.")
        return
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


##Main Function for streamlit app
def main():
    st.set_page_config("PDF Wizard")
    st.header("Chat with multiple PDFs📄")

    user_question = st.text_input("📎Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()