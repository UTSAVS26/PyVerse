## **EDUHELPER: Chat with PDF Files**

### 🎯 **Goal**
The main goal of EDUHELPER is to provide an interactive platform where users can upload PDF files and ask questions based on the content of those files. The application utilizes Google Generative AI API for efficient text embedding and accurate question answering, making it useful for quickly retrieving information from large documents.

### 🧵 **Dataset**
The dataset consists of user-uploaded PDF files. No specific dataset is preloaded since the application processes custom files.

### 🧾 **Description**
EDUHELPER is a Streamlit application that allows users to upload one or more PDF files and interact with the content by asking questions. The application processes the uploaded files, extracts text, and uses a conversational AI model to answer user queries based on the content. It’s ideal for educational and research purposes, enabling users to efficiently search for information in documents without manually going through the entire text.

### 🧮 **What I had done!**
1. Set up a Streamlit interface to upload PDF files.
2. Created a function (`get_pdf_text`) to extract text from the uploaded PDF files using PyPDF2.
3. Split the extracted text into smaller chunks (`get_text_chunks`) for efficient retrieval.
4. Created a vector store (`get_vector_store`) using FAISS indexing and Google Generative AI embeddings to store the text chunks.
5. Set up a conversational chain (`get_conversational_chain`) to process questions and provide answers using Google's Generative AI API.
6. Developed the user interaction flow, where users can enter questions to receive relevant answers from the uploaded content.

### 🚀 **Models Implemented**
- **FAISS Indexing with Google Generative AI API**: FAISS indexing was chosen to efficiently retrieve relevant chunks from large documents. Google's Generative AI API was used for both embedding the text and generating responses. The combination of FAISS and Generative AI provides fast and accurate results.

### 📚 **Libraries Needed**
- **Streamlit**: To build an interactive user interface.
- **PyPDF2**: To extract text from PDF files.
- **LangChain**: To implement conversational flows using language models.
- **Google Generative AI**: To perform text embedding and question answering.
- **FAISS**: To perform efficient similarity searches in text chunks.
- **python-dotenv**: To securely manage the Google API key.

### 📊 **Exploratory Data Analysis Results**
- **(No EDA Performed)**: As this project is not focused on data analysis but rather on interaction with textual content from PDFs, there are no visualizations included.

### 📈 **Performance of the Models based on the Accuracy Scores**
- **Google Generative AI (QA Performance)**: Since this project utilizes a pre-trained AI model, accuracy evaluation was based on the quality of responses to test queries. The model performed well, providing contextually accurate and relevant answers to the questions based on the content of the PDFs.

### 📢 **Conclusion**
EDUHELPER offers an effective solution for interacting with PDF content, saving users the time and effort of manually searching for information. The FAISS index allows for efficient text retrieval, and Google's Generative AI API ensures the answers are contextually appropriate. The setup enables users to ask questions and receive accurate responses based on the content, making the platform especially useful for students, researchers, and professionals.

### ✒️ **Your Signature**

**Aritro Saha**  
[GitHub](https://github.com/halcyon-past) | [LinkedIn](https://www.linkedin.com/in/aritro-saha/) | [Portfolio](https://aritro.tech/)