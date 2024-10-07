# **NaviBot-Voice-Assistant**

### 🎯 **Goal**

The main goal of NaviBot-Voice-Assistant is to assist visually impaired users by providing a voice-based interaction system that helps them navigate complex routes. The assistant uses voice commands and PDF-based route data to offer step-by-step guidance. It enhances accessibility by combining voice recognition, text-to-speech, and advanced AI-powered query systems.

### 🧵 **Dataset**

NaviBot does not require any pre-existing dataset. Instead, it utilizes user-uploaded PDF route data for navigation assistance. However, if you're working with specific datasets for training or additional use cases, please link them here.

### 🧾 **Description**

NaviBot-Voice-Assistant allows users to upload PDF route data, process it into text chunks, and query it using either text or voice inputs. It uses Google Generative AI for question answering, FAISS for vector storage, and Langchain to build conversational AI. The assistant is designed for seamless interaction and voice guidance, making navigation more accessible for blind users.

### 🧮 **What I had done!**

- Integrated **speech recognition** using `speech_recognition` to enable voice inputs.
- Added **text-to-speech** functionality using `pyttsx3` for voice-based responses.
- Allowed users to **upload PDF documents** to extract and process route data.
- Utilized **Google Generative AI embeddings** to transform text data into vectors.
- Implemented **FAISS** for efficient similarity search of navigation instructions.
- Built a conversational chain for answering questions based on the processed text.
- Deployed the application using **Streamlit** to create a user-friendly web interface.

### 🚀 **Models Implemented**

- **Google Generative AI for Embeddings**: Chosen for its advanced natural language understanding and high accuracy in generating meaningful embeddings.
- **FAISS (Facebook AI Similarity Search)**: Selected for its high-performance similarity search on large-scale vectors, allowing for efficient retrieval of relevant route instructions.

### 📚 **Libraries Needed**

- `streamlit`
- `pyttsx3`
- `speech_recognition`
- `PyPDF2`
- `langchain`
- `langchain_google_genai`
- `google.generativeai`
- `langchain_community.vectorstores`
- `FAISS`
- `dotenv`

### 📊 **Exploratory Data Analysis Results**

This project does not involve a traditional EDA, as it focuses on real-time PDF processing and AI-powered voice interaction. However, if visualizations or processing statistics (e.g., text chunk sizes, similarity scores) are generated, they can be displayed here.

### 📈 **Performance of the Models based on the Accuracy Scores**

The performance of the system can be evaluated based on:
- **Response accuracy**: How well the system provides relevant answers from the context.
- **Speech recognition accuracy**: How accurately it converts voice input into text.
- **Embedding quality**: How well the AI model embeds text for similarity search.

### 💻 How to run

To get started with NaviBot-Voice-Assistant, follow these steps:

1. Navigate to the project directory:

    ```bash
    cd NaviBot-Voice-Assistant
    ```

3. Activating venv (optional) 

    ```bash
    conda create -n venv python=3.10+
    conda activate venv
    ```

4. Install dependencies:

    ```python
    pip install -r requirements.txt
    ```

5. Configure environment variables
    ```
    Rename `.env-sample` to `.env` file
    Replace the API your Google API Key, 
    ```
    Kindly follow refer to this site for getting [your own key](https://ai.google.dev/tutorials/setup)
    <br/>

6. Run the chatbot:

    ```bash
    streamlit run app.py
    ```

    PS: Try running other files as well for different use case. 

### 📢 **Conclusion**

NaviBot-Voice-Assistant successfully integrates voice recognition and AI-powered question-answering to assist visually impaired users in navigating routes. It ensures high interaction accuracy by leveraging state-of-the-art models like Google Generative AI and FAISS for efficient retrieval and processing. The integration of these technologies provides a reliable and accessible navigation experience for its users.

### ✒️ **Signature**

**[J B Mugundh]**  
GitHub: [Github](https://github.com/J-B-Mugundh)  
LinkedIn: [Linkedin](https://www.linkedin.com/in/mugundhjb/)

