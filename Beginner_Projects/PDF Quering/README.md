# Query with multiple PDFs

### **Goal**
Developed a model that can take multiple PDFs as an input and then the user can ask different questions to the uploaded PDFs!!

### **Description**
Users can upload multiple PDFs (200 mb per file) and can ask questions related to the PDFs. Created an interactive UI for the same using streamlit and deployed on the streamlit community.

### **What I Have Done!**
1. Upload multiple PDFs.
2. Ask questions related to the pdf.
3. Get suitable answer.
4. Summarizes the pdf.

### **Libraries Needed**
- `streamlit`
- `textwrap`

### **Usage**
Run the server and client scripts using Python:

1. Clone the repo:
```bash
  git clone https://github.com/Yash-Bhatnagar-02
  Chat-with-multiple-PDFs.git
```

2. Install requirements in the terminal:
```bash
  pip install -r requirements
```

3. Create an environment variable (.env) in the root directory of
your project folder and add your Google API key:
```bash
  GOOGLE_API_KEY=""
```

4. Run the application with the following command:
```bash
  streamlit run app.py
```


### How to Use:
- Upon starting, upload the PDF(s).
- Type any question related to the PDF.
- A response will be generated containing the suitable answer.
