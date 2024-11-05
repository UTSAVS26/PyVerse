from flask import Flask, jsonify, request
from flask_cors import CORS
import config
import pymupdf
import google.generativeai as genai

genai.configure(api_key=config.API_KEY)

generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction='''
                        Analyze the text of the uploaded product manual thoroughly and 
                        provide accurate, detailed responses to any questions 
                        asked about its content. If questions regarding previous conversations are 
                        asked, answer them accordingly. The responses should be directly 
                        related to the information found within the manual or the 
                        information previously provided by the model and not 
                        include extraneous data from other sources. If necessary, 
                        refer to specific sections, tables, or figures in the manual
                        for clarity. You should also offer step-by-step 
                        instructions when asked about operational processes or
                        troubleshooting issues. Ensure all responses are concise, 
                        easy to understand, and directly relevant to the product 
                        being discussed.
                    '''
)

app = Flask(__name__)
CORS(app)
chat_history = []

@app.route('/generate', methods=['POST'])
def generate_response():
    global chat_history
    
    try:
        chat_session = model.start_chat(history=chat_history)

        if 'fileUploaded' not in request.files:
            return jsonify({'error': 'fileUploaded field is missing'}), 400

        file = request.files["fileUploaded"]
        input_prompt = request.form["fileDoubt"]

        pdf_text = extract_text(file)   

        chat_history.append({
            "role": "user",
            "parts": [input_prompt]
        })

        try:
            response = chat_session.send_message([input_prompt, pdf_text])
        except Exception as e:
            return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500

        chat_history.append({
            "role": "model",
            "parts": [response.text]
        })

        json_response = {
            "prompt": input_prompt,
            "response": response.text,
        }

        return jsonify(json_response)

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


def extract_text(file):
    try:
        pdf = pymupdf.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
        return text
    except Exception as e:
        raise Exception(f'Error extracting text: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
