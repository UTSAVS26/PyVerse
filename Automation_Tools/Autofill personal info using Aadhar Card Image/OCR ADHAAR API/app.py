from flask import Flask, request, jsonify
import easyocr
import re

app = Flask(__name__)
reader = easyocr.Reader(['en', 'hi'])  # Load EasyOCR with English and Hindi support

def extract_info(ocr_result):
    first_name, middle_name, last_name, gender, dob, year_of_birth, aadhaar_number = None, None, None, None, None, None, None
    
    for item in ocr_result:
        text = item[1]

        # Check for gender and extract names
        if re.search(r'Male|Female|पुरुष|महिला', text):
            name_match = re.findall(r'[A-Za-z]+', text)
            if len(name_match) >= 3:
                first_name, middle_name, last_name = name_match[:3]
            gender = 'Male' if 'Male' in text or 'पुरुष' in text else 'Female'
            
            # Extract DOB or Year of Birth
            dob_match = re.search(r'\b(\d{2}/\d{2}/\d{4})\b', text)
            if dob_match:
                dob = dob_match.group(1)
            elif 'Year of Birth' in text or 'जन्म वर्ष' in text:
                yob_match = re.search(r'Year of Birth\s*:\s*([\d]+)', text)
                year_of_birth = yob_match.group(1) if yob_match else None
        
        # Extract Aadhaar number
        aadhaar_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', text)
        if aadhaar_match:
            aadhaar_number = aadhaar_match.group(0)

    return {
        "First Name": first_name,
        "Middle Name": middle_name,
        "Last Name": last_name,
        "Gender": gender,
        "DOB": dob,
        "Year of Birth": year_of_birth,
        "Aadhaar Number": aadhaar_number
    }

@app.route('/extract', methods=['POST'])
def extract_data():
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path:
        return jsonify({"error": "Image path is required"}), 400

    # Process the image with EasyOCR
    ocr_result = reader.readtext(image_path, paragraph=True)
    extracted_info = extract_info(ocr_result)

    return jsonify(extracted_info)

if __name__ == '__main__':
    app.run(debug=True)
