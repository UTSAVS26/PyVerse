# Result Comparison: Aadhaar Information Extraction

This document showcases the results of extracting Aadhaar card information using two different approaches:

1. **Tesseract OCR with Image Pre-processing**
2. **EasyOCR with Multi-language Support (English and Hindi)**

Screenshots are provided for the extracted information, along with an API result screenshot.

---

## 1. Tesseract OCR Approach

In this approach, the Aadhaar card image is first converted to greyscale and then passed through the Tesseract OCR engine. Regular expressions (`re`) are used to extract key information such as names, gender, date of birth, and Aadhaar number from the extracted text.

### Screenshot for Tesseract OCR Result:
![Tesseract OCR Result](assets/images/tesseract.png)

#### Challenges:
- **Accuracy**: Tesseract struggles with mixed-language documents (English + Hindi).
- **Pre-processing Required**: The image needs to be pre-processed (converted to greyscale) to improve text extraction.
- **Hindi Text**: Tesseract doesn't handle Hindi text as well, which reduces its accuracy for Aadhaar cards that include Hindi.

---

## 2. EasyOCR Approach

The EasyOCR approach uses multi-language support for both Hindi and English, making it a better fit for Aadhaar card text recognition. The extracted text is processed using regular expressions to find relevant details.

### Output for EasyOCR:
- **First Name**: `Rahul`
- **Middle Name**: `Ramesh`
- **Last Name**: `Gaikwad`
- **Gender**: `Male`
- **DOB**: `23/08/1995`
- **Aadhaar Number**: `2058 6470 5393`

### Screenshot for EasyOCR Result:
![EasyOCR Result](assets/images/easyocr.png)

### After Extraction
![EasyOCR Result](assets/images/Output.png)

#### Advantages:
- **Higher Accuracy**: EasyOCR performs significantly better with mixed-language documents, making it ideal for Aadhaar cards.
- **Multi-language Support**: Supports both English and Hindi, improving text extraction accuracy.
- **No Heavy Pre-processing**: Works well without needing extensive image manipulation.

---

## Comparison of Results

| Feature              | Tesseract OCR                      | EasyOCR                          |
|----------------------|------------------------------------|----------------------------------|
| **Languages**         | English only                      | English and Hindi support        |
| **Accuracy**          | Low to Medium                     | High                             |
| **Pre-processing**    | Requires greyscale conversion     | Minimal pre-processing needed    |
| **Performance**       | Faster but less accurate          | Bit slower and more accurate     |
| **Aadhaar Extraction**| Struggles with Hindi and complex fonts | Handles both languages well       |

---

## API Result Screenshot

Here is the expected result returned from the API after extracting information from the Aadhaar card image:

![EasyOCR Result](assets/images/api_response.png)

### Input  Body JSON: 
{

    "image_path": "C:\\Users\\mayan\\Downloads\\fs.jpeg"

}

### API Response:
```json
{
  "First Name": "Rahul",
  "Middle Name": "Ramesh",
  "Last Name": "Gaikwad",
  "Gender": "Male",
  "DOB": "23/08/1995",
  "Aadhaar Number": "2058 6470 5393"
}
