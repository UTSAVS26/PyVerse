
import re
import spacy
import json
from extractor import extract_text_from_pdf

nlp = spacy.load("en_core_web_sm")

def extract_email(text):
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w{2,}\b", text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r"(\+?\d{1,3}[\s-]?)?(\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}", text)
    return match.group(0) if match else None

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_skills(text):
    skills_list = ['Python', 'Java', 'SQL', 'C++', 'HTML', 'CSS', 'JavaScript', 'Machine Learning', 'React']
    found_skills = [skill for skill in skills_list if skill.lower() in text.lower()]
    return list(set(found_skills))

def scan_resume(file_path):
    text = extract_text_from_pdf(file_path)
    data = {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "Skills": extract_skills(text),
    }
    return data

if __name__ == "__main__":
    result = scan_resume("data/sample_resume.pdf")
    with open("output/extracted_data.json", "w") as f:
        json.dump(result, f, indent=4)
    print("Extraction complete! Check output/extracted_data.json")
