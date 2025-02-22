import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

# Load template
template_path = "C:/Users//Desktop//sample_certificate.pdf" ##Path of the template certificate
output_path = "C:/Users/Arnab/OneDrive/Desktop/" ##Path to output

os.makedirs(output_path, exist_ok=True)

# Function to generate certificates
def generate_certificate(student_name, output_filename):
    doc = fitz.open(template_path)
    page = doc[0]  # Get first page

    # Define text position and font
    text_x, text_y = 250, 290  # Adjust based on layout
    font_size = 40

    # Add text directly onto the certificate
    page.insert_text((text_x, text_y), student_name, fontsize=font_size, fontname="helv", color=(0, 0, 0))

    # Save the generated certificate
    save_path = os.path.join(output_path, output_filename)
    doc.save(save_path)
    print(f"Certificate saved: {save_path}")

# Example Usage
students = ["Rajkumar Ghosh", "Arnab", "Ayon iqoo"]
for student in students:
    generate_certificate(student, f"{student.replace(' ', '_')}.pdf")