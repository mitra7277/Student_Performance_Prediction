import os
from fpdf import FPDF
import zipfile

# -----------------------------
# Generate PDF for one student
# -----------------------------
def generate_pdf(student_data, predicted_score, pdf_file):
    pdf = FPDF()
    pdf.add_page()
    
    # Title (no emojis to avoid Unicode errors)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Student Performance Report", ln=True, align="C")
    pdf.ln(10)

    # Student details
    pdf.set_font("Arial", '', 12)
    for key, value in student_data.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Predicted Score: {predicted_score:.2f}/100", ln=True, align="L")

    # Save PDF file
    pdf.output(pdf_file)


# -----------------------------
# Zip multiple student PDFs
# -----------------------------
def zip_pdfs(pdf_list, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in pdf_list:
            if os.path.exists(file):
                zipf.write(file, os.path.basename(file))
    return zip_filename
