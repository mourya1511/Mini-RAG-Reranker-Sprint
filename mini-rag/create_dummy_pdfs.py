from pathlib import Path
from fpdf import FPDF

# Folder to save PDFs
pdf_dir = Path("pdfs")
pdf_dir.mkdir(exist_ok=True)

# Dummy PDF content
pdf_contents = {
    "Industrial_Safety_Basics.pdf": "Industrial Safety Basics:\n\n- Always wear PPE.\n- Follow safety signs.\n- Report hazards immediately.",
    "PPE_Guidelines.pdf": "Personal Protective Equipment (PPE) Guidelines:\n\n- Helmets must be worn in construction areas.\n- Gloves should be used when handling chemicals.\n- Safety goggles protect eyes from debris.",
    "Emergency_Procedures.pdf": "Emergency Procedures:\n\n- Know the fire exits.\n- Follow evacuation plans.\n- Call emergency contacts in case of accidents."
}

# Create PDFs
for filename, text in pdf_contents.items():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(pdf_dir / filename)

print("✅ 3 dummy PDFs created in the 'pdfs' folder.")
