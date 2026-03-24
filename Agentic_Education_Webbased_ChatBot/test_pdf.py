#!/usr/bin/env python
"""Test PDF generation functionality"""

from fpdf import FPDF
import warnings

# Test the PDF generation with suppressed warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    # Create a simple PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, txt="Test Worksheet", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 10, txt="QUESTIONS", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    questions = ["Question 1: What is Python?", "Question 2: What is Flask?"]
    for q in questions:
        pdf.multi_cell(190, 10, txt=q)
    
    # Get the PDF bytes
    pdf_bytes = pdf.output(dest='S')
    print(f"✅ PDF generated successfully! Size: {len(pdf_bytes)} bytes")
    print("✅ Worksheet PDF test PASSED!")
    
    # Save test PDF
    with open('test_output.pdf', 'wb') as f:
        f.write(pdf_bytes)
    print("✅ Test PDF saved to test_output.pdf")
