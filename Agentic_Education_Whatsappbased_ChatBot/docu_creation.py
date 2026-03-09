# create_test_documents.py
from fpdf import FPDF
import os

def create_sample_pdf(filename, title, content):
    """Create a sample PDF document"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=title, ln=1, align='C')
    pdf.ln(10)
    
    # Content
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        pdf.multi_cell(0, 10, txt=line)
    
    os.makedirs('accuracy_evaluation/test_data/test_documents', exist_ok=True)
    filepath = f"accuracy_evaluation/test_data/test_documents/{filename}"
    pdf.output(filepath)
    print(f"Created: {filepath}")

def create_test_documents():
    """Create various test documents"""
    
    # Physics document
    create_sample_pdf(
        "physics_chapter.pdf",
        "Introduction to Physics",
        """Newton's Laws of Motion

First Law: An object at rest stays at rest, and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.

Second Law: The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass.

Third Law: For every action, there is an equal and opposite reaction.

Key Concepts:
- Force: A push or pull upon an object
- Mass: The amount of matter in an object  
- Acceleration: Rate of change of velocity
"""
    )
    
    # Math document
    create_sample_pdf(
        "math_worksheet.pdf", 
        "Algebra Practice Problems",
        """Solve the following equations:

1. 2x + 5 = 15
2. 3(x - 4) = 21
3. x² + 5x + 6 = 0
4. 2x - 7 = 3x + 1

Word Problems:
1. The sum of two numbers is 25. One number is 5 more than the other. Find the numbers.
2. A rectangle has a length that is 3 times its width. If the perimeter is 48 cm, find the dimensions.
"""
    )
    
    # Biology document
    create_sample_pdf(
        "biology_notes.pdf",
        "Cell Biology Basics",
        """The Cell Theory:
1. All living organisms are composed of cells
2. The cell is the basic unit of life
3. All cells come from pre-existing cells

Cell Types:
- Prokaryotic: Simple cells without nucleus (bacteria)
- Eukaryotic: Complex cells with nucleus (plants, animals)

Cell Organelles:
- Nucleus: Contains genetic material
- Mitochondria: Powerhouse of the cell
- Ribosomes: Protein synthesis
- Endoplasmic Reticulum: Transport system
"""
    )

if __name__ == "__main__":
    create_test_documents()