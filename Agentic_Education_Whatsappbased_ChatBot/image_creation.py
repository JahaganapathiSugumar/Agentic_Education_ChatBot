# create_test_images.py
from PIL import Image, ImageDraw, ImageFont
import os

def create_text_image(filename, text, size=(800, 600)):
    """Create an image with educational text"""
    # Create a white background
    img = Image.new('RGB', size, color='white')
    d = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Simple text wrapping
    lines = []
    words = text.split()
    current_line = ""
    
    for word in words:
        test_line = f"{current_line} {word}".strip()
        # Simple width check (approximate)
        if len(test_line) < 60:  # Character limit
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    
    # Draw text
    y_position = 50
    for line in lines:
        d.text((50, y_position), line, font=font, fill='black')
        y_position += 40
    
    os.makedirs('accuracy_evaluation/test_data/test_images', exist_ok=True)
    filepath = f"accuracy_evaluation/test_data/test_images/{filename}"
    img.save(filepath)
    print(f"Created: {filepath}")

def create_test_images():
    """Create various test images"""
    
    # Textbook page image
    create_text_image(
        "textbook_page.jpg",
        "Introduction to Computer Science\n\nComputer science is the study of computers and computational systems. Unlike electrical and computer engineers, computer scientists deal mostly with software and software systems."
    )
    
    # Diagram image
    create_text_image(
        "diagram.png", 
        "Photosynthesis Process\n\nLight Energy + Carbon Dioxide + Water → Glucose + Oxygen\n\nThis process occurs in the chloroplasts of plant cells and is essential for life on Earth."
    )
    
    # Handwritten notes image
    create_text_image(
        "handwritten_notes.jpg",
        "Class Notes - History\n\nKey Events:\n- 1776: American Declaration of Independence\n- 1789: French Revolution begins\n- 1914: World War I starts\n- 1945: World War II ends"
    )

if __name__ == "__main__":
    create_test_images()