import easyocr

def recognize_text(image_path):
    """
    Recognize handwritten text from an image using EasyOCR.
    
    Args:
        image_path (str): Path to the preprocessed image.
    
    Returns:
        str: Recognized raw text from the image.
    """
    reader = easyocr.Reader(['en'])  # Initialize OCR with English language
    results = reader.readtext(image_path, detail=0, paragraph=True)  # Combine text into paragraphs
    return " ".join(results)
