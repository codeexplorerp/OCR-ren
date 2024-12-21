import re
from spellchecker import SpellChecker

def clean_and_normalize_text(raw_text):
    """
    Cleans, normalizes, and structures the OCR output text.
    
    Args:
        raw_text (str): Unprocessed text from OCR output.
    
    Returns:
        dict: Contains cleaned and optionally structured text.
    """
    # Step 1: Remove unwanted spaces, symbols, and line breaks
    cleaned_text = re.sub(r'\s+', ' ', raw_text)  # Replace multiple spaces/newlines with a single space
    cleaned_text = re.sub(r'[^\w\s.,!?-]', '', cleaned_text)  # Allow only basic punctuation
    
    # Step 2: Fix common errors (enhanced for cursive styles)
    corrections = {
        '0': 'O', '1': 'I', 'l': 'I', 'rn': 'm', 'vv': 'w', '11': 'll'
    }
    for key, value in corrections.items():
        cleaned_text = cleaned_text.replace(key, value)
    
    # Step 3: Normalize sentence capitalization and punctuation
    cleaned_text = re.sub(r'\s([.,!?])', r'\1', cleaned_text)  # Remove space before punctuation
    cleaned_text = re.sub(r'([.!?])\s*(\w)', lambda m: f"{m.group(1)} {m.group(2).capitalize()}", cleaned_text)  # Capitalize sentences
    
    # Step 4: Spell correction
    spell = SpellChecker()
    words = cleaned_text.split()
    corrected_text = ' '.join([spell.correction(word) if spell.correction(word) else word for word in words])
    
    # Step 5: Structure the text into paragraphs
    structured_data = {"paragraphs": corrected_text.split('.')}
    
    return {
        "cleaned_text": corrected_text.strip(),
        "structured_data": structured_data
    }
