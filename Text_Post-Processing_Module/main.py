import os
from modules.image_preprocessing import preprocess_image
from modules.text_recognition import recognize_text
from modules.post_processing import clean_and_normalize_text
from modules.domain_rules import apply_domain_rules

def main():
    # Input and output paths
    input_image_path = "data/input_image2.png"
    processed_image_path = "output/processed_image.png"
    raw_text_path = "output/extracted_text.txt"
    cleaned_text_path = "output/cleaned_text.json"
    
    # Step 1: Preprocess the image
    processed_image = preprocess_image(input_image_path, processed_image_path)
    print(f"Preprocessed image saved at: {processed_image}")
    
    # Step 2: Recognize text from the preprocessed image
    raw_text = recognize_text(processed_image)
    print("\nRaw OCR Text:\n", raw_text)
    with open(raw_text_path, 'w') as f:
        f.write(raw_text)
    print(f"Raw OCR text saved at: {raw_text_path}")
    
    # Step 3: Clean and normalize the text
    normalized = clean_and_normalize_text(raw_text)
    print("\nCleaned Text:\n", normalized["cleaned_text"])
    
    # Step 4: Apply domain-specific rules
    final_text = apply_domain_rules(normalized["cleaned_text"])
    print("\nFinal Text after Domain Rules:\n", final_text)
    
    # Save normalized text as JSON
    import json
    with open(cleaned_text_path, 'w') as f:
        json.dump({"cleaned_text": final_text, "structured_data": normalized["structured_data"]}, f, indent=4)
    print(f"Cleaned and structured text saved to {cleaned_text_path}")

if __name__ == "__main__":
    main()
