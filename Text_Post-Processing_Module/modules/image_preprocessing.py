import cv2
import numpy as np

def preprocess_image(image_path, output_path="output/processed_image.png"):
    """
    Preprocess the image to enhance OCR recognition for handwritten cursive text.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the preprocessed image.
    
    Returns:
        str: Path to the preprocessed image.
    """
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Use adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
    )
    
    # Morphological operations to clean noise and join broken characters
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Save and return the processed image path
    cv2.imwrite(output_path, cleaned)
    return output_path
