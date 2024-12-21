import numpy as np
import cv2
from pyzbar.pyzbar import decode
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass

@dataclass
class BarcodeResult:
    """Data class to store barcode detection results"""
    data: str
    bbox: Tuple[int, int, int, int]
    type: str

class BarcodeDetector:
    """Class for detecting and decoding barcodes in images"""
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize the barcode detector
        
        Args:
            min_confidence: Minimum confidence threshold for detection
        """
        self.min_confidence = min_confidence
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better barcode detection
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image
        """
        # Resize image if too large
        max_size = 1000
        height, width = image.shape
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Apply different preprocessing techniques
        preprocessed_images = []
        
        # 1. Original image
        preprocessed_images.append(image)
        
        # 2. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        preprocessed_images.append(adaptive)
        
        # 3. Otsu's thresholding
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(otsu)
        
        # 4. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_contrast = clahe.apply(image)
        preprocessed_images.append(enhanced_contrast)
        
        return preprocessed_images
    
    def get_barcode_bbox(self, barcode) -> Tuple[int, int, int, int]:
        """
        Extract bounding box coordinates from barcode object
        
        Args:
            barcode: Decoded barcode object from pyzbar
            
        Returns:
            Tuple of (x1, y1, x2, y2) coordinates
        """
        points = barcode.polygon
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        
        return (x1, y1, x2, y2)
    
    def detect_and_decode(self, 
                         image: np.ndarray,
                         debug: bool = False) -> Optional[BarcodeResult]:
        """
        Detect and decode barcode from input image
        
        Args:
            image: Preprocessed grayscale image as numpy array
            debug: If True, show debug information
            
        Returns:
            BarcodeResult object containing decoded data and bbox coordinates,
            or None if no barcode detected
        """
        # Verify input image
        if len(image.shape) > 2:
            raise ValueError("Input image must be grayscale")
            
        # Try multiple preprocessing techniques
        preprocessed_images = self.enhance_image(image)
        
        for idx, enhanced_image in enumerate(preprocessed_images):
            # Detect and decode barcodes
            barcodes = decode(enhanced_image)
            
            if debug:
                cv2.imshow(f"Debug: Preprocessing {idx}", enhanced_image)
                cv2.waitKey(100)  # Show each preprocessing step briefly
            
            if barcodes:
                if debug:
                    cv2.destroyAllWindows()
                
                # Process first detected barcode
                barcode = barcodes[0]
                
                # Extract data and coordinates
                bbox = self.get_barcode_bbox(barcode)
                try:
                    data = barcode.data.decode('utf-8')
                except UnicodeDecodeError:
                    data = barcode.data.decode('utf-8', errors='ignore')
                
                return BarcodeResult(
                    data=data,
                    bbox=bbox,
                    type=barcode.type
                )
        
        if debug:
            cv2.destroyAllWindows()
        
        return None
    
    def draw_bbox(self, 
                  image: np.ndarray,
                  bbox: Tuple[int, int, int, int],
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box on image
        
        Args:
            image: Input image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding box
        """
        x1, y1, x2, y2 = bbox
        return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color, thickness)

def process_image(image_path: str, debug: bool = False) -> Dict[str, Union[str, Tuple[int, int, int, int]]]:
    """
    Process image file and extract barcode information
    
    Args:
        image_path: Path to input image file
        debug: If True, show debug information
        
    Returns:
        Dictionary containing barcode data and coordinates
    """
    # Read and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    if debug:
        print(f"Image shape: {image.shape}")
        cv2.imshow("Original Image", image)
        cv2.waitKey(1000)
    
    # Initialize detector and process image
    detector = BarcodeDetector()
    result = detector.detect_and_decode(image, debug=debug)
    
    if result is None:
        return {
            "barcode_data": "",
            "barcode_coordinates": None,
            "error": "No barcode detected"
        }
    
    return {
        "barcode_data": result.data,
        "barcode_coordinates": result.bbox,
        "error": None
    }