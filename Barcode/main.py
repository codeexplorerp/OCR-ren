# In your main.py
from Barcode.Barcode_Detector import process_image

result = process_image("pic.png", debug=True)
print(result)