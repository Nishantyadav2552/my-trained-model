from ultralytics import YOLO
import cv2
import pytesseract
import re

model = YOLO("best.pt")
results = model("image.png")
image = cv2.imread("image.png")

for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box[:4])
    plate_crop = image[y1:y2, x1:x2]

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 7')
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

    print(f"Plate {i+1}: {clean_text}")

    cv2.imshow(f"Plate {i+1}", plate_crop)
    cv2.waitKey(0)

cv2.destroyAllWindows()
