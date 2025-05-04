from ultralytics import YOLO
import cv2
import pytesseract
import re
import numpy as np

# Load your custom YOLOv8 model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)
frame_count = 0
processed_frame = None  # To store last processed frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process every 3rd frame only
    if frame_count % 3 == 0:
        results = model(frame)
        processed_frame = results[0].plot()

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = frame[y1:y2, x1:x2]

            # OCR preprocessing
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Run OCR
            text = pytesseract.image_to_string(thresh, config='--psm 7')
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

            # Display text on the image
            cv2.putText(processed_frame, f"{clean_text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Show last processed frame (even if skipped)
    if processed_frame is not None:
        cv2.imshow("Live Plate Detection + OCR", processed_frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
