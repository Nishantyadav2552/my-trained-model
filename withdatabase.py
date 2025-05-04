from ultralytics import YOLO
import cv2
import pytesseract
import re
import numpy as np
from datetime import datetime
import cx_Oracle

# Connect to Oracle DB
connection = cx_Oracle.connect("Nishant", "Nishant", "localhost:1521/xe")
cursor = connection.cursor()

# Load YOLOv8 model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)
frame_count = 0
processed_frame = None

# Regex for raw plate (no space)
plate_pattern_raw = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$'

# Haryana RTO code mapping
rto_mapping = {
    "HR01": "Ambala", "HR02": "Jagadhri", "HR03": "Panchkula", "HR04": "Naraingarh", "HR05": "Karnal",
    "HR06": "Panipat", "HR07": "Kurukshetra", "HR08": "Kaithal", "HR09": "Gulah", "HR10": "Sonepat",
    "HR11": "Gohana", "HR12": "Rohtak", "HR13": "Jhajjar", "HR14": "Jhajjar", "HR15": "Meham",
    "HR16": "Bhiwani", "HR17": "Bhiwani", "HR18": "Loharu", "HR19": "Charkhi Dadri", "HR20": "Hisar",
    "HR21": "Hansi", "HR22": "Fatehabad", "HR23": "Tohana", "HR24": "Sirsa", "HR25": "Sirsa",
    "HR26": "Gurgaon", "HR27": "Hissar", "HR28": "Ferozepur Jhirka", "HR29": "Faridabad", "HR30": "Palwal",
    "HR31": "Jind", "HR32": "Narwana", "HR33": "Safidon", "HR34": "Mahendragarh", "HR35": "Narnaul",
    "HR36": "Rewari", "HR37": "Ambala", "HR38": "Faridabad", "HR39": "Hissar", "HR40": "Assandh",
    "HR41": "Pehowa", "HR42": "Ganaur", "HR43": "Rewari", "HR44": "Sirsa", "HR45": "Karnal",
    "HR46": "Rohtak", "HR47": "Rewari", "HR48": "Bhiwani", "HR49": "Kalka", "HR50": "Palwal",
    "HR51": "Faridabad", "HR52": "Hathin", "HR53": "Adampur", "HR54": "Ambala", "HR55": "Gurgaon",
    "HR56": "Jind", "HR57": "Sirsa", "HR58": "Yamuna Nagar", "HR59": "Fatehabad", "HR60": "Samalkha",
    "HR61": "Bhiwani", "HR62": "Fatehabad", "HR63": "Jhajjar", "HR64": "Kaithal", "HR65": "Kurukshetra",
    "HR66": "Narnaul", "HR67": "Panipat", "HR68": "Panchkula", "HR69": "Sonepat", "HR71": "Yamuna Nagar",
    "HR72": "Gurgaon", "HR73": "Palwal", "HR74": "Hissar", "HR75": "Karnal", "HR76": "Pataudi",
    "HR77": "Beri", "HR78": "Shahabad Markanda", "HR79": "Kharkhoda", "HR80": "Hissar", "HR81": "Bawal",
    "HR82": "Kanina", "HR83": "Kalayat", "HR84": "Charkhi Dadri", "HR85": "Rai", "HR86": "Sonipat",
    "HR87": "Badkhal", "HR88": "Kundli", "HR89": "Badli", "HR90": "Jind", "HR91": "Gharaunda",
    "HR93": "Punhana", "HR94": "Kalanwali", "HR95": "Sampla", "HR96": "Tauru", "HR97": "Ladwa",
    "HR98": "Badshahpur", "HR99": "Temporary Registration"
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 3 == 0:
        results = model(frame)
        processed_frame = results[0].plot()

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = frame[y1:y2, x1:x2]

            # Preprocessing for OCR
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR
            text = pytesseract.image_to_string(thresh, config='--psm 7')
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())  # Clean non-alphanumerics

            # Validate clean plate (no space)
            if re.match(plate_pattern_raw, clean):
                # Format to "HR 26 DK 1234"
                formatted = re.sub(r'([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})', r'\1 \2 \3 \4', clean)
                code_key = clean[:4]  # HR26, HR29, etc.
                location = rto_mapping.get(code_key, "Unknown")

                now = datetime.now()

                # Check for duplicates within last 1 minute
                cursor.execute("""
                    SELECT COUNT(*) FROM plate_log
                    WHERE plate_number = :1
                    AND detected_time >= (SYSTIMESTAMP - INTERVAL '1' MINUTE)
                """, (formatted,))
                count = cursor.fetchone()[0]

                if count == 0:
                    # Insert to DB
                    cursor.execute("""
                        INSERT INTO plate_log (plate_number, detected_time, location)
                        VALUES (:1, :2, :3)
                    """, (formatted, now, location))
                    connection.commit()

                    cv2.putText(processed_frame, f"{formatted} | {location}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(processed_frame, "Duplicate (1 min)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # Not a valid plate
                cv2.putText(processed_frame, "Invalid Plate", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show output frame
    if processed_frame is not None:
        cv2.imshow("Live Number Plate Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
cursor.close()
connection.close()
