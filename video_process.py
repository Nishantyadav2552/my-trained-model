from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Make sure best.pt is in the same folder or give full path

# Open webcam (0 is default webcam)
cap = cv2.VideoCapture(0)

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Run YOLOv8 on the frame
    results = model(frame, verbose=False)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("Number Plate Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
