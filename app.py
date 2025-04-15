from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

# Run detection
results = model("image.png")  # your test image

# Get annotated image
annotated_frame = results[0].plot()

# Display the result
cv2.imshow("YOLOv8 Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
