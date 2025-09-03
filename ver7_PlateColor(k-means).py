from ultralytics import YOLO
import cv2
import numpy as np

# ------------------- Setup -------------------
# Load your custom YOLO model (trained for vehicles + number plates)
model = YOLO("license-plate-finetune-v1s.pt")

# Vehicle type mapping by number plate color
plate_color_mapping = {
    "white": "Private Vehicle",
    "yellow": "Commercial Vehicle",
    "green": "Electric Vehicle",
    "black": "Rental/Official Vehicle",
    "red": "Temporary Registration",
    "blue": "Diplomatic Vehicle"
}

# Function to classify plate color using OpenCV k-means
def classify_plate_color(cropped_plate, k=2):
    # Resize for consistency
    cropped_plate = cv2.resize(cropped_plate, (100, 40))
    data = cropped_plate.reshape((-1, 3))
    data = np.float32(data)

    # Define criteria and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Find the largest cluster (dominant color)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color = centers[np.argmax(counts)].astype(int)

    # Convert to HSV for robust classification
    dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = dominant_color_hsv

    # Color classification rules
    if v > 180 and s < 60:      # bright + low saturation → White
        return "white"
    elif 20 < h < 40 and s > 100:  # Yellow hue range
        return "yellow"
    elif 35 < h < 85:           # Green range
        return "green"
    elif v < 60:                # Very dark → Black
        return "black"
    elif 0 <= h <= 10 and s > 120 and v > 70:  # Red
        return "red"
    elif 90 < h < 130 and s > 80:  # Blue
        return "blue"
    else:
        return "unknown"

# ------------------- Image/Video Processing -------------------
cap = cv2.VideoCapture("sample5.mp4")  # Use video path OR image path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]

        # If detection is a number plate → crop it
        if label == "License_Plate":
            cropped = frame[y1:y2, x1:x2]

            # Identify plate color using k-means
            color = classify_plate_color(cropped)
            vehicle_type = plate_color_mapping.get(color, "Unknown")

            # Draw bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{vehicle_type} ({color})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

    cv2.imshow("Vehicle Plate Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
