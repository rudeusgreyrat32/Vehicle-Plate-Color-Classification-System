from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("license-plate-finetune-v1s.pt")

plate_color_mapping = {
    "white": "Private Vehicle",
    "yellow": "Commercial Vehicle",
    "green": "Electric Vehicle",
    "black": "Rental/Official Vehicle",
    "red": "Temporary Registration",
    "blue": "Diplomatic Vehicle"
}

def classify_plate_color(cropped_plate, k=2):
    # Resizing plates 
    cropped_plate = cv2.resize(cropped_plate, (100, 40))
    data = cropped_plate.reshape((-1, 3))
    data = np.float32(data)

    # Defining criteria and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Finding the largest cluster (dominant color)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color = centers[np.argmax(counts)].astype(int)

    # Converting to HSV
    dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = dominant_color_hsv

    # Color classification 
    if v > 180 and s < 60:   
        return "white"
    elif 20 < h < 40 and s > 100:  
        return "yellow"
    elif 35 < h < 85:          
        return "green"
    elif v < 60:                
        return "black"
    elif 0 <= h <= 10 and s > 120 and v > 70:  
        return "red"
    elif 90 < h < 130 and s > 80:  
        return "blue"
    else:
        return "unknown"


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame, verbose=False)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]

        if label == "License_Plate":
            cropped = frame[y1:y2, x1:x2]

            # Identifying plate color using k-means
            color = classify_plate_color(cropped)
            vehicle_type = plate_color_mapping.get(color, "Unknown")

            # Drawing bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{vehicle_type} ({color})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

    cv2.imshow("Vehicle Plate Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


