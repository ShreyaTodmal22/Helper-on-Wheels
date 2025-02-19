import cv2
from ultralytics import YOLO

# Load the YOLOv8 model 
model = YOLO(r'C:\Users\Shreya\Documents\helper_on_wheels\Object_detection\yolov8n.pt')

# Initialize the webcam (use 0 for the default camera or specify another index)
cap = cv2.VideoCapture(1)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # If the frame is not successfully captured, break the loop
    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Perform object detection on the frame
    results = model(frame)

    # Visualize the detection results on the frame
    annotated_frame = results[0].plot()

    # Display the frame with the detected objects
    cv2.imshow('YOLOv8 Live Object Detection', annotated_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
