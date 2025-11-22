
import cv2 as cv
from ultralytics import YOLO

# 1. Load the YOLO Model (yolov8n.pt should be in your folder)
model = YOLO("yolov8n.pt") 

# 2. Open the Video Source
# Ensure this path is correct, as discussed previously.
# NOTE: The .mpg format might be slow or unreliable; .mp4 is preferred.
cap = cv.VideoCapture("./Resources/Videos/Footage of jewellery store robbery.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set an inference confidence threshold (optional, but good practice)
CONFIDENCE_THRESHOLD = 0.1

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    # 3. Perform Object Detection on the current frame
    # Running the model on the frame returns a list of Results objects
    # Setting stream=True is efficient for processing video sequences
    results = model(frame, conf=CONFIDENCE_THRESHOLD, stream=True)

    # 4. Process and Display Results
    # The 'ultralytics' results object has a built-in 'plot()' function 
    # that draws boxes, labels, and confidence scores onto the frame.
    for result in results:
        # Get the frame with detections plotted
        annotated_frame = result.plot()

        # Display the frame
        cv.imshow("Smart Surveillance Feed", annotated_frame)

    # Press 'q' to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()