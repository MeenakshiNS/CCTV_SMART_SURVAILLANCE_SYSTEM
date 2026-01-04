import sys
import os
import smtplib
import ssl
from collections import deque
from email.message import EmailMessage

import cv2
import torch
import torchvision
import numpy as np
from ultralytics import YOLO
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda, Resize
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


# model path
MODEL_PATH = r"C:\Users\hirin\Downloads\Surveillance_final_file\Surveillance\Surveillance\final_theft_model"

# video source path
# VIDEO_SOURCE = r"C:\Users\hirin\Downloads\Surveillance_final_file\Surveillance\Surveillance\dataset\test\normal\12760321-uhd_2160_3840_24fps.mp4"
VIDEO_SOURCE=0

# Email credentials
SENDER_EMAIL = "hiringhrbot39@gmail.com"
RECEIVER_EMAIL = "hiringhrbot39@gmail.com"
EMAIL_PASSWORD = "qrze rztm psnb dljq"

# select GPU if available, otherwise run on CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
video_mae = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH)

# Load the image processor us
image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

# Move model to selected device and set to evaluation mode
video_mae.to(DEVICE)
video_mae.eval()

# Load YOLO model, YOLOv8 nano model for fast person detection
yolo_model = YOLO("yolov8n.pt")

# Mean and standard deviation for normalization
mean = image_processor.image_mean
std = image_processor.image_std

# Validation-time video transforms
val_transform = Compose([
    UniformTemporalSubsample(16),            # Sample 16 frames
    Lambda(lambda x: x / 255.0),              # Normalize pixel values to [0, 1]
    NormalizeVideo(mean, std),                # Apply ImageNet normalization
    Resize((224, 224)),                       # Resize frames
    CenterCropVideo(224),                     # Center crop to model input size
])


# to store last 16 frames
frame_buffer = deque(maxlen=16)

# Current system state
current_status = "Normal"

# Prevent repeated alert emails
alert_sent = False


#Email Alert Function
def send_alert(frame, confidence):
    """
    Sends an email alert with a captured frame as evidence
    when theft is detected.
    """
    global alert_sent
    if alert_sent:
        return

    print(f"--- Sending Email Alert ({confidence:.1%}) ---")

    # Save current frame as evidence image
    cv2.imwrite("evidence.jpg", frame)

    # Create email message
    msg = EmailMessage()
    msg['Subject'] = "🚨 ALERT: Theft Detected!"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg.set_content(f"Theft detected with {confidence:.1%} confidence.")

    # Attach evidence image
    with open("evidence.jpg", 'rb') as f:
        msg.add_attachment(
            f.read(),
            maintype='image',
            subtype='jpeg',
            filename='evidence.jpg'
        )

    # Send email securely using Gmail SMTP
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(SENDER_EMAIL, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("✅ Email Sent!")
        alert_sent = True
    except Exception as e:
        print(f"❌ Email Failed: {e}")


#Main Surveillance Loop
def main():
    global current_status

    # Open video stream
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    last_yolo_boxes = []

    # Adaptive skipping based on hardware for performance
    if DEVICE.type == 'cuda':
        YOLO_SKIP = 3      # Run YOLO every 3 frames
        MAE_SKIP = 8       # Run VideoMAE every 8 frames
    else:
        YOLO_SKIP = 5
        MAE_SKIP = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and store in buffer
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb)

        confidence_score = 0.0

        # Run VideoMAE only when buffer is full and skip condition met
        if len(frame_buffer) == 16 and frame_count % MAE_SKIP == 0:
            video_tensor = torch.from_numpy(np.stack(frame_buffer))
            video_tensor = video_tensor.permute(3, 0, 1, 2)
            video_tensor = val_transform(video_tensor)
            video_tensor = video_tensor.permute(1, 0, 2, 3)

            inputs = {
                "pixel_values": video_tensor.unsqueeze(0).to(DEVICE)
            }

            # Inference without gradient tracking
            with torch.no_grad():
                outputs = video_mae(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Theft class confidence (index 1)
                confidence_score = probs[0][1].item()

                # Update system status
                if confidence_score > 0.60:
                    current_status = "THEFT"
                else:
                    current_status = "Normal"

        # Run YOLO person detection periodically
        if frame_count % YOLO_SKIP == 0:
            results = yolo_model(frame, verbose=False, classes=[0])
            last_yolo_boxes = []

            # Store detected person bounding boxes
            for result in results:
                for box in result.boxes:
                    coords = map(int, box.xyxy[0])
                    last_yolo_boxes.append(tuple(coords))

        # Set visualization color based on system state
        box_color = (0, 0, 255) if current_status == "THEFT" else (0, 255, 0)
        label_text = f"{current_status}"

        # Draw bounding boxes and labels
        for (x1, y1, x2, y2) in last_yolo_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                box_color,
                2
            )

        # Display theft alert and send email once
        if current_status == "THEFT":
            cv2.putText(
                frame,
                "!!! THEFT DETECTED !!!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
            if not alert_sent:
                send_alert(frame, confidence_score)

        # Show live surveillance feed
        cv2.imshow("Surveillance Feed", frame)

        frame_count += 1

        # pressing 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Program Entry Point
if __name__ == "__main__":
    main()
