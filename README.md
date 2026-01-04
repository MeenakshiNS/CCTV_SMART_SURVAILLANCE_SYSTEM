## Project Structure

### Directories

- **`dataset/`**
  The main dataset directory containing video clips for the project.
  - **`train/`**: Labeled video data (Normal vs. Theft) used to train and fine-tune the VideoMAE model.
  - **`test/`**: Separate video data used to validate the model's performance on unseen examples.

- **`final_theft_model/`**
  The directory containing the final, fully trained VideoMAE model. This is the "production-ready" model loaded by the main surveillance script.

- **`theft_model_output/`**
  A temporary working directory containing intermediate model checkpoints and training logs. These files are generated automatically during the fine-tuning process to ensure training can resume if interrupted.

### Scripts & Code

- **`surveillance.py`**
  **[MAIN APPLICATION]**
  The core execution script for the project. It integrates the VideoMAE (Action Recognition) and YOLOv8 (Person Detection) models to monitor video feeds in real-time, detect theft, and send email alerts with evidence snapshots.

- **`train_theft_detector.py`**
  The training script used to fine-tune the pre-trained VideoMAE model on the custom dataset. It handles data loading, preprocessing, and the training loop.

- **`evaluate_theft_model.py`**
  A utility script used to audit the model's performance. It runs the trained model against the `test` dataset and generates accuracy metrics (Loss, Accuracy, Precision).

- **`test.py`**
  A sandbox/unit-testing file used for debugging specific code blocks or verifying that individual libraries (like OpenCV or PyTorch) are installed and working correctly.

### Model Weights

- **`yolov8n.pt`**
  The pre-trained weights for the **YOLOv8 Nano** model. This file acts as the "eyes" of the system, responsible for detecting and locating persons in the video frame with high speed and low latency.