import sys
import os

import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset

if not hasattr(torchvision.transforms, "functional_tensor"):
    sys.modules["torchvision.transforms.functional_tensor"] = torchvision.transforms.functional

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda, Resize
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Root project path
PROJECT_DIR = r"C:\Users\SHIRAN S K\Desktop\Projects\Surveillance\new"

# path containing test videos
TEST_DIR = r"C:\Users\SHIRAN S K\Desktop\Projects\Surveillance\new\dataset\test"

# path containing the VideoMAE model
MODEL_DIR = os.path.join(PROJECT_DIR, "final_theft_model")

# Base pretrained VideoMAE checkpoint
BASE_CKPT = "MCG-NJU/videomae-base"

# Evaluation batch size
BATCH_SIZE = 2

# Number of frames sampled per video clip
NUM_FRAMES = 16


#Custom Dataset Class
class TheftDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing video clips
    for theft vs normal classification.
    """
    def __init__(self, root_dir, transform=None):
        self.video_paths = []
        self.labels = []
        self.transform = transform

        # Mapping class names to numeric labels
        self.label_map = {"normal": 0, "theft": 1}

        # Collect video file paths and corresponding labels
        for name, idx in self.label_map.items():
            class_dir = os.path.join(root_dir, name)
            if os.path.exists(class_dir):
                for f in os.listdir(class_dir):
                    if f.lower().endswith((".mp4", ".avi", ".mov")):
                        self.video_paths.append(os.path.join(class_dir, f))
                        self.labels.append(idx)

    def __len__(self):
        # Total number of video samples
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Load video from disk
        video = EncodedVideo.from_path(self.video_paths[idx])

        # Get video duration
        duration = video.duration

        # Extract a clip
        clip = video.get_clip(start_sec=0.0, end_sec=min(duration, 4.0))

        # Extract raw video frames
        frames = clip["video"]

        # Apply preprocessing transforms if provided
        if self.transform:
            frames = self.transform(frames)

        # Rearrange dimensions to (T, C, H, W) for VideoMAE
        frames = frames.permute(1, 0, 2, 3)

        # Return input tensor and corresponding label
        return {
            "pixel_values": frames,
            "labels": torch.tensor(self.labels[idx])
        }


#Metric Computation Function
def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1-score
    during model evaluation.
    """
    logits, labels = eval_pred

    # Convert logits to predicted class indices
    preds = np.argmax(logits, axis=1)

    # Calculate evaluation metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


#Evaluation Pipeline
def main():
    # Load VideoMAE image processor for normalization values
    processor = VideoMAEImageProcessor.from_pretrained(BASE_CKPT)
    mean = processor.image_mean
    std = processor.image_std

    # Define video preprocessing pipeline
    transform = Compose([
        UniformTemporalSubsample(NUM_FRAMES),   # Sample fixed number of frames
        Lambda(lambda x: x / 255.0),             # Normalize pixel values to [0,1]
        NormalizeVideo(mean, std),               # Apply mean/std normalization
        Resize((224, 224)),                      # Resize frames
        CenterCropVideo(224),                    # Center crop to model input size
    ])

    # Load test dataset
    test_dataset = TheftDataset(TEST_DIR, transform)

    # Load trained VideoMAE classification model
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR)

    # Define evaluation configuration
    args = TrainingArguments(
        output_dir=os.path.join(PROJECT_DIR, "eval_output"),
        per_device_eval_batch_size=BATCH_SIZE,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to="none"
    )

    # Initialize HuggingFace Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Run evaluation on test dataset
    metrics = trainer.evaluate()

    # Print evaluation results
    print("\n===== TEST SET RESULTS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")


#Script Entry Point
if __name__ == "__main__":
    main()
