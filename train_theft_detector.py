import sys
import os

import torch
import torchvision
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

# Root project path
PROJECT_DIR = r"C:\Users\admin\Desktop\Surveillance\Surveillance"

# Dataset root path
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

# Training dataset path
TRAIN_DIR = os.path.join(DATASET_DIR, "train")

# Validation directory selection (test preferred if exists)
if os.path.exists(os.path.join(DATASET_DIR, "test")):
    VAL_DIR = os.path.join(DATASET_DIR, "test")
else:
    VAL_DIR = os.path.join(DATASET_DIR, "val")

# Base pretrained VideoMAE checkpoint
MODEL_CKPT = "MCG-NJU/videomae-base"

# Training hyperparameters
BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 5e-5
NUM_FRAMES = 16

#Custom Video Dataset
class TheftDataset(Dataset):
    """
    Custom PyTorch Dataset for loading video clips
    and preparing them for VideoMAE training.
    """
    def __init__(self, root_dir, transform=None):
        self.video_paths = []
        self.labels = []
        self.transform = transform

        # Class-to-label mapping
        self.label_map = {"normal": 0, "theft": 1}
        
        # Check if dataset directory exists
        if not os.path.exists(root_dir):
            print(f"ERROR: Directory not found: {root_dir}")
            return

        # Collect video file paths and labels
        for label_name, label_idx in self.label_map.items():
            class_dir = os.path.join(root_dir, label_name)
            if os.path.exists(class_dir):
                files = [
                    f for f in os.listdir(class_dir)
                    if f.lower().endswith(('.mp4', '.avi', '.mov'))
                ]
                for fname in files:
                    self.video_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label_idx)

                print(f"   Found {len(files)} '{label_name}' videos in {root_dir}")

    def __len__(self):
        # Return total number of video samples
        return len(self.video_paths)

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            label = self.labels[idx]

            # Load encoded video
            video = EncodedVideo.from_path(video_path)
            duration = video.duration

            # Extract up to the first 4 seconds of the video
            video_data = video.get_clip(
                start_sec=0.0,
                end_sec=min(duration, 4.0)
            )

            # Retrieve raw video tensor (C, T, H, W)
            video_tensor = video_data['video']

            # Apply preprocessing transforms
            if self.transform:
                video_tensor = self.transform(video_tensor)

            # Rearrange tensor to (T, C, H, W) for VideoMAE
            video_tensor = video_tensor.permute(1, 0, 2, 3)

            return {
                "pixel_values": video_tensor,
                "labels": torch.tensor(label)
            }

        except Exception as e:
            # If any video fails to load, skip to the next one
            return self.__getitem__((idx + 1) % len(self))


#Training Pipeline
def run_training():
    print(f"--- Starting Setup in {PROJECT_DIR} ---")
    print(f"--- Validation Folder: {VAL_DIR} ---")

    # Load VideoMAE image processor to get normalization parameters
    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    mean = image_processor.image_mean
    std = image_processor.image_std

    # Define preprocessing pipeline for training and validation
    train_transform = Compose([
        UniformTemporalSubsample(NUM_FRAMES),   # Sample fixed number of frames
        Lambda(lambda x: x / 255.0),             # Normalize pixel values
        NormalizeVideo(mean, std),               # Apply mean/std normalization
        Resize((224, 224)),                      # Resize frames
        CenterCropVideo(224),                    # Center crop to model input size
    ])

    # Load training and validation datasets
    print("Loading Training Data...")
    train_dataset = TheftDataset(root_dir=TRAIN_DIR, transform=train_transform)

    print("Loading Validation Data...")
    val_dataset = TheftDataset(root_dir=VAL_DIR, transform=train_transform)

    # Load pretrained VideoMAE model and adapt it for binary classification
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_CKPT,
        label2id={"normal": 0, "theft": 1},
        id2label={0: "normal", 1: "theft"},
        ignore_mismatched_sizes=True,
    )

    # Define HuggingFace training configuration
    args = TrainingArguments(
        output_dir=os.path.join(PROJECT_DIR, "theft_model_output"),
        remove_unused_columns=False,
        eval_strategy="epoch",          # Run validation after each epoch
        save_strategy="epoch",          # Save model after each epoch
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        fp16=False,                     # Disable mixed precision (safe default)
        logging_steps=1,
        load_best_model_at_end=True,    # Restore best model automatically
        save_total_limit=2,             # Limit saved checkpoints
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    # Initialize Trainer API
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Start training
    print("--- Starting Training ---")
    trainer.train()

    # Save final trained model
    save_path = os.path.join(PROJECT_DIR, "final_theft_model")
    trainer.save_model(save_path)
    print(f"SUCCESS! Model saved to: {save_path}")


#Script Entry Point
if __name__ == "__main__":
    run_training()
