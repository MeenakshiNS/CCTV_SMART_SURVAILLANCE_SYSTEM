import torch
import cv2
import ultralytics
import transformers
import pytorchvideo

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
