
# Pedestrian Detection on CityPerson Dataset

This project focuses on building a robust pedestrian detection model using the CityPerson dataset. The core challenge addressed is **class imbalance**, especially with underrepresented categories like Riders and Sitting Persons.

## 📌 Project Overview

- **Goal**: Improve detection accuracy on imbalanced pedestrian datasets.
- **Dataset**: [CityPerson](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.html) (subset of Cityscapes).
- **Techniques**: Data augmentation, class weighting, oversampling, and downsampling.
- **Models Used**: YOLOv5 (s/m/l/x), YOLOv8, YOLOv11, and MobileNetV2 with two types of classifier heads (FC and SVM).
- **Results**: Significant improvements using YOLOv5-l and balanced training.

## 📂 Project Structure

```
├── data/               # Raw and processed images
├── labels/             # YOLO-format label files
├── scripts/            # Preprocessing, training, evaluation
├── models/             # Model checkpoints
├── results/            # Output visualizations and metrics
└── README.md
```

## ⚙️ Installation & Requirements

Install the following dependencies:

```bash
pip install torch torchvision opencv-python-headless albumentations matplotlib seaborn Pillow numpy
```

Python modules used:
```python
import os
import zipfile
import shutil
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from PIL import Image
import cv2
import re
from collections import Counter, defaultdict
```

## 🚀 How to Run

### 1. Prepare Dataset
- Convert grayscale `.tif` images to `.jpg` in RGB.
- Convert JSON annotations to YOLO format: `[x_center, y_center, width, height]`.

### 2. Preprocessing
- Apply Albumentations-based augmentations (horizontal flip, rotations) **to both images and their corresponding labels**. Each augmented image must have its corresponding bounding boxes transformed and saved. For example, image number one should result in a flipped and rotated version with accurate YOLO-format bounding boxes.

- Balance dataset by:
  - Downsampling majority class (Pedestrian) **where many images contain over 30 pedestrian instances in one image**
  - Oversampling minorities (Rider, Sitting Person) **by applying targeted augmentations specifically to those classes**

### 3. Train Model

```bash
python train.py --img 1280 --batch 16 --epochs 50 --data cityperson.yaml --weights yolov5l.pt
python train.py --img 1280 --batch 16 --epochs 50 --data cityperson.yaml --weights yolov5m.pt
python train.py --img 1280 --batch 16 --epochs 50 --data cityperson.yaml --weights yolov8.pt
python train.py --img 1280 --batch 16 --epochs 50 --data cityperson.yaml --weights yolov11.pt
```

### 4. Evaluate

Results (sample):
```
Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    4/9     14.5G    0.05853    0.01663    0.00999         20       1280:1

                 Class     Images  Instances       P        R      mAP50   
                   all        500       4077      0.579    0.4     0.441     0.241
```

## 📊 Model Comparison

| Model     | Precision | Recall | F1-Score | Accuracy |
|-----------|-----------|--------|----------|----------|
| Pedestrian | 0.83     | 1.00   | 0.90     | 0.83     |
| Rider      | 0.83     | 0.08   | 0.15     |          |
| Sitting    | 0.72     | 0.35   | 0.47     |          |
| PersonGrp  | 0.99     | 0.86   | 0.92     |          |

## 🧠 Key Learnings

- High resolution images help with small object detection, **but they also demand more computational power, such as higher GPU memory and longer training times. This makes model selection and optimization important when working with limited resources**.
- Augmentation and resolution tuning are crucial for minority classes.
- YOLOv5-l offered the best trade-off between performance and speed.

## 🔮 Future Work

- Integrate advanced architectures (e.g., FPNs, Bayesian Uncertainty).
- Tune anchor boxes for better localization.
- Extend to real-time pedestrian tracking.

## 👨‍💻 Authors

- Mohammad Mahdi Heydari Asl
- Ramin Kazemi
- Reza Azari Aghouieh

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📚 Technical Report

> The technical report is private and not publicly shared.

## 🔄 Image Augmentation Example (Albumentations)

This project uses [Albumentations](https://albumentations.ai/) for powerful image and bounding box augmentation. Here are two key transformations:

```python
import cv2
import albumentations as A
import os

# 1. Horizontal Flip
transform_flip = A.Compose([
    A.HorizontalFlip(p=1.0),  # Always apply horizontal flip
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 2. Rotation 15° to the Right
transform_rotate_right_15 = A.Compose([
    A.Rotate(limit=[15, 15], p=1.0),  # Rotate exactly 15° to the right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

These augmentations rotate both the images and their corresponding bounding boxes.
