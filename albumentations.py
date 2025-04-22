import cv2
import albumentations as A
import os

# 1. Horizontal Flip only (already in your code)
transform_flip = A.Compose([
    A.HorizontalFlip(p=1.0),  # Always apply horizontal flip
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 2. Rotation 15° to the Right (already in your code)
transform_rotate_right_15 = A.Compose([
    A.Rotate(limit=[15, 15], p=1.0),  # Rotate exactly 15° to the right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 3. Rotation 15° to the Left (already in your code)
transform_rotate_left_15 = A.Compose([
    A.Rotate(limit=[-15, -15], p=1.0),  # Rotate exactly 15° to the left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 4. Rotation 5° to the Right
transform_rotate_right_5 = A.Compose([
    A.Rotate(limit=[5, 5], p=1.0),  # Rotate exactly 5° to the right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 5. Rotation 5° to the Left
transform_rotate_left_5 = A.Compose([
    A.Rotate(limit=[-5, -5], p=1.0),  # Rotate exactly 5° to the left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 6. Horizontal Flip + Rotation 5° to the Left
transform_flip_rotate_left_5 = A.Compose([
    A.HorizontalFlip(p=1.0),   # Flip horizontally
    A.Rotate(limit=[-5, -5], p=1.0),  # Then rotate 5° left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 7. Horizontal Flip + Rotation 5° to the Right
transform_flip_rotate_right_5 = A.Compose([
    A.HorizontalFlip(p=1.0),   # Flip horizontally
    A.Rotate(limit=[5, 5], p=1.0),   # Then rotate 5° right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 8. Rotation 3° to the Right
transform_rotate_right_3 = A.Compose([
    A.Rotate(limit=[3, 3], p=1.0),  # Rotate exactly 3° to the right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 9. Rotation 3° to the Left
transform_rotate_left_3 = A.Compose([
    A.Rotate(limit=[-3, -3], p=1.0),  # Rotate exactly 3° to the left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 10. Rotation 6° to the Right
transform_rotate_right_6 = A.Compose([
    A.Rotate(limit=[6, 6], p=1.0),  # Rotate exactly 6° to the right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 11. Rotation 6° to the Left
transform_rotate_left_6 = A.Compose([
    A.Rotate(limit=[-6, -6], p=1.0),  # Rotate exactly 6° to the left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 12. Horizontal Flip + Rotation 3° to the Left
transform_flip_rotate_left_3 = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.Rotate(limit=[-3, -3], p=1.0),  # Then rotate 3° left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 13. Horizontal Flip + Rotation 3° to the Right
transform_flip_rotate_right_3 = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.Rotate(limit=[3, 3], p=1.0),  # Then rotate 3° right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 14. Horizontal Flip + Rotation 6° to the Left
transform_flip_rotate_left_6 = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.Rotate(limit=[-6, -6], p=1.0),  # Then rotate 6° left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 15. Horizontal Flip + Rotation 6° to the Right
transform_flip_rotate_right_6 = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.Rotate(limit=[6, 6], p=1.0),  # Then rotate 6° right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 16. Rotation 13° to the Right
transform_rotate_right_13 = A.Compose([
    A.Rotate(limit=[13, 13], p=1.0),  # Rotate exactly 13° to the right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 17. Rotation 13° to the Left
transform_rotate_left_13 = A.Compose([
    A.Rotate(limit=[-13, -13], p=1.0),  # Rotate exactly 13° to the left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 18. Horizontal Flip + Rotation 13° to the Right
transform_flip_rotate_right_13 = A.Compose([
    A.HorizontalFlip(p=1.0),   # Flip horizontally
    A.Rotate(limit=[13, 13], p=1.0),  # Then rotate 13° right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 19. Horizontal Flip + Rotation 13° to the Left
transform_flip_rotate_left_13 = A.Compose([
    A.HorizontalFlip(p=1.0),   # Flip horizontally
    A.Rotate(limit=[-13, -13], p=1.0),  # Then rotate 13° left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 20. Rotation 14° to the Right
transform_rotate_right_14 = A.Compose([
    A.Rotate(limit=[14, 14], p=1.0),  # Rotate exactly 14° to the right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 21. Rotation 14° to the Left
transform_rotate_left_14 = A.Compose([
    A.Rotate(limit=[-14, -14], p=1.0),  # Rotate exactly 14° to the left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 22. Horizontal Flip + Rotation 14° to the Right
transform_flip_rotate_right_14 = A.Compose([
    A.HorizontalFlip(p=1.0),   # Flip horizontally
    A.Rotate(limit=[14, 14], p=1.0),  # Then rotate 14° right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 23. Horizontal Flip + Rotation 14° to the Left
transform_flip_rotate_left_14 = A.Compose([
    A.HorizontalFlip(p=1.0),   # Flip horizontally
    A.Rotate(limit=[-14, -14], p=1.0),  # Then rotate 14° left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 24. Rotation 11° to the Right
transform_rotate_right_11 = A.Compose([
    A.Rotate(limit=[11, 11], p=1.0),  # Rotate exactly 11° to the right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 25. Rotation 11° to the Left
transform_rotate_left_11 = A.Compose([
    A.Rotate(limit=[-11, -11], p=1.0),  # Rotate exactly 11° to the left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 26. Horizontal Flip + Rotation 11° to the Right
transform_flip_rotate_right_11 = A.Compose([
    A.HorizontalFlip(p=1.0),   # Flip horizontally
    A.Rotate(limit=[11, 11], p=1.0),  # Then rotate 11° right
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 27. Horizontal Flip + Rotation 11° to the Left
transform_flip_rotate_left_11 = A.Compose([
    A.HorizontalFlip(p=1.0),   # Flip horizontally
    A.Rotate(limit=[-11, -11], p=1.0),  # Then rotate 11° left
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Paths to your dataset
image_directory = "/kaggle/working/filtered/images"
label_directory = "/kaggle/working/filtered/labels"
output_directory = "/kaggle/working/filtered/augmented"

# Create output directories if they don't exist
output_image_dir = os.path.join(output_directory, "images")
output_label_dir = os.path.join(output_directory, "labels")
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Function to read YOLO labels
def read_yolo_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        class_label = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(class_label)
    return bboxes, class_labels

# List all augmentations including new ones
augmentations = [
    ("flip", transform_flip),
    ("rotate_right_15", transform_rotate_right_15),
    ("rotate_left_15", transform_rotate_left_15),
    ("rotate_right_5", transform_rotate_right_5),
    ("rotate_left_5", transform_rotate_left_5),
    ("flip_rotate_left_5", transform_flip_rotate_left_5),
    ("flip_rotate_right_5", transform_flip_rotate_right_5),
    ("rotate_right_3", transform_rotate_right_3),
    ("rotate_left_3", transform_rotate_left_3),
    ("rotate_right_6", transform_rotate_right_6),
    ("rotate_left_6", transform_rotate_left_6),
    ("flip_rotate_left_3", transform_flip_rotate_left_3),
    ("flip_rotate_right_3", transform_flip_rotate_right_3),
    ("flip_rotate_left_6", transform_flip_rotate_left_6),
    ("flip_rotate_right_6", transform_flip_rotate_right_6),

    # NEW AUGMENTATIONS
    ("rotate_right_13", transform_rotate_right_13),
    ("rotate_left_13", transform_rotate_left_13),
    ("flip_rotate_right_13", transform_flip_rotate_right_13),
    ("flip_rotate_left_13", transform_flip_rotate_left_13),
    ("rotate_right_14", transform_rotate_right_14),
    ("rotate_left_14", transform_rotate_left_14),
    ("flip_rotate_right_14", transform_flip_rotate_right_14),
    ("flip_rotate_left_14", transform_flip_rotate_left_14),
    ("rotate_right_11", transform_rotate_right_11),
    ("rotate_left_11", transform_rotate_left_11),
    ("flip_rotate_right_11", transform_flip_rotate_right_11),
    ("flip_rotate_left_11", transform_flip_rotate_left_11),
]

# Example for applying augmentations to a set of images and saving
for augmentation_name, augmentation_transform in augmentations:
    for image_name in os.listdir(image_directory):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_directory, image_name)
            label_path = os.path.join(label_directory, image_name.replace('.jpg', '.txt'))  # Adjust for your label format

            image = cv2.imread(image_path)
            bboxes, class_labels = read_yolo_labels(label_path)

            transformed = augmentation_transform(image=image, bboxes=bboxes, class_labels=class_labels)

            # Save the augmented image
            output_image_path = os.path.join(output_image_dir, f"{augmentation_name}_{image_name}")
            cv2.imwrite(output_image_path, transformed['image'])

            # Save the augmented label
            output_label_path = os.path.join(output_label_dir, f"{augmentation_name}_{image_name.replace('.jpg', '.txt')}")
            with open(output_label_path, 'w') as f:
                for bbox, class_label in zip(transformed['bboxes'], transformed['class_labels']):
                    f.write(f"{class_label} {' '.join(map(str, bbox))}\n")
