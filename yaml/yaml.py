import yaml

# Paths to your dataset folders
train_images_path = '/kaggle/working/gtFinePanopticParts/train'
val_images_path = '/kaggle/working/gtFinePanopticParts/val'

# Define class weights for imbalanced data
# Higher values mean the model will pay more attention to these classes
class_weights = {0: 1.0, 1: 30.0, 2: 5.0, 3: 5.0}  # Adjust as necessary <button class="citation-flag" data-index="3">

# YOLOv5/YOLOv8 configuration for custom dataset with class weights
data_config = {
    'train': train_images_path,
    'val': val_images_path,
    'nc': 4,  # Number of classes
    'names': ["pedestrian", "rider", "sitting person", "person group"],
    'class_weights': class_weights  # Add class weights to the config
}

# Save the YAML file
with open('my_dataset1.yaml', 'w') as file:
    yaml.dump(data_config, file)

print("YAML configuration file created as my_dataset1.yaml")
