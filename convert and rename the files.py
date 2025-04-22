def convert_and_copy_images(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith('.tif'):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    # Change the file extension to .jpg
                    new_file_name = os.path.splitext(file)[0] + '.jpg'
                    new_file_path = os.path.join(dest_dir, new_file_name)

                    img.convert("RGB").save(new_file_path, "JPEG")
source_folder = '/kaggle/input/city-persone/gtFinePanopticParts_trainval/gtFinePanopticParts/train'
destination_folder = '/kaggle/working/gtFinePanopticParts/train/images'
convert_and_copy_images(source_folder, destination_folder)
import json
import os

# Mapping of class labels to YOLOv5 class IDs
class_mapping = {
    "pedestrian": 0,
    "rider": 1,
    "sitting person": 2,
    "person group": 3,
}
def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bounding box coordinates from [x_min, y_min, width, height]
    to YOLO format (x_center, y_center, width, height) normalized to image size.
    """
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    w = width / img_width
    h = height / img_height
    return x_center, y_center, w, h

def convert_json_to_yolo_txt(json_path, output_dir):
    """
    Convert the JSON annotation to YOLO text file format.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    img_width = data["imgWidth"]
    img_height = data["imgHeight"]

    # Prepare the output filename based on the input JSON file
    txt_filename = os.path.splitext(os.path.basename(json_path))[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)

    with open(txt_path, 'w') as txt_file:
        for obj in data["objects"]:
            label = obj["label"]
            if label not in class_mapping:
                continue  # Skip labels that aren't in the class_mapping

            class_id = class_mapping[label]
            bbox = obj["bbox"]

            # Convert bounding box to YOLO format
            x_center, y_center, w, h = convert_bbox_to_yolo(bbox, img_width, img_height)

            # Write to the YOLO formatted .txt file
            txt_file.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

def process_json_files_in_dir(src_dir, output_dir):
    """
    Walk through each folder and subfolder in src_dir, and process all JSON files.
    Convert them to YOLO text format.
    """
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                convert_json_to_yolo_txt(json_path, output_dir)

# Define input and output directories
src_dir = '/kaggle/input/city-persone/gtBbox_cityPersons_trainval/gtBboxCityPersons/val'
output_dir = '/kaggle/working/gtFinePanopticParts/val/labels'

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all JSON files in the source directory and its subdirectories
process_json_files_in_dir(src_dir, output_dir)
