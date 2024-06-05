import os
import yaml
import json
from PIL import Image

def yolo_to_coco(yolo_dir, output_dir):
    # Read the data.yaml file to get class names
    with open(os.path.join(yolo_dir, 'data.yaml'), 'r') as file:
        data_yaml = yaml.safe_load(file)
    
    class_names = data_yaml['names']
    categories = [{"id": idx+1, "name": name} for idx, name in enumerate(class_names)]
    
    # Initialize COCO format data structure template
    def get_coco_template():
        return {
            "images": [],
            "annotations": [],
            "categories": categories
        }

    annotation_id = 1
    phase_dirs = ['train', 'val', 'test']

    # Prepare output directories
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    for phase in phase_dirs:
        coco_data = get_coco_template()
        images_dir = os.path.join(yolo_dir, phase, 'images')
        labels_dir = os.path.join(yolo_dir, phase, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            continue
        
        for idx, filename in enumerate(os.listdir(images_dir)):
            if not filename.endswith('.jpg') and not filename.endswith('.png'):
                continue
            
            # Read the corresponding label file
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            
            if not os.path.exists(label_path):
                continue
            
            with open(label_path, 'r') as file:
                lines = file.readlines()
                if not lines:
                    continue

                # Read the image to get dimensions
                image_path = os.path.join(images_dir, filename)
                image = Image.open(image_path)
                image_width, image_height = image.size
                image_id = len(coco_data["images"]) + 1

                # Add image info to COCO structure
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": filename,
                    "width": image_width,
                    "height": image_height
                })

                for line in lines:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())

                    # Convert YOLO format to COCO format
                    x_center_abs = x_center * image_width
                    y_center_abs = y_center * image_height
                    width_abs = width * image_width
                    height_abs = height * image_height
                    x_min = x_center_abs - width_abs / 2
                    y_min = y_center_abs - height_abs / 2

                    # Create annotation
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id) + 1,  # Assuming class_id starts from 0 in YOLO
                        "bbox": [x_min, y_min, width_abs, height_abs],
                        "area": width_abs * height_abs,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1

        # Save each phase annotations to separate JSON files
        output_json = os.path.join(output_dir, 'annotations', f'custom_{phase}.json')
        with open(output_json, 'w') as json_file:
            json.dump(coco_data, json_file, indent=4)

# Example usage


# Example usage
yolo_to_coco('/home/emailskpal/detr/datasets/IndoorObjectsDetectionYOLOv5/', '/home/emailskpal/detr/datasets/IndoorObjectsDetectionCOCO/')
