import os
import json
import shutil

# Define the new categories (excluding the ones to be removed)
new_categories = [
    {"id": 1, "name": "door"},
    {"id": 2, "name": "cabinet"},
    {"id": 3, "name": "refrigerator"},
    {"id": 4, "name": "window"},
    {"id": 5, "name": "chair"},
    {"id": 6, "name": "table"},
    {"id": 7, "name": "couch"},
    {"id": 8, "name": "bed"},
    {"id": 9, "name": "oven"},
    {"id": 10, "name": "tv"}
]

# Create a mapping of old category names to new category ids
category_mapping = {
    "door": 1, "openedDoor": 1,
    "cabinetDoor": 2, "cabinet": 2,
    "refrigeratorDoor": 3, "refrigerator": 3,
    "window": 4,
    "chair": 5, "Chair": 5,
    "table": 6, "Table": 6, "dining table": 6,
    "couch": 7, "sofa": 7, "Sofa": 7,
    "bed": 8,
    "oven": 9,
    "tv": 10
}

# Define the dataset directories
base_dir = "/home/emailskpal/detr/datasets"
dataset_dirs = [
    os.path.join(base_dir, "IndoorObjectsDetectionCOCO"),
    os.path.join(base_dir, "furniture"),
    os.path.join(base_dir, "furniture2"),
    os.path.join(base_dir, "householdappliance"),
    os.path.join(base_dir, "door")
]

# Function to load annotations
def load_annotations(dataset_dir, split):
    path = os.path.join(dataset_dir, "annotations", f"custom_{split}.json")
    with open(path) as f:
        return json.load(f)

# Function to update categories and annotations
def update_annotations(annotations, original_categories, category_mapping, annotation_id_offset, image_id_mapping):
    new_annotations = []
    for annotation in annotations:
        category_id = annotation["category_id"]
        category_name = [cat["name"] for cat in original_categories if cat["id"] == category_id][0]
        if category_name in category_mapping:
            new_category_id = category_mapping[category_name]
            annotation["category_id"] = new_category_id
            annotation["id"] += annotation_id_offset
            annotation["image_id"] = image_id_mapping[annotation["image_id"]]
            new_annotations.append(annotation)
    return new_annotations

# Combine datasets
def combine_datasets():
    combined_annotations = {"train": {"images": [], "annotations": [], "categories": new_categories},
                            "val": {"images": [], "annotations": [], "categories": new_categories},
                            "test": {"images": [], "annotations": [], "categories": new_categories}}
    global_image_id = {"train": 0, "val": 0, "test": 0}
    global_annotation_id = {"train": 0, "val": 0, "test": 0}

    for dataset_dir in dataset_dirs:
        for split in ["train", "val", "test"]:
            # Load annotations
            annotations = load_annotations(dataset_dir, split)
            images = annotations["images"]
            annos = annotations["annotations"]
            original_categories = annotations["categories"]

            # Map old image IDs to new global image IDs
            image_id_mapping = {}
            for image in images:
                new_id = global_image_id[split]
                image_id_mapping[image["id"]] = new_id
                image["id"] = new_id
                global_image_id[split] += 1

            annos = update_annotations(annos, original_categories, category_mapping, global_annotation_id[split], image_id_mapping)

            # Update annotation ids
            for anno in annos:
                anno["id"] = global_annotation_id[split]
                global_annotation_id[split] += 1

            # Append to combined annotations
            combined_annotations[split]["images"].extend(images)
            combined_annotations[split]["annotations"].extend(annos)

    return combined_annotations

# Save combined annotations
def save_combined_annotations(combined_annotations, split):
    os.makedirs("merged_dataset/annotations", exist_ok=True)
    path = f"merged_dataset/annotations/custom_{split}.json"
    with open(path, "w") as f:
        json.dump(combined_annotations[split], f)

# Copy image files
def copy_images(split):
    os.makedirs(f"merged_dataset/{split}", exist_ok=True)
    for dataset_dir in dataset_dirs:
        src_dir = os.path.join(dataset_dir, split)
        dst_dir = f"merged_dataset/{split}"
        for filename in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))

# Merge the datasets
combined_annotations = combine_datasets()
for split in ["train", "val", "test"]:
    save_combined_annotations(combined_annotations, split)
    copy_images(split)
