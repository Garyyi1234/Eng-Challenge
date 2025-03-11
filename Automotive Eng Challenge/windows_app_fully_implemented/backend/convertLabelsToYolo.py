import os

# Define paths
dataset_path = "/path/to/dataset"
images_dir = os.path.join(dataset_path, "images/train")  # Change if needed to val or test
labels_dir = os.path.join(dataset_path, "labels/train") # Change if needed to val or test
annotations_file = os.path.join(dataset_path, "train.txt") # Change if needed to val or test

# Create labels directory if not exists
os.makedirs(labels_dir, exist_ok=True)

# Read annotations and process them
with open(annotations_file, "r") as file:
    lines = file.readlines()

annotations = {}

for line in lines:
    parts = line.strip().split()
    img_filename = parts[0]
    class_id = int(parts[1]) - 1  # Convert 1-based index to 0-based for YOLO
    x_min, y_min, x_max, y_max = map(float, parts[2:])

    # Get image dimensions (assumes all images have the same size, otherwise fetch dynamically)
    img_path = os.path.join(images_dir, img_filename)
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_filename} not found. Skipping...")
        continue

    import cv2
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # Convert to YOLO format (normalize values)
    x_center = (x_min + x_max) / 2 / w
    y_center = (y_min + y_max) / 2 / h
    bbox_width = (x_max - x_min) / w
    bbox_height = (y_max - y_min) / h

    # Store annotations in a dictionary (grouped by image file)
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_filename)

    if label_filename not in annotations:
        annotations[label_filename] = []

    annotations[label_filename].append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

# Write YOLO formatted labels to individual .txt files
for label_filename, lines in annotations.items():
    label_path = os.path.join(labels_dir, label_filename)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

print("✅ Conversion complete! YOLO labels saved in:", labels_dir)
