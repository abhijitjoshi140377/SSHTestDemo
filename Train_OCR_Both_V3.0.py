import os
import torch


# Set environment variables to prevent OpenMP issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import cv2  # For image handling
import numpy as np  # For array manipulation
from ultralytics import YOLO
import easyocr  # Import EasyOCR
import pandas as pd

# Load the YOLOv8 model (pre-trained for "DisplayArea" detection)
model_path = r"C:\Users\003L0T744\PycharmProjects\yolov8_results\training_with_augment_7thOct24_5pm\training_with_augment_7thOct24_5pm\weights\best.pt"
model = YOLO(model_path)

# Run YOLO inference to detect "DisplayArea"
inference_results = model.predict(
    # source="C:/Users/003L0T744/PycharmProjects/yolov8/TestImagesYolov8",  # Path to images
    source= r"C:\Images\Cropped Images\Inference",  # Path to images
    imgsz=640,
    conf=0.2,  # Confidence threshold
    save=True,
    save_txt=True,  # Save labels as .txt files
    save_conf=True,  # Save confidence scores
    device='cpu'  # Set to 'cuda' for GPU
)

# Dynamically capture the most recent YOLO output directory
runs_dir = os.path.join("C:/Users/003L0T744/PycharmProjects/runs/detect")
yolo_output_dir = max([os.path.join(runs_dir, d) for d in os.listdir(runs_dir)], key=os.path.getmtime)
print(f"YOLO inference results saved to: {yolo_output_dir}")

# Path to the `labels` subfolder where YOLO saves label files
labels_dir = os.path.join(yolo_output_dir, "labels")
if not os.path.exists(labels_dir):
    print(f"Label folder does not exist: {labels_dir}")
else:
    print(f"Looking for label files in: {labels_dir}")

# Initialize EasyOCR reader (set gpu=False to use CPU)
reader = easyocr.Reader(['en'], gpu=False)

# Prepare to save OCR results in a CSV
ocr_summary = []

# Create OCR processed output folder inside the dynamically generated YOLO folder
final_output_dir = os.path.join(yolo_output_dir, "OCRProcessed")
if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)

# List to hold paths of processed OCR images
ocr_image_paths = []

# Ensure that OCR reads from the same directory where YOLO saved the inference results
for image_file in os.listdir(yolo_output_dir):  # Read images from the YOLO output directory
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image file
        img_path = os.path.join(yolo_output_dir, image_file)  # Full path to the image
        img = cv2.imread(img_path)  # Read the image for OCR processing

        # Ensure the image is loaded successfully
        if img is None:
            print(f"Error: Unable to load the image: {img_path}")
            continue

    # Get the corresponding label file for the YOLO bounding boxes from the `labels` subfolder
    label_file = os.path.splitext(image_file)[0] + ".txt"  # Label file should have the same name as the image
    label_path = os.path.join(labels_dir, label_file)  # Full path to the label file in the labels subfolder

    # Print the label path being read
    print(f"Looking for label file: {label_path}")

    if not os.path.exists(label_path):
        print(f"No label file found for {img_path}. Skipping...")
        continue

    # Read YOLO detection results from the label file
    with open(label_path, 'r') as file:
        boxes = file.readlines()

    img_height, img_width, _ = img.shape  # Get image dimensions

    display_area_detected = False  # To track if we detect a "DisplayArea"

    # Process each bounding box and convert from normalized to pixel coordinates
    for box in boxes:
        values = box.strip().split()
        class_id = int(float(values[0]))  # Convert from float to int

        # Process only if class_id == 5 (i.e., "DisplayArea")
        if class_id == 5:
            x_center_norm = float(values[1])
            y_center_norm = float(values[2])
            width_norm = float(values[3])
            height_norm = float(values[4])

            # Convert normalized coordinates to pixel values
            x_center = int(x_center_norm * img_width)
            y_center = int(y_center_norm * img_height)
            width = int(width_norm * img_width)
            height = int(height_norm * img_height)

            # Calculate the top-left and bottom-right coordinates of the bounding box
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            # Ensure bounding box is within image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)

            # Debugging: Print calculated pixel coordinates
            print(f"YOLO Bounding Box (pixel): x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

            # Crop the "DisplayArea" region from the image using pixel coordinates
            display_area = img[y_min:y_max, x_min:x_max]
            display_area_detected = True

            # Use EasyOCR to read text from the cropped display area
            ocr_results = reader.readtext(display_area, allowlist='0123456789')  # Modify allowlist if needed

            # Process OCR results
            for (bbox, text, prob) in ocr_results:
                # Draw the OCR result on the original image for visualization
                cv2.putText(img, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Append OCR result to the summary list
                ocr_summary.append({
                    'image': img_path,
                    'bounding_box': (x_min, y_min, x_max, y_max),
                    'text': text,
                    'confidence': prob
                })

    # Save the image with bounding boxes and OCR results only if a "DisplayArea" was detected
    if display_area_detected:
        output_image_path = os.path.join(final_output_dir, f"ocr_{os.path.basename(img_path)}")
        cv2.imwrite(output_image_path, img)
        ocr_image_paths.append(output_image_path)  # Store the path of the processed OCR image
        print(f"Processed image saved at: {output_image_path}")
    else:
        print(f"No 'DisplayArea' detected in {img_path}. Skipping saving.")

# Create a function to arrange images in a grid and save as a single image
def create_image_grid(image_paths, max_images_per_page=16):
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)

    # Calculate grid size
    num_images = len(images)
    rows = (num_images + 3) // 4  # 4 images per row
    grid_image = np.zeros((rows * 200, 4 * 200, 3), dtype=np.uint8)  # Create a blank image

    for idx, img in enumerate(images):
        row = idx // 4
        col = idx % 4
        # Resize each image to fit into the grid
        img_resized = cv2.resize(img, (200, 200))  # Resize to fit in the grid
        grid_image[row * 200:(row + 1) * 200, col * 200:(col + 1) * 200] = img_resized

    return grid_image

# Save combined OCR processed images as a grid
if ocr_image_paths:
    combined_image = create_image_grid(ocr_image_paths)
    grid_image_path = os.path.join(final_output_dir, "combined_ocr_images.png")
    cv2.imwrite(grid_image_path, combined_image)
    print(f"Combined OCR processed images saved at: {grid_image_path}")

# Save OCR results to CSV
if ocr_summary:
    ocr_df = pd.DataFrame(ocr_summary)
    ocr_csv_path = os.path.join(final_output_dir, "ocr_results_summary.csv")
    ocr_df.to_csv(ocr_csv_path, index=False)
    print(f"OCR results summary saved to {ocr_csv_path}")
else:
    print("No OCR results to save.")
