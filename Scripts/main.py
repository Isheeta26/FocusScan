import os
import cv2
import matplotlib.pyplot as plt
import shutil
import json
from tables.models.yolo_model import load_yolo_model, run_yolo_inference
from model.blip_model import load_blip_model, generate_summary
from model.text_ext_model_2 import extract_text  # Import the OCR module
from tables.models.image_processing import ensure_directories_exist, draw_bounding_boxes, save_bounding_box_image

# Paths
input_images_dir = r"C:\Users\Isheeta\NLP_ProjectS\object_detection\data\input_images"
output_dir = r'C:\Users\Isheeta\NLP_ProjectS\object_detection\data\bounded_images'
summary_dir = r'C:\Users\Isheeta\NLP_ProjectS\object_detection\data\summaries'
json_summary_file = os.path.join(summary_dir, 'summary.json')  # Define the JSON summary file path

# Ensuring that the directories exist
ensure_directories_exist(output_dir, summary_dir)

# Load models
yolo_model = load_yolo_model()
blip_processor, blip_model = load_blip_model()

# Collecting all summaries
all_summaries = []

# Processing each image
for image_idx, image_name in enumerate(os.listdir(input_images_dir)):
    image_path = os.path.join(input_images_dir, image_name)

    # Running inference with YOLOv8
    results = run_yolo_inference(yolo_model, image_path)

    # Loading the image
    img = cv2.imread(image_path)

    boxes = []
    confidences = []
    labels = []

    for result in results:
        boxes.extend(result.boxes.xyxy.cpu().numpy())
        confidences.extend(result.boxes.conf.cpu().numpy())
        labels.extend(result.boxes.cls.cpu().numpy())

    # Creating bounding boxes and saving detected images
    unique_ids = draw_bounding_boxes(img, boxes, confidences, labels, yolo_model.names)
    for obj_idx, box in enumerate(boxes):
        save_bounding_box_image(img, box, output_dir, unique_ids[obj_idx])

    # Generating summaries
    summaries = generate_summary(blip_processor, blip_model, image_path, boxes, unique_ids)

    # Extracting OCR text
    ocr_texts = extract_text(image_path, boxes)

    # Preparing JSON structure for this image
    image_summary = {
        "master_image": image_name,
        "objects": []
    }

    # Adding objects' details to the JSON structure
    for idx, (unique_id, description) in enumerate(summaries):
        ocr_text = ocr_texts[idx] if idx < len(ocr_texts) else "No text detected"
        image_summary["objects"].append({
            "unique_id": unique_id,
            "label": yolo_model.names[int(labels[idx])],
            "description": description,
            "ocr_text": ocr_text
        })

    # Appending image summary to the list
    all_summaries.append(image_summary)

    # Converting BGR to RGB for matplotlib and display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Detected Objects in {image_name}")
    plt.axis('off')
    plt.show()

# Writing all summaries to JSON file
with open(json_summary_file, 'w') as json_file:
    json.dump(all_summaries, json_file, indent=4)

# Deleting the bounded_images directory
shutil.rmtree(output_dir)
