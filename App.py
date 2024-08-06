import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd
from tables.models.yolo_model import load_yolo_model, run_yolo_inference
from model.blip_model import load_blip_model, generate_summary
from model.text_ext_model_2 import extract_text
from tables.models.image_processing import ensure_directories_exist, draw_bounding_boxes, save_bounding_box_image

# Here we are loading the required model
yolo_model = load_yolo_model()
blip_processor, blip_model = load_blip_model()

# Checking the directories
output_dir = 'bounded_images'
summary_dir = 'summaries'
ensure_directories_exist(output_dir, summary_dir)

st.title("FocusScan : From Detection to Insights")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    image_path = os.path.join(output_dir, uploaded_file.name)
    image.save(image_path)

    # Converting PIL image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Running YOLO inference
    results = run_yolo_inference(yolo_model, image_path)

    boxes = []
    confidences = []
    labels = []

    for result in results:
        boxes.extend(result.boxes.xyxy.cpu().numpy())
        confidences.extend(result.boxes.conf.cpu().numpy())
        labels.extend(result.boxes.cls.cpu().numpy())

    # Creating bounding boxes and saving the ROI images
    unique_ids = draw_bounding_boxes(img, boxes, confidences, labels, yolo_model.names)
    for obj_idx, box in enumerate(boxes):
        save_bounding_box_image(img, box, output_dir, unique_ids[obj_idx])

    # Converting the OpenCV image back to PIL format for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(img_rgb)

    # Displaying the uploaded image and the segmented objects side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image', width=350)

    with col2:
        st.image(annotated_image, caption='Annotated Image', width=350)

    # Displaying the detected objects with their label names
    st.subheader("Detected Objects")

    # Defining  desired size for cropped images
    desired_width = 150
    desired_height = 150

    # Here we are  calculating the number of columns for the grid layout
    num_columns = min(3, len(boxes))  # Maximum 3 columns or fewer if fewer boxes
    rows = (len(boxes) + num_columns - 1) // num_columns  # Calculate number of rows

    for row in range(rows):
        cols = st.columns(num_columns)
        for col in range(num_columns):
            idx = row * num_columns + col
            if idx < len(unique_ids):
                with cols[col]:
                    object_image_path = os.path.join(output_dir, f'bbox_{unique_ids[idx]}.jpg')
                    object_image = Image.open(object_image_path)
                    object_image_resized = object_image.resize((desired_width, desired_height))
                    st.image(object_image_resized, caption=f'Label: {yolo_model.names[int(labels[idx])]}', width=desired_width)

    # Generating summaries and extracting the OCR detected text
    summaries = generate_summary(blip_processor, blip_model, image_path, boxes, unique_ids)
    ocr_texts = extract_text(image_path, boxes)

    # Creating and displaying object details
    object_details = []
    for idx, (unique_id, description) in enumerate(summaries):
        ocr_text = ocr_texts[idx] if idx < len(ocr_texts) else "No text detected"
        object_detail = {
            "Unique ID": unique_id,
            "Label": yolo_model.names[int(labels[idx])],
            "OCR Text": ocr_text,
            "Description": description
        }
        object_details.append(object_detail)

    # Displaying the table containing all mapped data for each object in the master image
    st.subheader("Table")
    st.write(pd.DataFrame(object_details))

    # Saving the summaries to JSON file
    json_summary_file = os.path.join(summary_dir, 'summary.json')
    with open(json_summary_file, 'w') as json_file:
        json.dump(object_details, json_file, indent=4)

    st.success("Processing complete")
