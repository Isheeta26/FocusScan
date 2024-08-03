import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import json
import pandas as pd

from tables.models.yolo_model import load_yolo_model, run_yolo_inference
from model.blip_model import load_blip_model, generate_summary
from model.text_ext_model_2 import extract_text
from tables.models.image_processing import ensure_directories_exist, draw_bounding_boxes, save_bounding_box_image

# Load models
yolo_model = load_yolo_model()
blip_processor, blip_model = load_blip_model()

# Ensure directories exist
output_dir = 'bounded_images'
summary_dir = 'summaries'
ensure_directories_exist(output_dir, summary_dir)

st.title("Object Detection and Annotation App")

# File Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    image_path = os.path.join(output_dir, uploaded_file.name)
    image.save(image_path)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Run YOLO inference
    results = run_yolo_inference(yolo_model, image_path)

    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    boxes = []
    confidences = []
    labels = []

    for result in results:
        boxes.extend(result.boxes.xyxy.cpu().numpy())
        confidences.extend(result.boxes.conf.cpu().numpy())
        labels.extend(result.boxes.cls.cpu().numpy())

    # Draw bounding boxes and save ROI images
    unique_ids = draw_bounding_boxes(img, boxes, confidences, labels, yolo_model.names)
    for obj_idx, box in enumerate(boxes):
        save_bounding_box_image(img, box, output_dir, unique_ids[obj_idx])

    # Convert OpenCV image back to PIL format for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(img_rgb)

    # Display the segmented objects on the original image
    st.image(annotated_image, caption='Segmented Objects', use_column_width=True)

    # Generate summaries and extract OCR text
    summaries = generate_summary(blip_processor, blip_model, image_path, boxes, unique_ids)
    ocr_texts = extract_text(image_path, boxes)

    # Display object details
    st.subheader("Object Details")
    object_details = []
    for idx, (unique_id, description) in enumerate(summaries):
        ocr_text = ocr_texts[idx] if idx < len(ocr_texts) else "No text detected"
        object_detail = {
            "Unique ID": unique_id,
            "Label": yolo_model.names[int(labels[idx])],
            "Description": description,
            "OCR Text": ocr_text
        }
        object_details.append(object_detail)
        st.write(f"**Object {idx + 1}**")
        st.image(os.path.join(output_dir, f'bbox_{unique_id}.jpg'), caption=f'Object {idx + 1}', use_column_width=True)
        st.write(f"**Label**: {object_detail['Label']}")
        st.write(f"**Description**: {object_detail['Description']}")
        st.write(f"**OCR Text**: {object_detail['OCR Text']}")

    # Display final output image with annotations
    st.subheader("Final Output")
    st.image(annotated_image, caption='Final Output Image with Annotations', use_column_width=True)

    # Display table containing all mapped data for each object in the master image
    st.subheader("Mapped Data for Each Object")
    st.write(pd.DataFrame(object_details))

    # Save all summaries to JSON file
    json_summary_file = os.path.join(summary_dir, 'summary.json')
    with open(json_summary_file, 'w') as json_file:
        json.dump(object_details, json_file, indent=4)

    st.success("Processing complete and results saved.")

# To run this app, save the script as `app.py` and execute `streamlit run app.py` in the terminal.
