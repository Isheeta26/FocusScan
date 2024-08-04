# Object Detection and Annotation App

This Streamlit application allows users to upload images for object detection, segmentation, and annotation. It leverages YOLO for object detection, BLIP for image captioning, and EasyOCR for text extraction.

## Features

- **File Upload:**
  - Users can upload an input image.
- **Segmentation Display:**
  - Display the segmented objects on the original image.
- **Object Details:**
  - Show extracted object images with unique IDs.
  - Display descriptions, extracted text/data, and summarized attributes for each object.
- **Final Output:**
  - Display the final output image with annotations.
  - Present a table containing all mapped data for each object in the master image.
- **User Interaction:**
  - Allow users to interact with and review each step of the pipeline.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/object-detection-annotation-app.git
   cd object-detection-annotation-app

## Usage

1. **Run the Streamlit app:**

   ```bash
   streamlit run app.py

 2. **Upload an image:**

    Use the file uploader to select and upload an image.

 3. **View results:**

    The app will display segmented objects, object details, and the final output image with annotations.

## Live Application & Deployment
Explore the live version of FocusScan [here](https://isheeta-sharma-wasserstoff-aiinterntask.onrender.com).



