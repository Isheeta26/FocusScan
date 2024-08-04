# FocusScan

FocusScan is a Streamlit application designed for object detection, segmentation, and annotation. Leveraging YOLO for object detection, BLIP for image captioning, and EasyOCR for text extraction, FocusScan provides a comprehensive solution for image analysis.

## Features

- **File Upload:**
  - Allows users to upload an input image for processing.
- **Segmentation Display:**
  - Visualizes segmented objects on the original image.
- **Object Details:**
  - Displays extracted object images with unique IDs.
  - Provides descriptions, extracted text/data, and summarized attributes for each object.
- **Final Output:**
  - Shows the annotated image with all detected objects.
  - Presents a table with mapped data for each object in the master image.
- **User Interaction:**
  - Enables interaction with and review of each step in the analysis pipeline.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/focusscan.git
   cd focusscan

## Usage

1. **Run the Streamlit app:**

   ```bash
   streamlit run app.py

 2. **Upload an image:**

    Use the file uploader to select and upload an image.

 3. **View results:**

    The app will display segmented objects, object details, and the final output image with annotations.



