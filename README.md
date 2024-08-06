# FocusScan: From Detection to Insights
FocusScan is an innovative application designed to streamline the process of image analysis. It intelligently detects objects within images, identifies the presence of any text, extracts that text, and then provides a concise description of the detected objects.

## Functionality Overview
* Object Detection and identification: Utilized YOLOv8 for detecting and identifying the objects in an image.
* Text Extraction: Utilized Easy OCR ( Optical Character Recognition ) to extract the text if present in the detected objects.
* Description & Captioning: Utilized BLIP a vision transformer model to generate descriptions and captions for the detected objects.

![Screenshot 2024-08-06 001143](https://github.com/user-attachments/assets/0a4c0a65-e63a-49e6-bb96-f9710ae38d03)

![Screenshot 2024-08-06 000626](https://github.com/user-attachments/assets/9d09667a-225f-4e80-8e5b-d81c27b382ba)

![Screenshot 2024-08-06 000801](https://github.com/user-attachments/assets/d3cdef30-574e-4201-8d00-c632d35ffc03)


## Integrated Models and Techniques
### Yolov8
YOLOv8 (You Only Look Once version 8), is an object detection algorithm designed to detect and classify objects in images with remarkable speed and accuracy. YOLOv8's architecture includes three main components: the backbone, neck, and head. 
* The backbone, a custom CSPDarknet53 network, extracts features from the input image.
*  The neck, using a novel C2f module, integrates feature maps from different backbone stages to capture multi-scale information, improving detection, especially for small objects.
*  The head consists of multiple detection modules that predict bounding boxes, objectness scores, and class probabilities for each grid cell in the feature map.
### Easy OCR
EasyOCR is a Python computer language Optical Character Recognition (OCR) module that is both flexible and easy to use. EasyOCR is a powerful tool for text extraction from images, consisting of three core components.
* First, Feature Extraction employs deep learning models like ResNet and VGG to identify and extract relevant features from the image.
* Next, Sequence Labeling uses Long Short-Term Memory (LSTM) networks to understand and structure the sequential context of these features, facilitating accurate text pattern recognition.
* Finally, Decoding applies the Connectionist Temporal Classification (CTC) algorithm to translate the labeled sequences into readable text.
* This integrated approach, supported by the deep-text-recognition-benchmark framework, ensures that EasyOCR provides reliable and effective text recognition in diverse images.
### BLIP ( Bootstrapped Language Image Pre-training )
BLIP is a powerful model designed for generating detailed descriptions and captions for images. It combines visual and textual information to create meaningful summaries of visual content. It employs a two-step process: 
* First, it uses a visual encoder to extract features from the image, and then a language model generates coherent and contextually relevant descriptions based on these features.
* BLIP's pre-training on large-scale image-text pairs enables it to understand and articulate the relationship between visual content and textual descriptions effectively.

## Requirements
The following Python packages are required to run FocusScan. These can be installed using the requirements.txt file.
* transformers~=4.43.3
* pillow~=10.4.0
* easyocr~=1.7.1
* numpy~=1.26.4
* streamlit~=1.37.0
* opencv-python~=4.10.0.84
* pandas~=2.2.2
* torch~=2.4.0
* ultralytics~=8.2.72







