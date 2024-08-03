from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def load_blip_model(model_name="Salesforce/blip-image-captioning-base"):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model

def generate_summary(blip_processor, blip_model, image_path, boxes, unique_ids):
    img = Image.open(image_path).convert("RGB")
    summaries = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        roi = img.crop((x1, y1, x2, y2))  # Crop the region of interest
        inputs = blip_processor(images=roi, return_tensors="pt")

        # Generate the summary
        out = blip_model.generate(**inputs)
        description = blip_processor.decode(out[0], skip_special_tokens=True)

        summaries.append((unique_ids[i], description))

    return summaries
