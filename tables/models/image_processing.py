import cv2
import os
import uuid

def ensure_directories_exist(*dirs):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def draw_bounding_boxes(img, boxes, confidences, labels, model_names):
    unique_ids = []
    for obj_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        conf = confidences[obj_idx]
        label = labels[obj_idx]

        # Generate a unique ID for each object
        unique_id = f"{uuid.uuid4()}"
        unique_ids.append(unique_id)

        # Draw the bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f'{model_names[int(label)]}: {conf:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return unique_ids

def save_bounding_box_image(img, box, output_dir, unique_id):
    x1, y1, x2, y2 = map(int, box)
    bbox_img = img[y1:y2, x1:x2]

    if bbox_img.size > 0:
        bbox_filename = os.path.join(output_dir, f'bbox_{unique_id}.jpg')
        cv2.imwrite(bbox_filename, bbox_img)
        return bbox_filename
    return None

def save_summary(summary_dir, image_name, summaries):
    summary_filename = os.path.join(summary_dir, f'summary_{image_name}.txt')
    with open(summary_filename, 'w') as f:
        for unique_id, description in summaries:
            f.write(f"Unique ID: {unique_id}\nDescription: {description}\n")
            f.write("\n")
