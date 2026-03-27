import torch
import torchvision
from torchvision import models, transforms
from PIL import Image, ImageDraw 
import os
from tqdm import tqdm
import sys
import numpy as np # Import numpy

KAGGLE_INPUT_DATASET = "groupemow-full"
INPUT_DIR = f"/kaggle/input/{KAGGLE_INPUT_DATASET}/GroupEmoW"
OUTPUT_DIR = "/kaggle/working/groupemow-full-object-patches"
CONFIDENCE_THRESHOLD = 0.8
DRAW_BOUNDING_BOX = True 
DEBUG_BOX_DIR = "/kaggle/working/groupemow-object-bbox"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load Model
model = models.detection.fasterrcnn_resnet50_fpn(
    weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
)
model.eval()
model.to(device)
preprocess = transforms.Compose([transforms.ToTensor()])

def detect_and_crop_objects(source_base, output_base, model):
    if not os.path.exists(source_base):
        print(f"Source not found: {source_base}")
        return

    if DRAW_BOUNDING_BOX:
        os.makedirs(DEBUG_BOX_DIR, exist_ok=True)
    
    total_objects_saved = 0

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(source_base, split)
        if not os.path.exists(split_dir): continue
            
        for root, _, files in os.walk(split_dir):
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files: continue

            relative_path = os.path.relpath(root, source_base)
            output_save_dir = os.path.join(output_base, relative_path)
            os.makedirs(output_save_dir, exist_ok=True)
            
            debug_save_dir = None
            if DRAW_BOUNDING_BOX:
                debug_save_dir = os.path.join(DEBUG_BOX_DIR, relative_path)
                os.makedirs(debug_save_dir, exist_ok=True)

            for filename in tqdm(image_files, desc=f"Folder {relative_path}"):
                img_path = os.path.join(root, filename)
                base_filename, file_extension = os.path.splitext(filename)

                try:
                    img_pil = Image.open(img_path).convert('RGB')
                    orig_w, orig_h = img_pil.size # Lấy kích thước gốc

                    img_for_drawing = None
                    draw = None
                    if DRAW_BOUNDING_BOX:
                        img_for_drawing = img_pil.copy()
                        draw = ImageDraw.Draw(img_for_drawing)
                    
                    image_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        predictions = model(image_tensor)
                    
                    boxes = predictions[0]['boxes'].cpu()
                    scores = predictions[0]['scores'].cpu()
                    
                    obj_count = 0
                    for i, box in enumerate(boxes):
                        if scores[i] > CONFIDENCE_THRESHOLD:
                            box_coords = [int(b) for b in box] # [x1, y1, x2, y2]
                            
                            # --- 1. LƯU ẢNH CROP ---
                            object_patch = img_pil.crop(box_coords)
                            obj_count += 1
                            output_filename = f"{base_filename}_obj_{obj_count}{file_extension}"
                            output_path = os.path.join(output_save_dir, output_filename)
                            
                            if not os.path.exists(output_path):
                                object_patch.save(output_path)
                                
                                # --- 2. LƯU FILE TỌA ĐỘ (MỚI) ---
                                # Lưu [x1, y1, x2, y2, img_w, img_h]
                                bbox_info = np.array(box_coords + [orig_w, orig_h], dtype=np.float32)
                                bbox_filename = f"{base_filename}_obj_{obj_count}_bbox.npy"
                                bbox_path = os.path.join(output_save_dir, bbox_filename)
                                np.save(bbox_path, bbox_info)
                                
                            if DRAW_BOUNDING_BOX and draw:
                                draw.rectangle(box_coords, outline="red", width=2)
                                
                    total_objects_saved += obj_count
                    
                    if DRAW_BOUNDING_BOX and img_for_drawing:
                        debug_output_path = os.path.join(debug_save_dir, filename)
                        img_for_drawing.save(debug_output_path)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print(f"\n--- DONE OBJECT DETECTION. Saved {total_objects_saved} objects.")

if __name__ == "__main__":
    detect_and_crop_objects(INPUT_DIR, OUTPUT_DIR, model)