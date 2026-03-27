import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import timm
import sys

# --- 1. CẤU HÌNH ---
KAGGLE_INPUT_DATASET = "full-groupemow-object-cropped" 
INPUT_DIR = f"/kaggle/input/{KAGGLE_INPUT_DATASET}/GroupEmoW_Full_Face_Cropped"

# Output folder mới
OUTPUT_FEATURE_DIR = "/kaggle/working/facial_features_groupemow_dinov2_vits14"

INPUT_SIZE = (224, 224)  
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# --- 2. XÂY DỰNG MODEL PRE-TRAINED (dinov2_vits14) ---
def build_pretrained_extractor():
    print("Đang tải dinov2_vits14...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    print("Feature Dim: 384")    
    return model

extractor = build_pretrained_extractor()
extractor.to(device)
print("Model extractor (dinov2_vits14) đã sẵn sàng.")
# --- 3. DATASET VÀ TRÍCH XUẤT ---
extraction_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FaceExtractionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.output_paths = []
        print(f"Đang quét toàn bộ ảnh trong {root_dir}...")
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(root_dir, split)
            if not os.path.exists(split_dir): continue
            for root, _, files in os.walk(split_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        self.image_paths.append(img_path)
                        relative_path = os.path.relpath(root, root_dir)
                        output_save_dir = os.path.join(OUTPUT_FEATURE_DIR, relative_path)
                        os.makedirs(output_save_dir, exist_ok=True)
                        base_filename = os.path.splitext(file)[0]
                        output_npy_path = os.path.join(output_save_dir, f"{base_filename}.npy")
                        self.output_paths.append(output_npy_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, output_path = self.image_paths[idx], self.output_paths[idx]
        if os.path.exists(output_path): return None, output_path
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, output_path
        except Exception as e: return None, output_path

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return None, None
    return torch.utils.data.dataloader.default_collate(batch)

extract_dataset = FaceExtractionDataset(root_dir=INPUT_DIR, transform=extraction_transform)
extract_loader = DataLoader(extract_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn_skip_none)

print(f"Tìm thấy {len(extract_dataset)} ảnh. Bắt đầu trích xuất...")

for inputs, out_paths in tqdm(extract_loader, desc="Extracting features"):
    if inputs is None: continue
    inputs = inputs.to(device)
    with torch.no_grad():
        # DeiT-Small trả về features trực tiếp (B, 384)
        features = extractor(inputs)
    features_np = features.cpu().numpy()
    for feature_vec, path in zip(features_np, out_paths):
        try: np.save(path, feature_vec)
        except Exception as e: print(f"Lỗi khi lưu file {path}: {e}")

print("\n--- TRÍCH XUẤT FACE (dinov2_vits14) HOÀN TẤT ---")