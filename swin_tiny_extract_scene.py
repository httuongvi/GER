import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import os
import numpy as np
from tqdm import tqdm
import sys

# ==========================================
# TRÍCH XUẤT SCENE FEATURES (PATCH LEVEL) - 49 TOKENS
# ==========================================

print("\n" + "="*60)
print("  SWIN TRANSFORMER TINY - SCENE PATCH EXTRACTION (49x768)")
print("="*60 + "\n")

# --- 1. CẤU HÌNH ---
KAGGLE_SCENE_DATASET = "groupemow-full" 
INPUT_DIR = f"/kaggle/input/{KAGGLE_SCENE_DATASET}/GroupEmoW"

OUTPUT_FEATURE_DIR = "/kaggle/working/Scene_Features_GroupEmoW_swin_tiny_patch49"

INPUT_SIZE = (224, 224)  
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# --- 2. XÂY DỰNG MODEL PRE-TRAINED ---
def build_pretrained_extractor():
    print("Đang tải Swin Transformer Tiny...")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
    print("Feature dimension expected: 49 x 768")
    print("Model loaded successfully!\n")
    return model

extractor = build_pretrained_extractor()
extractor.eval() 
extractor.to(device)

# --- 3. DATASET ---
extraction_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

class SceneExtractionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.output_paths = []
        
        print(f"Đang quét ảnh trong: {root_dir}")
        
        for split in ['train', 'val', 'test']: 
            split_dir = os.path.join(root_dir, split)
            if not os.path.exists(split_dir): 
                continue
                
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
        
        print(f"Tổng số ảnh: {len(self.image_paths)}\n")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, output_path = self.image_paths[idx], self.output_paths[idx]
        if os.path.exists(output_path): 
            return None, output_path
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform: 
                image = self.transform(image)
            return image, output_path
        except Exception as e: 
            return None, output_path

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return None, None
    return torch.utils.data.dataloader.default_collate(batch)

# --- 4. TRÍCH XUẤT FEATURES ---
print("BẮT ĐẦU TRÍCH XUẤT SCENE FEATURES...\n")

extract_dataset = SceneExtractionDataset(root_dir=INPUT_DIR, transform=extraction_transform)
extract_loader = DataLoader(extract_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn_skip_none)

extracted_count = 0

for inputs, out_paths in tqdm(extract_loader, desc="Extracting features"):
    if inputs is None: continue
    inputs = inputs.to(device)
    
    with torch.no_grad():
        # SỬ DỤNG forward_features ĐỂ LẤY ĐẶC TRƯNG CHƯA POOLING
        features = extractor.forward_features(inputs) 
        
        if features.dim() == 4:
            B, H, W, C = features.shape
            features = features.view(B, H * W, C)
    
    features_np = features.cpu().numpy()
    
    for feature_seq, path in zip(features_np, out_paths):
        try: 
            # feature_seq lúc này có shape (49, 768)
            np.save(path, feature_seq)
            extracted_count += 1
        except Exception as e: 
            print(f" Lỗi lưu file {path}: {e}")

print("\n" + "="*60)
print(f"HOÀN TẤT TRÍCH XUẤT {extracted_count} SCENE FEATURES (49 TOKENS/ẢNH)")