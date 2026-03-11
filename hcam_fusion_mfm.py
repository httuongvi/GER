import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import copy
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================

FACE_FEATURE_DIR       = "/kaggle/input/datasets/wawuwaa/vi-feature-groupemow/Vi_Feature_GroupEmoW/facial_features_groupemow_dinov2_small"
OBJECT_FEATURE_DIR     = "/kaggle/input/datasets/wawuwaa/vi-feature-groupemow/Vi_Feature_GroupEmoW/Object_Features_GroupEmoW_dinov2_small"
SCENE_FEATURE_DIR      = "/kaggle/input/datasets/wawuwaa/vi-feature-groupemow/Vi_Feature_GroupEmoW/Scene_Features_GroupEmoW_swin_tiny"

OUTPUT_DIR = "/kaggle/working"
MODEL_PATH = os.path.join(OUTPUT_DIR, "vilt_hcam_exact_paper_logic.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_CLASSES = 3
LABEL_MAP = {"Positive": 0, "Negative": 1, "Neutral": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

FACE_FEAT_DIM = 384
OBJECT_FEAT_DIM = 384
SCENE_FEAT_DIM = 768

MAX_FACES = 10
MAX_OBJECTS = 14

D_MODEL = 512
NUM_HEADS = 8
FFN_DIM = 2048
DROPOUT = 0.4
HEAD_DROPOUT = 0.3

MASK_PROB_FACE   = 0.3
MASK_PROB_OBJECT = 0.15
ALPHA_MFM_FACE   = 1.0
ALPHA_MFM_OBJECT = 0.5

BATCH_SIZE = 64
LEARNING_RATE = 5e-5
NUM_EPOCHS = 200
EARLY_STOP_PATIENCE = 15
MIN_IMPROVEMENT = 0.005


# ==========================================
# 2. DATASET — RAM Caching
# ==========================================
class MultiModalFeatureDataset(Dataset):
    def __init__(self, split):
        self.split = split
        print(f"\nScanning {split} data...")
        scene_dir   = os.path.join(SCENE_FEATURE_DIR, split)
        raw_samples = []

        for label_name in LABEL_MAP.keys():
            label_dir = os.path.join(scene_dir, label_name)
            if not os.path.exists(label_dir): continue
            label = LABEL_MAP[label_name]

            for scene_file in os.listdir(label_dir):
                if not scene_file.endswith('.npy'): continue
                base_name  = os.path.splitext(scene_file)[0]
                scene_path = os.path.join(label_dir, scene_file)

                face_dir   = os.path.join(FACE_FEATURE_DIR, split, label_name)
                face_paths = []
                if os.path.exists(face_dir):
                    for f in os.listdir(face_dir):
                        if f.startswith(f"{base_name}_face_") and f.endswith('.npy'):
                            face_paths.append(os.path.join(face_dir, f))
                if not face_paths: continue

                object_dir   = os.path.join(OBJECT_FEATURE_DIR, split, label_name)
                object_paths = []
                if os.path.exists(object_dir):
                    for f in os.listdir(object_dir):
                        if f.startswith(f"{base_name}_obj_") and f.endswith('.npy'):
                            object_paths.append(os.path.join(object_dir, f))

                raw_samples.append({
                    'faces': face_paths, 'objects': object_paths,
                    'scene': scene_path, 'label': label,
                })

        self.cache_faces, self.cache_face_masks = [], []
        self.cache_objs, self.cache_obj_masks   = [], []
        self.cache_scenes, self.labels          = [], []

        for s in tqdm(raw_samples, desc=f"  RAM [{split}]"):
            ff = self._load_features(s['faces'], MAX_FACES, FACE_FEAT_DIM)
            num_f  = ff.size(0)
            f_mask = torch.cat([torch.ones(num_f), torch.zeros(MAX_FACES - num_f)]).bool()
            if num_f < MAX_FACES: ff = torch.cat([ff, torch.zeros(MAX_FACES - num_f, FACE_FEAT_DIM)])
            else: ff = ff[:MAX_FACES]

            if s['objects']: of = self._load_features(s['objects'], MAX_OBJECTS, OBJECT_FEAT_DIM)
            else: of = torch.zeros(1, OBJECT_FEAT_DIM)
            num_o  = of.size(0)
            o_mask = torch.cat([torch.ones(num_o), torch.zeros(MAX_OBJECTS - num_o)]).bool()
            if num_o < MAX_OBJECTS: of = torch.cat([of, torch.zeros(MAX_OBJECTS - num_o, OBJECT_FEAT_DIM)])
            else: of = of[:MAX_OBJECTS]

            try:
                sc = torch.tensor(np.load(s['scene']), dtype=torch.float32)
                sc = F.normalize(sc, p=2, dim=0)
            except: sc = torch.zeros(SCENE_FEAT_DIM)

            self.cache_faces.append(ff)
            self.cache_face_masks.append(f_mask)
            self.cache_objs.append(of)
            self.cache_obj_masks.append(o_mask)
            self.cache_scenes.append(sc)
            self.labels.append(s['label'])

    def __len__(self): return len(self.labels)

    def _extract_index(self, path):
        try: return int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
        except: return 9999

    def _load_features(self, feature_paths, max_tokens, feat_dim):
        feature_paths = sorted(feature_paths, key=self._extract_index)
        if len(feature_paths) > max_tokens: feature_paths = feature_paths[:max_tokens]
        features = []
        for path in feature_paths:
            try: features.append(torch.tensor(np.load(path), dtype=torch.float32))
            except: continue
        if not features: return torch.zeros(1, feat_dim)
        return F.normalize(torch.stack(features), p=2, dim=-1)

    def __getitem__(self, idx):
        return (self.cache_faces[idx], self.cache_face_masks[idx],
                self.cache_objs[idx], self.cache_obj_masks[idx],
                self.cache_scenes[idx], self.labels[idx])

def collate_fn_features(batch):
    faces, f_masks, objects, o_masks, scenes, labels = zip(*batch)
    return (torch.stack(faces), torch.stack(f_masks),
            torch.stack(objects), torch.stack(o_masks),
            torch.stack(scenes), torch.tensor(labels, dtype=torch.long))


# ==========================================
# 3. ViLT ARCHITECTURE — HCAM EXACT PAPER LOGIC
# ==========================================

class ViLTWithMFM(nn.Module):
    def __init__(self):
        super().__init__()

        self.face_proj   = nn.Linear(FACE_FEAT_DIM,   D_MODEL)
        self.object_proj = nn.Linear(OBJECT_FEAT_DIM, D_MODEL)
        self.scene_proj  = nn.Linear(SCENE_FEAT_DIM,  D_MODEL)

        self.type_embeddings = nn.Embedding(3, D_MODEL)
        self.register_buffer("type_ids", torch.tensor([0, 1, 2]))

        self.face_pos_embed    = nn.Parameter(torch.randn(1, MAX_FACES,   D_MODEL))
        self.obj_pos_embed     = nn.Parameter(torch.randn(1, MAX_OBJECTS, D_MODEL))
        self.face_mask_token   = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.object_mask_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))

        # STAGE I: INTRA-MODALITY
        self.face_encoder = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FFN_DIM, dropout=DROPOUT, batch_first=True
        )
        self.obj_encoder = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FFN_DIM, dropout=DROPOUT, batch_first=True
        )

        # STAGE I PREDICTORS (Chỉ có Face và Object, không có Scene)
        self.pred_f1 = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))
        self.pred_o1 = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))

        # STAGE II: STRICT CROSS-ATTENTION + REFINE
        self.face_cross  = nn.MultiheadAttention(D_MODEL, NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.obj_cross   = nn.MultiheadAttention(D_MODEL, NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.scene_cross = nn.MultiheadAttention(D_MODEL, NUM_HEADS, dropout=DROPOUT, batch_first=True)

        self.face_refine = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FFN_DIM, dropout=DROPOUT, batch_first=True
        )
        self.obj_refine = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FFN_DIM, dropout=DROPOUT, batch_first=True
        )
        self.scene_refine = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU()
        )

        # STAGE II PREDICTORS (Chỉ có Face và Object, không có Scene)
        self.pred_f2 = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))
        self.pred_o2 = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))

        # STAGE III: FINAL FUSION & PREDICTION
        self.fc_fusion_feat = nn.Sequential(
            nn.Linear(3 * D_MODEL, D_MODEL), 
            nn.LayerNorm(D_MODEL), 
            nn.GELU()
        )
        self.pred_c = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))

        # MFM HEADS
        self.face_mfm_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU(), nn.Linear(D_MODEL, FACE_FEAT_DIM)
        )
        self.object_mfm_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU(), nn.Linear(D_MODEL, OBJECT_FEAT_DIM)
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        nn.init.normal_(self.type_embeddings.weight, std=0.02)
        nn.init.normal_(self.face_mask_token,   std=0.02)
        nn.init.normal_(self.object_mask_token, std=0.02)

    def _masked_mean(self, seq, pad_mask):
        valid_mask = (~pad_mask).float().unsqueeze(-1)
        sum_feat = (seq * valid_mask).sum(dim=1)
        count = valid_mask.sum(dim=1).clamp(min=1e-6)
        return sum_feat / count

    def forward(self, face_feats, face_mask, object_feats, object_mask, scene_feat, masking=False):
        B = face_feats.size(0)
        device = face_feats.device

        f_embed = self.face_proj(face_feats)
        o_embed = self.object_proj(object_feats)
        s_embed = self.scene_proj(scene_feat).unsqueeze(1)
        
        bool_masked_pos, masked_labels = None, None
        object_bool_masked_pos, object_masked_labels = None, None

        if masking:
            rand = torch.rand(f_embed.shape[:2], device=device)
            rand_obj = torch.rand(o_embed.shape[:2], device=device)
            face_mask_indices = (rand < MASK_PROB_FACE) & face_mask.bool()
            obj_mask_indices  = (rand_obj < MASK_PROB_OBJECT) & object_mask.bool()

            if face_mask_indices.sum() > 0:
                bool_masked_pos = face_mask_indices
                masked_labels   = face_feats[face_mask_indices]
                mask_tokens_expanded = self.face_mask_token.to(f_embed.dtype).expand(B, f_embed.size(1), -1)
                f_embed = f_embed.clone()
                f_embed[face_mask_indices] = mask_tokens_expanded[face_mask_indices]

            if obj_mask_indices.sum() > 0:
                object_bool_masked_pos = obj_mask_indices
                object_masked_labels   = object_feats[obj_mask_indices]
                obj_mask_tokens = self.object_mask_token.to(o_embed.dtype).expand(B, o_embed.size(1), -1)
                o_embed = o_embed.clone()
                o_embed[obj_mask_indices] = obj_mask_tokens[obj_mask_indices]

        cls_input = s_embed + self.type_embeddings(self.type_ids[0])
        f_input   = f_embed + self.face_pos_embed[:, :f_embed.size(1), :] + self.type_embeddings(self.type_ids[1])
        o_input   = o_embed + self.obj_pos_embed[:, :o_embed.size(1), :] + self.type_embeddings(self.type_ids[2])

        f_pad_mask = (face_mask == 0)
        o_pad_mask = (object_mask == 0)
        cls_pad_mask = torch.zeros(B, 1, device=device, dtype=torch.bool)

        # STAGE I
        f_s1 = self.face_encoder(f_input, src_key_padding_mask=f_pad_mask)
        o_s1 = self.obj_encoder(o_input, src_key_padding_mask=o_pad_mask)
        s_s1 = cls_input

        f_pool1 = self._masked_mean(f_s1, f_pad_mask)
        o_pool1 = self._masked_mean(o_s1, o_pad_mask)

        # Chỉ có Face và Object dự đoán ở Stage 1
        p1_f = self.pred_f1(f_pool1)
        p1_o = self.pred_o1(o_pool1)

        # STAGE II: Strict Cross-Attention
        mem_f = torch.cat([s_s1, o_s1], dim=1)
        mem_pad_mask_f = torch.cat([cls_pad_mask, o_pad_mask], dim=1)
        f_c2, _ = self.face_cross(query=f_s1, key=mem_f, value=mem_f, key_padding_mask=mem_pad_mask_f)

        mem_o = torch.cat([s_s1, f_s1], dim=1)
        mem_pad_mask_o = torch.cat([cls_pad_mask, f_pad_mask], dim=1)
        o_c2, _ = self.obj_cross(query=o_s1, key=mem_o, value=mem_o, key_padding_mask=mem_pad_mask_o)

        mem_s = torch.cat([f_s1, o_s1], dim=1)
        mem_pad_mask_s = torch.cat([f_pad_mask, o_pad_mask], dim=1)
        s_c2, _ = self.scene_cross(query=s_s1, key=mem_s, value=mem_s, key_padding_mask=mem_pad_mask_s)

        f_r2 = self.face_refine(f_c2, src_key_padding_mask=f_pad_mask)
        o_r2 = self.obj_refine(o_c2, src_key_padding_mask=o_pad_mask)
        s_r2 = self.scene_refine(s_c2)

        f_pool2 = self._masked_mean(f_r2, f_pad_mask)
        o_pool2 = self._masked_mean(o_r2, o_pad_mask)
        s_pool2 = s_r2.squeeze(1)

        # Chỉ có Face và Object dự đoán ở Stage 2
        p2_f = self.pred_f2(f_pool2)
        p2_o = self.pred_o2(o_pool2)

        # STAGE III: FUSION
        fused = torch.cat([f_pool2, o_pool2, s_pool2], dim=-1)
        feat_c = self.fc_fusion_feat(fused)
        
        p_c = self.pred_c(feat_c)

        logits_face_mfm, logits_object_mfm = None, None
        if masking:
            if bool_masked_pos is not None:
                logits_face_mfm = self.face_mfm_head(f_r2[bool_masked_pos])
            if object_bool_masked_pos is not None:
                logits_object_mfm = self.object_mfm_head(o_r2[object_bool_masked_pos])

        # Trả về 5 predictions
        preds = (p1_f, p1_o, p2_f, p2_o, p_c)

        return preds, logits_face_mfm, masked_labels, logits_object_mfm, object_masked_labels


# ==========================================
# 4. LOSS FUNCTIONS
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma           = gamma
        self.reduction       = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else: self.alpha = None

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt         = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None: focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss


# ==========================================
# 5. TRAINING & EVALUATION LOOPS
# ==========================================
def train_epoch(model, loader, criterion_cls, criterion_mfm, optimizer):
    model.train()
    total_loss = total_cls = total_face_mfm = total_obj_mfm = 0
    correct = total = 0
    scaler  = torch.amp.GradScaler('cuda')

    for batch in tqdm(loader, desc="Training"):
        faces, f_mask, objects, o_mask, scene, labels = [b.to(DEVICE) for b in batch]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            preds, face_mfm_pred, face_mfm_targets, obj_mfm_pred, obj_mfm_targets = model(
                faces, f_mask, objects, o_mask, scene, masking=True
            )
            p1_f, p1_o, p2_f, p2_o, p_c = preds

            # 1. HCAM Loss: SIMPLE SUM (Không dùng weight)
            loss_cls = (
                criterion_cls(p1_f, labels) +
                criterion_cls(p1_o, labels) +
                criterion_cls(p2_f, labels) +
                criterion_cls(p2_o, labels) +
                criterion_cls(p_c, labels)
            )
            
            # 2. MSE MFM
            loss_face_mfm = torch.tensor(0.0, device=DEVICE)
            loss_obj_mfm  = torch.tensor(0.0, device=DEVICE)
            if face_mfm_pred is not None and face_mfm_targets is not None:
                loss_face_mfm = criterion_mfm(face_mfm_pred, face_mfm_targets)
            if obj_mfm_pred is not None and obj_mfm_targets is not None:
                loss_obj_mfm  = criterion_mfm(obj_mfm_pred,  obj_mfm_targets)

            loss = loss_cls + ALPHA_MFM_FACE * loss_face_mfm + ALPHA_MFM_OBJECT * loss_obj_mfm

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss     += loss.item()
        total_cls      += loss_cls.item()
        total_face_mfm += loss_face_mfm.item()
        total_obj_mfm  += loss_obj_mfm.item()

        # Trong Training, chỉ dùng bộ phân loại Fusion chính (p_c) để track Accuracy
        # (Không dùng ensemble trong training theo đúng paper)
        _, pred = torch.max(p_c, 1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    print(f"  CLS(Sum_5_Heads)={total_cls/n:.4f} | FaceMFM={total_face_mfm/n:.4f} | ObjMFM={total_obj_mfm/n:.4f}")
    return total_cls / n, 100 * correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    # Tham số Ensemble HCAM (Eq 9, 10, 11)
    ALPHA_STAGE1_TO_2 = 0.4
    ALPHA_FUSION = 0.5
    ALPHA_FACE = 0.25
    ALPHA_OBJ = 0.25

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            faces, f_mask, objects, o_mask, scene, labels = [b.to(DEVICE) for b in batch]
            
            preds, _, _, _, _ = model(faces, f_mask, objects, o_mask, scene, masking=False)
            p1_f, p1_o, p2_f, p2_o, p_c = preds
            
            # --- BƯỚC 1: CHUYỂN TOÀN BỘ LOGITS THÀNH PROBABILITIES ---
            prob1_f = F.softmax(p1_f, dim=1)
            prob1_o = F.softmax(p1_o, dim=1)
            prob2_f = F.softmax(p2_f, dim=1)
            prob2_o = F.softmax(p2_o, dim=1)
            prob_c  = F.softmax(p_c, dim=1)
            
            # --- BƯỚC 2: HCAM INFERENCE ENSEMBLE (TRÊN KHÔNG GIAN XÁC SUẤT) ---
            # Combine Stage 1 và Stage 2 cho từng modality
            y2_f_prob = (1 - ALPHA_STAGE1_TO_2) * prob2_f + ALPHA_STAGE1_TO_2 * prob1_f
            y2_o_prob = (1 - ALPHA_STAGE1_TO_2) * prob2_o + ALPHA_STAGE1_TO_2 * prob1_o
            
            # Combine tất cả vào Probabilities cuối cùng
            p_final = ALPHA_FUSION * prob_c + ALPHA_FACE * y2_f_prob + ALPHA_OBJ * y2_o_prob
            
            # --- BƯỚC 3: TÍNH LOSS VÀ ACCURACY ---
            # Vì criterion_cls (FocalLoss) mong đợi logits, ta bọc log() quanh p_final
            # để hàm CrossEntropy bên trong tự triệt tiêu và tính ra chính xác giá trị toán học
            loss_logits = torch.log(p_final + 1e-8)
            loss        = criterion(loss_logits, labels)
            
            total_loss += loss.item()
            
            # Lấy class có xác suất cao nhất
            _, pred     = torch.max(p_final, 1)
            correct    += (pred == labels).sum().item()
            total      += labels.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), 100 * correct / total, all_preds, all_labels

# ==========================================
# 6. VISUALIZATION
# ==========================================
def plot_confusion_matrix(labels_true, preds, save_dir):
    cm = confusion_matrix(labels_true, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    class_names = [INV_LABEL_MAP[i] for i in range(NUM_CLASSES)]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss (Sum of 5)')
    plt.plot(val_losses,   label='Val Loss (Ensemble)')
    plt.legend()
    plt.title('Classification Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc (Fusion Head)')
    plt.plot(val_accs,   label='Val Acc (Ensemble)')
    plt.legend()
    plt.title('Accuracy (%)')
    path = os.path.join(save_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  ViLT - EXACT HCAM (Simple Sum Loss, Inference Ensemble)")
    print("=" * 65)

    train_ds = MultiModalFeatureDataset('train')
    val_ds   = MultiModalFeatureDataset('val')
    test_ds  = MultiModalFeatureDataset('test')

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn_features, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate_fn_features, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate_fn_features, num_workers=0, pin_memory=True)

    print(f"\nInitializing ViLTWithMFM (Exact HCAM Paper Logic)...")
    model = ViLTWithMFM().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    class_weights = torch.tensor([1.0, 1.2, 1.5]).to(DEVICE)
    criterion_cls = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    criterion_mfm = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        t_loss, t_acc = train_epoch(model, train_loader, criterion_cls, criterion_mfm, optimizer)
        v_loss, v_acc, _, _ = eval_epoch(model, val_loader, criterion_cls)

        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc); val_accs.append(v_acc)
        scheduler.step(v_loss)

        print(f"  Train → Loss {t_loss:.4f}  Acc {t_acc:.2f}%  |  Val → Loss {v_loss:.4f}  Acc {v_acc:.2f}%")

        if v_acc > best_val_acc + MIN_IMPROVEMENT:
            best_val_acc = v_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("  >>> New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("  Early stopping triggered.")
                break

    print("\n" + "=" * 65)
    print("  TESTING BEST MODEL")
    print("=" * 65)
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), MODEL_PATH)
    test_loss, test_acc, preds, labels_test = eval_epoch(model, test_loader, criterion_cls)
    print(f"\n  Final Test Accuracy: {test_acc:.2f}%")
    print(classification_report(labels_test, preds, target_names=[INV_LABEL_MAP[i] for i in range(NUM_CLASSES)]))
    plot_confusion_matrix(labels_test, preds, OUTPUT_DIR)
    plot_training_history(train_losses, val_losses, train_accs, val_accs, OUTPUT_DIR)