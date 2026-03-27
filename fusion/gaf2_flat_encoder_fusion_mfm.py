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

# Dataset paths
FACE_FEATURE_DIR       = "/kaggle/input/datasets/huynhthituongviltp22/facial-features-gaf2-dinov2-vits14/facial_features_gaf2_dinov2_vits14"
OBJECT_FEATURE_DIR     = "/kaggle/input/datasets/huynhthituongviltp22/object-features-gaf2-dinov2-vits14/Object_Features_Gaf2_dinov2_vits14"
SCENE_FEATURE_DIR      = "/kaggle/input/datasets/huynhthituongviltp22/scene-features-gaf2-swin-tiny/Scene_Features_Gaf2_swin_tiny"

OUTPUT_DIR = "/kaggle/working"
MODEL_PATH = os.path.join(OUTPUT_DIR, "vilt_ablation_pure_vision_baseline_gaf2.pth")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Labels
NUM_CLASSES = 3
LABEL_MAP = {"Positive": 0, "Negative": 1, "Neutral": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Feature dimensions
FACE_FEAT_DIM = 384
OBJECT_FEAT_DIM = 384
SCENE_FEAT_DIM = 768

# Model config
MAX_FACES = 10
MAX_OBJECTS = 14

D_MODEL = 512
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 2
FFN_DIM = 2048
DROPOUT = 0.4
HEAD_DROPOUT = 0.3

MASK_PROB_FACE   = 0.45
MASK_PROB_OBJECT = 0.15
ALPHA_MFM_FACE   = 2.0
ALPHA_MFM_OBJECT = 1.0

# Training config
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
NUM_EPOCHS = 200
EARLY_STOP_PATIENCE = 15
MIN_IMPROVEMENT = 0.005

# ==========================================
# GAF2-SPECIFIC CONFIG
# ==========================================
# Vì tập test bị giấu nhãn, ta chạy 10 lần train/val
# và chọn model có val acc cao nhất để đánh giá cuối cùng trên val
NUM_RUNS = 10

# ==========================================
# 2. DATASET — RAM Caching
# ==========================================

class MultiModalFeatureDataset(Dataset):
    """
    RAM-Cached dataset. Toàn bộ feats + scene được load và padding vào RAM tại __init__.
    __getitem__ chỉ là list lookup thuần — zero disk I/O khi train.
    Giữ nguyên 6-tuple (không có BBox).
    """

    def __init__(self, split):
        self.split = split

        print(f"\nScanning {split} data...")
        scene_dir   = os.path.join(SCENE_FEATURE_DIR, split)
        raw_samples = []

        for label_name in LABEL_MAP.keys():
            label_dir = os.path.join(scene_dir, label_name)
            if not os.path.exists(label_dir):
                continue
            label = LABEL_MAP[label_name]

            for scene_file in os.listdir(label_dir):
                if not scene_file.endswith('.npy'):
                    continue
                base_name  = os.path.splitext(scene_file)[0]
                scene_path = os.path.join(label_dir, scene_file)

                face_dir   = os.path.join(FACE_FEATURE_DIR, split, label_name)
                face_paths = []
                if os.path.exists(face_dir):
                    for f in os.listdir(face_dir):
                        if f.startswith(f"{base_name}_face_") and f.endswith('.npy'):
                            face_paths.append(os.path.join(face_dir, f))
                if not face_paths:
                    continue

                object_dir   = os.path.join(OBJECT_FEATURE_DIR, split, label_name)
                object_paths = []
                if os.path.exists(object_dir):
                    for f in os.listdir(object_dir):
                        if f.startswith(f"{base_name}_obj_") and f.endswith('.npy'):
                            object_paths.append(os.path.join(object_dir, f))

                raw_samples.append({
                    'faces':   face_paths,
                    'objects': object_paths,
                    'scene':   scene_path,
                    'label':   label,
                })

        print(f"Found {len(raw_samples)} samples. Loading into RAM...")

        self.cache_faces      = []
        self.cache_face_masks = []
        self.cache_objs       = []
        self.cache_obj_masks  = []
        self.cache_scenes     = []
        self.labels           = []

        for s in tqdm(raw_samples, desc=f"  RAM [{split}]"):
            # --- faces ---
            ff = self._load_features(s['faces'], MAX_FACES, FACE_FEAT_DIM)
            num_f  = ff.size(0)
            f_mask = torch.cat([torch.ones(num_f),
                                 torch.zeros(MAX_FACES - num_f)]).bool()
            if num_f < MAX_FACES:
                ff = torch.cat([ff, torch.zeros(MAX_FACES - num_f, FACE_FEAT_DIM)])
            else:
                ff = ff[:MAX_FACES]

            # --- objects ---
            if s['objects']:
                of = self._load_features(s['objects'], MAX_OBJECTS, OBJECT_FEAT_DIM)
            else:
                of = torch.zeros(1, OBJECT_FEAT_DIM)
            num_o  = of.size(0)
            o_mask = torch.cat([torch.ones(num_o),
                                 torch.zeros(MAX_OBJECTS - num_o)]).bool()
            if num_o < MAX_OBJECTS:
                of = torch.cat([of, torch.zeros(MAX_OBJECTS - num_o, OBJECT_FEAT_DIM)])
            else:
                of = of[:MAX_OBJECTS]

            # --- scene ---
            try:
                sc = torch.tensor(np.load(s['scene']), dtype=torch.float32)
                sc = F.normalize(sc, p=2, dim=0)
            except Exception:
                sc = torch.zeros(SCENE_FEAT_DIM)

            self.cache_faces.append(ff)
            self.cache_face_masks.append(f_mask)
            self.cache_objs.append(of)
            self.cache_obj_masks.append(o_mask)
            self.cache_scenes.append(sc)
            self.labels.append(s['label'])

        print(f"  RAM cache done: {len(self.labels)} samples.")

    def __len__(self):
        return len(self.labels)

    def _extract_index(self, path):
        try:
            return int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
        except Exception:
            return 9999

    def _load_features(self, feature_paths, max_tokens, feat_dim):
        """Load, sort, L2-normalise DINOv2 features. Returns (N, D) tensor."""
        feature_paths = sorted(feature_paths, key=self._extract_index)
        if len(feature_paths) > max_tokens:
            feature_paths = feature_paths[:max_tokens]
        features = []
        for path in feature_paths:
            try:
                features.append(torch.tensor(np.load(path), dtype=torch.float32))
            except Exception:
                continue
        if not features:
            return torch.zeros(1, feat_dim)
        return F.normalize(torch.stack(features), p=2, dim=-1)

    def __getitem__(self, idx):
        return (self.cache_faces[idx],
                self.cache_face_masks[idx],
                self.cache_objs[idx],
                self.cache_obj_masks[idx],
                self.cache_scenes[idx],
                self.labels[idx])


def collate_fn_features(batch):
    faces, f_masks, objects, o_masks, scenes, labels = zip(*batch)
    return (torch.stack(faces), torch.stack(f_masks),
            torch.stack(objects), torch.stack(o_masks),
            torch.stack(scenes), torch.tensor(labels, dtype=torch.long))


# ==========================================
# 3. ViLT ARCHITECTURE — Pure Vision Baseline
# ==========================================

class ViLTWithMFM(nn.Module):
    """
    [ABL] Pure Vision Baseline — Xóa hoàn toàn nhánh Text & Decoder.

    THAY ĐỔI so với InfoNCE + AvgText:
      [ABL-1] Xóa: text_proj, semantic_children_raw, logit_scale, decoder
      [ABL-2] score_head đầu ra: nn.Linear(D_MODEL, NUM_CLASSES) thay vì Linear(D_MODEL, 1)
      [ABL-3] Classification trực tiếp từ global_visual_feat (bỏ Decoder)
      [ABL-4] return 6-tuple (bỏ averaged_labels)
    """

    def __init__(self):
        super().__init__()

        # =================================================================
        # A. VISUAL PROJECTIONS (GIỮ NGUYÊN)
        # =================================================================
        self.face_proj   = nn.Linear(FACE_FEAT_DIM,   D_MODEL)
        self.object_proj = nn.Linear(OBJECT_FEAT_DIM, D_MODEL)
        self.scene_proj  = nn.Linear(SCENE_FEAT_DIM,  D_MODEL)

        self.type_embeddings = nn.Embedding(3, D_MODEL)
        self.register_buffer("type_ids", torch.tensor([0, 1, 2]))

        # GIỮ NGUYÊN: random learnable positional embeddings
        self.face_pos_embed    = nn.Parameter(torch.randn(1, MAX_FACES,   D_MODEL))
        self.obj_pos_embed     = nn.Parameter(torch.randn(1, MAX_OBJECTS, D_MODEL))
        self.face_mask_token   = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.object_mask_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))

        # [ABL-1] Xóa: text_proj, semantic_children_raw, logit_scale, decoder

        # =================================================================
        # B. FLAT TRANSFORMER ENCODER (GIỮ NGUYÊN)
        # =================================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=NUM_TRANSFORMER_LAYERS
        )

        self.visual_pooler = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(D_MODEL, D_MODEL)
        )

        # [ABL-1] Xóa self.decoder

        self.layernorm_input = nn.LayerNorm(D_MODEL)
        self.dropout_input   = nn.Dropout(DROPOUT)

        # =================================================================
        # C. PREDICTION HEADS
        # =================================================================
        # [ABL-2] score_head: Linear(D_MODEL, NUM_CLASSES) — phân loại trực tiếp
        self.score_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(D_MODEL, NUM_CLASSES)   # [ABL-2] thay vì Linear(D_MODEL, 1)
        )
        self.face_mfm_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL),
            nn.GELU(), nn.Linear(D_MODEL, FACE_FEAT_DIM)
        )
        self.object_mfm_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL),
            nn.GELU(), nn.Linear(D_MODEL, OBJECT_FEAT_DIM)
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.type_embeddings.weight, std=0.02)
        nn.init.normal_(self.face_mask_token,   std=0.02)
        nn.init.normal_(self.object_mask_token, std=0.02)

    def forward(self, face_feats, face_mask, object_feats, object_mask,
                scene_feat, masking=False):
        """
        Returns (6-tuple):
            logits_cls           (B, NUM_CLASSES)
            logits_face_mfm      (N_masked, FACE_FEAT_DIM) or None
            masked_labels        (N_masked, FACE_FEAT_DIM) or None
            logits_object_mfm    (M_masked, OBJECT_FEAT_DIM) or None
            object_masked_labels (M_masked, OBJECT_FEAT_DIM) or None
            global_visual_feat   (B, D)

        [ABL-4] Bỏ averaged_labels khỏi return tuple
        """
        B      = face_feats.size(0)
        device = face_feats.device

        f_embed = self.face_proj(face_feats)                # (B, 10, D)
        o_embed = self.object_proj(object_feats)            # (B, 14, D)
        s_embed = self.scene_proj(scene_feat).unsqueeze(1)  # (B,  1, D)

        num_faces = f_embed.size(1)

        # ------ MFM Masking (GIỮ NGUYÊN logic + float16 fix) -------------
        bool_masked_pos        = None
        masked_labels          = None
        object_bool_masked_pos = None
        object_masked_labels   = None

        if masking:
            rand     = torch.rand(f_embed.shape[:2], device=device)
            rand_obj = torch.rand(o_embed.shape[:2], device=device)
            face_mask_indices = (rand < MASK_PROB_FACE) & face_mask.bool()
            obj_mask_indices  = (rand_obj < MASK_PROB_OBJECT) & object_mask.bool()

            if face_mask_indices.sum() > 0:
                bool_masked_pos = face_mask_indices
                masked_labels   = face_feats[face_mask_indices]
                mask_tokens_expanded = self.face_mask_token.to(f_embed.dtype).expand(
                    B, f_embed.size(1), -1
                )
                f_embed = f_embed.clone()
                f_embed[face_mask_indices] = mask_tokens_expanded[face_mask_indices]

            if obj_mask_indices.sum() > 0:
                object_bool_masked_pos = obj_mask_indices
                object_masked_labels   = object_feats[obj_mask_indices]
                obj_mask_tokens = self.object_mask_token.to(o_embed.dtype).expand(
                    B, o_embed.size(1), -1
                )
                o_embed = o_embed.clone()
                o_embed[obj_mask_indices] = obj_mask_tokens[obj_mask_indices]

        # ------ Positional + type embeddings (GIỮ NGUYÊN) ----------------
        cls_input = s_embed + self.type_embeddings(self.type_ids[0])
        f_input   = (f_embed
                     + self.face_pos_embed[:, :f_embed.size(1), :]
                     + self.type_embeddings(self.type_ids[1]))
        o_input   = (o_embed
                     + self.obj_pos_embed[:, :o_embed.size(1), :]
                     + self.type_embeddings(self.type_ids[2]))

        x = torch.cat([cls_input, f_input, o_input], dim=1)  # (B, 25, D)
        x = self.dropout_input(self.layernorm_input(x))

        cls_mask     = torch.ones(B, 1, device=device)
        padding_mask = torch.cat([cls_mask, face_mask, object_mask], dim=1) == 0

        # ------ Flat Encoder (GIỮ NGUYÊN) --------------------------------
        memory             = self.transformer(x, src_key_padding_mask=padding_mask)
        global_visual_feat = self.visual_pooler(memory[:, 0, :])  # (B, D)

        # ------ [ABL-3] Classify directly from visual feat ---------------
        logits_cls = self.score_head(global_visual_feat)            # (B, NUM_CLASSES)

        # ------ MFM heads (GIỮ NGUYÊN) -----------------------------------
        logits_face_mfm   = None
        logits_object_mfm = None
        if masking:
            if bool_masked_pos is not None:
                face_output        = memory[:, 1:1 + num_faces, :]
                masked_face_output = face_output[bool_masked_pos]
                logits_face_mfm    = self.face_mfm_head(masked_face_output)

            if object_bool_masked_pos is not None:
                obj_start         = 1 + num_faces
                obj_output        = memory[:, obj_start:obj_start + o_embed.size(1), :]
                masked_obj_output = obj_output[object_bool_masked_pos]
                logits_object_mfm = self.object_mfm_head(masked_obj_output)

        # [ABL-4] 6-tuple — bỏ averaged_labels
        return (logits_cls,
                logits_face_mfm, masked_labels,
                logits_object_mfm, object_masked_labels,
                global_visual_feat)


# ==========================================
# 4. LOSS FUNCTIONS
# ==========================================

class FocalLoss(nn.Module):
    """GIỮ NGUYÊN từ base code 91.06%."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma           = gamma
        self.reduction       = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(inputs, targets, reduction='none',
                                     label_smoothing=self.label_smoothing)
        pt         = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# [ABL] Xóa MFMInfoNCELoss và SAMLoss
# Chỉ giữ FocalLoss. MFM dùng nn.MSELoss() truyền thống.


# ==========================================
# 5. TRAINING & EVALUATION LOOPS
# ==========================================

def train_epoch(model, loader, criterion_cls, criterion_mfm, optimizer):
    # [ABL] Bỏ criterion_sam khỏi tham số
    model.train()
    total_loss = total_cls = total_face_mfm = total_obj_mfm = 0
    correct = total = 0
    scaler  = torch.amp.GradScaler('cuda')

    for batch in loader:
        faces, f_mask, objects, o_mask, scene, labels = [b.to(DEVICE) for b in batch]

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            # [ABL] unpack 6-tuple (bỏ averaged_labels)
            (cls_logits,
             face_mfm_pred,   face_mfm_targets,
             obj_mfm_pred,    obj_mfm_targets,
             visual_feat) = model(
                faces, f_mask, objects, o_mask, scene, masking=True
            )

            # --- Focal Loss (giữ nguyên) ---
            loss_cls = criterion_cls(cls_logits, labels)

            # --- MSE MFM [ABL] ---
            loss_face_mfm = torch.tensor(0.0, device=DEVICE)
            loss_obj_mfm  = torch.tensor(0.0, device=DEVICE)
            if face_mfm_pred is not None and face_mfm_targets is not None:
                loss_face_mfm = criterion_mfm(face_mfm_pred, face_mfm_targets)
            if obj_mfm_pred is not None and obj_mfm_targets is not None:
                loss_obj_mfm  = criterion_mfm(obj_mfm_pred,  obj_mfm_targets)

            # [ABL] Xóa loss_sam — không dùng SAM Loss
            loss = (loss_cls
                    + ALPHA_MFM_FACE   * loss_face_mfm
                    + ALPHA_MFM_OBJECT * loss_obj_mfm)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss     += loss.item()
        total_cls      += loss_cls.item()
        total_face_mfm += loss_face_mfm.item()
        total_obj_mfm  += loss_obj_mfm.item()

        _, pred = torch.max(cls_logits, 1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    print(f"  CLS={total_cls/n:.4f} | "
          f"FaceMFM(MSE)={total_face_mfm/n:.4f} | "
          f"ObjMFM(MSE)={total_obj_mfm/n:.4f}")
    return total_cls / n, 100 * correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            faces, f_mask, objects, o_mask, scene, labels = [b.to(DEVICE) for b in batch]
            # [ABL] 6-tuple unpack
            (cls_logits, _, _, _, _, _) = model(
                faces, f_mask, objects, o_mask, scene, masking=False
            )
            loss        = criterion(cls_logits, labels)
            total_loss += loss.item()
            _, pred     = torch.max(cls_logits, 1)
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
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    path = os.path.join(save_dir, "confusion_matrix_gaf2.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {path}")


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    plt.figure(figsize=(12, 5))

    # ----- LOSS -----
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.legend()
    plt.title('Total Loss (CLS + MSE MFM)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # ----- ACCURACY -----
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs,   label='Val Acc')
    plt.legend()
    plt.title('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    path = os.path.join(save_dir, "training_history_gaf2.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved training history → {path}")

# ==========================================
# 7. MAIN (GAF2 — 10 RUNS PROTOCOL)
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  ViLT Ablation — Pure Vision Baseline (MSE MFM, No Text, No SAM)")
    print(f"  Dataset: GAF2 | Protocol: {NUM_RUNS} runs, best val acc → final eval on val")
    print("=" * 65)

    # ---- 1. Load Data 1 lần duy nhất (chỉ train và val, không có test) --
    train_ds = MultiModalFeatureDataset('Train')
    val_ds   = MultiModalFeatureDataset('Val')

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_features,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn_features,
                              num_workers=0, pin_memory=True)

    # ---- 2. Losses -------------------------------------------------------
    class_weights = torch.tensor([1.0, 1.2, 1.8]).to(DEVICE)
    criterion_cls = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    criterion_mfm = nn.MSELoss()

    # ---- 3. 10-Run Loop --------------------------------------------------
    global_best_val_acc    = 0.0
    global_best_model_state = None
    all_run_val_accs       = []
    best_run               = 1

    # Lưu history của best run để vẽ biểu đồ
    best_run_train_losses, best_run_val_losses = [], []
    best_run_train_accs,   best_run_val_accs   = [], []

    for run_idx in range(NUM_RUNS):
        print("\n" + "=" * 65)
        print(f"  RUN {run_idx + 1}/{NUM_RUNS}")
        print("=" * 65)

        # Khởi tạo lại Model + Optimizer cho mỗi run (seed ngẫu nhiên tự nhiên)
        model = ViLTWithMFM().to(DEVICE)

        if run_idx == 0:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable parameters: {total_params:,}")

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        best_val_acc_this_run    = 0.0
        best_model_state_this_run = None
        patience_counter         = 0
        run_train_losses, run_val_losses = [], []
        run_train_accs,   run_val_accs   = [], []

        for epoch in range(NUM_EPOCHS):
            print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}")

            t_loss, t_acc = train_epoch(
                model, train_loader,
                criterion_cls, criterion_mfm, optimizer
            )
            v_loss, v_acc, _, _ = eval_epoch(model, val_loader, criterion_cls)

            run_train_losses.append(t_loss); run_val_losses.append(v_loss)
            run_train_accs.append(t_acc);   run_val_accs.append(v_acc)
            scheduler.step(v_loss)

            print(f"  Train → Loss {t_loss:.4f}  Acc {t_acc:.2f}%  |  "
                  f"Val → Loss {v_loss:.4f}  Acc {v_acc:.2f}%")

            if v_acc > best_val_acc_this_run + MIN_IMPROVEMENT:
                best_val_acc_this_run    = v_acc
                best_model_state_this_run = copy.deepcopy(model.state_dict())
                patience_counter         = 0
                print("  >>> New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print("  Early stopping triggered.")
                    break

        all_run_val_accs.append(best_val_acc_this_run)
        print(f"\n  => Run {run_idx + 1} best val acc: {best_val_acc_this_run:.2f}%")

        # Cập nhật global best qua các runs
        if best_val_acc_this_run > global_best_val_acc:
            global_best_val_acc    = best_val_acc_this_run
            global_best_model_state = copy.deepcopy(best_model_state_this_run)
            best_run               = run_idx + 1
            best_run_train_losses  = run_train_losses
            best_run_val_losses    = run_val_losses
            best_run_train_accs    = run_train_accs
            best_run_val_accs      = run_val_accs

    # ---- 4. Đánh giá cuối cùng trên VAL với best model ------------------
    print("\n" + "=" * 65)
    print("  FINAL EVALUATION ON VAL SET (BEST MODEL)")
    print("=" * 65)
    print(f"  Global best val acc qua {NUM_RUNS} runs: {global_best_val_acc:.2f}% (run {best_run})")
    print(f"  Mean val acc: {np.mean(all_run_val_accs):.2f}% ± {np.std(all_run_val_accs):.2f}%")
    print(f"  All runs: {[f'{a:.2f}%' for a in all_run_val_accs]}")

    final_model = ViLTWithMFM().to(DEVICE)
    final_model.load_state_dict(global_best_model_state)
    torch.save(final_model.state_dict(), MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")

    _, final_val_acc, preds, labels_val = eval_epoch(
        final_model, val_loader, criterion_cls
    )
    print(f"\n  Final Val Accuracy: {final_val_acc:.2f}%")
    print("\n  Classification Report (trên VAL):")
    print(classification_report(
        labels_val, preds,
        target_names=[INV_LABEL_MAP[i] for i in range(NUM_CLASSES)],
        digits=4
    ))

    plot_confusion_matrix(labels_val, preds, OUTPUT_DIR)
    plot_training_history(
        best_run_train_losses, best_run_val_losses,
        best_run_train_accs,   best_run_val_accs,
        OUTPUT_DIR
    )

    print(f"\n  Done! Results saved to {OUTPUT_DIR}")