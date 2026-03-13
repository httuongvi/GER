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
SCENE_FEATURE_DIR      = "/kaggle/input/full-scene-features-groupemow-swin-tiny/Scene_Features_GroupEmoW_swin_tiny_patch49"

OUTPUT_DIR = "/kaggle/working"
MODEL_PATH = os.path.join(OUTPUT_DIR, "vilt_hcam_attention_masking.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_CLASSES = 3
LABEL_MAP = {"Positive": 0, "Negative": 1, "Neutral": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

FACE_FEAT_DIM   = 384
OBJECT_FEAT_DIM = 384
SCENE_FEAT_DIM  = 768
NUM_SCENE_PATCHES = 49   

MAX_FACES   = 10
MAX_OBJECTS = 14

D_MODEL  = 512
NUM_HEADS = 8
FFN_DIM   = 2048
DROPOUT   = 0.4
HEAD_DROPOUT = 0.3

# Cân bằng Loss thủ công (Đã xóa Sparsity)
ALPHA_MFM_START = 0.5   
ALPHA_MFM_END   = 0.05  

W_MFM_FACE  = 1.0   # Khuôn mặt là quan trọng nhất
W_MFM_OBJ   = 0.5   # Vật thể hỗ trợ ngữ cảnh
W_MFM_SCENE = 0.5   # Bối cảnh hỗ trợ không gian

# Kỷ luật thép: Luôn che Top-3 token quan trọng nhất
TOP_K_FACE = 3
TOP_K_OBJ  = 3
TOP_K_SCENE = 15

BATCH_SIZE         = 64
LEARNING_RATE      = 5e-5
NUM_EPOCHS         = 200
EARLY_STOP_PATIENCE = 20
MIN_IMPROVEMENT    = 0.005

# ==========================================
# 2. DATASET (Giữ nguyên)
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
                assert sc.shape == (NUM_SCENE_PATCHES, SCENE_FEAT_DIM)
                sc = F.normalize(sc, p=2, dim=-1)
            except Exception as e:
                sc = torch.zeros(NUM_SCENE_PATCHES, SCENE_FEAT_DIM)

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
# 3. ViLT ARCHITECTURE (ATTENTION-GUIDED MASKING)
# ==========================================
class AttnExtractorLayer(nn.Module):
    """Custom Transformer Layer để trích xuất Attention Weights cho việc Masking"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        # Trích xuất attn_weights: shape (B, L, L)
        src2, attn_weights = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class ViLTWithMFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_proj   = nn.Linear(FACE_FEAT_DIM,   D_MODEL)
        self.object_proj = nn.Linear(OBJECT_FEAT_DIM, D_MODEL)
        self.scene_proj  = nn.Linear(SCENE_FEAT_DIM,  D_MODEL)

        # Mask Tokens
        self.face_mask_token   = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.object_mask_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.scene_mask_token  = nn.Parameter(torch.zeros(1, 1, D_MODEL))

        # --- STAGE I (Sử dụng Custom Layer để lấy Attention) ---
        self.face_encoder = AttnExtractorLayer(D_MODEL, NUM_HEADS, FFN_DIM, DROPOUT)
        self.obj_encoder  = AttnExtractorLayer(D_MODEL, NUM_HEADS, FFN_DIM, DROPOUT)
        self.scene_encoder = AttnExtractorLayer(D_MODEL, NUM_HEADS, FFN_DIM, DROPOUT)

        self.pred_f1 = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))
        self.pred_o1 = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))

        # --- STAGE II ---
        self.face_cross  = nn.MultiheadAttention(D_MODEL, NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.obj_cross   = nn.MultiheadAttention(D_MODEL, NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.scene_cross = nn.MultiheadAttention(D_MODEL, NUM_HEADS, dropout=DROPOUT, batch_first=True)

        self.face_refine  = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FFN_DIM, dropout=DROPOUT, batch_first=True)
        self.obj_refine   = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FFN_DIM, dropout=DROPOUT, batch_first=True)
        self.scene_refine = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FFN_DIM, dropout=DROPOUT, batch_first=True)

        self.pred_f2 = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))
        self.pred_o2 = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))

        # --- STAGE III ---
        self.fc_fusion_feat = nn.Sequential(nn.Linear(3 * D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU())
        self.pred_c = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(D_MODEL, NUM_CLASSES))

        # --- MFM HEADS ---
        self.face_mfm_head   = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU(), nn.Linear(D_MODEL, FACE_FEAT_DIM))
        self.object_mfm_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU(), nn.Linear(D_MODEL, OBJECT_FEAT_DIM))
        self.scene_mfm_head  = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU(), nn.Linear(D_MODEL, SCENE_FEAT_DIM))

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        nn.init.normal_(self.face_mask_token,   std=0.02)
        nn.init.normal_(self.object_mask_token, std=0.02)
        nn.init.normal_(self.scene_mask_token,  std=0.02)

    def _masked_mean(self, seq, pad_mask):
        valid_mask = (~pad_mask).float().unsqueeze(-1)
        sum_feat   = (seq * valid_mask).sum(dim=1)
        count      = valid_mask.sum(dim=1).clamp(min=1e-6)
        return sum_feat / count

    def _forward_clean_stage2_3(self, f_s1, f_pad_mask, o_s1, o_pad_mask, s_s1, s_pad_mask):
        # STAGE II
        mem_f = torch.cat([s_s1, o_s1], dim=1)
        mem_pad_mask_f = torch.cat([s_pad_mask, o_pad_mask], dim=1)
        f_c2, _ = self.face_cross(query=f_s1, key=mem_f, value=mem_f, key_padding_mask=mem_pad_mask_f)

        mem_o = torch.cat([s_s1, f_s1], dim=1)
        mem_pad_mask_o = torch.cat([s_pad_mask, f_pad_mask], dim=1)
        o_c2, _ = self.obj_cross(query=o_s1, key=mem_o, value=mem_o, key_padding_mask=mem_pad_mask_o)

        mem_s = torch.cat([f_s1, o_s1], dim=1)
        mem_pad_mask_s = torch.cat([f_pad_mask, o_pad_mask], dim=1)
        s_c2, _ = self.scene_cross(query=s_s1, key=mem_s, value=mem_s, key_padding_mask=mem_pad_mask_s)

        f_r2 = self.face_refine(f_c2, src_key_padding_mask=f_pad_mask)
        o_r2 = self.obj_refine(o_c2,  src_key_padding_mask=o_pad_mask)
        s_r2 = self.scene_refine(s_c2, src_key_padding_mask=s_pad_mask)

        f_pool2 = self._masked_mean(f_r2, f_pad_mask)
        o_pool2 = self._masked_mean(o_r2, o_pad_mask)
        s_pool2 = s_r2.mean(dim=1)

        p2_f = self.pred_f2(f_pool2)
        p2_o = self.pred_o2(o_pool2)

        # STAGE III (Chỉ gọi cho Luồng Sạch)
        fused  = torch.cat([f_pool2, o_pool2, s_pool2], dim=-1)
        feat_c = self.fc_fusion_feat(fused)
        p_c    = self.pred_c(feat_c)

        return p2_f, p2_o, p_c


    def forward(self, face_feats, face_mask, object_feats, object_mask, scene_feat, masking=False):
        B, device = face_feats.size(0), face_feats.device

        # 1. Chiếu đặc trưng
        f_embed_raw = self.face_proj(face_feats)
        o_embed_raw = self.object_proj(object_feats)
        s_embed_raw = self.scene_proj(scene_feat)

        f_pad_mask = (face_mask == 0)
        o_pad_mask = (object_mask == 0)
        s_pad_mask = torch.zeros(B, NUM_SCENE_PATCHES, device=device, dtype=torch.bool)

        # ==========================================
        # LUỒNG 1: CLEAN BRANCH (Đồng thời lấy Attention)
        # ==========================================
        # STAGE I
        f_s1_clean, attn_f = self.face_encoder(f_embed_raw, f_pad_mask)
        o_s1_clean, attn_o = self.obj_encoder(o_embed_raw, o_pad_mask)
        s_s1_clean, attn_s = self.scene_encoder(s_embed_raw, s_pad_mask)

        f_pool1 = self._masked_mean(f_s1_clean, f_pad_mask)
        o_pool1 = self._masked_mean(o_s1_clean, o_pad_mask)
        p1_f = self.pred_f1(f_pool1)
        p1_o = self.pred_o1(o_pool1)

        # STAGE II & III (Dự đoán thuần túy)
        p2_f, p2_o, p_c = self._forward_clean_stage2_3(
            f_s1_clean, f_pad_mask, o_s1_clean, o_pad_mask, s_s1_clean, s_pad_mask
        )
        preds_clean = (p1_f, p1_o, p2_f, p2_o, p_c)

        # ==========================================
        # LUỒNG 2: MASKED BRANCH
        # ==========================================
        logits_f_mfm, logits_o_mfm, logits_s_mfm = None, None, None
        masked_labels_f, masked_labels_o, masked_labels_s = None, None, None

        if masking:
            # --- TÌM TOP-K ---
            imp_f = attn_f.sum(dim=1).masked_fill(f_pad_mask, -1e9)
            imp_o = attn_o.sum(dim=1).masked_fill(o_pad_mask, -1e9)
            imp_s = attn_s.sum(dim=1).masked_fill(s_pad_mask, -1e9)

            _, topk_f = torch.topk(imp_f, k=TOP_K_FACE, dim=-1)
            _, topk_o = torch.topk(imp_o, k=TOP_K_OBJ, dim=-1)
            _, topk_s = torch.topk(imp_s, k=TOP_K_SCENE, dim=-1)

            # Lọc Mask an toàn
            mask_idx_f = torch.zeros_like(imp_f, dtype=torch.bool).scatter_(1, topk_f, True) & (~f_pad_mask)
            mask_idx_o = torch.zeros_like(imp_o, dtype=torch.bool).scatter_(1, topk_o, True) & (~o_pad_mask)
            mask_idx_s = torch.zeros_like(imp_s, dtype=torch.bool).scatter_(1, topk_s, True) & (~s_pad_mask)

            # --- TẠO QUERY BỊ MASK ---
            f_embed_masked = f_embed_raw.clone()
            o_embed_masked = o_embed_raw.clone()
            s_embed_masked = s_embed_raw.clone()

            if mask_idx_f.sum() > 0:
                masked_labels_f = face_feats[mask_idx_f]
                f_embed_masked[mask_idx_f] = self.face_mask_token.to(f_embed_masked.dtype).expand(B, MAX_FACES, -1)[mask_idx_f]
            if mask_idx_o.sum() > 0:
                masked_labels_o = object_feats[mask_idx_o]
                o_embed_masked[mask_idx_o] = self.object_mask_token.to(o_embed_masked.dtype).expand(B, MAX_OBJECTS, -1)[mask_idx_o]
            if mask_idx_s.sum() > 0:
                masked_labels_s = scene_feat[mask_idx_s]
                s_embed_masked[mask_idx_s] = self.scene_mask_token.to(s_embed_masked.dtype).expand(B, NUM_SCENE_PATCHES, -1)[mask_idx_s]

            # Đưa Query bị Mask qua Stage 1
            f_s1_mask, _ = self.face_encoder(f_embed_masked, f_pad_mask)
            o_s1_mask, _ = self.obj_encoder(o_embed_masked, o_pad_mask)
            s_s1_mask, _ = self.scene_encoder(s_embed_masked, s_pad_mask)

            # --- TRÍCH XUẤT NGỮ CẢNH SẠCH (CLEAN CONTEXT) ---
            #Chỉ lấy 15 token Top-K của Scene Sạch làm Memory
            s_s1_clean_topk = torch.gather(s_s1_clean, 1, topk_s.unsqueeze(-1).expand(-1, -1, D_MODEL))
            s_pad_mask_topk = torch.zeros(B, TOP_K_SCENE, device=device, dtype=torch.bool)

            # --- STAGE 2 MFM  ---
            # Query là Masked. Key/Value là clean và đã được lọc nhiễu
            
            # Masked Face hỏi Clean Obj và Top-15 Clean Scene
            mem_f_clean = torch.cat([s_s1_clean_topk, o_s1_clean], dim=1)
            mem_pad_mask_f_clean = torch.cat([s_pad_mask_topk, o_pad_mask], dim=1)
            f_c2_m, _ = self.face_cross(query=f_s1_mask, key=mem_f_clean, value=mem_f_clean, key_padding_mask=mem_pad_mask_f_clean)

            # Masked Obj hỏi Clean Face và Top-15 Clean Scene
            mem_o_clean = torch.cat([s_s1_clean_topk, f_s1_clean], dim=1)
            mem_pad_mask_o_clean = torch.cat([s_pad_mask_topk, f_pad_mask], dim=1)
            o_c2_m, _ = self.obj_cross(query=o_s1_mask, key=mem_o_clean, value=mem_o_clean, key_padding_mask=mem_pad_mask_o_clean)

            # Masked Scene hỏi Clean Face và Clean Obj
            mem_s_clean = torch.cat([f_s1_clean, o_s1_clean], dim=1)
            mem_pad_mask_s_clean = torch.cat([f_pad_mask, o_pad_mask], dim=1)
            s_c2_m, _ = self.scene_cross(query=s_s1_mask, key=mem_s_clean, value=mem_s_clean, key_padding_mask=mem_pad_mask_s_clean)

            # Refine
            f_r2_m = self.face_refine(f_c2_m, src_key_padding_mask=f_pad_mask)
            o_r2_m = self.obj_refine(o_c2_m,  src_key_padding_mask=o_pad_mask)
            s_r2_m = self.scene_refine(s_c2_m, src_key_padding_mask=s_pad_mask)

            # --- KHÔI PHỤC (MFM) ---
            if mask_idx_f.sum() > 0: logits_f_mfm = self.face_mfm_head(f_r2_m[mask_idx_f])
            if mask_idx_o.sum() > 0: logits_o_mfm = self.object_mfm_head(o_r2_m[mask_idx_o])
            if mask_idx_s.sum() > 0: logits_s_mfm = self.scene_mfm_head(s_r2_m[mask_idx_s])

        return preds_clean, (logits_f_mfm, masked_labels_f), \
                            (logits_o_mfm, masked_labels_o), \
                            (logits_s_mfm, masked_labels_s)

# ==========================================
# 4. LOSS FUNCTIONS
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma, self.reduction, self.label_smoothing = gamma, reduction, label_smoothing
        if alpha is not None: self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else: self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None: focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

def standard_cosine_loss(pred, target):
    if pred is None or target is None or pred.size(0) == 0:
        return torch.tensor(0.0, device=DEVICE)
    pred_norm = F.normalize(pred, p=2, dim=-1)
    return (1.0 - F.cosine_similarity(pred_norm, target, dim=-1)).mean()

# ==========================================
# 5. TRAINING & EVALUATION LOOPS
# ==========================================
def train_epoch(model, loader, criterion_cls, optimizer, current_alpha_mfm):
    model.train()
    # Biến tích lũy cho Tổng và MFM
    total_loss = total_cls = total_mfm = 0
    total_f_mfm = total_o_mfm = total_s_mfm = 0
    
    # Biến tích lũy cho 5 đầu Classification
    total_l1_f = total_l1_o = total_l2_f = total_l2_o = total_l_c = 0
    
    correct = total = 0
    scaler  = torch.amp.GradScaler('cuda')

    for batch in tqdm(loader, desc="Training"):
        faces, f_mask, objects, o_mask, scene, labels = [b.to(DEVICE) for b in batch]
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):
            preds, (f_pred, f_tgt), (o_pred, o_tgt), (s_pred, s_tgt) = model(
                faces, f_mask, objects, o_mask, scene, masking=True
            )
            p1_f, p1_o, p2_f, p2_o, p_c = preds

            # 1. Classification Loss (Tách riêng từng đầu)
            l1_f = criterion_cls(p1_f, labels)
            l1_o = criterion_cls(p1_o, labels)
            l2_f = criterion_cls(p2_f, labels)
            l2_o = criterion_cls(p2_o, labels)
            l_c  = criterion_cls(p_c, labels)

            loss_cls = l1_f + l1_o + l2_f + l2_o + l_c

            # 2. MFM Loss (Tách riêng)
            loss_f_raw = standard_cosine_loss(f_pred, f_tgt)
            loss_o_raw = standard_cosine_loss(o_pred, o_tgt)
            loss_s_raw = standard_cosine_loss(s_pred, s_tgt)
            
            # Gộp MFM Loss với trọng số (Bạn tự set W_MFM_* ở phần Config nhé)
            loss_mfm = (W_MFM_FACE * loss_f_raw) + (W_MFM_OBJ * loss_o_raw) + (W_MFM_SCENE * loss_s_raw)

            # 3. TỔNG LOSS
            loss = loss_cls + current_alpha_mfm * loss_mfm

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Tích lũy để tính trung bình
        total_loss += loss.item()
        total_cls  += loss_cls.item()
        total_mfm  += loss_mfm.item()
        
        total_f_mfm += loss_f_raw.item()
        total_o_mfm += loss_o_raw.item()
        total_s_mfm += loss_s_raw.item()

        total_l1_f += l1_f.item()
        total_l1_o += l1_o.item()
        total_l2_f += l2_f.item()
        total_l2_o += l2_o.item()
        total_l_c  += l_c.item()

        _, pred = torch.max(p_c, 1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    # IN RA MÀN HÌNH LOG CHI TIẾT
    print(f"  [Losses Tổng] CLS={total_cls/n:.4f} | MFM={total_mfm/n:.4f} | TỔNG={total_loss/n:.4f}")
    print(f"  [Chi tiết CLS] L1_F={total_l1_f/n:.4f} | L1_O={total_l1_o/n:.4f} | L2_F={total_l2_f/n:.4f} | L2_O={total_l2_o/n:.4f} | L_C={total_l_c/n:.4f}")
    print(f"  [Chi tiết MFM] Face={total_f_mfm/n:.4f} | Obj={total_o_mfm/n:.4f} | Scene={total_s_mfm/n:.4f}")
    
    return total_cls / n, 100 * correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            faces, f_mask, objects, o_mask, scene, labels = [b.to(DEVICE) for b in batch]
            preds, _, _, _ = model(faces, f_mask, objects, o_mask, scene, masking=False)
            _, _, _, _, p_c = preds

            loss = criterion(p_c, labels)
            total_loss += loss.item()
            
            _, pred = torch.max(p_c, 1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)

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
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss (CLS)')
    plt.plot(val_losses,   label='Val Loss (CLS)')
    plt.legend()
    plt.title('Classification Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs,   label='Val Acc')
    plt.legend()
    plt.title('Accuracy (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=150)
    plt.close()

# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  ViLT - ATTENTION-GUIDED HARD MASKING (XAI APPROACH)")
    print("=" * 65)

    train_ds = MultiModalFeatureDataset('train')
    val_ds   = MultiModalFeatureDataset('val')
    test_ds  = MultiModalFeatureDataset('test')

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate_fn_features, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate_fn_features, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate_fn_features, num_workers=0, pin_memory=True)

    print(f"\nInitializing Model...")
    model = ViLTWithMFM().to(DEVICE)
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    class_weights = torch.tensor([1.0, 1.2, 1.5]).to(DEVICE)
    criterion_cls = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(NUM_EPOCHS):
        current_alpha_mfm = ALPHA_MFM_START - (ALPHA_MFM_START - ALPHA_MFM_END) * (epoch / NUM_EPOCHS)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | Alpha MFM: {current_alpha_mfm:.4f}")
        
        t_loss, t_acc = train_epoch(model, train_loader, criterion_cls, optimizer, current_alpha_mfm)
        v_loss, v_acc, _, _ = eval_epoch(model, val_loader, criterion_cls)

        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc);   val_accs.append(v_acc)
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