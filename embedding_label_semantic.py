import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Config
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
OUTPUT_PATH = "semantic_hierarchical_384d_ordinal.npy" # Save file tên mới
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. ĐỊNH NGHĨA LABEL CHA (PARENT PROMPTS)
# Đây là vector sẽ đứng đầu chuỗi (vai trò như CLS token)
PARENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# 2. ĐỊNH NGHĨA LEXICON CON (CHILDREN) - Giữ nguyên của bạn
GROUP_EMOTION_LEXICON = {
    0: { # NEGATIVE
        "Fractured": "Group breaking into defensive sub-units",
        "Hostile": "Aggressive posturing and confrontational gazes",
        "Somber": "Collective heaviness, low energy, and downcast eyes",
        "Tense": "Rigid body language and high-stress atmosphere",
        "Disorganized": "Chaotic, non-purposeful movement and confusion",
        "Apathetic": "Total lack of engagement or collective energy",
        "Defensive": "Crossed arms and closed-off physical clusters",
        "Oppressive": "A heavy, stifling atmosphere dominated by one or two",
        "Agitated": "Restless, erratic collective movements",
        "Alienated": "High physical distance between group members",
        "Mournful": "Shared grief, slumped shoulders, and stillness",
        "Indignant": "Collective outrage or shared expressions of contempt",
        "Stagnant": "Lack of interaction despite physical proximity"
    },
    1: { # NEUTRAL
        "Passive": "Observation without emotional reaction",
        "Formal": "Structured, professional, and emotionally suppressed",
        "Attentive": "Focused on a central point without emotive bias",
        "Inert": "Physical stillness, waiting for direction",
        "Methodical": "Task-oriented focus with clinical efficiency",
        "Ambivalent": "Mixed signals or lack of a clear collective mood",
        "Stoic": "Controlled, unexpressive facial configurations",
        "Functional": "Interaction exists only to complete a task",
        "Observant": "External focus, group acting as an audience",
        "Undifferentiated": "No clear emotional peak or valley in the group",
        "Compliant": "Following protocol without enthusiasm or resistance",
        "Sedate": "Calm, low-energy, but not negative or sad",
        "Placid": "Tranquil and undisturbed collective state"
    },
    2: { # POSITIVE
        "Jubilant": "High-energy, celebratory atmosphere",
        "Cohesive": "Strong sense of unity and shared purpose",
        "Harmonious": "Balanced, peaceful, and synchronized interaction",
        "Exuberant": "Overflowing with enthusiasm and physical energy",
        "Synchronized": "Group movements or laughter occurring in unison",
        "Radiant": "Collective brightness in facial expressions",
        "Affirming": "Mutual nodding and supportive body language",
        "Communal": "Shared focus and open physical orientation",
        "Playful": "Lighthearted, teasing, and high-engagement",
        "Triumphant": "Victorious collective stance like arms raised",
        "Animated": "High gestural activity and lively discourse",
        "Inclusive": "Open group circles, welcoming posture",
        "Resonant": "Shared emotional vibration or flow state"
    }
}

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    emb = mean_pooling(outputs, inputs['attention_mask'])
    return F.normalize(emb, p=2, dim=1)

def extract_hierarchical():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    final_tensor_list = []

    with torch.no_grad():
        for label_id in range(3):
            print(f"--- Processing Label {label_id} ---")
            sequence_vectors = []
            
            # BƯỚC 1: TRÍCH XUẤT LABEL CHA (Vị trí 0)
            parent_text = PARENT_LABELS[label_id]
            parent_emb = get_embedding(parent_text, tokenizer, model)
            sequence_vectors.append(parent_emb) # Thêm vào đầu danh sách
            print(f"   + Added Parent: {parent_text[:30]}...")

            # BƯỚC 2: TRÍCH XUẤT LEXICONS CON (Vị trí 1 -> N)
            lexicon_dict = GROUP_EMOTION_LEXICON.get(label_id, {}) # .get để tránh lỗi nếu paste thiếu
            for word, description in lexicon_dict.items():
                child_text = f"Group emotion is {word}: {description}"
                child_emb = get_embedding(child_text, tokenizer, model)
                sequence_vectors.append(child_emb)
            
            # Stack lại: (1 + Num_Children, 384)
            # Với dictionary của bạn: 1 + 13 = 14 vectors
            class_tensor = torch.cat(sequence_vectors, dim=0)
            
            # Lưu ý: Nếu số lượng từ con không đều, bạn cần thêm code padding ở đây giống bài trước.
            # Giả sử lexicon đều 13 từ -> Tensor sẽ là (14, 384)
            
            final_tensor_list.append(class_tensor.unsqueeze(0))

    # Output Shape: (3, 14, 384)
    full_tensor = torch.cat(final_tensor_list, dim=0)
    print(f"Final Tensor Shape: {full_tensor.shape}")
    
    np.save(OUTPUT_PATH, full_tensor.cpu().numpy())
    print("Saved hierarchical vectors.")

if __name__ == "__main__":
    extract_hierarchical()