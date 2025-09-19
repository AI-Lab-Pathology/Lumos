import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from transformers import AutoModel

class CrossAttentionFusionClassifier(nn.Module):
    def __init__(self, num_classes: int, bert_model_path: str):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()

        self.bert = AutoModel.from_pretrained(bert_model_path)

        self.image_proj = nn.Linear(768, 768)
        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.vit(image)                
        img_feat = self.image_proj(img_feat).unsqueeze(1)  

        txt_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  

        fused, _ = self.cross_attn(query=txt_feat, key=img_feat, value=img_feat)  
        fused_pooled = fused.mean(dim=1)  
        logits = self.classifier(fused_pooled)
        return logits
