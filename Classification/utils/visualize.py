import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from .metrics import compute_confusion_matrix, compute_multiclass_roc_auc

def _save_fig(path_base):
    plt.savefig(path_base + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(path_base + ".svg", bbox_inches="tight")
    plt.close()

def save_training_curves(epoch_dir, train_losses, val_accuracies):
    os.makedirs(epoch_dir, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss"); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot([a*100 for a in val_accuracies]); plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Val Accuracy"); plt.grid(True)
    plt.tight_layout()
    _save_fig(os.path.join(epoch_dir, "training_curve"))

def save_per_class_acc(epoch_dir, per_class_correct, per_class_total, label_names):
    accs = [ (per_class_correct.get(i,0) / per_class_total.get(i,1)) if per_class_total.get(i,0)>0 else 0 for i in range(len(label_names)) ]
    plt.figure(figsize=(10,4))
    plt.bar(label_names, [a*100 for a in accs], color='skyblue')
    plt.xticks(rotation=45); plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy"); plt.grid(True, axis='y')
    plt.tight_layout()
    _save_fig(os.path.join(epoch_dir, "per_class_accuracy"))

def save_confusion_matrix_plot(epoch_dir, y_true, y_pred, label_names):
    cm = compute_confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix"); plt.tight_layout()
    _save_fig(os.path.join(epoch_dir, "confusion_matrix"))

def _extract_features(model, dataloader, device, mode="multimodal"):
    model.eval()
    feats, labs = [], []
    import torch
    with torch.no_grad():
        for batch in dataloader:
            img  = batch["image"].to(device)
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab  = batch["label"]
            if mode == "image":
                feat = model.vit(img)
            else:
                img_feat = model.vit(img)
                img_feat_proj = model.image_proj(img_feat).unsqueeze(1)
                txt_feat = model.bert(input_ids=ids, attention_mask=mask).last_hidden_state
                fused, _ = model.cross_attn(query=txt_feat, key=img_feat_proj, value=img_feat_proj)
                feat = fused.mean(dim=1)
            feats.append(feat.cpu())
            labs.extend(lab.tolist())
    return np.concatenate([f.numpy() for f in feats], axis=0), labs

def save_tsne_plots(epoch_dir, model, val_loader, label_names, device):
    for mode, title in [("image", "Image-only Features"), ("multimodal", "Multimodal Features")]:
        feats, labs = _extract_features(model, val_loader, device, mode=mode)
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        tsne_feats = tsne.fit_transform(feats)
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=tsne_feats[:,0], y=tsne_feats[:,1], hue=[label_names[i] for i in labs], palette="tab10", s=16, linewidth=0)
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        _save_fig(os.path.join(epoch_dir, f"tsne_{mode}"))

def save_roc_curves(epoch_dir, all_probs, all_labels, label_names):
    fpr, tpr, roc_auc = compute_multiclass_roc_auc(all_probs, all_labels, label_names)
    plt.figure(figsize=(8,6))
    for i, name in enumerate(label_names):
        plt.plot(fpr[i], tpr[i], label=f"{name} (AUC={roc_auc[i]:.2f})")
    plt.plot([0,1], [0,1], 'k--', label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout()
    _save_fig(os.path.join(epoch_dir, "roc_curve"))

def save_predictions(epoch_dir, all_labels, all_preds, all_probs):
    with open(os.path.join(epoch_dir, "predictions.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({"y_true": all_labels, "y_pred": all_preds}, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(epoch_dir, "probs.npy"), np.array(all_probs))
