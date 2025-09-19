import os
from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_scheduler
from .visualize import (
    save_training_curves, save_per_class_acc, save_confusion_matrix_plot,
    save_tsne_plots, save_roc_curves, save_predictions
)
from .metrics import compute_classification_report

def create_scheduler(optimizer, train_loader_len, epochs, warmup_ratio=0.1):
    num_training_steps = train_loader_len * epochs
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    return get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

def validate(model, val_loader, device, label_names):
    model.eval()
    correct, total = 0, 0
    per_class_correct = defaultdict(int); per_class_total = defaultdict(int)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(image, input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for p, t in zip(preds, labels):
                per_class_total[t.item()] += 1
                if p.item() == t.item():
                    per_class_correct[t.item()] += 1

    acc = correct / max(total, 1)
    report = compute_classification_report(all_labels, all_preds, label_names)
    return acc, report, per_class_correct, per_class_total, all_labels, all_preds, all_probs

def train_loop(
    model, train_loader, val_loader, optimizer, scheduler, criterion, device,
    epochs, save_path, base_vis_dir, label_names
):
    best_acc = 0.0
    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)

        
        acc, report, per_class_correct, per_class_total, all_labels, all_preds, all_probs = validate(
            model, val_loader, device, label_names
        )
        val_accuracies.append(acc)

        print(f"\n Epoch {epoch+1} - Loss: {avg_loss:.4f}, Val Acc: {acc*100:.2f}%")
        print("Classification Report:")
        print(report)

       
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print("Best model saved:", save_path)

        
        if (epoch + 1) % 5 == 0:
            epoch_dir = os.path.join(base_vis_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pt"))

            save_training_curves(epoch_dir, train_losses, val_accuracies)
            save_per_class_acc(epoch_dir, per_class_correct, per_class_total, label_names)
            save_confusion_matrix_plot(epoch_dir, all_labels, all_preds, label_names)
            save_tsne_plots(epoch_dir, model, val_loader, label_names, device)
            save_roc_curves(epoch_dir, all_probs, all_labels, label_names)
            save_predictions(epoch_dir, all_labels, all_preds, all_probs)

    print("\n{:.2f}%".format(best_acc*100))
