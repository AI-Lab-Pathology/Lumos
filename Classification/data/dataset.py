import os
import glob
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch

class MultimodalClassificationDataset(Dataset):
    def __init__(self, image_root, text_root, tokenizer, transform, label_encoder: LabelEncoder):
        self.samples: List[Tuple[str, str, str]] = []
        self.tokenizer = tokenizer
        self.transform = transform
        self.label_encoder = label_encoder

        for cls in os.listdir(image_root):
            img_dir = os.path.join(image_root, cls)
            if not os.path.isdir(img_dir):
                continue

            for img_path in glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg')):
                base = os.path.splitext(os.path.basename(img_path))[0]
                txt_folder = os.path.join(text_root, cls)
                found_txt = None

                if os.path.exists(txt_folder):
                    for fname in os.listdir(txt_folder):
                        if fname.startswith(base) and fname.endswith(".txt"):
                            found_txt = os.path.join(txt_folder, fname)
                            break

                if found_txt:
                    try:
                        with open(found_txt, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                        if len(text) < 5:
                            print(f"Text too short {found_txt}")
                            continue
                    except Exception as e:
                        print(f"Text read failed {found_txt} {e}")
                        continue

                    self.samples.append((img_path, found_txt, cls))
                else:
                    print(f"Missing text file {base} under {txt_folder}")

        class_names = [s[2] for s in self.samples]
        
        self.labels = self.label_encoder.fit_transform(class_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path, _ = self.samples[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def build_dataloaders(
    train_image_root, train_text_root,
    val_image_root, val_text_root,
    tokenizer, train_transform, val_transform,
    batch_size=16, num_workers=4
):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()

    train_set = MultimodalClassificationDataset(train_image_root, train_text_root, tokenizer, train_transform, label_encoder)

    val_set = MultimodalClassificationDataset(val_image_root, val_text_root, tokenizer, val_transform, label_encoder)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, label_encoder
