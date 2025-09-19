import torch
from transformers import AutoTokenizer
from models.cross_modal import CrossAttentionFusionClassifier
from losses.focal import FocalLoss
from utils.transforms import get_train_transform, get_val_transform
from data.dataset import build_dataloaders
from utils.train import create_scheduler, train_loop
import config as C

def set_seed(seed: int):
    import random, os, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    set_seed(C.SEED)


    tokenizer = AutoTokenizer.from_pretrained(C.BERT_MODEL_PATH)


    train_tf, val_tf = get_train_transform(), get_val_transform()
    train_loader, val_loader, label_encoder = build_dataloaders(
        C.IMAGE_ROOT, C.TEXT_ROOT,
        C.VAL_IMAGE_ROOT, C.VAL_TEXT_ROOT,
        tokenizer, train_tf, val_tf,
        batch_size=C.BATCH_SIZE, num_workers=4
    )
    label_names = list(label_encoder.classes_)

  
    model = CrossAttentionFusionClassifier(C.NUM_CLASSES, C.BERT_MODEL_PATH).to(C.DEVICE)

  
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR)
    scheduler = create_scheduler(optimizer, len(train_loader), C.EPOCHS, C.WARMUP_RATIO)
    criterion = FocalLoss(gamma=2.0)  
    # === Train ===
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=C.DEVICE,
        epochs=C.EPOCHS,
        save_path=C.SAVE_PATH,
        base_vis_dir=C.BASE_VIS_DIR,
        label_names=label_names
    )

if __name__ == "__main__":
    main()
