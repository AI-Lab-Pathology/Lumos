# LUMOS: A Multimodal Pathology Platform for Gastric Cancer
> **LUMOS** — multimodal pathology (histology + text) for **prognostic subtyping** and **immunotherapy decision support** in gastric cancer.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12-orange" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-green" />
</p>


---

## Table of Contents
- [Highlights](#highlights)
- [Modules](#modules)
- [Quickstart](#quickstart)
  - [Environment](#environment)
  - [Module 1 — Text Generation](#module-1--text-generation)
  - [Module 2 — Classification](#module-2--classification)
- [To-Do](#to-do)
- [Citation](#citation)
- [License](#license)

---

## Highlights
- **Two modules**:
  - **`txt-generation/`** — clinical/text generation pipeline: **original draft** → **refinement** (polishing).
  - **`Classification/`** — pathology image classification (training / evaluation / batch inference).
  


## Modules

> This repository **currently includes two modules**. We will expand the modular design as the project evolves.
- **`txt-generation/`** — clinical/text generation pipeline: **original draft** → **refinement** (polishing).
- **`Classification/`** — pathology image classification (training / evaluation / batch inference).






## Quickstart

### Environment

```bash


#Classification:
 pip install -r Classification/requirements.txt

```

---
```bash


#Text generation:
 pip install -r txt-generation/requirements.txt
```

---




### Module 1 — Text Generation 

1) **Download the model**  
   Get **InternVL2-8B** from Hugging Face and place it locally:  
   https://huggingface.co/OpenGVLab/InternVL2-8B

2) **Prepare images & set paths**  
   Put images under per-sample subfolders, then set the input/output paths in
   `txt-generation/For-all-txt-generation.py` (bottom of the file), e.g.:
   ```python
   path = r"\InternVL2-8B"   # local model dir
   input_folder  = r"\image" # e.g., \image\Sample_001\*.png|jpg
   output_folder = r"\txt"   # captions will mirror the folder structure
   generate_captions(input_folder, output_folder, model, tokenizer, max_num=12)



### Module 2 — Classification (Path-First Quick Guide)

> This section matches your current cross-modal code (ViT-B/16 + BioBERT + cross-attention).  
> **Goal:** set the paths correctly, then run.

---

##  Set dataset & model paths (edit in code)

```python
# ===== Required: training/validation data roots =====
IMAGE_ROOT     = r"\train\image"  # train images, organized by class
TEXT_ROOT      = r"\train\txt"    # train texts,   organized by class
VAL_IMAGE_ROOT = r"\val\image"    # val images,    organized by class
VAL_TEXT_ROOT  = r"\val\txt"      # val texts,     organized by class

# ===== Required: BioBERT (Hugging Face model dir or local path) =====
BERT_MODEL_PATH = r"E:\BioBERT"  # e.g., 'dmis-lab/biobert-base-cased-v1.1' if using HF cache

# ===== Optional: where to save model & visualizations =====
SAVE_PATH    = "cross_modal_classifier.pt"  # checkpoint (will load backbone weights if exists)
BASE_VIS_DIR = "visualization_report"       # loss/acc curves, confusion matrix, t-SNE, peak PDFs

```



---

## To-Do
* [ ] **Subtype classification**
* [ ] **Long-tail recognition**
* [ ] **Grad-CAM visualization**
  CAM/Grad-CAM(++) for ViT features and fused representations; export per-sample overlays.
* [ ] **Immunotherapy-related prediction**

---




## Citation

If you find this repository useful, please cite:

```bibtex
@misc{LUMOS2025,
  title  = {LUMOS: A Multimodal Pathology Platform for Gastric Cancer},
  author = {AI-Lab-Pathology},
  year   = {2025},
  url    = {https://github.com/AI-Lab-Pathology/Lumos}
}
```




## License

This project is released under the **Apache License 2.0**.
See [`LICENSE`](./LICENSE) for details.


