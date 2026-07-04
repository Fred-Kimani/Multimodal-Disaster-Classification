# Multimodal Disaster Classification Using Social Media Images and Text

This repository implements a lightweight multimodal disaster classification framework using social media images and text (tweets) from the **CrisisMMD** dataset. 

The architecture follows a middle fusion approach:
1. **Text Branch**: Features extracted via TF-IDF vectorization.
2. **Image Branch**: Features extracted via transfer learning using a pretrained CNN (e.g., ResNet50 or EfficientNet).
3. **Fusion Layer**: Fuses text and image features (using concatenation or cross-modal attention).
4. **Classifier**: Fully connected neural network mapping fused features to target classes (Informativeness, Disaster Type, Damage Severity, etc.).

---

## Installation & Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd Multimodal_Disaster_Classification
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Preparation

1. Download the **CrisisMMD** dataset.
2. Place the dataset files in the `data/raw/` folder:
   - TSV files (train, dev, test) under `data/raw/` or config-specified path.
   - Images in the corresponding structure.

---

## Project Structure

```text
Multimodal_Disaster_Classification/
├── requirements.txt         # Dependencies
├── config.yaml              # Hyperparameters and path configurations
├── data/                    # Dataset directory
├── src/                     # Source code directory
│   ├── data/                # Dataset class and preprocessing pipelines
│   ├── models/              # Model architectures (unimodal & multimodal)
│   ├── utils/               # Metrics and visualizations
│   ├── train_text.py        # Text unimodal training script
│   ├── train_image.py       # Image unimodal training script
│   └── train_multimodal.py  # Multimodal training script
```

---

## How to Run

### 1. Train Text Baseline
```bash
python -m src.train_text
```

### 2. Train Image Baseline
```bash
python -m src.train_image
```

### 3. Train Multimodal Model
```bash
python -m src.train_multimodal
```
