import os
import yaml
import argparse
from src.data.dataset import CrisisMMDDataset
from src.models.text_baseline import TextBaselineModel
from src.utils.metrics import calculate_metrics

def main(config_path: str, label_col: str):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"--- Training Text Baseline for {label_col} ---")
    
    # Initialize datasets
    # Note: Text baseline uses the text strings directly, so we don't need image_transforms
    train_dataset = CrisisMMDDataset(
        tsv_path=config["data"]["train_tsv"],
        images_dir=config["data"]["raw_dir"],
        label_column=label_col
    )
    val_dataset = CrisisMMDDataset(
        tsv_path=config["data"]["val_tsv"],
        images_dir=config["data"]["raw_dir"],
        label_column=label_col
    )
    
    # Extract text and labels for Scikit-Learn fitting
    train_texts = [train_dataset[i]["text"] for i in range(len(train_dataset))]
    train_labels = [train_dataset[i]["label"].item() for i in range(len(train_dataset))]
    
    val_texts = [val_dataset[i]["text"] for i in range(len(val_dataset))]
    val_labels = [val_dataset[i]["label"].item() for i in range(len(val_dataset))]
    
    # Initialize and fit TF-IDF + Logistic Regression baseline
    text_cfg = config["text_model"]
    model = TextBaselineModel(
        max_features=text_cfg.get("max_features", 5000),
        ngram_range=tuple(text_cfg.get("ngram_range", [1, 2]))
    )
    
    print("Fitting model...")
    model.fit(train_texts, train_labels)
    
    # Evaluate
    print("Evaluating model...")
    val_preds = model.predict(val_texts)
    metrics = calculate_metrics(val_labels, val_preds)
    
    # Print results
    print("\n--- Validation Metrics ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
            
    # Save the baseline model
    model_save_dir = "models/checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"text_baseline_{label_col}.pkl")
    model.save(model_path)
    print(f"\nSaved text baseline model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--label_col", type=str, default="label")  # e.g., label (informativeness), label_category (disaster type)
    args = parser.parse_args()
    
    main(args.config, args.label_col)
