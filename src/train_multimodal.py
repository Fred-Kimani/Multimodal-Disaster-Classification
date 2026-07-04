import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data.dataset import CrisisMMDDataset
from src.data.preprocessing import get_image_transforms
from src.models.classifier import MultimodalClassifier
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_training_curves, plot_confusion_matrix

def main(config_path: str, label_col: str):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"--- Training Multimodal Model for {label_col} ---")
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["training"]["device"] == "cuda"
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Initialize Transforms
    img_size = config["image_model"]["img_size"]
    train_transforms = get_image_transforms(img_size=img_size, is_train=True)
    val_transforms = get_image_transforms(img_size=img_size, is_train=False)
    
    # Datasets & Loaders
    train_dataset = CrisisMMDDataset(
        tsv_path=config["data"]["train_tsv"],
        images_dir=config["data"]["raw_dir"],
        image_transform=train_transforms,
        label_column=label_col
    )
    val_dataset = CrisisMMDDataset(
        tsv_path=config["data"]["val_tsv"],
        images_dir=config["data"]["raw_dir"],
        image_transform=val_transforms,
        label_column=label_col
    )
    
    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes: {num_classes}")
    
    # Fit TF-IDF Vectorizer on train texts to convert texts to features for Pytorch model
    print("Fitting TF-IDF Vectorizer on training corpus...")
    train_texts_raw = [train_dataset[i]["text"] for i in range(len(train_dataset))]
    text_cfg = config["text_model"]
    vectorizer = TfidfVectorizer(
        max_features=text_cfg.get("max_features", 5000),
        ngram_range=tuple(text_cfg.get("ngram_range", [1, 2]))
    )
    vectorizer.fit(train_texts_raw)
    text_in_dim = len(vectorizer.vocabulary_)
    print(f"TF-IDF Vocabulary size: {text_in_dim}")
    
    # Data loaders
    # We write a custom collate function to handle batching of raw texts
    def custom_collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        texts = [item["text"] for item in batch]
        return {
            "images": images,
            "labels": labels,
            "texts": texts
        }
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # Initialize Model
    model = MultimodalClassifier(
        text_in_dim=text_in_dim,
        image_backbone=config["image_model"]["backbone"],
        pretrained_image=config["image_model"]["pretrained"],
        freeze_image_backbone=config["image_model"]["freeze_backbone"],
        project_dim=config["multimodal_model"]["project_dim"],
        fusion_type=config["multimodal_model"]["fusion_type"],
        num_classes=num_classes,
        dropout=config["multimodal_model"]["dropout"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr"],
        weight_decay=float(config["training"]["weight_decay"])
    )
    
    # Track statistics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    epochs = config["training"]["epochs"]
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch in train_loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            
            # Vectorize text and load to GPU
            text_vecs = vectorizer.transform(batch["texts"]).toarray()
            text_tensors = torch.tensor(text_vecs, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            outputs = model(text_tensors, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                labels = batch["labels"].to(device)
                
                text_vecs = vectorizer.transform(batch["texts"]).toarray()
                text_tensors = torch.tensor(text_vecs, dtype=torch.float32).to(device)
                
                outputs = model(text_tensors, images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        metrics = calculate_metrics(all_labels, all_preds)
        val_acc = metrics["accuracy"]
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Checkpoint saving
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"models/checkpoints/multimodal_best_{label_col}.pth")
            
            # Save metrics & confusion matrix plot
            plot_confusion_matrix(
                cm_list=metrics["confusion_matrix"],
                class_names=train_dataset.unique_labels,
                save_path=f"models/checkpoints/confusion_matrix_{label_col}.png"
            )
            
    # Save training graphs
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=f"models/checkpoints/training_curves_{label_col}.png"
    )
    
    print(f"Training completed. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--label_col", type=str, default="label")
    args = parser.parse_args()
    
    main(args.config, args.label_col)
