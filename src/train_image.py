import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import CrisisMMDDataset
from src.data.preprocessing import get_image_transforms
from src.models.image_baseline import ImageBaselineModel
from src.utils.metrics import calculate_metrics

def main(config_path: str, label_col: str):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"--- Training Image Baseline for {label_col} ---")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    # Initialize Model
    model = ImageBaselineModel(
        backbone=config["image_model"]["backbone"],
        num_classes=num_classes,
        pretrained=config["image_model"]["pretrained"],
        freeze_backbone=config["image_model"]["freeze_backbone"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr"],
        weight_decay=float(config["training"]["weight_decay"])
    )
    
    # Training loop
    epochs = config["training"]["epochs"]
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        metrics = calculate_metrics(all_labels, all_preds)
        val_acc = metrics["accuracy"]
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save checkpoints
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"models/checkpoints/image_baseline_{label_col}.pth")
            
    print(f"Training completed. Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--label_col", type=str, default="label")
    args = parser.parse_args()
    
    main(args.config, args.label_col)
