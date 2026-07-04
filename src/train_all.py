import os
import yaml
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data.dataset import CrisisMMDDataset
from src.data.preprocessing import get_image_transforms
from src.models.classifier import MultimodalClassifier
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_training_curves, plot_confusion_matrix

def train_single_task(config, task_name, train_tsv, val_tsv, label_col, fusion_type="attention", epochs=10):
    print(f"\n=========================================")
    print(f"Starting Training for Task: {task_name.upper()} (column: {label_col})")
    print(f"Using fusion: {fusion_type} | Epochs: {epochs}")
    print(f"=========================================")
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["training"]["device"] == "cuda"
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Preprocessing & Transformations
    img_size = config["image_model"]["img_size"]
    train_transforms = get_image_transforms(img_size=img_size, is_train=True)
    val_transforms = get_image_transforms(img_size=img_size, is_train=False)
    
    # Load Datasets
    train_dataset = CrisisMMDDataset(
        tsv_path=train_tsv,
        images_dir=config["data"]["raw_dir"],
        image_transform=train_transforms,
        label_column=label_col
    )
    val_dataset = CrisisMMDDataset(
        tsv_path=val_tsv,
        images_dir=config["data"]["raw_dir"],
        image_transform=val_transforms,
        label_column=label_col
    )
    
    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes for {task_name}: {num_classes} ({train_dataset.unique_labels})")
    
    # Save the label mapping
    os.makedirs("models/checkpoints", exist_ok=True)
    mapping_path = os.path.join("models/checkpoints", f"mapping_{task_name}.pkl")
    with open(mapping_path, "wb") as f:
        pickle.dump({
            "idx_to_label": train_dataset.idx_to_label,
            "label_to_idx": train_dataset.label_to_idx,
            "unique_labels": train_dataset.unique_labels
        }, f)
    print(f"Saved class mapping to {mapping_path}")
    
    # Fit TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer...")
    train_texts_raw = [train_dataset[i]["text"] for i in range(len(train_dataset))]
    text_cfg = config["text_model"]
    vectorizer = TfidfVectorizer(
        max_features=text_cfg.get("max_features", 5000),
        ngram_range=tuple(text_cfg.get("ngram_range", [1, 2]))
    )
    vectorizer.fit(train_texts_raw)
    text_in_dim = len(vectorizer.vocabulary_)
    
    # Save the fitted TF-IDF Vectorizer
    vectorizer_path = os.path.join("models/checkpoints", f"vectorizer_{task_name}.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved TF-IDF Vectorizer to {vectorizer_path}")
    
    # Collate function
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
        fusion_type=fusion_type,
        num_classes=num_classes,
        dropout=config["multimodal_model"]["dropout"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr"],
        weight_decay=float(config["training"]["weight_decay"])
    )
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch in train_loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            
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
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"models/checkpoints/multimodal_best_{task_name}.pth")
            
            # Save confusion matrix plot
            plot_confusion_matrix(
                cm_list=metrics["confusion_matrix"],
                class_names=train_dataset.unique_labels,
                save_path=f"models/checkpoints/confusion_matrix_{task_name}.png"
            )
            
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=f"models/checkpoints/training_curves_{task_name}.png"
    )
    print(f"Finished {task_name.upper()}. Best accuracy: {best_acc:.4f}")

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # We will train 3 tasks sequentially
    tasks = [
        {
            "name": "informativeness",
            "train_tsv": "data/raw/crisismmd_datasplit_all/crisismmd_datasplit_all/task_informative_text_img_train.tsv",
            "val_tsv": "data/raw/crisismmd_datasplit_all/crisismmd_datasplit_all/task_informative_text_img_dev.tsv",
            "label_col": "label"
        },
        {
            "name": "disaster_type",
            "train_tsv": "data/raw/crisismmd_datasplit_all/crisismmd_datasplit_all/task_informative_text_img_train.tsv",
            "val_tsv": "data/raw/crisismmd_datasplit_all/crisismmd_datasplit_all/task_informative_text_img_dev.tsv",
            "label_col": "event_name"
        },
        {
            "name": "damage_severity",
            "train_tsv": "data/raw/crisismmd_datasplit_all/crisismmd_datasplit_all/task_damage_text_img_train.tsv",
            "val_tsv": "data/raw/crisismmd_datasplit_all/crisismmd_datasplit_all/task_damage_text_img_dev.tsv",
            "label_col": "label"
        }
    ]
    
    # Run sequential training with attention-based fusion for 5 epochs to maintain speed/feasibility
    for task in tasks:
        train_single_task(
            config=config,
            task_name=task["name"],
            train_tsv=task["train_tsv"],
            val_tsv=task["val_tsv"],
            label_col=task["label_col"],
            fusion_type="attention",
            epochs=5
        )

if __name__ == "__main__":
    main()
