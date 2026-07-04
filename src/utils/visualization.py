import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm_list: list, class_names: list, save_path: str = None):
    """
    Plots and optionally saves a confusion matrix heatmap.
    
    Args:
        cm_list (list): 2D array representation of confusion matrix.
        class_names (list): List of class names for axes.
        save_path (str, optional): Path where target image should be saved.
    """
    cm = np.array(cm_list)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_training_curves(train_losses: list, val_losses: list, train_accs: list, val_accs: list, save_path: str = None):
    """
    Plots training and validation loss/accuracy curves.
    
    Args:
        train_losses (list): Training losses over epochs.
        val_losses (list): Validation losses over epochs.
        train_accs (list): Training accuracies over epochs.
        val_accs (list): Validation accuracies over epochs.
        save_path (str, optional): Path where target image should be saved.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_accs, "r-", label="Validation Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
