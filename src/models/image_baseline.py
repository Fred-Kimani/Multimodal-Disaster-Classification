import torch
import torch.nn as nn
import torchvision.models as models

class ImageBaselineModel(nn.Module):
    """
    Unimodal image baseline classifier using transfer learning on a pretrained CNN.
    """
    def __init__(self, backbone: str = "resnet50", num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = True):
        super(ImageBaselineModel, self).__init__()
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            # Replace final fc layer with identity to act as feature extractor
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            # Replace final classifier with identity
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features prior to the final classification head (for multimodal fusion).
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features
