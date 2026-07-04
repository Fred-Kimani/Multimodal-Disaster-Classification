import torch
import torch.nn as nn
from .image_baseline import ImageBaselineModel
from .fusion import FusedFeatureProjector, CrossModalAttention

class MultimodalClassifier(nn.Module):
    """
    Multimodal classification model that combines textual and visual features
    using a middle fusion architecture (concatenation or cross-modal attention).
    """
    def __init__(
        self,
        text_in_dim: int,
        image_backbone: str = "resnet50",
        pretrained_image: bool = True,
        freeze_image_backbone: bool = True,
        project_dim: int = 256,
        fusion_type: str = "concat",  # Options: concat, attention
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super(MultimodalClassifier, self).__init__()
        self.fusion_type = fusion_type
        
        # Image branch (extracts raw CNN features)
        self.image_branch = ImageBaselineModel(
            backbone=image_backbone,
            num_classes=num_classes,
            pretrained=pretrained_image,
            freeze_backbone=freeze_image_backbone
        )
        # Determine image feature size based on backbone
        if image_backbone == "resnet50":
            image_in_dim = 2048
        elif image_backbone == "efficientnet_b0":
            image_in_dim = 1280
        else:
            raise ValueError(f"Unsupported image backbone: {image_backbone}")
            
        # Projectors to map text and image features to the same dimension
        self.projector = FusedFeatureProjector(
            text_in_dim=text_in_dim,
            image_in_dim=image_in_dim,
            project_dim=project_dim
        )
        
        # Fusion layer
        if fusion_type == "attention":
            self.attention_fusion = CrossModalAttention(embed_dim=project_dim, dropout=0.1)
            classifier_in_dim = project_dim
        else: # Default: concat
            classifier_in_dim = project_dim * 2
            
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, project_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(project_dim, num_classes)
        )
        
    def forward(self, text_feats: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feats (Tensor): TF-IDF feature tensor, shape [batch_size, text_in_dim]
            images (Tensor): Raw image tensor, shape [batch_size, 3, H, W]
            
        Returns:
            Tensor: Logits for each class, shape [batch_size, num_classes]
        """
        # 1. Extract image features
        img_feats = self.image_branch.extract_features(images)
        
        # 2. Project text and image features to shared dimension
        proj_text, proj_img = self.projector(text_feats, img_feats)
        
        # 3. Apply fusion
        if self.fusion_type == "attention":
            fused_feats = self.attention_fusion(proj_text, proj_img)
        else: # concat
            fused_feats = torch.cat([proj_text, proj_img], dim=1)
            
        # 4. Predict
        logits = self.classifier(fused_feats)
        return logits
