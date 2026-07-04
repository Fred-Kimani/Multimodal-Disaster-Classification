import torch
import torch.nn as nn

class FusedFeatureProjector(nn.Module):
    """
    Projects separate text and image features into a shared representation space.
    """
    def __init__(self, text_in_dim: int, image_in_dim: int, project_dim: int = 256):
        super(FusedFeatureProjector, self).__init__()
        self.text_project = nn.Sequential(
            nn.Linear(text_in_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.image_project = nn.Sequential(
            nn.Linear(image_in_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, text_feats: torch.Tensor, img_feats: torch.Tensor) -> tuple:
        """
        Args:
            text_feats (Tensor): Text feature tensor of shape [batch_size, text_in_dim]
            img_feats (Tensor): Image feature tensor of shape [batch_size, image_in_dim]
            
        Returns:
            tuple: (projected_text_feats, projected_img_feats)
        """
        proj_text = self.text_project(text_feats)
        proj_img = self.image_project(img_feats)
        return proj_text, proj_img

class CrossModalAttention(nn.Module):
    """
    Calculates cross-modal attention between text and image representations.
    Utilizes PyTorch's MultiheadAttention where:
      Query = Image features
      Key, Value = Text features
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(CrossModalAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, text_feats: torch.Tensor, img_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feats (Tensor): Projected text features of shape [batch_size, embed_dim]
            img_feats (Tensor): Projected image features of shape [batch_size, embed_dim]
            
        Returns:
            Tensor: Fused representation of shape [batch_size, embed_dim]
        """
        # MultiheadAttention expects sequence dimension: [batch_size, seq_len, embed_dim]
        # Treat features as sequence of length 1
        q = img_feats.unsqueeze(1) # [B, 1, D]
        k = text_feats.unsqueeze(1) # [B, 1, D]
        v = text_feats.unsqueeze(1) # [B, 1, D]
        
        # Attention pass
        attn_out, _ = self.mha(q, k, v)
        
        # Residual connection and layer normalization
        fused = self.norm(img_feats + attn_out.squeeze(1))
        return fused
