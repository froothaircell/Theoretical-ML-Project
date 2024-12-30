import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    """
    Vision Transformer for bounding box regression (Algorithm 3).
    Image is split into patches, then self-attention is applied.
    Output: 4D bounding box [x, y, w, h].
    """

    def __init__(self, 
                 image_size=224, 
                 patch_size=16, 
                 dim=768, 
                 depth=6, 
                 heads=8, 
                 mlp_dim=2048, 
                 channels=3, 
                 dropout=0.1):
        """
        Args:
            image_size (int): Input image size (assumed square).
            patch_size (int): Patch size.
            dim (int): Embedding dimension.
            depth (int): Number of Transformer encoder layers.
            heads (int): Number of attention heads.
            mlp_dim (int): Hidden dimension for feed-forward layers.
            channels (int): Input channels (3 for RGB).
            dropout (float): Dropout rate.
        """
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        # Patch embedding
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        # Class token replaced by "dummy" since we only need bounding box
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=heads,
                                                   dim_feedforward=mlp_dim,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Bounding box head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4)  # output [x, y, w, h]
        )

        self.patch_size = patch_size

    def forward(self, img):
        """
        img shape: (B, 3, 224, 224)
        """
        B, C, H, W = img.shape
        p = self.patch_size

        # Create patches
        # (B, C, H//p, p, W//p, p) => rearrange => (B, (H//p)*(W//p), p^2*C)
        patches = img.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(B, -1, p*p*C)

        # Linear embedding
        x = self.patch_to_embedding(patches)  # (B, num_patches, dim)

        # Prepend cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, dim)

        # Add positional embedding
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.dropout(x)

        # Transformer expects (sequence_length, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)

        # Pass through the transformer
        x = self.transformer(x)  # (num_patches+1, B, dim)
        x = x.permute(1, 0, 2)   # back to (B, num_patches+1, dim)

        # Take the "cls" token's representation
        cls_rep = x[:, 0]
        out = self.mlp_head(cls_rep)
        return out
