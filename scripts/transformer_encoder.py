import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderModel(nn.Module):
    """
    A generic Transformer Encoder for bounding box regression (Algorithm 4).
    Flattens the image into a sequence, then uses multi-head self-attention.
    Output: 4D bounding box [x, y, w, h].
    """

    def __init__(self, 
                 image_size=224, 
                 patch_size=16, 
                 dim=512, 
                 depth=6, 
                 heads=8, 
                 mlp_dim=1024, 
                 channels=3, 
                 dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        # We flatten the entire image as a sequence of (patch_size x patch_size) patches or even pixel-level tokens
        assert image_size % patch_size == 0, "Image must be divisible by patch_size."

        self.seq_len = (image_size // patch_size) ** 2  # number of patches
        patch_dim = channels * patch_size * patch_size

        # Simple patch embedding
        self.patch_embed = nn.Linear(patch_dim, dim)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=heads,
                                                   dim_feedforward=mlp_dim,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Final MLP for bounding box
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4)  # bounding box [x, y, w, h]
        )
        self.dropout = nn.Dropout(dropout)
        self.patch_size = patch_size

    def forward(self, x):
        """
        x shape: (B, 3, 224, 224)
        """
        B, C, H, W = x.size()
        p = self.patch_size

        # Create patches
        patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, 3, H/p, p, W/p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(B, -1, p*p*C)  # (B, seq_len, patch_dim)

        # Patch embedding
        patches = self.patch_embed(patches)  # (B, seq_len, dim)
        # Add positional embedding
        patches = patches + self.pos_embedding[:, : patches.size(1), :]
        patches = self.dropout(patches)

        # (seq_len, B, dim)
        patches = patches.permute(1, 0, 2)

        # Transformer Encoder
        enc_out = self.transformer(patches)  # (seq_len, B, dim)

        # Global average of sequence dimension or just take the first token
        enc_out = enc_out.mean(dim=0)  # (B, dim)

        # Output bounding box
        out = self.mlp_head(enc_out)
        return out
