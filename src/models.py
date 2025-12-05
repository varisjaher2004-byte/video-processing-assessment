import torch
import torch.nn as nn
import torchvision.models as models

# ======================================================
# BASELINE MODEL: CNN (ResNet18) + LSTM
# ======================================================

class BaselineModel(nn.nn.Module):
    def __init__(self, hidden_dim=256, vocab_size=1000):
        super(BaselineModel, self).__init__()

        # CNN feature extractor (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.feature_dim = 512

        # LSTM for sequence learning
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def encode_frames(self, frames):
        B, F, C, H, W = frames.shape
        frames = frames.view(B * F, C, H, W)
        feats = self.cnn(frames).squeeze(-1).squeeze(-1)  
        feats = feats.view(B, F, self.feature_dim)
        return feats

    def forward(self, frames):
        feats = self.encode_frames(frames)
        lstm_out, _ = self.lstm(feats)
        last = lstm_out[:, -1, :]
        logits = self.fc(last)
        return logits


# ======================================================
# IMPROVED MODEL: Transformer Encoder + Tag Fusion
# ======================================================

class TransformerTagModel(nn.Module):
    def __init__(
        self,
        vocab_size=1000,
        hidden_dim=256,
        n_heads=4,
        num_layers=2,
        tag_dim=5
    ):
        super(TransformerTagModel, self).__init__()

        # CNN Encoder
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.feature_dim = 512

        # Frame feature projection
        self.frame_proj = nn.Linear(self.feature_dim, hidden_dim)

        # Tag projection
        self.tag_proj = nn.Linear(tag_dim, hidden_dim)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 100, hidden_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def encode_frames(self, frames):
        B, F, C, H, W = frames.shape
        frames = frames.view(B * F, C, H, W)
        feats = self.cnn(frames).squeeze(-1).squeeze(-1)
        feats = feats.view(B, F, self.feature_dim)
        return feats

    def forward(self, frames, tag_vec):
        B, F, C, H, W = frames.shape

        # CNN encoding
        feats = self.encode_frames(frames)
        feats = self.frame_proj(feats)

        # Add positional embedding
        pos = self.pos_embed[:, :F, :]
        feats = feats + pos

        # Add tag embedding
        tag_emb = self.tag_proj(tag_vec).unsqueeze(1)
        feats = feats + tag_emb

        # Transformer input format: [F, B, H]
        feats = feats.permute(1, 0, 2)

        out = self.transformer(feats)
        last = out[-1]
        logits = self.fc(last)

        return logits
