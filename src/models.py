import torch
import torch.nn as nn
import torchvision.models as models

class BaselineModel(nn.Module):
    """
    CNN (ResNet18) + LSTM model for video → caption generation.
    """

    def __init__(self, hidden_dim=256, vocab_size=5000):
        super(BaselineModel, self).__init__()

        # 1. CNN Feature Extractor (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]   # remove final FC layer
        self.cnn = nn.Sequential(*modules)

        self.feature_dim = 512  # ResNet18 output channels

        # 2. LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 3. Decoder: Convert hidden state → vocabulary distribution
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, frames):
        """
        frames = list of tensors, shape: [batch, frames, 3, H, W]
        """

        batch_size, num_frames, C, H, W = frames.shape

        # Merge batch and frames for CNN
        frames = frames.view(batch_size * num_frames, C, H, W)

        # CNN feature extraction
        feats = self.cnn(frames).squeeze()  # → [batch*num_frames, 512]

        # Reshape back to [batch, frames, feature_dim]
        feats = feats.view(batch_size, num_frames, self.feature_dim)

        # LSTM
        lstm_out, _ = self.lstm(feats)  # → sequence output

        # last time step output
        last_output = lstm_out[:, -1, :]  # → [batch, hidden_dim]

        # Decoder
        out = self.fc(last_output)  # → vocabulary logits

        return out
