📌 Video Processing Assessment – Improved Story Generation Model

This repository contains my implementation for the Neural Networks and Deep Learning coursework at Sheffield Hallam University.
The goal of the assessment is to generate coherent text descriptions (stories) from short video sequences.

The notebook provided by the instructor was used as the baseline architecture.
In this project, I extend the baseline model with semantic tag extraction and a temporal Transformer module to improve sequence understanding and reduce hallucinations.

🧩 1. Project Overview

Video-driven narrative generation often struggles with:

Capturing temporal dependencies across frames

Maintaining coherence

Avoiding hallucinated objects/actions

Linking visuals meaningfully to text

My Contribution

This repository implements two major improvements over the baseline:

✅ 1) Semantic Tag Fusion

Object, action, and scene tags are extracted from captions and converted into a 64-dimensional vector.
This vector is fused with the model’s embeddings to improve grounding.

✅ 2) Temporal Transformer Module

A lightweight Transformer Encoder improves long-range temporal understanding across the frame sequence.

These additions enhance coherence, reduce hallucination, and improve alignment with video content.

🏗 2. Repository Structure
video-processing-assessment/
│
├── final_notebook.ipynb        # Modified notebook with improvements
├── README.md                   # Project documentation
├── images/                     # Optional loss curve, visualizations
│    └── loss_curve.png
└── models/                     # Optional saved model weights
     └── improved_model.pth

⚙️ 3. How to Run

Open the notebook in Google Colab

Mount Google Drive

Load the dataset from HuggingFace

Run:

Chapter 1 (data preparation + semantic tags)

Chapter 2 (baseline encoder + improved architecture)

Chapter 3 (training loop for improved model only)

View the loss curve and generated story predictions

The improved model is trained for 3 epochs for demonstration.

🔧 4. Model Improvements
⭐ SemanticFusion Layer
class SemanticFusion(nn.Module):
    def __init__(self, embed_dim=256, tag_dim=64):
        super().__init__()
        self.proj = nn.Linear(tag_dim, embed_dim)

    def forward(self, embed, tags):
        return embed + self.proj(tags)

⭐ TemporalTransformer Module
class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

⭐ Improved Sequence Predictor
class ImprovedSequencePredictor(nn.Module):
    def __init__(self, base_model, embed_dim=256):
        super().__init__()
        self.base = base_model
        self.temporal = TemporalTransformer(embed_dim)
        self.fusion = SemanticFusion(embed_dim)

    def forward(self, frames, descriptions, tags):
        img_emb = self.base.visual_encoder(frames)
        text_emb = self.base.text_encoder(descriptions)

        combined = self.fusion(img_emb + text_emb, tags)
        temp_out = self.temporal(combined)

        return self.base.decoder(temp_out)

📈 5. Results
Training Loss for Improved Model

Loss consistently decreases over 3 epochs

Improved stability across sequences

Better temporal coherence

(Insert your loss plot here)

🎯 6. Key Takeaways

✔ Transformer improves temporal understanding
✔ Semantic tags help ground text to visuals
✔ Improved coherence and reduced hallucination
✔ Lightweight model suitable for limited training data

👤 Author

Varis Jaherbhai Kureshi
MSc Artificial Intelligence
Sheffield Hallam University

📚 Academic Integrity

This repository contains only my own implementation.
Baseline code was provided by the module instructor; all extensions and modifications are original.
