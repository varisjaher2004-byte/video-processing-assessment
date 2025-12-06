import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, frame_dirs, captions, tags, transform=None):
        self.frame_dirs = frame_dirs
        self.captions = captions
        self.tags = tags
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def load_frames(self, folder_path):
        frames = []
        files = sorted(os.listdir(folder_path))

        for f in files:
            if f.endswith(".jpg") or f.endswith(".png"):
                img_path = os.path.join(folder_path, f)
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)
                frames.append(img)

        # ⭐⭐ BIG FIX HERE ⭐⭐
        return torch.stack(frames)  # shape: [num_frames, 3, 224, 224]

    def __getitem__(self, idx):
        folder = self.frame_dirs[idx]
        caption = self.captions[idx]
        tag = self.tags[idx]

        frames_tensor = self.load_frames(folder)

        return {
            "frames": frames_tensor,   # ⭐ tensor not list
            "caption": caption,
            "tags": tag,
        }

    def __len__(self):
        return len(self.frame_dirs)
