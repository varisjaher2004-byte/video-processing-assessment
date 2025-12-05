import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, frame_dirs, captions, tags, transform=None):
        self.frame_dirs = frame_dirs          # list of folders
        self.captions = captions              # list of captions
        self.tags = tags                      # list of semantic tags
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def load_frames(self, folder_path):
        frames = []
        files = sorted(os.listdir(folder_path))  # sorted list of frame names

        for f in files:
            if f.endswith(".jpg") or f.endswith(".png"):
                img_path = os.path.join(folder_path, f)
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)
                frames.append(img)

        return frames

    def __getitem__(self, idx):
        folder = self.frame_dirs[idx]
        caption = self.captions[idx]
        tag = self.tags[idx]

        frames = self.load_frames(folder)

        return {
            "frames": frames,
            "caption": caption,
            "tags": tag,
        }

    def __len__(self):
        return len(self.frame_dirs)
