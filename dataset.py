import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageOps

MAX_SIDE = 1024


class SignDataset(Dataset):
    def __init__(self, root, split="train"):
        self.img_dir = Path(root) / split / "images"
        with open(Path(root) / split / "annotations.json") as f:
            data = json.load(f)
        self.images = data["images"]
        self.by_image = {}
        for a in data["annotations"]:
            self.by_image.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = ImageOps.exif_transpose(Image.open(self.img_dir / info["file_name"])).convert("RGB")
        scale = min(1.0, MAX_SIDE / max(img.size))
        if scale < 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)))

        boxes, labels = [], []
        for a in self.by_image.get(info["id"], []):
            x, y, w, h = a["bbox"]
            boxes.append([x * scale, y * scale, (x + w) * scale, (y + h) * scale])
            labels.append(a["category_id"] + 1)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))
