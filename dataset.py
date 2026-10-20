import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageOps

MAX_SIDE = 1024


def get_data_root():
    local = Path(__file__).resolve().parent / "data"
    if (local / "train" / "annotations.json").exists():
        return local

    cache = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "competitions"
        / "traffic-sign-object-detection-challenge"
    )
    if (cache / "train" / "annotations.json").exists():
        return cache

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.getenv("KAGGLE_API_TOKEN"):
        raise SystemExit("Need KAGGLE_API_TOKEN in .env, or put the dataset in ./data")

    import kagglehub

    return Path(kagglehub.competition_download("traffic-sign-object-detection-challenge"))


class SignDataset(Dataset):
    def __init__(self, root, image_ids=None):
        self.img_dir = Path(root) / "train" / "images"
        with open(Path(root) / "train" / "annotations.json", encoding="utf-8") as f:
            data = json.load(f)

        self.images = data["images"]
        if image_ids is not None:
            keep = set(image_ids)
            self.images = [im for im in self.images if im["id"] in keep]

        self.by_image = {}
        for a in data["annotations"]:
            self.by_image.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = ImageOps.exif_transpose(
            Image.open(self.img_dir / info["file_name"])
        ).convert("RGB")

        scale = min(1.0, MAX_SIDE / max(img.size))
        if scale < 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)))

        boxes, labels = [], []
        for a in self.by_image.get(info["id"], []):
            x, y, w, h = a["bbox"]
            boxes.append([x * scale, y * scale, (x + w) * scale, (y + h) * scale])
            labels.append(a["category_id"] + 1)

        return to_tensor(img), {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([info["id"]]),
        }


def collate_fn(batch):
    return tuple(zip(*batch))
