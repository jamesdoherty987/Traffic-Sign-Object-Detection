import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SignDataset, collate_fn, get_data_root
from model import build_model

ROOT = Path(__file__).resolve().parent
SPLIT_FILE = ROOT / "splits.json"
SEEDS = [0, 1, 2]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_split(data_root):
    if SPLIT_FILE.exists():
        return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))

    with open(data_root / "train" / "annotations.json", encoding="utf-8") as f:
        data = json.load(f)

    ids = [im["id"] for im in data["images"]]
    random.Random(42).shuffle(ids)
    n_val = max(1, int(len(ids) * 0.15))
    split = {"train_ids": ids[n_val:], "val_ids": ids[:n_val]}
    SPLIT_FILE.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"wrote {SPLIT_FILE} train={len(split['train_ids'])} val={len(split['val_ids'])}")
    return split


@torch.no_grad()
def val_map50(model, loader, device):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError:
        print("install torchmetrics for mAP")
        return None

    metric = MeanAveragePrecision(iou_thresholds=[0.5])
    was_training = model.training
    model.eval()
    for images, targets in loader:
        images = [im.to(device) for im in images]
        preds = [{k: v.cpu() for k, v in p.items()} for p in model(images)]
        tgts = [{k: v.cpu() for k, v in t.items() if k in ("boxes", "labels")} for t in targets]
        metric.update(preds, tgts)
    if was_training:
        model.train()
    return float(metric.compute()["map"].item())


def train_one(freeze, seed, epochs, batch_size, lr, out_base):
    set_seed(seed)
    device = get_device()
    data_root = get_data_root()
    split = ensure_split(data_root)

    train_loader = DataLoader(
        SignDataset(data_root, split["train_ids"]),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SignDataset(data_root, split["val_ids"]),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model(freeze_backbone=freeze, seed=seed).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    tag = "freeze" if freeze else "unfreeze"
    out_dir = Path(out_base) / f"{tag}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "freeze_backbone": freeze,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\n=== {tag} seed={seed} device={device} ===")

    history = []
    best = -1.0
    model.train()

    for epoch in range(1, epochs + 1):
        total = 0.0
        for images, targets in train_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / max(1, len(train_loader))
        m = val_map50(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": avg, "val_map50": m})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch {epoch}/{epochs}  loss={avg:.3f}  val_mAP50={m}")

        if m is not None and m >= best:
            best = m
            torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")
    print(f"saved {out_dir} (best mAP50={best})")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--unfreeze", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=None, help="run one seed only (for a quick test)")
    p.add_argument("--out", default=str(ROOT / "runs"), help="base dir for results (e.g. a Drive path)")
    args = p.parse_args()

    seeds = [args.seed] if args.seed is not None else SEEDS
    for seed in seeds:
        train_one(args.freeze, seed, args.epochs, args.batch_size, args.lr, args.out)


if __name__ == "__main__":
    main()
