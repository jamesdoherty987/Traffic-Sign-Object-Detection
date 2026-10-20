import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset import SignDataset, collate_fn
from model import build_model

N_STEPS = 20
BATCH_SIZE = 2
ROOT = Path(__file__).resolve().parent

device = "mps" if torch.backends.mps.is_available() else "cpu"
ds = SignDataset(ROOT, "train")
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = build_model(freeze_backbone=True, seed=0).to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

print(f"device={device} images={len(ds)} batch={BATCH_SIZE} timing {N_STEPS} steps...")
model.train()
t0 = None
for step, (images, targets) in enumerate(dl):
    if step >= N_STEPS:
        break
    images = [im.to(device) for im in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)
    loss = sum(loss_dict.values())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step == 0:
        t0 = time.time()
        print(f"  step 0 loss={loss.item():.3f} (warmup, not timed)")
    else:
        print(f"  step {step} loss={loss.item():.3f}")

elapsed = time.time() - t0
per_step = elapsed / (N_STEPS - 1)
steps_per_epoch = len(ds) / BATCH_SIZE
epoch_frozen = per_step * steps_per_epoch
print(f"\nper step: {per_step:.2f}s")
print(f"est frozen epoch ({steps_per_epoch:.0f} steps): {epoch_frozen/60:.1f} min")
print(f"est 15-epoch frozen run: {epoch_frozen*15/60:.0f} min")
print(f"est 3 frozen runs: {epoch_frozen*15*3/3600:.1f} h")
print(f"unfrozen ~2-3x slower -> 3 unfrozen runs: {epoch_frozen*15*3*2.5/3600:.1f} h")
