# Traffic Sign Detection — Project Plan

**Module:** Advanced AI (CS6514)  
**Team:** James Doherty (23374039), Oisín Frizzell (23368276)  
**Dataset:** [Traffic Sign Object Detection Challenge](https://www.kaggle.com/competitions/traffic-sign-object-detection-challenge/data)

---

## Research question

Does fine-tuning the full Faster R-CNN backbone improve held-out detection (mAP@0.5) over a frozen backbone, once we account for run-to-run variation?

---

## What we are comparing

| | Baseline | Change |
|---|---|---|
| Model | Pretrained Faster R-CNN | same |
| Backbone | **Frozen** | **Unfrozen** |
| Head | Trained on our 8 classes | same |
| Everything else | identical | identical |
| Runs | 3 seeds | same 3 seeds |

Only one thing changes: whether the backbone can update during training.

Unlocked is **not** guaranteed to win. On a small dataset it can overfit. A null result is fine.

---

## Dataset

| | |
|---|---|
| Source | Kaggle competition (downloaded via `download-dataset.py`) |
| Format | COCO JSON (`annotations.json`) |
| Classes (8) | Bus stop, Crossroad, No entry, No parking, No stopping, Speed limit, Yield, direction of road |
| Train | 708 images, 865 boxes |
| Test | 167 images (83 have public labels) |
| Licence | Kaggle competition terms (confirm exact wording before final submit) |

Images are large (often 3060×4080). We resize consistently for training.

---

## Data splits

1. Split original **train** into **train / val** (~85% / 15%), stratified so all 8 classes appear in val.
2. Save the split so it stays fixed.
3. Use **val** for checkpoint selection and decisions.
4. Use **test** once at the end only. Never for tuning.

---

## Model and metric

- **Model:** torchvision Faster R-CNN (COCO pretrained), head replaced for 8 classes (+ background).
- **Primary metric:** mAP@0.5  
- **Secondary:** mAP@0.5:0.95, per-class AP  
- **Compute:** local / free T4-class GPU · **6 runs total** (2 conditions × 3 seeds)

---

## Who does what

| Area | Owner |
|---|---|
| Data pipeline, split, augmentation, repo/README, error analysis, submission CSV | Oisín |
| Model + freeze switch, training loop, seeding, 6 runs, metrics, results tables/figures | James |
| Proposal, writeup, slides, understanding everything | Both |

---

## Phases

### Week 4 — Proposal
- Submit question, dataset, licence, model, baseline, comparison, metric, splits, compute, risks.

### Week 5 — Baseline gate
- End-to-end pipeline working.
- One full **frozen** run saved (config + metric + checkpoint).

### Week 6 — Results checkpoint
- All 6 runs logged (every seed, not just best).
- Results table + one figure showing seed spread.
- Short error analysis (e.g. confused red circular signs).

### Week 7 — Freeze and submit
- Written analysis, presentation (≤8 min), contribution statement.
- No new experiments unless agreed.
- Rehearse oral defence individually.

---

## Risks

- Small data → effect may sit inside seed noise (null result OK).
- Large images → GPU memory; resize and record the size used.
- Similar signs (No entry / No parking / No stopping) may confuse the model.
- Test leakage — only score test once at the end.

**Backup comparison (if freeze/unfreeze is rejected):** minimal augmentation vs one added augmentation. Same structure otherwise.

---

## Submission checklist

- [ ] Code + `requirements.txt` + README
- [ ] Configs, seeds, all 6 run results
- [ ] Written analysis
- [ ] Slides + contribution statement
- [ ] No secrets (`.env` gitignored)
- [ ] Dataset/model provenance and licence noted
- [ ] Conclusion matches the evidence (including uncertainty)
