# Traffic Sign Detection

Advanced AI (CS6514) group project by James Doherty and Oisín Frizzell.

## What this project does

We test one question. Does letting the whole model learn (unfreezing the backbone) detect
traffic signs better than keeping most of it fixed (freezing the backbone)?

We use Faster R-CNN, a pretrained object detector, and adapt it to 8 traffic sign classes.
We train it two ways, frozen and unfrozen, with everything else kept the same. We run each
way three times with different random seeds so we can tell a real difference from luck.

## What we found

Unfreezing wins every time. The unfrozen model scored about 0.935 on our validation set
and the frozen model about 0.908. All three unfrozen runs beat all three frozen runs, and
the gap is much bigger than the difference between runs, so it is not just noise.

## Files

- `model.py` builds the detector and holds the freeze switch
- `dataset.py` loads the images and makes the train and validation split
- `train.py` runs the training and saves the results
- `splits.json` is the fixed train and validation split
- `runs/` holds the saved settings and scores for every run

## How to run it

Install the requirements first.

```
pip install -r requirements.txt
```

Then run each condition. Each command trains all three seeds.

```
python train.py --freeze
python train.py --unfreeze
```

The data comes from the Traffic Sign Object Detection Challenge on Kaggle. It downloads
automatically if you add your Kaggle token to a `.env` file, or you can place it in a
`data` folder yourself.
