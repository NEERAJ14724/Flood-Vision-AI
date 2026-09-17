import os
from dataset import FloodNetDataset

# BASE PATH
BASE = os.environ.get("FLOODNET_PATH", "../data/supervised/FloodNet-Supervised_v1.0")
if os.path.isdir(os.path.join(BASE, "FloodNet-Supervised_v1.0")):
    BASE = os.path.join(BASE, "FloodNet-Supervised_v1.0")

TRAIN_IMG = os.path.join(BASE, "train/train-org-img")
TRAIN_MASK = os.path.join(BASE, "train/train-label-img")

dataset = FloodNetDataset(TRAIN_IMG, TRAIN_MASK)

print("Total samples:", len(dataset))

img, mask = dataset[0]
print("Sample loaded successfully")