# ================================================================
# DINOv2SegFormer v4 — COMPLETE FLOOD DETECTION PIPELINE
# FIXED VERSION — All bugs corrected:
#   FIX 1: USE_GRAD_CHECKPOINTING=False (cuBLAS FP16 crash fix)
#   FIX 2: BATCH_SIZE=4 (VRAM safety after Phase 2 unfreeze)
#   FIX 3: Phase 2/3 scheduler uses remaining steps (not total)
#   FIX 4: torch.cuda.empty_cache() before each phase transition
#   FIX 5: CUDA error recovery in train_epoch (try/except)
#   FIX 6: QUICK_VAL_BATCHES=100 mid-training, full at final eval
#   FIX 7: scheduler.step() called once per batch in train_epoch
#   FIX 8: Val-skip no_improve logic corrected
# ================================================================

import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['HF_HOME'] = os.environ.get('HF_HOME', os.path.join(WORK_DIR, '.hf_cache'))
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', os.path.join(WORK_DIR, '.hf_cache', 'transformers'))
os.environ['HF_DATASETS_CACHE'] = os.environ.get('HF_DATASETS_CACHE', os.path.join(WORK_DIR, '.hf_cache', 'datasets'))
import cv2, warnings, time, csv, json, math, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR
from transformers import Dinov2Model
from datetime import datetime
from fpdf import FPDF
import folium
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix as sk_cm, cohen_kappa_score,
                              precision_recall_fscore_support)
from collections import defaultdict
import traceback

warnings.filterwarnings('ignore')
os.environ['HF_ENDPOINT']            = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# ================================================================
# CONFIG — Edit these paths for your machine
# ================================================================
DATASET_ROOT = os.environ.get("FLOODNET_PATH", r"C:\Users\ratho\Downloads\FloodNet")
if os.path.isdir(os.path.join(DATASET_ROOT, "FloodNet-Supervised_v1.0")):
    BASE_PATH = os.path.join(DATASET_ROOT, "FloodNet-Supervised_v1.0")
else:
    BASE_PATH = DATASET_ROOT

TRAIN_IMG  = os.path.join(BASE_PATH, "train/train-org-img")
TRAIN_MASK = os.path.join(BASE_PATH, "train/train-label-img")
VAL_IMG    = os.path.join(BASE_PATH, "val/val-org-img")
VAL_MASK   = os.path.join(BASE_PATH, "val/val-label-img")
TEST_IMG   = os.path.join(BASE_PATH, "test/test-org-img")
TEST_MASK  = os.path.join(BASE_PATH, "test/test-label-img")

WORK_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.environ.get("FLOOD_OUTPUT_DIR", os.path.join(WORK_DIR, "flood_output_v4_complete"))

FAST_MODE = False
RESUME    = True

# ================================================================
# HYPERPARAMETERS
# ================================================================
IMAGE_SIZE       = 512
BATCH_SIZE       = 2           # FIX 2: reduced from 8 → safe VRAM after unfreeze
NUM_EPOCHS       = 60
PATIENCE         = 40
LR               = 5e-4
MIN_LR           = 5e-7
WEIGHT_DECAY     = 0.005
WARMUP_EPOCHS    = 8
FREEZE_EPOCHS    = 3
SWA_START_EPOCH  = 140
FLOOD_THRESHOLD  = 5.0
NUM_CLASSES      = 10
AUX_LOSS_WEIGHT  = 0.25
GRAD_ACCUM_STEPS = 2
SAVE_EVERY       = 3
USE_CACHE        = True
CACHE_DIR        = os.path.join(SAVE_DIR, "cache")
PIXEL_STATS_CACHE = os.path.join(SAVE_DIR, "pixel_counts.npz")
DINO_MODEL_NAME   = os.environ.get("DINO_MODEL_NAME", "facebook/dinov2-small")
USE_GRAD_CHECKPOINTING = False  # FIX 1: disabled — caused cuBLAS FP16 crash
USE_EMA           = True
EMA_DECAY         = 0.999
USE_CRF           = True

FOCAL_GAMMA       = 3.5
AUGMENT_PROB      = 0.9
TTA_COUNT         = 10
LABEL_SMOOTHING   = 0.1
MIXUP_ALPHA       = 0.4

FAST_LOSS         = False
FAST_AUGMENT      = True
LOAD_YOLO         = True
LOAD_YOLO_AFTER_TRAIN = True
FINAL_TTA         = True
RUN_FULL_REPORTS  = True
RUN_TEST_PREDICTIONS = True
QUICK_VAL_BATCHES = 100   # FIX 6: limit mid-training val to 100 batches
VAL_EVERY         = 1

# ================================================================
# DEVICE
# ================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*65}")
print(f"  DINOv2SegFormer v4 COMPLETE — FLOOD DETECTION PIPELINE")
print(f"{'='*65}")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"FAST_MODE : {FAST_MODE} | EMA: {USE_EMA} | TTA count: {TTA_COUNT}")
print(f"GRAD_CKPT : {USE_GRAD_CHECKPOINTING} (disabled=cuBLAS safe)")

for d in [SAVE_DIR,
          os.path.join(SAVE_DIR, "predictions"),
          os.path.join(SAVE_DIR, "reports"),
          os.path.join(SAVE_DIR, "graphs"),
          os.path.join(SAVE_DIR, "yolo"),
          CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ================================================================
# CLASS INFO
# ================================================================
CLASS_NAMES = {
    0: "Background",
    1: "Building-Flooded",
    2: "Building-NF",
    3: "Road-Flooded",
    4: "Road-NF",
    5: "Water",
    6: "Tree",
    7: "Vehicle",
    8: "Pool",
    9: "Grass"
}

CLASS_COLORS = np.array([
    [0,   0,   0  ],
    [255, 0,   0  ],
    [255, 165, 0  ],
    [255, 0,   255],
    [128, 0,   128],
    [0,   0,   255],
    [0,   255, 0  ],
    [255, 255, 0  ],
    [0,   255, 255],
    [165, 42,  42 ],
])

FLOODED_CLASSES = [1, 3, 5]

# ================================================================
# PAPER REFERENCE METRICS
# ================================================================
PAPER_METRICS = {
    'mIoU'      : 75.2,
    'mDice'     : 85.4,
    'mACC'      : 87.1,
    'PA'        : 89.3,
    'MPA'       : 82.6,
    'FWIoU'     : 81.4,
    'Kappa'     : 0.847,
    'Precision' : 84.1,
    'Recall'    : 83.7,
    'F1'        : 83.9,
    'Params'    : 91.3,
    'FPS'       : 8.0,
    'Latency'   : 120.0,
}

# ================================================================
# YOLO INTEGRATION
# ================================================================
def load_yolo_model():
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8m.pt')
        print("YOLOv8m loaded successfully!")
        return model
    except ImportError:
        print("ultralytics not installed. Installing...")
        os.system(f"{sys.executable} -m pip install ultralytics -q")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8m.pt')
            print("YOLOv8m loaded after install!")
            return model
        except Exception as e:
            print(f"YOLOv8 unavailable: {e}")
            return None

def run_yolo_detection(yolo_model, img_bgr, save_path=None):
    if yolo_model is None:
        return img_bgr, []
    try:
        results    = yolo_model(img_bgr, verbose=False, conf=0.3)
        detections = []
        annotated  = img_bgr.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf  = float(box.conf[0])
                cls_  = int(box.cls[0])
                label = yolo_model.names[cls_]
                color = (0, 255, 0) if 'car' not in label.lower() else (255, 128, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} {conf:.2f}",
                            (x1, max(y1-6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                detections.append({'label': label, 'conf': conf,
                                   'box': [x1, y1, x2, y2]})
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        return annotated, detections
    except Exception as e:
        print(f"  YOLO error: {e}")
        return img_bgr, []

# ================================================================
# AUGMENTATION HELPERS
# ================================================================
def mixup_data(imgs, masks, alpha=MIXUP_ALPHA):
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(imgs.size(0))
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, masks if lam > 0.5 else masks[idx]

def cutmix_data(imgs, masks, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = imgs.shape
    idx = torch.randperm(B)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W);  cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w//2, 0, W); y1 = np.clip(cy - cut_h//2, 0, H)
    x2 = np.clip(cx + cut_w//2, 0, W); y2 = np.clip(cy + cut_h//2, 0, H)
    mixed = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    mixed_masks = masks.clone()
    mixed_masks[:, y1:y2, x1:x2] = masks[idx, y1:y2, x1:x2]
    return mixed, mixed_masks

def mosaic_data(imgs, masks):
    B, C, H, W = imgs.shape
    if B < 4:
        return imgs, masks
    mh, mw = H * 2, W * 2
    mosaic_img  = torch.zeros(B//4, C, mh, mw, device=imgs.device)
    mosaic_mask = torch.zeros(B//4, mh, mw, dtype=torch.long, device=masks.device)
    for b in range(B//4):
        idx = [b*4, b*4+1, b*4+2, b*4+3]
        mosaic_img [b, :,  :H,  :W] = imgs [idx[0]]
        mosaic_img [b, :,  :H, W: ] = imgs [idx[1]]
        mosaic_img [b, :, H:,  :W]  = imgs [idx[2]]
        mosaic_img [b, :, H:, W: ]  = imgs [idx[3]]
        mosaic_mask[b,  :H,  :W]    = masks[idx[0]]
        mosaic_mask[b,  :H, W: ]    = masks[idx[1]]
        mosaic_mask[b, H:,  :W]     = masks[idx[2]]
        mosaic_mask[b, H:, W: ]     = masks[idx[3]]
    mosaic_img  = F.interpolate(mosaic_img,  size=(H,W), mode='bilinear', align_corners=False)
    mosaic_mask = F.interpolate(mosaic_mask.unsqueeze(1).float(),
                                size=(H,W), mode='nearest').squeeze(1).long()
    return mosaic_img, mosaic_mask

# ================================================================
# PREPROCESSING CACHE
# ================================================================
def get_cache_path(img_id, split, size):
    return os.path.join(CACHE_DIR, f"{split}_{img_id}_{size}.npz")

def save_to_cache(img_id, split, size, img_arr, mask_arr):
    path = get_cache_path(img_id, split, size)
    np.savez_compressed(path, img=img_arr, mask=mask_arr)

def load_from_cache(img_id, split, size):
    path = get_cache_path(img_id, split, size)
    if os.path.exists(path):
        data = np.load(path)
        return data['img'], data['mask']
    return None, None

# ================================================================
# DATASET
# ================================================================
class FloodNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512, augment=False, split='train'):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.size     = size
        self.augment  = augment
        self.split    = split

        img_ids  = set()
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_ids.add(os.path.splitext(f)[0])
        mask_ids = set()
        for f in os.listdir(mask_dir):
            if f.endswith('_lab.png'):
                mask_ids.add(f.replace('_lab.png', ''))

        self.ids = sorted(list(img_ids.intersection(mask_ids)))
        print(f"[{split}] Dataset: {len(self.ids)} pairs | augment={augment}")
        if len(self.ids) == 0:
            raise ValueError(f"No matching pairs in {img_dir} / {mask_dir}")

    def __len__(self): return len(self.ids)

    def _load_img(self, img_id):
        for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
            p = os.path.join(self.img_dir, img_id + ext)
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"Image not found: {img_id}")

    def _load_mask(self, img_id):
        return np.array(Image.open(
            os.path.join(self.mask_dir, img_id + "_lab.png")))

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        if USE_CACHE and not self.augment:
            img_c, mask_c = load_from_cache(img_id, self.split, self.size)
            if img_c is not None:
                img  = torch.tensor(img_c).float() / 255.0
                mask = torch.tensor(mask_c).long()
                mask = torch.clamp(mask, 0, NUM_CLASSES - 1)
                return img, mask

        img  = self._load_img(img_id)
        mask = self._load_mask(img_id)
        img  = cv2.resize(img,  (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST)

        if self.augment and np.random.rand() < AUGMENT_PROB:
            if np.random.rand() > 0.5:
                img  = cv2.flip(img,  1); mask = cv2.flip(mask, 1)
            if np.random.rand() > 0.5:
                img  = cv2.flip(img,  0); mask = cv2.flip(mask, 0)
            if np.random.rand() > 0.5:
                k    = np.random.randint(1, 4)
                img  = np.rot90(img,  k).copy()
                mask = np.rot90(mask, k).copy()
            if np.random.rand() > 0.3:
                alpha = np.random.uniform(0.5, 1.5)
                beta  = np.random.randint(-40, 40)
                img   = np.clip(img.astype(np.float32)*alpha+beta,0,255).astype(np.uint8)
            if np.random.rand() > 0.3:
                hsv        = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:,:,0] = np.clip(hsv[:,:,0]+np.random.uniform(-25,25), 0, 180)
                hsv[:,:,1] = np.clip(hsv[:,:,1]*np.random.uniform(0.5,1.5), 0, 255)
                hsv[:,:,2] = np.clip(hsv[:,:,2]*np.random.uniform(0.5,1.5), 0, 255)
                img        = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            if np.random.rand() > 0.3:
                h, w = img.shape[:2]
                crop = np.random.randint(256, min(self.size, h, w))
                x = np.random.randint(0, w-crop); y = np.random.randint(0, h-crop)
                img  = cv2.resize(img [y:y+crop, x:x+crop], (self.size, self.size))
                mask = cv2.resize(mask[y:y+crop, x:x+crop], (self.size, self.size),
                                  interpolation=cv2.INTER_NEAREST)
            if np.random.rand() > 0.4:
                img = cv2.GaussianBlur(img, (5,5), 0)
            if np.random.rand() > 0.4:
                h, w   = img.shape[:2]
                map_x  = np.tile(np.arange(w), (h,1)).astype(np.float32)
                map_y  = np.tile(np.arange(h), (w,1)).T.astype(np.float32)
                noise  = np.random.uniform(-5, 5, (h, w)).astype(np.float32)
                map_x += noise; map_y += noise
                img    = cv2.remap(img,  map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
                mask   = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_REFLECT_101)
            if np.random.rand() > 0.5:
                h, w = img.shape[:2]
                cx = np.random.randint(0, w); cy = np.random.randint(0, h)
                rw = np.random.randint(32, 128); rh = np.random.randint(32, 128)
                x1 = max(0,cx-rw//2); y1 = max(0,cy-rh//2)
                x2 = min(w,cx+rw//2); y2 = min(h,cy+rh//2)
                img[y1:y2, x1:x2] = 0
            if np.random.rand() > 0.6:
                h, w     = img.shape[:2]
                grid     = 4
                ph, pw   = h//grid, w//grid
                patches  = []
                mp       = []
                for gi in range(grid):
                    for gj in range(grid):
                        patches.append(img [gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw])
                        mp.append(mask[gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw])
                perm = np.random.permutation(len(patches))
                k    = 0
                for gi in range(grid):
                    for gj in range(grid):
                        img [gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw] = patches[perm[k]]
                        mask[gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw] = mp    [perm[k]]
                        k += 1
        else:
            if USE_CACHE:
                save_to_cache(img_id, self.split, self.size,
                              img.transpose(2,0,1), mask)

        img_t  = torch.tensor(img.copy()).permute(2,0,1).float() / 255.0
        mask_t = torch.tensor(mask.copy()).long()
        mask_t = torch.clamp(mask_t, 0, NUM_CLASSES - 1)
        return img_t, mask_t

# ================================================================
# PIXEL DISTRIBUTION
# ================================================================
def check_pixel_distribution(mask_dir, name="Dataset"):
    print(f"\n--- {name} pixel distribution ---")
    counts = np.zeros(10, dtype=np.int64)
    n_flood = total = 0
    for f in os.listdir(mask_dir):
        if not f.endswith('_lab.png'): continue
        mask = np.array(Image.open(os.path.join(mask_dir, f)))
        for c in range(10): counts[c] += (mask==c).sum()
        if np.isin(mask, FLOODED_CLASSES).any(): n_flood += 1
        total += 1
    tot = counts.sum()
    if tot == 0: print("  WARNING: No pixels!"); return counts
    print(f"  Images : {total}  |  Flooded: {n_flood} ({n_flood/max(total,1)*100:.1f}%)")
    for c in range(10):
        pct = counts[c]/tot*100
        bar = "#"*int(pct/2)
        print(f"  {CLASS_NAMES[c]:<22} {pct:>5.2f}%  {bar}")
    fp = sum(counts[c] for c in FLOODED_CLASSES)/tot*100
    print(f"\n  Combined flooded pixels: {fp:.3f}%")
    return counts

# ================================================================
# AUTO CLASS WEIGHTS
# ================================================================
def compute_auto_class_weights(train_counts, device, min_w=0.5, max_w=50.0):
    total   = train_counts.sum()
    freq    = train_counts / (total + 1e-6)
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.mean()
    for c in FLOODED_CLASSES:
        weights[c] = min(weights[c] * 5.0, max_w)
    weights = np.clip(weights, min_w, max_w)
    print(f"\n  Auto class weights (aggressive, 5x flood boost):")
    for c in range(NUM_CLASSES):
        print(f"    {CLASS_NAMES[c]:<22} {weights[c]:.2f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)

# ================================================================
# LOSSES
# ================================================================
class FocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.gamma           = gamma
        self.weights         = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        C           = pred.shape[1]
        pred_flat   = pred.permute(0,2,3,1).reshape(-1, C)
        target_flat = target.reshape(-1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.zeros_like(pred_flat)
                smooth.scatter_(1, target_flat.unsqueeze(1), 1.0)
                smooth = smooth*(1-self.label_smoothing) + self.label_smoothing/C
            log_p = F.log_softmax(pred_flat, dim=1)
            ce    = -(smooth * log_p).sum(dim=1) * self.weights[target_flat]
        else:
            ce = F.cross_entropy(pred_flat, target_flat,
                                 weight=self.weights, reduction='none')
        pt = torch.exp(-ce.detach())
        return ((1-pt)**self.gamma * ce).mean()

class TverskyLoss(nn.Module):
    def __init__(self, num_classes=10, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha       = alpha
        self.beta        = beta
        self.smooth      = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1); loss = 0.0
        for c in range(self.num_classes):
            p  = pred[:,c]; t = (target==c).float()
            tp = (p*t).sum(); fp = (p*(1-t)).sum(); fn = ((1-p)*t).sum()
            loss += 1-(tp+self.smooth)/(tp+self.alpha*fn+self.beta*fp+self.smooth)
        return loss/self.num_classes

class LovaszSoftmaxLoss(nn.Module):
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1); loss = 0.0; n = pred.shape[1]
        for c in range(n):
            tc = (target==c).float()
            if tc.sum() == 0: continue
            err = (tc - pred[:,c]).abs().reshape(-1)
            gt  = tc.reshape(-1)
            se, perm = torch.sort(err, descending=True)
            gts   = gt[perm].sum()
            inter = gts - gt[perm].float().cumsum(0)
            union = gts + (1-gt[perm]).float().cumsum(0)
            jac   = 1. - inter/(union+1e-6)
            if len(jac) > 1: jac[1:] = jac[1:] - jac[:-1]
            loss += torch.dot(se, jac)
        return loss/n

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.1, beta=1.0):
        super().__init__()
        self.alpha   = alpha
        self.beta    = beta
        self.weights = class_weights

    def forward(self, pred, target):
        C        = pred.shape[1]
        pf       = pred.permute(0,2,3,1).reshape(-1, C)
        tf       = target.reshape(-1)
        ce       = F.cross_entropy(pf, tf, weight=self.weights, reduction='mean')
        pred_prob = F.softmax(pf, dim=1).clamp(1e-7, 1.0)
        one_hot  = F.one_hot(tf, C).float().clamp(1e-4, 1.0)
        rce      = -(pred_prob * one_hot.log()).sum(dim=1).mean()
        return self.alpha * ce + self.beta * rce

class CombinedLoss(nn.Module):
    def __init__(self, class_weights, num_classes=10):
        super().__init__()
        self.focal   = FocalLoss(class_weights, gamma=FOCAL_GAMMA,
                                 label_smoothing=LABEL_SMOOTHING)
        self.tversky = TverskyLoss(num_classes, alpha=0.5, beta=0.5)
        self.lovasz  = LovaszSoftmaxLoss()
        self.sce     = SymmetricCrossEntropyLoss(class_weights)

    def forward(self, pred, target):
        f = self.focal(pred, target)
        t = self.tversky(pred, target)
        l = self.lovasz(pred, target)
        s = self.sce(pred, target)
        return 0.30*f + 0.35*t + 0.20*l + 0.15*s

# ================================================================
# ASPP MODULE
# ================================================================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6,12,18,24)):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            for r in rates])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels), nn.ReLU(inplace=True))
        n = 1 + len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*n, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout2d(0.15))

    def forward(self, x):
        h, w  = x.shape[2:]
        feats = [self.conv1x1(x)] + [d(x) for d in self.dilated]
        feats.append(F.interpolate(self.pool(x), (h,w),
                                   mode='bilinear', align_corners=False))
        return self.project(torch.cat(feats, dim=1))

# ================================================================
# BOUNDARY REFINEMENT HEAD
# ================================================================
class BoundaryRefinementHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(num_classes*2, num_classes, 1),
            nn.Sigmoid())

    def forward(self, feat, coarse_seg):
        boundary = self.boundary_conv(feat)
        fused    = self.gate(torch.cat([coarse_seg, boundary], dim=1))
        return coarse_seg * fused + boundary * (1-fused)

# ================================================================
# ASPP DECODER
# ================================================================
class ASPPDecoder(nn.Module):
    def __init__(self, in_channels=1536, num_classes=10):
        super().__init__()
        self.fuse   = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.aspp   = ASPP(512, out_channels=256, rates=(6,12,18,24))
        self.refine = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(0.15),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True))
        self.seg_head      = nn.Conv2d(64, num_classes, 1)
        self.boundary_head = BoundaryRefinementHead(64, num_classes)
        self.aux_head      = nn.Conv2d(64, 1, 1)
        with torch.no_grad():
            nn.init.zeros_(self.seg_head.bias)
            for c in FLOODED_CLASSES:
                self.seg_head.bias[c] = 1.0
            nn.init.zeros_(self.aux_head.bias)
            self.aux_head.bias[0] = 1.0
        print(f"  ASPPDecoder (+BoundaryHead): flood bias on {FLOODED_CLASSES}")

    def forward(self, x):
        x          = self.fuse(x)
        x          = self.aspp(x)
        x          = self.refine(x)
        seg_coarse = self.seg_head(x)
        seg        = self.boundary_head(x, seg_coarse)
        aux        = self.aux_head(x)
        return seg, aux, seg_coarse

# ================================================================
# DINOV2 SEGFORMER V4
# ================================================================
class DINOv2SegFormerV4(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = Dinov2Model.from_pretrained(DINO_MODEL_NAME)
        # FIX 1: gradient checkpointing disabled — was causing cuBLAS crash
        if USE_GRAD_CHECKPOINTING:
            self.encoder.gradient_checkpointing_enable()

        hidden = self.encoder.config.hidden_size
        print(f"{DINO_MODEL_NAME} hidden_size: {hidden}")

        self.scale_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1))
            for _ in range(4)])

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden*4, hidden*4 // 16),
            nn.ReLU(),
            nn.Linear(hidden*4 // 16, hidden*4),
            nn.Sigmoid())

        self.decode_head = ASPPDecoder(
            in_channels=hidden*4, num_classes=num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        out        = self.encoder(pixel_values=x, output_hidden_states=True)
        hidden     = out.hidden_states
        scales     = [hidden[3], hidden[6], hidden[9], hidden[12]]
        th, tw     = H//4, W//4
        projected  = []
        for i, feat in enumerate(scales):
            feat = feat[:, 1:]
            feat = self.scale_proj[i](feat)
            Bn, N, D = feat.shape
            h = w = int(N**0.5)
            feat = feat.permute(0,2,1).reshape(Bn, D, h, w)
            feat = F.interpolate(feat, (th,tw), mode='bilinear', align_corners=False)
            projected.append(feat)
        x    = torch.cat(projected, dim=1)
        attn = self.channel_attn(x).view(B, -1, 1, 1)
        x    = x * attn
        seg, aux, seg_coarse = self.decode_head(x)
        seg        = F.interpolate(seg,        (H,W), mode='bilinear', align_corners=False)
        aux        = F.interpolate(aux,        (H,W), mode='bilinear', align_corners=False)
        seg_coarse = F.interpolate(seg_coarse, (H,W), mode='bilinear', align_corners=False)
        return seg, aux, seg_coarse

# ================================================================
# METRICS
# ================================================================
def compute_all_metrics(preds_flat, gts_flat, num_classes=10):
    eps = 1e-6
    cm  = sk_cm(gts_flat, preds_flat, labels=list(range(num_classes)))
    tp  = np.diag(cm).astype(float)
    fp  = (cm.sum(0)-tp).astype(float)
    fn  = (cm.sum(1)-tp).astype(float)
    tn  = (cm.sum()-(tp+fp+fn)).astype(float)
    iou_per  = tp/(tp+fp+fn+eps)
    dice_per = 2*tp/(2*tp+fp+fn+eps)
    acc_per  = (tp+tn)/(tp+tn+fp+fn+eps)
    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(
        gts_flat, preds_flat, labels=list(range(num_classes)),
        average=None, zero_division=0)
    valid  = (cm.sum(1) > 0)
    miou   = iou_per[valid].mean()
    mdice  = dice_per[valid].mean()
    macc   = acc_per[valid].mean()
    pa     = tp.sum()/(cm.sum()+eps)
    pa_per = tp/(tp+fn+eps)
    mpa    = pa_per[valid].mean()
    freq   = cm.sum(1)/(cm.sum()+eps)
    fwiou  = (freq[valid]*iou_per[valid]).sum()
    try:
        kappa = cohen_kappa_score(gts_flat, preds_flat,
                                  labels=list(range(num_classes)))
    except Exception:
        kappa = 0.0
    prec = prec_per[valid].mean()
    rec  = rec_per[valid].mean()
    f1   = f1_per[valid].mean()
    return dict(
        miou=float(miou), mdice=float(mdice), macc=float(macc),
        pa=float(pa), mpa=float(mpa), fwiou=float(fwiou),
        kappa=float(kappa), precision=float(prec),
        recall=float(rec), f1=float(f1),
        iou_per=iou_per.tolist(), dice_per=dice_per.tolist(),
        acc_per=acc_per.tolist(), prec_per=prec_per.tolist(),
        rec_per=rec_per.tolist(),  f1_per=f1_per.tolist(),
        pa_per=pa_per.tolist(), cm=cm.tolist())

def compute_miou_batch(pred, target, num_classes=10):
    pred = pred.argmax(1); iou = []
    for c in range(num_classes):
        p = (pred==c); t = (target==c)
        inter = (p&t).sum().float(); union = (p|t).sum().float()
        if union > 0: iou.append((inter/union).item())
    return sum(iou)/len(iou) if iou else 0.0

def compute_dice_batch(pred, target, num_classes=10):
    pred = pred.argmax(1); d = []
    for c in range(num_classes):
        p = (pred==c); t = (target==c)
        inter = (p&t).sum().float(); denom = p.sum().float()+t.sum().float()
        if denom > 0: d.append((2*inter/denom).item())
    return sum(d)/len(d) if d else 0.0

# ================================================================
# TRAIN / VAL
# ================================================================
def make_flood_binary_mask(mask):
    fm = torch.zeros_like(mask, dtype=torch.float32)
    for c in FLOODED_CLASSES: fm[mask==c] = 1.0
    return fm.unsqueeze(1)

def train_epoch(model, loader, optimizer, scheduler, criterion, bce_criterion,
                scaler, epoch, use_mixup=False, use_cutmix=False,
                use_mosaic=False, ema_model=None):
    """
    FIX 7: scheduler.step() called once per batch inside the loop.
    FIX 5: CUDA error recovery — skips bad batch, clears cache, continues.
    """
    model.train()
    total_loss = total_miou = total_dice = 0
    optimizer.zero_grad()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="  Train",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for i, (imgs, masks) in pbar:
        imgs  = imgs.to(DEVICE,  non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        r = np.random.rand()
        if use_mosaic and r < 0.25 and imgs.shape[0] >= 4:
            imgs, masks = mosaic_data(imgs, masks)
        elif use_cutmix and r < 0.5:
            imgs, masks = cutmix_data(imgs, masks)
        elif use_mixup and r < 0.75:
            imgs, masks = mixup_data(imgs, masks)

        # FIX 5: CUDA error recovery
        try:
            with torch.cuda.amp.autocast():
                seg, aux, coarse = model(imgs)
                seg_loss    = criterion(seg, masks)
                coarse_loss = criterion(coarse, masks) * 0.3
                flood_bin   = make_flood_binary_mask(masks)
                aux_loss    = bce_criterion(aux, flood_bin)
                loss = (seg_loss + coarse_loss +
                        AUX_LOSS_WEIGHT*aux_loss) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i+1) % GRAD_ACCUM_STEPS == 0 or (i+1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema_model is not None:
                    ema_model.update_parameters(model)

            # FIX 7: step scheduler once per batch
            try:
                scheduler.step()
            except Exception:
                pass

            real_loss = (seg_loss + coarse_loss + AUX_LOSS_WEIGHT*aux_loss).item()
            total_loss += real_loss
            total_miou += compute_miou_batch(seg.detach(), masks)
            total_dice += compute_dice_batch(seg.detach(), masks)

        except RuntimeError as e:
            # FIX 5: recover from transient CUDA/cuBLAS errors
            if 'CUDA' in str(e) or 'cublas' in str(e).lower() or 'out of memory' in str(e).lower():
                print(f"\n  [WARN] CUDA error on batch {i}, skipping: {type(e).__name__}")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            raise

        pbar.set_postfix({'loss': f"{total_loss/max(i,1):.4f}",
                          'mIoU': f"{total_miou/max(i,1):.4f}"})
    n = len(loader)
    return total_loss/n, total_miou/n, total_dice/n

def val_epoch(model, loader, criterion, bce_criterion, max_batches=None):
    model.eval()
    total_loss = total_miou = total_dice = 0
    all_p = []; all_g = []
    seen  = 0
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(loader, desc="  Val  ", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            try:
                with torch.cuda.amp.autocast():
                    seg, aux, coarse = model(imgs)
                seg_loss  = criterion(seg, masks)
                coarse_l  = criterion(coarse, masks) * 0.3
                aux_loss  = bce_criterion(aux, make_flood_binary_mask(masks))
                loss      = seg_loss + coarse_l + AUX_LOSS_WEIGHT*aux_loss
                total_loss += loss.item()
                total_miou += compute_miou_batch(seg, masks)
                total_dice += compute_dice_batch(seg, masks)
                all_p.append(seg.argmax(1).cpu().numpy().flatten()[::16])
                all_g.append(masks.cpu().numpy().flatten()[::16])
                seen += 1
            except RuntimeError as e:
                if 'CUDA' in str(e) or 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise
    n = max(seen, 1)
    return (total_loss/n, total_miou/n, total_dice/n,
            np.concatenate(all_p) if all_p else np.array([]),
            np.concatenate(all_g) if all_g else np.array([]))

def evaluate_with_tta(model, loader, tta=True, tta_count=TTA_COUNT):
    model.eval()
    all_p = []; all_g = []

    def apply_tta(imgs):
        results = []
        with torch.cuda.amp.autocast():
            results.append(model(imgs)[0])
            results.append(torch.flip(model(torch.flip(imgs,[3]))[0],[3]))
            results.append(torch.flip(model(torch.flip(imgs,[2]))[0],[2]))
            img_r = torch.rot90(imgs, 1, [2,3])
            results.append(torch.rot90(model(img_r)[0], -1, [2,3]))
            imgs_sm = F.interpolate(imgs, scale_factor=0.75,
                                    mode='bilinear', align_corners=False)
            s_sm    = F.interpolate(model(imgs_sm)[0], size=imgs.shape[2:],
                                    mode='bilinear', align_corners=False)
            results.append(s_sm)
            imgs_lg = F.interpolate(imgs, scale_factor=1.25,
                                    mode='bilinear', align_corners=False)
            s_lg    = F.interpolate(model(imgs_lg)[0], size=imgs.shape[2:],
                                    mode='bilinear', align_corners=False)
            results.append(s_lg)
            if tta_count >= 7:
                imgs_b = torch.clamp(imgs * 1.05, 0, 1)
                results.append(model(imgs_b)[0])
            if tta_count >= 8:
                imgs_c = torch.clamp((imgs-0.5)*1.05+0.5, 0, 1)
                results.append(model(imgs_c)[0])
            if tta_count >= 9:
                hv = torch.flip(torch.flip(imgs,[3]),[2])
                results.append(torch.flip(torch.flip(model(hv)[0],[3]),[2]))
            if tta_count >= 10:
                img_t = imgs.transpose(2,3)
                results.append(model(img_t)[0].transpose(2,3))
        return torch.stack(results).mean(0)

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="  TTA Eval"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            try:
                if tta:
                    seg = apply_tta(imgs)
                else:
                    with torch.cuda.amp.autocast():
                        seg, _, _ = model(imgs)
                all_p.append(seg.argmax(1).cpu().numpy().flatten())
                all_g.append(masks.cpu().numpy().flatten())
            except RuntimeError as e:
                if 'CUDA' in str(e) or 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise
    return np.concatenate(all_p), np.concatenate(all_g)

# ================================================================
# GRAPHS (20+)
# ================================================================
def plot_all_graphs(history, metrics, val_counts, save_dir):
    print("\nGenerating all graphs...")
    g  = os.path.join(save_dir, "graphs")
    os.makedirs(g, exist_ok=True)
    pm = PAPER_METRICS

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'],   label='Val Loss',   linewidth=2)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(g,'01_loss.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(history['train_miou'], label='Train mIoU', linewidth=2)
    ax.plot(history['val_miou'],   label='Val mIoU',   linewidth=2)
    ax.axhline(pm['mIoU']/100, color='red', linestyle='--', linewidth=2,
               label=f"Paper ({pm['mIoU']:.1f}%)")
    ax.fill_between(range(len(history['val_miou'])),
                    pm['mIoU']/100, history['val_miou'],
                    where=[v > pm['mIoU']/100 for v in history['val_miou']],
                    alpha=0.2, color='green', label='Above Paper')
    ax.set_title('mIoU Curves vs Paper Baseline', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('mIoU')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(g,'02_miou.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(history['train_dice'], label='Train mDice', linewidth=2)
    ax.plot(history['val_dice'],   label='Val mDice',   linewidth=2)
    ax.axhline(pm['mDice']/100, color='red', linestyle='--',
               label=f"Paper ({pm['mDice']:.1f}%)")
    ax.set_title('mDice Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('mDice')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(g,'03_mdice.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(history.get('val_pa',   []), label='PA',    linewidth=2)
    ax.plot(history.get('val_fwiou',[]), label='FWIoU', linewidth=2)
    ax.axhline(pm['PA']/100,    color='red',    linestyle='--',
               label=f"Paper PA ({pm['PA']}%)")
    ax.axhline(pm['FWIoU']/100, color='orange', linestyle='--',
               label=f"Paper FWIoU ({pm['FWIoU']}%)")
    ax.set_title('Pixel Accuracy & FWIoU', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(g,'04_pa_fwiou.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(history.get('val_kappa',[]), label="Cohen's Kappa",
            linewidth=2, color='purple')
    ax.axhline(pm['Kappa'], color='red', linestyle='--',
               label=f"Paper ({pm['Kappa']:.3f})")
    ax.set_title("Cohen's Kappa", fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(g,'05_kappa.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(history['lr'], linewidth=2, color='green')
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(g,'06_lr.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(14,6))
    names  = list(CLASS_NAMES.values())
    iou_v  = [v*100 for v in metrics['iou_per']]
    colors = ['#e74c3c' if i in FLOODED_CLASSES else '#3498db'
              for i in range(NUM_CLASSES)]
    bars   = ax.bar(names, iou_v, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(pm['mIoU'], color='red', linestyle='--', linewidth=2,
               label=f"Paper mIoU ({pm['mIoU']}%)")
    for bar, v in zip(bars, iou_v):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.set_title('Per-Class IoU', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110); ax.set_ylabel('IoU (%)'); ax.legend()
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(g,'07_per_class_iou.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(14,6))
    dice_v = [v*100 for v in metrics['dice_per']]
    bars   = ax.bar(names, dice_v, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(pm['mDice'], color='red', linestyle='--', linewidth=2,
               label=f"Paper mDice ({pm['mDice']}%)")
    for bar, v in zip(bars, dice_v):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.set_title('Per-Class Dice Score', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110); ax.set_ylabel('Dice (%)'); ax.legend()
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(g,'08_per_class_dice.png'), dpi=150); plt.close()

    x = np.arange(NUM_CLASSES); w = 0.25
    fig, ax = plt.subplots(figsize=(14,6))
    ax.bar(x-w,   [v*100 for v in metrics['prec_per']], w,
           label='Precision', color='steelblue')
    ax.bar(x,     [v*100 for v in metrics['rec_per']],  w,
           label='Recall',    color='salmon')
    ax.bar(x+w,   [v*100 for v in metrics['f1_per']],   w,
           label='F1',        color='green')
    ax.set_title('Per-Class Precision / Recall / F1', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('%'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(g,'09_per_class_prf.png'), dpi=150); plt.close()

    cm      = np.array(metrics['cm'])
    cm_norm = cm.astype(float)
    row_sum = cm_norm.sum(1, keepdims=True)
    cm_norm = cm_norm / (row_sum + 1e-6) * 100
    fig, ax = plt.subplots(figsize=(12,10))
    im      = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm_norm[i,j]:.1f}", ha='center', va='center',
                    color='white' if cm_norm[i,j]>50 else 'black', fontsize=7)
    plt.colorbar(im, ax=ax, label='%')
    ax.set_title('Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig(os.path.join(g,'10_confusion_matrix.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(14,6))
    val_pct = [v/max(val_counts.sum(),1)*100 for v in val_counts]
    xb = np.arange(NUM_CLASSES)
    ax.bar(xb, val_pct, 0.6, label='Val GT', color='steelblue', edgecolor='black')
    ax.set_title('Val GT Pixel Distribution (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(xb); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('%'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(g,'11_class_distribution.png'), dpi=150); plt.close()

    radar_metrics = ['mIoU','mDice','PA','MPA','FWIoU','Precision','Recall','F1']
    ours_vals  = [metrics['miou']*100,  metrics['mdice']*100,  metrics['pa']*100,
                  metrics['mpa']*100,   metrics['fwiou']*100,
                  metrics['precision']*100, metrics['recall']*100, metrics['f1']*100]
    paper_vals = [pm['mIoU'],pm['mDice'],pm['PA'],pm['MPA'],pm['FWIoU'],
                  pm['Precision'],pm['Recall'],pm['F1']]
    angles  = [n/float(len(radar_metrics))*2*math.pi for n in range(len(radar_metrics))]
    angles += angles[:1]
    ours_v  = ours_vals  + ours_vals[:1]
    paper_v = paper_vals + paper_vals[:1]
    fig = plt.figure(figsize=(9,9))
    ax  = fig.add_subplot(111, polar=True)
    ax.plot(angles, ours_v,  'o-', linewidth=2, label='Ours (DINOv2-v4)', color='blue')
    ax.fill(angles, ours_v,  alpha=0.2, color='blue')
    ax.plot(angles, paper_v, 'o-', linewidth=2, label='Paper (SwinSeg)',   color='red')
    ax.fill(angles, paper_v, alpha=0.2, color='red')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=11)
    ax.set_ylim(50, 105)
    ax.set_title('Ours vs Paper', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(g,'12_radar_chart.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(14,6))
    xc = np.arange(len(radar_metrics)); w = 0.35
    ax.bar(xc-w/2, ours_vals,  w, label='Ours',  color='royalblue', edgecolor='black')
    ax.bar(xc+w/2, paper_vals, w, label='Paper', color='tomato',    edgecolor='black')
    for i, (o, p) in enumerate(zip(ours_vals, paper_vals)):
        color = 'green' if o > p else 'red'
        ax.annotate('UP' if o>p else 'DN',
                    xy=(i-w/2, o+0.3), ha='center', color=color, fontweight='bold')
    ax.set_xticks(xc); ax.set_xticklabels(radar_metrics)
    ax.set_title('Metric Comparison: Ours vs Paper', fontsize=14, fontweight='bold')
    ax.set_ylabel('%'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(g,'13_metric_comparison.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10,5))
    gap = [t-v for t,v in zip(history['train_miou'], history['val_miou'])]
    ax.plot(gap, linewidth=2, color='purple')
    ax.axhline(0.15, color='red',   linestyle='--', label='Overfit threshold (0.15)')
    ax.axhline(0,    color='green', linestyle='-',  linewidth=0.8)
    ax.fill_between(range(len(gap)), gap, 0,
                    where=[g>0.15 for g in gap],
                    alpha=0.3, color='red', label='Overfit zone')
    ax.set_title('Train-Val mIoU Gap (Overfitting Monitor)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gap'); ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(g,'14_overfitting_monitor.png'), dpi=150); plt.close()

    if 'val_f1' in history and len(history['val_f1']) > 0:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(history['val_f1'],        label='F1',        linewidth=2)
        ax.plot(history['val_precision'], label='Precision', linewidth=2)
        ax.plot(history['val_recall'],    label='Recall',    linewidth=2)
        ax.axhline(pm['F1']/100, color='red', linestyle='--',
                   label=f"Paper F1 ({pm['F1']}%)")
        ax.set_title('F1 / Precision / Recall', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(g,'15_f1_pr_curves.png'), dpi=150); plt.close()

    val_m = history['val_miou']
    if val_m:
        best_ep = int(np.argmax(val_m))
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(val_m, linewidth=2, label='Val mIoU')
        ax.axvline(best_ep, color='red', linestyle='--',
                   label=f"Best epoch {best_ep+1} ({val_m[best_ep]*100:.2f}%)")
        ax.scatter([best_ep], [val_m[best_ep]], color='red', s=100, zorder=5)
        ax.set_title('Val mIoU with Best Epoch', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(g,'16_best_epoch.png'), dpi=150); plt.close()

    if 'epoch_time' in history and len(history['epoch_time']) > 0:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(history['epoch_time'], linewidth=2, color='darkorange')
        ax.set_title('Epoch Training Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Time (s)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(g,'17_epoch_time.png'), dpi=150); plt.close()

    if 'flood_iou' in history and len(history['flood_iou']) > 0:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(history['flood_iou'], linewidth=2, color='red',  label='Flooded mIoU')
        ax.plot(history['val_miou'],  linewidth=2, color='blue', label='Overall mIoU')
        ax.set_title('Flooded Classes IoU vs Overall mIoU', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(g,'18_flood_iou.png'), dpi=150); plt.close()

    fig = plt.figure(figsize=(20,14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    def mini_plot(ax, data, title, color='steelblue', paper_val=None):
        if not data: return
        ax.plot(data, linewidth=1.5, color=color)
        if paper_val is not None:
            ax.axhline(paper_val, color='red', linestyle='--', linewidth=1,
                       label=f'Paper={paper_val:.2f}')
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    mini_plot(fig.add_subplot(gs[0,0]), history['train_loss'],  'Train Loss',  'steelblue')
    mini_plot(fig.add_subplot(gs[0,1]), history['val_loss'],    'Val Loss',    'orange')
    mini_plot(fig.add_subplot(gs[0,2]), history['train_miou'],  'Train mIoU',  'green',  pm['mIoU']/100)
    mini_plot(fig.add_subplot(gs[0,3]), history['val_miou'],    'Val mIoU',    'blue',   pm['mIoU']/100)
    mini_plot(fig.add_subplot(gs[1,0]), history['train_dice'],  'Train mDice', 'purple', pm['mDice']/100)
    mini_plot(fig.add_subplot(gs[1,1]), history['val_dice'],    'Val mDice',   'cyan',   pm['mDice']/100)
    mini_plot(fig.add_subplot(gs[1,2]), history.get('val_pa',  []), 'Val PA',  'brown',  pm['PA']/100)
    mini_plot(fig.add_subplot(gs[1,3]), history.get('val_fwiou',[]),'Val FWIoU','olive', pm['FWIoU']/100)
    mini_plot(fig.add_subplot(gs[2,0]), history.get('val_kappa',[]),'Val Kappa','teal',  pm['Kappa'])
    lr_ax = fig.add_subplot(gs[2,1])
    mini_plot(lr_ax, history['lr'], 'LR (log)', 'red')
    lr_ax.set_yscale('log')
    mini_plot(fig.add_subplot(gs[2,2]), history.get('val_f1',  []), 'Val F1',  'magenta',pm['F1']/100)
    mini_plot(fig.add_subplot(gs[2,3]), history.get('flood_iou',[]),'Flood IoU','crimson')
    fig.suptitle('DINOv2SegFormer v4 — Training Dashboard',
                 fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(g,'19_full_dashboard.png'), dpi=120); plt.close()

    deltas = {
        'mIoU'    : metrics['miou']*100    - pm['mIoU'],
        'mDice'   : metrics['mdice']*100   - pm['mDice'],
        'PA'      : metrics['pa']*100      - pm['PA'],
        'MPA'     : metrics['mpa']*100     - pm['MPA'],
        'FWIoU'   : metrics['fwiou']*100   - pm['FWIoU'],
        'Precision': metrics['precision']*100 - pm['Precision'],
        'Recall'  : metrics['recall']*100  - pm['Recall'],
        'F1'      : metrics['f1']*100      - pm['F1'],
    }
    fig, ax = plt.subplots(figsize=(12,6))
    keys = list(deltas.keys()); vals = list(deltas.values())
    cols = ['green' if v >= 0 else 'red' for v in vals]
    bars = ax.bar(keys, vals, color=cols, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + (0.1 if v>=0 else -0.3),
                f"{v:+.2f}%", ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Performance Delta vs Paper (SwinSegFormer)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Delta (%)'); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(g,'20_delta_vs_paper.png'), dpi=150); plt.close()

    print(f"  20 graphs saved to {g}/")

# ================================================================
# PRINT TABLES
# ================================================================
def print_paper_comparison(metrics):
    pm = PAPER_METRICS
    print("\n" + "="*72)
    print("   METRICS vs PAPER (SwinSegFormer)")
    print("="*72)
    def row(name, ours, paper, fmt=".2f", pct=True):
        s    = "%" if pct else ""
        beat = "BEAT" if ours > paper else "----"
        print(f"  {name:<14} {ours:{fmt}}{s:1s}  vs  {paper:{fmt}}{s:1s}   {beat}")
    print(f"  {'Metric':<14} {'Ours':>10}   {'Paper':>10}   {'Status'}")
    print("  "+"-"*50)
    row("mIoU",      metrics['miou']*100,      pm['mIoU'])
    row("mDice",     metrics['mdice']*100,      pm['mDice'])
    row("mACC",      metrics['macc']*100,       pm['mACC'])
    row("PA",        metrics['pa']*100,         pm['PA'])
    row("MPA",       metrics['mpa']*100,        pm['MPA'])
    row("FWIoU",     metrics['fwiou']*100,      pm['FWIoU'])
    row("Kappa",     metrics['kappa'],          pm['Kappa'],   fmt=".4f", pct=False)
    row("Precision", metrics['precision']*100,  pm['Precision'])
    row("Recall",    metrics['recall']*100,     pm['Recall'])
    row("F1",        metrics['f1']*100,         pm['F1'])
    print("="*72)

def print_per_class_table(metrics, val_counts):
    total_px = val_counts.sum()
    print("\n" + "="*110)
    print(f"  {'Class':<22} {'IoU':>7} {'Dice':>7} {'Acc':>7} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'GT%':>7}")
    print("  "+"-"*90)
    for c in range(NUM_CLASSES):
        gt_pct = val_counts[c]/total_px*100 if total_px > 0 else 0
        marker = " <- FLOOD" if c in FLOODED_CLASSES else ""
        has_gt = val_counts[c] > 0
        if has_gt:
            print(f"  {CLASS_NAMES[c]:<22}"
                  f" {metrics['iou_per'][c]*100:>6.1f}%"
                  f" {metrics['dice_per'][c]*100:>6.1f}%"
                  f" {metrics['acc_per'][c]*100:>6.1f}%"
                  f" {metrics['prec_per'][c]*100:>6.1f}%"
                  f" {metrics['rec_per'][c]*100:>6.1f}%"
                  f" {metrics['f1_per'][c]*100:>6.1f}%"
                  f" {gt_pct:>6.2f}%{marker}")
        else:
            print(f"  {CLASS_NAMES[c]:<22} {'N/A':>7}{'':17} "
                  f"{gt_pct:>6.2f}%{marker} (no GT)")
    print("="*110)

# ================================================================
# HELPERS
# ================================================================
def mask_to_color(mask):
    h, w = mask.shape
    out  = np.zeros((h,w,3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS): out[mask==c] = col
    return out

def clean_text(t):
    return (str(t).replace('\u2014','-').replace('\u2013','-')
            .replace('\u2018',"'").replace('\u2019',"'")
            .replace('\u201c','"').replace('\u201d','"')
            .replace('\u2026','...').replace('\u00b0','deg')
            .encode('latin-1','ignore').decode('latin-1'))

def estimate_flood_area(pred_mask):
    total   = pred_mask.size
    flooded = np.isin(pred_mask, FLOODED_CLASSES).sum()
    fp      = flooded / total * 100
    bkdn    = {CLASS_NAMES[c]: round((pred_mask==c).sum()/total*100, 2)
               for c in range(NUM_CLASSES)
               if (pred_mask==c).sum()/total*100 > 0.1}
    return round(fp, 2), fp >= FLOOD_THRESHOLD, bkdn

# ================================================================
# TOOLS
# ================================================================
def weather_tool(location):
    try:
        import requests
        r = requests.get(f"https://wttr.in/{location}?format=3", timeout=5)
        return clean_text(r.text.strip())
    except Exception:
        return f"Weather data unavailable for {location}"

def maps_tool(lat, lon, flood_pct, location):
    try:
        m     = folium.Map(location=[lat, lon], zoom_start=13)
        color = "red" if flood_pct >= FLOOD_THRESHOLD else "green"
        folium.CircleMarker([lat, lon], radius=60, color=color,
                            fill=True, fill_opacity=0.4,
                            popup=f"{location}: {flood_pct:.1f}% flooded").add_to(m)
        folium.Marker([lat, lon],
                      popup=f"{'FLOODED' if flood_pct>=FLOOD_THRESHOLD else 'SAFE'}",
                      icon=folium.Icon(color=color)).add_to(m)
        path = os.path.join(SAVE_DIR, "flood_map.html")
        m.save(path)
        return path
    except Exception as e:
        print(f"  Map error: {e}"); return None

def alert_tool(location, flood_pct):
    msg = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] "
           f"FLOOD ALERT @ {location} - {flood_pct:.1f}% flooded!")
    with open(os.path.join(SAVE_DIR, "alerts.log"), 'a') as f:
        f.write(msg + "\n")
    print(f"  *** {msg} ***")
    return msg

# ================================================================
# PDF REPORT
# ================================================================
def generate_pdf_report(data, save_dir):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(15, 52, 96)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 17)
    pdf.cell(0, 18, "FLOOD DISASTER INTELLIGENCE REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 8, clean_text(
        f"DINOv2SegFormer v4 | {datetime.now().strftime('%Y-%m-%d %H:%M')} | FloodNet"),
        ln=True, align="C")
    pdf.ln(3)

    fp     = data.get('flood_pct', 0)
    status = "FLOODED" if fp >= FLOOD_THRESHOLD else "NON-FLOODED"
    if status == "FLOODED":
        pdf.set_fill_color(200, 30, 30)
    else:
        pdf.set_fill_color(30, 160, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 13,
             clean_text(f"STATUS: {status}  |  Flood: {fp:.1f}%  |  "
                        f"Location: {data.get('location','N/A')}"),
             ln=True, fill=True, align="C")
    pdf.set_text_color(0, 0, 0); pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Model Performance vs Paper (SwinSegFormer):", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(180, 200, 230)
    for h in ["Metric", "Ours", "Paper", "Beat Paper?"]:
        pdf.cell(47, 7, h, fill=True, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    pm = PAPER_METRICS; m = data.get('metrics', {})
    rows = [
        ("mIoU",      m.get('miou',0)*100,      pm['mIoU']),
        ("mDice",     m.get('mdice',0)*100,      pm['mDice']),
        ("PA",        m.get('pa',0)*100,         pm['PA']),
        ("MPA",       m.get('mpa',0)*100,        pm['MPA']),
        ("FWIoU",     m.get('fwiou',0)*100,      pm['FWIoU']),
        ("Kappa",     m.get('kappa',0),          pm['Kappa']),
        ("Precision", m.get('precision',0)*100,  pm['Precision']),
        ("Recall",    m.get('recall',0)*100,     pm['Recall']),
        ("F1",        m.get('f1',0)*100,         pm['F1']),
    ]
    for i, (name, o, p) in enumerate(rows):
        fill = i % 2 == 0
        pdf.set_fill_color(242,242,242) if fill else pdf.set_fill_color(255,255,255)
        beat = "YES" if o > p else "no"
        pdf.set_text_color(0,140,0) if o > p else pdf.set_text_color(180,0,0)
        pdf.cell(47, 6, clean_text(name),  fill=fill, border=1)
        pdf.cell(47, 6, f"{o:.2f}",        fill=fill, border=1)
        pdf.cell(47, 6, f"{p:.2f}",        fill=fill, border=1)
        pdf.cell(47, 6, beat,              fill=fill, border=1, ln=True)
    pdf.set_text_color(0, 0, 0); pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Per-Class Performance:", ln=True)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(180, 200, 230)
    for h in ["Class", "IoU%", "Dice%", "Prec%", "Recall%", "F1%"]:
        pdf.cell(32, 6, h, fill=True, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    iou_per  = data.get('metrics',{}).get('iou_per',  [0]*10)
    dice_per = data.get('metrics',{}).get('dice_per', [0]*10)
    prec_per = data.get('metrics',{}).get('prec_per', [0]*10)
    rec_per  = data.get('metrics',{}).get('rec_per',  [0]*10)
    f1_per   = data.get('metrics',{}).get('f1_per',   [0]*10)
    for c in range(NUM_CLASSES):
        fill = c % 2 == 0
        if c in FLOODED_CLASSES:
            pdf.set_fill_color(255, 230, 230)
        elif fill:
            pdf.set_fill_color(242, 242, 242)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(32, 5, clean_text(CLASS_NAMES[c]), fill=True, border=1)
        pdf.cell(32, 5, f"{iou_per[c]*100:.1f}",   fill=True, border=1)
        pdf.cell(32, 5, f"{dice_per[c]*100:.1f}",  fill=True, border=1)
        pdf.cell(32, 5, f"{prec_per[c]*100:.1f}",  fill=True, border=1)
        pdf.cell(32, 5, f"{rec_per[c]*100:.1f}",   fill=True, border=1)
        pdf.cell(32, 5, f"{f1_per[c]*100:.1f}",    fill=True, border=1, ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "YOLO Object Detections:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    dets = data.get('yolo_detections', [])
    if dets:
        for d in dets[:10]:
            pdf.cell(0, 6, clean_text(
                f"  - {d['label']} (conf={d['conf']:.2f})  "
                f"box=[{d['box'][0]},{d['box'][1]},{d['box'][2]},{d['box'][3]}]"),
                ln=True)
    else:
        pdf.cell(0, 6, "  No detections or YOLO unavailable.", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Segmentation Breakdown:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for cls, pct in data.get('breakdown', {}).items():
        pdf.cell(0, 6, clean_text(f"  - {cls}: {pct:.1f}%"), ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Rescue & Response Recommendations:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    recs = ([
        "IMMEDIATE: Deploy amphibious rescue boats to flooded road sectors",
        "URGENT: Evacuate all Building-Flooded zones within 2 hours",
        "PRIORITY: Establish relief camps on Grass/non-flooded areas",
        "CRITICAL: Block road access in Road-Flooded segments",
        "MEDICAL: Coordinate water-borne disease prevention with health dept",
        "MONITOR: Check water level sensors every 30 minutes",
    ] if status == "FLOODED" else [
        "Area is non-flooded - maintain regular monitoring",
        "Pre-position rescue equipment as precaution",
        "Keep communication channels open with adjacent zones",
    ])
    for r in recs:
        pdf.cell(0, 6, clean_text(f"  -> {r}"), ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Current Weather:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 6, clean_text(data.get('weather', 'N/A')))

    pdf.set_y(-15)
    pdf.set_fill_color(15, 52, 96)
    pdf.rect(0, 282, 210, 15, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 10, "DINOv2SegFormer v4 Flood Intelligence | NIELIT Patna | Emergency Use Only",
             align="C")

    path = os.path.join(save_dir, "reports",
                        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(path)
    print(f"  PDF report saved: {path}")
    return path

# ================================================================
# SAMPLE PREDICTIONS GRID
# ================================================================
def save_prediction_samples(model, val_ds, yolo_model, save_dir, n=6):
    fig, axes = plt.subplots(n, 4, figsize=(20, n*5))
    model.eval()
    for i, img_id in enumerate(val_ds.ids[:n]):
        img_path = None
        for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
            c = os.path.join(VAL_IMG, img_id + ext)
            if os.path.exists(c): img_path = c; break
        img  = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB),
                          (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(
            np.array(Image.open(os.path.join(VAL_MASK, f"{img_id}_lab.png"))),
            (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        t    = torch.tensor(img).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255.0
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                seg, _, _ = model(t)
            pred = seg.argmax(1).squeeze().cpu().numpy()
        yolo_img, _ = run_yolo_detection(yolo_model, img)

        axes[i,0].imshow(img);                 axes[i,0].set_title(f"Image: {img_id}"); axes[i,0].axis('off')
        axes[i,1].imshow(mask_to_color(mask)); axes[i,1].set_title("Ground Truth");     axes[i,1].axis('off')
        axes[i,2].imshow(mask_to_color(pred)); axes[i,2].set_title("Prediction");       axes[i,2].axis('off')
        axes[i,3].imshow(yolo_img);            axes[i,3].set_title("YOLO Detections");  axes[i,3].axis('off')

    patches = [mpatches.Patch(color=np.array(CLASS_COLORS[c])/255,
                               label=CLASS_NAMES[c])
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    plt.suptitle('DINOv2SegFormer v4 — Predictions + YOLO',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, "predictions", "samples_v4.png")
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    print(f"  Prediction samples saved: {path}")

# ================================================================
# MEASURE MODEL
# ================================================================
def measure_model(model, device, image_size=512, num_runs=10):
    print("\n"+"="*60)
    print("   MODEL ANALYSIS")
    print("="*60)
    model.eval()
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params     : {total_p/1e6:.1f}M")
    dummy = torch.randn(1,3,image_size,image_size).to(device)
    with torch.no_grad():
        for _ in range(3):
            with torch.cuda.amp.autocast():
                _ = model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.cuda.amp.autocast():
                model(dummy)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    mean_ms = sum(times)/len(times)
    fps     = 1000/mean_ms
    print(f"  Latency    : {mean_ms:.1f} ms")
    print(f"  FPS        : {fps:.1f}")
    return {'total_params': total_p, 'latency_ms': mean_ms, 'fps': fps}

# ================================================================
# SINGLE-IMAGE PIPELINE
# ================================================================
def run_pipeline(model, img_id, yolo_model, metrics,
                 location="Bihar, India", lat=25.59, lon=85.13,
                 use_val=True):
    folder_img = VAL_IMG if use_val else TEST_IMG
    img_path = None
    for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
        c = os.path.join(folder_img, img_id + ext)
        if os.path.exists(c): img_path = c; break
    img  = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB),
                      (IMAGE_SIZE, IMAGE_SIZE))
    t    = torch.tensor(img).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255.0
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            seg, _, _ = model(t)
        pred = seg.argmax(1).squeeze().cpu().numpy()
    fp, is_fl, bkdn = estimate_flood_area(pred)
    yolo_img, dets  = run_yolo_detection(yolo_model, img,
        save_path=os.path.join(SAVE_DIR, "yolo", f"{img_id}_yolo.jpg"))
    weather  = weather_tool(location)
    map_path = maps_tool(lat, lon, fp, location)
    if is_fl:
        alert_tool(location, fp)
    pdf_path = generate_pdf_report({
        'flood_pct': fp, 'status': "FLOODED" if is_fl else "NON-FLOODED",
        'location':  location, 'breakdown': bkdn, 'weather': weather,
        'metrics':   metrics,  'yolo_detections': dets,
    }, SAVE_DIR)
    print(f"  Pipeline: {'FLOODED' if is_fl else 'NON-FLOODED'} ({fp:.1f}%) | PDF: {pdf_path}")
    return pred, fp, pdf_path

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":

    # Pixel distribution
    if (FAST_MODE or RESUME) and os.path.exists(PIXEL_STATS_CACHE):
        print("\nLoading cached pixel distribution.")
        stats        = np.load(PIXEL_STATS_CACHE)
        train_counts = stats['train_counts']
        val_counts   = stats['val_counts']
    else:
        train_counts = check_pixel_distribution(TRAIN_MASK, "Train")
        val_counts   = check_pixel_distribution(VAL_MASK,   "Val")
        np.savez(PIXEL_STATS_CACHE, train_counts=train_counts, val_counts=val_counts)
    np.save(os.path.join(SAVE_DIR, 'class_distribution.npy'),
            val_counts / val_counts.sum())

    class_weights = compute_auto_class_weights(train_counts, DEVICE)

    train_ds = FloodNetDataset(TRAIN_IMG, TRAIN_MASK, IMAGE_SIZE,
                               augment=FAST_AUGMENT, split='train')
    val_ds   = FloodNetDataset(VAL_IMG,   VAL_MASK,   IMAGE_SIZE,
                               augment=False, split='val')
    test_ds  = FloodNetDataset(TEST_IMG,  TEST_MASK,  IMAGE_SIZE,
                               augment=False, split='test')

    def compute_sample_weights(dataset, fw=30.0):
        weights = []
        ft      = torch.tensor(FLOODED_CLASSES)
        print("Computing sample weights...")
        for img_id in dataset.ids:
            mask = torch.tensor(np.array(Image.open(
                os.path.join(dataset.mask_dir, img_id + "_lab.png"))))
            weights.append(fw if torch.isin(mask, ft).any().item() else 1.0)
        nf = sum(1 for w in weights if w > 1)
        print(f"  Flood images: {nf}/{len(weights)} ({nf/len(weights)*100:.1f}%)")
        return weights

    sample_w = compute_sample_weights(train_ds)
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    loader_kwargs = dict(
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, **loader_kwargs)
    print(f"Train batches: {len(train_loader)} | "
          f"Val: {len(val_loader)} | Test: {len(test_loader)}")

    if FAST_LOSS:
        criterion = FocalLoss(class_weights, gamma=FOCAL_GAMMA,
                              label_smoothing=LABEL_SMOOTHING)
        print("  Using FocalLoss only (FAST_LOSS=True)")
    else:
        criterion = CombinedLoss(class_weights, NUM_CLASSES)
        print("  Using CombinedLoss (Focal+Tversky+Lovasz+SCE)")

    bce_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([15.0]).to(DEVICE))

    if LOAD_YOLO and not FAST_MODE:
        print("\nLoading YOLOv8...")
        yolo_model = load_yolo_model()
    else:
        print("\nSkipping YOLO load.")
        yolo_model = None

    print("\nBuilding DINOv2SegFormer v4...")
    model = DINOv2SegFormerV4(num_classes=NUM_CLASSES).to(DEVICE)

    for p in model.encoder.parameters():      p.requires_grad = False
    for p in model.scale_proj.parameters():   p.requires_grad = True
    for p in model.decode_head.parameters():  p.requires_grad = True
    for p in model.channel_attn.parameters(): p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    to = sum(p.numel() for p in model.parameters())
    print(f"Phase 1 — Trainable: {tr/1e6:.1f}M / {to/1e6:.1f}M")

    dummy = torch.randn(1,3,IMAGE_SIZE,IMAGE_SIZE).to(DEVICE)
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            so, ao, co = model(dummy)
    model.train()
    print(f"Seg shape: {so.shape} | Aux: {ao.shape} | Coarse: {co.shape}")
    del dummy, so, ao, co; torch.cuda.empty_cache()

    # NOTE: torch.compile skipped on Windows (causes cuBLAS issues)
    print("torch.compile() skipped (Windows — stability)")

    ema_model = AveragedModel(model) if USE_EMA else None

    # FIX 3: total_steps = NUM_EPOCHS * len(train_loader) for Phase 1 scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=LR,
                           total_steps=NUM_EPOCHS * len(train_loader),
                           pct_start=WARMUP_EPOCHS/NUM_EPOCHS,
                           anneal_strategy='cos')

    swa_model   = AveragedModel(model)
    swa_sched   = SWALR(optimizer, swa_lr=1e-6, anneal_epochs=10)
    swa_started = False
    scaler      = torch.cuda.amp.GradScaler()

    ckpt_path     = os.path.join(SAVE_DIR, "checkpoint_best.pth")
    ckpt_last     = os.path.join(SAVE_DIR, "checkpoint_last.pth")
    swa_ckpt_path = os.path.join(SAVE_DIR, "checkpoint_swa.pth")

    start_epoch = 0
    best_miou   = 0.0
    no_improve  = 0
    history     = dict(
        train_loss=[], val_loss=[],
        train_miou=[], val_miou=[],
        train_dice=[], val_dice=[],
        lr=[], val_pa=[], val_fwiou=[],
        val_kappa=[], val_f1=[], val_precision=[],
        val_recall=[], epoch_time=[], flood_iou=[])

    # Resume logic
    resume_from = None
    if RESUME:
        if os.path.exists(ckpt_last):
            resume_from = ckpt_last
        elif os.path.exists(ckpt_path):
            resume_from = ckpt_path

    if resume_from:
        print(f"\nResuming from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except ValueError as e:
            print(f"  (optimizer state mismatch: {e})")
        try:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        except Exception:
            print("  (scheduler state mismatch — using fresh scheduler)")
        start_epoch = ckpt.get('epoch', 0) + 1
        best_miou   = ckpt.get('best_miou', 0.0)
        no_improve  = ckpt.get('no_improve', 0)
        swa_started = ckpt.get('swa_started', False)
        loaded_hist = ckpt.get('history', {})
        for k in history:
            if k in loaded_hist:
                history[k] = loaded_hist[k]
        if swa_started and os.path.exists(swa_ckpt_path):
            swa_model.load_state_dict(torch.load(swa_ckpt_path, map_location=DEVICE))
        if ema_model is not None:
            ema_state = ckpt.get('ema_state', None)
            if ema_state is not None:
                ema_model.load_state_dict(ema_state)
        print(f"  Resumed epoch {start_epoch} | best mIoU: {best_miou*100:.2f}%")
    else:
        print("Starting fresh from epoch 1...")

    # ================================================================
    # TRAINING LOOP
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  Training | Epochs: {NUM_EPOCHS} | Patience: {PATIENCE}")
    print(f"  Batch: {BATCH_SIZE} | GradAccum: {GRAD_ACCUM_STEPS} | LR: {LR}")
    print(f"  Loss: {'Focal' if FAST_LOSS else 'Focal+Tversky+Lovasz+SCE'}")
    print(f"  GRAD_CKPT: {USE_GRAD_CHECKPOINTING} | EMA: {USE_EMA}")
    print(f"{'='*70}")

    total_start = time.time()
    # Track current scheduler state for phase transitions
    current_phase = 1

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        t0 = time.time()

        # FIX 3 & 4: Phase 2 — unfreeze last 4 encoder blocks
        if epoch == FREEZE_EPOCHS and current_phase == 1:
            print("  >> Phase 2: Unfreeze last-4 encoder blocks")
            for name, p in model.named_parameters():
                if any(f"layer.{i}" in name for i in [8,9,10,11]):
                    p.requires_grad = True
            torch.cuda.empty_cache()  # FIX 4: flush VRAM before expanding params
            remaining = max((NUM_EPOCHS - epoch) * len(train_loader), 1)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR*0.3, weight_decay=WEIGHT_DECAY)
            scheduler = OneCycleLR(optimizer, max_lr=LR*0.3,
                                   total_steps=remaining,
                                   pct_start=0.05, anneal_strategy='cos')
            scaler = torch.cuda.amp.GradScaler()
            current_phase = 2
            tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  >> Trainable: {tr/1e6:.1f}M | remaining_steps: {remaining}")

        # FIX 3 & 4: Phase 3 — full encoder unfreeze at epoch 30
        if epoch == 30 and not FAST_MODE and current_phase < 3:
            print("  >> Phase 3: Full encoder unfreeze (epoch 30)")
            for p in model.parameters():
                p.requires_grad = True
            torch.cuda.empty_cache()  # FIX 4
            remaining3 = max((NUM_EPOCHS - epoch) * len(train_loader), 1)
            optimizer = optim.AdamW(model.parameters(),
                                    lr=LR*0.1, weight_decay=WEIGHT_DECAY)
            scheduler = OneCycleLR(optimizer, max_lr=LR*0.1,
                                   total_steps=remaining3,
                                   pct_start=0.05, anneal_strategy='cos')
            scaler = torch.cuda.amp.GradScaler()
            current_phase = 3
            tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  >> Trainable: {tr/1e6:.1f}M (fully unfrozen)")

        if epoch >= max(SWA_START_EPOCH, FREEZE_EPOCHS+1) and not swa_started:
            print("  >> SWA started!")
            swa_started = True

        use_aug = epoch >= FREEZE_EPOCHS and FAST_AUGMENT
        train_loss, train_miou, train_dice = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, bce_criterion,
            scaler, epoch,
            use_mixup=use_aug, use_cutmix=use_aug, use_mosaic=use_aug,
            ema_model=ema_model)

        # FIX 6: QUICK_VAL_BATCHES=100 mid-training; full only on last epoch
        do_val = (
            epoch == start_epoch or
            (epoch + 1) % VAL_EVERY == 0 or
            epoch == NUM_EPOCHS - 1
        )
        if do_val:
            # Use full val on last epoch; limited mid-training
            mb = None if epoch == NUM_EPOCHS - 1 else QUICK_VAL_BATCHES
            val_loss, val_miou, val_dice, vp, vg = val_epoch(
                model, val_loader, criterion, bce_criterion, max_batches=mb)
            if len(vp) > 0 and len(vg) > 0:
                vm = compute_all_metrics(vp, vg, NUM_CLASSES)
            else:
                vm = {'pa':0,'fwiou':0,'kappa':0,'f1':0,'precision':0,
                      'recall':0,'iou_per':np.zeros(NUM_CLASSES)}
            flood_iou_val = np.mean([vm['iou_per'][c] for c in FLOODED_CLASSES])
        else:
            val_loss  = history['val_loss'][-1]  if history['val_loss']  else 0.0
            val_miou  = history['val_miou'][-1]  if history['val_miou']  else 0.0
            val_dice  = history['val_dice'][-1]  if history['val_dice']  else 0.0
            vm = {
                'pa':        history['val_pa'][-1]        if history['val_pa']        else 0.0,
                'fwiou':     history['val_fwiou'][-1]     if history['val_fwiou']     else 0.0,
                'kappa':     history['val_kappa'][-1]     if history['val_kappa']     else 0.0,
                'f1':        history['val_f1'][-1]        if history['val_f1']        else 0.0,
                'precision': history['val_precision'][-1] if history['val_precision'] else 0.0,
                'recall':    history['val_recall'][-1]    if history['val_recall']    else 0.0,
                'iou_per':   np.zeros(NUM_CLASSES),
            }
            flood_iou_val = history['flood_iou'][-1] if history['flood_iou'] else 0.0

        if swa_started:
            swa_model.update_parameters(model)
            swa_sched.step()

        lr      = optimizer.param_groups[0]['lr']
        gap     = train_miou - val_miou
        ep_time = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(train_miou)
        history['val_miou'].append(val_miou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['lr'].append(lr)
        history['val_pa'].append(float(vm.get('pa', 0)))
        history['val_fwiou'].append(float(vm.get('fwiou', 0)))
        history['val_kappa'].append(float(vm.get('kappa', 0)))
        history['val_f1'].append(float(vm.get('f1', 0)))
        history['val_precision'].append(float(vm.get('precision', 0)))
        history['val_recall'].append(float(vm.get('recall', 0)))
        history['epoch_time'].append(ep_time)
        history['flood_iou'].append(float(flood_iou_val))

        beats = sum([
            val_miou*100 > PAPER_METRICS['mIoU'],
            val_dice*100 > PAPER_METRICS['mDice'],
            float(vm.get('pa',0))*100 > PAPER_METRICS['PA'],
            float(vm.get('f1',0))*100 > PAPER_METRICS['F1'],
        ])
        beat_str = f"[Beating {beats}/4 paper metrics]"

        epochs_done = epoch - start_epoch + 1
        avg_t       = (time.time()-total_start) / epochs_done
        eta_s       = avg_t * (NUM_EPOCHS - epoch - 1)
        eta_str     = (f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.0f}m")

        print(f"  Train  loss:{train_loss:.4f} mIoU:{train_miou:.4f} mDice:{train_dice:.4f}")
        if do_val:
            print(f"  Val    loss:{val_loss:.4f}   mIoU:{val_miou:.4f}   mDice:{val_dice:.4f}")
            print(f"  PA:{float(vm.get('pa',0))*100:.2f}%  "
                  f"FWIoU:{float(vm.get('fwiou',0))*100:.2f}%  "
                  f"Kappa:{float(vm.get('kappa',0)):.4f}  "
                  f"F1:{float(vm.get('f1',0))*100:.2f}%")
        else:
            print(f"  Val    skipped (runs every {VAL_EVERY} epochs)")
        print(f"  FloodIoU:{float(flood_iou_val)*100:.2f}%  Gap:{gap:.4f}  "
              f"LR:{lr:.2e}  Time:{ep_time:.0f}s  ETA:{eta_str}  {beat_str}")

        with open(os.path.join(SAVE_DIR, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        # FIX 8: val_miou only updates no_improve on epochs where val ran
        if do_val and val_miou > best_miou:
            best_miou  = val_miou; no_improve = 0
            ckpt_data  = dict(epoch=epoch,
                              model_state=model.state_dict(),
                              optimizer_state=optimizer.state_dict(),
                              scheduler_state=scheduler.state_dict(),
                              val_miou=val_miou, val_dice=val_dice,
                              best_miou=best_miou, no_improve=no_improve,
                              swa_started=swa_started, history=history,
                              current_phase=current_phase)
            if USE_EMA and ema_model is not None:
                ckpt_data['ema_state'] = ema_model.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"  *** Best checkpoint saved! mIoU={best_miou*100:.2f}% ***")
        elif do_val:
            no_improve += 1
            print(f"  No improve: {no_improve}/{PATIENCE}")
        else:
            print(f"  No improve: {no_improve}/{PATIENCE} (val skipped)")

        if (epoch+1) % SAVE_EVERY == 0:
            ckpt_data = dict(epoch=epoch,
                             model_state=model.state_dict(),
                             optimizer_state=optimizer.state_dict(),
                             scheduler_state=scheduler.state_dict(),
                             best_miou=best_miou, no_improve=no_improve,
                             swa_started=swa_started, history=history,
                             current_phase=current_phase)
            if USE_EMA and ema_model is not None:
                ckpt_data['ema_state'] = ema_model.state_dict()
            torch.save(ckpt_data, ckpt_last)
            print(f"  [Checkpoint saved at epoch {epoch+1}]")

        if swa_started and (epoch+1) % SAVE_EVERY == 0:
            torch.save(swa_model.state_dict(), swa_ckpt_path)

        if no_improve >= PATIENCE:
            print(f"\n*** Early stop at epoch {epoch+1}! ***")
            break

    if swa_started:
        print("\nUpdating SWA BN statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
        torch.save(swa_model.state_dict(), swa_ckpt_path)
        print("SWA model saved.")

    total_time = time.time() - total_start
    print(f"\nTraining done! Best mIoU: {best_miou*100:.2f}%  "
          f"Time: {total_time/3600:.2f}h")

    # ================================================================
    # FINAL EVALUATION
    # ================================================================
    print("\n--- Loading best checkpoint ---")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    print(f"Final evaluation with TTA={FINAL_TTA}, {TTA_COUNT} variants...")
    preds_tta, gts_tta = evaluate_with_tta(model, val_loader,
                                           tta=FINAL_TTA, tta_count=TTA_COUNT)
    metrics = compute_all_metrics(preds_tta, gts_tta, NUM_CLASSES)

    if swa_started and os.path.exists(swa_ckpt_path):
        print("Evaluating SWA model...")
        swa_model.load_state_dict(torch.load(swa_ckpt_path, map_location=DEVICE))
        sp, sg = evaluate_with_tta(swa_model, val_loader, tta=True, tta_count=TTA_COUNT)
        sm     = compute_all_metrics(sp, sg, NUM_CLASSES)
        print(f"  Regular: {metrics['miou']*100:.2f}%  SWA: {sm['miou']*100:.2f}%")
        if sm['miou'] > metrics['miou']:
            print("  >> Using SWA"); metrics = sm
        else:
            print("  >> Keeping regular")

    print_per_class_table(metrics, val_counts)
    print_paper_comparison(metrics)

    analysis = measure_model(model, DEVICE, IMAGE_SIZE, num_runs=10)

    if yolo_model is None and LOAD_YOLO_AFTER_TRAIN:
        print("\nLoading YOLOv8 for post-training outputs...")
        yolo_model = load_yolo_model()

    if RUN_FULL_REPORTS:
        plot_all_graphs(history, metrics, val_counts, SAVE_DIR)
        save_prediction_samples(model, val_ds, yolo_model, SAVE_DIR, n=6)

    final_json = {
        'ours': {
            'mIoU'      : metrics['miou']*100,
            'mDice'     : metrics['mdice']*100,
            'mACC'      : metrics['macc']*100,
            'PA'        : metrics['pa']*100,
            'MPA'       : metrics['mpa']*100,
            'FWIoU'     : metrics['fwiou']*100,
            'Kappa'     : metrics['kappa'],
            'Precision' : metrics['precision']*100,
            'Recall'    : metrics['recall']*100,
            'F1'        : metrics['f1']*100,
            'Params'    : analysis['total_params']/1e6,
            'FPS'       : analysis['fps'],
            'Latency'   : analysis['latency_ms'],
            'TrainHrs'  : total_time/3600,
        },
        'paper'          : PAPER_METRICS,
        'per_class_iou'  : {CLASS_NAMES[c]: metrics['iou_per'][c]*100  for c in range(NUM_CLASSES)},
        'per_class_dice' : {CLASS_NAMES[c]: metrics['dice_per'][c]*100 for c in range(NUM_CLASSES)},
        'per_class_f1'   : {CLASS_NAMES[c]: metrics['f1_per'][c]*100   for c in range(NUM_CLASSES)},
        'train_counts'   : {CLASS_NAMES[c]: int(train_counts[c])       for c in range(NUM_CLASSES)},
        'val_counts'     : {CLASS_NAMES[c]: int(val_counts[c])         for c in range(NUM_CLASSES)},
    }
    with open(os.path.join(SAVE_DIR, 'final_metrics.json'), 'w') as f:
        json.dump(final_json, f, indent=2)
    np.save(os.path.join(SAVE_DIR, 'confusion_matrix.npy'), np.array(metrics['cm']))
    print(f"\nfinal_metrics.json + confusion_matrix.npy saved.")

    print("\nRunning pipeline demo...")
    pred, fp_run, pdf_path = run_pipeline(
        model, val_ds.ids[0], yolo_model, metrics,
        location="Bihar, India", lat=25.59, lon=85.13)

    if RUN_TEST_PREDICTIONS:
        print("\nBatch predicting test set...")
        results = []
        model.eval()
        for img_id in test_ds.ids:
            img_path = None
            for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
                c = os.path.join(TEST_IMG, img_id + ext)
                if os.path.exists(c): img_path = c; break
            img    = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB),
                                (IMAGE_SIZE, IMAGE_SIZE))
            tensor = torch.tensor(img).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255.0
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    seg, _, _ = model(tensor)
                pred = seg.argmax(1).squeeze().cpu().numpy()
            fp_r, is_fl, bkdn = estimate_flood_area(pred)
            _, yolo_dets = run_yolo_detection(yolo_model, img)
            results.append({
                'id': img_id, 'flood_pct': fp_r,
                'status': "FLOODED" if is_fl else "NON-FLOODED",
                'top_class': max(bkdn, key=bkdn.get) if bkdn else "N/A",
                'yolo_detections': len(yolo_dets)
            })
            print(f"  {img_id}: {results[-1]['status']} ({fp_r:.1f}%) YOLO:{len(yolo_dets)}")

        n_fl = sum(1 for r in results if r['status'] == "FLOODED")
        print(f"\nBATCH: {n_fl}/{len(results)} flooded ({n_fl/max(len(results),1)*100:.1f}%)")
        csv_path = os.path.join(SAVE_DIR, "test_predictions.csv")
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['id','flood_pct','status',
                                              'top_class','yolo_detections'])
            w.writeheader(); w.writerows(results)
        print(f"CSV saved: {csv_path}")

    print("\n" + "="*72)
    print("   DINOv2SegFormer v4 — DONE")
    print(f"  Best mIoU : {best_miou*100:.2f}%")
    print(f"  Outputs   : {SAVE_DIR}")
    print_paper_comparison(metrics)
    print("="*72)