import os
import sys

# Limit CPU BLAS threads to reduce memory pressure.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['HF_HOME'] = os.environ.get('HF_HOME', os.path.join(WORK_DIR, '.hf_cache'))
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', os.path.join(WORK_DIR, '.hf_cache', 'transformers'))
os.environ['HF_DATASETS_CACHE'] = os.environ.get('HF_DATASETS_CACHE', os.path.join(WORK_DIR, '.hf_cache', 'datasets'))

import torch
import numpy as np
from torch.utils.data import DataLoader
import main

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_DIR = os.path.join(WORK_DIR, 'flood_output_v4_complete')
CKPT_PATH = os.path.join(SAVE_DIR, 'checkpoint_best.pth')
CKPT_LAST = os.path.join(SAVE_DIR, 'checkpoint_last.pth')
if os.path.exists(CKPT_PATH):
    checkpoint_file = CKPT_PATH
elif os.path.exists(CKPT_LAST):
    checkpoint_file = CKPT_LAST
else:
    raise FileNotFoundError(f'No checkpoint found in {SAVE_DIR}')

print('Using checkpoint:', checkpoint_file)
print('Device:', DEVICE)

model = main.DINOv2SegFormerV4(num_classes=main.NUM_CLASSES).to(DEVICE)

print('Loading checkpoint...')
with torch.serialization.safe_globals([np.core.multiarray.scalar]):
    ckpt = torch.load(checkpoint_file, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

print('Checkpoint loaded.')
print('Checkpoint epoch:', ckpt.get('epoch'))
print('Checkpoint best_miou:', ckpt.get('best_miou'))

print('Preparing validation dataset...')
val_ds = main.FloodNetDataset(main.VAL_IMG, main.VAL_MASK, main.IMAGE_SIZE, augment=False, split='val')
val_loader = DataLoader(val_ds, batch_size=main.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print('Evaluating on validation set (no TTA)...')
preds, gts = main.evaluate_with_tta(model, val_loader, tta=False)
metrics = main.compute_all_metrics(preds, gts, main.NUM_CLASSES)

print('\n=== Validation Results ===')
print(f"mIoU      : {metrics['miou']*100:.2f}%")
print(f"mDice     : {metrics['mdice']*100:.2f}%")
print(f"mACC      : {metrics['macc']*100:.2f}%")
print(f"PA        : {metrics['pa']*100:.2f}%")
print(f"MPA       : {metrics['mpa']*100:.2f}%")
print(f"FWIoU     : {metrics['fwiou']*100:.2f}%")
print(f"Kappa     : {metrics['kappa']:.4f}")
print(f"Precision : {metrics['precision']*100:.2f}%")
print(f"Recall    : {metrics['recall']*100:.2f}%")
print(f"F1        : {metrics['f1']*100:.2f}%")

print('\nPer-class IoU:')
for c in range(main.NUM_CLASSES):
    print(f"  {main.CLASS_NAMES[c]:<20}: {metrics['iou_per'][c]*100:.2f}%")

print('\nSaving sample predictions for first 6 val images...')
out_dir = os.path.join(SAVE_DIR, 'inference_samples')
os.makedirs(out_dir, exist_ok=True)

for idx in range(min(6, len(val_ds))):
    img, mask = val_ds[idx]
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    with torch.no_grad():
        seg, _, _ = model(img.unsqueeze(0).to(DEVICE))
    pred = seg.argmax(1).squeeze().cpu().numpy().astype('uint8')
    color_pred = main.mask_to_color(pred)
    color_gt = main.mask_to_color(mask.cpu().numpy())
    import cv2
    combined = np.hstack([img_np, color_gt, color_pred])
    cv2.imwrite(os.path.join(out_dir, f'sample_{idx+1}.png'), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

print('Sample outputs saved to:', out_dir)
print('Done.')
