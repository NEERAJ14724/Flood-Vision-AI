import os, sys, cv2, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import Dinov2Model

# ================================================================
# CONFIG
# ================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = r"C:\Users\ratho\OneDrive\Desktop\nielit project\flood\flood_output_v5"
CKPT_PATH = os.path.join(SAVE_DIR, "checkpoint_best.pth")
IMAGE_SIZE = 512
NUM_CLASSES = 10

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
# MODEL ARCHITECTURE (same as training)
# ================================================================
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(6,12,18,24)):
        super().__init__()
        self.c1=nn.Sequential(nn.Conv2d(in_ch,out_ch,1,bias=False),
                              nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
        self.dilated=nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=r,dilation=r,bias=False),
                          nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
            for r in rates])
        self.pool=nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_ch,out_ch,1,bias=False),
                                nn.GroupNorm(32,out_ch),nn.ReLU(inplace=True))
        n=1+len(rates)+1
        self.proj=nn.Sequential(nn.Conv2d(out_ch*n,out_ch,1,bias=False),
                                nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
                                nn.Dropout2d(0.15))

    def forward(self,x):
        h,w=x.shape[2:]
        feats=[self.c1(x)]+[d(x) for d in self.dilated]
        feats.append(torch.nn.functional.interpolate(self.pool(x),(h,w),mode='bilinear',align_corners=False))
        return self.proj(torch.cat(feats,dim=1))

class BoundaryRefinementHead(nn.Module):
    def __init__(self, in_ch, nc):
        super().__init__()
        self.bconv=nn.Sequential(
            nn.Conv2d(in_ch,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.Conv2d(64,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,nc,1))
        self.gate=nn.Sequential(nn.Conv2d(nc*2,nc,1),nn.Sigmoid())

    def forward(self,feat,coarse):
        b=self.bconv(feat)
        g=self.gate(torch.cat([coarse,b],dim=1))
        return coarse*g+b*(1-g)

class ASPPDecoder(nn.Module):
    def __init__(self, in_ch=1536, nc=10):
        super().__init__()
        self.fuse=nn.Sequential(nn.Conv2d(in_ch,512,1,bias=False),
                                nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.aspp=ASPP(512,256,(6,12,18,24))
        self.ref=nn.Sequential(
            nn.Conv2d(256,128,3,padding=1,bias=False),nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),nn.Dropout2d(0.15),
            nn.Conv2d(128,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.seg_head=nn.Conv2d(64,nc,1)
        self.bhead=BoundaryRefinementHead(64,nc)
        self.aux_head=nn.Conv2d(64,1,1)

    def forward(self,x):
        x=self.fuse(x); x=self.aspp(x); x=self.ref(x)
        coarse=self.seg_head(x)
        seg=self.bhead(x,coarse)
        aux=self.aux_head(x)
        return seg,aux,coarse

class DINOv2SegFormerV5(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.encoder=Dinov2Model.from_pretrained("facebook/dinov2-small")
        hidden=self.encoder.config.hidden_size
        self.scale_proj=nn.ModuleList([
            nn.Sequential(nn.Linear(hidden,hidden),nn.LayerNorm(hidden),
                          nn.GELU(),nn.Dropout(0.1))
            for _ in range(4)])
        self.channel_attn=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),
            nn.Linear(hidden*4,hidden*4//16),nn.ReLU(),
            nn.Linear(hidden*4//16,hidden*4),nn.Sigmoid())
        self.decode_head=ASPPDecoder(hidden*4,nc)

    def forward(self,x):
        B,C,H,W=x.shape
        out=self.encoder(pixel_values=x,output_hidden_states=True)
        hs=out.hidden_states
        scales=[hs[3],hs[6],hs[9],hs[12]]
        th,tw=H//4,W//4
        proj=[]
        for i,feat in enumerate(scales):
            feat=feat[:,1:]
            feat=self.scale_proj[i](feat)
            Bn,N,D=feat.shape
            h=w=int(N**0.5)
            feat=feat.permute(0,2,1).reshape(Bn,D,h,w)
            feat=torch.nn.functional.interpolate(feat,(th,tw),mode='bilinear',align_corners=False)
            proj.append(feat)
        x=torch.cat(proj,dim=1)
        attn=self.channel_attn(x).view(B,-1,1,1)
        x=x*attn
        seg,aux,coarse=self.decode_head(x)
        seg=torch.nn.functional.interpolate(seg,(H,W),mode='bilinear',align_corners=False)
        aux=torch.nn.functional.interpolate(aux,(H,W),mode='bilinear',align_corners=False)
        coarse=torch.nn.functional.interpolate(coarse,(H,W),mode='bilinear',align_corners=False)
        return seg,aux,coarse

import torch.nn as nn
import torch.nn.functional as F

# ================================================================
# LOAD MODEL
# ================================================================
print("="*60)
print("Loading Trained Model...")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CKPT_PATH}")

model = DINOv2SegFormerV5(nc=NUM_CLASSES).to(DEVICE)
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"✓ Model loaded successfully!")
print(f"  Best mIoU: {checkpoint.get('best_miou', 0)*100:.2f}%")
print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
print("="*60)

# ================================================================
# RUN INFERENCE ON TEST IMAGE
# ================================================================
# Find a test image
DATASET_ROOT = r"C:\Users\ratho\Downloads\FloodNet"
if os.path.isdir(os.path.join(DATASET_ROOT, "FloodNet-Supervised_v1.0")):
    BASE_PATH = os.path.join(DATASET_ROOT, "FloodNet-Supervised_v1.0")
else:
    BASE_PATH = DATASET_ROOT

TEST_IMG_DIR = os.path.join(BASE_PATH, "test/test-org-img")

if os.path.exists(TEST_IMG_DIR):
    test_images = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if test_images:
        test_img_path = os.path.join(TEST_IMG_DIR, test_images[0])
        print(f"\nRunning inference on: {test_images[0]}")
        
        # Load and preprocess image
        img_bgr = cv2.imread(test_img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = torch.tensor(img_resized).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0
        
        # Run inference
        with torch.no_grad():
            seg, aux, coarse = model(img_tensor)
        
        # Get prediction
        pred = seg.argmax(1).squeeze(0).cpu().numpy()
        
        # Calculate flood percentage
        total = pred.size
        flooded = np.isin(pred, FLOODED_CLASSES).sum()
        flood_pct = flooded / total * 100
        status = "FLOODED" if flood_pct >= 5.0 else "NON-FLOODED"
        
        # Class breakdown
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Status: {status}")
        print(f"Flood Percentage: {flood_pct:.2f}%")
        print(f"\nClass Breakdown:")
        for c in range(NUM_CLASSES):
            count = (pred == c).sum()
            pct = count / total * 100
            marker = "🌊" if c in FLOODED_CLASSES else ""
            print(f"  {CLASS_NAMES[c]:<22} {pct:>6.2f}%  {marker}")
        
        print(f"\nDominant Class: {CLASS_NAMES[pred.flatten().mode()[0][0]]}")
        print("="*60)
        
        # Save visualization
        colored_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for c, col in enumerate(CLASS_COLORS):
            colored_mask[pred == c] = col
        
        overlay = cv2.addWeighted(img_resized, 0.6, colored_mask, 0.4, 0)
        output_path = os.path.join(SAVE_DIR, "inference_result.png")
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"\n✓ Visualization saved to: {output_path}")
        
    else:
        print("No test images found in dataset directory.")
else:
    print(f"Test image directory not found: {TEST_IMG_DIR}")
    print("\nTo run inference, provide a test image path:")
    print("  python inference_simple.py <image_path>")
