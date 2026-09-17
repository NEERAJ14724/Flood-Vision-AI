# ================================================================
# FLOOD DETECTION API — FastAPI Application
# Serves DINOv2SegFormer v4 + YOLOv8 + EMA/SWA models
#
# Endpoints:
#   POST /predict          — upload image → full flood analysis
#   POST /predict/segment  — segmentation mask only
#   POST /predict/yolo     — YOLO detection only
#   POST /predict/tta      — TTA ensemble prediction
#   GET  /metrics          — trained model metrics vs paper
#   GET  /health           — health check
#   GET  /classes          — class info
#   GET  /map/{lat}/{lon}  — flood risk map HTML
#
# Run: uvicorn flood_api:app --host 0.0.0.0 --port 8000 --reload
# ================================================================

import os, sys, io, cv2, json, time, base64, tempfile, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FloodAPI")

# ── FastAPI imports ─────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Model imports (same architecture as training script) ─────────
from transformers import Dinov2Model
import folium
from fpdf import FPDF

# ================================================================
# CONFIG — point to your trained model outputs
# ================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.environ.get(
    "FLOOD_SAVE_DIR",
    os.path.join(PROJECT_ROOT, "flood_output_v5")
)
CKPT_BEST       = os.path.join(SAVE_DIR, "checkpoint_best.pth")
CKPT_SWA        = os.path.join(SAVE_DIR, "checkpoint_swa.pth")
FINAL_METRICS   = os.path.join(SAVE_DIR, "training_history.json")
DINO_MODEL_NAME = "facebook/dinov2-small"
IMAGE_SIZE      = 512
NUM_CLASSES     = 10
FLOOD_THRESHOLD = 5.0
TTA_COUNT       = 6          # reduced for API speed (was 10 in training)
USE_TTA_DEFAULT = False      # TTA off by default for /predict; on for /predict/tta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

PAPER_METRICS = {
    'mIoU': 75.2, 'mDice': 85.4, 'mACC': 87.1,
    'PA': 89.3,   'MPA': 82.6,   'FWIoU': 81.4,
    'Kappa': 0.847, 'Precision': 84.1,
    'Recall': 83.7, 'F1': 83.9,
}

# ================================================================
# MODEL ARCHITECTURE (identical to training script)
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
        feats.append(F.interpolate(self.pool(x),(h,w),mode='bilinear',align_corners=False))
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
        self.encoder=Dinov2Model.from_pretrained(DINO_MODEL_NAME)
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
            feat=F.interpolate(feat,(th,tw),mode='bilinear',align_corners=False)
            proj.append(feat)
        x=torch.cat(proj,dim=1)
        attn=self.channel_attn(x).view(B,-1,1,1)
        x=x*attn
        seg,aux,coarse=self.decode_head(x)
        seg=F.interpolate(seg,(H,W),mode='bilinear',align_corners=False)
        aux=F.interpolate(aux,(H,W),mode='bilinear',align_corners=False)
        coarse=F.interpolate(coarse,(H,W),mode='bilinear',align_corners=False)
        return seg,aux,coarse

# ================================================================
# MODEL REGISTRY — loads all models once at startup
# ================================================================
class ModelRegistry:
    """Holds all loaded models. Loaded once at API startup."""

    def __init__(self):
        self.seg_model   = None   # best checkpoint
        self.swa_model   = None   # SWA averaged model
        self.yolo_model  = None   # YOLOv8
        self.final_metrics = {}
        self._ready      = False

    def load_all(self):
        logger.info("="*60)
        logger.info("Loading all models...")
        logger.info(f"Device: {DEVICE}")

        def _load_checkpoint(path: str):
            try:
                return torch.load(path, map_location=DEVICE, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=DEVICE)

        # ── Segmentation model (best checkpoint) ──────────────────
        if not os.path.exists(CKPT_BEST):
            logger.warning(f"Best checkpoint not found: {CKPT_BEST}")
            logger.warning("API will run in demo mode (random weights).")
        else:
            logger.info(f"Loading best checkpoint: {CKPT_BEST}")

        self.seg_model = DINOv2SegFormerV5(nc=NUM_CLASSES).to(DEVICE)
        if os.path.exists(CKPT_BEST):
            ckpt = _load_checkpoint(CKPT_BEST)
            self.seg_model.load_state_dict(ckpt['model_state'])
            logger.info(f"  Seg model loaded. Best mIoU: {ckpt.get('best_miou',0)*100:.2f}%")
        self.seg_model.eval()

        # ── SWA model ────────────────────────────────────────────
        if os.path.exists(CKPT_SWA):
            try:
                from torch.optim.swa_utils import AveragedModel
                self.swa_model = AveragedModel(
                    DINOv2SegFormerV5(nc=NUM_CLASSES).to(DEVICE))
                self.swa_model.load_state_dict(
                    _load_checkpoint(CKPT_SWA))
                self.swa_model.eval()
                logger.info("  SWA model loaded.")
            except Exception as e:
                logger.warning(f"  SWA model load failed: {e}")
                self.swa_model = None
        else:
            logger.info("  SWA checkpoint not found — using best only.")

        # ── YOLO ─────────────────────────────────────────────────
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8m.pt')
            logger.info("  YOLOv8m loaded.")
        except Exception as e:
            logger.warning(f"  YOLO unavailable: {e}")
            self.yolo_model = None

        # ── Final metrics JSON ────────────────────────────────────
        if os.path.exists(FINAL_METRICS):
            with open(FINAL_METRICS) as f:
                self.final_metrics = json.load(f)
            logger.info("  Final metrics loaded.")
        else:
            self.final_metrics = {'ours': {}, 'paper': PAPER_METRICS}

        self._ready = True
        logger.info("All models ready!")
        logger.info("="*60)

    @property
    def ready(self):
        return self._ready


REGISTRY = ModelRegistry()

# ================================================================
# FASTAPI APP
# ================================================================
app = FastAPI(
    title="Flood Detection API",
    description=(
        "DINOv2SegFormer v4 + YOLOv8 Flood Detection API.\n\n"
        "Upload aerial/satellite images and receive:\n"
        "- Semantic segmentation (10 classes)\n"
        "- Flood area percentage\n"
        "- Flood/Non-flood classification\n"
        "- Object detection (YOLO)\n"
        "- Coloured mask image (base64)\n"
        "- Per-class breakdown\n"
        "- Disaster intelligence report data"
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# STARTUP / SHUTDOWN
# ================================================================
@app.on_event("startup")
async def startup_event():
    REGISTRY.load_all()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API shutting down.")

# ================================================================
# RESPONSE SCHEMAS
# ================================================================
class ClassBreakdown(BaseModel):
    class_name: str
    class_id: int
    percentage: float
    is_flooded: bool

class YoloDetection(BaseModel):
    label: str
    confidence: float
    box: List[int]   # [x1, y1, x2, y2]

class FloodPredictionResponse(BaseModel):
    # Core result
    status: str               # "FLOODED" or "NON-FLOODED"
    flood_percentage: float
    flood_threshold: float
    confidence: float

    # Breakdown
    class_breakdown: List[ClassBreakdown]
    dominant_class: str

    # Images (base64 PNG)
    segmentation_mask_b64: str    # coloured segmentation overlay
    yolo_overlay_b64: Optional[str] = None

    # Detections
    yolo_detections: List[YoloDetection]

    # Model info
    model_used: str
    inference_time_ms: float
    device: str

    # Recommendations
    recommendations: List[str]

    # Timestamp
    timestamp: str

# ================================================================
# UTILITY FUNCTIONS
# ================================================================
def preprocess_image(image_bytes: bytes) -> tuple:
    """Decode bytes → numpy RGB + torch tensor."""
    nparr  = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Ensure it is a valid JPG/PNG.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rsz = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    tensor  = torch.tensor(img_rsz).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0
    return img_rsz, tensor


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS):
        out[mask == c] = col
    return out


def overlay_mask_on_image(img_rgb: np.ndarray, mask: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
    colored = mask_to_color(mask)
    return cv2.addWeighted(img_rgb, 1-alpha, colored, alpha, 0)


def numpy_to_b64(arr: np.ndarray) -> str:
    """Convert numpy HWC uint8 image to base64 PNG string."""
    pil_img = Image.fromarray(arr.astype(np.uint8))
    buf     = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def estimate_flood_area(pred_mask: np.ndarray):
    total   = pred_mask.size
    flooded = np.isin(pred_mask, FLOODED_CLASSES).sum()
    pct     = flooded / total * 100
    bkdn    = []
    for c in range(NUM_CLASSES):
        cnt = (pred_mask == c).sum()
        p   = cnt / total * 100
        bkdn.append(ClassBreakdown(
            class_name=CLASS_NAMES[c],
            class_id=c,
            percentage=round(p, 3),
            is_flooded=(c in FLOODED_CLASSES)))
    bkdn.sort(key=lambda x: x.percentage, reverse=True)
    dominant = max(CLASS_NAMES.values(),
                   key=lambda n: next(b.percentage for b in bkdn if b.class_name==n))
    return round(pct, 3), pct >= FLOOD_THRESHOLD, bkdn, dominant


def run_segmentation(model, tensor: torch.Tensor, use_tta: bool = False):
    """Run segmentation with optional TTA."""
    model.eval()
    with torch.no_grad():
        if use_tta:
            results = []
            with torch.cuda.amp.autocast():
                results.append(model(tensor)[0])
                results.append(torch.flip(model(torch.flip(tensor,[3]))[0],[3]))
                results.append(torch.flip(model(torch.flip(tensor,[2]))[0],[2]))
                img_r = torch.rot90(tensor, 1, [2,3])
                results.append(torch.rot90(model(img_r)[0], -1, [2,3]))
                if TTA_COUNT >= 5:
                    sm = F.interpolate(tensor, scale_factor=0.75,
                                       mode='bilinear', align_corners=False)
                    results.append(F.interpolate(model(sm)[0],
                                                 size=tensor.shape[2:],
                                                 mode='bilinear', align_corners=False))
                if TTA_COUNT >= 6:
                    lg = F.interpolate(tensor, scale_factor=1.25,
                                       mode='bilinear', align_corners=False)
                    results.append(F.interpolate(model(lg)[0],
                                                 size=tensor.shape[2:],
                                                 mode='bilinear', align_corners=False))
            seg = torch.stack(results).mean(0)
        else:
            with torch.cuda.amp.autocast():
                seg, _, _ = model(tensor)

    probs = F.softmax(seg, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)
    pred  = seg.argmax(1).squeeze(0).cpu().numpy()           # (H, W)
    # confidence = mean probability of predicted class at each pixel
    confidence = float(probs.max(0).mean())
    return pred, confidence


def run_yolo(img_rgb: np.ndarray):
    if REGISTRY.yolo_model is None:
        return img_rgb, []
    try:
        results    = REGISTRY.yolo_model(img_rgb, verbose=False, conf=0.3)
        annotated  = img_rgb.copy()
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf  = float(box.conf[0])
                cls_  = int(box.cls[0])
                label = REGISTRY.yolo_model.names[cls_]
                color = (0, 255, 0) if 'car' not in label.lower() else (255, 128, 0)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                cv2.putText(annotated, f"{label} {conf:.2f}",
                            (x1, max(y1-6,10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1)
                detections.append(YoloDetection(
                    label=label, confidence=round(conf,3),
                    box=[x1, y1, x2, y2]))
        return annotated, detections
    except Exception as e:
        logger.warning(f"YOLO error: {e}")
        return img_rgb, []


def get_recommendations(status: str, flood_pct: float,
                         breakdown: List[ClassBreakdown]) -> List[str]:
    recs = []
    
    # Create a dictionary of class percentages for easy lookup
    class_pct = {b.class_name: b.percentage for b in breakdown}
    
    if status == "FLOODED":
        if flood_pct >= 40:
            # SEVERE FLOODING (≥40%) - High-priority emergency response
            recs.append(f"🚨 SEVERE FLOOD ALERT: {flood_pct:.1f}% area submerged - Immediate evacuation required.")
            recs.append("Deploy NDRF/SDRF rescue teams, amphibious boats, and emergency medical units to affected zones.")
            recs.append("Immediate evacuation of residents from submerged and high-risk areas.")
            recs.append("Closure of flooded roads, bridges, and transport routes.")
            recs.append("Establish temporary relief camps with food, clean water, electricity, and medical aid.")
            recs.append("Continuous monitoring of river water levels, dams, and drainage systems every 15-30 minutes.")
            recs.append("Public warning: Electrocution risk from submerged power lines - avoid contact with floodwater.")
            recs.append("Strong currents and debris present - do not attempt to walk or drive through floodwaters.")
            recs.append("Priority rescue operations for hospitals, schools, elderly citizens, and children.")
            recs.append("Coordinate with police, fire departments, and local authorities for unified response.")
            recs.append("Distribution of emergency supplies and communication alerts to affected communities.")
            
            # Add specific class-based recommendations
            if "Building-Flooded" in class_pct and class_pct["Building-Flooded"] > 0:
                recs.append(f"CRITICAL: {class_pct['Building-Flooded']:.1f}% buildings submerged - structural collapse risk.")
            if "Road-Flooded" in class_pct and class_pct["Road-Flooded"] > 0:
                recs.append(f"URGENT: {class_pct['Road-Flooded']:.1f}% roads impassable - water transport required.")
            if "Bridge" in class_pct and class_pct["Bridge"] > 0:
                recs.append(f"INFRASTRUCTURE: {class_pct['Bridge']:.1f}% bridges affected - immediate inspection required.")
                
        elif 20 <= flood_pct < 40:
            # MODERATE FLOODING (20-39%) - Warning and preparedness
            recs.append(f"⚠️ MODERATE FLOOD WARNING: {flood_pct:.1f}% area affected - Stay alert.")
            recs.append("Keep emergency response teams and evacuation vehicles on standby.")
            recs.append("Restrict movement through waterlogged roads and underpasses.")
            recs.append("Monitor rainfall intensity and nearby river or canal water levels continuously.")
            recs.append("Inform residents in low-lying areas about possible evacuation if conditions worsen.")
            recs.append("Prepare temporary shelters and stock emergency supplies.")
            recs.append("Inspect drainage systems and clear blockages to reduce water accumulation.")
            recs.append("Advise schools, markets, and public transport systems to remain cautious.")
            recs.append("Encourage residents to store drinking water, medicines, and essential items.")
            
            # Add specific class-based recommendations
            if "Road-Flooded" in class_pct and class_pct["Road-Flooded"] > 5:
                recs.append(f"CAUTION: {class_pct['Road-Flooded']:.1f}% roads waterlogged - avoid non-essential travel.")
            if "Building-Flooded" in class_pct and class_pct["Building-Flooded"] > 0:
                recs.append(f"ALERT: {class_pct['Building-Flooded']:.1f}% buildings at risk - prepare for possible evacuation.")
            if "Water" in class_pct and class_pct["Water"] > 15:
                recs.append(f"MONITOR: Water levels at {class_pct['Water']:.1f}% - watch for rising trend.")
                
        else:
            # MINOR FLOODING (<20%) - Preventive and monitoring
            recs.append(f"📢 FLOOD ADVISORY: {flood_pct:.1f}% area affected - Monitor situation.")
            recs.append("Continue monitoring rainfall and local water bodies.")
            recs.append("Clear drainage channels and remove debris to prevent waterlogging.")
            recs.append("Issue advisory alerts for low-lying and flood-prone areas.")
            recs.append("Encourage residents to avoid unnecessary travel during heavy rain.")
            recs.append("Keep local emergency contacts and medical support ready as a precaution.")
            recs.append("Monitor roads, bridges, and small settlements for early signs of flooding.")
            recs.append("Promote public awareness and safety precautions without creating panic.")
            
            # Add specific class-based recommendations
            if "Water" in class_pct and class_pct["Water"] > 5:
                recs.append(f"MONITOR: {class_pct['Water']:.1f}% water bodies present - track level changes.")
            if "Road-Flooded" in class_pct and class_pct["Road-Flooded"] > 0:
                recs.append(f"CAUTION: {class_pct['Road-Flooded']:.1f}% minor road flooding - drive carefully.")
            if "Grass" in class_pct and class_pct["Grass"] > 20:
                recs.append("SAFE: High ground areas identified - potential evacuation sites if needed.")
        
    else:
        # NON-FLOODED - General monitoring
        recs.append("✅ Area is currently non-flooded - Normal conditions.")
        recs.append("Continue regular monitoring of rainfall and water bodies.")
        recs.append("Maintain drainage systems and clear any blockages.")
        recs.append("Keep emergency response teams on standby for rapid deployment if needed.")
        recs.append("Ensure communication channels remain active for alerts.")
        
        # Add specific insights from breakdown
        if "Water" in class_pct and class_pct["Water"] > 5:
            recs.append(f"MONITOR: {class_pct['Water']:.1f}% water bodies present - watch for sudden level changes.")
        if "Grass" in class_pct and class_pct["Grass"] > 20:
            recs.append("SAFE: High ground areas available for potential evacuation sites.")
    
    return recs


def generate_pdf_report(prediction_data: dict, original_image_b64: str, mask_image_b64: str):
    """Generate a PDF report from prediction data."""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 15, "Flood Detection Report", ln=True, align='C')
    pdf.ln(10)
    
    # Author info
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "NEERAJ RATHORE", ln=True, align='C')
    pdf.cell(0, 10, "M.Sc. Data Science & Applied Statistics", ln=True, align='C')
    pdf.ln(10)
    
    # Status
    pdf.set_font("Arial", 'B', 16)
    status_color = (255, 0, 0) if prediction_data['status'] == 'FLOODED' else (0, 128, 0)
    pdf.set_text_color(*status_color)
    pdf.cell(0, 12, f"Status: {prediction_data['status']}", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Flood percentage
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Flood Area: {prediction_data['flood_percentage']:.2f}%", ln=True, align='C')
    pdf.ln(10)
    
    # Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Model Metrics:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Confidence: {prediction_data['confidence']*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Inference Time: {prediction_data['inference_time_ms']:.2f}ms", ln=True)
    pdf.cell(0, 8, f"Dominant Class: {prediction_data['dominant_class']}", ln=True)
    pdf.cell(0, 8, f"YOLO Detections: {len(prediction_data['yolo_detections'])}", ln=True)
    pdf.ln(10)
    
    # Class breakdown
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Class Breakdown:", ln=True)
    pdf.set_font("Arial", '', 10)
    for item in prediction_data['class_breakdown']:
        marker = " [FLOODED]" if item['is_flooded'] else ""
        pdf.cell(0, 7, f"  {item['class_name']}: {item['percentage']:.2f}%{marker}", ln=True)
    pdf.ln(10)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Recommendations:", ln=True)
    pdf.set_font("Arial", '', 10)
    for rec in prediction_data['recommendations']:
        pdf.multi_cell(0, 7, f"  • {rec}")
    pdf.ln(10)
    
    # Images
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Images:", ln=True)
    pdf.ln(5)
    
    # Original image
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, "Original Image:", ln=True)
    pdf.image(io.BytesIO(base64.b64decode(original_image_b64)), x=20, w=80)
    pdf.ln(5)
    
    # Segmentation mask
    pdf.cell(0, 8, "Segmentation Mask:", ln=True)
    pdf.image(io.BytesIO(base64.b64decode(mask_image_b64)), x=110, w=80)
    
    # Timestamp
    pdf.ln(10)
    pdf.set_font("Arial", '', 9)
    pdf.cell(0, 8, f"Generated: {prediction_data['timestamp']}", ln=True, align='C')
    pdf.cell(0, 8, f"Model: {prediction_data['model_used']}", ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')


# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flood Detection API v4</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;
                   padding: 20px; background: #0a1628; color: #e0e6f0; }
            h1 { color: #4fc3f7; border-bottom: 2px solid #1976d2; padding-bottom: 10px; }
            h2 { color: #81c784; }
            .endpoint { background: #132338; border-left: 4px solid #4fc3f7;
                        padding: 12px 16px; margin: 10px 0; border-radius: 4px; }
            .method { font-weight: bold; padding: 2px 8px; border-radius: 3px;
                      margin-right: 8px; }
            .post { background: #1b5e20; color: #a5d6a7; }
            .get  { background: #0d47a1; color: #90caf9; }
            code { background: #1e3a5f; padding: 2px 6px; border-radius: 3px; }
            a { color: #4fc3f7; }
            .badge { display: inline-block; padding: 4px 12px; border-radius: 12px;
                     font-size: 12px; margin: 2px; }
            .badge-green { background: #1b5e20; color: #a5d6a7; }
            .badge-blue  { background: #0d47a1; color: #90caf9; }
        </style>
    </head>
    <body>
        <h1>🌊 Flood Detection API v4</h1>
        <p>
            <span class="badge badge-green">DINOv2SegFormer v4</span>
            <span class="badge badge-blue">YOLOv8m</span>
            <span class="badge badge-green">10-Class Segmentation</span>
            <span class="badge badge-blue">FloodNet Dataset</span>
        </p>

        <h2>Quick Start</h2>
        <p>Upload an aerial image to <code>POST /predict</code> and receive a full flood analysis.</p>

        <h2>Endpoints</h2>

        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/predict</code>
            — Full analysis: segmentation + YOLO + flood status + recommendations
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/predict/segment</code>
            — Segmentation mask only (faster)
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/predict/yolo</code>
            — YOLO object detection only
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/predict/tta</code>
            — Full TTA ensemble prediction (slower, more accurate)
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/metrics</code>
            — Trained model metrics vs paper baseline
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/classes</code>
            — Segmentation class information
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/map/{lat}/{lon}?flood_pct=25.0</code>
            — Interactive flood risk map
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/health</code>
            — API health check
        </div>

        <p>
            Interactive docs: <a href="/docs">/docs</a> |
            ReDoc: <a href="/redoc">/redoc</a>
        </p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Serve the web interface HTML file."""
    web_path = os.path.join(PROJECT_ROOT, "web_interface.html")
    if os.path.exists(web_path):
        with open(web_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Web interface not found</h1>", status_code=404)


@app.get("/health")
async def health():
    """Health check — returns model loading status."""
    return {
        "status"       : "ok" if REGISTRY.ready else "loading",
        "seg_model"    : "loaded" if REGISTRY.seg_model   is not None else "missing",
        "swa_model"    : "loaded" if REGISTRY.swa_model   is not None else "not available",
        "yolo_model"   : "loaded" if REGISTRY.yolo_model  is not None else "not available",
        "device"       : str(DEVICE),
        "timestamp"    : datetime.now().isoformat(),
    }


@app.get("/classes")
async def get_classes():
    """Return all segmentation classes with their colors and flood status."""
    return {
        "num_classes"    : NUM_CLASSES,
        "flooded_classes": FLOODED_CLASSES,
        "flood_threshold": FLOOD_THRESHOLD,
        "classes": [
            {
                "id"        : c,
                "name"      : CLASS_NAMES[c],
                "color_rgb" : CLASS_COLORS[c].tolist(),
                "color_hex" : "#{:02X}{:02X}{:02X}".format(*CLASS_COLORS[c]),
                "is_flooded": c in FLOODED_CLASSES,
            }
            for c in range(NUM_CLASSES)
        ]
    }


@app.get("/metrics")
async def get_metrics():
    """Return trained model performance metrics vs paper baseline."""
    ours = REGISTRY.final_metrics.get('ours', {})
    paper = PAPER_METRICS
    comparison = {}
    for key in ['mIoU','mDice','PA','MPA','FWIoU','Precision','Recall','F1']:
        ours_val  = ours.get(key, 0.0)
        paper_val = paper.get(key, 0.0)
        comparison[key] = {
            "ours"       : round(ours_val, 3),
            "paper"      : paper_val,
            "delta"      : round(ours_val - paper_val, 3),
            "beats_paper": ours_val > paper_val,
        }
    return {
        "model"         : "DINOv2SegFormer v4",
        "baseline"      : "SwinSegFormer (paper)",
        "comparison"    : comparison,
        "per_class_iou" : REGISTRY.final_metrics.get('per_class_iou', {}),
        "per_class_dice": REGISTRY.final_metrics.get('per_class_dice', {}),
        "per_class_f1"  : REGISTRY.final_metrics.get('per_class_f1', {}),
    }


@app.post("/predict", response_model=FloodPredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Aerial/satellite image (JPG or PNG)"),
    use_swa: bool = Query(False, description="Use SWA model if available"),
    return_overlay: bool = Query(True, description="Return segmentation overlay on original image"),
):
    """
    Full flood analysis:
    - Semantic segmentation (DINOv2SegFormer v4)
    - Object detection (YOLOv8)
    - Flood percentage and status
    - Colour-coded mask image (base64)
    - Class breakdown and recommendations
    """
    if not REGISTRY.ready:
        raise HTTPException(503, "Models still loading. Try again in a moment.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected image file, got: {file.content_type}")

    t_start = time.time()

    try:
        image_bytes = await file.read()
        img_rgb, tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(400, f"Image processing failed: {e}")

    # Choose model
    model = (REGISTRY.swa_model
             if (use_swa and REGISTRY.swa_model is not None)
             else REGISTRY.seg_model)
    model_used = "SWA" if (use_swa and REGISTRY.swa_model) else "Best Checkpoint"

    # Segmentation
    try:
        pred_mask, confidence = run_segmentation(model, tensor, use_tta=USE_TTA_DEFAULT)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")

    # Flood analysis
    flood_pct, is_flooded, breakdown, dominant = estimate_flood_area(pred_mask)
    status = "FLOODED" if is_flooded else "NON-FLOODED"

    # Mask image
    if return_overlay:
        mask_img = overlay_mask_on_image(img_rgb, pred_mask, alpha=0.5)
    else:
        mask_img = mask_to_color(pred_mask)
    mask_b64 = numpy_to_b64(mask_img)

    # YOLO
    yolo_img, yolo_dets = run_yolo(img_rgb)
    yolo_b64 = numpy_to_b64(yolo_img) if yolo_dets else None

    # Recommendations
    recs = get_recommendations(status, flood_pct, breakdown)

    inference_ms = (time.time() - t_start) * 1000

    return FloodPredictionResponse(
        status=status,
        flood_percentage=flood_pct,
        flood_threshold=FLOOD_THRESHOLD,
        confidence=round(confidence, 4),
        class_breakdown=breakdown,
        dominant_class=dominant,
        segmentation_mask_b64=mask_b64,
        yolo_overlay_b64=yolo_b64,
        yolo_detections=yolo_dets,
        model_used=model_used,
        inference_time_ms=round(inference_ms, 1),
        device=str(DEVICE),
        recommendations=recs,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict/segment")
async def predict_segment(
    file: UploadFile = File(..., description="Image to segment"),
    return_overlay: bool = Query(True, description="Overlay on original image"),
):
    """
    Fast segmentation-only endpoint.
    Returns coloured mask + per-class pixel percentages.
    No YOLO, no TTA — optimised for speed.
    """
    if not REGISTRY.ready:
        raise HTTPException(503, "Models still loading.")

    t_start = time.time()
    try:
        image_bytes      = await file.read()
        img_rgb, tensor  = preprocess_image(image_bytes)
        pred_mask, conf  = run_segmentation(REGISTRY.seg_model, tensor, use_tta=False)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")

    flood_pct, is_flooded, breakdown, dominant = estimate_flood_area(pred_mask)

    if return_overlay:
        mask_img = overlay_mask_on_image(img_rgb, pred_mask, alpha=0.5)
    else:
        mask_img = mask_to_color(pred_mask)

    return {
        "status"              : "FLOODED" if is_flooded else "NON-FLOODED",
        "flood_percentage"    : flood_pct,
        "confidence"          : round(conf, 4),
        "dominant_class"      : dominant,
        "segmentation_mask_b64": numpy_to_b64(mask_img),
        "class_breakdown"     : [b.dict() for b in breakdown],
        "inference_time_ms"   : round((time.time()-t_start)*1000, 1),
        "timestamp"           : datetime.now().isoformat(),
    }


@app.post("/predict/yolo")
async def predict_yolo(
    file: UploadFile = File(..., description="Image for object detection"),
):
    """
    YOLO-only endpoint.
    Returns detected objects with bounding boxes, labels, and confidence scores.
    """
    if not REGISTRY.ready:
        raise HTTPException(503, "Models still loading.")
    if REGISTRY.yolo_model is None:
        raise HTTPException(503, "YOLOv8 model not available.")

    t_start = time.time()
    try:
        image_bytes     = await file.read()
        img_rgb, _      = preprocess_image(image_bytes)
        yolo_img, dets  = run_yolo(img_rgb)
    except Exception as e:
        raise HTTPException(500, f"YOLO detection failed: {e}")

    return {
        "num_detections"   : len(dets),
        "detections"       : [d.dict() for d in dets],
        "annotated_b64"    : numpy_to_b64(yolo_img),
        "inference_time_ms": round((time.time()-t_start)*1000, 1),
        "timestamp"        : datetime.now().isoformat(),
    }


@app.post("/predict/tta")
async def predict_tta(
    file: UploadFile = File(..., description="Image for TTA ensemble prediction"),
):
    """
    TTA (Test-Time Augmentation) ensemble endpoint.
    Uses 6-variant augmentation ensemble for higher accuracy.
    Slower than /predict but more robust.
    """
    if not REGISTRY.ready:
        raise HTTPException(503, "Models still loading.")

    t_start = time.time()
    try:
        image_bytes     = await file.read()
        img_rgb, tensor = preprocess_image(image_bytes)
        pred_mask, conf = run_segmentation(REGISTRY.seg_model, tensor, use_tta=True)
    except Exception as e:
        raise HTTPException(500, f"TTA prediction failed: {e}")

    flood_pct, is_flooded, breakdown, dominant = estimate_flood_area(pred_mask)
    overlay  = overlay_mask_on_image(img_rgb, pred_mask, alpha=0.5)
    mask_img = mask_to_color(pred_mask)
    recs     = get_recommendations("FLOODED" if is_flooded else "NON-FLOODED",
                                   flood_pct, breakdown)

    _, yolo_dets = run_yolo(img_rgb)

    return {
        "status"              : "FLOODED" if is_flooded else "NON-FLOODED",
        "flood_percentage"    : flood_pct,
        "confidence"          : round(conf, 4),
        "tta_variants"        : TTA_COUNT,
        "dominant_class"      : dominant,
        "class_breakdown"     : [b.dict() for b in breakdown],
        "segmentation_overlay_b64": numpy_to_b64(overlay),
        "segmentation_mask_b64"  : numpy_to_b64(mask_img),
        "yolo_detections"     : [d.dict() for d in yolo_dets],
        "recommendations"     : recs,
        "inference_time_ms"   : round((time.time()-t_start)*1000, 1),
        "timestamp"           : datetime.now().isoformat(),
    }


@app.post("/predict/pdf")
async def predict_pdf(
    file: UploadFile = File(..., description="Image for PDF report generation"),
):
    """
    Generate and download a PDF report for flood analysis.
    Returns a PDF file with all prediction results, images, and recommendations.
    """
    if not REGISTRY.ready:
        raise HTTPException(503, "Models still loading.")

    t_start = time.time()
    try:
        image_bytes = await file.read()
        img_rgb, tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(400, f"Image processing failed: {e}")

    # Run prediction
    try:
        pred_mask, conf = run_segmentation(REGISTRY.seg_model, tensor, use_tta=USE_TTA_DEFAULT)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")

    # Flood analysis
    flood_pct, is_flooded, breakdown, dominant = estimate_flood_area(pred_mask)
    status = "FLOODED" if is_flooded else "NON-FLOODED"

    # Generate images
    mask_img = mask_to_color(pred_mask)
    overlay = overlay_mask_on_image(img_rgb, pred_mask, alpha=0.5)

    # YOLO detection
    yolo_img, yolo_dets = run_yolo(img_rgb)

    # Get recommendations
    recs = get_recommendations(status, flood_pct, breakdown)

    # Prepare prediction data
    prediction_data = {
        "status": status,
        "flood_percentage": flood_pct,
        "confidence": conf,
        "dominant_class": dominant,
        "class_breakdown": [b.dict() for b in breakdown],
        "yolo_detections": [d.dict() for d in yolo_dets],
        "recommendations": recs,
        "inference_time_ms": round((time.time() - t_start) * 1000, 1),
        "timestamp": datetime.now().isoformat(),
        "model_used": "Best Checkpoint",
    }

    # Generate PDF
    try:
        original_b64 = numpy_to_b64(img_rgb)
        mask_b64 = numpy_to_b64(mask_img)
        pdf_bytes = generate_pdf_report(prediction_data, original_b64, mask_b64)
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {e}")

    # Return PDF file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flood_report_{timestamp}.pdf"
    
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple images (max 20)"),
):
    """
    Batch prediction endpoint.
    Send multiple images and receive flood status for each.
    Maximum 20 images per request.
    """
    if not REGISTRY.ready:
        raise HTTPException(503, "Models still loading.")
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 images per batch request.")

    results = []
    for f in files:
        t_start = time.time()
        try:
            image_bytes     = await f.read()
            img_rgb, tensor = preprocess_image(image_bytes)
            pred_mask, conf = run_segmentation(REGISTRY.seg_model, tensor, use_tta=False)
            flood_pct, is_fl, breakdown, dominant = estimate_flood_area(pred_mask)
            _, yolo_dets    = run_yolo(img_rgb)
            results.append({
                "filename"        : f.filename,
                "status"          : "FLOODED" if is_fl else "NON-FLOODED",
                "flood_percentage": flood_pct,
                "confidence"      : round(conf, 4),
                "dominant_class"  : dominant,
                "yolo_count"      : len(yolo_dets),
                "inference_ms"    : round((time.time()-t_start)*1000, 1),
                "error"           : None,
            })
        except Exception as e:
            results.append({
                "filename": f.filename,
                "status"  : "ERROR",
                "error"   : str(e),
            })

    n_flooded = sum(1 for r in results if r.get('status') == "FLOODED")
    return {
        "total"           : len(results),
        "flooded"         : n_flooded,
        "non_flooded"     : len(results) - n_flooded,
        "flood_rate_pct"  : round(n_flooded / max(len(results),1) * 100, 1),
        "results"         : results,
        "timestamp"       : datetime.now().isoformat(),
    }


@app.get("/map/{lat}/{lon}", response_class=HTMLResponse)
async def get_flood_map(
    lat: float,
    lon: float,
    flood_pct: float = Query(0.0, description="Flood percentage to display"),
    location: str    = Query("Unknown", description="Location name"),
    zoom: int        = Query(13, description="Map zoom level"),
):
    """
    Interactive Folium map showing flood risk at the given coordinates.
    Returns an HTML page with the map embedded.
    """
    try:
        m     = folium.Map(location=[lat, lon], zoom_start=zoom)
        is_fl = flood_pct >= FLOOD_THRESHOLD
        color = "red" if is_fl else "green"
        status_text = f"{'FLOODED' if is_fl else 'NON-FLOODED'}: {flood_pct:.1f}%"

        folium.CircleMarker(
            [lat, lon], radius=60, color=color,
            fill=True, fill_opacity=0.35,
            popup=folium.Popup(
                f"<b>{location}</b><br>{status_text}", max_width=200),
        ).add_to(m)

        folium.Marker(
            [lat, lon],
            popup=status_text,
            icon=folium.Icon(color=color, icon="tint", prefix="fa"),
        ).add_to(m)

        if is_fl:
            folium.Circle(
                [lat, lon], radius=2000, color='red',
                fill=True, fill_opacity=0.1,
                popup="Estimated flood radius (2km)",
            ).add_to(m)

        html_content = m._repr_html_()
        return HTMLResponse(content=html_content)

    except Exception as e:
        raise HTTPException(500, f"Map generation failed: {e}")


# Serve the standalone frontend HTML if present
@app.get("/app")
async def serve_frontend():
    ui_path = os.path.join(PROJECT_ROOT, "web_interface.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    raise HTTPException(404, "Frontend not found")


@app.get("/mask/legend", response_class=HTMLResponse)
async def mask_legend():
    """Returns an HTML page showing the segmentation colour legend."""
    rows = ""
    for c in range(NUM_CLASSES):
        rgb   = CLASS_COLORS[c]
        hex_c = "#{:02X}{:02X}{:02X}".format(*rgb)
        flood = " (FLOODED)" if c in FLOODED_CLASSES else ""
        rows += (f"<tr>"
                 f"<td style='background:{hex_c};width:50px;height:25px;border:1px solid #333'></td>"
                 f"<td style='padding:4px 12px'>{c}</td>"
                 f"<td style='padding:4px 12px'><b>{CLASS_NAMES[c]}</b>{flood}</td>"
                 f"<td style='padding:4px 12px'>{hex_c}</td>"
                 f"</tr>")
    html = f"""
    <html><head><title>Colour Legend</title>
    <style>body{{font-family:Arial;margin:20px;background:#111;color:#eee}}
    table{{border-collapse:collapse}}td{{border:1px solid #333}}</style>
    </head><body>
    <h2>Segmentation Class Legend</h2>
    <table>{rows}</table>
    </body></html>
    """
    return HTMLResponse(content=html)


# ================================================================
# SIMPLE PYTHON CLIENT EXAMPLE (printed on import for reference)
# ================================================================
CLIENT_EXAMPLE = '''
# ── Python client example ─────────────────────────────────────
import requests, base64
from PIL import Image
import io

API_URL = "http://localhost:8000"

# 1. Health check
r = requests.get(f"{API_URL}/health")
print(r.json())

# 2. Full prediction
with open("my_aerial_image.jpg", "rb") as f:
    resp = requests.post(
        f"{API_URL}/predict",
        files={"file": ("image.jpg", f, "image/jpeg")},
        params={"use_swa": False, "return_overlay": True},
    )
result = resp.json()
print("Status     :", result["status"])
print("Flood %    :", result["flood_percentage"])
print("Confidence :", result["confidence"])
print("Model      :", result["model_used"])
print("Inference  :", result["inference_time_ms"], "ms")
print("Detections :", len(result["yolo_detections"]))
print("Recs       :", result["recommendations"][0])

# Decode and show segmentation mask
mask_bytes = base64.b64decode(result["segmentation_mask_b64"])
mask_img   = Image.open(io.BytesIO(mask_bytes))
mask_img.show()

# 3. Segmentation only (faster)
with open("my_aerial_image.jpg", "rb") as f:
    resp = requests.post(f"{API_URL}/predict/segment",
                         files={"file": ("image.jpg", f, "image/jpeg")})
print(resp.json()["class_breakdown"][:3])

# 4. YOLO only
with open("my_aerial_image.jpg", "rb") as f:
    resp = requests.post(f"{API_URL}/predict/yolo",
                         files={"file": ("image.jpg", f, "image/jpeg")})
print("Objects detected:", resp.json()["num_detections"])

# 5. TTA ensemble (most accurate)
with open("my_aerial_image.jpg", "rb") as f:
    resp = requests.post(f"{API_URL}/predict/tta",
                         files={"file": ("image.jpg", f, "image/jpeg")})
print("TTA Status:", resp.json()["status"])

# 6. Batch prediction
files = [("files", open(f"img_{i}.jpg","rb")) for i in range(5)]
resp  = requests.post(f"{API_URL}/predict/batch", files=files)
print("Batch results:", resp.json()["flood_rate_pct"], "% flooded")

# 7. Model metrics
resp = requests.get(f"{API_URL}/metrics")
print(resp.json()["comparison"]["mIoU"])

# 8. Interactive map
# Open in browser:
# http://localhost:8000/map/25.59/85.13?flood_pct=35.2&location=Patna
'''

# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*65)
    print("  FLOOD DETECTION API v4 — Starting")
    print("="*65)
    print(CLIENT_EXAMPLE)
    print("="*65)
    print(f"  Docs    : http://localhost:8000/docs")
    print(f"  ReDoc   : http://localhost:8000/redoc")
    print(f"  Health  : http://localhost:8000/health")
    print("="*65 + "\n")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )