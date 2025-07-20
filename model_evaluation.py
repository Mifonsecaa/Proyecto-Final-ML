import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms as T
from pathlib import Path
import io
import contextlib
from datetime import datetime
import utils  # Tu utils.py
from engine import evaluate

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent / 'Proyecto Final ML' / 'data'/'annotations'
DATA_ROOT = BASE_DIR.parent / 'Proyecto Final ML' / 'data' / 'VSAIv1' / 'split_ss_444_lsv'
TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test", "images")
TEST_ANN_FILE = os.path.join(ROOT, "annotations_test.json")
MODELS_DIR = os.path.join(DATA_ROOT, "models")

# --- Constantes ---
NUM_CLASSES = 3  # background + small-vehicle + large-vehicle
BATCH_SIZE = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- Transforms ---
def get_transform(train):
    transforms = [T.ToTensor()]
    return T.Compose(transforms)

# --- Dataset y DataLoader ---
test_dataset = CocoDetection(TEST_IMAGES_DIR, TEST_ANN_FILE, transform=get_transform(train=False))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=utils.collate_fn)

# --- Modelos ---
def load_faster_rcnn():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    weights_path = os.path.join(MODELS_DIR, "rcnn_quad_final.pth")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    return model

def load_retinanet():
    model = retinanet_resnet50_fpn(weights="DEFAULT")
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head = RetinaNetHead(in_channels, num_anchors, NUM_CLASSES)
    weights_path = os.path.join(MODELS_DIR, "retinanet_quad_final.pth")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    return model

# --- Evaluación ---
def evaluate_model(model, name):
    print(f"Evaluando modelo: {name}")
    '''model.eval()
    evaluate(model, test_loader, device=DEVICE)'''
    # Capturar salida del evaluador
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        evaluate(model, test_loader, device=DEVICE)

    output = buffer.getvalue()

    # Filtrar líneas relevantes
    metrics_lines = [line for line in output.splitlines() if
                     "IoU metric:" in line or line.strip().startswith(("Average Precision", "Average Recall"))]

    # Guardar en archivo
    log_filename = f"metrics_{name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = BASE_DIR / log_filename
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("IoU metric: bbox\n")
        for line in metrics_lines:
            f.write(f"{line}\n")

    print(f"✅ Métricas de evaluación guardadas en: {log_path}")



# --- Ejecutar ---
if __name__ == "__main__":
    model_rcnn = load_faster_rcnn()
    evaluate_model(model_rcnn, "Faster R-CNN")

    model_retina = load_retinanet()
    evaluate_model(model_retina, "RetinaNet")



