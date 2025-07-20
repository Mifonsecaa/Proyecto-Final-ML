import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
import random

# --- Rutas ---
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent / 'Proyecto Final ML' / 'data' / 'annotations'
DATA_ROOT = BASE_DIR.parent / 'Proyecto Final ML' / 'data' / 'VSAIv1' / 'split_ss_444_lsv'
TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test", "images")
TEST_ANN_FILE = os.path.join(ROOT, "annotations_test.json")
MODELS_DIR = os.path.join(DATA_ROOT, "models")

# --- Constantes ---
NUM_CLASSES = 3  # fondo, small-vehicle, large-vehicle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transformaciones ---
def get_transform(train):
    transforms = [T.ToTensor()]
    return T.Compose(transforms)

# --- Cargar modelos ---
def load_faster_rcnn():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    weights_path = os.path.join(MODELS_DIR, "rcnn_quad_final.pth")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

def load_retinanet():
    model = retinanet_resnet50_fpn(weights="DEFAULT")
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head = RetinaNetHead(in_channels, num_anchors, NUM_CLASSES)
    weights_path = os.path.join(MODELS_DIR, "retinanet_quad_final.pth")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

# --- Visualización de predicciones ---
def visualize_predictions(model, image_path, threshold=0.5, save_path=None):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(train=False)
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    boxes = outputs['boxes'].cpu()
    scores = outputs['scores'].cpu()
    labels = outputs['labels'].cpu()

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min

        color = 'red' if label == 1 else 'blue'
        label_name = "small-vehicle" if label == 1 else "large-vehicle"

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        #ax.text(x_min, y_min - 5, f"{label_name}: {score:.2f}", color=color, fontsize=10, weight='bold')

    plt.axis('off')
    # Guardar o mostrar
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

# --- Ejecutar predicción sobre una imagen de test ---
if __name__ == "__main__":
    # Ruta a la carpeta de imágenes
    images_dir = Path("data/VSAIv1/split_ss_444_lsv/test/images")
    output_dir = Path("data/VSAIv1/split_ss_444_lsv/results")

    # Obtener solo nombres de archivos (no rutas completas)
    nombres_archivos = [f.name for f in images_dir.iterdir() if f.is_file()]
    for i in range(0,30):
        image_filename = random.choice(nombres_archivos)  # Cambia esto por una imagen real de test
        image_path = os.path.join(TEST_IMAGES_DIR, image_filename)
        output_path = output_dir / f"pred_{i + 1}_{image_filename}"

        # Cargar modelo
        model = load_faster_rcnn()

        #model = load_retinanet()
        # Visualizar y guardar predicciones
        visualize_predictions(model, image_path, threshold=0.5, save_path=output_path)

    print(f"✅ Imágenes guardadas en: {output_dir.resolve()}")

