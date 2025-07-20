
import os
import subprocess
import sys
import random
import torch
import shutil
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import cv2
import torch.optim as optim
from torchvision.transforms import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead

try:
    from engine import train_one_epoch, evaluate
    import utils
except ImportError:
    # este bloque correra si falta alguna dependencia dentro de engine o utils
    print("ADVERTENCIA: Uno o más módulos de entrenamiento (engine, utils, coco_eval, coco_utils) o sus dependencias (como 'pycocotools') no están disponibles.")
    print("El entrenamiento continuará, pero la evaluación post-época será omitida.")
    # Define funciones ficticias para que el script pueda ejecutarse sin la parte de evaluación.
    def evaluate(model, data_loader, device):
        print("Función de evaluación omitida por falta de dependencias.")
        pass
# --- clases de transformacion personalizadas ---
class Compose:
    """compone varias transformaciones juntas.
    Args:
        transforms (lista de objetos 'transform': lista de funciones para componer.
    """
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
class ToTensor:
    """convierte una imagen PIL. a un tensor."""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
class RandomHorizontalFlip:
    '''Voltea horizontalmente la imagen dada de manera aleatoria con una probabilidad dada.'''
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            # image is a tensor, so we get its width from the shape
            _, _, width = image.shape
            image = F.hflip(image)
            if "boxes" in target:
                bbox = target["boxes"]
                # Flip the x-coordinates of the bounding boxes
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
        return image, target
def get_transform(train):
    '''define las transformacionespara aplicar al dataset'''
    transforms = [ToTensor()]
    if train:
        # añade aumento de data para el conjunto de entrenamiento
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
# --- configuracion del proyecto ---
# Rutas de datos (ajustadas a la estructura real del dataset)
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent / 'Proyecto Final ML'/ 'data' / 'VSAIv1' / 'split_ss_444_lsv'
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")


# Directorios de imágenes
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, "images")
VAL_IMAGES_DIR = os.path.join(VAL_DIR, "images")
TEST_IMAGES_DIR = os.path.join(TEST_DIR, "images")

# Directorios de anotaciones
TRAIN_ANN_DIR = os.path.join(TRAIN_DIR, "annfiles")
VAL_ANN_DIR = os.path.join(VAL_DIR, "annfiles")
TEST_ANN_DIR = os.path.join(TEST_DIR, "annfiles")

# Directorio de salida para modelos y resultados
OUTPUT_DIR = os.path.join(DATA_ROOT, "models")
RESULTS_DIR = os.path.join(DATA_ROOT, "results")



# Crear directorios de salida si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_CLASSES = 3

# Configuración de modelos
MODEL_CONFIG = {
    'faster_rcnn': {
        'name': 'rcnn_quad',
        'backbone': 'resnet50',
        'pretrained': True,
        'num_classes': 3,  # background + small-vehicle + large-vehicle
        'min_size': 800,
        'max_size': 1333,
        'batch_size': 4,
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'num_epochs': 10
    },
    'retinanet': {
        'name': 'retinanet_quad',
        'backbone': 'resnet50',
        'pretrained': True,
        'num_classes': 3,  # background + small-vehicle + large-vehicle
        'min_size': 800,
        'max_size': 1333,
        'batch_size': 4,
        'learning_rate': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'num_epochs': 10
    },
    'yolov5': {
        'repo_path': 'yolov5',
        'train_script': 'train.py',
        'weights': 'yolov5s.pt',
        'img_size': 640,
        'epochs': 50,
        'batch_size': 16,
        'name': 'yolov5s_vsaid_results'
    }

}
class VSAIDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.img_ids = [os.path.splitext(os.path.basename(p))[0] for p in self.img_paths]
        self.annotations = self._load_annotations(self.img_ids)

        # Solo conservar imágenes con anotaciones válidas
        self.img_ids = [img_id for img_id in self.img_ids if img_id in self.annotations and self.annotations[img_id]]
        self.img_paths = [os.path.join(self.img_dir, f"{img_id}.png") for img_id in self.img_ids]
        print(f"Dataset cargado: {len(self.img_ids)} imágenes con anotaciones.")

    def _load_annotations(self, img_ids):
        annotations = {}
        for img_id in img_ids:
            ann_file = os.path.join(self.ann_dir, f"{img_id}.txt")
            if not os.path.exists(ann_file):
                continue

            with open(ann_file, 'r') as f:
                lines = f.readlines()

            image_annotations = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue

                # Leer clase
                class_name = parts[8]
                if class_name == 'small-vehicle':
                    class_id = 1
                elif class_name == 'large-vehicle':
                    class_id = 2
                else:
                    continue  # ignorar otras clases

                coords = [float(x) for x in parts[:8]]  # 4 puntos (x, y)
                image_annotations.append({
                    'class': class_id,
                    'coordinates': coords
                })

            annotations[img_id] = image_annotations
        return annotations

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in self.annotations.get(img_id, []):
            coords = np.array(ann['coordinates'], dtype=np.float32).reshape(4, 2)

            # ¡Importante!: Ya están en píxeles absolutos, NO normalizar
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]

            x_min = np.min(x_coords)
            y_min = np.min(y_coords)
            x_max = np.max(x_coords)
            y_max = np.max(y_coords)

            # Validar cajas
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['class'])  # 1 o 2
            areas.append((x_max - x_min) * (y_max - y_min))
            iscrowd.append(0)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
def get_faster_rcnn_model(config):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model
def get_retinanet_model(config):
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head = RetinaNetHead(in_channels, num_anchors, NUM_CLASSES)
    return model

def prepare_yolo_data():
    """
    Prepara los datos para YOLOv5

    Returns:
        Ruta al archivo YAML de configuración
    """
    # Crear directorios para YOLOv5
    yolo_dir = os.path.join(DATA_ROOT, "yolo_quad")
    os.makedirs(yolo_dir, exist_ok=True)

    # Crear subdirectorios
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(yolo_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, split, 'labels'), exist_ok=True)

    # Procesar cada conjunto
    for split, img_dir, ann_dir in [
        ('train', TRAIN_IMAGES_DIR, TRAIN_ANN_DIR),
        ('val', VAL_IMAGES_DIR, VAL_ANN_DIR),
        ('test', TEST_IMAGES_DIR, TEST_ANN_DIR)
    ]:
        # Obtener rutas de imágenes
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

        for img_path in tqdm(img_paths, desc=f"Procesando {split}"):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            ann_file = os.path.join(ann_dir, f"{img_id}.txt")

            if not os.path.exists(ann_file):
                continue

            # Copiar imagen
            dst_img_path = os.path.join(yolo_dir, split, 'images', f"{img_id}.png")
            shutil.copy(img_path, dst_img_path)

            # Convertir anotaciones al formato YOLO
            with open(ann_file, 'r') as f:
                lines = f.readlines()

            # Obtener dimensiones de la imagen
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            yolo_lines = []
            for line in lines:
                parts = line.strip().split()

                # Formato Quad: [x1] [y1] [x2] [y2] [x3] [y3] [x4] [y4] [clase] [tasa_oclusión]
                if len(parts) >= 10:
                    class_name = parts[8]
                    class_id = 0 if class_name == 'small-vehicle' else 1

                    # Convertir cuadrilátero a AABB
                    coords = np.array([
                        (float(parts[0]), float(parts[1])),
                        (float(parts[2]), float(parts[3])),
                        (float(parts[4]), float(parts[5])),
                        (float(parts[6]), float(parts[7]))
                    ])

                    x_min = np.min(coords[:, 0])
                    y_min = np.min(coords[:, 1])
                    x_max = np.max(coords[:, 0])
                    y_max = np.max(coords[:, 1])

                    # Convertir a formato YOLO
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min

                    # Normalizar por las dimensiones de la imagen
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = width / img_width
                    height_norm = height / img_height

                    # Crear la línea para el archivo de etiqueta con los valores normalizados
                    yolo_line = f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"
                    # --- FIN DE LA MODIFICACIÓN ---

                    yolo_lines.append(yolo_line)

            # Guardar anotaciones en formato YOLO
            dst_label_path = os.path.join(yolo_dir, split, 'labels', f"{img_id}.txt")
            with open(dst_label_path, 'w') as f:
                f.writelines(yolo_lines)

    # Crear archivo YAML de configuración
    yaml_path = os.path.join(DATA_ROOT, "yolo_data_quad.yaml")
    yaml_content = {
        'path': os.path.abspath(yolo_dir).replace('\\', '/'), # Usar rutas absolutas para mayor robustez
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 2,  # Número de clases
        'names': ['small-vehicle', 'large-vehicle']

    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    return yaml_path
def train_pytorch_model(model_type, device):
    """
    Entrena un modelo PyTorch (Faster R-CNN o RetinaNet)
    """
    config = MODEL_CONFIG[model_type]
    # crea datasets
    train_dataset = VSAIDataset(TRAIN_IMAGES_DIR, TRAIN_ANN_DIR, transforms=get_transform(train=True))
    val_dataset = VSAIDataset(VAL_IMAGES_DIR, VAL_ANN_DIR, transforms=get_transform(train=False))
    # verifica que los datasets no estan vacios
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty. Check paths: {TRAIN_IMAGES_DIR}, {TRAIN_ANN_DIR}")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty. Check paths: {VAL_IMAGES_DIR}, {VAL_ANN_DIR}")
    # crea dataloaders
    # Nota: en Windows, Es posible que necesites configurar num_workers=0 para evitar ciertos errores de multiprocesamiento.
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=0, collate_fn=utils.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=0, collate_fn=utils.collate_fn
    )
    # Crea el modelo
    model = get_faster_rcnn_model(config) if model_type == 'faster_rcnn' else get_retinanet_model(config)
    model.to(device)
    # Crear optimizador y (scheduler).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay']
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # entrena el modelo
    for epoch in range(config['num_epochs']):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        # La siguiente línea está comentada para evitar el ImportError causado por 'pycocotools'.
        # evaluate(model, val_loader, device=device)

        model_path = os.path.join(OUTPUT_DIR, f"{config['name']}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
    return model


def train_yolov5_model(device, config):
    """
    Prepara los datos, clona YOLOv5 (si no existe) y entrena el modelo.
    La configuración se toma del diccionario global YOLO_CONFIG.

    Args:
        device (str): Dispositivo a utilizar ('cuda' o 'cpu').

    Returns:
        str: Ruta al archivo de pesos del mejor modelo (.pt) o None si falla.
    """
    print("Iniciando la implementación y entrenamiento de YOLOv5...")

    # 1. Preparar los datos y obtener la ruta al archivo .yaml
    print("Preparando los datos al formato requerido por YOLOv5...")
    data_yaml_path = prepare_yolo_data()
    if not data_yaml_path:
        print("\nNo se pudo crear el archivo de configuración .yaml. Abortando entrenamiento.")
        return None
    print(f"Archivo de configuración de datos creado en: {data_yaml_path}")

    # 2. Cargar toda la configuración desde el diccionario global YOLO_CONFIG
    repo_path = config['repo_path']
    train_script = os.path.join(repo_path, config['train_script'])

    # 3. Clonar el repositorio si no existe e instalar dependencias
    if not os.path.exists(repo_path):
        print(f"Clonando el repositorio de ultralytics/yolov5 en {repo_path}...")
        try:
            subprocess.run(
                ['git', 'clone', 'https://github.com/ultralytics/yolov5.git', repo_path],
                check=True, capture_output=True, text=True
            )
            requirements_path = os.path.join(repo_path, 'requirements.txt')
            print("Instalando dependencias para YOLOv5...")
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', requirements_path],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error al clonar YOLOv5 o instalar dependencias: {e.stderr}")
            return None

    # 4. Construir y ejecutar el comando de entrenamiento
    device_id = '0' if 'cuda' in str(device) else 'cpu'
    command = [
        sys.executable,
        train_script,
        '--data', data_yaml_path,
        '--weights', config['weights'],
        '--img-size', str(config['img_size']),
        '--epochs', str(config['epochs']),
        '--batch-size', str(config['batch_size']),
        '--name', config['name'],
        '--device', device_id,
        '--exist-ok'
    ]

    print("\n" + "=" * 50)
    print("Ejecutando comando de entrenamiento de YOLOv5:")
    print(f"  {' '.join(command)}")
    print("=" * 50 + "\n")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace'
)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        rc = process.poll()
        if rc != 0:
            print(f"\nEl entrenamiento de YOLOv5 falló (código de salida: {rc}).")
            return None

    except Exception as e:
        print(f"Ocurrió un error inesperado durante el entrenamiento: {e}")
        return None

    # 5. Devolver la ruta al mejor modelo
    model_path = os.path.join('runs', 'train', config['name'], 'weights', 'best.pt')
    if os.path.exists(model_path):
        print("\nEntrenamiento de YOLOv5 finalizado con éxito.")
        return model_path
    else:
        print(f"\nEl entrenamiento pareció finalizar, pero no se encontró 'best.pt' en la ruta esperada: {model_path}")
        expected_path_in_repo = os.path.join(repo_path, 'runs', 'train', config['name'], 'weights', 'best.pt')
        if os.path.exists(expected_path_in_repo):
            return expected_path_in_repo
        return None


def main():
    """Función principal para implementar y entrenar modelos."""
    print("Iniciando implementación y entrenamiento de modelos para el dataset VSAI...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Usando dispositivo: {device}")
    # Train PyTorch models
    try:
        for model_type in ['faster_rcnn', 'retinanet']:
            print(f"\nEntrenando modelo {model_type}...")
            model = train_pytorch_model(model_type, device)
            config = MODEL_CONFIG[model_type]
            model_path = os.path.join(OUTPUT_DIR, f"{config['name']}_final.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Modelo guardado en: {model_path}")
    except NameError as e:
        print("\nERROR: El entrenamiento no pudo iniciar debido a un problema con las importaciones.")
        print(
            "Asegúrese de que los archivos 'engine.py', 'utils.py', 'coco_eval.py', 'coco_utils.py' y sus dependencias estén disponibles.")
        print(f"Detalle del error: {e}")
        # Train YOLOv5 model
    print("\nEntrenando modelo YOLOv5...")
    yolo_config = MODEL_CONFIG['yolov5']
    model_path = train_yolov5_model(device, yolo_config)
    if model_path:
        print(f"Modelo guardado en: {model_path}")
    else:
        print("Fallo en el entrenamiento de YOLOv5.")

    print("\nImplementación y entrenamiento de modelos completados.")


if __name__ == "__main__":
    # Establecer num_workers a 0 en el DataLoader también puede ayudar a evitar problemas en Windows.
    main()

