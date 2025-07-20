import os
import json
from PIL import Image
from pathlib import Path

# Directorios base
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent / 'Proyecto Final ML' / 'data' / 'VSAIv1' / 'split_ss_444_lsv'

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, "images")
VAL_IMAGES_DIR = os.path.join(VAL_DIR, "images")
TEST_IMAGES_DIR = os.path.join(TEST_DIR, "images")

TRAIN_ANN_DIR = os.path.join(TRAIN_DIR, "annfiles")
VAL_ANN_DIR = os.path.join(VAL_DIR, "annfiles")
TEST_ANN_DIR = os.path.join(TEST_DIR, "annfiles")

# Categorías COCO
categories = [
    {"id": 1, "name": "small-vehicle", "supercategory": "vehicle"},
    {"id": 2, "name": "large-vehicle", "supercategory": "vehicle"}
]
category_name_to_id = {cat["name"]: cat["id"] for cat in categories}

# Configuración de los splits
splits = {
    "train": {"images_dir": TRAIN_IMAGES_DIR, "ann_dir": TRAIN_ANN_DIR},
    "val": {"images_dir": VAL_IMAGES_DIR, "ann_dir": VAL_ANN_DIR},
    "test": {"images_dir": TEST_IMAGES_DIR, "ann_dir": TEST_ANN_DIR}
}

# Crear carpeta de salida
os.makedirs("data/annotations", exist_ok=True)

for split_name, paths in splits.items():
    print(f"Procesando {split_name}...")

    images_dir = paths["images_dir"]
    ann_dir = paths["ann_dir"]
    output_json = f"data/annotations/annotations_{split_name}.json"

    # Estructura COCO completa
    coco = {
        "info": {
            "description": "VSAI Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Universidad Nacional de Colombia",
            "date_created": "2025-07-19"
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "http://creativecommons.org/licenses/by/4.0/"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    annotation_id = 1
    image_id = 1

    for filename in sorted(os.listdir(ann_dir)):
        if not filename.endswith(".txt"):
            continue

        txt_path = os.path.join(ann_dir, filename)
        base_name = os.path.splitext(filename)[0]
        possible_extensions = [".png", ".jpg", ".jpeg"]

        image_path = None
        for ext in possible_extensions:
            tentative_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(tentative_path):
                image_path = tentative_path
                break

        if image_path is None:
            print(f"No se encontró imagen para {base_name}.txt. Saltando.")
            continue

        image_filename = os.path.basename(image_path)

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error al abrir {image_filename}: {e}")
            continue

        coco["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height,
            "license": 1
        })

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            coords = list(map(float, parts[:8]))
            class_name = parts[8]
            occlusion = int(parts[9]) if len(parts) > 9 else 0

            xs = coords[0::2]
            ys = coords[1::2]
            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)

            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = bbox[2] * bbox[3]

            category_id = category_name_to_id.get(class_name)
            if category_id is None:
                print(f"Clase no reconocida: {class_name}")
                continue

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    # Guardar archivo JSON
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Guardado {output_json}")
