import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2
import glob
from tqdm import tqdm
import json
import yaml

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


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

# Directorio de salida para resultados procesados
OUTPUT_DIR = os.path.join(DATA_ROOT, "processed")

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image_paths():
    """Carga las rutas de todas las imágenes en el dataset."""
    train_images = glob.glob(os.path.join(TRAIN_IMAGES_DIR, "*.png"))
    val_images = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.png"))
    test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))

    all_images = train_images + val_images + test_images

    print(f"Total de imágenes encontradas: {len(all_images)}")
    print(f"  - Entrenamiento: {len(train_images)}")
    print(f"  - Validación: {len(val_images)}")
    print(f"  - Prueba: {len(test_images)}")

    return all_images, train_images, val_images, test_images



def load_annotations(ann_dir, annotation_type="quad"):
    """
    Carga las anotaciones desde los archivos .txt

    Args:
        ann_dir: Directorio que contiene los archivos de anotación
        annotation_type: Tipo de anotación ('obb' o 'quad')

    Returns:
        Un diccionario con las anotaciones por imagen
    """
    annotations = {}

    annotation_files = glob.glob(os.path.join(ann_dir, "*.txt"))
    print(f"Total de archivos de anotación en {ann_dir}: {len(annotation_files)}")

    for ann_file in tqdm(annotation_files, desc=f"Cargando anotaciones de {os.path.basename(ann_dir)}"):
        image_id = os.path.splitext(os.path.basename(ann_file))[0]
        with open(ann_file, 'r') as f:
            lines = f.readlines()

        image_annotations = []
        for line in lines:
            parts = line.strip().split()

            if annotation_type == "obb":
                # Formato OBB: [clase] [x_centro] [y_centro] [ancho] [alto] [ángulo] [tasa_oclusión]
                if len(parts) < 10:
                    annotation = {
                        'class': 'small-vehicle' if parts[0] == '0' else 'large-vehicle',
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4]),
                        'angle': float(parts[5]),
                        'occlusion': int(parts[6])
                    }
                    image_annotations.append(annotation)

            else:  # quad
                # Formato Quad: [x1] [y1] [x2] [y2] [x3] [y3] [x4] [y4] [clase] [tasa_oclusión]
                if len(parts) >= 10:
                    annotation = {
                        'class': 'small-vehicle' if parts[8] == 'small-vehicle' else 'large-vehicle',
                        'coordinates': [
                            (float(parts[0]), float(parts[1])),
                            (float(parts[2]), float(parts[3])),
                            (float(parts[4]), float(parts[5])),
                            (float(parts[6]), float(parts[7]))
                        ],
                        'occlusion': int(parts[9])
                    }
                    image_annotations.append(annotation)

        annotations[image_id] = image_annotations

    return annotations


def create_annotation_dataframe(annotations, annotation_type="quad"):
    """
    Convierte las anotaciones a un DataFrame para análisis

    Args:
        annotations: Diccionario de anotaciones
        annotation_type: Tipo de anotación ('obb' o 'quad')

    Returns:
        DataFrame con las anotaciones
    """
    data = []

    for image_id, image_annotations in annotations.items():
        for ann in image_annotations:
            row = {
                'image_id': image_id,
                'class': ann['class'],
                'occlusion': ann['occlusion']
            }

            if annotation_type == "obb":
                row.update({
                    'x_center': ann['x_center'],
                    'y_center': ann['y_center'],
                    'width': ann['width'],
                    'height': ann['height'],
                    'angle': ann['angle'],
                    'area': ann['width'] * ann['height']
                })
            else:  # quad
                # Calcular área aproximada del cuadrilátero
                coords = np.array(ann['coordinates'])
                area = cv2.contourArea(coords.astype(np.float32))

                # Calcular centro aproximado
                x_center = np.mean(coords[:, 0])
                y_center = np.mean(coords[:, 1])

                row.update({
                    'x_center': x_center,
                    'y_center': y_center,
                    'area': area
                })

            data.append(row)

    df = pd.DataFrame(data)

    # Agregar columnas derivadas
    df['vehicle_size'] = df['class'].map({'small-vehicle': 'small', 'large-vehicle': 'large'})
    df['occlusion_level'] = df['occlusion'].map({0: 'N', 1: 'S', 2: 'M', 3: 'L'})

    return df


def analyze_class_distribution(df):
    """Analiza la distribución de clases en el dataset."""
    class_counts = df['class'].value_counts()

    print("\nDistribución de clases:")
    print(class_counts)
    print(f"Porcentaje de vehículos pequeños: {class_counts['small-vehicle'] / len(df) * 100:.2f}%")
    print(f"Porcentaje de vehículos grandes: {class_counts['large-vehicle'] / len(df) * 100:.2f}%")

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='class', data=df, palette='Set2')
    plt.title('Distribución de Clases de Vehículos')
    plt.xlabel('Clase')
    plt.ylabel('Cantidad')

    # Añadir etiquetas con valores
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=300)
    plt.close()


def analyze_occlusion_distribution(df):
    """Analiza la distribución de niveles de oclusión en el dataset."""
    occlusion_counts = df['occlusion_level'].value_counts()

    print("\nDistribución de niveles de oclusión:")
    print(occlusion_counts)

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='occlusion_level', data=df, palette='Set3', order=['N', 'S', 'M', 'L'])
    plt.title('Distribución de Niveles de Oclusión')
    plt.xlabel('Nivel de Oclusión')
    plt.ylabel('Cantidad')

    # Añadir etiquetas con valores
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'occlusion_distribution.png'), dpi=300)
    plt.close()

    # Análisis cruzado de oclusión por clase
    plt.figure(figsize=(12, 7))
    cross_tab = pd.crosstab(df['class'], df['occlusion_level'])
    cross_tab.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Distribución de Oclusión por Clase de Vehículo')
    plt.xlabel('Clase de Vehículo')
    plt.ylabel('Cantidad')
    plt.legend(title='Nivel de Oclusión')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'occlusion_by_class.png'), dpi=300)
    plt.close()


def analyze_size_distribution(df):
    """Analiza la distribución de tamaños de objetos."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df[df['class'] == 'small-vehicle']['area'], bins=30, kde=True, color='blue')
    plt.title('Distribución de Área - Vehículos Pequeños')
    plt.xlabel('Área')
    plt.ylabel('Frecuencia')

    plt.subplot(1, 2, 2)
    sns.histplot(df[df['class'] == 'large-vehicle']['area'], bins=30, kde=True, color='red')
    plt.title('Distribución de Área - Vehículos Grandes')
    plt.xlabel('Área')
    plt.ylabel('Frecuencia')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'size_distribution.png'), dpi=300)
    plt.close()

    # Estadísticas de tamaño
    size_stats = df.groupby('class')['area'].describe()
    print("\nEstadísticas de tamaño por clase:")
    print(size_stats)

    # Guardar estadísticas en CSV
    size_stats.to_csv(os.path.join(OUTPUT_DIR, 'size_statistics.csv'))


def get_image_set_from_path(image_path):
    """Determina a qué conjunto (train, val, test) pertenece una imagen basado en su ruta."""
    if TRAIN_IMAGES_DIR in image_path:
        return 'train'
    elif VAL_IMAGES_DIR in image_path:
        return 'val'
    elif TEST_IMAGES_DIR in image_path:
        return 'test'
    else:
        return 'unknown'


def analyze_dataset_splits(all_images, train_images, val_images, test_images, annotations_train, annotations_val,
                           annotations_test):
    """
    Analiza la distribución de los conjuntos de datos ya divididos

    Args:
        all_images: Lista de todas las rutas de imágenes
        train_images, val_images, test_images: Listas de rutas por conjunto
        annotations_train, annotations_val, annotations_test: Diccionarios de anotaciones por conjunto
    """
    # Crear DataFrame con información de cada imagen
    image_data = []

    for img_path in all_images:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img_set = get_image_set_from_path(img_path)

        # Determinar qué conjunto de anotaciones usar
        if img_set == 'train' and img_id in annotations_train:
            annotations = annotations_train[img_id]
        elif img_set == 'val' and img_id in annotations_val:
            annotations = annotations_val[img_id]
        elif img_set == 'test' and img_id in annotations_test:
            annotations = annotations_test[img_id]
        else:
            annotations = []

        # Contar vehículos pequeños y grandes
        small_vehicles = sum(1 for ann in annotations if ann['class'] == 'small-vehicle')
        large_vehicles = sum(1 for ann in annotations if ann['class'] == 'large-vehicle')

        # Niveles de oclusión presentes
        occlusion_levels = set(ann['occlusion'] for ann in annotations)
        max_occlusion = max(occlusion_levels) if occlusion_levels else 0

        # Crear categoría para análisis
        if large_vehicles > 0:
            if small_vehicles > 0:
                category = 'mixed'
            else:
                category = 'large_only'
        else:
            category = 'small_only'

        image_data.append({
            'image_id': img_id,
            'path': img_path,
            'set': img_set,
            'small_vehicles': small_vehicles,
            'large_vehicles': large_vehicles,
            'total_vehicles': small_vehicles + large_vehicles,
            'category': category,
            'max_occlusion': max_occlusion
        })

    df_images = pd.DataFrame(image_data)

    # Guardar información de imágenes
    df_images.to_csv(os.path.join(OUTPUT_DIR, 'image_metadata.csv'), index=False)

    # Analizar distribución por conjunto
    split_stats = df_images.groupby('set').agg({
        'image_id': 'count',
        'small_vehicles': 'sum',
        'large_vehicles': 'sum',
        'total_vehicles': 'sum'
    }).reset_index()

    split_stats.columns = ['Split', 'Images', 'Small Vehicles', 'Large Vehicles', 'Total Vehicles']

    # Calcular porcentajes
    total_images = split_stats['Images'].sum()
    total_small = split_stats['Small Vehicles'].sum()
    total_large = split_stats['Large Vehicles'].sum()

    split_stats['Images %'] = split_stats['Images'] / total_images * 100
    split_stats['Small Vehicles %'] = split_stats['Small Vehicles'] / total_small * 100
    split_stats['Large Vehicles %'] = split_stats['Large Vehicles'] / total_large * 100

    print("\nEstadísticas de la división del dataset:")
    print(split_stats)

    # Guardar estadísticas
    split_stats.to_csv(os.path.join(OUTPUT_DIR, 'split_statistics.csv'), index=False)

    # Visualizar distribución de clases por conjunto
    plt.figure(figsize=(12, 6))

    # Preparar datos para gráfico
    splits = split_stats['Split'].tolist()
    small_counts = split_stats['Small Vehicles'].tolist()
    large_counts = split_stats['Large Vehicles'].tolist()

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, small_counts, width, label='Vehículos Pequeños')
    rects2 = ax.bar(x + width / 2, large_counts, width, label='Vehículos Grandes')

    ax.set_title('Distribución de Clases por Conjunto')
    ax.set_xlabel('Conjunto')
    ax.set_ylabel('Cantidad')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()

    # Añadir etiquetas con valores
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution_by_split.png'), dpi=300)
    plt.close()

    return df_images


def normalize_features(df):
    """
    Normaliza las características numéricas del DataFrame

    Args:
        df: DataFrame con características

    Returns:
        DataFrame con características normalizadas y scaler
    """
    # Seleccionar características numéricas
    numeric_features = ['x_center', 'y_center', 'area']
    if 'width' in df.columns and 'height' in df.columns:
        numeric_features.extend(['width', 'height', 'angle'])

    # Crear y aplicar el scaler
    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Guardar el scaler para uso futuro
    import joblib
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'feature_scaler.pkl'))

    return df_normalized, scaler


def perform_pca_analysis(df):
    """
    Realiza análisis de componentes principales para visualización

    Args:
        df: DataFrame con características

    Returns:
        DataFrame con componentes principales
    """
    # Seleccionar características numéricas
    numeric_features = ['x_center', 'y_center', 'area']
    if 'width' in df.columns and 'height' in df.columns:
        numeric_features.extend(['width', 'height', 'angle'])

    # Aplicar PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df[numeric_features])

    # Crear DataFrame con componentes principales
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['class'] = df['class'].values
    pca_df['occlusion_level'] = df['occlusion_level'].values

    # Visualizar PCA
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    sns.scatterplot(x='PC1', y='PC2', hue='class', data=pca_df, palette='Set1', alpha=0.7)
    plt.title('PCA por Clase de Vehículo')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
    plt.legend(title='Clase')

    plt.subplot(2, 1, 2)
    sns.scatterplot(x='PC1', y='PC2', hue='occlusion_level', data=pca_df, palette='Set2', alpha=0.7)
    plt.title('PCA por Nivel de Oclusión')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
    plt.legend(title='Oclusión')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_analysis.png'), dpi=300)
    plt.close()

    # Guardar información de varianza explicada
    explained_variance = pd.DataFrame({
        'Component': ['PC1', 'PC2'],
        'Explained Variance Ratio': pca.explained_variance_ratio_,
        'Cumulative Explained Variance': np.cumsum(pca.explained_variance_ratio_)
    })

    explained_variance.to_csv(os.path.join(OUTPUT_DIR, 'pca_explained_variance.csv'), index=False)

    return pca_df


def main():
    """Función principal para ejecutar el análisis exploratorio."""
    print("Iniciando preparación de datos y análisis exploratorio del dataset VSAI...")

    # 1. Cargar datos
    all_images, train_images, val_images, test_images = load_image_paths()

    # 2. Cargar anotaciones por conjunto
    print("\nCargando anotaciones...")
    annotations_train = load_annotations(TRAIN_ANN_DIR, "quad")
    annotations_val = load_annotations(VAL_ANN_DIR, "quad")
    annotations_test = load_annotations(TEST_ANN_DIR, "quad")

    # 3. Crear DataFrames para análisis
    print("\nCreando DataFrames para análisis...")
    df_train = create_annotation_dataframe(annotations_train, "quad")
    df_val = create_annotation_dataframe(annotations_val, "quad")
    df_test = create_annotation_dataframe(annotations_test, "quad")

    # Combinar todos los DataFrames para análisis global
    df_all = pd.concat([df_train, df_val, df_test])

    # 4. Análisis exploratorio
    print("\n=== Análisis Global de Anotaciones ===")
    analyze_class_distribution(df_all)
    analyze_occlusion_distribution(df_all)
    analyze_size_distribution(df_all)

    # 5. Análisis de la división del dataset
    print("\n=== Análisis de la División del Dataset ===")
    df_images = analyze_dataset_splits(
        all_images, train_images, val_images, test_images,
        annotations_train, annotations_val, annotations_test
    )

    # 6. Normalización de características
    print("\n=== Normalización de Características ===")
    df_all_norm, _ = normalize_features(df_all)

    # 7. Análisis PCA
    print("\n=== Análisis de Componentes Principales ===")
    perform_pca_analysis(df_all_norm)

    # 8. Guardar DataFrames procesados
    print("\n=== Guardando Datos Procesados ===")
    df_all.to_csv(os.path.join(OUTPUT_DIR, 'annotations_all.csv'), index=False)
    df_train.to_csv(os.path.join(OUTPUT_DIR, 'annotations_train.csv'), index=False)
    df_val.to_csv(os.path.join(OUTPUT_DIR, 'annotations_val.csv'), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, 'annotations_test.csv'), index=False)
    df_all_norm.to_csv(os.path.join(OUTPUT_DIR, 'annotations_all_normalized.csv'), index=False)

    print("\nPreparación de datos y análisis exploratorio completados.")
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

