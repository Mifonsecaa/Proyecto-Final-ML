# Proyecto Final de Machine Learning

## Descripción del Proyecto

Este repositorio alberga el proyecto final del curso de Machine Learning, centrado en la **detección de vehículos en escenarios complejos**. El objetivo principal es evaluar la eficacia de diferentes modelos de detección de objetos, específicamente Faster R-CNN, RetinaNet y YOLOv5, y analizar cómo su rendimiento varía en función del tamaño del vehículo (pequeño o grande) y el nivel de oclusión.

El proyecto aborda la preparación de datos, el entrenamiento de modelos y la evaluación de su rendimiento utilizando métricas estándar de detección de objetos.

## Características Principales

*   **Detección de Objetos**: Implementación y evaluación de modelos de vanguardia como Faster R-CNN, RetinaNet y YOLOv5 para la detección de vehículos.
*   **Análisis de Datos**: Procesamiento y análisis del dataset VSAIv1, incluyendo la carga de imágenes, el parseo de anotaciones (cuadriláteros), y la derivación de características como el tamaño del vehículo y el nivel de oclusión.
*   **Entrenamiento de Modelos**: Configuración de entornos de entrenamiento con datasets personalizados y técnicas de aumento de datos (e.g., volteo horizontal).
*   **Evaluación Exhaustiva**: Medición del rendimiento de los modelos utilizando métricas de COCO (Average Precision y Average Recall) para diferentes clases de vehículos y niveles de oclusión.
*   **Tecnologías**: Desarrollado principalmente con Python, utilizando bibliotecas clave como PyTorch, torchvision, OpenCV, scikit-learn, pandas, numpy, matplotlib y seaborn.

## Instalación y Configuración

Para configurar el entorno del proyecto y ejecutar los scripts, siga los siguientes pasos:

### 1. Clonar el Repositorio

```bash
git clone https://github.com/Mifonsecaa/Proyecto-Final-ML.git
cd Proyecto-Final-ML
```

### 2. Crear y Activar un Entorno Virtual

Se recomienda el uso de un entorno virtual para gestionar las dependencias del proyecto.

**En Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**En macOS y Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

Instale todas las bibliotecas necesarias listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Descargar y Organizar el Dataset

El dataset utilizado es VSAIv1, disponible en Kaggle. Descárguelo y organícelo según la estructura esperada por los scripts.

*   **Descargar el dataset**: Acceda a [VSAIv1 Dataset en Kaggle](https://www.kaggle.com/datasets/dronevision/vsaiv1/data).
*   Una vez descargado (ej. `dataset.zip`), descomprímalo y coloque su contenido en la ruta `Proyecto-Final-ML/data/VSAIv1/split_ss_444_lsv/`.

## Uso

### Preparación de Datos

El script `data_preparation.py` se encarga de cargar las imágenes y anotaciones, así como de realizar un análisis exploratorio inicial.

```bash
python data_preparation.py
```

### Entrenamiento de Modelos

El script `model_train.py` permite entrenar los modelos de detección de objetos configurados (Faster R-CNN, RetinaNet, YOLOv5).

```bash
python model_train.py
```

### Evaluación de Modelos

Para evaluar el rendimiento de los modelos entrenados, utilice el script `model_evaluation.py`.

```bash
python model_evaluation.py
```

## Resultados y Conclusiones

Los resultados de este proyecto buscan determinar qué modelo de detección de objetos (Faster R-CNN, RetinaNet, YOLOv5) ofrece el mejor rendimiento en la identificación de vehículos, prestando especial atención a la precisión en la detección de vehículos de diferentes tamaños y bajo diversas condiciones de oclusión. Las métricas de evaluación de COCO proporcionan una base cuantitativa para comparar la robustez y la eficacia de cada enfoque.

## Autor

Miguel Fonseca -mifonsecaa
David Urrego - shirohigexe

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo `LICENSE` para más detalles.
