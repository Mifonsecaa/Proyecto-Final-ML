# 🚀 Proyecto Final de Machine Learning

Este repositorio contiene el código y los recursos para el proyecto final del curso de Machine Learning. El objetivo es [**Determinar qué modelo es mejor en la detección precisa de vehículos en escenarios complejos, y cómo varía esta efectividad según el tamaño del vehículo.**].

---

## ⚙️ Requisitos Previos

Asegúrate de tener instalado lo siguiente:
* Python 3.8 o superior
* pip (manejador de paquetes de Python)

---

## 🛠️ Instalación y Configuración

Sigue estos pasos para configurar el entorno del proyecto.

**1. Clonar el Repositorio**
```bash
git clone [https://github.com/Mifonsecaa/Proyecto-Final-ML.git]
cd Proyecto-Final-ML
```

**2. Crear y Activar un Entorno Virtual**

Es una buena práctica trabajar en un entorno virtual para aislar las dependencias del proyecto.

* **En Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
* **En macOS y Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

**3. Instalar Dependencias**

Instala todas las bibliotecas necesarias que se encuentran en `requirements.txt`.
```bash
pip install -r requirements.txt
```

**4. Descargar y Organizar el Dataset** 📥

Este es el paso más importante para que los scripts funcionen correctamente.

* **Descarga el dataset** desde el siguiente enlace:
    > **[https://www.kaggle.com/datasets/dronevision/vsaiv1/data]**

* Una vez descargado, obtendrás un archivo comprimido (ej: `dataset.zip`). Descomprímelo.

* Dentro del archivo descomprimido, encontrarás tres carpetas: **`train`**, **`test`** y **`val`**.

* **Mueve** estas tres carpetas (`train`, `test`, `val`) a la siguiente ruta exacta dentro del proyecto:
    > `Proyecto-Final-ML/data/VSAlv1/split_ss_444_lsv/`

    Si las carpetas `data`, `VSAlv1` o `split_ss_444_lsv` no existen, debes crearlas manualmente.

---

## 📂 Estructura de Carpetas Final

Para asegurarte de que todo está correcto, la estructura de tus directorios debe verse así:

```
Proyecto-Final-ML/
|
├── data/
│   └── VSAlv1/
│       └── split_ss_444_lsv/
│           ├── train/
│           │   ├── (carpeta imágenes y carpetas de anotaciones)...
│           ├── test/
│           │   ├── (carpeta imágenes y carpetas de anotaciones)...
│           └── val/
│               ├── (carpeta imágenes y carpetas de anotaciones)...
|      
├── model_train.py
└── model_evaluation.py
|
├── requirements.txt    # Lista de dependencias
└── README.md           # Este archivo
```

---

