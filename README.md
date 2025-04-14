# **Understanding CNNs**
** Ángel Visedo, Pablo Rodríguez y José Carlos Riego**

### **Índice**

- [**1. Introducción**](#1-introduccion)
- [**2. Requisitos**](#2-requisitos)
- [**3. Desarrollo**](#3-desarrollo-del-proyecto)
  - [**3.1 Entrenamiento de modelos**](#31-entrenamiento-de-modelos)
    - [**3.1.1 Estudio del efecto del _learning rate_**](#311-estudio-del-efecto-del-learning-rate)
    - [**3.1.2 Comparación entre modelos**](#312-comparacion-entre-modelos)
    - [**3.1.3 Obtención de métricas por clase**](#313-obtencion-de-metricas-por-clase)
  - [**3.2 Despliegue de una app en `Streamlit`**](#32-despliegue-de-una-app-en-streamlit)
    - [**3.2.1 Ejecución**](#321-ejecucion)
    - [**3.2.2 Resultados**](#322-resultados)
- [**4. Conclusiones**](#4-conclusiones)

## **1. Introducción**

Este proyecto tiene como objetivo el estudio e implementación de modelos basados en redes neuronales convolucionales (CNNs), mediante la realización de diversas comparativas en las que se analizan distintos modelos y parámetros, así como su impacto en la calidad de las predicciones.

Para ello, se ha empleado el dataset utilizado en el artículo:

> **Lazebnik, S., Schmid, C. y Ponce, J. (2006).** _Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories_. En: Proc. IEEE Conf. Computer Vision and Pattern Recognition, Vol. 2, pp. 2169–2178, 17–22 de junio de 2006.

Este dataset consta de diversas imágenes en blanco y negro de escenas naturales y urbanas, con hasta 15 clases distintas, y está disponible en la página oficial de `Figshare`: [15-Scene Image Dataset](https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177).

## **2. Requisitos**

Para poder ejecutar el proyecto, es necesario tener instalado Python 3.12.9 o superior y las siguientes librerías:

```bash
pip install -r requirements.txt
```

Con esto, ya tenremos todas las dependencias necesarias para ejecutar el proyecto.

## **3. Desarrollo**

Este proyecto está dividido en varias partes:

### **2.1 Entrenamiento modelos**

- Comparar modelos según métricas e hiperparámetros usando weight and bias

- Validación de cada modelo

- Predicciones de cada modelo con Interfaz usuairo mediante streamlit
