# **Understanding CNNs**

**Ángel Visedo, Pablo Rodríguez y José Carlos Riego**

### **Índice**

- [**1. Introducción**](#1-Introduccion)
- [**2. Requisitos**](#2-requisitos)
- [**3. Desarrollo**](#3-desarrollo-del-proyecto)
  - [**3.1 Entrenamiento de modelos CNN**](#31-entrenamiento-de-modelos)
    - [**3.1.1 Estudio del efecto del _learning rate_**](#311-estudio-del-efecto-del-learning-rate)
    - [**3.1.2 Comparación entre modelos**](#312-comparacion-entre-modelos)
    - [**3.1.3 Obtención de métricas por clase**](#313-obtencion-de-metricas-por-clase)
  - [**3.2 Despliegue de una app en `Streamlit`**](#32-despliegue-de-una-app-en-streamlit)
    - [**3.2.1 Ejecución**](#321-ejecucion)
    - [**3.2.2 Resultados**](#322-resultados)
- [**4. Conclusiones**](#4-conclusiones)

## **1. Introducción**

Este proyecto tiene como objetivo el estudio e implementación de modelos basados en redes neuronales convolucionales (CNNs), mediante la realización de diversas comparativas en las que se analizan distintos modelos y parámetros, registrando su impacto en la calidad de las predicciones mediante `Weights and Biases`.

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

### **3.1 Entrenamiento de modelos CNN**

Durante el entrenamiento de los modelos, se han fijado los siguiente parámetros:

- `Batch size`: 8. Esto permite un tiempo razonable de entrenamiento, sin sobrecargar los recursos disponibles.
- `Número de épocas`: 5
- `Optimizador`: Adam. Escogido por su rápida convergencia, al implementar learning rate adaptativo y momentum.
- `Image size`: 224 píxeles. Tamaño de las imágenes de muestra.
- `Loss criterion`: Cross entropy. Para comparar las distribuciones de probabilidad entre las clases predicha y real.

#### **3.1.1 Estudio del efecto del _learning rate_**

Para estudiar el efecto del learning rate, se ha entrenado el modelo `convnext-large` con diferentes valores de learning rate, y se han registrado las métricas obtenidas en cada caso.

Cabe destacar que, para el caso de learning rate = 0.0005, se ha empleado un `learning rate scheduler`, la cuál es una técnica que permite ajustar el learning rate durante el entrenamiento, para mejorar la convergencia del modelo.

A continuación, se muestran los resultados obtenidos:

| lr                      | 0.0001 | 0.001 | 0.0005 |
| ----------------------- | ------ | ----- | ------ |
| Validation Accuracy (%) | 41.0   | 86.3  | 92.9   |
| Training Accuracy (%)   | 49.1   | 95.8  | 98.4   |

[GRÁFICA DE ACCURACY EN VALIDACION Y TRAINING]

A parir de estos resultados, podemos concluir lo siguiente:

- El learning rate óptimo de los probados para este modelo es 0.0005, ya que es el que ha obtenido la mejor accuracy en validación.
- El modelo con learning rate = 0.0001 empeora ha sufrido un empeoramiento significativo del accuracy a partir de la primera época. Esto podría ser debido a que, con un learning rate tan bajo, e medida que el modelo va aprendiendo y se van incluyendo imágenes nuevas, el optimizador no es capaz de realizar los cambios necesarios en los pesos de la red para mejorar la predicción.
- En todos los casos, observanmos una diferencia de accuracy en train y validación de al menos 6 puntos porcentuales. Esto indica la presencia clara de overfitting, que podría evitarse con técnicas como el dropout o la regularización L2.
- El uso de `learning rate scheduler` ha permitido mejorar la convergencia del modelo, ya que el learning rate se ha ido ajustando a medida que el modelo iba aprendiendo. Esto se puede observar en la gráfica de accuracy, donde la curva de validación es más suave y presenta menos picos.

#### **3.1.2 Comparación entre modelos**

En este apartado, se ha fijado el learning rate a 0.0005, y se han entrenado los siguientes modelos:

- `convnext-large`
- `efficientnet-b0`
