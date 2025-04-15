### **Table of Contents**

- [**1. Introduction**](#1-introduction)
- [**2. Requirements**](#2-requirements)
- [**3. Development**](#3-development)
  - [**3.1 Training CNN Models**](#31-training-cnn-models)
    - [**3.1.1 Study of Learning Rate Effect**](#311-study-of-learning-rate-effect)
    - [**3.1.2 Comparison Between Models**](#312-comparison-between-models)
    - [**3.1.3 Obtaining Metrics by Class**](#313-obtaining-metrics-by-class)
  - [**3.2 Deployment of a Streamlit App**](#32-deployment-of-a-streamlit-app)
    - [**3.2.1 Execution**](#321-execution)
    - [**3.2.2 Results**](#322-results)
- [**4. Conclusions**](#4-conclusions)

## **1. Introduction**

This project aims to study and implement models based on Convolutional Neural Networks (CNNs) by conducting various comparisons that analyze different models and parameters, recording their impact on prediction quality using Weights and Biases (W&B).

For this purpose, we used the dataset from the article:

> **Lazebnik, S., Schmid, C., and Ponce, J. (2006).** _Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories_. In: Proc. IEEE Conf. Computer Vision and Pattern Recognition, Vol. 2, pp. 2169–2178, June 17–22, 2006.

This dataset consists of various black and white images of natural and urban scenes, with up to 15 different classes, and is available on the official Figshare page: [15-Scene Image Dataset](https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177).

## **2. Requirements**

To run the project, you need Python 3.12.9 or higher and the following libraries:

```bash
pip install -r requirements.txt
```

Also, if you want to track the training progress of any model, you can modify the W&B configuration in train_models.py and run the following command in the terminal to log into the service:

```bash
wandb login
```

After this, you will be asked to enter the API key previously created in your account dashboard on the W&B website (https://wandb.ai/site).

Finally, you will have all the necessary dependencies to run the project.

## **3. Development**

### **3.1 Training CNN Models**

During model training, the following parameters were set:

- Batch size: 8. This allows for a reasonable training time without overloading the available resources.
- Number of epochs: 5. This was initially estimated to be an adequate number to achieve good results without overfitting the model.
- Optimizer: Adam. Chosen for its rapid convergence by implementing adaptive learning rate and momentum.
- Image size: 224 pixels. Size of the sample images.
- Loss criterion: Cross entropy. To compare probability distributions between predicted and actual classes.

#### **3.1.1 Study of Learning Rate Effect**

To study the effect of the learning rate, the ConvNeXt-Large model was trained with different learning rate values, and the metrics obtained in each case were recorded.

It is worth noting that, for the case of learning rate = 0.0005, a learning rate scheduler was used, which is a technique that allows adjusting the learning rate during training to improve model convergence.

The results obtained are shown below:

| lr                      | 0.0001 | 0.001 | 0.0005 |
| ----------------------- | ------ | ----- | ------ |
| Validation Accuracy (%) | 41.0   | 86.3  | 92.9   |
| Training Accuracy (%)   | 49.1   | 95.8  | 98.4   |

From these results, we can conclude the following:

- The optimal learning rate among those tested for this model is 0.0005, as it achieved the best validation accuracy.
- The model with learning rate = 0.0001 experienced a significant deterioration in accuracy after the first epoch. This could be due to the fact that, with such a low learning rate, as the model learns and new images are included, the optimizer is unable to make the necessary changes to the network weights to improve prediction.
- In all cases, we observe a difference in accuracy between training and validation of at least 6 percentage points. This clearly indicates the presence of overfitting, which could have been prevented by reducing the number of epochs or using techniques such as dropout or L2 regularization.
- The use of a learning rate scheduler has improved model convergence, as the learning rate has been adjusted as the model learned. This can be observed in the accuracy graph, where the validation curve is smoother and shows fewer peaks.

#### **3.1.2 Comparison Between Models**

In this section, the learning rate was set to 0.0005 with a learning rate scheduler, and the following models were trained:

- ConvNeXt-Large
- EfficientNet-B0

| Model                   | EfficientNet-B0 | ConvNeXt-Large |
| ----------------------- | --------------- | -------------- |
| Validation Accuracy (%) | 89.5            | 92.9           |
| Training Accuracy (%)   | 93.6            | 98.4           |

As can be observed, ConvNeXt-Large consistently outperforms EfficientNet-B0 in this problem. The reason for this could be that, despite the high efficiency of EfficientNet-B0 (around 5.3M parameters), ConvNeXt-Large, with approximately 198M parameters, offers a greater capacity to learn complex and detailed representations, which translates into superior performance in accuracy and generalization.

#### **3.1.3 Obtaining Metrics by Class**

[ANGEL'S GRAPH AND CONSIDERATION OF ENSEMBLING CNN MODELS]

### **3.2 Deployment of a Streamlit App**

For the deployment of the application, Streamlit was used, a tool that allows creating web applications easily and quickly, ideal for visualizing machine learning models.

#### **3.2.1 Execution**

There are two ways to run the application.

First, you can access it through the following link:
[Streamlit App](https://idealistai.streamlit.app/)

Alternatively, if you want to run it locally, simply execute the following command in the terminal:

```bash
streamlit run app.py
```

This will open a new window in the browser where you can interact with the application.

To predict the class of an image, simply choose the desired model and upload the image. Then, the 3 classes with the highest probability of being correct will be shown, along with their associated probability.

#### **3.2.2 Results**

As an example, the following figure shows the result of predicting an image of a coast, where it can be observed that the model has correctly predicted the class of the image:

### **4. Conclusions**

In this project, the following conclusions have been drawn:

1. The learning rate is a key parameter in training machine learning models, and its choice can significantly affect the convergence and performance of the model. It is necessary to find a value that avoids optimization divergence without being excessively small to not hinder the learning process.

2. The use of a learning rate scheduler has improved model convergence by adjusting the learning rate during training.

3. The ConvNeXt-Large model has proven to be superior to EfficientNet-B0 in this problem, highlighting the improvement that comes with increasing the number of model parameters, at the cost of longer training time and resource usage.

4. Overfitting has been the biggest challenge identified in this project, and in future implementations, parameters such as the number of epochs should be adjusted, or techniques such as dropout or L2 regularization should be employed to avoid it.

5. In order to achieve adequate precision across all classes, the ideal approach would be to implement an ensemble model that combines the results of several models to improve the accuracy and robustness of the system.
