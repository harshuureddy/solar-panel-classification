# Solar Panel Defect Classification Using Deep Learning

## Project Overview

This project focuses on detecting and classifying defects in solar panels using Deep Learning and Computer Vision techniques. The objective is to automatically identify different types of solar panel conditions from images and classify them into predefined categories.

The project implements Convolutional Neural Networks (CNNs), Transfer Learning, Hyperparameter Optimization, and a Streamlit web application for real-time predictions.

---

## Business Problem

Solar panels are exposed to various environmental conditions that can reduce their efficiency and energy output. Manual inspection of solar panels is time-consuming, expensive, and difficult to scale.

An automated defect detection system can:

* Reduce inspection costs
* Improve maintenance efficiency
* Detect faults early
* Increase solar energy production
* Enable real-time monitoring

---

## Project Objectives

* Build an image classification model for solar panel defect detection.
* Compare custom CNN models with Transfer Learning models.
* Improve model performance using Hyperparameter Optimization.
* Deploy the final model as a Streamlit web application.
* Enable users to upload solar panel images and receive instant predictions.

---

## Dataset Information

The dataset contains approximately 885 solar panel images belonging to six classes.

| Class             | Description                        |
| ----------------- | ---------------------------------- |
| Clean             | Normal solar panel without defects |
| Bird-Drop         | Bird droppings on panel surface    |
| Dusty             | Dust accumulation on panel         |
| Electrical-Damage | Electrical faults or damaged cells |
| Physical-Damage   | Cracks or physical damage          |
| Snow-Covered      | Solar panel covered with snow      |

### Dataset Distribution

| Class             | Images |
| ----------------- | ------ |
| Clean             | ~194   |
| Bird-Drop         | ~200   |
| Dusty             | ~190   |
| Electrical-Damage | ~103   |
| Physical-Damage   | ~69    |
| Snow-Covered      | ~120   |

Total Images: ~885

---

## Technology Stack

### Programming Language

* Python

### Deep Learning Framework

* TensorFlow
* Keras

### Computer Vision

* OpenCV
* PIL (Python Imaging Library)

### Data Processing

* NumPy
* Matplotlib

### Model Optimization

* Keras Tuner

### Web Application

* Streamlit

### Deployment

* AWS EC2

---

## Project Workflow

### Step 1: Data Understanding

The dataset was analyzed to understand:

* Class distribution
* Data imbalance
* Image quality
* Defect patterns

Several classes had fewer samples, making the dataset moderately imbalanced.

---

### Step 2: Data Preprocessing

Images were resized to:

```python
224 x 224
```

Additional preprocessing included:

* Image normalization
* Tensor conversion
* EfficientNet preprocessing

---

### Step 3: Base CNN Model

A custom CNN model was created using:

* Conv2D Layers
* MaxPooling Layers
* Flatten Layer
* Dense Layers
* Softmax Output Layer

Architecture:

```text
Input Image
      ↓
Conv2D
      ↓
MaxPooling
      ↓
Conv2D
      ↓
MaxPooling
      ↓
Conv2D
      ↓
MaxPooling
      ↓
Flatten
      ↓
Dense
      ↓
Output (6 Classes)
```

---

## Initial Results

| Metric              | Result |
| ------------------- | ------ |
| Training Accuracy   | ~99%   |
| Validation Accuracy | ~62%   |

Observation:

The model suffered from severe overfitting.

---

## Step 4: Overfitting Reduction Techniques

Several regularization techniques were applied:

### Early Stopping

Automatically stops training when validation performance stops improving.

### Batch Normalization

Added after convolution layers to stabilize learning.

### Dropout

Randomly disables neurons during training to improve generalization.

### Learning Rate Scheduling

Gradually reduces learning rate during training.

### Data Augmentation

Implemented:

* Random Rotation
* Random Zoom
* Horizontal Flip

Despite these improvements, validation accuracy remained unsatisfactory.

---

## Step 5: Transfer Learning

To improve performance, pretrained CNN models were used.

### MobileNetV2

Configuration:

* ImageNet weights
* Frozen base model
* GlobalAveragePooling2D
* Dense Layer
* Softmax Output

Results:

| Metric              | Result |
| ------------------- | ------ |
| Training Accuracy   | ~91%   |
| Validation Accuracy | ~74%   |

MobileNetV2 improved performance significantly compared to the custom CNN.

---

### EfficientNetB0

Configuration:

* ImageNet pretrained weights
* Transfer Learning
* Data Augmentation
* Class Weights

Results:

| Metric              | Result |
| ------------------- | ------ |
| Training Accuracy   | ~96%   |
| Validation Accuracy | ~81%   |

EfficientNetB0 became the best-performing model.

---

## Step 6: Class Weight Balancing

Since some classes had fewer samples, class weights were calculated.

Formula:

```python
class_weight = total_images / (num_classes * images_per_class)
```

Benefits:

* Improved minority class learning
* Reduced bias toward majority classes

---

## Step 7: Hyperparameter Optimization

Keras Tuner was used to automatically search for the best model configuration.

### Tuned Parameters

* Learning Rate
* Dropout Rate
* Dense Units
* Rotation Factor
* Zoom Factor

### Search Configuration

* 20 Trials
* Early Stopping Enabled
* EfficientNetB0 Base Model

Execution Time:

```text
Approximately 7 Hours
```

---

## Final Model Performance

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | ~96%  |
| Validation Accuracy | ~83%  |

Final Selected Model:

```text
EfficientNetB0 + Hyperparameter Tuning
```

---

## Model Saving

The final model was saved as:

```text
trained_effnet_finetune.h5
```

or

```text
trained_effnet_finetune.keras
```

---

## Streamlit Application

A Streamlit application was developed to allow real-time image classification.

### Features

* Upload solar panel image
* Predict defect category
* Display confidence score
* Show Top-3 Predictions
* Display probabilities for all classes

---

## Application Workflow

```text
Upload Image
      ↓
Resize Image
      ↓
Preprocess Input
      ↓
Load EfficientNet Model
      ↓
Generate Predictions
      ↓
Identify Highest Probability Class
      ↓
Display Result
```

---

## Sample Output

```text
Prediction:
Dusty

Confidence:
60.2%

Top Predictions:
1. Dusty
2. Bird-Drop
3. Physical-Damage
```

---

## Deployment

The Streamlit application was deployed using AWS EC2.

### Deployment Steps

1. Launch EC2 Instance
2. Install Python Dependencies
3. Upload Project Files
4. Install Requirements
5. Run Streamlit Server
6. Open Port 8501
7. Access Application via Public IP

---

## Project Structure

```text
Solar-Panel-Defect-Classifier/
│
├── app.py
├── trained_effnet_finetune.h5
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── Clean/
│   ├── Bird-Drop/
│   ├── Dusty/
│   ├── Electrical-Damage/
│   ├── Physical-Damage/
│   └── Snow-Covered/
│
├── notebooks/
│   ├── CNN_Model.ipynb
│   ├── MobileNetV2.ipynb
│   ├── EfficientNetB0.ipynb
│   └── Hyperparameter_Tuning.ipynb
│
└── screenshots/
```

---

## Key Learnings

* Image Classification using CNNs
* Transfer Learning with MobileNetV2
* Transfer Learning with EfficientNetB0
* Data Augmentation Techniques
* Hyperparameter Optimization using Keras Tuner
* Streamlit Application Development
* AWS Cloud Deployment
* End-to-End Deep Learning Project Lifecycle

---

## Future Enhancements

* Increase dataset size
* Add object detection capabilities
* Implement defect localization
* Integrate drone-based image collection
* Deploy using Docker and Kubernetes
* Add monitoring dashboard
* Build REST APIs using FastAPI

---

## Conclusion

This project demonstrates a complete end-to-end Deep Learning workflow for solar panel defect detection. Starting from dataset exploration and custom CNN models, the project evolved into a high-performing Transfer Learning solution using EfficientNetB0 and Hyperparameter Optimization. The final model achieved approximately 83% validation accuracy and was successfully integrated into a Streamlit web application for real-time predictions and cloud deployment.
