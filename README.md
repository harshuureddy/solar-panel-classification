# Solar Panel Defect Classification Using Deep Learning

## Project Overview

This project is an end-to-end deep learning application for classifying solar panel images into different defect categories. The goal is to detect whether a solar panel is clean or affected by issues such as dust, bird droppings, snow coverage, electrical damage, or physical damage.

The project covers dataset analysis, CNN model building, transfer learning, hyperparameter tuning, Streamlit application development, and cloud deployment preparation.

---

## Problem Statement

Solar panels can lose efficiency due to dust, snow, physical cracks, bird droppings, and electrical faults. Manual inspection is time-consuming and difficult to scale. This project uses computer vision to automatically classify solar panel conditions from images.

---

## Classes

The dataset contains six classes:

| Class             | Description                        |
| ----------------- | ---------------------------------- |
| Bird-drop         | Bird droppings on the solar panel  |
| Clean             | Solar panel without visible defect |
| Dusty             | Dust accumulation on the panel     |
| Electrical-damage | Electrical fault or damaged cells  |
| Physical-Damage   | Cracks or visible physical damage  |
| Snow-Covered      | Panel covered with snow            |

---

## Dataset Summary

Total images:

```text
885 images
```

Training images:

```text
708 images
```

Validation images:

```text
177 images
```

Image size used:

```text
224 × 224 × 3
```

Batch size:

```text
32
```

Validation split:

```text
20%
```

---

## Technologies Used

* Python
* TensorFlow
* Keras
* EfficientNetB0
* MobileNetV2
* Keras Tuner
* NumPy
* Matplotlib
* PIL
* Streamlit

---

## Project Workflow

```text
Dataset Loading
        ↓
Data Preprocessing
        ↓
Base CNN Model
        ↓
Overfitting Analysis
        ↓
Regularization Experiments
        ↓
Transfer Learning
        ↓
EfficientNetB0 Model
        ↓
Hyperparameter Tuning
        ↓
Model Saving
        ↓
Streamlit Web App
        ↓
Deployment Preparation
```

---

## Base CNN Model

A custom CNN model was first built using:

* Rescaling
* Conv2D
* MaxPooling2D
* Flatten
* Dense layer
* Softmax output layer

Architecture:

```text
Input Image
    ↓
Conv2D(32)
    ↓
MaxPooling2D
    ↓
Conv2D(64)
    ↓
MaxPooling2D
    ↓
Conv2D(128)
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense(128)
    ↓
Dense(6, softmax)
```

---

## Base CNN Results

### 10 Epochs

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 97.46% |
| Validation Accuracy | 64.41% |
| Training Loss       | 0.1042 |
| Validation Loss     | 2.0586 |

Observation:

The model was overfitting because training accuracy became very high while validation accuracy stayed low.

---

### 20 Epochs

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 99.72% |
| Validation Accuracy | 63.28% |
| Training Loss       | 0.0184 |
| Validation Loss     | 1.7914 |

Best validation accuracy during this run:

```text
70.62%
```

Observation:

Increasing epochs did not solve overfitting.

---

## Overfitting Reduction Attempts

The following techniques were tried:

* EarlyStopping
* Dropout
* Batch Normalization
* Learning rate scheduling
* Data augmentation

However, these approaches did not produce strong validation performance with the custom CNN.

---

## Transfer Learning

Since the custom CNN was overfitting, transfer learning was used.

Transfer learning allows the model to use pretrained image features from large datasets such as ImageNet.

---

## MobileNetV2 Model

MobileNetV2 was used with:

```python
include_top=False
weights="imagenet"
base_model.trainable = False
```

### MobileNetV2 Result After 10 Epochs

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 85.88% |
| Validation Accuracy | 69.49% |
| Training Loss       | 0.3907 |
| Validation Loss     | 0.8831 |

Best validation accuracy during this run:

```text
75.71%
```

MobileNetV2 improved performance compared to the base CNN.

---

## EfficientNetB0 Model

EfficientNetB0 was then used because it is a strong transfer learning model for image classification.

Configuration:

```python
EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
```

The base EfficientNetB0 model was frozen:

```python
base_model.trainable = False
```

---

## Class Weights

The dataset was imbalanced, so class weights were calculated.

Class weights used:

```python
{
    0: 0.7530,
    1: 0.7491,
    2: 0.7649,
    3: 1.4110,
    4: 2.1063,
    5: 1.1816
}
```

This helped give more importance to minority classes.

---

## EfficientNetB0 Results

### 15 Epochs

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 95.20% |
| Validation Accuracy | 79.10% |
| Training Loss       | 0.1690 |
| Validation Loss     | 0.6418 |

Best validation accuracy during this run:

```text
80.23%
```

EfficientNetB0 gave the best performance before tuning.

---

## Hyperparameter Tuning

Keras Tuner was used for hyperparameter optimization.

Tuned parameters:

* Rotation factor
* Zoom factor
* Dropout rate
* Dense units
* Learning rate

Search method:

```text
Random Search
```

Number of trials:

```text
3 trials
```

Total tuning time shown in notebook:

```text
28 minutes 18 seconds
```

---

## Best Hyperparameters

```python
{
    "rotation_factor": 0.05,
    "zoom_factor": 0.25,
    "dropout_rate": 0.0,
    "dense_units": 64,
    "learning_rate": 0.002240080343891231
}
```

---

## Final Model Performance

Final validation accuracy after tuning:

```text
79.66%
```

Final validation loss:

```text
0.6468
```

Final selected model:

```text
EfficientNetB0 with Keras Tuner optimization
```

---

## Model Saving

The final model was saved as:

```text
trained_effnet_finetune.keras
```

---

## Streamlit Application

A Streamlit web application was created to make predictions from uploaded solar panel images.

### App Features

* Upload solar panel image
* Display uploaded image
* Resize image to 224×224
* Apply EfficientNet preprocessing
* Predict defect class
* Show confidence score
* Show top 3 predictions
* Show all class probabilities

---

## Streamlit Prediction Flow

```text
User uploads image
        ↓
Image converted to RGB
        ↓
Image resized to 224×224
        ↓
Image converted to NumPy array
        ↓
Batch dimension added
        ↓
EfficientNet preprocessing applied
        ↓
Model predicts probabilities
        ↓
Highest probability class selected
        ↓
Prediction and confidence displayed
```

---

## Example App Output

```text
Prediction: Bird-drop
Confidence: 68.0%

Top 3 Predictions:
1. Bird-drop
2. Dusty
3. Physical-Damage
```

Another tested output:

```text
Prediction: Physical-Damage
Confidence: 45.8%
```

---

## How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone <repository-url>
cd solar-panel-defect-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

### 4. Open in Browser

```text
http://localhost:8501
```

---

## Requirements

Example `requirements.txt`:

```text
streamlit
tensorflow
numpy
pillow
matplotlib
keras-tuner
```

---

## Project Structure

```text
solar-panel-defect-classification/
│
├── app.py
├── trained_effnet_finetune.keras
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── solar_panel_classification.ipynb
│
├── dataset/
│   ├── Bird-drop/
│   ├── Clean/
│   ├── Dusty/
│   ├── Electrical-damage/
│   ├── Physical-Damage/
│   └── Snow-Covered/
│
└── screenshots/
```

---

## Deployment

The application can be deployed using:

* AWS EC2
* Azure
* GCP
* Streamlit Community Cloud
* Docker-based deployment

For AWS EC2 deployment:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then open port `8501` in the EC2 security group.

---

## Key Learnings

* CNN-based image classification
* Overfitting detection
* Transfer learning
* MobileNetV2 implementation
* EfficientNetB0 implementation
* Class imbalance handling
* Keras Tuner hyperparameter optimization
* Streamlit app development
* Model deployment preparation

---

## Final Conclusion

This project successfully built a solar panel defect classification system using deep learning. A custom CNN model was first tested but suffered from overfitting. Transfer learning improved performance, with EfficientNetB0 giving the best results. After hyperparameter tuning, the final model achieved approximately 79.66% validation accuracy and was integrated into a Streamlit application for real-time image classification.

The final app can classify solar panel images into six categories: Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, and Snow-Covered.

