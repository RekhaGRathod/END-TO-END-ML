# 🩺 Pediatric Chest X-ray Pneumonia Detection
This project presents a deep learning-based system for detecting pneumonia from pediatric chest X-ray images using **DenseNet-161** with transfer learning. The system includes model training, evaluation, and a Streamlit-based web application for real-time predictions.
---
## 📌 Overview
Pneumonia is a serious lung infection requiring early detection. Manual analysis of X-ray images can be slow and inconsistent. This project automates pneumonia detection using deep learning to improve accuracy and assist medical diagnosis.
---
## 🎯 Problem Statement
Traditional X-ray diagnosis depends on expert radiologists and can vary between interpretations. This project aims to build an AI system that provides consistent and accurate pneumonia detection from chest X-ray images.
---
## 🧠 Approach
- Transfer learning using **DenseNet-161 (pretrained on ImageNet)**
- Modified final layer for binary classification:
  - Normal
  - Pneumonia
- Frozen pretrained layers for efficient training
---
## ⚙️ Methodology
### Data Preprocessing
- Resize images to 224x224
- Normalize pixel values
- Apply data augmentation:
  - Random flip
  - Rotation
  - Grayscale transformation
### Model Training
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Learning Rate Scheduler used
- Class imbalance handled using weighted loss
---
## 🧪 Experiments
- Compared training vs validation performance
- Monitored overfitting using loss curves
- Applied augmentation to improve generalization
---
## 📊 Results
The model achieved strong performance on the test dataset:
- Accuracy: **~96%**
- High recall for pneumonia detection (~97%)
The model is highly effective in identifying pneumonia cases, which is critical in medical diagnosis.
---
## 📉 Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)
The model shows excellent detection of pneumonia cases with very few false negatives. Some normal cases are misclassified as pneumonia, indicating a bias toward safer predictions in healthcare applications.
---
## 📈 Training Performance
![Training Plot](images/training_plot.png)
The model shows stable convergence with minimal overfitting, indicating effective learning.
---
## 🏗️ Model Architecture
The system uses **DenseNet-161**, a deep convolutional neural network with dense connections that improve feature reuse and gradient flow.
- Initial convolution layer
- Dense blocks for feature learning
- Transition layers for dimensionality reduction
- Global average pooling
- Fully connected classification layer

![Architecture](images/architecture.png)
---
## 💡 Key Insight
The model prioritizes minimizing false negatives (missing pneumonia cases), which is highly desirable in healthcare applications where early detection is critical.
---
## ⚠️ Limitations
- Performance depends on dataset quality and balance
- Limited generalization to real-world hospital data
- Sensitive to variations in X-ray image quality
---
## 🔮 Future Work
- Add explainability using Grad-CAM
- Compare with ResNet / EfficientNet
- Train on larger real-world datasets
- Deploy as a clinical decision support system
---
## 🌐 Web Application
A **Streamlit-based frontend** is implemented for interactive predictions.
### Features:
- Upload chest X-ray images
- Get real-time predictions (Normal / Pneumonia)
- View confidence scores
### Run the app:
```bash
streamlit run app.py
```
---
## 📊 Dataset
The dataset used is the Chest X-Ray Pneumonia dataset from Kaggle.
Note: The dataset is not included in this repository due to size constraints.
---
## ▶️ How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/test.py
```
---
## 📂 Project Structure
```text
pneumonia-detection/
│
├── src/
├── models/
├── images/
├── app.py
├── requirements.txt
└── README.md
```
---
## 👩‍💻 Author
**Rekha Rathod**
AI & ML Engineer | Focused on LLM Systems & AI Research
