🫁 Pneumonia Prediction Using Deep Learning (DenseNet-161)
📌 Introduction

Pneumonia is a serious respiratory infection that inflames the air sacs in one or both lungs. These air sacs may fill with fluid or pus, leading to symptoms such as cough with phlegm, fever, chills, and difficulty breathing. Pneumonia can be caused by bacteria, viruses, or fungi and can range from mild to life-threatening, especially for infants, elderly people, and individuals with weak immune systems.

Early and accurate diagnosis is critical for effective treatment. This project uses deep learning to automatically detect pneumonia from chest X-ray images, helping doctors make faster and more reliable decisions.

🎯 Problem Statement

Manual analysis of chest X-ray images by radiologists can be:

Time-consuming

Subjective and inconsistent

Limited in regions with fewer medical experts

To address this problem, we propose an AI-based pneumonia detection system using DenseNet-161, a powerful convolutional neural network. The system automatically classifies chest X-ray images as NORMAL or PNEUMONIA, improving diagnosis speed and accuracy.

🏗️ Proposed Architecture
DenseNet-161

DenseNet-161 is a deep convolutional neural network where each layer is connected to every other layer in a feed-forward fashion. This improves feature reuse and reduces the vanishing gradient problem.

🔹 Initial Convolution Layer

Convolution: 7×7, stride 2, padding 3

Batch Normalization

ReLU Activation

Max Pooling: 3×3, stride 2, padding 1

This stage extracts basic features such as edges and textures from the X-ray images.

🔹 Dense Blocks

DenseNet-161 contains 4 dense blocks

Each layer receives inputs from all previous layers

Encourages feature reuse and efficient learning

🔹 Transition Layers

Placed between dense blocks

Reduce the number of feature maps

Apply average pooling to downsample feature maps

Helps control model complexity

🔹 Final Layers

Global Average Pooling

Fully Connected Layer

Softmax Activation

These layers produce probability scores for each class (NORMAL or PNEUMONIA).

🔄 Flow Graph (Data Flow)

Data Input

Chest X-ray images labeled as NORMAL or PNEUMONIA

Data Preprocessing

Resize images to 256 × 256

Center crop to 224 × 224

Data augmentation (random flips, rotations, grayscale transformations)

Convert images to tensors

Model Training

Use pre-trained DenseNet-161 (ImageNet weights)

Replace final classifier layer

Loss Function: Cross Entropy Loss

Optimizer: Adam

Learning rate scheduler

Model Evaluation

Accuracy on validation dataset

Confusion Matrix

Classification Report

Save best performing model

Inference

Load new chest X-ray images

Apply same preprocessing steps

Predict pneumonia

Output predicted class with probability

⚙️ Technologies Used

Python

PyTorch

DenseNet-161

NumPy

OpenCV

Matplotlib / Seaborn

Scikit-learn

📊 Dataset

Chest X-ray images

Two classes:

NORMAL

PNEUMONIA

✅ Results

Improved accuracy compared to manual inspection

Faster diagnosis

Reliable classification

Suitable for real-world healthcare assistance

🧠 Conclusion

This project demonstrates the effective use of DenseNet-161 for pneumonia detection from chest X-ray images. The AI system helps doctors quickly and accurately diagnose pneumonia, especially in areas with limited medical expertise. By automating the diagnostic process, this project aims to improve healthcare outcomes and save lives.

🚀 Future Enhancements

Deploy as a web or mobile application

Support multi-class lung disease detection

Integrate with hospital management systems

Use explainable AI (Grad-CAM) for model interpretability

👤 Author

Rekha Rathod
Artificial Intelligence & Machine Learning
