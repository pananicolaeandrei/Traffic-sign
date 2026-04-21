#  Traffic Sign Recognition (CNN + Raspberry Pi)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red)
![License](https://img.shields.io/badge/License-MIT-green)

##  Overview
A **computer vision project** that uses a Convolutional Neural Network (CNN) to classify traffic signs.  
The model is trained on GPU for efficiency and deployed on a **Raspberry Pi** for real-time inference in a low-resource environment.

##  Features
-  CNN-based image classification  
-  GPU training for faster convergence  
-  Deployment on Raspberry Pi (edge AI)  
-  Data augmentation for improved generalization  
-  Lightweight inference pipeline  


##  Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas, Matplotlib  
- Scikit-learn  

## Model Details
- Input: `32x32` grayscale images  
- Architecture: Custom CNN  
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Epochs: 100  
- Batch Size: 128  

##  Workflow
1. Load and label dataset  
2. Preprocess (grayscale, equalization, normalization)  
3. Apply data augmentation  
4. Train CNN model (GPU)  
5. Evaluate on test data  
6. Export model (`.h5`)  
7. Deploy on Raspberry Pi  

##  Raspberry Pi Deployment
- Model transferred to Raspberry Pi  
- Used for real-time traffic sign detection via camera  
- Optimized for low computational resources  

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/traffic-sign-recognition.git
cd traffic-sign-recognition

```
### 2. Install dependencies
```bash 
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```
