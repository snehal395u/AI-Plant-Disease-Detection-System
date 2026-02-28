![header](https://capsule-render.vercel.app/api?type=waving&color=0:11998e,100:38ef7d&height=200&section=header&text=🌿%20AI%20Plant%20Disease%20Detection%20using%20CNN&fontSize=35&fontColor=ffffff)

<img align="left" height="150" src="https://i.imgflip.com/65efzo.gif"  />
<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg" height="65" alt="javascript logo"  />
  <img width="14" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/anaconda/anaconda-original.svg" height="65" alt="anaconda logo"  />
  <img width="14" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" height="65" alt="jupyter logo"  />
  <img width="14" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" height="65" alt="html5 logo"  />
  <img width="14" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/css3/css3-original.svg" height="65" alt="css logo"  />
  <img width="14" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/chrome/chrome-original.svg" height="65" alt="chrome logo"  />
  <img width="14" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" height="65" alt="git logo"  />
  <img width="14" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="65" alt="github logo"  />
  <img width="14" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="65" alt="kaggle logo"  />
  <img width="14" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="65" alt="linkedin logo"  />
  <img width="14" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="65" alt="python logo"  />
  <img width="14" />
  <img src="https://skillicons.dev/icons?i=vscode" height="65" alt="vscode logo"  />
</div>

<div align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=snehal395u.snehal395u&"  />
</div>


![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-ImageProcessing-green?logo=opencv)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-88.7%25-brightgreen)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.88-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

# 🌿 AI Plant Disease Detection System

An intelligent deep learning-based system that detects plant diseases from leaf images using Convolutional Neural Networks built with TensorFlow & Keras.

This project helps farmers, researchers, and agricultural experts detect crop diseases early and prevent yield loss.


## 🎥 Project Demo

![Project Demo](demo.gif)

## 🎥 Demo Preview

👉 [Click here to watch full screen demo](https://screenrec.com/share/vZUruQBYP4)


## 📊 Model Performance

| Metric | Validation | Test |
|--------|------------|------|
| 🎯 Accuracy | 94% | 92% |
| 📊 F1 Score | 0.93 | 0.91 |

---

# 📂 Dataset

Dataset Source: Kaggle PlantVillage Dataset

The dataset contains multiple plant leaf images categorized into:

- Healthy
- Early Blight
- Late Blight
- Leaf Mold
- Powdery Mildew
- Rust
- Bacterial Spot

---

## 📈 Data Augmentation

### Why Data Augmentation?

The original dataset had limited samples per class and class imbalance.

Data augmentation was used to:

- Increase dataset size
- Improve generalization
- Reduce overfitting
- Handle class imbalance

### Techniques Used:

- Rotation
- Horizontal & Vertical Flip
- Zoom
- Width/Height Shift
- Shear Transformation

Before augmentation:
- ~3000 images

After augmentation:
- ~12000+ images

---

## 🔍 Data Preprocessing

Each image goes through:

1. Leaf extraction / background noise reduction
2. Resize to (224, 224, 3)
3. Pixel normalization (0–1 scaling)
4. Label encoding

---

## 📊 Data Split

- 70% Training
- 15% Validation
- 15% Testing

---

# 🧠 Neural Network Architecture

The CNN architecture:

1. Conv2D (32 filters, 3x3)
2. ReLU Activation
3. MaxPooling
4. Conv2D (64 filters)
5. Batch Normalization
6. MaxPooling
7. Dropout
8. Flatten
9. Dense Layer
10. Softmax Output (Multi-class)

---

## Why This Architecture?

Initially, transfer learning was tested using:

- ResNet50
- VGG16

However, due to hardware limitations and overfitting on small dataset size, a custom lightweight CNN was designed.

This resulted in:

- Faster training
- Lower memory usage
- Better generalization
- Stable validation accuracy

---

# 🏋️ Training

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 25
- Batch Size: 32

Best validation accuracy achieved at epoch 21.

---

## 📉 Training Graphs

![Loss plot](loss.JPEG)
![Accuracy plot](accuracy.JPEG)

---

# 🧪 Final Results

The best model detects plant diseases with:

✅ 92% Accuracy on Test Set  
✅ 0.91 F1 Score  
✅ Balanced Precision & Recall  

---

# 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Plant-Disease-Detection-System.git
cd AI-Plant-Disease-Detection-System




