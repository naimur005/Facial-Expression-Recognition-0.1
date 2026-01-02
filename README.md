# Facial-Expression-Recognition-0.1

Binary classification: **Happy** vs **Disappointed**

## 📝 Overview

This project aims to build a Facial Expression Recognition (FER) system using a Convolutional Neural Network (CNN). The model is designed to classify facial expressions into two categories: Happy vs Disappointed. By using the FER2013 dataset, the model learns to identify emotional expressions from facial images, which can be applied in various fields like human-computer interaction, emotion recognition, and AI-driven applications.


## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your FER dataset in the `Dataset/` folder with the structure shown above.

### 3. Train the Model

```bash
python train.py
```

This will:
- Train the CNN model
- Save the best model as `fer_model_best.keras`
- Save the final model as `fer_model_final.keras`
- Generate `training_history.png`

### 4. Make Predictions

```bash
python predict.py

# Single image
python predict.py path/to/image.jpg

# Multiple images
python predict.py img1.jpg img2.jpg img3.jpg

# All images in a folder
python predict.py --folder Pictures/
```


## ⚙️ Configuration

```python
# In train.py, modify these lines:
TRAIN_PATH = './Dataset/Train'    
VALID_PATH = './Dataset/Test'     
```


## 🎓 Training

| Parameter | Value |
|-----------|-------|
| Image Size | 48 × 48 |
| Batch Size | 32 |
| Initial Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Max Epochs | 50 |


## 📊 Expected Results

- Training accuracy: ~85%
- Validation accuracy: ~78-82%
- Minimal overfitting (train/val curves close together)


## 🙏 Acknowledgments

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) - Facial Expression Recognition Challenge
- [TensorFlow/Keras](https://www.tensorflow.org/) - Deep learning framework


## 📧 Contact

- Md. Naimur Rahman - naimur005@gmail.com
