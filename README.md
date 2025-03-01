# SRGAN - Super Resolution using GANs

## 📌 Project Overview
**SRGAN (Super-Resolution Generative Adversarial Network)** is a deep learning-based model that enhances the resolution of low-quality images to high-quality images. This project implements SRGAN from scratch using Keras and TensorFlow, which includes building the generator, discriminator, and VGG-based feature extractor models.

The dataset used is from **[MIRFLICKR-25K](http://press.liacs.nl/mirflickr/mirdownload.html)**, where images are resized to **128x128 pixels** as high-resolution (HR) and **32x32 pixels** as low-resolution (LR).

## 🚀 Features
- Image Super Resolution using GANs
- Generator model with residual blocks
- Discriminator model for adversarial training
- VGG19-based perceptual loss
- Real-time Image Upscaling
- Custom Image Testing

## 🛠️ Tech Stack
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- tqdm

## 🔑 Installation

### Prerequisites
- Python 3.8+
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

### Steps
```bash
# Clone the repository
git clone https://github.com/your-username/SRGAN.git

# Navigate to the project directory
cd SRGAN

# Install dependencies
pip install -r requirements.txt

# Run dataset preparation
python dataset_prep.py

# Train the model
python srgan.py
```

## 📄 Usage
1. Prepare your dataset with high-resolution images.
2. Run `dataset_prep.py` to resize images into **HR (128x128)** and **LR (32x32)**.
3. Train the SRGAN model using `srgan.py`.
4. Use the trained model to upscale new low-resolution images.

## 📌 Dataset Preparation
The `dataset_prep.py` script resizes images into two directories:
- `hr_images/`: High-Resolution images (128x128)
- `lr_images/`: Low-Resolution images (32x32)

## ⚙️ Model Architecture
### Generator
- Convolutional layers with PReLU activation
- Residual Blocks
- Upsampling layers
- Final Convolutional Layer

### Discriminator
- Convolutional layers with LeakyReLU activation
- Batch Normalization
- Fully Connected Layers

### Perceptual Loss
- VGG19 feature extraction for perceptual loss calculation
- Combination of adversarial and content loss

## 🎯 Future Improvements
- Use deeper ResNet blocks
- Implement Progressive GANs
- Add Noise Reduction Preprocessing
- Use Transfer Learning for faster convergence

## ⭐ Acknowledgements
- Original SRGAN paper by Ledig et al.
- Keras and TensorFlow community
- MIRFLICKR Dataset providers

