

# 🍎 Fruit Classification with CNN

## 🚀 Project Overview

This project implements a **Convolutional Neural Network (CNN)** in PyTorch to classify images of fruits from the [Fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits). The model achieves **96.12% accuracy** on the test set and demonstrates data preprocessing, normalization, and custom CNN design from scratch.

---

## 🖼 Dataset

* **Name:** Fruits-360
* **Source:** [Kaggle - Fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits)
* **Images:** 141,679 images of 208 fruit classes
* **Resolution:** 100x100 pixels
* **Format:** JPEG
* **Split:** Training and testing sets provided

![Sample images from the Fruits-360 dataset](https://www.researchgate.net/profile/Anwar-Ali-Sathio/publication/387745966/figure/fig4/AS:1067760909932544@1661303445082/Sample-Images-From-The-Fruit-360-Dataset-Classification-In-Fruit-360-each-fruit.png)

---

## ⚙️ Methodology

1. **Data Preprocessing**

   * Resize images to 64x64
   * Convert to tensors
   * Normalize RGB channels with mean=0.5 and std=0.5

2. **CNN Architecture**

   * `Conv2d → ReLU → MaxPool` layers
   * Fully connected layers at the end
   * Output layer with `num_classes`

3. **Training**

   * Loss function: `CrossEntropyLoss`
   * Optimizer: `Adam` with learning rate `0.001`
   * Batch size: 64
   * Epochs: 5

4. **Evaluation**

   * Accuracy calculated on test set
   * Test accuracy achieved: **96.12%**

---

## 📈 Results

| Epoch | Training Loss |
| ----- | ------------- |
| 1     | 0.4833        |
| 2     | 0.0484        |
| 3     | 0.0265        |
| 4     | 0.0220        |
| 5     | 0.0186        |

**Test Accuracy:** 96.12%

---

## 🔮 Future Improvements

* Add **data augmentation** (flips, rotations) for better generalization
* Train for more epochs to further improve accuracy
* Deploy the model with **FastAPI** or **Streamlit** for interactive predictions

---

## 💻 How to Run

1. Clone the repo:

   ```bash
   git clone <your-repo-url>
   cd project
   ```

2. Install dependencies:

   ```bash
   pip install torch torchvision
   ```

3. Train the model:

   ```bash
   python train.py
   ```

4. Evaluate on test set:

   ```bash
   python test.py
   ```

