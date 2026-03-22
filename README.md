# Cat vs Dog Image Classification (Random Forest)

## Description

This project aims to classify images of cats and dogs using traditional machine learning techniques.

Instead of using deep learning, this approach relies on handcrafted feature extraction methods (HOG and LBP) combined with a Random Forest classifier.

---

## Methodology

### 🔹 Feature extraction

* **HOG (Histogram of Oriented Gradients)** → captures shapes and edges
* **LBP (Local Binary Patterns)** → captures texture information

### 🔹 Model

* Random Forest classifier

---

## Dataset

* Source: Kaggle (Cat vs Dog dataset)
* Automatically downloaded using KaggleHub

---

## Results

The model is evaluated using:

* Accuracy
* Confusion matrix
* Classification report

---

## Limitations

* Traditional ML approach (no deep learning)
* Feature engineering is manually designed
* Performance is lower than CNN-based models

---

## How to run

```bash id="z2p4lm"
pip install kagglehub scikit-learn numpy opencv-python scikit-image
```

```bash id="x8qj2n"
python detectionChatChien.py
```

---

## Improvements

* Use Convolutional Neural Networks (CNN)
* Data augmentation
* Hyperparameter tuning
* Better image preprocessing

---

## 👨‍💻 Author

Project developed as a learning exercise in machine learning and computer vision.
