# 🧬 Colorectal Cancer Detection & Stage Estimation Using Deep Learning + Unsupervised Clustering

This project presents a **hybrid AI pipeline** capable of detecting colorectal cancer from histopathology images and estimating its stage **without requiring manually labeled stage data**.  
The system uses **VGG16 transfer learning**, **deep feature embeddings**, **KMeans clustering**, and a **final 5-class classifier** to predict:

- **Normal tissue**
- **Cancer Stage 1**
- **Cancer Stage 2**
- **Cancer Stage 3**
- **Cancer Stage 4**

The model achieves **99% accuracy** for cancer detection and **93% accuracy** for stage prediction.

---

## 🚀 Project Highlights

- 🔬 **Binary cancer detection** using VGG16 (99% accuracy)
- 🤖 **Automatic stage discovery** with KMeans clustering
- 🧠 **5-class classifier** trained on cluster-derived labels (93% accuracy)
- 📊 PCA cluster visualization for explainability
- 🧮 Confusion matrix + accuracy/loss curves for evaluation
- 🏥 Designed for real-world digital pathology workflows
- 💡 Works **without stage labels** — fully annotation-efficient

---

## 📁 Dataset

**LC25000 Colon Histopathology Dataset**

- `colon_n` — Normal tissue  
- `colon_aca` — Adenocarcinoma (malignant)

Malignant images are embedded using VGG16 and clustered into **4 stage groups** with KMeans.

📌 Dataset source:  
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

---

## 🧠 Model Architecture

### **1️⃣ Binary Cancer Detector (VGG16)**
- Transfer learning from ImageNet  
- Output: Normal / Cancer  
- Accuracy: **99%**

### **2️⃣ Deep Feature Embeddings**
- Extracted from VGG16 FC layer  
- 4096-dimensional embedding vectors

### **3️⃣ KMeans Stage Clustering**
- Clusters malignant embeddings into 4 groups  
- Used as surrogate stage labels

### **4️⃣ Final 5-Class Stage Classifier**
- Predicts: Normal + Stages 1–4  
- Accuracy: **93%**

---

## 📊 Visual Results

### **Binary Classification Performance**
**(Insert your images)**  
- Accuracy Curve  
- Loss Curve  

---

### **KMeans Clustering Visualization**
**(Insert PCA Cluster Plot)**

---

### **Stage Classification Performance**
**(Insert Accuracy Curve + Confusion Matrix)**

---

## 📂 Project Structure
├── binary_model/ # VGG16 cancer detection model
├── feature_extraction/ # Deep embedding extraction
├── clustering/ # KMeans clustering + PCA
├── stage_classifier/ # 5-class stage classifier
├── Final_Dataset/ # Dataset after clustering
├── results/ # Plots: accuracy, loss, PCA, confusion matrix
├── block_diagram.png # Pipeline architecture diagram
├── README.md # Documentation
└── requirements.txt # Dependencies


---

## 💻 Installation & Usage

Clone the repository:

```bash
git clone https://github.com/Jitendar-369/Detection-and-Stage-Prediction-of-Colorectal-Cancer.git
cd colorectal-cancer-staging

Install dependencies:

pip install -r requirements.txt

Train binary classifier:
python train_binary_classifier.py

Extract deep features:
python extract_features.py

Run KMeans clustering:
python cluster_stages.py

Train 5-class classifier:
python train_stage_classifier.py
```
## 🛠️ Tech Stack

Python 3.x

TensorFlow / Keras

VGG16

Scikit-Learn (KMeans, PCA)

NumPy

Matplotlib

OpenCV

---

## 🌟 Key Achievements

✔ 99% accuracy on binary cancer detection

✔ 93% accuracy on final 5-class stage classification

✔ Fully automated staging without stage labels

✔ PCA + confusion matrix visualizers

✔ Scalable pipeline suitable for clinical workflows

---

## 🔮 Future Enhancements

Extend to whole-slide images (WSI)

Use transformer-based models (ViT, Swin)

Deploy as a web app (Flask/Streamlit)

Apply self-supervised learning (SimCLR, MoCo)

Incorporate clinical metadata for multimodal predictions

---

## 📄 License


MIT License © 2025

---

## 🤝 Acknowledgements

LC25000 Dataset

TensorFlow / Keras

Medical research in colorectal cancer pathology

---
