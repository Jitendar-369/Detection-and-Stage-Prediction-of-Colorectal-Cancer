🧬 Colorectal Cancer Detection & Stage Estimation Using Deep Learning + Unsupervised Clustering

This project presents a hybrid AI pipeline capable of detecting colorectal cancer from histopathology images and estimating its cancer stage without requiring manually labeled stage data.
The system uses VGG16 transfer learning, deep feature embeddings, KMeans clustering, and a final 5-class classifier to predict:

Normal tissue

Cancer Stage 1

Cancer Stage 2

Cancer Stage 3

Cancer Stage 4

The model achieves 99% accuracy for cancer detection and 93% accuracy for stage prediction.

🚀 Project Highlights

🔬 Binary cancer detection using VGG16 (99% accuracy)

🤖 Automatic stage discovery using KMeans clustering on deep embeddings

🧠 5-class stage classifier trained on cluster-derived labels (93% accuracy)

📊 PCA cluster visualization for explainability

🧮 Confusion matrix + accuracy/loss curves for evaluation

🏥 Designed for real-world digital pathology workflows

💡 Requires no stage labels — fully annotation-efficient

📁 Dataset

LC25000 Colon Histopathology Dataset

Classes used:

colon_n — Normal tissue

colon_aca — Adenocarcinoma (malignant)

Malignant class further divided into 4 clusters using KMeans

📌 Dataset source:
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

Dataset Pipeline:

Load images from colon_n and colon_aca

Train VGG16 binary classifier

Extract 4096-dim embeddings for malignant samples

Cluster embeddings into 4 stage groups

Train final 5-class classifier

🧠 Model Architecture
1️⃣ VGG16 Binary Cancer Detector

Input: 224×224 histopathology image

Output: Normal / Cancer

Accuracy: 99%

2️⃣ Deep Embedding Extractor

Extracts 4096-dim features from VGG16’s FC layer

3️⃣ KMeans Stage Clustering

Clusters malignant embeddings into 4 groups

Provides surrogate stage labels

4️⃣ Five-Class Stage Classifier

Learns to classify:

Normal

Stage 1

Stage 2

Stage 3

Stage 4

Accuracy: 93%

📊 Visual Results
Binary Classification Performance
Accuracy	Loss
(Insert Fig 1)	(Insert Fig 2)
KMeans Clustering Visualization
PCA Cluster Plot
(Insert Fig 3)
Stage Classification Performance
5-Class Accuracy Curve	Confusion Matrix
(Insert Fig 4)	(Insert Fig 5)
📂 Project Structure
├── binary_model/                # VGG16 cancer detection model
├── feature_extraction/          # Deep embedding extraction scripts
├── clustering/                  # KMeans clustering + PCA visualization
├── stage_classifier/            # 5-class classifier training
├── Final_Dataset/               # Dataset used after clustering
├── results/                     # Accuracy plots, PCA, confusion matrix
├── block_diagram.png            # Pipeline architecture
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies

💻 Installation & Usage

Clone the repository:

git clone https://github.com/yourusername/colorectal-cancer-staging.git
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

🛠️ Tech Stack

Python 3

TensorFlow / Keras

VGG16 Transfer Learning

Scikit-Learn (KMeans, PCA)

NumPy

Matplotlib

OpenCV

🌟 Key Achievements

✔ 99% accuracy on binary cancer detection

✔ 93% accuracy on final stage classification

✔ Fully automated staging without labelled stages

✔ PCA-based cluster explainability

✔ High-quality confusion matrix performance

🔮 Future Enhancements

Train on whole-slide images (WSI)

Integrate ViT/Transformers for richer embeddings

Deploy as a web app (Streamlit / Flask)

Apply self-supervised learning (SimCLR, MoCo)

Multi-modal fusion (image + clinical metadata)

📄 License

MIT License © 2025

🤝 Acknowledgements

LC25000 Dataset (Kaggle)

TensorFlow / Keras

Medical researchers involved in colorectal cancer pathology
