# Plantar Pressure Analysis System 🦶

A comprehensive deep learning application for analyzing plantar pressure maps to classify foot health patterns. This system combines supervised learning, unsupervised clustering, and an interactive labeling interface for medical research and clinical applications.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🔬 Three Analysis Approaches

1. **Supervised Learning**
   - ResNet18-based CNN architecture with transfer learning
   - Binary classification (Normal/Abnormal)
   - Custom augmentation pipeline
   - Early stopping and learning rate scheduling
   - Gradient clipping for stable training

2. **Unsupervised Learning**
   - K-Means and DBSCAN clustering
   - Comprehensive feature extraction (26+ features)
   - PCA visualization
   - Silhouette score evaluation
   - Anomaly pattern detection

3. **Interactive Labeling Tool**
   - User-friendly Tkinter GUI
   - Keyboard shortcuts for efficiency
   - Progress tracking
   - Session persistence

### 📊 Feature Extraction

- **Pressure Statistics**: Mean, std, max, min, median, range
- **Pressure Distribution**: 10-bin histogram, high-pressure area ratio
- **Spatial Features**: Center of gravity, contact area ratio
- **Asymmetry Analysis**: Left-right comparison, mean pressure differences

### 🎯 Model Architecture

- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Custom Classifier**: 3-layer fully connected network with dropout
- **Input Size**: 224×224×3 RGB images
- **Optimization**: AdamW with weight decay
- **Loss**: Cross-entropy with class weighting support

## 🚀 Installation

### Prerequisites

```bash
Python >= 3.8
CUDA-capable GPU (optional but recommended)
```

### Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install scikit-learn
pip install pandas numpy
pip install matplotlib seaborn
pip install Pillow
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Clone Repository

```bash
git clone https://github.com/yourusername/plantar-pressure-analysis.git
cd plantar-pressure-analysis
```

## 📖 Usage

### 1. Prepare Your Dataset

Organize your plantar pressure images in a directory:

```
Dataset/
├── image001.jpg
├── image002.jpg
├── image003.png
└── ...
```

Update the dataset path in the code:

```python
DATASET_DIR = r"path/to/your/Dataset"
```

### 2. Label Images

Run the application and open the labeling tool:

```bash
python foot_plantar_classification.py
```

- Go to **File → Label Images**
- Use keyboard shortcuts:
  - `1`: Mark as Normal
  - `2`: Mark as Abnormal
  - `Space`: Skip
  - `←/→`: Navigate
- Click **Save & Exit** when done

### 3. Train the Model

From the GUI:
- Go to **Model → Train Model**
- Confirm to start training
- Monitor progress in the console

Training features:
- 80/20 train-validation split
- Data augmentation (flip, rotation, color jitter)
- Early stopping (patience: 15 epochs)
- Model checkpointing (best validation loss)

### 4. Make Predictions

**Single Image:**
- Go to **Controls → Load Image**
- Select an image
- View classification, confidence, and extracted features

**Batch Prediction:**
- Go to **Controls → Batch Predict**
- Select folder with images

### 5. Unsupervised Analysis

**Extract Features:**
- Go to **Analysis → Extract Features**
- Features saved to `features/extracted_features.csv`

**Clustering:**
- Go to **Analysis → Cluster Analysis**
- Results saved to `results/clustering_kmeans_results.csv`
- Visualization saved to `results/clustering_visualization.png`

## 📁 Project Structure

```
plantar-pressure-analysis/
│
├── foot_plantar_classification.ipynb  # Main Jupyter notebook
├── foot_plantar_classification.py     # Standalone Python script
├── README.md                           # This file
├── requirements.txt                    # Dependencies
│
├── Dataset/                            # Your images (not included)
│   └── *.jpg/png
│
├── models/                             # Saved models
│   └── plantar_model.pth
│
├── features/                           # Extracted features
│   └── extracted_features.csv
│
├── results/                            # Analysis results
│   ├── clustering_kmeans_results.csv
│   └── clustering_visualization.png
│
└── logs/                               # Training logs
```

## 🔬 Methodology

### Feature Engineering

The system extracts 26 features from each pressure map:

1. **Basic Statistics** (6): Pressure mean, std, max, min, median, range
2. **Distribution** (11): 10-bin histogram + high-pressure ratio
3. **Spatial** (3): CoG coordinates (normalized) + contact area
4. **Asymmetry** (3): L-R asymmetry score + left/right mean pressures

### CNN Architecture

```
Input (224×224×3)
    ↓
ResNet18 Backbone (pretrained)
    ↓
Dropout(0.5) → FC(512) → ReLU
    ↓
Dropout(0.3) → FC(256) → ReLU
    ↓
Dropout(0.2) → FC(2)
    ↓
Softmax → [Normal, Abnormal]
```

### Training Strategy

- **Optimizer**: AdamW (lr=0.0001, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Augmentation**: Random flip, rotation (±15°), color jitter
- **Regularization**: Dropout, gradient clipping (max_norm=1.0)
- **Validation**: 20% holdout, stratified split

### Clustering

- **Method**: K-Means (k=3) or DBSCAN
- **Preprocessing**: StandardScaler normalization
- **Evaluation**: Silhouette score
- **Visualization**: PCA (2 components)

## 📊 Results

### Expected Performance

With properly labeled data (~200+ images):
- **Training Accuracy**: 70-85%
- **Validation Accuracy**: 65-75%
- **Clustering Silhouette Score**: 0.15-0.30

### Interpretation

The system identifies:
- ✅ Normal pressure distribution patterns
- ⚠️ Abnormal gait patterns
- 🔍 Outliers requiring manual review
- 📈 Distinct pressure profile clusters

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
BATCH_SIZE = 8  # or 4
```

**2. Image Loading Errors**
```python
# Already handled with truncated image support
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

**3. DataLoader Worker Issues (Windows)**
```python
# Already set to safe values
num_workers = 0
pin_memory = False
```

**4. Insufficient Data**
- Minimum recommended: 50+ labeled images per class
- Use data augmentation to increase effective dataset size

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to new functions
- Include unit tests for new features
- Update README for significant changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Research Project** - *Initial work*

## 🙏 Acknowledgments

- ResNet architecture from torchvision models
- Plantar pressure imaging research community
- Open-source PyTorch and scikit-learn contributors

## 📧 Contact

For questions, issues, or collaborations:
- Open an issue on GitHub
- Email: your.email@example.com

## 🔮 Future Work

- [ ] Multi-class classification (specific conditions)
- [ ] Grad-CAM visualization for interpretability
- [ ] Real-time video analysis
- [ ] Mobile deployment
- [ ] Integration with medical record systems
- [ ] Longitudinal patient tracking

---

**⭐ Star this repository if you find it helpful!**

---

## 📚 References

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
2. Scikit-learn: Machine Learning in Python
3. PyTorch: An Imperative Style, High-Performance Deep Learning Library

---

*Last Updated: November 2025*
