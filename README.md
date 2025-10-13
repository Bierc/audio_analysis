# Timbre Features & Instrument Classification
Project developed for the **Pattern Recognition** course — UTFPR.

## Objective
Investigate which **audio features** are most strongly associated with **timbre**, analyzing different groups of descriptors (temporal, spectral, and perceptual) and validating their discriminative power through the **automatic classification of musical instruments**.

---

## Project Overview

### 1. Exploratory Analysis
- Initial dataset with **five instruments** (guitar, piano, flute, bass, drums), each playing the **same note and intensity** (~5 seconds).
- Visualizations:
  - **Waveform + Amplitude Envelope**
  - **Mel-spectrogram**
  - **Comparative feature plots**
- Goal: understand how each feature relates to the perceptual differences between instrument timbres.

---

### 2. Dataset Expansion
- Add new samples with **short melodies (3–5 seconds)** using the same instruments.
- Normalize **volume**, **pitch**, and **duration** to control for non-timbral factors.

---

### 3. Feature Extraction and Grouping
Features are organized into three main groups:

| Group | Examples | Relation to Timbre |
|--------|-----------|--------------------|
| **Temporal** | RMS, Attack Time, Zero-Crossing Rate | Percussiveness, transient behavior |
| **Spectral** | Centroid, Bandwidth, Rolloff, Flatness, Flux, Contrast | Brightness, noise level, harmonic texture |
| **Perceptual** | MFCCs, Chroma, HNR | Tonal color, harmonic quality |

- Extract statistical descriptors (mean, std, median) for each feature.  
- Analyze correlations and redundancy among features using **PCA**, **heatmaps**, and **pairplots**.

---

### 4. Instrument Classification
- Split dataset: **80% training / 20% testing**.  
- Test multiple classifiers:
  - **k-NN** – similarity-based baseline  
  - **SVM** – margin-based separability  
  - **Random Forest** – interpretable feature importance
- Evaluate models using:
  - Accuracy  
  - F1-Score  
  - Confusion Matrix  

- Compare results for:
  - Temporal features only  
  - Spectral features only  
  - Perceptual features only  
  - All features combined

---

### 5. Timbre Space Visualization
- Dimensionality reduction with **PCA** and **UMAP**.  
- Visualize instruments in a 2D/3D timbre space.  
- Interpret axes in perceptual terms: *brightness*, *attack strength*, *harmonicity*, etc.  
- Interactive plots built with **Plotly**.

---

## Expected Outcomes
- Deeper understanding of how each feature group describes timbre.  
- Identification of the most discriminative features for instrument recognition.  
- A **reproducible and interpretable** pipeline for timbre-based pattern recognition.

---

## Tools and Libraries
- Python 3.10+  
- Librosa  
- Scikit-learn  
- Pandas  
- Matplotlib / Plotly  
- Jupyter Notebook

---

##  Repository Structure
```
├── data/ # Original and expanded audio datasets
├── notebooks/
│ ├── 01_analysis.ipynb
│ ├── 02_feature_groups.ipynb
│ ├── 03_classification.ipynb
│ └── 04_timbre_space.ipynb
├── scripts/
│ └── extract_features.py
├── outputs/
│ ├── plots/
│ ├── features_summary.csv
│ └── models/
└── README.md
```

