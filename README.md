# Coral Bleaching Classification: Hybrid BoF Approach

This project implements an advanced machine learning pipeline for classifying coral bleaching using a hybrid feature extraction method and Bag-of-Features (BoF).

## Implementation Overview

The script processes coral reef images from the Orpheus Island area to distinguish between **Bleached** and **Healthy** (Unbleached) specimens. It leverages a hybrid framework to capture both high-level semantic features and low-level color variations essential for detecting heat-stressed corals.

### Pipeline & Parameters

* **Data Source**: Images of branching, soft, and massive corals collected via an OM SYSTEM Tough TG-7 camera at Orpheus Island.
* **Feature Extraction (Hybrid)**:
    * **Deep Features**: Extracted using a pre-trained `AlexNet` (features layer) as a backbone.
    * **Color-Texture Features**: 256-bin RGB histograms to capture bleaching-specific color shifts.
* **Vector Quantization**:
    * `dim = 512`: Input image rescaling for high-resolution feature capture.
    * `patch = 50`, `overlap = 0.5`: Sliding window patch extraction to generate local feature descriptors.
    * `cluster_size = 100`: K-means codebook size for the Visual Vocabulary.
* **Classification**:
    * **Model**: Support Vector Machine (SVM) with a quadratic kernel (`poly`, `degree=2`).
    * `k = 4`: Stratified K-Fold Cross-Validation to ensure robust performance metrics.

## Setup and Usage

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/sparkrones/coral_bleach_detection_dl.git
cd coral_bleach_detection_dl
pip install -r requirements.txt
```

### 2. Execution
Run the detection script:
```bash
python bleach_detection.py
```

## References

* Fawad, Ahmad, I., Ullah, A., & Choi, W. (2023). Machine learning framework for precise localization of bleached corals using bag-of-hybrid visual feature classification. *Scientific Reports*, 13(1). doi: [https://doi.org/10.1038/s41598-023-46971-7](https://doi.org/10.1038/s41598-023-46971-7).
