# Coral Bleaching Classification: Hybrid BoF Approach

This repository implements an advanced machine learning pipeline for classifying coral bleaching using a hybrid feature extraction method and Bag-of-Features (BoF). [cite_start]This approach is designed to overcome the limitations of simpler CNN models—such as the 60.0% accuracy achieved by the initial VGG16 model [cite: 80, 128]—by integrating deep visual features with handcrafted color-texture descriptors.

## Implementation Overview

[cite_start]The script processes coral reef images from the Orpheus Island area [cite: 5, 50] to distinguish between **Bleached** and **Healthy** (Unbleached) specimens. [cite_start]It leverages a hybrid framework to capture both high-level semantic features and low-level color variations essential for detecting heat-stressed corals[cite: 131, 133].

### Pipeline & Parameters

* [cite_start]**Data Source**: Images of branching, soft, and massive corals collected via an OM SYSTEM Tough TG-7 camera at Orpheus Island[cite: 58].
* **Feature Extraction (Hybrid)**:
    * [cite_start]**Deep Features**: Extracted using a pre-trained `AlexNet` (features layer) as a backbone[cite: 133].
    * [cite_start]**Color-Texture Features**: 256-bin RGB histograms to capture bleaching-specific color shifts[cite: 133].
* **Vector Quantization**:
    * `dim = 512`: Input image rescaling for high-resolution feature capture.
    * `patch = 50`, `overlap = 0.5`: Sliding window patch extraction to generate local feature descriptors.
    * `cluster_size = 100`: K-means codebook size for the Visual Vocabulary.
* **Classification**:
    * **Model**: Support Vector Machine (SVM) with a quadratic kernel (`poly`, `degree=2`).
    * `k = 4`: Stratified K-Fold Cross-Validation to ensure robust performance metrics.

## Expected Outcomes

[cite_start]By combining spatial invariant features (BoF) with color-texture data, this model aims to improve the **Recall** (initially 40.0% [cite: 82, 129][cite_start]) and **Accuracy** (initially 60.0% [cite: 80, 128]) reported in preliminary studies. [cite_start]The objective is to align performance with state-of-the-art benchmarks that reach up to 96.2% accuracy using similar hybrid frameworks[cite: 133].

## References

* Fawad, Ahmad, I., Ullah, A., & Choi, W. (2023). Machine learning framework for precise localization of bleached corals using bag-of-hybrid visual feature classification. [cite_start]*Scientific Reports*, 13(1). doi: [https://doi.org/10.1038/s41598-023-46971-7](https://doi.org/10.1038/s41598-023-46971-7).
