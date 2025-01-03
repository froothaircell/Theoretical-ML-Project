# Comparative Study on Deep Learning Models for Semi-Supervised Bounding Box Regression

Welcome to the repository for **Comparative Study on Deep Learning Models for Semi-Supervised Bounding Box Regression**. This repository contains the code, datasets, and analysis for evaluating the performance of different deep learning architectures in semi-supervised bounding box regression tasks.

## Overview

Bounding box regression is essential for object detection tasks in computer vision, enabling precise localization of objects. While supervised methods require extensive labeled datasets, this study explores the semi-supervised approach, leveraging both labeled and unlabeled data. Our work compares the performance of four prominent deep learning architectures under these conditions

## Key Features

Deep Learning Models Evaluated
- MLP (Multi-Layer Perceptron): Baseline feed-forward architecture.
- CNN (Convolutional Neural Network)**: Captures local spatial features effectively.
- Vision Transformer (ViT)**: Utilizes self-attention mechanisms for global context.
- Transformer Encoder: A versatile architecture for capturing dependencies.

## Dataset
- Modified COCO 2017 Dataset: Adapted for semi-supervised learning by introducing unlabeled data.  
  Preprocessing includes normalization and resizing to \(224 \times 224\).

## Evaluation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (\(R^2\))

## Highlights
- Automated preprocessing pipeline for data preparation.
- Comparative analysis of model performance on semi-supervised tasks.
- Insights into trade-offs between model complexity, accuracy, and generalization.

## Repository Structure

```
repo-name/
├── data/            # datasets and preprocessing scripts
├── models/          # DL model architectures (MLP, CNN, Transformer, ViT)
├── notebooks/       # Jupyter notebooks (practice)
├── scripts/         # Training scripts
├── utils/           # Helper scripts (e.g., data loaders, metrics)
├── results/         # Saved results and visualizations
├── README.md        # Project documentation
└── main.py          # Primary entry script
```

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/froothaircell/Theoretical-ML-Project/tree/main.git
   cd desired file/bounding-box-regression
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install numpy, pandas, transformers, voxelmorph
   ```
## Contact and Acknowledgments

For questions, feedback, or collaboration opportunities, please reach out:

- **Name:** Syed Farasat Ali
- **Email:** sali666@uwo.ca
- **Affiliation:** MSc Student, Western University, London Ontario

- **Name:** Memoona Aziz  
- **Email:** maziz86@uwo.ca  
- **Affiliation:** PhD Student, Western University, London, Ontario  

Special thanks to the open-source community and the creators of the COCO dataset for enabling this research.
```
