# Comparative Study on Deep Learning Models for Semi-Supervised Bounding Box Regression

This repository provides a comprehensive framework for evaluating the performance of four distinct deep learning models in the context of semi-supervised bounding box regression. The project aims to deliver actionable insights into the efficacy and behavior of these models in addressing this complex and significant task.

---

## Key Features

### **Deep Learning Models**
- **MLP (Multi-Layer Perceptron)**
- **CNN (Convolutional Neural Network)**
- **ViT (Vision Transformer)**
- **Transformer Encoder**

### **Dataset Preprocessing and Modification**
- Automated pipeline for data normalization and resizing.
- Utilizes a **modified COCO 2017 dataset** tailored for experimentation.

### **Evaluation Metrics**
- Regression-based metrics to ensure robust model evaluation:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)

### **Comparative Analysis**
- Systematic comparison of model performance, providing detailed insights into strengths and weaknesses across models.

---

## Repository Structure

```plaintext
LoanApprovalPrediction-XAI/
├── data/            # Sample datasets and preprocessing scripts
├── models/          # Deep learning model architectures (MLP, CNN, ViT, Transformer)
├── notebooks/       # Jupyter notebooks for experimentation and analysis
├── scripts/         # Core scripts for training and evaluation (e.g., transformer_encoder.py)
├── utils/           # Utility scripts such as train.py for model training and pipeline integration
└── README.md        # Project documentation
