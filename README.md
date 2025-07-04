# 🧬 Breast Cancer Classification using Machine Learning

This project applies machine learning techniques to classify **breast tumors** as **malignant** (cancerous) or **benign** (non-cancerous) using the **Breast Cancer Wisconsin Dataset**. It focuses on building a robust logistic regression model with evaluation via accuracy, ROC curve, and confusion matrix.

---

## 📌 Project Objectives

- Load and understand the Breast Cancer dataset from Scikit-learn.
- Perform exploratory data analysis (EDA) and visualize feature correlations.
- Train a **Logistic Regression** model to classify tumor types.
- Evaluate model performance using accuracy, ROC-AUC, and confusion matrix.
- Use **PCA** to visualize high-dimensional data in 2D.

---

## 📁 Dataset Details

- 📦 Source: `sklearn.datasets.load_breast_cancer()`
- 👩‍⚕️ Samples: 569 patients
- 🔢 Features: 30 numeric features (e.g., radius, texture, perimeter, area)
- 🎯 Target:
  - `0` → Malignant (cancerous)
  - `1` → Benign (non-cancerous)

---

## 📊 Exploratory Data Analysis (EDA)

- Checked for missing values
- Generated statistical summary using `.describe()`
- Created a **correlation heatmap** to identify important feature relationships
- Visualized data using **PCA (Principal Component Analysis)** to project onto 2D

![PCA Projection](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Principal_Component_Analysis_visualization.svg/1200px-Principal_Component_Analysis_visualization.svg.png)

---

## 🧠 Machine Learning Model

- Algorithm: **Logistic Regression**
- Data Split: `80% training / 20% testing`
- Libraries used: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

### 🔍 Evaluation Metrics

| Metric              | Value     |
|---------------------|-----------|
| Training Accuracy   | ~98.24%   |
| Testing Accuracy    | ~94.73%   |
| ROC-AUC Score       | ~0.99     |
| Confusion Matrix    | TP, TN, FP, FN shown visually |

### 📈 ROC Curve & Confusion Matrix

- ROC Curve plotted using `roc_auc_score`, `roc_curve`
- Confusion Matrix displayed with `ConfusionMatrixDisplay`

---

## 🧪 Predictive System

You can input raw feature values like:
```python
input_data = (17.2, 15.8, 110.0, 910.2, 0.1023, 0.1304, 0.1505, 0.0894, 0.1901, 0.0623,
              0.4201, 1.250, 3.150, 30.21, 0.00955, 0.01234, 0.02011, 0.00505, 0.01878, 0.00105,
              21.5, 23.0, 140.5, 1300.2, 0.1201, 0.2105, 0.2455, 0.1123, 0.2401, 0.0851)
