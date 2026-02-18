# Machine Learning Classification Practice – Decision Tree & KNN

This repository contains Python implementations of supervised machine learning classification algorithms using Decision Trees and K-Nearest Neighbors (KNN).  

The project demonstrates how structured datasets are processed, how models are trained, how predictions are generated, and how performance changes when model parameters are adjusted.

---

## Repository Purpose

This repository was created to demonstrate hands-on implementation of:

- Supervised classification
- Decision Tree modeling
- K-Nearest Neighbors (KNN)
- Train/Test data splitting (80:20)
- Accuracy evaluation
- Confusion matrix analysis
- Model comparison (K=1 vs K=5)
- Feature-based rule extraction

The goal is to show practical understanding of machine learning fundamentals using Python and Scikit-Learn.

---

## Datasets Used

### 1. Customer1.csv – Credit Risk Classification

This dataset simulates a financial risk evaluation problem.

Features:
- Debt
- Collateral
- CreditHistory
- Income

Target:
- Risk (Low Risk, Medium Risk, High Risk)

What the code does:
- Trains a Decision Tree classifier using financial indicators
- Learns classification rules from numeric features
- Outputs readable decision rules using export_text
- Evaluates predictions using accuracy and confusion matrix
- Demonstrates how structured financial data can be used for risk modeling

This simulates a simplified credit risk analysis system.

---

### 2. iris_100.csv – Flower Species Classification

This dataset is based on the classic Iris dataset and contains normalized numerical measurements.

Features:
- Sepal length
- Sepal width
- Petal length
- Petal width

Target:
- 0 → Iris Setosa
- 1 → Iris Versicolor
- 2 → Iris Virginica

What the code does:
- Implements K-Nearest Neighbors classification
- Compares K=1 and K=5
- Demonstrates distance-based prediction
- Shows how K affects overfitting and generalization
- Optionally applies 80:20 train-test split for evaluation

This demonstrates instance-based learning and the bias-variance tradeoff.

---

## Machine Learning Techniques Implemented

### Decision Tree
- Supervised classification algorithm
- Feature-based splitting
- Interpretable rule extraction
- Structured decision boundaries

### K-Nearest Neighbors (KNN)
- Distance-based classification
- Majority voting
- Sensitivity to noise
- Impact of K value on model stability

---

## Key Learning Outcomes

- Loading and processing CSV data using Pandas
- Feature selection using iloc
- Training models with Scikit-Learn
- Evaluating performance with accuracy_score
- Generating confusion matrices
- Understanding overfitting vs generalization
- Comparing tree-based and distance-based algorithms

---

## Technologies Used

- Python 3
- NumPy
- Pandas
- Scikit-Learn

---

## How to Run

Install dependencies:

pip install pandas numpy scikit-learn

Run any script:

python Code_03_05.py

---

## Professional Relevance

This repository demonstrates foundational competencies in:

- Data preprocessing
- Classification modeling
- Algorithm comparison
- Performance evaluation
- Practical machine learning implementation

These skills are applicable in Data Science, Machine Learning Engineering, Risk Modeling, and Predictive Analytics roles.

