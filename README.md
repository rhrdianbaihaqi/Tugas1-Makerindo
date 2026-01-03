# Telco Customer Churn Prediction

## 📌 Project Overview

This project is a machine learning experiment to predict customer churn
(customers who stop using a service) in a telecommunications company.
The model is designed as a decision support system to help identify
customers at risk of churn so that preventive actions can be taken.

---

## 🎯 Problem Statement

Customer churn is a critical business problem because acquiring new
customers is more expensive than retaining existing ones. The goal of
this project is to build a supervised machine learning model that can
predict whether a customer will churn based on historical customer data.

---

## 📊 Dataset

- **Name:** Telco Customer Churn Dataset
- **Source:** Kaggle (Open Dataset)
- **Type:** Tabular data
- **Target Variable:** `Churn` (Yes / No)

---

## ⚙️ Machine Learning Pipeline

The project follows a standard machine learning workflow:

1. Data Preparation
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
   - Handling missing values
   - One-Hot Encoding for categorical features
   - Feature scaling for numerical features
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Hyperparameter Tuning
8. Model Comparison

---

## 🤖 Models Used

- **Logistic Regression** (Baseline Model)
- **Random Forest Classifier** (Comparison Model)

Logistic Regression is used as a baseline due to its simplicity and
interpretability, while Random Forest is used to capture non-linear
relationships in the data.

---

## 📈 Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Special attention is given to **recall for the churn class (Yes)**,
as failing to detect churned customers is more costly from a business
perspective.

---

## 🧪 Hyperparameter Tuning

GridSearchCV is used to tune the Logistic Regression model, with recall
for the churn class (`Yes`) as the primary optimization metric.

---

## 📊 Results Summary

- Logistic Regression achieved stable performance with good interpretability.
- Random Forest did not always outperform Logistic Regression in terms of accuracy.
- This indicates that simpler models can be effective for structured,
  tabular datasets with mostly linear relationships.

---

## 🧠 Key Insights

- The dataset is imbalanced, with fewer churned customers.
- Model evaluation should not rely solely on accuracy.
- Recall is a critical metric for churn prediction problems.

---

## 🛠️ Tools & Technologies

- Python
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Jupyter Notebook
- Git & GitHub

---

## 📁 Project Structure

telco-churn-prediction/
├── data/
│ └── WA*Fn-UseC*-Telco-Customer-Churn.csv
├── notebook/
│ └── churn_prediction.ipynb
├── README.md
└── requirements.txt

---

## ✅ Conclusion

This project demonstrates a complete and clean implementation of a
machine learning pipeline for a real-world classification problem.
It is suitable for academic assignments and as a portfolio project
for junior AI / Machine Learning Engineer roles.
