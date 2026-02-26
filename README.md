# Machine Learning Internship - Bahash.ai 🤖

This repository contains the projects, assignments, and code implemented during my Machine Learning internship at **Bahash.ai**. The work focuses on building a strong foundation in Machine Learning using Python, covering supervised and unsupervised learning algorithms, model evaluation, and data preprocessing techniques.

---

## 📌 Internship Overview

* **Organization:** Bahash.ai
* **Focus Area:** Machine Learning, Data Science, Python
* **Tools Used:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

---

## 📂 Projects Implemented

During the internship, I developed the following key projects, applying various ML algorithms and techniques:

### 1. Ecommerce Customer Analysis (Regression)
* **File:** `Linear Regression/project.ipynb`
* **Description:** Analyzed customer data for an Ecommerce business to predict the **Yearly Amount Spent** based on user behavior.
* **Key Techniques:**
    * Data Visualization using **Seaborn** (Jointplots for feature correlation).
    * **Linear Regression** (Multi-variable) to determine whether the company should focus on their Mobile App or Website.
    * Feature analysis: *Avg. Session Length, Time on App, Time on Website, Length of Membership*.

### 2. Heart Disease Prediction (Classification & PCA)
* **File:** `heart_disease/heart_disease.ipynb`
* **Description:** Built a diagnostic model to predict the presence of heart disease using the UCI Heart Disease dataset.
* **Key Techniques:**
    * **Data Preprocessing:** Handling missing values (median/mode imputation) and Standard Scaling.
    * **Dimensionality Reduction:** Applied **Principal Component Analysis (PCA)** (0.95 variance) to reduce feature space.
    * **Models Implemented:** Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and Naive Bayes (GaussianNB).
    * **Evaluation:** Cross-validation scores and accuracy metrics.

### 3. Telecom Customer Churn Prediction (Classification)
* **File:** `telecom_churn/telecom_churn.ipynb`
* **Description:** Developed a system to classify telecom customers likely to churn based on service usage and contract details.
* **Key Techniques:**
    * **Preprocessing:** Label Encoding for categorical variables (e.g., Contract type, Payment method).
    * **Models Implemented:** Logistic Regression, SVM, Random Forest, Decision Trees, and Gaussian Naive Bayes.
    * **Performance:** Compared multiple classifiers to identify the best model for churn retention strategies.

---

## 📚 Applied Curriculum & Modules

The internship covered various ML concepts, with the following core modules directly applied in the implementation of the projects above:

* **Data Preprocessing:** Train/Test data splitting, handling missing values, Feature Scaling (StandardScaler), and Label Encoding for categorical data.
* **Supervised Learning (Regression):** Multi-variable Linear Regression.
* **Supervised Learning (Classification):** Logistic Regression, Decision Trees, Support Vector Machine (SVM), Random Forest, and Gaussian Naive Bayes.
* **Unsupervised Learning:** Principal Component Analysis (PCA) for dimensionality reduction.
* **Model Evaluation:** K-Fold Cross Validation and accuracy metrics.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:**
    * `scikit-learn`: For ML algorithms, PCA, and preprocessing.
    * `pandas`: For data manipulation and analysis.
    * `numpy`: For numerical computations.
    * `matplotlib` / `seaborn`: For data visualization and correlation heatmaps.
