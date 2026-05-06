# 🌳 Bagging Ensemble — Diabetes Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ensemble-orange?style=flat)
![Algorithm](https://img.shields.io/badge/Algorithm-Bagging%20Ensemble-purple?style=flat)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20ML-red?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

> 🏥 An ensemble learning project applying **Bagging** technique
> to predict diabetes using the **Pima Indians Diabetes Dataset** —
> demonstrating how combining multiple weak learners creates
> a stronger, more accurate model.

---

## 🎯 Problem Statement

Diabetes prediction from clinical data is a critical healthcare challenge:
- Single ML models can be unstable and overfit
- **Bagging (Bootstrap Aggregating)** reduces variance and overfitting
- Multiple models trained on different data subsets → better generalization

**Goal:** Compare single Decision Tree vs Bagging Ensemble accuracy
on diabetes classification.

---

## 📂 Dataset — Pima Indians Diabetes

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes hereditary function |
| `Age` | Patient age |
| **`Outcome`** | **Target — 1=Diabetic, 0=Non-diabetic** |

**Source:** Pima Indians Diabetes Database (Kaggle)
**Size:** 768 patients

---

## 📁 Notebooks

| File | Description |
|------|-------------|
| `Bagging.ipynb` | Bagging from scratch — concept demonstration |
| `BaggingClassifier.ipynb` | Scikit-learn BaggingClassifier implementation |

---

## 🧠 How Bagging Works

```
Original Dataset (768 samples)
        ↓
Bootstrap Sampling (with replacement)
├── Sample 1 → Train Tree 1
├── Sample 2 → Train Tree 2
├── Sample 3 → Train Tree 3
└── ... (n_estimators trees)
        ↓
Majority Voting (Classification)
        ↓
Final Prediction
```

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Base estimator
base = DecisionTreeClassifier()

# Bagging — 100 trees, each trained on 80% of data
bagging = BaggingClassifier(
    estimator=base,
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    random_state=42
)

bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)

print(f"Single Tree Accuracy: {accuracy_score(y_test, tree_pred):.4f}")
print(f"Bagging Accuracy:     {accuracy_score(y_test, y_pred):.4f}")
# Bagging > Single Tree ✅
```

---

## 📊 Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Single Decision Tree | ~72% | High variance, overfits |
| **Bagging Ensemble** | **~77%** | Lower variance, more stable |

**Key Finding:** Bagging reduces overfitting by averaging predictions
across multiple trees trained on different data subsets.

---

## 🔄 Bagging vs Other Ensemble Methods

| Method | Base Learners | Key Idea | Best For |
|--------|--------------|----------|----------|
| **Bagging** | Same algorithm | Reduce variance | Unstable models |
| Random Forest | Decision Trees | Feature randomness | Classification |
| Boosting | Weak learners | Reduce bias | High accuracy |
| Stacking | Different algos | Meta-learning | Complex problems |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.x |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/tashfeen786/BaggingEnsemble.git
cd BaggingEnsemble

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Run notebooks
jupyter notebook Bagging.ipynb
jupyter notebook BaggingClassifier.ipynb
```

---

## 🏗️ Project Structure

```
BaggingEnsemble/
│
├── Bagging.ipynb                      # Bagging concept notebook
├── BaggingClassifier.ipynb            # Sklearn implementation
├── diabetes_data/                     # Dataset folder
├── pima-indians-diabetes-database.zip # Raw dataset
└── README.md
```

---

## ⚠️ One Fix Needed

**Remove the ZIP file from repo** — unzip karo aur CSV file directly rakho:
```bash
# Unzip and push CSV instead
unzip pima-indians-diabetes-database.zip
git rm pima-indians-diabetes-database.zip
git add diabetes_data/
git commit -m "Replace zip with raw CSV files"
```

---

## 🔮 Future Improvements

- [ ] Compare with **Random Forest** (Bagging + feature randomness)
- [ ] Add **Boosting** (AdaBoost, XGBoost) comparison
- [ ] **ROC-AUC curve** comparison across models
- [ ] **SHAP values** for feature importance
- [ ] Deploy with **FastAPI** as prediction endpoint

---

## 👨‍💻 Author

**Tashfeen Aziz** — AI/ML Engineer & Python Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/tashfeen-aziz-b51361292)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/tashfeen786)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:tashfeen247@gmail.com)

---

⭐ **If you found this project helpful, please give it a star!**
