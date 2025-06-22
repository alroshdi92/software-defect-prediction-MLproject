
# 🐞 Software Defect Prediction (Binary Classification)

This project aims to predict whether a software module contains a defect (bug) or not based on static code metrics. It follows a full machine learning pipeline using Python and scikit-learn.

---

## 📌 Problem Statement

Early detection of software defects improves code quality, reduces maintenance costs, and speeds up release cycles. We use static features such as lines of code, complexity, and variable counts to classify software as **defective (1)** or **non-defective (0)**.

---

## 🧠 ML Approach

| Step | Description |
|------|-------------|
| **1. Load Data** | Load and inspect the dataset from Kaggle |
| **2. Explore** | View summary stats, check for imbalance, correlation |
| **3. Clean** | Handle missing values ('?'), convert to numeric |
| **4. Transform** | Scale features using `StandardScaler` |
| **5. Train/Test Split** | Use `train_test_split` to create validation sets |
| **6. Model Training** | Test different classifiers: Logistic, RF, GB |
| **7. Tune Models** | Use `GridSearchCV` for best hyperparameters |
| **8. Final Model** | Train best model on all data, evaluate performance |
| **9. Save Model** | Save using `joblib` for reuse/deployment |

---

## 📊 Data Source

- Kaggle Dataset: [Software Defect Prediction](https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction)

---

## 🔧 Requirements

- Python 3.8+
- scikit-learn
- pandas
- seaborn, matplotlib
- joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Clone the repo:
```bash
https://github.com/alroshdi92/software-defect-prediction-MLproject.git
```

2. Run the notebook or Python script:
```bash
python main.py
```

3. Use the saved model:
```python
import joblib
model = joblib.load('software_bug_classifier.pkl')
scaler = joblib.load('scaler.pkl')
```

---

## 📈 Results

- Best Model: Random Forest with GridSearchCV
- Accuracy: ~90%
- ROC AUC Score: ~0.93
- Supports binary prediction: `0 = No Defect`, `1 = Defect`

---

## 📂 Project Structure

```
.
├── main.ipynb              # Jupyter notebook with full workflow
├── main.py                 # Python script version
├── software_bug_classifier.pkl  # Final saved model
├── scaler.pkl              # Saved StandardScaler
├── README.md               # Project documentation
└── requirements.txt        # List of dependencies
```

---

## 🧪 Future Work

- Add multi-class severity classification (e.g. Low/High bugs)
- Visual dashboard (Streamlit or Flask)
- Integrate with static code analysis tools

---

## 👩‍💻 Author

Made with 💻 by Hajer Alroshdi  
Bootcamp: Makeen V2 – ML & DL

