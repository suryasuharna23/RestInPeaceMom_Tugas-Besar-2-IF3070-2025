# Tugas Besar 2 - IF3070 Dasar Artificial Intelligence
## Kelompok 12
### Fraud Detection using Logistic Regression & Decision Tree

---

## 👥 Workload Division (3 People)

### 🔹 Person 1 — Data Cleaning ✅ **COMPLETED**
**Responsibilities:**
- Data Cleaning (membersihkan data dari missing values, duplikat, data tidak valid)
- Handling Missing Data
- Removing Duplicates
- Dealing With Outliers
- Data Validation
- Feature Engineering (only cleaning-related: scaling/normalizing if needed during cleaning)

**Deliverables:**
- ✅ Clean dataset (`train_cleaned.csv`, `test_cleaned.csv`)
- ✅ Missing value report
- ✅ Outlier detection summary
- ✅ Duplicate removal log
- ✅ Data validity check report

**Status:** ✅ **100% COMPLETE** - All tasks finished successfully!

**Documentation:** See [PERSON1_DATA_CLEANING_REPORT.md](PERSON1_DATA_CLEANING_REPORT.md)

---

### 🔹 Person 2 — Logistic Regression Model
**Responsibilities:**
- Feature Engineering (interaction features, polynomial features, feature selection)
- Hyperparameter Tuning untuk Logistic Regression
- Model Training & Evaluation (Logistic Regression)
- Comparison with Decision Tree

**Deliverables:**
- Logistic Regression trained model
- Hyperparameter tuning report
- Model evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Feature importance analysis
- Comparison report

---

### 🔹 Person 3 — Decision Tree Model
**Responsibilities:**
- Feature Engineering (if different from Person 2)
- Hyperparameter Tuning untuk Decision Tree
- Model Training & Evaluation (Decision Tree)
- Comparison with Logistic Regression

**Deliverables:**
- Decision Tree trained model
- Hyperparameter tuning report
- Model evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Feature importance analysis
- Comparison report

---

## 📁 Project Structure

```
RestInPeaceMom_Tugas-Besar-2-IF3070-2025/
│
├── README.md                              # This file
├── PERSON1_DATA_CLEANING_REPORT.md        # Person 1 documentation
│
├── src/
│   ├── requirements.txt                   # Python dependencies
│   ├── data_cleaning.py                   # Person 1: Data cleaning script
│   │
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py         # Person 2: Logistic Regression
│   │   └── decision_tree.py               # Person 3: Decision Tree
│   │
│   ├── data/
│   │   ├── train.csv                      # Original training data
│   │   ├── test.csv                       # Original test data
│   │   ├── train_cleaned.csv              # ✅ Cleaned training data
│   │   ├── test_cleaned.csv               # ✅ Cleaned test data
│   │   └── sample_submission.csv          # Submission format
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_handler.py                # ✅ Data loading & preprocessing utilities
│   │   ├── model_saver.py                 # Model saving utilities
│   │   └── visualizations.py              # Visualization utilities
│   │
│   ├── reports/                           # ✅ Generated reports
│   │   ├── missing_value_report_*.txt
│   │   ├── duplicate_removal_log_*.txt
│   │   ├── outlier_detection_summary_*.txt
│   │   ├── data_validity_check_*.txt
│   │   └── cleaning_summary_*.txt
│   │
│   └── Kelompok 12_ IF3070 Dasar Artificial Intelligence _ Tugas Besar 2 Notebook Template.ipynb
│
└── models/                                # Saved trained models
    ├── logistic_regression_model.pkl
    └── decision_tree_model.pkl
```

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/suryasuharna23/RestInPeaceMom_Tugas-Besar-2-IF3070-2025.git
cd RestInPeaceMom_Tugas-Besar-2-IF3070-2025
```

### 2. Install Dependencies
```bash
cd src
pip install -r requirements.txt
```

### 3. Open Jupyter Notebook ✅ 
**IMPORTANT**: All work is done in the Jupyter Notebook!

```bash
cd src
jupyter notebook "Kelompok 12_ IF3070 Dasar Artificial Intelligence _ Tugas Besar 2 Notebook Template.ipynb"
```

### 4. Work in Notebook

**Person 1 Work (COMPLETED ✅):**
- All data cleaning code is in the notebook cells
- Simply run cells sequentially from top to bottom
- Results show in output cells

**Person 2 & 3 Work (TODO):**
- Baseline models already provided in notebook
- Add hyperparameter tuning in designated sections
- Implement advanced feature engineering
- Improve model performance

---

## 📊 Dataset Information

### Original Dataset
- **Training data**: 100,000 rows × 31 columns
- **Test data**: 100,000 rows × 30 columns (no `is_fraud` label)
- **Target variable**: `is_fraud` (binary: 0 = legitimate, 1 = fraudulent)

### Features
**Categorical Features (6):**
- `gender` (M/F)
- `country` (UK, US, IN, BR, CA, DE, ID)
- `device_type` (mobile, desktop, tablet)
- `device_os` (Android, iOS, Windows, Linux)
- `merchant_category` (clothing, restaurants, electronics, gas, etc.)
- `transaction_type` (purchase, topup, withdrawal, transfer)

**Numeric Features (21):**
- `age` - User age
- `transaction_amount` - Transaction amount
- `time_of_day` - Hour of transaction (0-23)
- `day_of_week` - Day of week (0-6)
- `transaction_duration` - Transaction duration in minutes
- `num_prev_transactions` - Number of previous transactions
- `avg_transaction_amount` - Average transaction amount
- `std_transaction_amount` - Standard deviation of transaction amounts
- `transactions_last_24h` - Transactions in last 24 hours
- `transactions_last_1h` - Transactions in last hour
- `failed_login_attempts` - Failed login attempts
- `ip_risk_score` - IP risk score (0-1)
- `device_trust_score` - Device trust score (0-1)
- `account_age_days` - Account age in days
- `has_chargeback_history` - Has chargeback history (binary)
- `shared_ip_users` - Number of users sharing IP
- `shared_device_users` - Number of users sharing device
- `merchant_risk` - Merchant risk score (0-1)
- `country_risk` - Country risk score (0-1)
- `distance_from_home` - Distance from home location (km)
- `is_new_country` - Is new country (binary)

---

## 📈 Data Cleaning Results (Person 1) ✅

### Issues Found & Fixed:
1. **Missing Values**: 7,257 missing values (2.42%)
   - ✅ Fixed using median imputation for numeric features
   
2. **Duplicates**: 0 duplicates found
   - ✅ No action needed - data integrity maintained

3. **Outliers**: 56,784 outlier values detected
   - ✅ Handled using IQR capping method

4. **Data Validation**: All checks passed
   - ✅ Age range valid (18-120)
   - ✅ Transaction amounts non-negative
   - ✅ All probability scores in range (0-1)
   - ✅ Binary columns have valid values (0/1)

### Clean Dataset:
- **Size**: 100,000 rows × 31 columns (0 rows removed)
- **Missing values**: 0 (100% handled)
- **Data quality**: Excellent ✅
- **Ready for modeling**: Yes ✅

---

## 🛠️ Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
ipykernel>=6.25.0
```

---

## 📖 How to Use Data Handler (For Person 2 & 3)

### Quick Start
```python
from utils.data_handler import load_and_prepare_data

# Load and prepare data in one step
X_train, X_val, y_train, y_val, X_test, features = load_and_prepare_data(
    data_dir='data',
    encoding='label',      # or 'onehot'
    scaling='standard',    # or 'minmax'
    test_size=0.2,
    random_state=42
)

# Now ready for model training!
```

### Custom Pipeline
```python
from utils.data_handler import DataHandler

# Initialize
handler = DataHandler(data_dir='data')

# Step by step
handler.load_cleaned_data()
handler.identify_features()
handler.encode_categorical_features(method='label')
handler.scale_numeric_features(method='standard')
X_train, X_val, y_train, y_val = handler.prepare_train_validation_split()
X_test = handler.prepare_test_data()
```

---

## 📝 Notes

### ⚠️ IMPORTANT: All Work in Jupyter Notebook!

**All team members must work in the single Jupyter notebook:**
- `src/Kelompok 12_ IF3070 Dasar Artificial Intelligence _ Tugas Besar 2 Notebook Template.ipynb`

### For Person 2 (Logistic Regression):
1. Open the Jupyter notebook
2. Person 1's cleaning work is already done in cells above
3. Navigate to "4. Modeling and Validation - B. Logistic Regression"
4. Baseline model is already provided
5. Add hyperparameter tuning (C, penalty, solver)
6. Implement feature engineering in designated cells
7. Improve model performance metrics
8. Add visualizations (ROC curve, confusion matrix)

### For Person 3 (Decision Tree):
1. Open the Jupyter notebook
2. Person 1's cleaning work is already done in cells above
3. Navigate to "4. Modeling and Validation - A. DTL"
4. Baseline model is already provided
5. Add hyperparameter tuning (max_depth, min_samples_split, criterion)
6. Implement feature engineering if different from Person 2
7. Improve model performance metrics
8. Add visualizations (tree diagram, feature importance)

---

## 🎯 Evaluation Metrics

Both models should be evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification results

---

## 📊 Expected Outputs

### Model Files:
- `models/logistic_regression_model.pkl`
- `models/decision_tree_model.pkl`

### Reports:
- Hyperparameter tuning results
- Model evaluation metrics
- Feature importance analysis
- Model comparison report

### Predictions:
- `predictions_logistic_regression.csv`
- `predictions_decision_tree.csv`

---

## 🏆 Success Criteria

- ✅ Data cleaning complete (Person 1) - **DONE**
- ⏳ Both models trained successfully
- ⏳ All evaluation metrics calculated
- ⏳ Comprehensive comparison report
- ⏳ Predictions generated for test set
- ⏳ Documentation complete

---

## 📞 Contact

For questions or issues:
- **Person 1 (Data Cleaning)**: Refer to `PERSON1_DATA_CLEANING_REPORT.md`
- **Person 2 (Logistic Regression)**: Check `algorithms/logistic_regression.py`
- **Person 3 (Decision Tree)**: Check `algorithms/decision_tree.py`

---

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/
- Fraud Detection Best Practices: [Add relevant papers/links]

---

**Last Updated**: 2025-11-27  
**Status**: Person 1 Complete ✅ | Person 2 In Progress ⏳ | Person 3 In Progress ⏳
