💳 Online Payment Fraud Detection using Machine Learning

A machine learning project that automatically detects fraudulent online payment transactions using Logistic Regression, Random Forest and XGBoost, evaluated with AUC-ROC, confusion matrices and classification reports.

📌 Overview

Digital payment platforms process millions of transactions every day, and only a tiny fraction are fraudulent. Checking them manually is impossible, so this project builds an automated system that flags suspicious transactions.

The main challenge is extreme class imbalance: only ~0.11% of transactions are fraud. Because of this, accuracy is misleading (a model that predicts "legitimate" every time would score ~99.89%), so AUC-ROC is used as the primary evaluation metric.

This was a team project of four members. The work was divided into data exploration, preprocessing, model training and model evaluation.

📊 Dataset
Source: PaySim – Synthetic Financial Dataset (Kaggle)
Used in this project: ~10.5 lakh (1,048,575) transactions
Target column: isFraud (0 = Legitimate, 1 = Fraud)
Column	Description
step	Time unit (1 step = 1 hour)
type	Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
amount	Transaction amount
nameOrig	Sender account ID
oldbalanceOrg	Sender balance before transaction
newbalanceOrig	Sender balance after transaction
nameDest	Receiver account ID
oldbalanceDest	Receiver balance before transaction
newbalanceDest	Receiver balance after transaction
isFraud	Target – 1 = Fraud, 0 = Legitimate

Class distribution

Class	Count	Share
Legitimate (0)	1,047,433	99.89%
Fraud (1)	1,142	0.11%
🛠️ Tech Stack
Category	Tools
Language	Python 3.x
Data handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine learning	Scikit-learn, XGBoost
Environment	Jupyter Notebook
🔄 Project Workflow
Load Data → EDA → Preprocessing → Model Training → Model Evaluation → Prediction
Data loading – read the CSV with Pandas and inspect shape, types and missing values.
Exploratory Data Analysis – transaction type counts, fraud vs legitimate split, fraud rate by type, amount distribution, correlation heatmap.
Preprocessing
One-hot encode the type column
Drop nameOrig and nameDest (random account IDs)
Separate features (X) and target (y)
70/30 train–test split with stratify=y to keep the same fraud ratio in both sets
Model training – Logistic Regression, Random Forest and XGBoost trained on the same training data.
Model evaluation – AUC-ROC, confusion matrix, precision, recall and F1-score for all three models.
Prediction – a helper function that takes a new transaction and returns a fraud probability.
🔍 Key Findings from EDA
Fraud is extremely rare (~0.11%), a highly imbalanced problem.
Fraud occurs only in TRANSFER and CASH_OUT transactions.
Fraudulent transactions typically involve larger amounts and drain the sender's account to zero.
🤖 Models
Model	How it works	Key parameters
Logistic Regression	Linear decision boundary, baseline model	max_iter=1000
Random Forest	Many decision trees voting together	n_estimators=7, criterion='entropy'
XGBoost	Trees built sequentially, each correcting the previous one's errors	n_estimators=200, max_depth=6, learning_rate=0.1
📈 Model Evaluation
Why AUC-ROC and not accuracy?

With 99.89% legitimate transactions, accuracy rewards a model that never catches any fraud. AUC-ROC measures how well a model separates fraud from legitimate transactions across all probability thresholds, so it stays meaningful under heavy imbalance.

AUC-ROC comparison
Model	AUC-ROC
Logistic Regression	0.9804
Random Forest	0.9372
XGBoost	0.9979
Confusion matrices (test set: 314,573 transactions, 343 fraud)
Model	TN	FP	FN	TP
Logistic Regression	314,217	13	202	141
Random Forest	314,223	7	73	270
XGBoost	314,226	4	56	287
Fraud-class metrics (derived from the confusion matrices)
Model	Precision	Recall	F1-score
Logistic Regression	0.92	0.41	0.57
Random Forest	0.97	0.79	0.87
XGBoost	0.99	0.84	0.91

Formulas used

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-score  = 2 × (Precision × Recall) / (Precision + Recall)
Takeaway
Logistic Regression misses 202 of 343 frauds (recall 41%), which is too risky for a banking use case.
Random Forest improves recall to 79%, but only 7 trees were used and it has the lowest AUC.
XGBoost is the best overall: highest AUC, highest precision and recall, only 4 false alarms and 56 missed frauds.

---

## 📁 Repository Structure

```
online-payment-fraud-detection/
│
├── fraud_detection.ipynb     # Main notebook (EDA, training, evaluation)
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── images/                   # Saved plots (optional)
```

> The dataset (`new_data.csv`) is **not** included in the repository because of its size. Download it from Kaggle (link above), rename it to `new_data.csv` and place it in the project folder.

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/korsalasya/online-payment-fraud-detection.git
   cd online-payment-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the dataset**: download from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1), rename it to `new_data.csv` and keep it in the same folder as the notebook.

4. **Launch Jupyter and run all cells**
   ```bash
   jupyter notebook fraud_detection.ipynb
   ```

**`requirements.txt`**
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

---

## 🔮 Future Improvements

- Handle imbalance with SMOTE or class weights
- Tune the decision threshold below 0.5 to raise fraud recall
- Add engineered features such as balance-error and account-drained flags
- Add k-fold cross-validation and hyperparameter tuning
- Add SHAP for model explainability
- Deploy as a Streamlit web app for real-time prediction

---

## 🙌 Acknowledgements

- Dataset: [PaySim – Synthetic Financial Dataset for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Reference article: [GeeksforGeeks – Online Payment Fraud Detection using Machine Learning in Python](https://www.geeksforgeeks.org/online-payment-fraud-detection-using-machine-learning-in-python/)
