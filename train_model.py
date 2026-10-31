import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json, warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('transactions.csv')
df['transaction_time'] = pd.to_datetime(df['transaction_time'])

features = ['amount', 'time_since_last_transaction', 'transaction_hour', 'is_international', 
            'previous_transaction_amount', 'is_home_city', 'is_preferred_device']

# 3-WAY TIME SPLIT
t1, t2 = df['transaction_time'].quantile([0.60, 0.80])
train_df = df[df['transaction_time'] < t1]
val_df = df[(df['transaction_time'] >= t1) & (df['transaction_time'] < t2)]
test_df = df[df['transaction_time'] >= t2]

X_train, y_train = train_df[features], train_df['is_fraud']
X_val, y_val = val_df[features], val_df['is_fraud']
X_test, y_test = test_df[features], test_df['is_fraud']

# True class ratio
spw = (y_train == 0).sum() / (y_train == 1).sum() 

model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=spw, random_state=42, eval_metric='aucpr')
model.fit(X_train, y_train)

# TUNE THRESHOLD ON VALIDATION (F2-Score)
val_proba = model.predict_proba(X_val)[:, 1]
p, r, thr = precision_recall_curve(y_val, val_proba)
f2 = (5 * p * r) / (4 * p + r + 1e-12)
THRESHOLD = float(thr[np.argmax(f2[:-1])])

# FINAL EVALUATION ON TEST
test_proba = model.predict_proba(X_test)[:, 1]
y_pred = (test_proba >= THRESHOLD).astype(int)
pr_auc = average_precision_score(y_test, test_proba)

print(f"Tuned Threshold: {THRESHOLD:.4f}")
print(f"Test PR-AUC: {pr_auc:.4f}")
print(classification_report(y_test, y_pred))

# SAVE ARTIFACTS
model.save_model("fraud_model.json")
json.dump({"features": features, "threshold": THRESHOLD, "version": "2.0.0", "pr_auc": round(pr_auc, 4)}, open("model_meta.json", "w"))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.savefig('confusion_matrix.png')
plt.close()