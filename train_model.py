import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')

# --- Step 1: Load Data ---
print("Loading data...")
df = pd.read_csv('transactions.csv')
df['transaction_time'] = pd.to_datetime(df['transaction_time'])

print(f"Dataset shape: {df.shape}")
print(f"Fraud percentage: {df['is_fraud'].mean()*100:.2f}%")

# --- Step 2: Time-Based Split (Reviewer Fix) ---
print("\nPerforming time-based split...")
cutoff_date = df['transaction_time'].quantile(0.70)

train_df = df[df['transaction_time'] < cutoff_date]
test_df = df[df['transaction_time'] >= cutoff_date]

print(f"Cutoff Date: {cutoff_date}")
print(f"Training set: {train_df.shape[0]:,} transactions")
print(f"Testing set: {test_df.shape[0]:,} transactions")

# --- Step 3: Feature Selection ---
features = [
    'amount',
    'time_since_last_transaction',
    'transaction_hour',
    'is_international',
    'previous_transaction_amount',
    'is_home_city',
    'is_preferred_device'
]

X_train = train_df[features]
y_train = train_df['is_fraud']

X_test = test_df[features]
y_test = test_df['is_fraud']

# --- Step 4: Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 5: Train XGBoost Model ---
print("\nTraining XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=10, # Balanced approach: penalizes missing fraud 10x more than a false alarm
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)

# --- Step 6: Evaluate with Custom Threshold ---
print("\n--- Model Evaluation ---")
THRESHOLD = 0.4

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= THRESHOLD).astype(int)

accuracy = (y_pred == y_test).mean()
roc_auc = roc_auc_score(y_test, y_pred_proba)

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print(f"Threshold: {THRESHOLD}")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC (Precision-Recall): {pr_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Step 7: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Threshold = {THRESHOLD})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# --- Step 8: Save the Model ---
model.save_model("fraud_model.json")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'fraud_model.pkl') 
print("\nModel saved as 'fraud_model.json' and 'fraud_model.pkl'")
print("Scaler saved as 'scaler.pkl'")

print("\n✅ Model training complete!")