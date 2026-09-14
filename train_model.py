import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')

# --- Step 1: Load Data ---
print("Loading data...")
df = pd.read_csv('transactions.csv')

print(f"Dataset shape: {df.shape}")
print(f"Fraud percentage: {df['is_fraud'].mean()*100:.2f}%")

# --- Step 2: Feature Engineering ---
print("\nEngineering features...")

features = [
    'amount',
    'time_since_last_transaction',
    'transaction_hour',
    'is_international',
    'previous_transaction_amount',
    'is_home_city',        # NEW: 1 if transaction is in home city, 0 otherwise
    'is_preferred_device'  # NEW: 1 if device matches preferred, 0 otherwise
]
# Create new features from existing data
df['is_home_city'] = (df['location'] == df['user_id'].map(
    df.groupby('user_id')['location'].first()
)).astype(int)

# For device, we need to check against the user's most common device
user_preferred_device = df.groupby('user_id')['device_type'].agg(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Mobile'
).to_dict()

df['is_preferred_device'] = df.apply(
    lambda row: 1 if row['device_type'] == user_preferred_device.get(row['user_id'], 'Mobile') else 0,
    axis=1
)

X = df[features]
y = df['is_fraud']

# --- Step 3: Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 4: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} transactions")
print(f"Testing set: {X_test.shape[0]:,} transactions")

# --- Step 5: Train XGBoost Model ---
print("\nTraining XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=10,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# --- Step 6: Evaluate ---
print("\n--- Model Evaluation ---")

y_pred_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.1  # Lower threshold catches more fraud
y_pred = (y_pred_proba >= threshold).astype(int)


accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Step 7: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# --- Step 8: Feature Importance ---
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('XGBoost Feature Importance')
plt.savefig('feature_importance.png')
print("Feature importance saved as 'feature_importance.png'")

# --- Step 9: Save the Model ---
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel saved as 'fraud_model.pkl'")
print("Scaler saved as 'scaler.pkl'")

print("\n✅ Model training complete!")