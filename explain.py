import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load model and scaler ---
print("Loading model and scaler...")
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')
print("Loaded!")

# --- Feature names (must match training order) ---
FEATURES = [
    'amount',
    'time_since_last_transaction',
    'transaction_hour',
    'is_international',
    'previous_transaction_amount',
    'is_home_city',
    'is_preferred_device'
]

# --- Load a sample of transactions ---
print("\nLoading transactions...")
df = pd.read_csv('transactions.csv')

# Take a random sample of 500 transactions
sample = df.sample(500, random_state=42)
X_sample = sample[FEATURES].values
X_scaled = scaler.transform(X_sample)

# --- Create SHAP explainer (TreeSHAP for XGBoost) ---
print("\nCreating SHAP explainer...")
explainer = shap.TreeExplainer(model)

# Compute SHAP values
print("Computing SHAP values...")
shap_values = explainer.shap_values(X_scaled)
print(f"SHAP values computed for {len(shap_values)} transactions")

# --- Global Explanation: Feature Importance ---
print("\n--- Global Feature Importance (SHAP) ---")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved shap_summary.png")

# --- Global Explanation: Bar Chart ---
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, feature_names=FEATURES, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved shap_bar.png")

# --- Local Explanation: Explain ONE fraud prediction ---
print("\n--- Explaining ONE fraud prediction ---")
fraud_rows = sample[sample['is_fraud'] == 1]
if len(fraud_rows) > 0:
    fraud_idx = fraud_rows.index[0]
    fraud_pos = list(sample.index).index(fraud_idx)

    prob = model.predict_proba(X_scaled[fraud_pos:fraud_pos+1])[0][1]
    print(f"Predicted fraud probability: {prob:.4f}")

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[fraud_pos],
            base_values=explainer.expected_value,
            data=X_sample[fraud_pos],
            feature_names=FEATURES
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_local.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved shap_local.png")

    print("\n--- Feature Contributions for this Prediction ---")
    contributions = pd.DataFrame({
        'feature': FEATURES,
        'shap_value': shap_values[fraud_pos],
        'feature_value': X_sample[fraud_pos]
    }).sort_values('shap_value', key=abs, ascending=False)

    for _, row in contributions.iterrows():
        direction = "→ FRAUD" if row['shap_value'] > 0 else "→ NOT FRAUD"
        print(f"{row['feature']:35s} = {row['feature_value']:10.2f}   SHAP: {row['shap_value']:+.4f}  {direction}")

print("\n✅ SHAP analysis complete!")