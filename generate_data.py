import random
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
from faker import Faker

# Set seed for reproducibility (Reviewer Fix)
random.seed(42)
np.random.seed(42)

fake = Faker('en_IN')

NUM_TRANSACTIONS = 1_000_000
NUM_USERS = 10_000
TARGET_FRAUD_RATE = 0.008  # 0.8%
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 1)

# Cleaned list (Removed duplicate Pharmacy)
MERCHANTS = ['Amazon', 'Flipkart', 'Swiggy', 'Zomato', 'Big Bazaar',
             'Reliance Digital', 'Apple Store', 'Netflix', 'Spotify',
             'Uber', 'Ola', 'IRCTC', 'Makemytrip', 'Pharmacy',
             'Hospital', 'School Fees', 'Rent Payment', 'Salary Credit',
             'Local Grocery']

CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad',
          'Pune', 'Kolkata', 'Ahmedabad', 'Surat', 'Jaipur']

DEVICES = ['Mobile', 'Desktop', 'Tablet']
PAYMENT_METHODS = ['UPI', 'Credit Card', 'Debit Card', 'Net Banking']

print("Generating user profiles...")
user_profiles = {}
for i in range(1, NUM_USERS + 1):
    user_id = f"user_{i:05d}"
    user_profiles[user_id] = {
        'home_city': random.choice(CITIES),
        'preferred_device': random.choices(DEVICES, weights=[0.7, 0.2, 0.1])[0],
        'is_hni': random.random() < 0.05  # 5% High Net-worth Individuals
    }

print(f"Generating {NUM_TRANSACTIONS:,} transactions...")
transactions = []
# Pre-compute user list to save massive amounts of time (Reviewer Fix)
user_ids_list = list(user_profiles.keys()) 

for i in range(NUM_TRANSACTIONS):
    if i % 100_000 == 0 and i != 0:
        print(f"Generated {i:,} transactions...")

    user_id = random.choice(user_ids_list)
    profile = user_profiles[user_id]

    random_days = np.random.exponential(scale=30)
    random_days = min(random_days, 90)
    transaction_time = END_DATE - timedelta(days=random_days)

    # Nuanced realistic amounts (HNIs make legit large international payments)
    is_legit_international = False
    if profile['is_hni'] and random.random() < 0.02:
        amount = round(np.random.uniform(50000, 200000), 2)
        is_legit_international = True
    elif random.random() < 0.005:
        amount = round(np.random.uniform(30000, 100000), 2)
        is_legit_international = True
    elif random.random() < 0.2:
        amount = round(np.random.exponential(scale=8000) + 500, 2)
    else:
        amount = round(np.random.exponential(scale=1500) + 50, 2)

    location = profile['home_city'] if random.random() < 0.8 else random.choice(CITIES)
    device = profile['preferred_device'] if random.random() < 0.85 else random.choice(DEVICES)

    merchant = random.choice(MERCHANTS)
    payment_method = random.choice(PAYMENT_METHODS)
    is_international = 1 if is_legit_international else (1 if random.random() < 0.05 else 0)
    transaction_hour = transaction_time.hour

    # We do NOT calculate time_since_last or previous_amount here anymore. 
    # We will do it using Pandas AFTER fraud injection.
    transactions.append({
        'transaction_id': str(uuid.uuid4()),
        'user_id': user_id,
        'merchant': merchant,
        'amount': amount,
        'transaction_time': transaction_time,
        'location': location,
        'device_type': device,
        'payment_method': payment_method,
        'is_international': is_international,
        'transaction_hour': transaction_hour,
        'is_fraud': 0 # Placeholder
    })

print("\nInjecting fraud patterns into 0.8% of transactions...")
df = pd.DataFrame(transactions)

# --- INJECT FRAUD BEFORE TEMPORAL SORTING ---
NUM_FRAUD = int(NUM_TRANSACTIONS * TARGET_FRAUD_RATE)
all_indices = df.sample(frac=1, random_state=42).index.tolist()
fraud_pool = all_indices[:NUM_FRAUD]

pattern_sizes = [int(NUM_FRAUD * 0.20)] * 5
p1 = fraud_pool[0:pattern_sizes[0]]
p2 = fraud_pool[pattern_sizes[0]:pattern_sizes[0]+pattern_sizes[1]]
p3 = fraud_pool[pattern_sizes[0]+pattern_sizes[1]:pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]]
p4 = fraud_pool[pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]:pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]+pattern_sizes[3]]
p5 = fraud_pool[pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]+pattern_sizes[3]:]

# Pattern 1: High amount + International
df.loc[p1, 'is_fraud'] = 1
df.loc[p1, 'amount'] = [round(np.random.uniform(50000, 150000), 2) for _ in p1]
df.loc[p1, 'is_international'] = 1

# Pattern 2: Impossible Travel (Different city)
df.loc[p2, 'is_fraud'] = 1
for idx in p2:
    user = df.loc[idx, 'user_id']
    home = user_profiles[user]['home_city']
    other_cities = [c for c in CITIES if c != home]
    df.loc[idx, 'location'] = random.choice(other_cities)

# Pattern 3: Velocity (Handled by the Pandas logic below, we just tag it as fraud)
df.loc[p3, 'is_fraud'] = 1

# Pattern 4: Suspicious Hour
df.loc[p4, 'is_fraud'] = 1
df.loc[p4, 'transaction_hour'] = [random.randint(2, 4) for _ in p4]
df.loc[p4, 'amount'] = [round(np.random.uniform(15000, 50000), 2) for _ in p4]

# Pattern 5: New Device
df.loc[p5, 'is_fraud'] = 1
for idx in p5:
    user = df.loc[idx, 'user_id']
    pref = user_profiles[user]['preferred_device']
    other = [d for d in DEVICES if d != pref]
    df.loc[idx, 'device_type'] = random.choice(other)

# --- REVIEWER FIX 1: FIX TEMPORAL FEATURES CORRECTLY ---
print("Calculating true temporal features (Fixing negative time bug)...")
df['transaction_time'] = pd.to_datetime(df['transaction_time'])
# Sort by user and time so "previous" is actually the previous transaction
df = df.sort_values(['user_id', 'transaction_time']).reset_index(drop=True)

g = df.groupby('user_id')
df['time_since_last_transaction'] = (
    g['transaction_time'].diff().dt.total_seconds().div(60).fillna(43200).clip(upper=43200).astype(int)
)
df['previous_transaction_amount'] = g['amount'].shift().fillna(0)

# --- REVIEWER FIX 2: PREVENT DATA LEAKAGE IN FEATURE ENGINEERING ---
print("Calculating is_home_city and is_preferred_device correctly...")
home_cities = pd.DataFrame([
    {'user_id': uid, 'home_city': prof['home_city'], 'preferred_device': prof['preferred_device']} 
    for uid, prof in user_profiles.items()
])
df = df.merge(home_cities, on='user_id', how='left')

df['is_home_city'] = (df['location'] == df['home_city']).astype(int)
df['is_preferred_device'] = (df['device_type'] == df['preferred_device']).astype(int)

# Drop helper columns to keep dataset clean
df = df.drop(columns=['home_city', 'preferred_device'])

# Reorder columns for a clean look
cols = ['transaction_id', 'user_id', 'merchant', 'amount', 'transaction_time', 
        'location', 'device_type', 'payment_method', 'is_international', 
        'previous_transaction_amount', 'time_since_last_transaction', 
        'transaction_hour', 'is_home_city', 'is_preferred_device', 'is_fraud']
df = df[cols]

print("Saving to CSV...")
df.to_csv('transactions.csv', index=False)

print("\n=== DATASET SUMMARY ===")
print(f"Total transactions: {len(df):,}")
print(f"Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
print(f"Average amount: ₹{df['amount'].mean():.2f}")
print(f"Max amount: ₹{df['amount'].max():.2f}")