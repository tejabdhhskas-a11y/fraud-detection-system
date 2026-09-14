import random
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
from faker import Faker

fake = Faker('en_IN')

NUM_TRANSACTIONS = 1_000_000
NUM_USERS = 10_000
TARGET_FRAUD_RATE = 0.008  # 0.8%
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 1)

MERCHANTS = ['Amazon', 'Flipkart', 'Swiggy', 'Zomato', 'Big Bazaar',
             'Reliance Digital', 'Apple Store', 'Netflix', 'Spotify',
             'Uber', 'Ola', 'IRCTC', 'Makemytrip', 'Pharmacy',
             'Hospital', 'School Fees', 'Rent Payment', 'Salary Credit',
             'Local Grocery', 'Pharmacy']

CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad',
          'Pune', 'Kolkata', 'Ahmedabad', 'Surat', 'Jaipur']

DEVICES = ['Mobile', 'Desktop', 'Tablet']
PAYMENT_METHODS = ['UPI', 'Credit Card', 'Debit Card', 'Net Banking']

# =====================================================
# STEP 1: Generate user profiles
# =====================================================
print("Generating user profiles...")
user_profiles = {}
for i in range(1, NUM_USERS + 1):
    user_id = f"user_{i:05d}"
    user_profiles[user_id] = {
        'home_city': random.choice(CITIES),
        'preferred_device': random.choices(DEVICES, weights=[0.7, 0.2, 0.1])[0],
    }

# =====================================================
# STEP 2: Generate NORMAL transactions (no fraud yet)
# =====================================================
print(f"Generating {NUM_TRANSACTIONS:,} transactions...")
user_last_transaction = {user_id: None for user_id in user_profiles.keys()}
user_last_amount = {user_id: 0 for user_id in user_profiles.keys()}
transactions = []

for i in range(NUM_TRANSACTIONS):
    if i % 100_000 == 0 and i != 0:
        print(f"Generated {i:,} transactions...")

    user_id = random.choice(list(user_profiles.keys()))
    profile = user_profiles[user_id]

    random_days = np.random.exponential(scale=30)
    random_days = min(random_days, 90)
    transaction_time = END_DATE - timedelta(days=random_days)

    if random.random() < 0.2:
        amount = round(np.random.exponential(scale=8000) + 500, 2)
    else:
        amount = round(np.random.exponential(scale=1500) + 50, 2)

    location = profile['home_city'] if random.random() < 0.8 else random.choice(CITIES)
    device = profile['preferred_device'] if random.random() < 0.85 else random.choice(DEVICES)

    merchant = random.choice(MERCHANTS)
    payment_method = random.choice(PAYMENT_METHODS)
    is_international = 1 if random.random() < 0.05 else 0
    transaction_hour = transaction_time.hour

    last_txn = user_last_transaction[user_id]
    if last_txn:
        time_since_last = int((transaction_time - last_txn).total_seconds() / 60)
    else:
        time_since_last = 99999

    prev_amount = user_last_amount[user_id]

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
        'previous_transaction_amount': prev_amount,
        'time_since_last_transaction': time_since_last,
        'transaction_hour': transaction_hour,
        'is_fraud': 0
    })

    user_last_transaction[user_id] = transaction_time
    user_last_amount[user_id] = amount

# =====================================================
# STEP 3: Inject fraud into a controlled subset
# =====================================================
print("\nInjecting fraud patterns into 0.8% of transactions...")
df = pd.DataFrame(transactions)

# How many fraud cases we want
NUM_FRAUD = int(NUM_TRANSACTIONS * TARGET_FRAUD_RATE)  # 8,000

# Shuffle all indices, then split fraud across 5 patterns
all_indices = df.sample(frac=1, random_state=42).index.tolist()

fraud_pool = all_indices[:NUM_FRAUD]
remaining = all_indices[NUM_FRAUD:]

# Distribute across 5 patterns
pattern_sizes = [
    int(NUM_FRAUD * 0.20),  # Pattern 1: High amount + international
    int(NUM_FRAUD * 0.20),  # Pattern 2: Impossible travel
    int(NUM_FRAUD * 0.20),  # Pattern 3: Velocity
    int(NUM_FRAUD * 0.20),  # Pattern 4: Suspicious hour
    int(NUM_FRAUD * 0.20),  # Pattern 5: New device
]

p1_indices = fraud_pool[0:pattern_sizes[0]]
p2_indices = fraud_pool[pattern_sizes[0]:pattern_sizes[0]+pattern_sizes[1]]
p3_indices = fraud_pool[pattern_sizes[0]+pattern_sizes[1]:pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]]
p4_indices = fraud_pool[pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]:pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]+pattern_sizes[3]]
p5_indices = fraud_pool[pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]+pattern_sizes[3]:]

# --- Pattern 1: High amount + International ---
df.loc[p1_indices, 'is_fraud'] = 1
df.loc[p1_indices, 'amount'] = [round(np.random.uniform(50000, 100000), 2) for _ in p1_indices]
df.loc[p1_indices, 'is_international'] = 1

# --- Pattern 2: Impossible Travel (different city, tiny time gap) ---
df.loc[p2_indices, 'is_fraud'] = 1
for idx in p2_indices:
    user = df.loc[idx, 'user_id']
    home = user_profiles[user]['home_city']
    # Set location to a different city
    other_cities = [c for c in CITIES if c != home]
    df.loc[idx, 'location'] = random.choice(other_cities)
    # Set time since last to < 15 minutes
    df.loc[idx, 'time_since_last_transaction'] = random.randint(1, 14)

# --- Pattern 3: Velocity (many transactions in very short time) ---
df.loc[p3_indices, 'is_fraud'] = 1
df.loc[p3_indices, 'time_since_last_transaction'] = [random.randint(1, 5) for _ in p3_indices]

# --- Pattern 4: Suspicious hour ---
df.loc[p4_indices, 'is_fraud'] = 1
df.loc[p4_indices, 'transaction_hour'] = [random.randint(2, 4) for _ in p4_indices]
df.loc[p4_indices, 'amount'] = [round(np.random.uniform(15000, 50000), 2) for _ in p4_indices]

# --- Pattern 5: New device + high amount ---
df.loc[p5_indices, 'is_fraud'] = 1
df.loc[p5_indices, 'amount'] = [round(np.random.uniform(10000, 40000), 2) for _ in p5_indices]
for idx in p5_indices:
    user = df.loc[idx, 'user_id']
    pref = user_profiles[user]['preferred_device']
    other_devices = [d for d in DEVICES if d != pref]
    df.loc[idx, 'device_type'] = random.choice(other_devices)

# =====================================================
# STEP 4: Save
# =====================================================
print("Saving to CSV...")
df.to_csv('transactions.csv', index=False)

print("\n=== DATASET SUMMARY ===")
print(f"Total transactions: {len(df):,}")
print(f"Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
print(f"Average amount: ₹{df['amount'].mean():.2f}")
print(f"Max amount: ₹{df['amount'].max():.2f}")
print("\nSample fraud transaction:")
print(df[df['is_fraud'] == 1].head(1))
print("\nSample normal transaction:")
print(df[df['is_fraud'] == 0].head(1))