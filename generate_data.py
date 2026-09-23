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

print("Generating user profiles...")
user_profiles = {}
for i in range(1, NUM_USERS + 1):
    user_id = f"user_{i:05d}"
    user_profiles[user_id] = {
        'home_city': random.choice(CITIES),
        'preferred_device': random.choices(DEVICES, weights=[0.7, 0.2, 0.1])[0],
        'is_hnI': random.random() < 0.05  # 5% are High Net-worth Individuals
    }

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

    # --- NUANCE #1: Legitimate high-value international transactions ---
    # 2% of HNIs and 0.5% of regular users make legitimate international payments
    is_legit_international = False
    if profile['is_hnI'] and random.random() < 0.02:
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

    last_txn = user_last_transaction[user_id]
    if last_txn:
        time_since_last = int((transaction_time - last_txn).total_seconds() / 60)
        # --- FIX: Prevent negative time deltas ---
        if time_since_last < 0 or time_since_last > 43200:
            time_since_last = 43200  # Cap at 30 days
    else:
        time_since_last = 43200  # Cap at 30 days

    prev_amount = user_last_amount[user_id]

    is_fraud = 0
    is_home_city = 1 if location == profile['home_city'] else 0
    is_preferred_device = 1 if device == profile['preferred_device'] else 0

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
        'is_home_city': is_home_city,
        'is_preferred_device': is_preferred_device,
        'is_fraud': is_fraud
    })

    user_last_transaction[user_id] = transaction_time
    user_last_amount[user_id] = amount

print("\nInjecting fraud patterns into 0.8% of transactions...")
df = pd.DataFrame(transactions)

NUM_FRAUD = int(NUM_TRANSACTIONS * TARGET_FRAUD_RATE)

all_indices = df.sample(frac=1, random_state=42).index.tolist()
fraud_pool = all_indices[:NUM_FRAUD]

pattern_sizes = [
    int(NUM_FRAUD * 0.20),
    int(NUM_FRAUD * 0.20),
    int(NUM_FRAUD * 0.20),
    int(NUM_FRAUD * 0.20),
    int(NUM_FRAUD * 0.20),
]

p1 = fraud_pool[0:pattern_sizes[0]]
p2 = fraud_pool[pattern_sizes[0]:pattern_sizes[0]+pattern_sizes[1]]
p3 = fraud_pool[pattern_sizes[0]+pattern_sizes[1]:pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]]
p4 = fraud_pool[pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]:pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]+pattern_sizes[3]]
p5 = fraud_pool[pattern_sizes[0]+pattern_sizes[1]+pattern_sizes[2]+pattern_sizes[3]:]

# Pattern 1: High amount + International
df.loc[p1, 'is_fraud'] = 1
df.loc[p1, 'amount'] = [round(np.random.uniform(50000, 150000), 2) for _ in p1]
df.loc[p1, 'is_international'] = 1

# Pattern 2: Impossible Travel
df.loc[p2, 'is_fraud'] = 1
for idx in p2:
    user = df.loc[idx, 'user_id']
    home = user_profiles[user]['home_city']
    other_cities = [c for c in CITIES if c != home]
    df.loc[idx, 'location'] = random.choice(other_cities)
    df.loc[idx, 'time_since_last_transaction'] = random.randint(1, 14)
    df.loc[idx, 'is_home_city'] = 0

# Pattern 3: Velocity
df.loc[p3, 'is_fraud'] = 1
df.loc[p3, 'time_since_last_transaction'] = [random.randint(1, 5) for _ in p3]

# Pattern 4: Suspicious Hour
df.loc[p4, 'is_fraud'] = 1
df.loc[p4, 'transaction_hour'] = [random.randint(2, 4) for _ in p4]
df.loc[p4, 'amount'] = [round(np.random.uniform(15000, 50000), 2) for _ in p4]

# Pattern 5: New Device (with some small amounts - NUANCE #2)
df.loc[p5, 'is_fraud'] = 1
for idx in p5:
    user = df.loc[idx, 'user_id']
    pref = user_profiles[user]['preferred_device']
    other = [d for d in DEVICES if d != pref]
    df.loc[idx, 'device_type'] = random.choice(other)
    df.loc[idx, 'is_preferred_device'] = 0
# Some are small "test" transactions
for idx in random.sample(list(p5), k=int(len(p5) * 0.3)):
    df.loc[idx, 'amount'] = round(np.random.uniform(100, 1000), 2)

print("Saving to CSV...")
df.to_csv('transactions.csv', index=False)

print("\n=== DATASET SUMMARY ===")
print(f"Total transactions: {len(df):,}")
print(f"Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
print(f"Average amount: ₹{df['amount'].mean():.2f}")
print(f"Max amount: ₹{df['amount'].max():.2f}")