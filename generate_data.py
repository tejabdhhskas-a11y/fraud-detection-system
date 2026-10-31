import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

random.seed(42)
np.random.seed(42)
fake = Faker('en_IN')

NUM_TRANSACTIONS = 1_000_000
NUM_USERS = 10_000
TARGET_FRAUD_RATE = 0.008
END_DATE = datetime(2025, 4, 1)

MERCHANTS = ['Amazon', 'Flipkart', 'Swiggy', 'Zomato', 'Big Bazaar', 'Reliance Digital', 
             'Apple Store', 'Netflix', 'Spotify', 'Uber', 'Ola', 'IRCTC', 'Makemytrip', 
             'Pharmacy', 'Hospital', 'School Fees', 'Rent Payment', 'Salary Credit', 'Local Grocery']
CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad', 'Surat', 'Jaipur']
DEVICES = ['Mobile', 'Desktop', 'Tablet']
PAYMENT_METHODS = ['UPI', 'Credit Card', 'Debit Card', 'Net Banking']

user_profiles = {}
for i in range(1, NUM_USERS + 1):
    user_profiles[f"user_{i:05d}"] = {
        'home_city': random.choice(CITIES),
        'preferred_device': random.choices(DEVICES, weights=[0.7, 0.2, 0.1])[0],
        'is_hni': random.random() < 0.05
    }

print("Generating transactions...")
transactions = []
user_ids_list = list(user_profiles.keys()) 

for i in range(NUM_TRANSACTIONS):
    user_id = random.choice(user_ids_list)
    profile = user_profiles[user_id]

    # Resample instead of clipping (prevents identical timestamps)
    random_days = np.random.exponential(scale=30)
    while random_days > 90:
        random_days = np.random.exponential(scale=30)
    transaction_time = END_DATE - timedelta(days=random_days)

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

    transactions.append({
        'transaction_id': f"txn_{i:07d}", 
        'user_id': user_id,
        'merchant': random.choice(MERCHANTS),
        'amount': amount,
        'transaction_time': transaction_time,
        'location': location,
        'device_type': device,
        'payment_method': random.choice(PAYMENT_METHODS),
        'is_international': 1 if is_legit_international else (1 if random.random() < 0.05 else 0),
        'is_fraud': 0,
        'is_velocity': False 
    })

df = pd.DataFrame(transactions)
NUM_FRAUD = int(NUM_TRANSACTIONS * TARGET_FRAUD_RATE)
fraud_pool = df.sample(frac=1, random_state=42).index.tolist()[:NUM_FRAUD]

sizes = [int(NUM_FRAUD * 0.2)] * 5
p1, p2, p3, p4, p5 = (fraud_pool[0:sizes[0]], 
                      fraud_pool[sizes[0]:sizes[0]+sizes[1]], 
                      fraud_pool[sizes[0]+sizes[1]:sizes[0]+sizes[1]+sizes[2]],
                      fraud_pool[sizes[0]+sizes[1]+sizes[2]:sizes[0]+sizes[1]+sizes[2]+sizes[3]],
                      fraud_pool[sizes[0]+sizes[1]+sizes[2]+sizes[3]:])

# --- FIX: Split the multi-column assignment into 3 separate lines ---
df.loc[p1, 'is_fraud'] = 1
df.loc[p1, 'amount'] = np.random.uniform(50000, 150000, len(p1))
df.loc[p1, 'is_international'] = 1

for idx in p2: 
    df.loc[idx, 'location'] = random.choice([c for c in CITIES if c != user_profiles[df.loc[idx, 'user_id']]['home_city']])
df.loc[p2, 'is_fraud'] = 1

df.loc[p3, 'is_fraud'] = 1
df.loc[p3, 'is_velocity'] = True

df.loc[p4, 'is_fraud'] = 1

for idx in p5: 
    df.loc[idx, 'device_type'] = random.choice([d for d in DEVICES if d != user_profiles[df.loc[idx, 'user_id']]['preferred_device']])
df.loc[p5, 'is_fraud'] = 1

# FIX 1 & 3: Apply temporal shifts and sort
df['transaction_time'] = pd.to_datetime(df['transaction_time'])
df.loc[p4, 'transaction_time'] = df.loc[p4, 'transaction_time'].apply(lambda x: x.replace(hour=random.randint(2, 4)))

df = df.sort_values(['user_id', 'transaction_time']).reset_index(drop=True)
g = df.groupby('user_id')
prev_t = g['transaction_time'].shift()

m = df['is_velocity'] & prev_t.notna()
df.loc[m, 'transaction_time'] = prev_t[m] + pd.to_timedelta(np.random.randint(1, 6, m.sum()), unit='m')

df = df.sort_values(['user_id', 'transaction_time']).reset_index(drop=True)
g = df.groupby('user_id')

df['time_since_last_transaction'] = g['transaction_time'].diff().dt.total_seconds().div(60).fillna(43200).clip(upper=43200).astype(int)
df['previous_transaction_amount'] = g['amount'].shift().fillna(0)
df['transaction_hour'] = df['transaction_time'].dt.hour 

home_cities = pd.DataFrame([{'user_id': uid, 'home_city': p['home_city'], 'preferred_device': p['preferred_device']} for uid, p in user_profiles.items()])
df = df.merge(home_cities, on='user_id', how='left')
df['is_home_city'] = (df['location'] == df['home_city']).astype(int)
df['is_preferred_device'] = (df['device_type'] == df['preferred_device']).astype(int)

df.drop(columns=['home_city', 'preferred_device', 'is_velocity'], inplace=True)
df.to_csv('transactions.csv', index=False)
print("Dataset generated successfully!")