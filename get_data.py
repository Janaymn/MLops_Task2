import pandas as pd
import os

# URL for the Penguins dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv'
df = pd.read_csv(url).dropna()  # drop any missing values (very few)

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Save the clean data
df.to_csv('data/penguins_clean.csv', index=False)