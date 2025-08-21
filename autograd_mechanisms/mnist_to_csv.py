# convert_parquet_to_csv.py
import pandas as pd
import numpy as np

# Read parquet
df = pd.read_parquet('mnist.parquet')

# Save as CSV
df.to_csv('mnist.csv', index=False)
