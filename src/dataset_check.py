import pandas as pd

from settings import RAW_FAKE, RAW_TRUE

# Load the datasets (run from repo root: python src/dataset_check.py)
true_data = pd.read_csv(RAW_TRUE)
fake_data = pd.read_csv(RAW_FAKE)


# ==============================
# TRUE NEWS DATASET
# ==============================

print("========== TRUE NEWS DATASET ==========")

print("\nFirst 5 rows:")
print(true_data.head())

print("\nDataset shape:")
print(true_data.shape)

print("\nColumn names:")
print(true_data.columns.tolist())

print("\nData information:")
true_data.info()

print("\nMissing values:")
print(true_data.isnull().sum())

print("\nDuplicate rows:")
print(true_data.duplicated().sum())


# ==============================
# FAKE NEWS DATASET
# ==============================

print("\n========== FAKE NEWS DATASET ==========")

print("\nFirst 5 rows:")
print(fake_data.head())

print("\nDataset shape:")
print(fake_data.shape)

print("\nColumn names:")
print(fake_data.columns.tolist())

print("\nData information:")
fake_data.info()

print("\nMissing values:")
print(fake_data.isnull().sum())

print("\nDuplicate rows:")
print(fake_data.duplicated().sum())