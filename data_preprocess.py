import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
print("Loading data...")
csv_path = r"C:\Users\selva\OneDrive\Attachments\New folder\archive (2)\crop_yield.csv"
if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found at: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Data loaded. Shape: {df.shape}")
print("Data info:")
print(df.info())

# Step 1: Handle Missing Values
print("\nStep 1: Handling missing values...")
# Numerical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Impute numerical with mean
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Impute categorical with most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("Missing values handled.")

# Step 2: Remove Duplicates
print("\nStep 2: Removing duplicates...")
initial_count = len(df)
df = df.drop_duplicates()
final_count = len(df)
print(f"Removed {initial_count - final_count} duplicates. Row count now: {final_count}")

# Step 3: Data Cleaning
print("\nStep 3: Data cleaning...")
# Trim whitespace and lowercase for categorical
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Remove invalid data (negative yields)
yield_cols = [col for col in df.columns if 'yield' in col.lower()]
if yield_cols:
    for col in yield_cols:
        before = len(df)
        df = df[df[col] >= 0]
        after = len(df)
        print(f"Removed {before - after} rows with negative {col}")
else:
    print("No yield column found for filtering.")

print("Data cleaning completed.")

# Step 4: Outlier Detection and Handling (IQR for numerical)
print("\nStep 4: Outlier detection...")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    after = len(df)
    print(f"Removed {before - after} outliers from {col}")

print("Outlier handling completed.")

# Step 5: Feature Engineering
print("\nStep 5: Feature engineering...")
# Example: yield per area if columns exist
area_cols = [col for col in df.columns if 'area' in col.lower()]
if area_cols and yield_cols:
    df['yield_per_area'] = df[yield_cols[0]] / df[area_cols[0]]
    print("Created yield_per_area feature")
else:
    print("No area/yield columns for feature engineering")

print("Feature engineering completed.")

# Step 6: Encoding and Scaling
print("\nStep 6: Encoding and scaling...")
# Define target if exists
target_col = yield_cols[0] if yield_cols else None
if target_col:
    y = df[target_col]
    X = df.drop(columns=[target_col])
    # Update num_cols and cat_cols to exclude target
    num_cols = [col for col in num_cols if col != target_col]
    cat_cols = [col for col in cat_cols if col != target_col]
else:
    X = df
    y = None

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# Fit and transform
X_processed = preprocessor.fit_transform(X)

# Convert back to DataFrame
encoded_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_cols = num_cols + list(encoded_cat_cols)
df_processed = pd.DataFrame(X_processed, columns=all_cols)

if y is not None:
    df_processed[target_col] = y.values

print("Encoding and scaling completed.")

# Save processed data
print("\nSaving processed data...")
df_processed.to_csv("processed_crop_yield_data_pandas.csv", index=False)
print("Data saved to 'processed_crop_yield_data_pandas.csv'")

print("Data preprocessing completed.")
