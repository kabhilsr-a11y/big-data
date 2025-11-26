import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, when, isnan, count, trim, lower
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import pyspark.sql.functions as F

# First read with pandas to verify data
print("Reading CSV with pandas first...")
pd_df = pd.read_csv(r"C:\Users\selva\OneDrive\Attachments\New folder\archive (2)\crop_yield.csv")
print("\nPandas DataFrame Info:")
print(pd_df.info())

# Create Spark session with explicit local mode configuration
print("\nInitializing Spark...")
spark = SparkSession.builder \
    .appName("Crop Yield Data Preprocessing") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

# Load data
print("Starting data load...")
# Use absolute path to CSV file
csv_path = r"C:\Users\selva\OneDrive\Attachments\New folder\archive (2)\crop_yield.csv"
if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found at: {csv_path}")
    print("Please ensure the file exists at this location.")
    spark.stop()
    exit(1)

print(f"Found CSV file at: {csv_path}")
df = spark.read.csv(csv_path, header=True, inferSchema=True)
print("Data loaded successfully.")

# Display initial data info
print("Initial Data Schema:")
df.printSchema()
print(f"Initial Row Count: {df.count()}")

# Step 1: Handle Missing Values
print("Step 1: Handling missing values...")
# Check for missing values
# use col(c) inside isnan() to pass a Column object (was passing a string previously)
missing_counts = df.select([count(when(isnan(col(c)) | col(c).isNull(), col(c))).alias(c) for c in df.columns])
missing_counts.show()
print("Missing values checked.")

# Fill missing values for numerical columns with mean
# be more permissive with dtype names and normalize to lowercase
num_types = set(['int', 'bigint', 'double', 'float', 'long', 'integer', 'decimal', 'short'])
numerical_cols = [col_name for col_name, dtype in df.dtypes if dtype and dtype.lower() in num_types]
print(f"Numerical columns: {numerical_cols}")
for col_name in numerical_cols:
    mean_val = df.select(mean(col(col_name))).collect()[0][0]
    if mean_val is None:
        print(f"Warning: column {col_name} has no non-null values; filling with 0")
        mean_val = 0
    print(f"Filling missing values in {col_name} with mean: {mean_val}")
    df = df.na.fill(mean_val, [col_name])

# Fill missing values for categorical columns with mode or 'Unknown'
categorical_cols = [col_name for col_name, dtype in df.dtypes if dtype and dtype.lower() == 'string']
print(f"Categorical columns: {categorical_cols}")
for col_name in categorical_cols:
    try:
        mode_row = df.groupBy(col_name).count().orderBy('count', ascending=False).first()
        mode_val = mode_row[0] if mode_row else 'Unknown'
    except Exception:
        mode_val = 'Unknown'
    print(f"Filling missing values in {col_name} with mode: {mode_val}")
    df = df.na.fill(mode_val, [col_name])
print("Missing values handled.")

# Step 2: Remove Duplicates
print("Step 2: Removing duplicates...")
initial_count = df.count()
df = df.dropDuplicates()
final_count = df.count()
print(f"Removed {initial_count - final_count} duplicates. Row count now: {final_count}")

# Step 3: Data Cleaning
print("Step 3: Data cleaning...")
# Trim whitespace and convert to lowercase for string columns
for col_name in categorical_cols:
    df = df.withColumn(col_name, trim(lower(col(col_name))))
    print(f"Trimmed and lowercased {col_name}")

# Remove rows with invalid data (example: negative yields)
if 'yield' in df.columns:
    before_filter = df.count()
    df = df.filter(col('yield') >= 0)
    after_filter = df.count()
    print(f"Removed {before_filter - after_filter} rows with negative yield. Row count now: {after_filter}")
elif 'Yield_tons_per_hectare' in df.columns:
    before_filter = df.count()
    df = df.filter(col('Yield_tons_per_hectare') >= 0)
    after_filter = df.count()
    print(f"Removed {before_filter - after_filter} rows with negative yield. Row count now: {after_filter}")
else:
    print("No yield column found for filtering negative values.")
print("Data cleaning completed.")

# Step 4: Outlier Detection and Handling (using IQR method for numerical columns)
print("Step 4: Outlier detection and handling...")
for col_name in numerical_cols:
    print(f"Processing outliers for {col_name}")
    quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
    if quantiles and len(quantiles) == 2:
        Q1, Q3 = quantiles
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before_outlier = df.count()
        df = df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))
        after_outlier = df.count()
        print(f"Removed {before_outlier - after_outlier} outliers from {col_name}. Row count now: {after_outlier}")
    else:
        print(f"Could not calculate quantiles for {col_name}")
print("Outlier handling completed.")

# Step 5: Feature Engineering
print("Step 5: Feature engineering...")
# Example: Create new features
if 'area' in df.columns and 'yield' in df.columns:
    df = df.withColumn('yield_per_area', col('yield') / col('area'))
    print("Created yield_per_area feature")
elif 'area' in df.columns and 'Yield_tons_per_hectare' in df.columns:
    df = df.withColumn('yield_per_area', col('Yield_tons_per_hectare') / col('area'))
    print("Created yield_per_area feature using Yield_tons_per_hectare")
else:
    print("No area and yield columns found for feature engineering")
print("Feature engineering completed.")

# Step 6: Encoding Categorical Variables
print("Step 6: Encoding categorical variables...")
# Identify categorical columns for encoding
categorical_cols = [col_name for col_name, dtype in df.dtypes if dtype == 'string']
print(f"Categorical columns for encoding: {categorical_cols}")

# String Indexing
indexers = [StringIndexer(inputCol=col_name, outputCol=col_name+"_index", handleInvalid="keep") for col_name in categorical_cols]
print(f"Created {len(indexers)} string indexers")

# One-Hot Encoding
encoders = [OneHotEncoder(inputCol=col_name+"_index", outputCol=col_name+"_encoded") for col_name in categorical_cols]
print(f"Created {len(encoders)} one-hot encoders")
print("Categorical encoding setup completed.")

# Step 7: Feature Scaling for Numerical Columns
print("Step 7: Feature scaling...")
# Assemble numerical features
assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features")
print(f"Created assembler with input cols: {numerical_cols}")
scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_features")
print("Created standard scaler")
print("Feature scaling setup completed.")

# Step 8: Create Pipeline
print("Step 8: Creating and fitting pipeline...")
pipeline_stages = indexers + encoders + [assembler, scaler]
print(f"Pipeline stages: {len(pipeline_stages)}")
pipeline = Pipeline(stages=pipeline_stages)
print("Pipeline created.")

# Fit and transform the data
print("Fitting pipeline...")
model = pipeline.fit(df)
print("Pipeline fitted. Transforming data...")
df_processed = model.transform(df)
print("Data transformed.")

# Select final features
print("Selecting final features...")
encoded_cols = [col_name+"_encoded" for col_name in categorical_cols]
final_cols = encoded_cols + ["scaled_features"]
print(f"Encoded columns: {encoded_cols}")

# If there's a target column (e.g., 'yield'), keep it separate
target_col = 'yield' if 'yield' in df.columns else ('Yield_tons_per_hectare' if 'Yield_tons_per_hectare' in df.columns else None)
if target_col:
    final_cols.append(target_col)
    print(f"Target column: {target_col}")
else:
    print("No target column found")

df_final = df_processed.select(final_cols)
print("Final features selected.")

# Display processed data info
print("Processed Data Schema:")
df_final.printSchema()
print(f"Processed Row Count: {df_final.count()}")

# Save processed data
print("Saving processed data...")
df_final.write.mode("overwrite").parquet("processed_crop_yield_data.parquet")
print("Data saved to 'processed_crop_yield_data.parquet'")

# Stop Spark session
print("Stopping Spark session...")
spark.stop()
print("Spark session stopped.")

print("Data preprocessing completed and saved to 'processed_crop_yield_data.parquet'")