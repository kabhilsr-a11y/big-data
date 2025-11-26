import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib

# Load processed data
print("Loading processed data...")
df = pd.read_csv("processed_crop_yield_data_pandas.csv")
print(f"Data loaded. Shape: {df.shape}")

# Convert yield to categorical for classification
target_col = 'Yield_tons_per_hectare'
if target_col in df.columns:
    # Create yield categories (Low, Medium, High)
    y_continuous = df[target_col]
    y = pd.cut(y_continuous, bins=3, labels=['Low', 'Medium', 'High'])
    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X = df.drop(columns=[target_col])
    print(f"Target column: {target_col} (converted to categories)")
    print(f"Yield categories: {y.value_counts()}")
    print(f"Encoded classes: {label_encoder.classes_}")
    print(f"Features shape: {X.shape}")
else:
    print("Target column not found.")
    exit(1)

# Split data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train Linear Regression model
print("\nTraining Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("Model trained.")

# Make predictions
print("\nMaking predictions...")
y_pred_continuous = lr_model.predict(X_test)

# Convert continuous predictions to class labels for evaluation
y_pred = np.round(y_pred_continuous).astype(int)
y_pred = np.clip(y_pred, 0, 2)  # Ensure predictions are within valid class range

# Evaluate model
print("\nEvaluating model...")
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Note: AUC metric not available for Linear Regression (only for probabilistic classifiers)")

# Feature coefficients (for the first class vs rest)
print("\nFeature coefficients (top 10 by absolute value, class 0 vs rest):")
coefficients = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0]
}).assign(abs_coefficient=lambda x: x['coefficient'].abs()).sort_values('abs_coefficient', ascending=False)
print(coefficients.head(10))

# Save model and label encoder
print("\nSaving model...")
joblib.dump(lr_model, 'linear_regression_crop_yield.pkl')
joblib.dump(label_encoder, 'label_encoder_crop_yield.pkl')
print("Model saved to 'linear_regression_crop_yield.pkl'")
print("Label encoder saved to 'label_encoder_crop_yield.pkl'")

print("Linear Regression modeling completed.")