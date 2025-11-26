import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# Load processed data
print("Loading processed data...")
df = pd.read_csv("processed_crop_yield_data_pandas.csv")
print(f"Data loaded. Shape: {df.shape}")

# Prepare data (same as in individual models)
target_col = 'Yield_tons_per_hectare'
y_continuous = df[target_col]
y = pd.cut(y_continuous, bins=3, labels=['Low', 'Medium', 'High'])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X = df.drop(columns=[target_col])
# Split data (same split as models)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Load models
print("\nLoading models...")
models = {
    'Random Forest': joblib.load('random_forest_crop_yield.pkl'),
    'Naive Bayes': joblib.load('naive_bayes_crop_yield.pkl'),
    'XGBoost': joblib.load('xgboost_crop_yield.pkl'),
    'Linear Regression': joblib.load('linear_regression_crop_yield.pkl')
}

# Evaluate each model
results = []
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)

    # Convert predictions to integers if they are strings (for Random Forest)
    if isinstance(y_pred[0], str):
        y_pred = label_encoder.transform(y_pred)
    # Convert continuous predictions to class labels for Linear Regression
    elif name == 'Linear Regression':
        y_pred = np.round(y_pred).astype(int)
        y_pred = np.clip(y_pred, 0, 2)  # Ensure predictions are within valid class range

    # For AUC, need predict_proba (not available for Linear Regression)
    if hasattr(model, 'predict_proba') and name != 'Linear Regression':
        y_pred_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    else:
        auc = None  # AUC not available for Linear Regression

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Append to results list for CSV
    results.append({'Model': name, 'Metric': 'Accuracy', 'Value': accuracy})
    results.append({'Model': name, 'Metric': 'F1 Score', 'Value': f1})
    if auc is not None:
        results.append({'Model': name, 'Metric': 'AUC', 'Value': auc})

# Create DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_metrics_for_powerbi.csv', index=False)
print("\nModel metrics exported to 'model_metrics_for_powerbi.csv' for Power BI visualization.")
print("CSV format: Model, Metric, Value")
print("This format allows easy creation of bar charts, line charts, and comparisons in Power BI.")
