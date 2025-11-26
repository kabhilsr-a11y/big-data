import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt

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
results = {}
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
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    results[name] = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall
    }

# Print comparison table
print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10}")
print("-"*80)
for model_name, metrics in results.items():
    auc_str = f"{metrics['AUC']:<10.4f}" if metrics['AUC'] is not None else "N/A       "
    print(f"{model_name:<20} {metrics['Accuracy']:<10.4f} {metrics['F1 Score']:<10.4f} {auc_str} {metrics['Precision']:<10.4f} {metrics['Recall']:<10.4f}")
print("="*80)

# Find best model for each metric
best_accuracy = max(results.items(), key=lambda x: x[1]['Accuracy'])
best_f1 = max(results.items(), key=lambda x: x[1]['F1 Score'])
# Only consider models with AUC available
models_with_auc = {k: v for k, v in results.items() if v['AUC'] is not None}
best_auc = max(models_with_auc.items(), key=lambda x: x[1]['AUC']) if models_with_auc else None

print("\nBest Models:")
print(f"Accuracy: {best_accuracy[0]} ({best_accuracy[1]['Accuracy']:.4f})")
print(f"F1 Score: {best_f1[0]} ({best_f1[1]['F1 Score']:.4f})")
if best_auc:
    print(f"AUC: {best_auc[0]} ({best_auc[1]['AUC']:.4f})")
else:
    print("AUC: N/A (not available for Linear Regression)")

print("\nModel comparison completed.")

# Visualization
print("\nGenerating visualizations...")

# Prepare data for plotting
model_names = list(results.keys())
accuracies = [results[name]['Accuracy'] for name in model_names]
precisions = [results[name]['Precision'] for name in model_names]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy bar plot
ax1.bar(model_names, accuracies, color='skyblue')
ax1.set_title('Model Accuracy Comparison')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Models')
ax1.set_ylim(0, 1)
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# Precision bar plot
ax2.bar(model_names, precisions, color='lightgreen')
ax2.set_title('Model Precision Comparison')
ax2.set_ylabel('Precision')
ax2.set_xlabel('Models')
ax2.set_ylim(0, 1)
for i, v in enumerate(precisions):
    ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_accuracy_precision_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'model_accuracy_precision_comparison.png'")