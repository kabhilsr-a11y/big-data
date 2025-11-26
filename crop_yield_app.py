import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# Define feature columns
NUM_COLS = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
CAT_COLS = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']

# Define categories for one-hot encoding
CATEGORIES = {
    'Region': ['east', 'north', 'south', 'west'],
    'Soil_Type': ['chalky', 'clay', 'loam', 'peaty', 'sandy', 'silt'],
    'Crop': ['barley', 'cotton', 'maize', 'rice', 'soybean', 'wheat'],
    'Weather_Condition': ['cloudy', 'rainy', 'sunny']
}

# Load the trained classification models
@st.cache_resource
def load_classification_models():
    models = {}
    model_files = {
        'Random Forest': 'random_forest_crop_yield.pkl',
        'Linear Regression': 'linear_regression_crop_yield.pkl',
        'XGBoost': 'xgboost_crop_yield.pkl',
        'Naive Bayes': 'naive_bayes_crop_yield.pkl'
    }

    for name, file in model_files.items():
        try:
            if os.path.exists(file):
                models[name] = joblib.load(file)
        except Exception as e:
            st.warning(f"Could not load {name} model: {str(e)}")

    return models

# Load the trained regression models
@st.cache_resource
def load_regression_models():
    models = {}
    model_files = {
        'Linear Regression': 'linear_regression_yield.pkl',
        'Random Forest Regressor': 'random_forest_regressor_yield.pkl'
    }

    for name, file in model_files.items():
        try:
            if os.path.exists(file):
                models[name] = joblib.load(file)
        except Exception as e:
            st.warning(f"Could not load {name} model: {str(e)}")

    return models

# Load label encoder
@st.cache_resource
def load_label_encoder():
    try:
        if os.path.exists('label_encoder_crop_yield.pkl'):
            return joblib.load('label_encoder_crop_yield.pkl')
    except Exception:
        return None
    return None

# Create preprocessor
@st.cache_resource
def get_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(categories=[CATEGORIES[col] for col in CAT_COLS], handle_unknown='ignore'), CAT_COLS)
        ])
    return preprocessor

# Function to preprocess input features
def preprocess_features(features_df):
    preprocessor = get_preprocessor()
    # Fit on dummy data with varied values to avoid zero variance
    dummy_data = pd.DataFrame({
        'Rainfall_mm': [1000, 2000], 'Temperature_Celsius': [20, 30], 'Days_to_Harvest': [100, 150],
        'Region': ['east', 'west'], 'Soil_Type': ['clay', 'sandy'], 'Crop': ['wheat', 'rice'], 'Weather_Condition': ['sunny', 'rainy']
    })
    preprocessor.fit(dummy_data)
    return preprocessor.transform(features_df)

# Function to make classification predictions
def predict_yield_category(model, features_df):
    features_processed = preprocess_features(features_df)
    prediction = model.predict(features_processed)
    probabilities = model.predict_proba(features_processed)
    return prediction, probabilities

# Function to make regression predictions
def predict_yield_continuous(model, features_df):
    features_processed = preprocess_features(features_df)
    prediction = model.predict(features_processed)
    return prediction

# Function to plot metrics
def plot_model_metrics(metrics_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return fig

# Main app
def main():
    st.title("🌾 Crop Yield Prediction System")

    # Load label encoder
    label_encoder = load_label_encoder()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict Yield Category", "Predict Continuous Yield"])

    if page == "Predict Yield Category":
        st.header("Predict Crop Yield Category")

        # Load models
        models = load_classification_models()

        if not models:
            st.error("No classification models found. Please ensure model files are present in the application directory.")
            return

        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0)
                temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=1.0)
                days_to_harvest = st.number_input("Days to Harvest", min_value=1, max_value=365, value=120)

            with col2:
                region = st.selectbox("Region", CATEGORIES['Region'])
                soil_type = st.selectbox("Soil Type", CATEGORIES['Soil_Type'])
                crop = st.selectbox("Crop", CATEGORIES['Crop'])
                weather_condition = st.selectbox("Weather Condition", CATEGORIES['Weather_Condition'])

            model_choice = st.selectbox("Select Model", list(models.keys()))

            submitted = st.form_submit_button("Predict Yield Category")

            if submitted:
                # Prepare input features
                features = pd.DataFrame({
                    'Rainfall_mm': [rainfall],
                    'Temperature_Celsius': [temperature],
                    'Days_to_Harvest': [days_to_harvest],
                    'Region': [region],
                    'Soil_Type': [soil_type],
                    'Crop': [crop],
                    'Weather_Condition': [weather_condition]
                })

                # Make prediction
                model = models[model_choice]
                prediction, probabilities = predict_yield_category(model, features)

                # Handle prediction decoding based on model type
                if isinstance(prediction[0], str):
                    # Model predicts string labels directly (e.g., Random Forest, Naive Bayes)
                    predicted_class = prediction[0]
                    classes = ['Low', 'Medium', 'High']
                elif isinstance(prediction[0], (int, np.integer)):
                    # Model predicts encoded labels (e.g., Linear Regression, XGBoost)
                    if label_encoder:
                        predicted_class = label_encoder.inverse_transform(prediction)[0]
                        classes = label_encoder.classes_
                    else:
                        predicted_class = prediction[0]
                        classes = ['Low', 'Medium', 'High']
                else:
                    predicted_class = prediction[0]
                    classes = ['Low', 'Medium', 'High']

                # Display results
                st.success(f"Predicted Yield Category: **{predicted_class}**")

                # Show prediction probabilities
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Category': classes,
                    'Probability': probabilities[0]
                })
                st.bar_chart(prob_df.set_index('Category'))

                # Show feature importance if available
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    # Get feature names after preprocessing
                    preprocessor = get_preprocessor()
                    dummy_data = pd.DataFrame({
                        'Rainfall_mm': [1000, 2000], 'Temperature_Celsius': [20, 30], 'Days_to_Harvest': [100, 150],
                        'Region': ['east', 'west'], 'Soil_Type': ['clay', 'sandy'], 'Crop': ['wheat', 'rice'], 'Weather_Condition': ['sunny', 'rainy']
                    })
                    preprocessor.fit(dummy_data)
                    feature_names = preprocessor.get_feature_names_out()

                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)

                    st.bar_chart(importance_df.set_index('Feature'))

    elif page == "Predict Continuous Yield":
        st.header("Predict Continuous Crop Yield")

        # Load regression models
        reg_models = load_regression_models()

        if not reg_models:
            st.error("No regression models found. Please ensure model files are present in the application directory.")
            return

        # Input form
        with st.form("regression_form"):
            col1, col2 = st.columns(2)

            with col1:
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0, key="reg_rainfall")
                temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=1.0, key="reg_temp")
                days_to_harvest = st.number_input("Days to Harvest", min_value=1, max_value=365, value=120, key="reg_days")

            with col2:
                region = st.selectbox("Region", CATEGORIES['Region'], key="reg_region")
                soil_type = st.selectbox("Soil Type", CATEGORIES['Soil_Type'], key="reg_soil")
                crop = st.selectbox("Crop", CATEGORIES['Crop'], key="reg_crop")
                weather_condition = st.selectbox("Weather Condition", CATEGORIES['Weather_Condition'], key="reg_weather")

            model_choice = st.selectbox("Select Model", list(reg_models.keys()), key="reg_model")

            submitted = st.form_submit_button("Predict Continuous Yield")

            if submitted:
                # Prepare input features
                features = pd.DataFrame({
                    'Rainfall_mm': [rainfall],
                    'Temperature_Celsius': [temperature],
                    'Days_to_Harvest': [days_to_harvest],
                    'Region': [region],
                    'Soil_Type': [soil_type],
                    'Crop': [crop],
                    'Weather_Condition': [weather_condition]
                })

                # Make prediction
                model = reg_models[model_choice]
                prediction = predict_yield_continuous(model, features)

                # Display results
                st.success(f"Predicted Yield: **{prediction[0]:.2f} tons per hectare**")

                # Show feature importance if available (for Random Forest Regressor)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    # Get feature names after preprocessing
                    preprocessor = get_preprocessor()
                    dummy_data = pd.DataFrame({
                        'Rainfall_mm': [1000, 2000], 'Temperature_Celsius': [20, 30], 'Days_to_Harvest': [100, 150],
                        'Region': ['east', 'west'], 'Soil_Type': ['clay', 'sandy'], 'Crop': ['wheat', 'rice'], 'Weather_Condition': ['sunny', 'rainy']
                    })
                    preprocessor.fit(dummy_data)
                    feature_names = preprocessor.get_feature_names_out()

                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)


if __name__ == "__main__":
    main()