import os
import pandas as pd
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Define dataset paths
DATASET_PATHS = {
    "General": "C:/Users/vchan/OneDrive/Desktop/Medical-Diagnosis-System/datasets/general.csv",
    "Diabetes": "C:/Users/vchan/OneDrive/Desktop/Medical-Diagnosis-System/datasets/diabetes.csv",
    "Heart attack": "C:/Users/vchan/OneDrive/Desktop/Medical-Diagnosis-System/datasets/heart.csv",
    "Parkinsons": "C:/Users/vchan/OneDrive/Desktop/Medical-Diagnosis-System/datasets/parkinsons.csv"
}

# Model storage path
MODEL_PATH = "C:/Users/vchan/OneDrive/Desktop/Medical-Diagnosis-System/models/"
os.makedirs(MODEL_PATH, exist_ok=True)

# Load datasets function
def load_data():
    """Loads all datasets from predefined file paths."""
    datasets = {}
    for disease, path in DATASET_PATHS.items():
        try:
            datasets[disease] = pd.read_csv(path)
            print(f"📂 Loaded dataset for {disease}, shape: {datasets[disease].shape}")
        except Exception as e:
            print(f"❌ Failed to load {disease} dataset: {e}")
    return datasets

# Train models function
def train_models():
    datasets = load_data()
    models = {}

    for disease, df in datasets.items():
        print(f"\n🚀 Training {disease} Model...")

        if df.empty:
            print(f"⚠️ Skipping {disease} - Dataset is empty!")
            continue

        # Select features (X) and target variable (y)
        if disease == "Parkinsons":
            if "status" in df.columns:
                y = df["status"]
                X = df.drop(columns=["status", "name"], errors="ignore")
            else:
                print(f"❌ No 'status' column found in Parkinsons dataset! Skipping...")
                continue
        else:
            X = df.iloc[:, :-1].copy()
            y = df.iloc[:, -1]

        # Fix missing values
        y = y.dropna()
        X = X.loc[y.index]

        # Encode categorical features
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le

        # Encode target variable
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model with class balancing
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save model
        model_filename = os.path.join(MODEL_PATH, f"{disease}_model.pkl")
        joblib.dump((model, label_encoder, label_encoders, list(X.columns)), model_filename)

        print(f"✅ Model for {disease} saved successfully!")
        print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

    return models

# Load trained models
def load_models():
    models = {}
    for disease in DATASET_PATHS.keys():
        model_path = os.path.join(MODEL_PATH, f"{disease}_model.pkl")
        try:
            model_data = joblib.load(model_path)
            if len(model_data) == 4:
                models[disease] = model_data
            else:
                return train_models()
        except FileNotFoundError:
            return train_models()
    return models

# Streamlit UI
def main():
    st.markdown("<h1 style='text-align: center; color: #003366;'>🩺 Advanced Medical Diagnosis System</h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #00509e;'>Select symptoms to predict possible diseases.</h3>",
                unsafe_allow_html=True)

    models = load_models()
    datasets = load_data()

    # Add the disease type selection dropdown here
    disease_type = st.selectbox("Select Disease Type", ['Heart attack', 'Diabetes', 'Parkinsons', 'General'])

    # Ensure the selected disease type is in the datasets dictionary
    if disease_type not in datasets:
        st.error(f"❌ No dataset available for {disease_type}. Please try again.")
        return

    df = datasets[disease_type]

    # Define symptom columns based on disease type (must match the dataset's columns)
    symptom_columns = {
        'Heart attack': {
            'Chest Pain': 'cp', 'Shortness of Breath': 'exang', 'High Blood Pressure': 'trestbps',
            'Irregular Heartbeat': 'thal', 'Fatigue': 'oldpeak', 'Swelling in Legs': 'ca',
            'Dizziness': 'slope', 'Nausea': 'thalach', 'Cold Sweats': 'chol'
        },
        'Diabetes': {
            'Increased Thirst': 'polyuria', 'Frequent Urination': 'polydipsia',
            'Extreme Hunger': 'sudden weight loss', 'Unexplained Weight Loss': 'weakness',
            'Fatigue': 'fatigue', 'Blurred Vision': 'blurred vision',
            'Slow Healing Wounds': 'delayed healing', 'Tingling in Hands/Feet': 'partial paresis'
        },
        'Parkinsons': {
            'Tremor': 'tremor', 'Slowness of Movement': 'bradykinesia',
            'Muscle Rigidity': 'rigidity', 'Balance Problems': 'postural instability',
            'Small Handwriting': 'micrographia', 'Loss of Smell': 'anosmia',
            'Soft/Slurred Speech': 'dysarthria', 'Stooped Posture': 'stooped posture'
        },
        'General': {col: col for col in df.columns[:-1]}
    }

    # Select symptoms for the chosen disease
    symptoms = list(symptom_columns[disease_type].keys())
    selected_symptoms = [symptom for symptom in symptoms if st.checkbox(symptom)]

    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            # Debugging: Print the selected symptoms
            print(f"Selected Symptoms: {selected_symptoms}")

            model, label_encoder, label_encoders, feature_names = models[disease_type]

            # Prepare the input data (set all features to 0 first)
            input_data = pd.DataFrame([{col: 0 for col in feature_names}])

            # Set the corresponding feature value to 1 for the selected symptoms
            for symptom in selected_symptoms:
                col_name = symptom_columns[disease_type].get(symptom, None)
                if col_name in feature_names:
                    input_data[col_name] = 1  # Set the corresponding symptom feature to 1

            # Ensure proper alignment of features in the input data
            input_data = input_data[feature_names]  # Reorder input data to match feature columns

            # Debugging: Print the input data
            print(f"Input Data:\n{input_data}")
            print(f"Feature Names: {feature_names}")

            # Make the prediction
            prediction_encoded = model.predict(input_data)[0]

            # Debugging: Print the prediction
            print(f"Prediction Encoded: {prediction_encoded}")

            # Decode the prediction
            if disease_type == "General":
                predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]
            else:
                predicted_label = "No " + disease_type if prediction_encoded == 0 else disease_type

            # Display the prediction result
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px; color: #003366;">
                    🎯 <b>Predicted Disease:</b> {predicted_label}
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
