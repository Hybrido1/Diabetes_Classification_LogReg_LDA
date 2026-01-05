import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Config
st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")


# Load ML Artifact (Model + Scaler + Threshold + Feature Order)
@st.cache_resource
def load_artifact(pkl_path: str = "diabetes_final_model.pkl") -> dict:
    with open(pkl_path, "rb") as f:
        artifact = pickle.load(f)

    required = {"model", "scaler", "threshold", "feature_names"}
    missing = required - set(artifact.keys())
    if missing:
        raise ValueError(f"PKL artifact is missing keys: {missing}")

    # Basic validation
    if not isinstance(artifact["feature_names"], (list, tuple)) or len(artifact["feature_names"]) == 0:
        raise ValueError("feature_names in the artifact must be a non-empty list/tuple.")

    return artifact


try:
    artifact = load_artifact()
    model = artifact["model"]
    scaler = artifact["scaler"]
    threshold = float(artifact["threshold"])
    feature_names = list(artifact["feature_names"])
except Exception as e:
    st.error("❌ Unable to load `diabetes_final_model.pkl`.")
    st.write("Make sure `diabetes_final_model.pkl` is present in the same folder as `app.py`.")
    st.exception(e)
    st.stop()



# UI Header

st.title("Diabetes Risk Prediction")
st.caption("Enter patient details below. The app uses the trained ML model to estimate diabetes risk.")

with st.expander("Model Details"):
    st.write(
        f"""
- **Model**: Logistic Regression (class_weight='balanced')
- **Decision Threshold**: {threshold:.2f}  
- **Output**:
  - `Outcome = 1` → Diabetic
  - `Outcome = 0` → Non-diabetic
        """
    )


# Input Form (Auto from feature_names)
st.subheader("Patient Inputs")

# Provide sensible defaults / ranges for common Pima dataset features.
# If your dataset has different features, it will still work (defaults to general numeric input).
default_config = {
    "Pregnancies": {"min": 0.0, "max": 20.0, "value": 2.0, "step": 1.0, "format": "%.0f"},
    "Glucose": {"min": 1.0, "max": 250.0, "value": 120.0, "step": 1.0, "format": "%.0f"},
    "BloodPressure": {"min": 1.0, "max": 200.0, "value": 72.0, "step": 1.0, "format": "%.0f"},
    "SkinThickness": {"min": 1.0, "max": 100.0, "value": 29.0, "step": 1.0, "format": "%.0f"},
    "Insulin": {"min": 1.0, "max": 900.0, "value": 125.0, "step": 1.0, "format": "%.0f"},
    "BMI": {"min": 1.0, "max": 80.0, "value": 32.3, "step": 0.1, "format": "%.1f"},
    "DiabetesPedigreeFunction": {"min": 0.0, "max": 3.0, "value": 0.47, "step": 0.01, "format": "%.2f"},
    "Age": {"min": 1.0, "max": 120.0, "value": 33.0, "step": 1.0, "format": "%.0f"},
}

# Create a form so the page doesn't re-run on every input change
with st.form("patient_form"):
    user_inputs = {}

    # Use 2 columns for a more professional layout
    col1, col2 = st.columns(2)

    for i, feat in enumerate(feature_names):
        cfg = default_config.get(
            feat,
            {"min": -1e9, "max": 1e9, "value": 0.0, "step": 1.0, "format": "%.4f"},
        )

        target_col = col1 if i % 2 == 0 else col2

        with target_col:
            user_inputs[feat] = st.number_input(
                label=f"{feat}",
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(cfg["value"]),
                step=float(cfg["step"]),
                format=cfg["format"],
            )

    submitted = st.form_submit_button("🔮 Predict")



# Prediction + Professional Output Message
if submitted:
    # Build input row exactly in model feature order
    X_input = pd.DataFrame([[user_inputs[f] for f in feature_names]], columns=feature_names)

    # Scale input
    X_scaled = scaler.transform(X_input)

    # Predict probability for class 1 (diabetes)
    prob_diabetes = float(model.predict_proba(X_scaled)[:, 1][0])

    # Apply threshold
    pred_class = 1 if prob_diabetes >= threshold else 0

    st.markdown("---")
    st.subheader("Prediction Result")

    if pred_class == 1:
        st.error(" **Result: High likelihood of Diabetes (Outcome = 1)**")
        st.write(
            "Based on the input values, the model indicates a **higher risk of diabetes**. "
            "This result is intended for **screening support** and should be followed by clinical confirmation."
        )
    else:
        st.success("**Result: Low likelihood of Diabetes (Outcome = 0)**")
        st.write(
            "Based on the input values, the model indicates a **lower risk of diabetes**. "
            "This result is intended for **screening support** and should not replace medical advice."
        )

    st.info(f"**Predicted Probability of Diabetes:** `{prob_diabetes:.3f}`  |  **Decision Threshold:** `{threshold:.2f}`")

    with st.expander("View Entered Inputs"):
        st.dataframe(X_input)

    st.caption("Disclaimer: This tool is for educational/demo purposes only and is not medical advice.")
