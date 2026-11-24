import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "optimized_rf_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoders.pkl"


# ---------------------------------------------------------
# Load the existing model and label encoders
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(ENCODER_PATH, "rb") as encoder_file:
        label_encoders = pickle.load(encoder_file)

    return model, label_encoders


st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.35rem;
            font-weight: 750;
            margin-bottom: 0.15rem;
        }
        .subtitle {
            color: #9CA3AF;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 18px;
            border-radius: 14px;
        }
        .result-caption {
            color: #9CA3AF;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🍔 Food Delivery Time Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Predict estimated food delivery time using your trained Random Forest model.</div>',
    unsafe_allow_html=True,
)

try:
    model, label_encoders = load_artifacts()
except FileNotFoundError as error:
    st.error(
        f"Could not find {error.filename or 'a required model file'}. "
        "Keep main.py, optimized_rf_model.pkl and label_encoders.pkl "
        "in the same folder."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the model or encoders: {error}")
    st.info(
        "Pickle files should be loaded with compatible Python and "
        "scikit-learn versions. If this error persists, check the versions "
        "used when the model was trained."
    )
    st.stop()


# Use the exact feature order saved in the trained model.
expected_features = list(getattr(model, "feature_names_in_", []))
fallback_features = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]
feature_order = expected_features if expected_features else fallback_features

required_features = set(fallback_features)
if set(feature_order) != required_features or len(feature_order) != 7:
    st.error(
        "The model's expected input features do not match this app. "
        f"Model expects: {feature_order}"
    )
    st.stop()

required_encoders = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
missing_encoders = [name for name in required_encoders if name not in label_encoders]
if missing_encoders:
    st.error(f"Missing label encoders: {', '.join(missing_encoders)}")
    st.stop()


# ---------------------------------------------------------
# Sidebar: model information
# ---------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "This application uses a pre-trained Random Forest Regressor "
        "to estimate food delivery time."
    )
    st.divider()
    st.subheader("Model inputs")
    st.write("7 features")
    st.write("4 categorical features")
    st.write("3 numerical features")
    st.caption("Order_ID is intentionally excluded.")
    st.divider()
    st.caption("Model: optimized_rf_model.pkl")
    st.caption("Encoders: label_encoders.pkl")


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.subheader("Enter delivery details")
st.write("Fill in the information below and click **Predict delivery time**.")

with st.form("delivery_prediction_form"):
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### 📍 Delivery information")

        distance = st.number_input(
            "Delivery distance (km)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Distance between the restaurant and delivery destination.",
        )

        weather_options = list(label_encoders["Weather"].classes_)
        weather = st.selectbox("Weather condition", weather_options)

        traffic_options = list(label_encoders["Traffic_Level"].classes_)
        traffic = st.selectbox("Traffic level", traffic_options)

        time_options = list(label_encoders["Time_of_Day"].classes_)
        time_of_day = st.selectbox("Time of day", time_options)

    with right:
        st.markdown("#### 🛵 Order and courier information")

        vehicle_options = list(label_encoders["Vehicle_Type"].classes_)
        vehicle = st.selectbox("Vehicle type", vehicle_options)

        preparation_time = st.number_input(
            "Food preparation time (minutes)",
            min_value=1,
            max_value=180,
            value=15,
            step=1,
            help="Time required by the restaurant to prepare the order.",
        )

        courier_experience = st.number_input(
            "Courier experience (years)",
            min_value=0.0,
            max_value=50.0,
            value=2.0,
            step=0.5,
        )

    submitted = st.form_submit_button(
        "🚀 Predict delivery time",
        use_container_width=True,
        type="primary",
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    try:
        # Apply the exact encoders used during model training.
        input_values = {
            "Distance_km": float(distance),
            "Weather": int(label_encoders["Weather"].transform([weather])[0]),
            "Traffic_Level": int(
                label_encoders["Traffic_Level"].transform([traffic])[0]
            ),
            "Time_of_Day": int(
                label_encoders["Time_of_Day"].transform([time_of_day])[0]
            ),
            "Vehicle_Type": int(
                label_encoders["Vehicle_Type"].transform([vehicle])[0]
            ),
            "Preparation_Time_min": int(preparation_time),
            "Courier_Experience_yrs": float(courier_experience),
        }

        # Preserve the feature order used when the model was trained.
        input_df = pd.DataFrame(
            [[input_values[column] for column in feature_order]],
            columns=feature_order,
        )

        prediction = float(model.predict(input_df)[0])

        st.divider()
        st.subheader("Prediction result")

        result_col, details_col = st.columns([1, 1.4], gap="large")

        with result_col:
            st.metric(
                label="Estimated delivery time",
                value=f"{prediction:.1f} min",
            )
            st.markdown(
                '<div class="result-caption">Predicted by your trained Random Forest model.</div>',
                unsafe_allow_html=True,
            )

        with details_col:
            st.markdown("#### Delivery summary")
            summary = pd.DataFrame(
                {
                    "Input feature": [
                        "Distance",
                        "Weather",
                        "Traffic level",
                        "Time of day",
                        "Vehicle",
                        "Preparation time",
                        "Courier experience",
                    ],
                    "Selected value": [
                        f"{distance:.1f} km",
                        weather,
                        traffic,
                        time_of_day,
                        vehicle,
                        f"{preparation_time} min",
                        f"{courier_experience:g} years",
                    ],
                }
            )
            st.dataframe(summary, hide_index=True, use_container_width=True)

        st.success(
            f"The model estimates that this order will be delivered in "
            f"approximately {prediction:.0f} minutes."
        )
        st.caption(
            "This is a model estimate, not a guaranteed delivery time. "
            "Actual delivery time can vary."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the pickle files belong to this model and that "
            "the label encoders were saved from the same training workflow."
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption("Food Delivery Time Prediction • Streamlit • Random Forest Regression")
