# Importing the libraries
import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

# Importing the functions and tenure mapping from utils
from utils import (
    plot_csat_distibution,
    plot_tenure_distribution,
    plot_channel_name,
    plot_csat_channel,
    plot_category_distribution,
    plot_response_distribution,
    plot_avg_csat_tenure,
    plot_csat_response,
    plot_avg_responsetime_channel,
    plot_csat_category,
    plot_avg_responsetime_category,
    plot_agent_interaction_vol,
    tenure
)

# App configuration
st.set_page_config(
    page_title="Customer Satisfaction Score Prediction",
    layout="wide")

st.title("🏪Shopzilla Customer Satisfaction Score Prediction")

# Loading the dataframes, trained model, scaler and column list
@st.cache_resource
def load_artifacts():
    # Loading the dataframes
    df_eda = pd.read_csv('../unclean_data/df_unclean.csv')
    df_model = pd.read_csv('../clean_data/df_clean.csv')

    # Loading the trained model
    model = load_model(f"../code_files/best_model.keras")

    # Loading the scaler and column list
    scaler = joblib.load('../artifacts/scaler.pkl')
    columns = joblib.load('../artifacts/columns.pkl')

    return df_eda, df_model, model, scaler, columns

df_eda, df_model, model, scaler, columns = load_artifacts()

# Sidebar controls
st.sidebar.header("🔧 Controls")

section = st.sidebar.radio(
    "Choose Section",
    ["EDA Analytics", "CSAT Prediction"])

# EDA Section
if section == "EDA Analytics":

    st.subheader("📊Exploratory Data Analysis")

    eda_option = st.selectbox(
        "Select Analysis",
        [
            'Distribution of CSAT Scores',
            'Tenure Bucket Distribution of Agents',
            'Customer Interaction Channel Distribution',
            'Average CSAT Score by Channel',
            'Distribution of Customer Interaction by Categories',
            'Response Time Distribution',
            'Average CSAT Scores across Tenure Buckets',
            'CSAT Scores vs Response Time',
            'Average Response Time by Channel',
            'Average CSAT Scores by Category',
            'Average Response Time by Category',
            'Interaction Volume by Agent Shift'
        ]
    )

    if eda_option == "Distribution of CSAT Scores":
        st.pyplot(plot_csat_distibution(df_eda))

    elif eda_option == "Tenure Bucket Distribution of Agents":
        st.pyplot(plot_tenure_distribution(df_eda))

    elif eda_option == "Customer Interaction Channel Distribution":
        st.pyplot(plot_channel_name(df_eda))

    elif eda_option == "Average CSAT Score by Channel":
        st.pyplot(plot_csat_channel(df_eda))

    elif eda_option == "Distribution of Customer Interaction by Categories":
        st.pyplot(plot_category_distribution(df_eda))

    elif eda_option == "Response Time Distribution":
        st.pyplot(plot_response_distribution(df_eda))

    elif eda_option == "Average CSAT Scores across Tenure Buckets":
        st.pyplot(plot_avg_csat_tenure(df_eda))

    elif eda_option == "CSAT Scores vs Response Time":
        st.pyplot(plot_csat_response(df_eda))
    
    elif eda_option == "Average Response Time by Channel":
        st.pyplot(plot_avg_responsetime_channel(df_eda))

    elif eda_option == "Average CSAT Scores by Category":
        st.pyplot(plot_csat_category(df_eda))

    elif eda_option == "Average Response Time by Category":
        st.pyplot(plot_avg_responsetime_category(df_eda))

    elif eda_option == "Interaction Volume by Agent Shift":
        st.pyplot(plot_agent_interaction_vol(df_eda))

# CSAT Prediction Section
else:
    st.subheader("Customer Satisfaction Score Prediction💯")

    # User Inputs
    channel = st.selectbox(
        "Channel Name",
        sorted(df_model["channel_name"].unique())
    )

    category = st.selectbox(
        "Category",
        sorted(df_model["category"].unique())
    )

    filtered_subcats = df_model[df_model['category'] == category]['Sub-category'].unique()
    sub_category = st.selectbox(
        "Sub-category",
        sorted(filtered_subcats)
    )

    shift = st.selectbox(
        "Agent Shift",
        sorted(df_model["Agent Shift"].unique())
    )

    tenure_bucket = st.selectbox(
        "Agent Tenure Bucket",
        sorted(tenure.keys())
    )

    response_time = st.number_input(
        "Response Time (minutes)",
        min_value=0.0,
        value=5.0
    )

    # Prediction Button
    if st.button("Predict CSAT Score"):

        # Encoding tenure using the tenure mapping
        tenure_encoded = tenure[tenure_bucket]

        # Creating input dictionary
        input_data = {
            "channel_name": channel,
            "category": category,
            "Sub-category": sub_category,
            "Agent Shift": shift,
            "Tenure_Bucket_enc": tenure_encoded,
            "responsetime_in_mins": response_time
        }

        input_df = pd.DataFrame([input_data])

        # One-hot encoding the categorical columns by creating dummy columns
        categorical_cols = [
            "channel_name",
            "category",
            "Sub-category",
            "Agent Shift"
        ]

        input_df = pd.get_dummies(input_df, columns=categorical_cols)

        # Aligning with the training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Scaling response time
        input_df[["responsetime_in_mins"]] = scaler.transform(input_df[["responsetime_in_mins"]])

        # Predicting the CSAT scores
        pred_prob = model.predict(input_df)
        pred_class = pred_prob.argmax(axis=1)[0] + 1

        st.success(f"Predicted CSAT Score: {pred_class}")