import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# Load data and model
df = pd.read_csv("df_to_viz.csv")
model = joblib.load("rf_model_base.joblib")  # Replace with your actual model file path

df = df[df["ADR"] != 5400]


# Function to grab column names
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Function for EDA summaries
def target_summary_with_cat(dataframe, target, categorical_col):
    return pd.DataFrame({
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
        "Count": dataframe[categorical_col].value_counts(),
        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)
    })

# Create tabs
st.set_page_config(layout="wide")
tab1, tab2, tab3 = st.tabs(["Home", "EDA", "Prediction"])

# --- Home Tab ---
with tab1:
    st.title("Hotel Booking Cancellation Examiner App")
    st.subheader("Welcome to the Hotel Booking Prediction Tool!")
    st.write("Content for the home tab will go here. You can describe your app's purpose, provide instructions, or add visuals.")

# --- EDA Tab ---
with tab2:
    st.title("Exploratory Data Analysis")

    # Dropdown to select a categorical column
    selected_cat_col = st.selectbox("Select a Categorical Column for Analysis", options=cat_cols)

    if selected_cat_col:
        # Display target summary for the selected categorical column
        st.subheader(f"Analysis of {selected_cat_col}")
        cat_summary = target_summary_with_cat(df, "IsCanceled", selected_cat_col)
        st.dataframe(cat_summary)

        # Plot bar charts
        fig_ratio = px.bar(
            cat_summary.reset_index(),
            x="index",
            y="Ratio",
            text="Ratio",
            title=f"Ratio (%) for {selected_cat_col}",
            labels={"index": selected_cat_col, "Ratio": "Ratio (%)"},
        )
        fig_ratio.update_traces(textposition="inside", texttemplate='%{text:.2f} %')
        st.plotly_chart(fig_ratio, use_container_width=True)

        fig_target_mean = px.bar(
            cat_summary.reset_index(),
            x="index",
            y="TARGET_MEAN",
            text="TARGET_MEAN",
            title=f"Target Mean for {selected_cat_col}",
            labels={"index": selected_cat_col, "TARGET_MEAN": "Target Mean"},
        )
        fig_target_mean.update_traces(textposition="inside", texttemplate='%{text:.2f}')
        st.plotly_chart(fig_target_mean, use_container_width=True)

    # Dropdown to select a numerical column for boxplots
    selected_num_col = st.selectbox("Select a Numerical Column for Boxplot", options=num_cols)

    if selected_num_col:
        st.subheader(f"Boxplot of {selected_num_col} by 'IsCanceled'")
        fig_box = px.box(
            df,
            x="IsCanceled",
            y=selected_num_col,
            color="IsCanceled",
            title=f"{selected_num_col} Distribution by IsCanceled",
            labels={"IsCanceled": "Cancellation Status", selected_num_col: selected_num_col},
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Correlation Matrix
    st.subheader("Heatmap for Correlations of Numerical Variables")
    corr_matrix = df[num_cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="Viridis",
        title="Correlation Matrix of Numeric Features",
        labels=dict(color="Correlation"),
    )
    fig_corr.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
        height=700,
        width=1200,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# --- Prediction Tab ---
with tab3:
    st.title("Hotel Booking Cancellation Prediction")

    # Default values for numerical inputs
    default_values = {
        "LeadTime": 50,
        "ArrivalDateWeekNumber": 20,
        "ArrivalDateDayOfMonth": 15,
        "StaysInWeekendNights": 1,
        "StaysInWeekNights": 3,
        "PreviousBookingsNotCanceled": 0,
        "BookingChanges": 0,
        "DaysInWaitingList": 0,
        "ADR": 100,
        "NEW_ADR_by_Adults": 50,
        "NEW_ADR_by_Children": 30,
        "NEW_ADR_by_LeadTime": 2,
        "NEW_Weeknights_by_LeadTime": 1,
    }

    # Categorical variable mappings
    meal_mapping = ["HB", "SC", "Rare", "FB"]
    country_mapping = ["BRA", "DEU", "ESP", "FRA", "GBR", "ITA", "NLD", "PRT", "USA", "Rare"]
    market_segment_mapping = ["Direct", "Groups", "Offline TA/TO", "Online TA", "Rare"]
    distribution_channel_mapping = ["Direct", "TA/TO", "Rare", "Corporate"]

    # Sidebar for inputs
    st.sidebar.header("Input Features")

    # Numerical inputs
    inputs = {}
    for feature, default in default_values.items():
        inputs[feature] = st.sidebar.number_input(f"Input {feature}", value=default)

    # Dropdowns for categorical inputs
    inputs["Meal"] = st.sidebar.selectbox("Meal Plan", meal_mapping)
    inputs["Country"] = st.sidebar.selectbox("Country", country_mapping)
    inputs["MarketSegment"] = st.sidebar.selectbox("Market Segment", market_segment_mapping)
    inputs["DistributionChannel"] = st.sidebar.selectbox("Distribution Channel", distribution_channel_mapping)

    # Prepare data for prediction
    def encode_inputs(inputs):
        encoded = {}
        for key in default_values.keys():
            encoded[key] = [inputs[key]]
        return pd.DataFrame(encoded)

    input_data = encode_inputs(inputs)

    # Align columns with the model
    expected_features = model.feature_names_in_
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    # Prediction button
    if st.sidebar.button("Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.success(f"Prediction: Booking is LIKELY to be canceled.")
        else:
            st.success(f"Prediction: Booking is UNLIKELY to be canceled.")