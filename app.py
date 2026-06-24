import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Set page config
st.set_page_config(page_title="Airline Price Prediction Dashboard", layout="wide")

# Title
st.title("✈️ Airline Ticket Price Prediction Dashboard")
st.markdown("Predict flight prices and visualize trends!")

# Load dataset
@st.cache_data
def load_data():
    # Replace with your dataset path
    data = pd.read_csv("flight_price_data.csv")  # Update this path
    return data

data = load_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
options = st.sidebar.selectbox("Choose an option", ["Dataset Overview", "Price Prediction with Plot"])

# Display raw data
if options == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("Here’s a glimpse of the raw flight price dataset:")
    st.dataframe(data.head())
    st.write(f"Dataset Shape: {data.shape}")
    st.write("Columns:", list(data.columns))

# Price Prediction with Plot
elif options == "Price Prediction with Plot":
    st.header("Price Prediction Model")
    st.write("Predict flight ticket prices based on your inputs!")

    # Preprocessing
    df = data.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Airline', 'Source', 'Destination']
    encoders = {}  # Store encoders for prediction
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col].astype(str))

    # Features and target
    X = df[['Airline', 'Source', 'Destination', 'Month', 'Year']]  # Use only specified columns
    y = df["Price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # User input for prediction
    st.subheader("Predict Your Ticket Price")
    input_data = {}

    # Select Airline
    input_data['Airline'] = st.selectbox("Select Airline", data['Airline'].unique())

    # Select Source
    input_data['Source'] = st.selectbox("Select Source", data['Source'].unique())

    # Select Destination
    input_data['Destination'] = st.selectbox("Select Destination", data['Destination'].unique())

    # Enter Month
    input_data['Month'] = st.number_input("Enter Month", min_value=1, max_value=12, value=3)

    # Enter Year
    input_data['Year'] = st.number_input("Enter Year", min_value=2019, max_value=2025, value=2019)

    # Predict button
    if st.button("Predict Price"):
        # Prepare input for prediction
        input_df = pd.DataFrame([input_data])
        for col in categorical_cols:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))  # Use fitted encoders
        prediction = model.predict(input_df)
        st.success(f"Predicted Ticket Price: ₹{prediction[0]:.2f}")

        # Plotly visualization
        st.subheader("Price Visualization")
        # Create a dataframe for plotting (predicted price vs inputs)
        plot_data = pd.DataFrame({
            "Category": ["Predicted Price"],
            "Price": [prediction[0]]
        })
        fig = px.bar(plot_data, x="Category", y="Price", title="Predicted Price",
                     labels={"Price": "Price (₹)"}, color="Category")
        st.plotly_chart(fig)

        # Optional: Scatter plot of actual vs predicted prices from test set
        test_results = pd.DataFrame({
            "Actual Price": y_test,
            "Predicted Price": y_pred
        })
        fig2 = px.scatter(test_results, x="Actual Price", y="Predicted Price",
                          title="Actual vs Predicted Prices",
                          labels={"Actual Price": "Actual Price (₹)", "Predicted Price": "Predicted Price (₹)"},
                          trendline="ols")
        st.plotly_chart(fig2)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ by k7")
st.sidebar.write(f"Date: March 23, 2025")