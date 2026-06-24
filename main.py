import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Set page config
st.set_page_config(page_title="Advanced Airline Price Prediction Dashboard", 
                  layout="wide", 
                  initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0078D7;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: #0078D7;
        margin-top: 1.5rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f8ff;
        border-left: 5px solid #0078D7;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title with styled header
st.markdown("<h1 class='main-header'>✈️ Advanced Airline Ticket Price Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Comprehensive analysis and prediction of flight prices with advanced ML models</p>", unsafe_allow_html=True)

# Load dataset with progress bar
@st.cache_data
def load_data():
    with st.spinner('Loading flight price data...'):
        # Replace with your dataset path
        data = pd.read_csv("flight_price_data.csv")  # Update this path
        
        # Data cleaning
        # Remove outliers (example: remove prices that are beyond 3 standard deviations)
        mean_price = data['Price'].mean()
        std_price = data['Price'].std()
        data = data[(data['Price'] > mean_price - 3*std_price) & 
                   (data['Price'] < mean_price + 3*std_price)]
        
        # Create additional features
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data['Day'] = data['Date'].dt.day
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Convert date columns if they exist in string format
        date_cols = [col for col in data.columns if 'date' in col.lower() and data[col].dtype == 'object']
        for col in date_cols:
            try:
                data[col] = pd.to_datetime(data[col])
            except:
                pass
        
        return data

# Sidebar for navigation
with st.sidebar:
    st.image("https://www.example.com/images/airline_logo.png", width=100)  # Replace with your logo
    st.title("Navigation")
    
    options = st.radio(
        "Choose a section",
        ["Dashboard Overview", "Data Exploration", "Advanced Price Prediction", "Market Analysis", "Seasonal Trends", "About"]
    )
    
    st.sidebar.markdown("---")
    
    # Model selection in sidebar for prediction page
    if options == "Advanced Price Prediction":
        st.subheader("Model Settings")
        model_choice = st.selectbox(
            "Select ML Model",
            ["Random Forest", "Gradient Boosting", "Ensemble (Average)"]
        )
        
        advanced_options = st.expander("Advanced Options")
        with advanced_options:
            n_estimators = st.slider("Number of Estimators", 50, 500, 100, 50)
            max_depth = st.slider("Max Depth", 3, 20, 10)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
            use_hyperparameter_tuning = st.checkbox("Use Hyperparameter Tuning", False)
    
    # Download section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Options")
    
    # Date information
    st.sidebar.markdown("---")
    st.sidebar.info(f"Last updated: March 23, 2025")
    st.sidebar.markdown(" Advanced Version")

# Load the data
try:
    data = load_data()
    
    # Main content based on selected option
    if options == "Dashboard Overview":
        st.markdown("<h2 class='subheader'>Dashboard Overview</h2>", unsafe_allow_html=True)
        
        # Summary metrics in a row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Routes", f"{len(data['Source'].unique()) * len(data['Destination'].unique())}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Average Price", f"₹{data['Price'].mean():.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Price Range", f"₹{data['Price'].min():.2f} - ₹{data['Price'].max():.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Airlines", f"{len(data['Airline'].unique())}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick visualizations
        st.markdown("<h3>Quick Insights</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Average Price by Airline")
            avg_price_by_airline = data.groupby('Airline')['Price'].mean().reset_index()
            fig = px.bar(
                avg_price_by_airline.sort_values('Price'), 
                x='Airline', 
                y='Price',
                title="Average Price by Airline",
                color='Price',
                color_continuous_scale=px.colors.sequential.Bluyl
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Top Routes by Price")
            route_data = data.groupby(['Source', 'Destination'])['Price'].mean().reset_index()
            route_data['Route'] = route_data['Source'] + ' → ' + route_data['Destination']
            route_data = route_data.sort_values('Price', ascending=False).head(10)
            fig = px.bar(
                route_data,
                x='Route',
                y='Price',
                title="Top 10 Most Expensive Routes",
                color='Price',
                color_continuous_scale=px.colors.sequential.Bluyl
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Dataset preview
        with st.expander("Dataset Preview"):
            st.write("Here's a glimpse of the flight price dataset:")
            st.dataframe(data.head())
            st.write(f"Dataset Shape: {data.shape}")
            st.write(f"Columns: {', '.join(data.columns)}")
    
    elif options == "Data Exploration":
        st.markdown("<h2 class='subheader'>Interactive Data Exploration</h2>", unsafe_allow_html=True)
        
        # Data filtering options
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_airlines = st.multiselect(
                "Select Airlines", 
                options=data['Airline'].unique(),
                default=data['Airline'].unique()[:3]
            )
        
        with col2:
            selected_sources = st.multiselect(
                "Select Source Airports",
                options=data['Source'].unique(),
                default=data['Source'].unique()[:2]
            )
        
        with col3:
            selected_destinations = st.multiselect(
                "Select Destination Airports",
                options=data['Destination'].unique(),
                default=data['Destination'].unique()[:2]
            )
        
        # Filter data based on selections
        filtered_data = data[
            (data['Airline'].isin(selected_airlines)) &
            (data['Source'].isin(selected_sources)) &
            (data['Destination'].isin(selected_destinations))
        ]
        
        # Show filter summary
        st.write(f"Showing {len(filtered_data)} flights matching your filters.")
        
        # Interactive plots for exploration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Price Distribution")
            
            chart_type = st.radio("Select Chart Type", ["Histogram", "Box Plot", "Violin Plot"], horizontal=True)
            
            if chart_type == "Histogram":
                fig = px.histogram(
                    filtered_data, 
                    x="Price", 
                    color="Airline", 
                    marginal="box",
                    opacity=0.7,
                    barmode="overlay",
                    title="Price Distribution by Airline"
                )
            elif chart_type == "Box Plot":
                fig = px.box(
                    filtered_data,
                    x="Airline",
                    y="Price",
                    color="Airline",
                    title="Price Distribution by Airline"
                )
            else:  # Violin Plot
                fig = px.violin(
                    filtered_data,
                    x="Airline",
                    y="Price",
                    color="Airline",
                    box=True,
                    points="all",
                    title="Price Distribution by Airline"
                )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Route Price Comparison")
            
            # Create Route column
            filtered_data['Route'] = filtered_data['Source'] + ' → ' + filtered_data['Destination']
            
            # Calculate average price by route and airline
            route_airline_avg = filtered_data.groupby(['Route', 'Airline'])['Price'].mean().reset_index()
            
            # Get top routes by volume
            top_routes = filtered_data['Route'].value_counts().nlargest(8).index.tolist()
            route_airline_avg = route_airline_avg[route_airline_avg['Route'].isin(top_routes)]
            
            fig = px.bar(
                route_airline_avg,
                x="Route",
                y="Price",
                color="Airline",
                barmode="group",
                title="Average Price by Route and Airline"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Correlation heatmap
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Feature Correlation")
        
        # Select only numerical columns
        numerical_cols = filtered_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numerical_cols) > 1:
            correlation = filtered_data[numerical_cols].corr()
            
            # Generate heatmap
            fig = px.imshow(
                correlation,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Correlation Between Numerical Features"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Not enough numerical features for correlation analysis.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data table with sorting and filtering
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Filtered Data Table")
        st.dataframe(filtered_data, height=300)
        
        # Download filtered data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(filtered_data)
        st.download_button(
            "Download Filtered Data as CSV",
            csv,
            "filtered_flight_data.csv",
            "text/csv",
            key='download-csv'
        )
        st.markdown("</div>", unsafe_allow_html=True)
            
    elif options == "Advanced Price Prediction":
        st.markdown("<h2 class='subheader'>Advanced Price Prediction</h2>", unsafe_allow_html=True)
        
        # Data preprocessing for modeling
        df = data.copy()
        
        # Define feature engineering function
        def engineer_features(df):
            """Add engineered features to improve model performance"""
            feature_df = df.copy()
            
            # Route combination (if not already present)
            if 'Route' not in feature_df.columns:
                feature_df['Route'] = feature_df['Source'] + '_' + feature_df['Destination']
            
            # Day features if date is available
            if 'Date' in feature_df.columns and pd.api.types.is_datetime64_dtype(feature_df['Date']):
                feature_df['DayOfWeek'] = feature_df['Date'].dt.dayofweek
                feature_df['IsWeekend'] = feature_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
                feature_df['DayOfMonth'] = feature_df['Date'].dt.day
                feature_df['WeekOfYear'] = feature_df['Date'].dt.isocalendar().week
            
            # Seasonal features
            if 'Month' in feature_df.columns:
                # Add seasonality with sine and cosine transforms
                feature_df['Month_sin'] = np.sin(2 * np.pi * feature_df['Month']/12)
                feature_df['Month_cos'] = np.cos(2 * np.pi * feature_df['Month']/12)
            
            return feature_df
        
        # Apply feature engineering
        df = engineer_features(df)
        
        # Encode categorical variables
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        encoders = {}
        
        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
        
        # Select features
        feature_cols = [col for col in df.columns if col not in ['Price', 'Date']]
        X = df[feature_cols]
        y = df["Price"]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        if not 'model_choice' in locals():
            model_choice = "Random Forest"
            n_estimators = 100
            max_depth = 10
            min_samples_split = 5
            use_hyperparameter_tuning = False
            
        # Show progress to user
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Setting up model...")
        progress_bar.progress(10)
        
        # Model training with hyperparameter tuning if selected
        if use_hyperparameter_tuning:
            status_text.text("Performing hyperparameter tuning...")
            
            if model_choice == "Random Forest":
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [n_estimators],
                    'max_depth': [max_depth, None],
                    'min_samples_split': [min_samples_split],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:  # Gradient Boosting
                model = GradientBoostingRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [n_estimators],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [max_depth],
                    'min_samples_split': [min_samples_split]
                }
            
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            
            progress_bar.progress(70)
            status_text.text(f"Best parameters found: {grid_search.best_params_}")
            
        else:
            status_text.text("Training model...")
            
            if model_choice == "Random Forest":
                best_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
            elif model_choice == "Gradient Boosting":
                best_model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
            else:  # Ensemble
                rf_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                gb_model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                
                # Train both models
                rf_model.fit(X_train_scaled, y_train)
                gb_model.fit(X_train_scaled, y_train)
                
                # Create ensemble prediction function
                def ensemble_predict(X):
                    return (rf_model.predict(X) + gb_model.predict(X)) / 2
                
                best_model = rf_model  # Just for feature importance, use ensemble for predictions
        
        # Train the model if not ensemble or not already trained
        if model_choice != "Ensemble" or not 'ensemble_predict' in locals():
            best_model.fit(X_train_scaled, y_train)
        
        progress_bar.progress(80)
        status_text.text("Evaluating model...")
        
        # Evaluate model
        if model_choice == "Ensemble" and 'ensemble_predict' in locals():
            y_pred = ensemble_predict(X_test_scaled)
        else:
            y_pred = best_model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        progress_bar.progress(100)
        status_text.text("Model ready!")
        
        # Display model metrics
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("MSE", f"{mse:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("RMSE", f"{rmse:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("MAE", f"{mae:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("R² Score", f"{r2:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature importance
        if model_choice != "Ensemble" or not 'ensemble_predict' in locals():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Feature Importance")
            
            # Get feature importances
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    feature_importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Feature Importance ({model_choice})",
                    color='Importance',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Feature importance not available for this model.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Prediction vs Actual plot
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Prediction vs Actual")
        
        # Create a dataframe for plotting
        result_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': y_test - y_pred
        })
        
        # Scatter plot
        fig = px.scatter(
            result_df,
            x='Actual',
            y='Predicted',
            title="Actual vs Predicted Prices",
            color='Error',
            color_continuous_scale='RdBu_r',
            labels={"Actual": "Actual Price (₹)", "Predicted": "Predicted Price (₹)"},
            trendline="ols"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # User input for prediction
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Predict Your Ticket Price")
        
        # Create form for inputs
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Select Airline
                input_airline = st.selectbox("Select Airline", data['Airline'].unique())
                
                # Select Source
                input_source = st.selectbox("Select Source", data['Source'].unique())
                
                # Select Destination
                input_destination = st.selectbox("Select Destination", data['Destination'].unique())
                
            with col2:
                # Enter Month
                input_month = st.number_input("Enter Month", min_value=1, max_value=12, value=3)
                
                # Enter Year
                input_year = st.number_input("Enter Year", min_value=2019, max_value=2025, value=2023)
                
                # Day of Week
                input_day_of_week = st.selectbox("Day of Week", 
                    options=[0, 1, 2, 3, 4, 5, 6], 
                    format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
                )
                
            with col3:
                # Additional features if available
                if 'Duration' in data.columns:
                    input_duration = st.number_input("Flight Duration (hours)", min_value=0.5, max_value=24.0, value=2.0, step=0.5)
                
                if 'Stops' in data.columns:
                    input_stops = st.selectbox("Number of Stops", options=[0, 1, 2, 3], format_func=lambda x: f"{x} {'Stop' if x == 1 else 'Stops'}")
            
            submitted = st.form_submit_button("Predict Price")
        
        if submitted:
            # Prepare input for prediction
            input_data = {
                'Airline': input_airline,
                'Source': input_source,
                'Destination': input_destination,
                'Month': input_month,
                'Year': input_year,
                'DayOfWeek': input_day_of_week,
                'IsWeekend': 1 if input_day_of_week >= 5 else 0,
                'Month_sin': np.sin(2 * np.pi * input_month/12),
                'Month_cos': np.cos(2 * np.pi * input_month/12)
            }
            
            # Add additional features if available in the model
            if 'Duration' in data.columns and 'input_duration' in locals():
                input_data['Duration'] = input_duration
            
            if 'Stops' in data.columns and 'input_stops' in locals():
                input_data['Stops'] = input_stops
            
            # Create Route feature
            input_data['Route'] = input_source + '_' + input_destination
            
            # Create a dataframe from input
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col in categorical_cols:
                if col in input_df.columns:
                    # Handle new categories not seen during training
                    try:
                        input_df[col] = encoders[col].transform(input_df[col].astype(str))
                    except:
                        # Use most frequent category if new value is encountered
                        most_common = encoders[col].transform([encoders[col].classes_[0]])[0]
                        input_df[col] = most_common
            
            # Select only features that were used for training
            input_features = input_df[input_df.columns.intersection(feature_cols)]
            
            # Add missing columns with zeros
            for col in feature_cols:
                if col not in input_features.columns:
                    input_features[col] = 0
            
            # Reorder columns to match training data
            input_features = input_features[feature_cols]
            
            # Scale input features
            input_scaled = scaler.transform(input_features)
            
            # Make prediction
            if model_choice == "Ensemble" and 'ensemble_predict' in locals():
                prediction = ensemble_predict(input_scaled)
            else:
                prediction = best_model.predict(input_scaled)
            
            # Display prediction with animation
            st.balloons()
            st.success(f"Predicted Ticket Price: ₹{prediction[0]:.2f}")
            
            # Confidence interval (using model's prediction variation)
            if hasattr(best_model, 'estimators_'):
                if model_choice == "Random Forest":
                    individual_preds = np.array([estimator.predict(input_scaled)[0] 
                                                for estimator in best_model.estimators_])
                    pred_std = individual_preds.std()
                    lower_bound = prediction[0] - 1.96 * pred_std
                    upper_bound = prediction[0] + 1.96 * pred_std
                    
                    st.write(f"95% Confidence Interval: ₹{lower_bound:.2f} - ₹{upper_bound:.2f}")
            
            # Context for the prediction
            similar_routes = data[(data['Source'] == input_source) & 
                                 (data['Destination'] == input_destination)]
            
            if len(similar_routes) > 0:
                avg_price = similar_routes['Price'].mean()
                min_price = similar_routes['Price'].min()
                max_price = similar_routes['Price'].max()
                
                st.write(f"Average price for this route: ₹{avg_price:.2f}")
                st.write(f"Historical price range: ₹{min_price:.2f} - ₹{max_price:.2f}")
                
                # Is the price a good deal?
                if prediction[0] < avg_price * 0.8:
                    st.info("🔥 This appears to be a good deal compared to average prices!")
                elif prediction[0] > avg_price * 1.2:
                    st.warning("⚠️ This price is higher than the average for this route.")
            
            # Price visualization
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode = "number+gauge",
                value = prediction[0],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Predicted Price (₹)"},
                gauge = {
                    'axis': {'range': [min_price if 'min_price' in locals() else 0, 
                                      max_price if 'max_price' in locals() else prediction[0]*2]},
                    'steps': [
                        {'range': [0, avg_price*0.8 if 'avg_price' in locals() else prediction[0]*0.8], 'color': "green"},
                        {'range': [avg_price*0.8 if 'avg_price' in locals() else prediction[0]*0.8, 
                                  avg_price*1.2 if 'avg_price' in locals() else prediction[0]*1.2], 'color': "yellow"},
                        {'range': [avg_price*1.2 if 'avg_price' in locals() else prediction[0]*1.2, 
                                  max_price if 'max_price' in locals() else prediction[0]*2], 'color': "red"},
                    ],
                    'bar': {'color': "blue"}
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Nearby dates suggestion if Date is a feature
            if 'Date' in data.columns:
                st.subheader("Want to find cheaper flights?")
                st.write("Here are prices for nearby dates:")
                
                # Create date range around the selected date
                today = datetime.now().date()
                base_date = datetime(year=int(input_year), month=int(input_month), day=15).date()
                
                # Only show future dates
                if base_date < today:
                    base_date = today
                
                date_range = pd.date_range(start=base_date, periods=14, freq='D')
                
                # Generate predictions for different dates
                date_predictions = []
                
                for date in date_range:
                    temp_input = input_data.copy()
                    temp_input['Month'] = date.month
                    temp_input['Year'] = date.year
                    temp_input['DayOfWeek'] = date.dayofweek
                    temp_input['IsWeekend'] = 1 if date.dayofweek >= 5 else 0
                    temp_input['DayOfMonth'] = date.day
                    temp_input['Month_sin'] = np.sin(2 * np.pi * date.month/12)
                    temp_input['Month_cos'] = np.cos(2 * np.pi * date.month/12)
                    
                    temp_df = pd.DataFrame([temp_input])
                    
                    # Encode categorical variables
                    for col in categorical_cols:
                        if col in temp_df.columns:
                            try:
                                temp_df[col] = encoders[col].transform(temp_df[col].astype(str))
                            except:
                                most_common = encoders[col].transform([encoders[col].classes_[0]])[0]
                                temp_df[col] = most_common
                    
                    # Select features and scale
                    temp_features = temp_df[temp_df.columns.intersection(feature_cols)]
                    for col in feature_cols:
                        if col not in temp_features.columns:
                            temp_features[col] = 0
                    temp_features = temp_features[feature_cols]
                    temp_scaled = scaler.transform(temp_features)
                    
                    # Make prediction
                    if model_choice == "Ensemble" and 'ensemble_predict' in locals():
                        temp_pred = ensemble_predict(temp_scaled)[0]
                    else:
                        temp_pred = best_model.predict(temp_scaled)[0]
                    
                    date_predictions.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'Day': date.strftime('%a'),
                        'Price': temp_pred,
                        'Savings': prediction[0] - temp_pred
                    })
                
                # Create a dataframe for date predictions
                date_df = pd.DataFrame(date_predictions)
                date_df['Price_Formatted'] = date_df['Price'].apply(lambda x: f"₹{x:.2f}")
                date_df['Savings_Formatted'] = date_df['Savings'].apply(
                    lambda x: f"+₹{x:.2f}" if x < 0 else f"₹{x:.2f}"
                )
                
                # Show as a heatmap calendar
                fig = px.imshow(
                    date_df.pivot_table(
                        values='Price', 
                        index=[date_df['Date'].str[:7]], 
                        columns=date_df['Day'],
                        aggfunc='mean'
                    ),
                    color_continuous_scale='RdYlGn_r',
                    title="Price Forecast by Day"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif options == "Market Analysis":
        st.markdown("<h2 class='subheader'>Airline Market Analysis</h2>", unsafe_allow_html=True)
        
        # Market share analysis
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Airline Market Share Analysis")
        
        # By volume
        airline_volume = data['Airline'].value_counts().reset_index()
        airline_volume.columns = ['Airline', 'Flight Count']
        airline_volume['Market Share (%)'] = (airline_volume['Flight Count'] / airline_volume['Flight Count'].sum() * 100).round(2)
        
        # By revenue (estimated from price)
        airline_revenue = data.groupby('Airline')['Price'].sum().reset_index()
        airline_revenue.columns = ['Airline', 'Total Revenue']
        airline_revenue['Revenue Share (%)'] = (airline_revenue['Total Revenue'] / airline_revenue['Total Revenue'].sum() * 100).round(2)
        
        # Join the data
        market_analysis = pd.merge(airline_volume, airline_revenue, on='Airline')
        
        # Create tabs for different visualization
        tab1, tab2 = st.tabs(["Market Share", "Revenue Analysis"])
        
        with tab1:
            # Market share pie chart
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "domain"}, {"type": "domain"}]],
                subplot_titles=["Flight Volume Share", "Revenue Share"]
            )
            
            fig.add_trace(
                go.Pie(
                    labels=airline_volume['Airline'],
                    values=airline_volume['Flight Count'],
                    textinfo='percent+label',
                    hole=0.4
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Pie(
                    labels=airline_revenue['Airline'],
                    values=airline_revenue['Total Revenue'],
                    textinfo='percent+label',
                    hole=0.4
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Market share table
            st.write("Market Share Details:")
            st.dataframe(market_analysis.sort_values('Market Share (%)', ascending=False))
            
        with tab2:
            # Revenue analysis
            st.write("Average Price by Airline:")
            
            avg_price = data.groupby('Airline')['Price'].mean().reset_index()
            avg_price.columns = ['Airline', 'Average Price']
            revenue_data = pd.merge(airline_volume, avg_price, on='Airline')
            
            fig = px.scatter(
                revenue_data,
                x='Flight Count',
                y='Average Price',
                size='Flight Count',
                color='Airline',
                hover_name='Airline',
                text='Airline',
                log_x=True,
                size_max=60,
                title="Volume vs Price Strategy"
            )
            
            fig.update_traces(textposition='top center')
            fig.update_layout(height=600)
            
            # Add quadrant labels
            avg_count = revenue_data['Flight Count'].median()
            avg_price_val = revenue_data['Average Price'].median()
            
            fig.add_shape(
                type="line", line=dict(dash="dash", width=1),
                x0=avg_count, y0=0, x1=avg_count, y1=revenue_data['Average Price'].max() * 1.1
            )
            
            fig.add_shape(
                type="line", line=dict(dash="dash", width=1),
                x0=0, y0=avg_price_val, x1=revenue_data['Flight Count'].max() * 1.1, y1=avg_price_val
            )
            
            # Add annotations for quadrants
            fig.add_annotation(
                x=avg_count/2, y=avg_price_val*1.5,
                text="Premium (Low Volume, High Price)",
                showarrow=False, font=dict(size=10)
            )
            
            fig.add_annotation(
                x=avg_count*5, y=avg_price_val*1.5,
                text="Market Leaders (High Volume, High Price)",
                showarrow=False, font=dict(size=10)
            )
            
            fig.add_annotation(
                x=avg_count/2, y=avg_price_val*0.5,
                text="Niche Players (Low Volume, Low Price)",
                showarrow=False, font=dict(size=10)
            )
            
            fig.add_annotation(
                x=avg_count*5, y=avg_price_val*0.5,
                text="Budget Carriers (High Volume, Low Price)",
                showarrow=False, font=dict(size=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Route profitability analysis
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Route Analysis")
        
        # Create route column
        data['Route'] = data['Source'] + ' → ' + data['Destination']
        
        # Route popularity
        route_popularity = data['Route'].value_counts().reset_index()
        route_popularity.columns = ['Route', 'Flight Count']
        
        # Average price by route
        route_price = data.groupby('Route')['Price'].mean().reset_index()
        route_price.columns = ['Route', 'Average Price']
        
        # Join data
        route_analysis = pd.merge(route_popularity, route_price, on='Route')
        route_analysis['Estimated Revenue'] = route_analysis['Flight Count'] * route_analysis['Average Price']
        
        # Show top routes by volume
        st.write("Top Routes by Volume:")
        fig = px.bar(
            route_analysis.sort_values('Flight Count', ascending=False).head(10),
            x='Route',
            y='Flight Count',
            title="Most Popular Routes",
            color='Average Price',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top routes by revenue
        st.write("Top Routes by Estimated Revenue:")
        fig = px.bar(
            route_analysis.sort_values('Estimated Revenue', ascending=False).head(10),
            x='Route',
            y='Estimated Revenue',
            title="Most Profitable Routes",
            color='Average Price',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive route analysis
        st.subheader("Airline Competition by Route")
        
        # Select a route
        selected_route = st.selectbox(
            "Select a Route to Analyze",
            options=route_analysis.sort_values('Flight Count', ascending=False)['Route'].tolist()
        )
        
        # Show airlines operating on this route
        route_data = data[data['Route'] == selected_route]
        airline_share = route_data['Airline'].value_counts().reset_index()
        airline_share.columns = ['Airline', 'Flight Count']
        airline_share['Market Share (%)'] = (airline_share['Flight Count'] / airline_share['Flight Count'].sum() * 100).round(2)
        
        airline_price = route_data.groupby('Airline')['Price'].mean().reset_index()
        airline_price.columns = ['Airline', 'Average Price']
        
        # Join data
        route_airline_analysis = pd.merge(airline_share, airline_price, on='Airline')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                route_airline_analysis,
                values='Flight Count',
                names='Airline',
                title=f"Market Share on {selected_route}",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                route_airline_analysis.sort_values('Average Price'),
                x='Airline',
                y='Average Price',
                title=f"Average Prices on {selected_route}",
                color='Airline'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("Airline Competition on Selected Route:")
        st.dataframe(route_airline_analysis.sort_values('Flight Count', ascending=False))
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif options == "Seasonal Trends":
        st.markdown("<h2 class='subheader'>Seasonal Price Analysis</h2>", unsafe_allow_html=True)
        
        # Check if month data is available
        if 'Month' in data.columns:
            # Create month name column
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                5: 'May', 6: 'June', 7: 'July', 8: 'August', 
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            
            data['Month_Name'] = data['Month'].map(month_names)
            
            # Monthly average prices
            monthly_avg = data.groupby('Month')['Price'].mean().reset_index()
            monthly_avg['Month_Name'] = monthly_avg['Month'].map(month_names)
            monthly_avg = monthly_avg.sort_values('Month')
            
            # Monthly price trends - overall
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Monthly Price Trends")
            
            fig = px.line(
                monthly_avg,
                x='Month_Name',
                y='Price',
                markers=True,
                title="Average Flight Prices by Month",
                line_shape='linear'
            )
            
            fig.update_xaxes(categoryorder='array', categoryarray=[month_names[i] for i in range(1, 13)])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights about seasonality
            peak_month = monthly_avg.loc[monthly_avg['Price'].idxmax()]['Month_Name']
            low_month = monthly_avg.loc[monthly_avg['Price'].idxmin()]['Month_Name']
            max_price = monthly_avg['Price'].max()
            min_price = monthly_avg['Price'].min()
            price_diff = ((max_price - min_price) / min_price * 100).round(2)
            
            st.info(f"Insight: {peak_month} has the highest average prices (₹{max_price:.2f}), while {low_month} has the lowest (₹{min_price:.2f}). The seasonal price difference is {price_diff}%.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Monthly trends by airline
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Monthly Trends by Airline")
            
            # Select airlines to compare
            selected_airlines = st.multiselect(
                "Select Airlines to Compare",
                options=data['Airline'].unique(),
                default=data['Airline'].unique()[:3]
            )
            
            if selected_airlines:
                # Filter data by selected airlines
                airline_data = data[data['Airline'].isin(selected_airlines)]
                
                # Group by airline and month
                airline_monthly = airline_data.groupby(['Airline', 'Month'])['Price'].mean().reset_index()
                airline_monthly['Month_Name'] = airline_monthly['Month'].map(month_names)
                airline_monthly = airline_monthly.sort_values(['Airline', 'Month'])
                
                fig = px.line(
                    airline_monthly,
                    x='Month_Name',
                    y='Price',
                    color='Airline',
                    markers=True,
                    title="Monthly Price Trends by Airline",
                    line_shape='linear'
                )
                
                fig.update_xaxes(categoryorder='array', categoryarray=[month_names[i] for i in range(1, 13)])
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap of airline prices by month
                pivot_data = airline_monthly.pivot(index='Airline', columns='Month_Name', values='Price')
                pivot_data = pivot_data.reindex(columns=[month_names[i] for i in range(1, 13)])
                
                fig = px.imshow(
                    pivot_data,
                    labels=dict(x="Month", y="Airline", color="Price (₹)"),
                    x=[month_names[i] for i in range(1, 13)],
                    y=selected_airlines,
                    color_continuous_scale='Viridis',
                    text_auto=True,
                    title="Price Heatmap by Airline and Month"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Please select at least one airline to see the comparison.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Seasonal pricing by route
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Seasonal Pricing by Route")
            
            # Create route column if not already done
            if 'Route' not in data.columns:
                data['Route'] = data['Source'] + ' → ' + data['Destination']
            
            # Get popular routes
            popular_routes = data['Route'].value_counts().nlargest(10).index.tolist()
            
            # Select a route to analyze
            selected_route = st.selectbox(
                "Select a Route",
                options=popular_routes
            )
            
            # Filter data for the selected route
            route_data = data[data['Route'] == selected_route]
            
            # Group by month
            route_monthly = route_data.groupby('Month')['Price'].agg(['mean', 'min', 'max']).reset_index()
            route_monthly['Month_Name'] = route_monthly['Month'].map(month_names)
            route_monthly = route_monthly.sort_values('Month')
            
            # Create a range column for the error bars
            route_monthly['price_range'] = route_monthly['max'] - route_monthly['min']
            
            fig = go.Figure()
            
            # Add the mean line
            fig.add_trace(go.Scatter(
                x=route_monthly['Month_Name'],
                y=route_monthly['mean'],
                mode='lines+markers',
                name='Average Price',
                line=dict(color='blue')
            ))
            
            # Add the min and max lines
            fig.add_trace(go.Scatter(
                x=route_monthly['Month_Name'],
                y=route_monthly['min'],
                mode='lines',
                name='Minimum Price',
                line=dict(color='green', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=route_monthly['Month_Name'],
                y=route_monthly['max'],
                mode='lines',
                name='Maximum Price',
                line=dict(color='red', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ))
            
            fig.update_layout(
                title=f"Seasonal Price Trends for {selected_route}",
                xaxis_title="Month",
                yaxis_title="Price (₹)",
                legend_title="Price",
                height=500
            )
            
            fig.update_xaxes(categoryorder='array', categoryarray=[month_names[i] for i in range(1, 13)])
            st.plotly_chart(fig, use_container_width=True)
            
            # Show airlines on this route by month
            st.write(f"Airlines Operating on {selected_route} by Month:")
            
            # Group by airline and month
            route_airline_monthly = route_data.groupby(['Airline', 'Month'])['Price'].mean().reset_index()
            route_airline_monthly['Month_Name'] = route_airline_monthly['Month'].map(month_names)
            
            # Pivot table
            pivot = route_airline_monthly.pivot(index='Airline', columns='Month_Name', values='Price')
            pivot = pivot.reindex(columns=[month_names[i] for i in range(1, 13)])
            
            # Format pivot table
            formatted_pivot = pivot.copy()
            for col in formatted_pivot.columns:
                formatted_pivot[col] = formatted_pivot[col].apply(lambda x: f"₹{x:.2f}" if not pd.isna(x) else "N/A")
            
            st.dataframe(formatted_pivot)
            
            # Best time to book recommendation
            best_month = route_monthly.loc[route_monthly['mean'].idxmin()]['Month_Name']
            worst_month = route_monthly.loc[route_monthly['mean'].idxmax()]['Month_Name']
            saving_potential = ((route_monthly['mean'].max() - route_monthly['mean'].min()) / route_monthly['mean'].max() * 100).round(2)
            
            st.success(f"Recommendation: For the route {selected_route}, the best time to book is typically in {best_month} (avg. ₹{route_monthly.loc[route_monthly['Month_Name'] == best_month, 'mean'].values[0]:.2f}), while {worst_month} is usually the most expensive month (avg. ₹{route_monthly.loc[route_monthly['Month_Name'] == worst_month, 'mean'].values[0]:.2f}). By booking at the right time, you could save up to {saving_potential}%.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        else:
            st.warning("Month data is not available in the dataset. Cannot perform seasonal analysis.")
    
    elif options == "About":
        st.markdown("<h2 class='subheader'>About This Dashboard</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h3>Advanced Airline Price Prediction Dashboard</h3>
            <p>This dashboard provides comprehensive analysis and prediction of flight prices using machine learning models. The application leverages data science and AI techniques to help users make informed decisions about airline ticket purchases.</p>
            
            <h4>Features:</h4>
            <ul>
                <li><strong>Data Exploration:</strong> Interactive visualization of flight price data across airlines, routes, and time periods.</li>
                <li><strong>Advanced Price Prediction:</strong> Machine learning models to predict flight prices based on multiple factors.</li>
                <li><strong>Market Analysis:</strong> Insights into airline market share, route profitability, and competitive dynamics.</li>
                <li><strong>Seasonal Trends:</strong> Analysis of price variations by month to identify the best time to book.</li>
            </ul>
            
            <h4>Technologies Used:</h4>
            <ul>
                <li>Streamlit for the web application framework</li>
                <li>Pandas and NumPy for data manipulation</li>
                <li>Scikit-learn for machine learning models</li>
                <li>Plotly and Matplotlib for data visualization</li>
            </ul>
            
            <h4>Dataset Information:</h4>
            <p>The dataset contains information about flight prices across various airlines, routes, and time periods. The data includes features such as airline, source, destination, price, and temporal information.</p>
            
            <h4>Model Information:</h4>
            <p>The prediction models used in this dashboard include:</p>
            <ul>
                <li><strong>Random Forest:</strong> An ensemble learning method that builds multiple decision trees and merges their predictions.</li>
                <li><strong>Gradient Boosting:</strong> A machine learning technique that builds an ensemble of shallow decision trees sequentially, with each tree correcting the errors of its predecessors.</li>
                <li><strong>Ensemble Model:</strong> A combination of multiple models to improve prediction accuracy and robustness.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Contact information and feedback
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Feedback and Suggestions")
        
        with st.form("feedback_form"):
            st.write("We'd love to hear your thoughts on how we can improve this dashboard!")
            feedback = st.text_area("Your Feedback:")
            rating = st.slider("Rate this dashboard (1-5):", 1, 5, 3)
            email = st.text_input("Your Email (optional):")
            
            submitted = st.form_submit_button("Submit Feedback")
            
            if submitted:
                st.success("Thank you for your feedback! We appreciate your input.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Version information
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Version Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("Dashboard Version: 2.0")
        
        with col2:
            st.info("Last Updated: March 23, 2025")
        
        with col3:
            st.info("Data Last Refreshed: March 21, 2025")
        
        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check if the dataset file is correctly placed and the format is appropriate.")