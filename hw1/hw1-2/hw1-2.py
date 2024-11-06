import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to load data
def load_data():
    # Load the CSV file and select the desired rows and columns
    data = pd.read_csv('2330-training.csv', header=None)  # Read without header
    # Select rows 2 to 218 (index 1 to 217) and columns for features and target
    data = data.iloc[1:218, 0:7]  # Adjusting for zero-based indexing (A to G)
    data.columns = ['date', 'y', 'x1', 'x2', 'x3', 'x4', 'x5']  # Set the column names
    
    # Convert columns to numeric, coercing errors to NaN, and convert date to datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop rows with NaN values
    data = data.dropna()
    
    return data

# Streamlit UI
st.title('Stock Price Prediction Using Multiple Linear Regression')

# Load data
data = load_data()
st.write("Dataset:", data)

# Feature selection
st.subheader('Feature Selection')
features = data.columns[2:]  # Use all columns except the date and target variable 'y'
selected_features = st.multiselect('Select Features for Prediction', features)

# Ensure the user selects at least one feature
if not selected_features:
    st.warning("Please select at least one feature.")
else:
    # Split the data
    X = data[selected_features]
    y = data['y']  # Set target variable to y
    X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(X, y, data['date'], test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions for the test set
    y_pred = model.predict(X_test)

    # Model evaluation
    st.subheader('Model Evaluation')
    st.write(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
    st.write(f'R² Score: {r2_score(y_test, y_pred):.2f}')

    # Prepare data for line chart
    predictions_df = pd.DataFrame({'Date': date_test, 'Actual': y_test, 'Predicted': y_pred})
    predictions_df.reset_index(drop=True, inplace=True)

    # Predictions vs Actual Values for the test set
    st.subheader('Predictions vs Actual Values (Historical)')
    st.line_chart(predictions_df.set_index('Date'))

    # Display model coefficients
    st.write('Model Coefficients:')
    coef_df = pd.DataFrame(model.coef_, index=selected_features, columns=['Coefficient'])
    st.write(coef_df)

    # Predict future stock prices for October 2024 using all historical data
    st.subheader('Predictions for October 2024')
    
    # Generate future feature data using historical data
    future_dates = pd.date_range(start='2024-10-01', end='2024-10-31', freq='D')
    
    # Create a DataFrame for future predictions
    future_data = pd.DataFrame([data[selected_features].mean()] * len(future_dates))  # Placeholder using mean of historical data

    # Predict future stock prices
    future_predictions = model.predict(future_data)

    # Prepare the DataFrame for plotting
    future_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_predictions})

    # Combine with historical predictions for plotting
    combined_predictions_df = pd.concat([
        predictions_df.set_index('Date'),
        future_predictions_df.set_index('Date')
    ])

    # Plot combined predictions as a line chart
    st.line_chart(combined_predictions_df[['Actual', 'Predicted']])

    # Future Value Prediction
    st.subheader('Future Value Prediction (Next Day)')
    # Input for future feature values
    next_day_features = {feature: st.number_input(f'Enter value for {feature}', value=0.0) for feature in selected_features}

    if st.button('Predict Next Day Stock Price'):
        next_day_data = pd.DataFrame([next_day_features])
        next_day_prediction = model.predict(next_day_data)
        st.write(f'Predicted Next Day Stock Price: {next_day_prediction[0]:.2f}')
