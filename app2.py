# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:28:15 2024

@author: Dell
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objs as go

# Load the LSTM model and scaler
with open("C:\\Users\\moham\\OneDrive\\文件\\project\\lstm_model.pkl", 'rb') as f:
    model_lstm = pickle.load(f)

with open("C:\\Users\\moham\\OneDrive\\文件\\project\\scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Apple Stock Price Prediction", 
                   page_icon='https://cdn.freebiesupply.com/images/large/2x/apple-logo-transparent.png',
                   layout="wide")


# Display logo and text with aligned layout
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="https://cdn.freebiesupply.com/images/large/2x/apple-logo-transparent.png" width="50" style="margin-right:10px;"/>
        <span style="font-size:24px; font-weight:bold;color:black;">Apple Stock Price Prediction</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("This app predicts Apple stock prices using a trained LSTM model.")



# User input: Upload CSV file
#st.sidebar.header("Historical Data")
st.sidebar.markdown(
    '<h3 style="color:black;">Historical Data</h3>',
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("Please upload a CSV file containing at least the last 10 closing prices.", type="csv")

# Initialize last_date variable
last_date = None

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Check if the DataFrame has the necessary column (assumed to be 'Close' and 'Date')
    if 'Close' in df.columns and 'Date' in df.columns:
        # Get the last date from the DataFrame
        last_date = pd.to_datetime(df['Date'].values[-1])  # Ensure the date column is correctly named

        # Get the last 10 closing prices
        closing_prices = df['Close'].values[-10:]

        if len(closing_prices) != 10:
            st.sidebar.error("The uploaded file must contain at least 10 closing prices.")
        else:
            # Prepare input data for prediction
            input_prices = closing_prices.reshape(-1, 1)
            input_scaled = scaler.transform(input_prices)
            X_input = input_scaled.reshape(1, 10, 1)  # Reshape for LSTM model

            # User input: Future date
            # Define the minimum date as the last date in the uploaded file
            #min_date = last_date.date()
            min_date = last_date.date() + timedelta(days=1)
            future_date = st.sidebar.date_input("Select a future date for prediction", min_value=min_date)

            if st.button('Predict'):
                # Predict stock prices from the last date to the specified future date
                predicted_prices = []
                num_days = (future_date - min_date).days + 1  # Include the last date in predictions

                # Display number of days being predicted
                st.write(f"Predicting stock prices from {min_date} to {future_date}")


                # Loop for the number of days to predict
                for _ in range(num_days):
                    # Make sure to skip weekends
                    if min_date.weekday() >= 5:  # If it's Saturday or Sunday, move to next weekday
                        min_date += timedelta(days=1)
                        continue
                    
                    predicted_price = model_lstm.predict(X_input)
                    predicted_price_inverse = scaler.inverse_transform(predicted_price)  # Inverse transform to original scale
                    predicted_prices.append(predicted_price_inverse[0][0])

                    # Update input for the next prediction
                    new_input = np.append(input_prices, predicted_price_inverse[0][0]).reshape(-1, 1)[-10:]  # Append predicted price
                    input_scaled = scaler.transform(new_input)
                    X_input = input_scaled.reshape(1, 10, 1)  # Reshape for LSTM model

                    # Move to the next day
                    min_date += timedelta(days=1)

                # Create a DataFrame for the predictions
                future_dates = []
                start_date = pd.to_datetime(last_date)+timedelta(days=1) # Start from the last date
                for i in range(len(predicted_prices)):
                    # Skip weekends when generating the future dates
                    while start_date.weekday() >= 5:  # If it's Saturday or Sunday, move to next weekday
                        start_date += timedelta(days=1)
                    future_dates.append(start_date.date())
                    start_date += timedelta(days=1)  # Move to the next day for the next prediction

                # Create a DataFrame for the predicted prices
                predictions_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': predicted_prices
                })

                # Display the line plot for closing prices and predicted prices
                #st.write("### Stock Prices and Predictions")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(df['Date'].values[-10:]),  # Last 10 dates
                    y=closing_prices,
                    mode='lines+markers',
                    name='Historical Prices',
                    line=dict(color='blue')
                ))
                
                # Add trace to join the last point of the blue line to the first point of the orange line
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(df['Date'].values[-1]), future_dates[0]],  # Last actual date and first future date
                    y=[closing_prices[-1], predicted_prices[0]],  # Last actual price and first predicted price
                    mode='lines',
                    line=dict(color='orange'),
                    showlegend=False  # Hide from legend
                ))
                
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(future_dates),
                    y=predicted_prices,
                    mode='lines+markers',
                    name='Predicted Prices',
                    line=dict(color='orange')
                ))


                # Update layout
                fig.update_layout(
                    title='Stock Price History and Predictions',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend=dict(x=0, y=1),
                    xaxis=dict(tickformat='%Y-%m-%d')
                )
                st.plotly_chart(fig)

                # Display the predictions as a table
                st.write("### Predicted Prices")
                st.dataframe(predictions_df)

                # Provide an option to download the predictions as a CSV file
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='predicted_prices.csv',
                    mime='text/csv'
                )

    else:
        st.sidebar.error("The CSV file must contain columns named 'Close' and 'Date'.")
else:
    st.sidebar.info("The CSV file must contain columns named 'Close' and 'Date'.")

with st.sidebar:
    footer_html = """<div style='text-align: left;'>
    <p style="margin-bottom:2cm;"> </p>
    <p style="color:black";> <b> Designed and Developed by </b> </br> <i> Mohammed Noman </i> </p>
    </div>"""
    st.markdown(footer_html, unsafe_allow_html=True)