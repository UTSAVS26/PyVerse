# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load the dataset
file_path = 'link_to_dataset//WHO-COVID-19-global-daily-data.csv'
df = pd.read_csv(file_path)

# Convert 'Date_reported' to datetime format
df['Date_reported'] = pd.to_datetime(df['Date_reported'])

# Streamlit app title
st.title("COVID-19 Data Analysis")

# Get input from user for countries to analyze
country_input = st.text_input("Enter the names of countries to analyze (comma-separated):")
if country_input:
    selected_countries = [country.strip() for country in country_input.split(',')]

    # Function to analyze data for selected countries
    def analyze_covid_data(countries):
        for country in countries:
            # Filter data for the specific country
            df_country = df[df['Country'] == country].copy()

            # Handle missing values (fill NaN with 0 for new cases)
            df_country['New_cases'].fillna(0, inplace=True)

            # Set the frequency of the time series to daily ('D')
            df_country.set_index('Date_reported', inplace=True)
            df_country = df_country.asfreq('D')

            # Plotting the new cases over time
            st.subheader(f'Daily New COVID-19 Cases in {country}')
            fig = px.line(df_country, y='New_cases', title=f'Daily New COVID-19 Cases in {country}')
            st.plotly_chart(fig)

            # Time Series Forecasting with ARIMA
            # Split into train and test data (80% train, 20% test)
            train_size = int(len(df_country) * 0.8)
            train_data, test_data = df_country['New_cases'][0:train_size], df_country['New_cases'][train_size:]

            # Fit ARIMA model (try different values for (p, d, q))
            arima_model = ARIMA(train_data, order=(3, 1, 2))  # Adjust order as necessary
            arima_model_fit = arima_model.fit()

            # Make predictions on the test data
            predictions = arima_model_fit.forecast(steps=len(test_data))
            test_data = test_data.reset_index(drop=True)

            # Plot actual vs predicted values
            st.subheader(f"Actual vs Predicted Daily New Cases for {country}")
            fig2 = px.line(title="Actual vs Predicted Daily New Cases")
            fig2.add_scatter(x=df_country.index[train_size:], y=test_data, mode='lines', name='Actual')
            fig2.add_scatter(x=df_country.index[train_size:], y=predictions, mode='lines', name='Predicted', line=dict(color='red'))
            st.plotly_chart(fig2)

            # Calculate and display error (MSE)
            mse = mean_squared_error(test_data, predictions)
            st.write(f"Mean Squared Error (MSE) of ARIMA model for {country}: {mse}")

            # Future forecasting
            forecast_steps = 14  # Forecast the next 14 days
            future_forecast = arima_model_fit.forecast(steps=forecast_steps)

            # Display future forecast
            st.subheader(f"Future {forecast_steps}-Day Forecast of New Cases for {country}")
            future_dates = pd.date_range(df_country.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
            forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted New Cases': future_forecast})
            st.write(forecast_df)

            # Plot future forecast
            fig3 = px.line(forecast_df, x='Date', y='Forecasted New Cases', title=f"{forecast_steps}-Day COVID-19 New Cases Forecast for {country}")
            st.plotly_chart(fig3)

    # Call the analysis function
    analyze_covid_data(selected_countries)
