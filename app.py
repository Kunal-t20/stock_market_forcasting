import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# Cached Data Loading
@st.cache_data
def load_and_preprocess_data(file_path, model_type):
    try:
        df = pd.read_csv(file_path)
        if model_type == 'Prophet':
            if 'ds' not in df.columns or 'y' not in df.columns:
                st.error("Prophet data must have 'ds' and 'y' columns.")
                return None
            df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
            ts_data = df[['ds', 'y']].rename(columns={'y':'Close'}).set_index('ds')
        else:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                ts_data = df.set_index('Date')
            else:
                ts_data = df.copy()
                ts_data['Date'] = pd.date_range(start="2000-01-01", periods=len(df), freq='D')
                ts_data = ts_data.set_index('Date')
            if 'Close' not in ts_data.columns:
                st.error(f"{model_type} data must have 'Close' column.")
                return None
            ts_data['Close'] = pd.to_numeric(ts_data['Close'], errors='coerce')
            ts_data.fillna(method='ffill', inplace=True)
        return ts_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Cached LSTM Model Load
@st.cache_resource
def load_lstm_model(path):
    return tf.keras.models.load_model(path)

# LSTM Forecast
def run_lstm(train_data, test_data, model):
    try:
        scaler = MinMaxScaler()
        full_data = pd.concat([train_data, test_data])
        scaler.fit(full_data)

        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)

        time_step = 100
        X_test = []

        if len(test_scaled) <= time_step:
            X_test.append(np.concatenate([train_scaled[-time_step:], test_scaled], axis=0)[:time_step])
        else:
            for i in range(len(test_scaled) - time_step):
                X_test.append(test_scaled[i:i+time_step, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(np.concatenate([np.zeros((predictions.shape[0], full_data.shape[1]-1)), predictions], axis=1))[:, -1]

        forecast_index = test_data.index[-len(predictions):]
        forecast = pd.Series(predictions, index=forecast_index)
        return forecast
    except Exception as e:
        st.error(f"LSTM failed: {e}")
        return pd.Series(np.nan, index=test_data.index)

# ARIMA Forecast
def run_arima(train_data, test_data):
    try:
        model = ARIMA(train_data['Close'], order=(1,1,1))
        fit = model.fit()
        forecast = fit.forecast(steps=len(test_data))
        forecast.index = test_data.index
        return forecast
    except Exception as e:
        st.error(f"ARIMA failed: {e}")
        return pd.Series(np.nan, index=test_data.index)

# SARIMA Forecast
def run_sarima(train_data, test_data):
    try:
        model = SARIMAX(train_data['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
        fit = model.fit(disp=False)
        forecast = fit.get_forecast(steps=len(test_data)).predicted_mean
        forecast.index = test_data.index
        return forecast
    except Exception as e:
        st.error(f"SARIMA failed: {e}")
        return pd.Series(np.nan, index=test_data.index)

# Prophet Forecast
def run_prophet(train_data, test_data):
    prophet_df = train_data.reset_index().rename(columns={'Date':'ds','Close':'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=len(test_data), include_history=False)
    forecast_df = model.predict(future)
    forecast = pd.Series(forecast_df['yhat'].values, index=test_data.index)
    return forecast

#Model Comparison
def compare_models():
    st.subheader("Model Comparison")
    files = {
        'Prophet': ('data/prophet.csv', run_prophet),
        'ARIMA': ('data/arima_data.csv', run_arima),
        'SARIMA': ('data/arima_data.csv', run_sarima),
        'LSTM': ('data/scaled_data.csv', run_lstm)
    }
    results = {}
    lstm_model = load_lstm_model('notebook/lstm_trained.h5')
    for name, (path, func) in files.items():
        data = load_and_preprocess_data(path, 'Prophet' if name=='Prophet' else 'ARIMA')
        if data is not None:
            train_size = int(len(data)*0.8)
            train = data[:train_size]
            test = data[train_size:]
            forecast = func(train, test, lstm_model) if name=='LSTM' else func(train, test)
            if forecast is not None and not forecast.isnull().all():
                actual = test['Close']
                min_len = min(len(actual), len(forecast))
                actual = actual.iloc[-min_len:]
                forecast = forecast.iloc[-min_len:]
                rmse = np.sqrt(mean_squared_error(actual, forecast))
                mae = mean_absolute_error(actual, forecast)
                results[name] = {'RMSE': f"{rmse:.2f}", 'MAE': f"{mae:.2f}"}
    if results:
        st.dataframe(pd.DataFrame.from_dict(results, orient='index'))
    else:
        st.error("No model ran successfully.")

# --- Streamlit UI ---
st.title("Stock Forecast Dashboard")
st.sidebar.header("Select Model")
model_choice = st.sidebar.selectbox("Forecasting Model", ('Prophet','ARIMA','SARIMA','LSTM'))

ts_data = None
if model_choice == 'Prophet':
    ts_data = load_and_preprocess_data('data/prophet.csv','Prophet')
elif model_choice == 'LSTM':
    ts_data = load_and_preprocess_data('data/scaled_data.csv','LSTM')
else:
    ts_data = load_and_preprocess_data('data/arima_data.csv','ARIMA/SARIMA')

if ts_data is not None:
    st.subheader("Historical Data")
    st.line_chart(ts_data)
    
    train_size = int(len(ts_data)*0.8)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]

    lstm_model = load_lstm_model('notebook/lstm_trained.h5')

    forecast = None
    if model_choice == 'ARIMA':
        forecast = run_arima(train_data, test_data)
    elif model_choice == 'SARIMA':
        forecast = run_sarima(train_data, test_data)
    elif model_choice == 'Prophet':
        forecast = run_prophet(train_data, test_data)
    elif model_choice == 'LSTM':
        forecast = run_lstm(train_data, test_data, lstm_model)

    if forecast is not None and not forecast.isnull().all():
        actual_values = test_data['Close']
        min_len = min(len(actual_values), len(forecast))
        actual_values = actual_values.iloc[-min_len:]
        forecast = forecast.iloc[-min_len:]

        rmse = np.sqrt(mean_squared_error(actual_values, forecast))
        mae = mean_absolute_error(actual_values, forecast)

        st.markdown("### Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("MAE", f"{mae:.2f}")

        st.markdown("### Forecast Plot")
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(ts_data.index, ts_data['Close'], label='Historical', color='blue')
        ax.plot(test_data.index[-min_len:], actual_values, label='Actual', color='green')
        ax.plot(forecast.index, forecast, label='Forecast', color='red', linestyle='--')
        ax.set_title(f"{model_choice} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# --- Compare Models Button ---
if st.button("Compare All Models"):
    compare_models()