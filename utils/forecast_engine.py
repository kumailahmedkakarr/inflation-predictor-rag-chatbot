import pandas as pd
from prophet import Prophet
import os


def generate_forecast():
    try:
        # Get base directory (project root)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(BASE_DIR, "data", "inflation_data.csv")

        # Read CSV
        df = pd.read_csv(file_path)

        # Ensure correct column names
        df.columns = ["ds", "y"]

        # Convert date column
        df["ds"] = pd.to_datetime(df["ds"])

        # Train Prophet model
        model = Prophet()
        model.fit(df)

        # Forecast next 60 months
        future = model.make_future_dataframe(periods=60, freq="M")
        forecast = model.predict(future)

        # Get last 5 predicted values
        forecast_summary = forecast[["ds", "yhat"]].tail(5)

        return forecast_summary.to_string(index=False)

    except Exception as e:
        return f"Forecast Error: {str(e)}"
