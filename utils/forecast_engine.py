import pandas as pd
from prophet import Prophet

def generate_forecast():
    df = pd.read_csv("data/inflation_data.csv")
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=60, freq="M")
    forecast = model.predict(future)

    forecast_summary = forecast[["ds", "yhat"]].tail(5)

    return forecast_summary.to_string(index=False)
