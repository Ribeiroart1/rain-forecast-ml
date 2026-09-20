import argparse
import joblib
import pandas as pd
import requests

from location import get_coordinates, slugify
from collect_data import DAILY_VARIABLES
from train_model import build_features


def fetch_recent_data(lat, lon, days=10):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARIABLES),
        "past_days": days,
        "forecast_days": 1,
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()["daily"]
    df = pd.DataFrame(data).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def predict(city):
    slug = slugify(city)
    model = joblib.load(f"rain_model_{slug}.pkl")
    model_columns = joblib.load(f"model_columns_{slug}.pkl")

    lat, lon, formatted_name = get_coordinates(city)
    df = fetch_recent_data(lat, lon)
    df, feature_columns = build_features(df)
    df = df.dropna(subset=feature_columns).reset_index(drop=True)

    last_row = df[feature_columns].iloc[[-1]][model_columns]
    probability = model.predict_proba(last_row)[0][1]
    forecast = "WILL RAIN" if probability >= 0.5 else "will NOT rain"

    print(f"Forecast for tomorrow in {formatted_name}: {forecast}")
    print(f"Rain probability: {probability:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Goiania, Brazil")
    args = parser.parse_args()
    predict(args.city)


if __name__ == "__main__":
    main()