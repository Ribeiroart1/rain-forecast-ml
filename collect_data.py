import argparse
import requests
import pandas as pd
from datetime import date, timedelta

from location import get_coordinates, slugify

YEARS_OF_HISTORY = 10

DAILY_VARIABLES = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "sunshine_duration",
    "shortwave_radiation_sum",
]


def fetch_historical_data(lat, lon, years=YEARS_OF_HISTORY):
    end_date = date.today() - timedelta(days=1)
    start_date = end_date.replace(year=end_date.year - years)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
    }

    print(f"Downloading data from {start_date} to {end_date} (lat={lat:.2f}, lon={lon:.2f})...")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    daily_data = response.json()["daily"]

    df = pd.DataFrame(daily_data)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Goiania, Brazil")
    args = parser.parse_args()

    slug = slugify(args.city)
    lat, lon, formatted_name = get_coordinates(args.city)
    print(f"City found: {formatted_name} (lat={lat}, lon={lon})")

    df = fetch_historical_data(lat, lon)
    print(f"{len(df)} days collected.")

    csv_path = f"weather_data_{slug}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


if __name__ == "__main__":
    main()