import re
import unicodedata
import requests


def slugify(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def get_coordinates(city_name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    results = response.json().get("results")

    if not results:
        raise ValueError(f"Could not find city '{city_name}'. Try another name.")

    r = results[0]
    formatted_name = f"{r['name']}, {r.get('country', '')}".strip(", ")
    return r["latitude"], r["longitude"], formatted_name