# Rain Forecast with Machine Learning

A project that uses free historical weather data from Open-Meteo to train
a model that predicts whether it will rain the next day, for any city in
the world.

## Setup (Windows)

1. Open this folder in VS Code.
2. Create and activate a virtual environment.
3. Install dependencies:
```
   pip install -r requirements.txt
```
   or, with uv:
```
   uv pip install -r requirements.txt
```

## Usage

All three scripts accept `--city "City Name, Country"`. If you don't pass
anything, it defaults to Goiania, Brazil.

```
python collect_data.py --city "Sao Paulo, Brazil"
python train_model.py --city "Sao Paulo, Brazil"
python predict_tomorrow.py --city "Sao Paulo, Brazil"
```

More examples:

```
python collect_data.py --city "Paris, France"
python train_model.py --city "Paris, France"
python predict_tomorrow.py --city "Paris, France"

python collect_data.py --city "Tokyo"
python train_model.py --city "Tokyo"
python predict_tomorrow.py --city "Tokyo"
```

Use the exact same `--city` value across all three commands for a given
city — that's how the scripts know which data/model file to use.

Each city generates its own files (`weather_data_<city>.csv`,
`rain_model_<city>.pkl`, `confusion_matrix_<city>.png`), so multiple
cities can be trained without overwriting each other.

## How location lookup works

`location.py` uses Open-Meteo's free geocoding API to turn a city name
into latitude/longitude — covers the whole world, no API key needed.

## About the model

- **Target**: 1 if it rained >= 1mm the next day, 0 otherwise.
- **Features**: temperature, precipitation, wind, sunshine and radiation
  for the current day, plus a 3-day rolling average of those same
  variables, plus the month.
- **Model**: `RandomForestClassifier`, trained from scratch per city
  (not a single global model — rainfall patterns vary too much by
  location for one model to generalize well).
- **Evaluation**: time-based train/test split (train on the past, test
  on the most recent period).

## Possible next steps

- Adjust `RAIN_THRESHOLD_MM` in `train_model.py`.
- Try other models (logistic regression, XGBoost) and compare metrics.
- Predict rainfall *amount* (regression) instead of just yes/no.
- Add a script that trains multiple cities at once from a list.
- Automate daily predictions with GitHub Actions.