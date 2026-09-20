import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from location import slugify

RAIN_THRESHOLD_MM = 1.0

NUMERIC_COLUMNS = [
    "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "sunshine_duration", "shortwave_radiation_sum",
]


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_features(df):
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        df[f"{col}_avg_3d"] = df[col].rolling(3).mean()
    df["month"] = df["date"].dt.month
    feature_columns = NUMERIC_COLUMNS + [f"{c}_avg_3d" for c in NUMERIC_COLUMNS] + ["month"]
    return df, feature_columns


def build_target(df):
    return (df["rain_sum"].shift(-1) >= RAIN_THRESHOLD_MM).astype(int)


def split_train_test_by_time(X, y, test_ratio=0.2):
    cutoff = int(len(X) * (1 - test_ratio))
    return X.iloc[:cutoff], X.iloc[cutoff:], y.iloc[:cutoff], y.iloc[cutoff:]


def train_and_evaluate(slug, display_name):
    df = load_data(f"weather_data_{slug}.csv")
    df, feature_columns = build_features(df)
    df["rain_tomorrow"] = build_target(df)
    df = df.dropna().reset_index(drop=True)

    X = df[feature_columns]
    y = df["rain_tomorrow"]
    X_train, X_test, y_train, y_test = split_train_test_by_time(X, y)

    print(f"City: {display_name}")
    print(f"Train: {len(X_train)} days | Test: {len(X_test)} days")
    print(f"Share of rainy days overall: {y.mean():.1%}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Classification report (test set) ---")
    print(classification_report(y_test, y_pred, target_names=["No rain", "Rain"]))

    matrix = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(matrix, display_labels=["No rain", "Rain"]).plot()
    plt.title(f"Confusion matrix - {display_name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{slug}.png")
    print(f"Confusion matrix saved to confusion_matrix_{slug}.png")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Top features ---")
    print(importances.head(10))

    joblib.dump(model, f"rain_model_{slug}.pkl")
    joblib.dump(list(X.columns), f"model_columns_{slug}.pkl")
    print(f"\nModel saved to rain_model_{slug}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Goiania, Brazil")
    args = parser.parse_args()
    slug = slugify(args.city)
    train_and_evaluate(slug, args.city)


if __name__ == "__main__":
    main()