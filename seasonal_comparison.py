"""Compare the non-seasonal ARIMA models against seasonal SARIMA equivalents.

The ARIMA models fitted in the notebooks produce forecasts that flatten out
within a few weeks. That is expected behaviour rather than a coding error: a
non-seasonal ARIMA has no mechanism to repeat an annual pattern, so its long
horizon forecast converges on the series mean.

The decomposition in notebook 01 shows a clear annual cycle in all three series,
which suggests the flat forecast is leaving real signal unused. This script fits
the same orders with a seasonal component added and compares the two on the same
held-out year.

Run from the repository root:  python seasonal_comparison.py
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

SEASONAL_PERIOD = 52  # weekly data, so one year is 52 observations

# The orders selected in notebook 02 from the ACF and PACF plots.
ORDERS = {
    "Data Analyst": (1, 1, 1),
    "Software Engineer": (1, 0, 1),
    "Cybersecurity Specialist": (1, 0, 1),
}


def load():
    df = pd.read_csv("data/tech_roles_trends.csv")
    df["Week"] = pd.to_datetime(df["Week"])
    return df.set_index("Week").sort_index()


def score(actual, forecast):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, forecast))),
        "MAE": float(mean_absolute_error(actual, forecast)),
        "MAPE": float(np.mean(np.abs((actual - forecast) / actual)) * 100),
    }


def main():
    df = load()
    train, test = df.iloc[:-52], df.iloc[-52:]

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    rows = []

    for ax, (role, order) in zip(axes, ORDERS.items()):
        arima_fc = ARIMA(train[role], order=order).fit().forecast(steps=52)

        sarima_fc = (
            SARIMAX(
                train[role],
                order=order,
                seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            .fit(disp=False)
            .forecast(steps=52)
        )

        a, s = score(test[role], arima_fc), score(test[role], sarima_fc)
        rows.append((role, a, s))

        ax.plot(train.index[-52:], train[role][-52:], color="#94a3b8", label="Training")
        ax.plot(test.index, test[role], color="#16a34a", lw=2, label="Actual")
        ax.plot(test.index, arima_fc, color="#dc2626", ls="--",
                label=f"ARIMA{order}  MAPE {a['MAPE']:.1f}%")
        ax.plot(test.index, sarima_fc, color="#2563eb", ls="-.",
                label=f"SARIMA{order}x(1,0,1,52)  MAPE {s['MAPE']:.1f}%")
        ax.set_title(f"{role} — non-seasonal forecast flattens, seasonal one tracks the cycle")
        ax.set_ylabel("Search interest")
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/10-arima-vs-sarima.png", dpi=130)

    header = f"{'Role':<26}{'Model':<14}{'RMSE':>8}{'MAE':>8}{'MAPE':>9}"
    print(header)
    print("-" * len(header))
    for role, a, s in rows:
        print(f"{role:<26}{'ARIMA':<14}{a['RMSE']:8.2f}{a['MAE']:8.2f}{a['MAPE']:8.2f}%")
        print(f"{'':<26}{'SARIMA':<14}{s['RMSE']:8.2f}{s['MAE']:8.2f}{s['MAPE']:8.2f}%")
        print()

    print("Saved figures/10-arima-vs-sarima.png")


if __name__ == "__main__":
    main()
