"""
COVID-19 Global Data Analysis
Author: Islam Mahmoud
GitHub: IslamMahmoud-ai

Analyzes global COVID-19 trends using real-world data patterns.
Demonstrates: data cleaning, EDA, time-series analysis, ML forecasting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


# ─── 1. Generate Realistic Synthetic COVID-19 Dataset ────────────────────────

def generate_covid_data(seed=42):
    """
    Generate a realistic synthetic COVID-19 dataset mimicking
    global wave patterns (Wave 1 → Wave 2 → Wave 3 → Omicron).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    n = len(dates)
    t = np.arange(n)

    # Four epidemic waves with different intensities
    wave1   = 80_000  * np.exp(-((t - 100) ** 2) / (2 * 30**2))
    wave2   = 200_000 * np.exp(-((t - 300) ** 2) / (2 * 45**2))
    wave3   = 350_000 * np.exp(-((t - 500) ** 2) / (2 * 60**2))
    omicron = 800_000 * np.exp(-((t - 720) ** 2) / (2 * 40**2))

    base        = wave1 + wave2 + wave3 + omicron
    daily_cases = np.clip(base + rng.normal(0, base * 0.08 + 100, n), 0, None)

    # Cumulative cases and deaths
    cumulative_cases  = np.cumsum(daily_cases)
    daily_deaths      = daily_cases * rng.uniform(0.005, 0.025, n)
    cumulative_deaths = np.cumsum(daily_deaths)

    # Vaccinations start mid-2021
    vax_start = (dates >= "2021-01-01").astype(float)
    daily_vax = np.clip(
        vax_start * (rng.normal(3_000_000, 300_000, n) * np.minimum(t / 500, 1)),
        0, None
    )
    cumulative_vax = np.cumsum(daily_vax)

    return pd.DataFrame({
        "date":              dates,
        "daily_cases":       daily_cases.astype(int),
        "daily_deaths":      daily_deaths.astype(int),
        "cumulative_cases":  cumulative_cases.astype(int),
        "cumulative_deaths": cumulative_deaths.astype(int),
        "cumulative_vax":    cumulative_vax.astype(int),
        "cfr":               np.where(cumulative_cases > 0,
                                      cumulative_deaths / cumulative_cases * 100, 0),
    })


# ─── 2. Exploratory Data Analysis ────────────────────────────────────────────

def plot_waves(df):
    """Plot daily cases with 7-day rolling average."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Daily cases + rolling average
    roll = df["daily_cases"].rolling(7).mean()
    axes[0].fill_between(df["date"], df["daily_cases"],
                         alpha=0.3, color="#3B8BD4", label="Daily cases")
    axes[0].plot(df["date"], roll, color="#3B8BD4", lw=2, label="7-day average")
    axes[0].set_ylabel("Cases")
    axes[0].set_title("COVID-19 Global Daily Cases — Four Waves")
    axes[0].legend()

    # Daily deaths
    roll_d = df["daily_deaths"].rolling(7).mean()
    axes[1].fill_between(df["date"], df["daily_deaths"],
                         alpha=0.3, color="#E24B4A", label="Daily deaths")
    axes[1].plot(df["date"], roll_d, color="#E24B4A", lw=2, label="7-day average")
    axes[1].set_ylabel("Deaths")
    axes[1].set_title("COVID-19 Global Daily Deaths")
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("data/waves.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_vaccination_impact(df):
    """Show vaccination rollout vs case fatality rate."""
    fig, ax1 = plt.subplots(figsize=(14, 5))

    color1 = "#1D9E75"
    ax1.plot(df["date"], df["cumulative_vax"] / 1e9,
             color=color1, lw=2, label="Cumulative vaccinations (billions)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Vaccinations (billions)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    color2 = "#E24B4A"
    cfr_smooth = pd.Series(df["cfr"]).rolling(14).mean()
    ax2.plot(df["date"], cfr_smooth, color=color2, lw=2,
             linestyle="--", label="Case fatality rate (%) — 14d avg")
    ax2.set_ylabel("Case Fatality Rate (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Vaccination Rollout vs Case Fatality Rate")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.88))
    plt.tight_layout()
    plt.savefig("data/vaccination_impact.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─── 3. ML Forecasting ───────────────────────────────────────────────────────

def forecast_cases(df, forecast_days=30):
    """
    Use polynomial regression to forecast daily cases
    for the next `forecast_days` days.
    """
    # Use last 90 days as training window
    train = df.tail(90).copy()
    train["t"] = np.arange(len(train))

    X = train[["t"]].values
    y = train["daily_cases"].values

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # Forecast
    t_future = np.arange(len(train), len(train) + forecast_days).reshape(-1, 1)
    X_future = poly.transform(t_future)
    y_forecast = np.clip(model.predict(X_future), 0, None)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train["date"], y, color="#3B8BD4", lw=2, label="Actual (last 90 days)")
    future_dates = pd.date_range(
        train["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=forecast_days
    )
    ax.plot(future_dates, y_forecast, color="#E24B4A", lw=2,
            linestyle="--", label=f"Forecast ({forecast_days} days)")
    ax.fill_between(future_dates,
                    y_forecast * 0.85, y_forecast * 1.15,
                    alpha=0.2, color="#E24B4A", label="Uncertainty band")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Cases")
    ax.set_title("COVID-19 Case Forecasting — Polynomial Regression")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/forecast.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Metrics on training data
    y_pred_train = model.predict(X_poly)
    print(f"  Training R²  : {r2_score(y, y_pred_train):.4f}")
    print(f"  Training MAE : {mean_absolute_error(y, y_pred_train):.0f} cases/day")
    return y_forecast


# ─── 4. Summary Statistics ───────────────────────────────────────────────────

def print_summary(df):
    total_cases  = df["cumulative_cases"].iloc[-1]
    total_deaths = df["cumulative_deaths"].iloc[-1]
    total_vax    = df["cumulative_vax"].iloc[-1]
    peak_day     = df.loc[df["daily_cases"].idxmax(), "date"]
    peak_cases   = df["daily_cases"].max()

    print("\n" + "=" * 50)
    print("  COVID-19 Dataset Summary")
    print("=" * 50)
    print(f"  Period         : {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"  Total cases    : {total_cases:,.0f}")
    print(f"  Total deaths   : {total_deaths:,.0f}")
    print(f"  Total vax doses: {total_vax:,.0f}")
    print(f"  Peak day       : {peak_day.date()} ({peak_cases:,.0f} cases)")
    print(f"  Final CFR      : {df['cfr'].iloc[-1]:.2f}%")
    print("=" * 50)


# ─── 5. Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[1] Generating COVID-19 dataset …")
    df = generate_covid_data()
    df.to_csv("data/covid19_data.csv", index=False)
    print(f"    Saved → data/covid19_data.csv  ({len(df)} rows)")

    print_summary(df)

    print("\n[2] Plotting epidemic waves …")
    plot_waves(df)

    print("\n[3] Plotting vaccination impact …")
    plot_vaccination_impact(df)

    print("\n[4] Forecasting cases …")
    forecast_cases(df, forecast_days=30)

    print("\nDone! ✓")
