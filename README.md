# COVID-19 Global Data Analysis

> End-to-end data analysis and ML forecasting project on COVID-19 global trends (2020–2022).

## What This Project Does

- 📊 **EDA** — Visualises four epidemic waves (daily cases + deaths)
- 💉 **Vaccination impact** — Correlates vaccine rollout with case fatality rate
- 🤖 **ML forecasting** — Polynomial regression to predict future daily cases
- 📅 **Monthly aggregation** — Identifies worst months across the pandemic

## Connection to SIR / SIRA

Real COVID-19 data follows SIR-like dynamics. Each wave has:
- A **susceptible** population pool
- An **infection peak** (I compartment)
- A **removal phase** (recovered + deceased)

This project shows how real epidemic data relates to the mathematical SIR model
explored in the companion [sir-epidemic-ml](https://github.com/IslamMahmoud-ai/sir-epidemic-ml) project.

## Project Structure

```
covid19-analysis/
├── src/
│   └── analysis.py            # Data generation + analysis functions
├── notebooks/
│   └── covid_analysis.ipynb   # Full interactive analysis
├── data/                      # Generated plots and CSV
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/IslamMahmoud-ai/covid19-analysis
cd covid19-analysis
pip install -r requirements.txt
python src/analysis.py
```

## Tech Stack

- Python 3.11+
- pandas, NumPy — data handling
- matplotlib — visualisation
- scikit-learn — ML forecasting
- SciPy — signal smoothing

## Author

**Islam Mahmoud** — [@IslamMahmoud-ai](https://github.com/IslamMahmoud-ai)
