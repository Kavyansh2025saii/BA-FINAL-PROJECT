# Krishi-Mausam AI — College Project Prototype

A Streamlit prototype with four stakeholder views: Punjab Agriculture Department, distributor/wholesaler, agricultural retailer, and farmer. Covers **Jalandhar, Amritsar, and Ludhiana districts** using representative city coordinates (not district-wide spatial averages).

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app requires internet access to fetch historical daily weather from the **Open-Meteo Archive API**. It caches the response. It trains **two separate regressors** (rainfall and temperature) using either Random Forest or XGBoost, with lagged weather and calendar features. The last 20% of chronological historical observations are used to calculate one-day-ahead MAE, then the models are refit on all observations. Future predictions are generated recursively.

## Important limitations

- Forecasts are **experimental**, not official meteorological forecasts. Iterative predictions for 7–14 days can have substantial errors. Historical observations stop approximately 7 days before today due to potential archive delays, so forecast dates may not start tomorrow.
- Historical data are grid-based weather estimates at district-headquarter coordinates, not complete district averages.
- Seasonal comparison is a simple same-calendar-month baseline; it is **not** a flood or drought diagnosis.
- Inventory baseline sales, stock, lead times, demand multipliers, and order recommendations are **illustrative assumptions**. They are not ML-trained demand predictions and should not be used for actual purchasing.
- Farmer advice is generic and not a substitute for crop-specific guidance.

## Suggested improvements for final submission

1. Add a reliable live forecast API (e.g. Open-Meteo forecast or IMD) and use it for operational advisories; keep the ML model for benchmarking.
2. Get historical product sales and inventory for real dealers in each district, then train a **separate demand model** and report demand MAE.
3. Use district-level gridded aggregation or multiple stations, rather than a single representative coordinate.
4. Evaluate weather models against a seasonal naive baseline and with rolling-origin validation.
5. Incorporate crop, sowing calendar, soil moisture, farmer needs, and supplier lead times before recommending specific products.
6. Distinguish weather forecast **lead time** from stock delivery lead time and send alerts early enough to act.

## Recommended ML comparison

Random Forest is a solid baseline. XGBoost can sometimes improve accuracy, but neither requires independent weather variables. Compare both on the same time-ordered test periods using MAE/RMSE, and choose based on observed performance.
