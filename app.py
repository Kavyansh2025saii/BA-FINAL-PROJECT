"""Krishi-Masam AI demo. Run: streamlit run app.py"""
from datetime import date, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Punjab AgriWeather", page_icon="🌾", layout="wide")
DISTRICTS = {
    "Jalandhar": (31.3260, 75.5762),
    "Amritsar": (31.6340, 74.8723),
    "Ludhiana": (30.9010, 75.8573),
}
PRODUCTS = {
    "Drip irrigation kits": {"weekly": 12, "stock": 22, "lead": 12, "category": "dry"},
    "Sprinkler sets": {"weekly": 8, "stock": 14, "lead": 10, "category": "dry"},
    "Irrigation pipe bundles": {"weekly": 18, "stock": 28, "lead": 8, "category": "dry"},
    "Drainage pipe bundles": {"weekly": 10, "stock": 15, "lead": 10, "category": "wet"},
    "Dewatering pump rentals": {"weekly": 5, "stock": 7, "lead": 7, "category": "wet"},
}
FEATURES = ["month", "day_sin", "day_cos", "rain_lag1", "rain_avg3", "rain_avg7", "temp_lag1", "temp_avg3", "temp_avg7"]

@st.cache_data(ttl=86400, show_spinner=False)
def load_history(district):
    lat, lon = DISTRICTS[district]
    end = date.today() - timedelta(days=7)  # archive observations may arrive late
    start = end - timedelta(days=365 * 4)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": lat, "longitude": lon, "start_date": str(start), "end_date": str(end),
              "daily": "precipitation_sum,temperature_2m_mean", "timezone": "Asia/Kolkata"}
    response = requests.get(url, params=params, timeout=35)
    response.raise_for_status()
    daily = response.json()["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"]),
                       "rain": pd.to_numeric(daily["precipitation_sum"], errors="coerce"),
                       "temp": pd.to_numeric(daily["temperature_2m_mean"], errors="coerce")})
    return df.dropna().sort_values("date").reset_index(drop=True)

def build_training(df):
    x = df.copy()
    x["month"] = x.date.dt.month
    day = x.date.dt.dayofyear
    x["day_sin"] = np.sin(2 * np.pi * day / 365.25)
    x["day_cos"] = np.cos(2 * np.pi * day / 365.25)
    for target, prefix in [("rain", "rain"), ("temp", "temp")]:
        prior = x[target].shift(1)
        x[f"{prefix}_lag1"] = prior
        x[f"{prefix}_avg3"] = prior.rolling(3).mean()
        x[f"{prefix}_avg7"] = prior.rolling(7).mean()
    return x.dropna().reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def fit_models(district, algorithm):
    raw = load_history(district)
    train = build_training(raw)
    if len(train) < 200:
        raise ValueError("Insufficient historical observations to train the model.")
    cut = int(len(train) * 0.8)  # time-based split; never shuffle a time series
    def estimator():
        if algorithm == "XGBoost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05,
                                subsample=0.85, colsample_bytree=0.9, random_state=42,
                                objective="reg:squarederror", n_jobs=2)
        return RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3,
                                     random_state=42, n_jobs=-1)
    models, metrics = {}, {}
    for target in ["rain", "temp"]:
        model = estimator()
        model.fit(train.iloc[:cut][FEATURES], train.iloc[:cut][target])
        pred = model.predict(train.iloc[cut:][FEATURES])
        pred = np.maximum(pred, 0) if target == "rain" else pred
        metrics[target] = mean_absolute_error(train.iloc[cut:][target], pred)
        model.fit(train[FEATURES], train[target])
        models[target] = model
    return raw, models, metrics

def predict_next_days(raw, models, days):
    # Iterative forecasts become increasingly uncertain; no future weather features are leaked.
    history = raw[["date", "rain", "temp"]].copy()
    output = []
    for i in range(days):
        when = history.date.iloc[-1] + timedelta(days=1)
        day = when.dayofyear
        recent = history.tail(7)
        features = pd.DataFrame([{
            "month": when.month, "day_sin": np.sin(2*np.pi*day/365.25),
            "day_cos": np.cos(2*np.pi*day/365.25),
            "rain_lag1": float(recent.rain.iloc[-1]),
            "rain_avg3": float(recent.rain.tail(3).mean()),
            "rain_avg7": float(recent.rain.mean()),
            "temp_lag1": float(recent.temp.iloc[-1]),
            "temp_avg3": float(recent.temp.tail(3).mean()),
            "temp_avg7": float(recent.temp.mean())
        }])[FEATURES]
        rain = max(0., float(models["rain"].predict(features)[0]))
        temp = float(models["temp"].predict(features)[0])
        history = pd.concat([history, pd.DataFrame([{"date": when, "rain": rain, "temp": temp}])], ignore_index=True)
        output.append({"date": when, "rain_mm": round(rain, 1), "temp_c": round(temp, 1)})
    return pd.DataFrame(output)

def condition(forecast, history):
    total = forecast.rain_mm.sum()
    # Compare against historical same-calendar-month mean daily rainfall, scaled to horizon.
    month = forecast.date.iloc[0].month
    subset = history[history.date.dt.month == month]
    expected = max(float(subset.rain.mean()) * len(forecast), 1.0)
    ratio = total / expected
    if ratio >= 1.5:
        return "Wetter than seasonal baseline", "wet", ratio
    if ratio <= 0.55:
        return "Drier than seasonal baseline", "dry", ratio
    return "Near seasonal baseline", "normal", ratio

def demand_factor(category, scenario):
    # Transparent illustrative assumptions, NOT fitted sales forecasts.
    if scenario == "dry":
        return 1.30 if category == "dry" else 0.85
    if scenario == "wet":
        return 1.35 if category == "wet" else 0.80
    return 1.0

def inventory_plan(scenario, weeks, sales_scale=1.0, stock_scale=1.0):
    rows = []
    for product, item in PRODUCTS.items():
        weekly = item["weekly"] * sales_scale
        current = round(item["stock"] * stock_scale)
        factor = demand_factor(item["category"], scenario)
        demand = int(np.ceil(weekly * weeks * factor))
        safety = int(np.ceil(weekly * 0.5))
        order = max(0, demand + safety - current)
        rows.append({"Product": product, "Current stock (example)": current,
                     "Expected demand (scenario)": demand,
                     "Safety stock": safety, "Suggested order (example)": order,
                     "Lead time (days, example)": item["lead"]})
    return pd.DataFrame(rows)

st.title("🌾 Krishi-Mausam AI")
st.caption("Prototype: agriculture department • distributors • local retailers • farmers")
st.warning("Research demo, not a certified weather forecast or agronomic prescription. Historical weather comes from Open-Meteo; inventory and product demand are illustrative rules, not measured local sales.")
with st.sidebar:
    st.header("Forecast settings")
    district = st.selectbox("District", list(DISTRICTS))
    algorithm = st.radio("ML algorithm", ["Random Forest", "XGBoost"])
    horizon = st.slider("Forecast horizon (days)", 3, 14, 7)
    st.caption("Beyond a few days, iterative ML forecasts can be unreliable. A production system should use meteorological forecasts and uncertainty ranges.")
    st.header("Inventory scenario inputs")
    sales_scale = st.slider("Sales multiplier vs example shop", 0.25, 3.0, 1.0, 0.25)
    stock_scale = st.slider("Current stock multiplier", 0.0, 3.0, 1.0, 0.25)

try:
    with st.spinner("Loading historical weather and training model..."):
        raw, models, metrics = fit_models(district, algorithm)
        forecast = predict_next_days(raw, models, horizon)
        status, scenario, ratio = condition(forecast, raw)
except Exception as exc:
    st.error(f"Weather data unavailable or training failed: {exc}")
    st.info("An internet connection is required for the Open-Meteo historical weather API. Try again later.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("District", district)
m2.metric("Forecast rain total", f"{forecast.rain_mm.sum():.1f} mm")
m3.metric("Mean forecast temp", f"{forecast.temp_c.mean():.1f} °C")
m4.metric("Seasonal comparison", f"{ratio:.2f}×")
st.write(f"**Scenario:** {status}. Comparison uses historical rainfall from the same calendar month, not a flood or drought warning.")
st.subheader("Weather model performance")
st.write(f"Chronological 80/20 holdout: rainfall MAE **{metrics['rain']:.2f} mm/day**, temperature MAE **{metrics['temp']:.2f} °C**. These are one-day-ahead test errors; they do **not** establish the accuracy of the iterative {horizon}-day forecast.")
st.line_chart(forecast.set_index("date")[["rain_mm"]])
st.line_chart(forecast.set_index("date")[["temp_c"]])
with st.expander("See daily forecast"):
    st.dataframe(forecast, use_container_width=True, hide_index=True)

st.subheader("Stakeholder dashboards")
tabs = st.tabs(["🏛️ Agriculture Department", "🚚 Distributor / Wholesaler", "🏪 Retailer / Small Business", "👨‍🌾 Farmer"])
plan = inventory_plan(scenario, horizon/7, sales_scale, stock_scale)
with tabs[0]:
    st.markdown(f"**District:** {district}  \n**Weather scenario:** {status}")
    st.write("Suggested action: circulate a district-level advisory, consult local agricultural officers and crop-specific bulletins, and coordinate with relevant supply networks. Do not treat model outputs as official warnings.")
    st.dataframe(pd.DataFrame({"Indicator": ["Rainfall total (model)", "Mean temperature (model)", "Relative seasonal rainfall"],
                              "Value": [f"{forecast.rain_mm.sum():.1f} mm", f"{forecast.temp_c.mean():.1f} °C", f"{ratio:.2f}×"]}), hide_index=True)
with tabs[1]:
    st.write("**Illustrative district replenishment signals** for participating dealers. These figures are a single example shop's scenario, not a district-wide demand estimate.")
    st.dataframe(plan[["Product", "Expected demand (scenario)", "Suggested order (example)", "Lead time (days, example)"]], use_container_width=True, hide_index=True)
    st.info("For a real district total, collect participating retailers' inventory, historical sales, service area and supplier lead times; aggregate without double-counting.")
with tabs[2]:
    st.write("Enter your own baseline sales and current stock in a later production version. Sliders in the sidebar scale the example values.")
    st.dataframe(plan, use_container_width=True, hide_index=True)
    st.download_button("Download illustrative stock plan CSV", plan.to_csv(index=False).encode(),
                       file_name=f"{district.lower()}_inventory_scenario.csv", mime="text/csv")
with tabs[3]:
    crop = st.selectbox("Crop (for advisory context)", ["Wheat", "Rice", "Maize", "Vegetables", "Other"])
    if scenario == "wet":
        st.write(f"**{district} / {crop}:** Monitor field drainage and standing water. Avoid unnecessary irrigation. Check local crop-specific advisories before buying crop-protection products.")
    elif scenario == "dry":
        st.write(f"**{district} / {crop}:** Check soil moisture and irrigation availability. Consider water-saving practices where appropriate. Do not buy equipment solely on a district-wide weather forecast.")
    else:
        st.write(f"**{district} / {crop}:** Continue monitoring field moisture and local forecasts. Avoid changing irrigation or input purchases based only on this general scenario.")
    st.caption("Advisories must be tailored to crop stage, soil, field conditions, and local agricultural guidance.")

st.divider()
st.caption("Data: Open-Meteo Historical Weather API. ML: RandomForestRegressor or XGBRegressor. Historical observations are gridded estimates, not field measurements. Forecasts may start several days before today due to archive reporting lag.")
