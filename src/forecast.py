import numpy as np
import pandas as pd

from src import config

GROWTH_CLIP = (-0.75, 0.75)


def _product_growth_rates(df: pd.DataFrame) -> pd.Series:
    yearly = (
        df[df["Year"].isin([2022, 2023])]
        .groupby(["Product", "Year"])["Boxes_Shipped"]
        .sum()
        .unstack("Year")
    )
    growth = (yearly[2023] - yearly[2022]) / yearly[2022].replace(0, np.nan)
    growth = growth.clip(*GROWTH_CLIP).fillna(0.0)
    return growth


def simulate_future_orders(
    df: pd.DataFrame, scenario: str = "flat", year: int = config.FORECAST_YEAR
) -> pd.DataFrame:
    last_real_year = int(df["Year"].max())
    base = df[df["Year"] == last_real_year].copy()

    sim = base.drop(columns=["Amount"], errors="ignore").copy()
    sim["Year"] = year

    if scenario == "trend":
        growth = _product_growth_rates(df)
        factor = sim["Product"].map(growth).fillna(0.0) + 1.0
        sim["Boxes_Shipped"] = (sim["Boxes_Shipped"] * factor).round().clip(lower=1)
        sim["Marketing_Spend"] = (sim["Marketing_Spend"] * factor).round(2)
    elif scenario != "flat":
        raise ValueError("scenario must be 'flat' or 'trend'")

    return sim.reset_index(drop=True)


def predict_2024(pipeline, historical_df: pd.DataFrame, scenario: str = "flat") -> pd.DataFrame:
    sim = simulate_future_orders(historical_df, scenario=scenario)
    sim["Predicted_Amount"] = pipeline.predict(sim[config.FEATURE_COLUMNS])
    return sim


def summarize_forecast(scored_df: pd.DataFrame, group_cols) -> pd.DataFrame:
    return (
        scored_df.groupby(group_cols)["Predicted_Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Predicted_Amount": "Predicted_2024_Revenue"})
    )
