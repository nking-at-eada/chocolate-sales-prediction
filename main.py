import sys

from src import analyze, config, data_loader, forecast, preprocessing, train


def main():
    print("1/5  Loading raw data...")
    raw_df = data_loader.load_raw_data()
    print(f"      {len(raw_df):,} raw rows from {config.RAW_DATA_PATH}")

    print("2/5  Cleaning data...")
    clean_df = preprocessing.clean_data(raw_df)
    dropped = len(raw_df) - len(clean_df)
    print(f"      {len(clean_df):,} rows after cleaning ({dropped:,} dropped: "
          f"unparseable dates / missing target)")

    print("3/5  Training candidate models (tracked in MLflow)...")
    result = train.train_and_select_best(clean_df)
    best = result["best"]
    print(f"      Best model: {best['name']}")
    for row in result["leaderboard"]:
        print(f"        {row['model']:<24} RMSE={row['RMSE']:.2f}  "
              f"MAE={row['MAE']:.2f}  MAPE={row['MAPE']:.1f}%  R2={row['R2']:.3f}")

    print(f"4/5  Saving best model to {config.MODEL_PATH}...")
    train.save_model(best["pipeline"], best["name"], best["metrics"])

    print("5/5  Forecasting 2024 and ranking products/channels...")
    scored = forecast.predict_2024(best["pipeline"], clean_df, scenario="flat")
    total_2023 = clean_df.loc[clean_df["Year"] == 2023, "Amount"].sum()
    total_2024 = scored["Predicted_Amount"].sum()
    print(f"      2023 actual revenue:        ${total_2023:,.0f}")
    print(f"      2024 predicted revenue:     ${total_2024:,.0f} "
          f"({(total_2024/total_2023 - 1)*100:+.1f}% vs 2023, 'flat' scenario)")

    product_summary = forecast.summarize_forecast(scored, "Product")
    channel_summary = forecast.summarize_forecast(scored, "Channel")
    ranked_products = analyze.rank_performers(clean_df, product_summary, "Product")
    ranked_channels = analyze.rank_performers(clean_df, channel_summary, "Channel")

    print()
    print(analyze.recommend_drops(ranked_products, "Product", n=2))
    print()
    print(analyze.recommend_drops(ranked_channels, "Channel", n=1))

    print()
    print("Done. Model + metadata saved in models/. "
          "Run notebooks/experiments.ipynb for full EDA and MLflow results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
