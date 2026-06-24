import pandas as pd


def _historical_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rev_by_year = (
        df[df["Year"].isin([2022, 2023])]
        .groupby([group_col, "Year"])["Amount"]
        .sum()
        .unstack("Year")
    )
    yoy_growth = (
        (rev_by_year[2023] - rev_by_year[2022]) / rev_by_year[2022].replace(0, pd.NA)
    ).fillna(0.0)

    spend_2023 = df[df["Year"] == 2023].groupby(group_col)["Marketing_Spend"].sum()
    revenue_2023 = df[df["Year"] == 2023].groupby(group_col)["Amount"].sum()
    marketing_roi = (revenue_2023 / spend_2023.replace(0, pd.NA)).fillna(0.0)

    return pd.DataFrame(
        {
            "Revenue_2023": revenue_2023,
            "YoY_Growth_2022_to_2023": yoy_growth,
            "Marketing_ROI_2023": marketing_roi,
        }
    )


def rank_performers(
    historical_df: pd.DataFrame, forecast_summary: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    hist = _historical_summary(historical_df, group_col)
    table = hist.merge(
        forecast_summary.set_index(group_col), left_index=True, right_index=True
    )

    table["rank_predicted_2024"] = table["Predicted_2024_Revenue"].rank()
    table["rank_growth"] = table["YoY_Growth_2022_to_2023"].rank()
    table["rank_marketing_roi"] = table["Marketing_ROI_2023"].rank()
    table["composite_score"] = table[
        ["rank_predicted_2024", "rank_growth", "rank_marketing_roi"]
    ].mean(axis=1)

    table = table.sort_values("composite_score").reset_index()
    table = table.rename(columns={"index": group_col})
    return table


def recommend_drops(ranked_table: pd.DataFrame, group_col: str, n: int = 1) -> str:
    worst = ranked_table.head(n)
    lines = [f"Lowest-ranked {group_col.lower()}(s), candidates to discontinue:"]
    for _, row in worst.iterrows():
        lines.append(
            f"  - {row[group_col]}: predicted 2024 revenue "
            f"${row['Predicted_2024_Revenue']:,.0f}, "
            f"2022->2023 growth {row['YoY_Growth_2022_to_2023']*100:.1f}%, "
            f"marketing ROI {row['Marketing_ROI_2023']:.2f}x"
        )
    return "\n".join(lines)
