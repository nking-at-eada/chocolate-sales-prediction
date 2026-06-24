import numpy as np
import pandas as pd


def clean_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stray currency symbols from Amount and cast to float."""
    df = df.copy()
    df["Amount"] = (
        df["Amount"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    )
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    return df


def fix_boxes_shipped(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Boxes_Shipped"] = df["Boxes_Shipped"].abs()
    return df


def parse_order_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    known_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y"]

    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    remaining = df["Order_Date"].notna()

    for fmt in known_formats:
        if not remaining.any():
            break
        attempt = pd.to_datetime(
            df.loc[remaining, "Order_Date"], format=fmt, errors="coerce"
        )
        newly_parsed = attempt.notna()
        parsed.loc[attempt.index[newly_parsed]] = attempt[newly_parsed]
        remaining.loc[attempt.index[newly_parsed]] = False

    df["Order_Date"] = parsed
    return df


def impute_missing_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Discount_Pct", "Price_per_Box", "Marketing_Spend"]:
        df[col] = df.groupby("Product")[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col] = df[col].fillna(df[col].median())
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df["Order_Date"].dt.year
    df["Month"] = df["Order_Date"].dt.month
    df["Quarter"] = df["Order_Date"].dt.quarter
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_amount(df)
    df = fix_boxes_shipped(df)
    df = parse_order_date(df)
    df = df.dropna(subset=["Order_Date", "Amount"])
    df = impute_missing_numeric(df)
    df = add_calendar_features(df)
    df = df.reset_index(drop=True)
    return df


def clean_batch_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = fix_boxes_shipped(df)
    df = parse_order_date(df)
    if df["Order_Date"].isna().any():
        raise ValueError(
            "One or more rows have an Order_Date that could not be parsed. "
            "Fix the date format and resubmit."
        )
    df = impute_missing_numeric(df)
    df = add_calendar_features(df)
    return df
