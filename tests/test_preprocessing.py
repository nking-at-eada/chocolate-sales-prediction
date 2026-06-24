
import pandas as pd
import pytest

from src import preprocessing


def test_clean_amount_strips_dollar_sign():
    df = pd.DataFrame({"Amount": ["$383.66", "245.91", "$1,000.00"]})
    out = preprocessing.clean_amount(df)
    assert list(out["Amount"]) == [383.66, 245.91, 1000.00]


def test_fix_boxes_shipped_takes_absolute_value():
    df = pd.DataFrame({"Boxes_Shipped": [-50, 30, -1]})
    out = preprocessing.fix_boxes_shipped(df)
    assert list(out["Boxes_Shipped"]) == [50, 30, 1]


def test_parse_order_date_handles_mixed_formats():
    df = pd.DataFrame(
        {"Order_Date": ["2023-12-11", "15/04/2023", "12-19-2023", "not-a-date"]}
    )
    out = preprocessing.parse_order_date(df)
    parsed = out["Order_Date"]
    assert parsed.iloc[0] == pd.Timestamp("2023-12-11")  
    assert parsed.iloc[1] == pd.Timestamp("2023-04-15")  
    assert parsed.iloc[2] == pd.Timestamp("2023-12-19")  
    assert pd.isna(parsed.iloc[3])


def test_impute_missing_numeric_uses_product_median():
    df = pd.DataFrame(
        {
            "Product": ["A", "A", "A", "B", "B"],
            "Discount_Pct": [10.0, None, 20.0, 5.0, None],
            "Price_per_Box": [3.0, 3.0, 3.0, 2.0, 2.0],
            "Marketing_Spend": [50.0, 50.0, 50.0, 30.0, 30.0],
        }
    )
    out = preprocessing.impute_missing_numeric(df)
    assert out["Discount_Pct"].isna().sum() == 0
    assert out.loc[1, "Discount_Pct"] == 15.0
    assert out.loc[4, "Discount_Pct"] == 5.0


def test_clean_data_end_to_end_no_missing_values():
    df = pd.DataFrame(
        {
            "Order_ID": ["A1", "A2"],
            "Product": ["Truffle Gift Box", "70% Dark Bar"],
            "Country": ["Australia", "Brazil"],
            "Channel": ["Retail", "Online"],
            "Salesperson": ["X", "Y"],
            "Order_Date": ["2023-01-15", "15/02/2023"],
            "Discount_Pct": [10.0, None],
            "Price_per_Box": [3.0, 12.0],
            "Marketing_Spend": [50.0, 60.0],
            "Boxes_Shipped": [-30, 40],
            "Amount": ["$300.00", "480.00"],
        }
    )
    out = preprocessing.clean_data(df)
    assert out.isna().sum().sum() == 0
    assert (out["Boxes_Shipped"] >= 0).all()
    assert out.loc[0, "Amount"] == 300.00
    assert "Year" in out.columns and "Month" in out.columns and "Quarter" in out.columns


def test_clean_batch_input_does_not_drop_rows():
    df = pd.DataFrame(
        {
            "Product": ["Truffle Gift Box", "70% Dark Bar"],
            "Country": ["Australia", "Brazil"],
            "Channel": ["Retail", "Online"],
            "Order_Date": ["2024-01-15", "2024-02-15"],
            "Discount_Pct": [10.0, 5.0],
            "Price_per_Box": [3.0, 12.0],
            "Marketing_Spend": [50.0, 60.0],
            "Boxes_Shipped": [-30, 40],
        }
    )
    out = preprocessing.clean_batch_input(df)
    assert len(out) == len(df)  
    assert (out["Boxes_Shipped"] >= 0).all()


def test_clean_batch_input_raises_on_unparseable_date():
    df = pd.DataFrame(
        {
            "Product": ["Truffle Gift Box"],
            "Country": ["Australia"],
            "Channel": ["Retail"],
            "Order_Date": ["not-a-date"],
            "Discount_Pct": [10.0],
            "Price_per_Box": [3.0],
            "Marketing_Spend": [50.0],
            "Boxes_Shipped": [30],
        }
    )
    with pytest.raises(ValueError):
        preprocessing.clean_batch_input(df)