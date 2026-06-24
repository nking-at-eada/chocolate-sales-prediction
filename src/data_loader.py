"""
Data ingestion layer.

Today this reads a static CSV export. It is kept behind a single function
on purpose: if the source later becomes a database table or an API
(e.g. an ERP/Order Management System export job), only `load_raw_data()`
needs to change -- preprocessing, feature engineering, training and
prediction code never touch the file system directly.
"""
from pathlib import Path
import pandas as pd

from src import config


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw chocolate sales transactions.

    Parameters
    ----------
    path : Path
        Location of the CSV file. Defaults to datasets/Chocolate_Sales.csv.

    Returns
    -------
    pd.DataFrame
        Raw, uncleaned data exactly as found in the source file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find raw data at {path}. "
            "Place Chocolate_Sales.csv in the datasets/ folder."
        )
    df = pd.read_csv(path)
    return df


def load_batch_input(path: Path = config.ON_DEMAND_INPUT_PATH) -> pd.DataFrame:
    """Load a batch of new/hypothetical orders for on-demand prediction.

    Same schema as the raw data, minus the Amount target column
    (Amount is what we are predicting).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find batch input at {path}. "
            "Place a CSV there before running the on-demand workflow."
        )
    return pd.read_csv(path)
