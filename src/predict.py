import sys

import joblib
import pandas as pd

from src import config, data_loader, preprocessing


def run_on_demand_prediction(
    input_path=config.ON_DEMAND_INPUT_PATH,
    output_path=config.ON_DEMAND_OUTPUT_PATH,
    model_path=config.MODEL_PATH,
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run `python main.py` (or the CD pipeline) to train one first."
        )

    pipeline = joblib.load(model_path)
    raw_batch = data_loader.load_batch_input(input_path)
    clean_batch = preprocessing.clean_batch_input(raw_batch)

    predictions = pipeline.predict(clean_batch[config.FEATURE_COLUMNS])

    result = raw_batch.copy()
    result["Predicted_Amount"] = predictions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    out = run_on_demand_prediction()
    print(f"Scored {len(out)} orders.")
    print(f"Predictions written to {config.ON_DEMAND_OUTPUT_PATH}")
    sys.exit(0)
