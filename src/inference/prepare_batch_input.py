from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "artifacts/evaluation/test.csv"
)

OUTPUT_DIR = Path(
    "artifacts/batch_inference"
)

OUTPUT_FILE = OUTPUT_DIR / "batch_input.csv"


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        INPUT_FILE,
        header=None
    )

    # column 0 = churn_value
    # remaining columns = model features
    features = df.iloc[:, 1:]

    features.to_csv(
        OUTPUT_FILE,
        index=False,
        header=False
    )

    print(
        f"Batch input created: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(features)}"
    )

    print(
        f"Features: {features.shape[1]}"
    )


if __name__ == "__main__":
    main()