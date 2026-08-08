import os

import pandas as pd


OUTPUT = "./output"


def select_row(path, model_contains, label, contribution):
    frame = pd.read_csv(os.path.join(OUTPUT, path))
    if model_contains == "Light-GMTC":
        selected = frame[frame["Model"] == model_contains]
    else:
        selected = frame[frame["Model"].str.contains(model_contains, regex=False)]
    row = selected.iloc[-1].to_dict()
    row["Experiment"] = label
    row["Contribution"] = contribution
    return row


def main():
    rows = [
        select_row(
            "ablation_no_transfer_oselm.csv", "Light-GMTC",
            "A0 Light-GMTC", "depthwise-separable TimesNet",
        ),
        select_row(
            "ablation_no_transfer_oselm.csv", "OSELM-online",
            "A1 A0+Robust-OSELM", "online residual correction",
        ),
        select_row(
            "ablation_no_transfer_reliability.csv", "Causal-MultiRate",
            "A2 A0+Causal-MultiRate+Reliability", "causal rates and intervals",
        ),
        select_row(
            "ablation_full_transfer_oselm.csv", "OSELM-online",
            "A3 Full-transfer+OSELM", "unrestricted source transfer",
        ),
        select_row(
            "ablation_selective_transfer_oselm.csv", "OSELM-online",
            "A4 Selective-Trend-transfer+OSELM", "negative-transfer control",
        ),
        select_row(
            "ablation_selective_transfer_reliability.csv", "Causal-MultiRate",
            "A5 A4+Causal-MultiRate+Reliability", "combined reliability branch",
        ),
    ]
    result = pd.DataFrame(rows)
    first = ["Experiment", "Contribution", "MAE%", "MAPE%", "RMSE%", "R2"]
    result = result[first + [c for c in result.columns if c not in first and c != "Model"]]
    result.to_csv(os.path.join(OUTPUT, "innovation_complete_ablation.csv"), index=False)
    print("=" * 92)
    print("创新模块完整消融汇总")
    print("=" * 92)
    print(result[first].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
