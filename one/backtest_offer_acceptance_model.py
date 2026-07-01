import os
from pathlib import Path

import numpy as np
import pandas as pd

from offer_model_utils import compounded_return_and_drawdown, load_model, predict_artifact_full


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
MODEL_OUTPUT_TAG = os.getenv("MODEL_OUTPUT_TAG", "live").strip() or "live"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "live_training" / f"{SYMBOL}_offer_training_rows.csv"
INPUT_PATH = Path(os.getenv("OFFER_TRAINING_PATH", str(DEFAULT_INPUT_PATH)))
if not INPUT_PATH.is_absolute():
    INPUT_PATH = PROJECT_ROOT / INPUT_PATH

ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / f"{SYMBOL}_offer_acceptance_model.json"
CANDIDATE_MODEL_PATH_VALUE = os.getenv("CANDIDATE_MODEL_PATH", "").strip()
CANDIDATE_MODEL_PATH = (
    Path(CANDIDATE_MODEL_PATH_VALUE) if CANDIDATE_MODEL_PATH_VALUE else None
)
if CANDIDATE_MODEL_PATH is not None and not CANDIDATE_MODEL_PATH.is_absolute():
    CANDIDATE_MODEL_PATH = PROJECT_ROOT / CANDIDATE_MODEL_PATH

ACCEPT_THRESHOLD = float(os.getenv("ACCEPT_THRESHOLD", "0.60"))
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
ALLOCATION_THRESHOLDS = [
    ("> 0%", 0.0000001),
    (">= 1%", 0.01),
    (">= 2.5%", 0.025),
    (">= 5%", 0.05),
    (">= 10%", 0.10),
    (">= 20%", 0.20),
]
BUCKET_TO_FRACTION = {
    0: 0.00,
    1: 0.01,
    2: 0.025,
    3: 0.05,
    4: 0.10,
    5: 0.20,
}
TOP_SLICES = [0.005, 0.01, 0.02, 0.05, 0.10]


def load_rows():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing offer training rows: {INPUT_PATH}. Run npm run offer-build first."
        )
    frame = pd.read_csv(INPUT_PATH)
    required = ["timestamp", "offer_side", "net_return", "opportunity_score", "hit_result"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Offer training CSV is missing required columns: {missing}")
    for column in frame.columns:
        if column not in {"time", "symbol", "offer_side", "hit_result"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "net_return", "opportunity_score"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def choose_model_path():
    if CANDIDATE_MODEL_PATH is not None:
        return CANDIDATE_MODEL_PATH, "candidate"
    return ACTIVE_MODEL_PATH, "active"


def summarize_rows(rows):
    if len(rows) == 0:
        return {
            "accepted_count": 0,
            "accept_rate": 0.0,
            "average_opportunity_score": 0.0,
            "average_net_return": 0.0,
            "win_rate_after_costs": 0.0,
            "max_drawdown": 0.0,
            "long_count": 0,
            "short_count": 0,
            "average_target_allocation_fraction": 0.0,
            "average_predicted_allocation_fraction": 0.0,
            "average_paper_weighted_return": 0.0,
            "paper_weighted_total_return": 0.0,
            "paper_weighted_max_drawdown": 0.0,
            "average_predicted_quality_score": 0.0,
            "average_actual_quality_score": 0.0,
        }
    _, drawdown = compounded_return_and_drawdown(rows["net_return"])
    weighted_returns = (
        rows["paper_weighted_return"]
        if "paper_weighted_return" in rows.columns
        else rows["net_return"] * 0.0
    )
    weighted_total, weighted_drawdown = compounded_return_and_drawdown(weighted_returns)
    predicted_quality_column = (
        "predicted_opportunity_score"
        if "predicted_opportunity_score" in rows.columns
        else None
    )
    return {
        "accepted_count": int(len(rows)),
        "accept_rate": float(len(rows) / max(1, rows.attrs.get("total_rows", len(rows)))),
        "average_opportunity_score": float(rows["opportunity_score"].mean()),
        "average_net_return": float(rows["net_return"].mean()),
        "win_rate_after_costs": float((rows["net_return"] > 0).mean()),
        "max_drawdown": float(drawdown),
        "long_count": int((rows["offer_side"] == "LONG").sum()),
        "short_count": int((rows["offer_side"] == "SHORT").sum()),
        "average_target_allocation_fraction": float(rows["target_allocation_fraction"].mean()) if "target_allocation_fraction" in rows.columns else 0.0,
        "average_predicted_allocation_fraction": float(rows["predicted_allocation_fraction"].mean()) if "predicted_allocation_fraction" in rows.columns else 0.0,
        "average_paper_weighted_return": float(weighted_returns.mean()) if len(weighted_returns) else 0.0,
        "paper_weighted_total_return": float(weighted_total),
        "paper_weighted_max_drawdown": float(weighted_drawdown),
        "average_predicted_quality_score": float(rows[predicted_quality_column].mean()) if predicted_quality_column else 0.0,
        "average_actual_quality_score": float(rows["opportunity_score"].mean()),
    }


def format_percent(value):
    return f"{value:.2%}"


def print_table_header(title):
    print(f"\n{title}")
    print(
        "label | accepted_count | accept_rate | avg_opportunity | avg_net | "
        "win_after_costs | max_drawdown | LONG | SHORT | avg_target_alloc | "
        "avg_pred_alloc | avg_weighted_return | weighted_total | weighted_dd | "
        "avg_pred_quality | avg_actual_quality"
    )


def print_summary_row(label, summary):
    print(
        f"{label} | "
        f"{summary['accepted_count']} | "
        f"{format_percent(summary['accept_rate'])} | "
        f"{summary['average_opportunity_score']:.6f} | "
        f"{format_percent(summary['average_net_return'])} | "
        f"{format_percent(summary['win_rate_after_costs'])} | "
        f"{format_percent(summary['max_drawdown'])} | "
        f"{summary['long_count']} | "
        f"{summary['short_count']} | "
        f"{format_percent(summary['average_target_allocation_fraction'])} | "
        f"{format_percent(summary['average_predicted_allocation_fraction'])} | "
        f"{format_percent(summary['average_paper_weighted_return'])} | "
        f"{format_percent(summary['paper_weighted_total_return'])} | "
        f"{format_percent(summary['paper_weighted_max_drawdown'])} | "
        f"{summary['average_predicted_quality_score']:.6f} | "
        f"{summary['average_actual_quality_score']:.6f}"
    )


def add_predictions(rows, model):
    if "w_bucket" not in model.get("model", {}):
        raise RuntimeError(
            "Selected model does not include the allocation bucket head. "
            "Retrain the offer model after the allocation-sizing update."
        )
    accept_probability, bucket_probabilities, regression = predict_artifact_full(model, rows)
    output = rows.copy()
    output["accept_probability"] = accept_probability
    output["predicted_allocation_bucket"] = np.argmax(bucket_probabilities, axis=1)
    output["predicted_allocation_fraction"] = output["predicted_allocation_bucket"].map(
        BUCKET_TO_FRACTION
    ).astype(float)
    target_columns = model.get("target_columns", [])
    for index, column in enumerate(target_columns):
        output[f"predicted_{column}"] = regression[:, index]
    if "predicted_opportunity_score" not in output.columns and regression.shape[1] > 0:
        output["predicted_opportunity_score"] = regression[:, 0]
    if "predicted_target_allocation_fraction" in output.columns:
        output["predicted_allocation_fraction"] = output[
            "predicted_target_allocation_fraction"
        ].clip(lower=0.0, upper=0.20)
    output["paper_weighted_return"] = output["predicted_allocation_fraction"] * output["net_return"]
    return output


def threshold_table(rows):
    print_table_header("Accept probability threshold table")
    total_rows = len(rows)
    for threshold in THRESHOLDS:
        accepted = rows[rows["accept_probability"] >= threshold].copy()
        accepted.attrs["total_rows"] = total_rows
        print_summary_row(f"{threshold:.2f}", summarize_rows(accepted))


def allocation_threshold_table(rows):
    print_table_header("Predicted allocation fraction threshold table")
    total_rows = len(rows)
    for label, threshold in ALLOCATION_THRESHOLDS:
        selected = rows[rows["predicted_allocation_fraction"] >= threshold].copy()
        selected.attrs["total_rows"] = total_rows
        print_summary_row(label, summarize_rows(selected))


def top_slice_table(rows):
    print_table_header("Top-ranked slices by accept_probability")
    total_rows = len(rows)
    ranked = rows.sort_values(
        ["accept_probability", "predicted_opportunity_score"],
        ascending=False,
    )
    for fraction in TOP_SLICES:
        count = max(1, int(np.ceil(len(ranked) * fraction)))
        selected = ranked.head(count).copy()
        selected.attrs["total_rows"] = total_rows
        print_summary_row(f"top {fraction:.1%}", summarize_rows(selected))


def print_baselines(rows):
    print_table_header("Baselines")
    total_rows = len(rows)
    for label, subset in [
        ("always accept LONG", rows[rows["offer_side"] == "LONG"]),
        ("always accept SHORT", rows[rows["offer_side"] == "SHORT"]),
        ("always reject", rows.iloc[0:0]),
    ]:
        subset = subset.copy()
        subset.attrs["total_rows"] = total_rows
        print_summary_row(label, summarize_rows(subset))

    if len(rows) > 0:
        grouped = rows.groupby(
            [
                "offer_side",
                "offer_horizon_minutes",
                "offer_take_profit",
                "offer_stop_loss",
            ]
        )["net_return"].mean()
        best_key = grouped.idxmax()
        best_rows = rows[
            (rows["offer_side"] == best_key[0])
            & (rows["offer_horizon_minutes"] == best_key[1])
            & (rows["offer_take_profit"] == best_key[2])
            & (rows["offer_stop_loss"] == best_key[3])
        ].copy()
        best_rows.attrs["total_rows"] = total_rows
        print_summary_row(f"best simple offer {best_key}", summarize_rows(best_rows))


def main():
    rows = load_rows()
    model_path, model_kind = choose_model_path()
    model = load_model(model_path)

    print("Paper-only offer acceptance backtest")
    print(f"SYMBOL: {SYMBOL}")
    print(f"OFFER_TRAINING_PATH: {INPUT_PATH}")
    print(f"MODEL_OUTPUT_TAG: {MODEL_OUTPUT_TAG}")
    print(f"CANDIDATE_MODEL_PATH: {CANDIDATE_MODEL_PATH if CANDIDATE_MODEL_PATH else 'not set'}")
    print(f"Model path used: {model_path if model else 'not available'}")
    print(f"Model kind: {model_kind if model else 'none'}")
    print(f"Total offer rows: {len(rows)}")
    print("No trades are placed. No models are promoted.")

    if model is None:
        print("\nNo model found. Showing baselines only.")
        print_baselines(rows)
        return

    scored = add_predictions(rows, model)
    accepted = scored[scored["accept_probability"] >= ACCEPT_THRESHOLD].copy()
    accepted.attrs["total_rows"] = len(scored)
    print_summary_row(
        f"configured threshold {ACCEPT_THRESHOLD:.2f}",
        summarize_rows(accepted),
    )

    winners = accepted[accepted["net_return"] > 0]
    losers = accepted[accepted["net_return"] <= 0]
    print(
        "winner avg accept_probability: "
        f"{winners['accept_probability'].mean() if len(winners) else 0:.4f}"
    )
    print(
        "loser avg accept_probability: "
        f"{losers['accept_probability'].mean() if len(losers) else 0:.4f}"
    )

    threshold_table(scored)
    allocation_threshold_table(scored)
    top_slice_table(scored)
    print_baselines(scored)
    print("\nNo trades were placed. No promotion was attempted.")


if __name__ == "__main__":
    main()
