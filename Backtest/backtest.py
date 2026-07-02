"""
================================================================================
backtest.py  --  Member 4: Model Evaluation & Backtesting
================================================================================

PURPOSE
-------
This script does NOT retrain the model and does NOT touch the training pipeline.
It only *reuses* the already-trained network (best_timeseries_model.h5) and the
existing data pipeline in order to:

    1. Rebuild the test set (X_test / y_test) exactly the way `test.ipynb`
       (cell 0) prepared the data.
    2. Run the trained model on the test set.
    3. Report the regression error of the model  ->  RMSE, MAE.
    4. Run a simple trading simulation (backtest):
           predicted tomorrow close > today close  ->  BUY
           otherwise                               ->  SELL
    5. Report trading performance:
           Total Trades, Winning Trades, Losing Trades,
           Win Rate (%), Total Profit (%).

WHY THE SCRIPT REBUILDS THE DATA INSTEAD OF LOADING A .npy
----------------------------------------------------------
The notebook's cell 0 saves the prepared arrays to `dataset_ready_for_DL.npy`,
but that file is NOT included in the repository. Rather than force the user to
re-run the notebook, this script reproduces cell 0's preprocessing directly from
the CSV that *is* in the repo (`vix_features_1D_updated.csv`). If the .npy file
happens to exist AND matches the model's window size, we load it directly for a
1:1 reuse. Otherwise we rebuild it - the logic is identical to cell 0.

IMPORTANT DISCREPANCY WE HANDLE AUTOMATICALLY
---------------------------------------------
The trained model file `best_timeseries_model.h5` expects an input window of
length 10 (input shape = (None, 10, 14)), while `test.ipynb` cell 0 uses
WINDOW_SIZE = 60 and cell 4 builds a *different* architecture that is never
saved. In other words, the saved model was produced by code that is not in the
notebook. To stay correct no matter what, this script reads the window size and
feature count straight from the loaded model, so the reconstructed test set
always matches what the model actually expects.

HOW TO RUN
----------
    # from inside the project folder
    python backtest.py

    # or with explicit paths
    python backtest.py --csv vix_features_1D_updated.csv \
                       --model best_timeseries_model.h5 \
                       --report backtest_report.txt

Requirements: tensorflow / keras (same version used for training, Keras 3.x),
numpy, pandas, scikit-learn.
================================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ------------------------------------------------------------------ #
# 0. Model loading (works with both Keras 3 and tf.keras)            #
# ------------------------------------------------------------------ #
def load_trained_model(model_path):
    """
    Load the pre-trained .h5 model WITHOUT recompiling it.

    compile=False is used on purpose: we never train or evaluate with Keras'
    own loop here, we only call model.predict(). Skipping compilation avoids
    needing the exact loss/optimizer objects and makes loading more robust
    across Keras versions.
    """
    last_err = None
    # Keras 3 (the version this model was saved with: keras_version 3.15.0)
    try:
        import keras
        return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        last_err = e
    # Fallback: tensorflow.keras
    try:
        from tensorflow.keras.models import load_model
        return load_model(model_path, compile=False)
    except Exception as e:
        last_err = e
    raise RuntimeError(
        f"Could not load model '{model_path}'. Last error: {last_err}\n"
        "Make sure tensorflow/keras is installed (the model was saved with "
        "Keras 3.15.0)."
    )


# ------------------------------------------------------------------ #
# 1. Preprocessing  --  faithful reproduction of test.ipynb cell 0   #
# ------------------------------------------------------------------ #
# The exact 14-feature list used by the training pipeline (cell 0), in order.
FEATURES = [
    "open", "high", "low", "close", "volume",
    "SMA_10", "RSI_14", "lag_1", "lag_5",
    "SMA_20", "BB_upper", "BB_lower", "MACD", "MACD_Signal",
]


def build_feature_frame(csv_path):
    """
    Reproduce cell 0's feature engineering step-by-step.

    The base columns (open/high/low/close/volume/SMA_10/RSI_14/lag_1/lag_5) are
    already present in the CSV. Cell 0 additionally derives 5 indicators from the
    'close' column: SMA_20, BB_upper, BB_lower, MACD, MACD_Signal. We recompute
    them here with the *same formulas* so the result is byte-for-byte the same as
    the training pipeline (this is idempotent even though the "_updated" CSV
    already contains these columns).
    """
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # --- Bollinger Bands (20-period), exactly as cell 0 ---
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["BB_std"] = df["close"].rolling(window=20).std()
    df["BB_upper"] = df["SMA_20"] + (df["BB_std"] * 2)
    df["BB_lower"] = df["SMA_20"] - (df["BB_std"] * 2)

    # --- MACD (12, 26, 9), exactly as cell 0 ---
    df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Drop the warm-up rows that contain NaN (from rolling/lag windows).
    df = df.dropna().reset_index(drop=True)
    return df


def make_sliding_windows(features, target, window_size):
    """
    Turn a 2D feature matrix into 3D sequences, identical to cell 0's
    create_sliding_window():

        X[i] = features[i : i + window_size]      (window_size days of history)
        y[i] = target[i + window_size]            (the NEXT day's close)

    So each sample looks at `window_size` past days and the label is the close
    price of the day immediately after the window.
    """
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:(i + window_size)])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)


def prepare_test_set(csv_path, window_size, split_ratio=0.8):
    """
    Rebuild the test set the same way cell 0 does and return everything the
    backtest needs.

    Returns
    -------
    X_test        : np.ndarray (n_test, window_size, 14)  -- MinMax-scaled inputs
    y_test_scaled : np.ndarray (n_test,)                  -- MinMax-scaled target
    target_scaler : fitted MinMaxScaler for 'close' (to invert predictions)
    today_real    : np.ndarray (n_test,) real close of the LAST day in each window
    next_real     : np.ndarray (n_test,) real close of the day being predicted
    """
    df = build_feature_frame(csv_path)

    # Feature matrix (X source) and target vector (next-day close source).
    data_features = df[FEATURES].values          # shape (N, 14)
    data_target = df["close"].values             # shape (N,)

    # Cell 0 scales BOTH features and target to [0, 1] with MinMaxScaler.
    # NOTE: cell 0 fits the scalers on the WHOLE series (not train-only). We keep
    # that behaviour on purpose so the reconstruction matches the saved dataset.
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(data_features)
    scaled_target = target_scaler.fit_transform(data_target.reshape(-1, 1)).ravel()

    # Build sequences.
    X, y_scaled = make_sliding_windows(scaled_features, scaled_target, window_size)

    # Time-ordered train/test split (NO shuffling for time series).
    split_index = int(len(X) * split_ratio)
    X_test = X[split_index:]
    y_test_scaled = y_scaled[split_index:]

    # ----- Recover the REAL prices we need for the trading simulation -----
    # For a test sample j, its global window index is  i = split_index + j.
    #   * last input day  = row (i + window_size - 1)  -> "today's close"
    #   * predicted day   = row (i + window_size)      -> "tomorrow's close"
    # These come straight from the un-scaled `data_target`, so no inverse
    # transform is needed for the ground-truth prices.
    n_test = len(X_test)
    today_real = np.empty(n_test)
    next_real = np.empty(n_test)
    for j in range(n_test):
        i = split_index + j
        today_real[j] = data_target[i + window_size - 1]
        next_real[j] = data_target[i + window_size]

    return X_test, y_test_scaled, target_scaler, today_real, next_real


# ------------------------------------------------------------------ #
# 2. Turn model output into real predicted prices                    #
# ------------------------------------------------------------------ #
def predictions_to_real_price(raw_pred, target_scaler, next_real):
    """
    The model outputs values in the same [0, 1] space as the scaled target
    (cell 0 scaled 'close' with MinMaxScaler and the Dense(1) head is linear).
    We invert that scaling to get predicted prices in real currency units.

    A safety check is included: if the model was actually trained on a different
    target scaling than cell 0's MinMax, the inverse-transformed prices would be
    nonsensical. We compare the predicted price cloud against the real price
    cloud and warn loudly if they are wildly inconsistent, so the result is
    never silently wrong.
    """
    raw_pred = np.asarray(raw_pred).ravel()
    pred_real = target_scaler.inverse_transform(raw_pred.reshape(-1, 1)).ravel()

    # ---- Sanity diagnostics ----
    real_lo, real_hi = next_real.min(), next_real.max()
    pred_lo, pred_hi = pred_real.min(), pred_real.max()
    # Correlation between predicted and actual next-day price.
    if np.std(pred_real) > 1e-9 and np.std(next_real) > 1e-9:
        corr = float(np.corrcoef(pred_real, next_real)[0, 1])
    else:
        corr = 0.0

    print("\n[diagnostics] target-scaling sanity check")
    print(f"  real next-close range : [{real_lo:.3f}, {real_hi:.3f}]")
    print(f"  predicted price range : [{pred_lo:.3f}, {pred_hi:.3f}]")
    print(f"  corr(pred, actual)    : {corr:.3f}")

    # Heuristic warning: predictions far outside the real price band, or almost
    # no correlation, usually means the assumed scaling does not match training.
    span = max(real_hi - real_lo, 1e-9)
    if pred_lo < real_lo - span or pred_hi > real_hi + span or corr < 0.2:
        print(
            "  [WARNING] Predicted prices look inconsistent with real prices.\n"
            "            The trained model was likely fitted with a different\n"
            "            target scaling than cell 0's MinMax (for example an\n"
            "            extra StandardScaler as in cell 4). If so, adjust\n"
            "            predictions_to_real_price() to apply the matching\n"
            "            inverse transform. Backtest numbers below assume the\n"
            "            cell 0 MinMax target scaling."
        )
    return pred_real


# ------------------------------------------------------------------ #
# 3. Regression metrics: RMSE + MAE                                  #
# ------------------------------------------------------------------ #
def regression_metrics(y_true_real, y_pred_real):
    """RMSE and MAE computed in REAL price units (easy to interpret)."""
    rmse = float(np.sqrt(mean_squared_error(y_true_real, y_pred_real)))
    mae = float(mean_absolute_error(y_true_real, y_pred_real))
    return rmse, mae


# ------------------------------------------------------------------ #
# 4. Trading simulation (backtest)                                   #
# ------------------------------------------------------------------ #
def run_backtest(pred_real, today_real, next_real):
    """
    Simple daily long/short backtest.

    Rule (from the task spec):
        predicted tomorrow close > today close  ->  BUY  (go long)
        otherwise                               ->  SELL (go short)

    Outcome for each day:
        actual return r = (next_real - today_real) / today_real
        BUY  trade profit = +r   (you gain if the price actually went up)
        SELL trade profit = -r   (you gain if the price actually went down)

    A trade is "winning" if its trade profit is > 0. Total Profit (%) is the
    compounded return of putting the whole (notional) capital into one trade
    per day - i.e. an equity curve.
    """
    # +1 = BUY (long), -1 = SELL (short)
    signals = np.where(pred_real > today_real, 1, -1)

    # Real next-day return of the asset.
    actual_return = (next_real - today_real) / today_real

    # Profit of each trade given its direction.
    trade_return = signals * actual_return

    total_trades = int(len(trade_return))
    winning_trades = int(np.sum(trade_return > 0))
    losing_trades = int(np.sum(trade_return <= 0))
    win_rate = (winning_trades / total_trades * 100.0) if total_trades else 0.0

    # Compounded total profit (equity grows/shrinks trade by trade).
    equity_curve = np.cumprod(1.0 + trade_return)
    total_profit_pct = float((equity_curve[-1] - 1.0) * 100.0) if total_trades else 0.0

    # Simple (non-compounded) sum of returns, reported as extra context.
    simple_sum_pct = float(np.sum(trade_return) * 100.0)

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": win_rate,
        "total_profit_pct": total_profit_pct,
        "simple_sum_return_pct": simple_sum_pct,
        "n_buy": int(np.sum(signals == 1)),
        "n_sell": int(np.sum(signals == -1)),
        "equity_curve": equity_curve,
    }


# ------------------------------------------------------------------ #
# 5. Reporting                                                       #
# ------------------------------------------------------------------ #
def build_report(rmse, mae, bt, window_size, n_features):
    """Assemble the human-readable performance report as a string."""
    lines = []
    lines.append("=" * 60)
    lines.append("      MODEL EVALUATION & BACKTESTING REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Model input window : {window_size} days")
    lines.append(f"Model features     : {n_features}")
    lines.append("")
    lines.append("--- Regression metrics (real price units) ---")
    lines.append(f"RMSE               : {rmse:.4f}")
    lines.append(f"MAE                : {mae:.4f}")
    lines.append("")
    lines.append("--- Backtest results ---")
    lines.append(f"Total Trades       : {bt['total_trades']}")
    lines.append(f"  BUY signals      : {bt['n_buy']}")
    lines.append(f"  SELL signals     : {bt['n_sell']}")
    lines.append(f"Winning Trades     : {bt['winning_trades']}")
    lines.append(f"Losing Trades      : {bt['losing_trades']}")
    lines.append(f"Win Rate (%)       : {bt['win_rate_pct']:.2f}")
    lines.append(f"Total Profit (%)   : {bt['total_profit_pct']:.2f}   (compounded)")
    lines.append(f"Sum of returns (%) : {bt['simple_sum_return_pct']:.2f}   (non-compounded)")
    lines.append("")
    lines.append("Note: Win Rate here is also the model's directional accuracy")
    lines.append("(fraction of days the up/down direction was predicted right).")
    lines.append("=" * 60)
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# 6. Main                                                            #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Backtest & evaluate the pre-trained stock model."
    )
    # Default paths assume the script sits in the project root.
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--csv", default=os.path.join(here, "vix_features_1D_updated.csv"))
    parser.add_argument("--model", default=os.path.join(here, "best_timeseries_model.h5"))
    parser.add_argument("--npy", default=os.path.join(here, "dataset_ready_for_DL.npy"),
                        help="Optional prebuilt dataset from cell 0 (reused if it matches the model).")
    parser.add_argument("--report", default=os.path.join(here, "backtest_report.txt"))
    parser.add_argument("--split", type=float, default=0.8, help="Train ratio (test = 1 - split).")
    args = parser.parse_args()

    # --- Load the trained model and read what it expects ---
    print("Loading trained model ...")
    model = load_trained_model(args.model)
    # Input shape is (None, window_size, n_features).
    in_shape = model.input_shape
    window_size = int(in_shape[1])
    n_features = int(in_shape[2])
    print(f"  model input shape = {in_shape}  ->  window={window_size}, features={n_features}")
    if n_features != len(FEATURES):
        print(f"  [WARNING] model expects {n_features} features but FEATURES has "
              f"{len(FEATURES)}. Check the feature list matches training.")

    # --- Get X_test / y_test, either from the saved .npy or by rebuilding ---
    X_test = y_test_scaled = target_scaler = today_real = next_real = None
    if os.path.exists(args.npy):
        try:
            data = np.load(args.npy, allow_pickle=True).item()
            cand = data.get("X_test")
            if cand is not None and cand.shape[1] == window_size and cand.shape[2] == n_features:
                print(f"Reusing existing dataset: {os.path.basename(args.npy)}")
                # We still need real prices + target scaler, so rebuild those
                # from the CSV (cheap and keeps the .npy as the source of X/y).
                X_test = cand
                y_test_scaled = data["y_test"].ravel()
                _, _, target_scaler, today_real, next_real = prepare_test_set(
                    args.csv, window_size, args.split
                )
            else:
                print(f"  {os.path.basename(args.npy)} window/feature size does not "
                      f"match the model; rebuilding from CSV instead.")
        except Exception as e:
            print(f"  Could not use {args.npy} ({e}); rebuilding from CSV.")

    if X_test is None:
        print(f"Rebuilding test set from CSV (cell 0 pipeline): {os.path.basename(args.csv)}")
        X_test, y_test_scaled, target_scaler, today_real, next_real = prepare_test_set(
            args.csv, window_size, args.split
        )

    print(f"  X_test shape = {X_test.shape}")

    # --- Predict ---
    print("Running model.predict on the test set ...")
    raw_pred = model.predict(X_test, verbose=0)

    # --- Convert predictions to real prices ---
    pred_real = predictions_to_real_price(raw_pred, target_scaler, next_real)

    # --- Regression metrics vs the real next-day close ---
    rmse, mae = regression_metrics(next_real, pred_real)

    # --- Trading simulation ---
    bt = run_backtest(pred_real, today_real, next_real)

    # --- Report ---
    report = build_report(rmse, mae, bt, window_size, n_features)
    print("\n" + report)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved to: {args.report}")

    # --- Optional: per-trade detail CSV for the write-up / appendix ---
    detail_path = os.path.splitext(args.report)[0] + "_trades.csv"
    detail = pd.DataFrame({
        "today_close": today_real,
        "predicted_next_close": pred_real,
        "actual_next_close": next_real,
        "signal": np.where(pred_real > today_real, "BUY", "SELL"),
        "actual_return_pct": (next_real - today_real) / today_real * 100.0,
        "equity_curve": bt["equity_curve"],
    })
    detail.to_csv(detail_path, index=False)
    print(f"Per-trade detail saved to: {detail_path}")


if __name__ == "__main__":
    main()