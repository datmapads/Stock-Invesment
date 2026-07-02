"""
backtest.py  --  Model Evaluation & Backtesting (Member 4)
================================================================================
 load model, chạy dự đoán trên tập test, tính RMSE/MAE và
mô phỏng giao dịch.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========================== CẤU HÌNH ==========================
# Đường dẫn đến các file. Sửa lại cho đúng nếu cần.
CSV_PATH = "vix_features_1D_updated.csv"              # Dữ liệu đầu vào
MODEL_PATH = "best_timeseries_model.h5"               # File model .h5
NPY_PATH = "dataset_ready_for_DL.npy"                 # Dataset đã xử lý (nếu có)
REPORT_PATH = "backtest_report.txt"                   # File báo cáo đầu ra
SPLIT_RATIO = 0.8                                     # Tỉ lệ train/test (mặc định 80% train)
# ============================================================

FEATURES = [
    "open", "high", "low", "close", "volume",
    "SMA_10", "RSI_14", "lag_1", "lag_5",
    "SMA_20", "BB_upper", "BB_lower", "MACD", "MACD_Signal",
]

def load_trained_model(model_path):
    """Load model .h5, ưu tiên tensorflow.keras"""
    try:
        from tensorflow.keras.models import load_model
        return load_model(model_path, compile=False)
    except:
        import keras
        return keras.models.load_model(model_path, compile=False)

def build_feature_frame(csv_path):
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["BB_std"] = df["close"].rolling(window=20).std()
    df["BB_upper"] = df["SMA_20"] + (df["BB_std"] * 2)
    df["BB_lower"] = df["SMA_20"] - (df["BB_std"] * 2)

    df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df = df.dropna().reset_index(drop=True)
    return df

def make_sliding_windows(features, target, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(target[i+window_size])
    return np.array(X), np.array(y)

def prepare_test_set(csv_path, window_size, split_ratio=0.8):
    df = build_feature_frame(csv_path)
    data_features = df[FEATURES].values
    data_target = df["close"].values

    feature_scaler = MinMaxScaler(feature_range=(0,1))
    target_scaler = MinMaxScaler(feature_range=(0,1))
    scaled_features = feature_scaler.fit_transform(data_features)
    scaled_target = target_scaler.fit_transform(data_target.reshape(-1,1)).ravel()

    X, y_scaled = make_sliding_windows(scaled_features, scaled_target, window_size)
    split_index = int(len(X) * split_ratio)
    X_test = X[split_index:]
    y_test_scaled = y_scaled[split_index:]

    n_test = len(X_test)
    today_real = np.empty(n_test)
    next_real = np.empty(n_test)
    for j in range(n_test):
        i = split_index + j
        today_real[j] = data_target[i + window_size - 1]
        next_real[j] = data_target[i + window_size]

    return X_test, y_test_scaled, target_scaler, today_real, next_real

def predictions_to_real_price(raw_pred, target_scaler, next_real):
    raw_pred = np.asarray(raw_pred).ravel()
    pred_real = target_scaler.inverse_transform(raw_pred.reshape(-1,1)).ravel()

    real_lo, real_hi = next_real.min(), next_real.max()
    pred_lo, pred_hi = pred_real.min(), pred_real.max()
    corr = np.corrcoef(pred_real, next_real)[0,1] if np.std(pred_real)>1e-9 and np.std(next_real)>1e-9 else 0.0

    print("\n[Kiểm tra tỉ lệ]")
    print(f"  Khoảng giá thực tế: [{real_lo:.3f}, {real_hi:.3f}]")
    print(f"  Khoảng giá dự đoán: [{pred_lo:.3f}, {pred_hi:.3f}]")
    print(f"  Hệ số tương quan (corr): {corr:.3f}")

    span = max(real_hi - real_lo, 1e-9)
    if pred_lo < real_lo - span or pred_hi > real_hi + span or corr < 0.2:
        print("  [CẢNH BÁO] Giá dự đoán không nhất quán với giá thực tế. Có thể sai tỉ lệ target.")
    return pred_real

def run_backtest(pred_real, today_real, next_real):
    signals = np.where(pred_real > today_real, 1, -1)
    actual_return = (next_real - today_real) / today_real
    trade_return = signals * actual_return

    total_trades = len(trade_return)
    winning_trades = np.sum(trade_return > 0)
    losing_trades = np.sum(trade_return <= 0)
    win_rate = (winning_trades / total_trades * 100.0) if total_trades else 0.0

    equity_curve = np.cumprod(1.0 + trade_return)
    total_profit_pct = float((equity_curve[-1] - 1.0) * 100.0) if total_trades else 0.0
    simple_sum_pct = float(np.sum(trade_return) * 100.0)

    return {
        "total_trades": total_trades,
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "win_rate_pct": win_rate,
        "total_profit_pct": total_profit_pct,
        "simple_sum_return_pct": simple_sum_pct,
        "n_buy": int(np.sum(signals == 1)),
        "n_sell": int(np.sum(signals == -1)),
        "equity_curve": equity_curve,
    }

def main():
    # ---- Load model ----
    print("Loading model...")
    model = load_trained_model(MODEL_PATH)
    in_shape = model.input_shape
    window_size = int(in_shape[1])
    n_features = int(in_shape[2])
    print(f"  Model input shape = {in_shape}  ->  window={window_size}, features={n_features}")

    # ---- Chuẩn bị dữ liệu test ----
    X_test = y_test_scaled = target_scaler = today_real = next_real = None
    if os.path.exists(NPY_PATH):
        try:
            data = np.load(NPY_PATH, allow_pickle=True).item()
            cand = data.get("X_test")
            if cand is not None and cand.shape[1] == window_size and cand.shape[2] == n_features:
                print(f"Reusing existing dataset: {os.path.basename(NPY_PATH)}")
                X_test = cand
                y_test_scaled = data["y_test"].ravel()
                _, _, target_scaler, today_real, next_real = prepare_test_set(CSV_PATH, window_size, SPLIT_RATIO)
        except Exception as e:
            print(f"Could not use {NPY_PATH} ({e}); rebuilding from CSV.")

    if X_test is None:
        print(f"Building test set from CSV: {os.path.basename(CSV_PATH)}")
        X_test, y_test_scaled, target_scaler, today_real, next_real = prepare_test_set(CSV_PATH, window_size, SPLIT_RATIO)

    print(f"  X_test shape = {X_test.shape}")

    # ---- Dự đoán ----
    print("Running model.predict...")
    raw_pred = model.predict(X_test, verbose=0)
    pred_real = predictions_to_real_price(raw_pred, target_scaler, next_real)

    # ---- Regression metrics ----
    rmse = float(np.sqrt(mean_squared_error(next_real, pred_real)))
    mae = float(mean_absolute_error(next_real, pred_real))

    # ---- Backtest ----
    bt = run_backtest(pred_real, today_real, next_real)

    # ---- In báo cáo ----
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

    report = "\n".join(lines)
    print("\n" + report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved to: {REPORT_PATH}")

    # ---- Lưu chi tiết giao dịch ----
    detail_path = os.path.splitext(REPORT_PATH)[0] + "_trades.csv"
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
