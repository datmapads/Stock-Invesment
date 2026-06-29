import pandas as pd
import pandas_ta as ta
from PostgreSQL import get_conn 

def load_data():
    with get_conn() as conn:
        # Cập nhật câu lệnh SQL lấy đúng các cột đã lưu từ PostgresManager.py
        df = pd.read_sql("""
        SELECT price_open, price_high, price_low, price_close, total_volume, created_at
        FROM vix
        ORDER BY created_at ASC
        """, conn )
    return df

# ---------------------------------------------------------
# XỬ LÝ DỮ LIỆU ĐƯA VÀO AI (KHUNG 1D)
# ---------------------------------------------------------

print("Đang tải dữ liệu từ PostgreSQL...")
df = load_data()

# Time-based Features
df["created_at"] = pd.to_datetime(df["created_at"])
df.set_index("created_at", inplace=True)

# Nén dữ liệu thành khung 1 Ngày (1D)
# Vì script realtime lưu liên tục, ta cần tổng hợp lại điểm đầu/cuối của mỗi ngày
df_1d = df.resample("1D").agg({
    "price_open": "first",    # Lấy giá mở cửa phiên sáng
    "price_high": "max",      # Lấy mức giá cao nhất đạt được trong ngày
    "price_low": "min",       # Lấy mức giá thấp nhất trong ngày
    "price_close": "last",    # Lấy giá chốt phiên (đóng cửa)
    "total_volume": "last"    # Khối lượng giao dịch thường cộng dồn, lấy giá trị cuối ngày
})

# Loại bỏ các ngày nghỉ (Thứ 7, CN, Lễ) bị Pandas tự động sinh ra với giá trị NaN
df_1d.dropna(inplace=True)

# Technical Indicators (Chỉ báo kỹ thuật)
# Áp dụng các chỉ báo trên giá đóng cửa của khung 1D
df_1d["SMA_10"] = ta.sma(length=10, close=df_1d["price_close"])
df_1d["RSI_14"] = ta.rsi(length=14, close=df_1d["price_close"])

# 3. Lag Features (Dữ liệu quá khứ)
# Trích xuất giá đóng cửa của 1 ngày và 5 ngày (tương đương 1 tuần giao dịch) trước đó
df_1d["price_lag_1"] = df_1d["price_close"].shift(1)
df_1d["price_lag_5"] = df_1d["price_close"].shift(5)

# 4. Làm sạch dữ liệu
# Các dòng đầu tiên sẽ bị NaN vì chưa đủ số ngày gom dữ liệu để tính SMA/RSI/Lag
df_1d.dropna(inplace=True)
df_1d.reset_index(inplace=True)

# ---------------------------------------------------------
# LƯU DỮ LIỆU ĐỂ HUẤN LUYỆN AI
# ---------------------------------------------------------
output_file = "vix_features_1D.csv"
df_1d.to_csv(output_file, index=False)

print(f"\n✅ Đã xử lý xong! Dữ liệu 1D sẵn sàng cho AI được lưu tại: {output_file}")
print("\n--- 5 DÒNG DỮ LIỆU ĐẦU TIÊN ---")
# In ra để kiểm tra nhanh tính toàn vẹn của dữ liệu dạng bảng (tabular data)
print(df_1d[['created_at', 'price_close', 'total_volume', 'SMA_10', 'RSI_14', 'price_lag_1']].head())
