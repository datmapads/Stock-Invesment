import pandas as pd
import pandas_ta as ta
from PostgreSQL import get_conn 

def load_data():
    with get_conn() as conn:
        df = pd.read_sql("""
        SELECT price, pricereference, pricedayhigh, pricedaylow, created_at
        FROM vix
        ORDER BY created_at ASC
        """, conn )
    return df

# ---------------------------------------------------------
# Phần có thể tách File
# ---------------------------------------------------------


print("Đang tải dữ liệu từ PostgreSQL...")
df = load_data()

#Time-based Features
df["created_at"] = pd.to_datetime(df["created_at"])
df.set_index("created_at",inplace= True)

#Nén dữ liệu thành OHLC
df_1m = df["price"].resample("1min").ohlc()
df_1m.dropna(inplace= True)#Drop NaN bị mất khi lưu data


# Technical Indicators (Chỉ báo kỹ thuật)
df_1m["SMA_10"] = ta.sma(length=10,close=df_1m["close"])
df_1m["RSI_14"] = ta.rsi(length=14,close=df_1m["close"])

# 3. Lag Features (Dữ liệu quá khứ)
# Lấy giá của 1 phút và 5 phút trước đó
df_1m["price_lag_1"] = df_1m["close"].shift(1)
df_1m["price_lag_5"] = df_1m["close"].shift(5)

# 4. Làm sạch dữ liệu
# Các dòng đầu tiên sẽ bị NaN (Not a Number) vì không có quá khứ để tính lag hoặc RSI.
# Ta bắt buộc phải xóa các dòng lỗi này trước khi đưa cho AI.
df_1m.dropna(inplace = True)
df_1m.reset_index(inplace=True)
# ---------------------------------------------------------
# LƯU DỮ LIỆU ĐỂ HUẤN LUYỆN AI
# ---------------------------------------------------------
output_file = "vix_features.csv"
df_1m.to_csv(output_file, index = False)

print(f"\n✅ Đã xử lý xong! Dữ liệu sẵn sàng cho AI được lưu tại: {output_file}")
print("\n--- 5 DÒNG DỮ LIỆU ĐẦU TIÊN ---")
# Chỉ in ra một số cột quan trọng để dễ nhìn
print(df_1m[['created_at', 'close', 'SMA_10', 'RSI_14', 'price_lag_1']].head())