from .PostgreSQL import get_conn

def fetch_raw(limit=1000):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT price, created_at
            FROM vix
            ORDER BY created_at DESC 
            LIMIT %s
""",(limit,))
            return cur.fetchall()[::-1]

def ohlc_func(rows):
    candles = {}
    for price,ts in rows:
        minute = ts.replace(second=0,microsecond=0)
    
        if minute not in candles:
            candles[minute] = {
                "open":price,
                "high":price,
                "low":price,
                "close":price,
            }
        else:
            candles[minute]["high"]=max(candles[minute]["high"], price)
            candles[minute]["low"]=min(candles[minute]["low"], price)
            candles[minute]["close"] = price
    
    return candles

def print_ohlc(candles):
    for minute in sorted(candles):
        data = candles[minute]
        print(
            f"{minute.strftime('%H:%M')} | "
            f"O:{data['open']:.2f} "
            f"H:{data['high']:.2f} "
            f"L:{data['low']:.2f} "
            f"C:{data['close']:.2f}"
        )