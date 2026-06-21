import requests,os,time
from datetime import datetime

def send(stock,ticker:str):
    webhook_URL = os.getenv("WEBHOOK")
    if not webhook_URL:
        print("Thiếu WEBHOOK URL")
        return

    # ⏱️ time
    time_str = datetime.fromtimestamp(stock.ts).strftime("%d/%m %H:%M:%S")

    # 📊 tính toán
    change = stock.priceClose - stock.priceReference
    pct = (change / stock.priceReference) * 100

    # 🎨 màu
    if stock.priceClose > stock.priceReference:
        color = int("26ff3c", 16)  # xanh
    elif stock.priceClose < stock.priceReference:
        color = int("ff3c3c", 16)  # đỏ
    else:
        color = int("f1c40f", 16)  # vàng

    # 🚀 signal
    if pct > 1:
        signal = "🚀 BREAKOUT"
    elif pct < -1:
        signal = "🔻 DUMP"
    else:
        signal = "📊 NORMAL"

    # 📦 payload
    text = {
        "content": f"{signal} | VIX {stock.priceClose:.2f} ({pct:+.2f}%)",
        "embeds": [
            {
                "title": f"{signal} | VIX {stock.priceClose:.2f}",
                "color": color,
                "fields": [
                    {
                        "name": "📊 Change",
                        "value": f"{change:+.2f} ({pct:+.2f}%)",
                        "inline": False
                    },
                    {
                        "name": "📈 High",
                        "value": f"{stock.priceDayHigh:.2f}",
                        "inline": True
                    },
                    {
                        "name": "📉 Low",
                        "value": f"{stock.priceDayLow:.2f}",
                        "inline": True
                    },
                    {
                        "name": "⚖️ Ref",
                        "value": f"{stock.priceReference:.2f}",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": f"🕒 {time_str}"
                }
            }
        ]
    }

    try:
        res = requests.post(webhook_URL, json=text, timeout=5)
        if res.status_code != 204:
            print("Webhook lỗi:", res.status_code, res.text)
        return res
    except requests.RequestException as e:
        print("Lỗi gửi webhook:", e)
        
#Hàm Input
def info():
    while True:   
        os.system("cls" if os.name=="nt" else "clear")
        try:
            lineMax = int(input("Nhập số dòng muốn terminal hiển thị [≥1]:"))
            ky_vong = float(input("Nhập giá mục tiêu(Target price) mà bạn mong muốn:"))
            ky_vong = round(ky_vong,2)
            if lineMax >= 1:
                break
            else:
                print("Phải nhập số ≥ 1:")
                time.sleep(2)
                os.system("cls" if os.name=="nt" else "clear")
                continue
        except ValueError:
            print("Nhập sai")
            time.sleep(2)
            print("hãy nhập lại")
            time.sleep(2)
            os.system("cls" if os.name=="nt" else "clear")
            continue
        except EOFError:
            print("Hãy nhập số mong muốn:")
            time.sleep(3)
            os.system("cls" if os.name=="nt" else "clear")
            return