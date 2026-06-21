from engine.collector import fetch_data
from Core.models import model_stock
from Database.PostgreSQL import PostgresManager
from engine.notifier import send,info
from engine.realtime import get_next_action
import time
from rich.console import Console

console = Console()
#Hàm Run
def run():
    db = PostgresManager()
    last_save_time = 0
    last_prices = {"vix": None, "bsr": None, "vre": None}
    stocks = ["vix","bsr","vre"]
    while True:
        #Check Trading time
        should_run, sleep_time = get_next_action()
        if not should_run:
            console.print(f":pause_button: [dark_orange]Trading Suspended[/dark_orange]")
            time.sleep(sleep_time)
            continue
        
        #Loop for each stock
        now = time.time()
        for ticker in stocks:
            try:
                db.create_tables(ticker)
                raw_data = fetch_data(ticker)

                stock = model_stock(raw_data)
                stock.ticker = ticker

                db.save(stock,ticker)

                if stock.priceClose != last_prices[ticker] or now - last_save_time >= 30:
                    send(stock, ticker) 
                    print(f"[{ticker}] " + stock.printed())
                    last_prices[ticker] = stock.priceClose

            except Exception as e:
                console.print(f" [dark_orange]Lỗi khi xử lý mã {ticker}: {e}")

        #Wait 2s to next loop
        last_save_time = now
        time.sleep(2)

#Chạy chương trình
if __name__ == "__main__":
    run()