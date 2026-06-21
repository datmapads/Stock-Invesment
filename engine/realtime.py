from datetime import datetime
import time

morning_start = 9*60 #9:00 AM
morning_end = 11*60 + 30 #11:30 AM

afternoon_start = 13*60 #1:00 PM
afternoon_end = 15*60 #3:00 PM

def in_trading_time():
    now = datetime.now()
    current = now.hour * 60 + now.minute

    return (morning_start<=current <= morning_end) or (afternoon_start <= current <= afternoon_end)

def until_trading_time(hour,minute):
    now = datetime.now()
    #Transform to h:m:0:0, E.g: 11:30:5:5 -> 11:30:0:0
    target = now.replace(hour=hour,minute=minute,second=0,microsecond=0)

    # Adjust target to tomorrow if the current trading session has already passed
    from datetime import timedelta
    if target <= now:
        target += timedelta(days=1)

    return (target - now).total_seconds()

def get_next_action():
    if in_trading_time():
        return True, 0

    now = datetime.now()

    current_minutes = (now.hour * 60) + now.minute
    # Determine the start time of the next trading session
    if current_minutes < morning_start:
        # Before market open: wait until 09:00 today
        t = (9, 0)

    # nghỉ trưa → chờ 13h
    elif current_minutes < afternoon_start:
        t = (13, 0)

    # sau giờ → chờ ngày mai
    else:
        t = (9, 0)

    return False, until_trading_time(*t)
        