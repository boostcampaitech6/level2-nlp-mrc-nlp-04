from datetime import datetime
import pytz


def get_seoul_datetime_str():
    seoul_timezone = pytz.timezone('Asia/Seoul')
    now = datetime.now(seoul_timezone)
    return now.strftime("%m%d_%H:%M")
