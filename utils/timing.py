import dateparser
import humanfriendly
import numpy as np
from typing import List
from dateutil.relativedelta import relativedelta


class Timing:
    _units = {
        60: "1m",
        300: "5m",
        900: "15m",
        1800: "30m",
        3600: "60m",
        86400: "1d",
        604800: "7d",
        2592000: "30d",
        7776000: "1q",
        31536000: "1y",
    }
    _seconds = np.array(list(_units.keys()), dtype=float)

    @classmethod
    def nearest_unit(cls, interval: str) -> str:
        seconds = humanfriendly.parse_timespan(interval)
        return cls._units[cls._seconds[np.abs(cls._seconds - seconds).argmin()]]

    @staticmethod
    def parse_interval(interval: str) -> int:
        if interval.isalpha():
            interval = "1" + interval
        return int(humanfriendly.parse_timespan(interval))

    @staticmethod
    def intraday(interval: str) -> bool:
        return humanfriendly.parse_timespan(interval) < 86400

    @staticmethod
    def get_all_months(start_time: str, end_time: str) -> List[str]:
        start = dateparser.parse(start_time)
        end = dateparser.parse(end_time)
        months = []
        current = start
        while current <= end:
            months.append(current.strftime("%Y-%m"))
            current += relativedelta(months=1)
        return months
