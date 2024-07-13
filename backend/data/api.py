import humanfriendly
from yfinance import Tickers
from requests import Session
from typing import List, Dict, Any
from pyrate_limiter import Duration, RequestRate, Limiter
from requests_cache import CacheMixin, RedisCache, BaseCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket

from backend.data import get_all_months


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    def __init__(
        self,
        base_url: str | None,
        api_key: str | None,
        rate_limit: str,
        cache_backend: BaseCache | None = None,
    ) -> None:
        super().__init__(
            limiter=self._create_limiter(rate_limit),
            bucket_class=MemoryQueueBucket,
            backend=cache_backend or RedisCache(),
            cache_control=True,
        )

        self._base_url = base_url
        self._api_key = api_key

    def _create_limiter(self, rate_limit: str) -> Limiter:
        rate, interval = rate_limit.split("/")
        max_requests = int(rate)
        interval_seconds = self._interval_to_seconds(interval)
        return Limiter(RequestRate(max_requests, interval_seconds * Duration.SECOND))

    def _interval_to_seconds(self, interval: str) -> int:
        if interval.isalpha():
            interval = "1" + interval
        return int(humanfriendly.parse_timespan(interval))

    def make_request(self, url: str, **kwargs) -> Dict[str, Any]:
        response = self.get(url, **kwargs)
        try:
            response_json = response.json()
        except ValueError:
            response_json = {
                "ok": False,
                "error_message": "Failed to parse JSON",
                "content": response.text,
            }
        response_json["ok"] = True
        if not response.ok:
            response_json["ok"] = False
            response_json["status_code"] = response.status_code
            response_json["error_message"] = response.reason
        return response_json


class AlphaVantage(CachedLimiterSession):
    def __init__(
        self,
        api_key: str,
        rate_limit: str,
        cache_backend=None,
    ) -> None:
        super().__init__(
            base_url="https://www.alphavantage.co/query?",
            api_key=api_key,
            rate_limit=rate_limit,
            cache_backend=cache_backend,
        )

    def intraday(
        self,
        symbol: str,
        interval: str,
        month: str,
        extended_hours: bool = True,
        outputsize: str = "full",
    ) -> Dict[str, Any] | None:
        url = self._base_url + (
            f"function=TIME_SERIES_INTRADAY"
            f"&apikey={self._api_key}"
            f"&symbol={symbol}"
            f"&interval={interval}"
            f"&month={month}"
            f"&extended_hours={extended_hours}"
            f"&outputsize={outputsize}"
        )
        return self.make_request(url)

    def get_prices(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
        extended_hours: bool = True,
    ) -> Dict[str, Any]:
        key = f"Time Series ({interval})"
        data = {"Meta Data": None, key: dict(), "Errors": None}
        months = get_all_months(start, end)
        for i, month in enumerate(reversed(months)):
            res = self.intraday(symbol, interval, month, extended_hours)
            if not res["ok"]:
                data["Errors"] = res["error_message"]
                break
            data[key].update({**res[key]})
            if i == len(months) - 1:
                data["Meta Data"] = res["Meta Data"]
        return data


class YahooFinance(Tickers, CachedLimiterSession):
    def __init__(
        self,
        tickers: List[str],
        cache_backend=None,
    ) -> None:
        CachedLimiterSession.__init__(
            base_url=None,
            api_key=None,
            rate_limit="2/5s",
            cache_backend=cache_backend,
        )
        Tickers.__init__(
            session=self,
            tickers=tickers,
        )

    def get_prices(self):
        pass

    def get_news(self):
        pass
