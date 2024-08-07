from yfinance import Tickers
from requests import Session
from typing import List, Dict, Any
from pyrate_limiter import Duration, RequestRate, Limiter
from requests_cache import CacheMixin, RedisCache, BaseCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket

from automoonbot.moonpy.utils import Timing


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
        interval_seconds = Timing.parse_interval(interval)
        return Limiter(RequestRate(max_requests, interval_seconds * Duration.SECOND))

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

    def _asset_intraday(
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

    def _asset_interday(
        self,
        symbol: str,
        interval: str,
        outputsize: str = "full",
    ) -> Dict[str, Any]:
        url = self._base_url + (
            f"function=TIME_SERIES_{interval}_ADJUSTED"
            f"&apikey={self._api_key}"
            f"&symbol={symbol}"
            f"&outputsize={outputsize}"
        )
        return self.make_request(url)

    def _news_sentiment(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> Dict[str, Any]:
        url = self._base_url + (
            "function=NEWS_SENTIMENT"
            "&sort=RELEVANCE"
            "&limit=1000"
            f"&apikey={self._api_key}"
            f"&tickers={symbol}"
            f"&time_from={start}"
            f"&time_to={end}"
        )
        return self.make_request(url)

    def _options_eod(self, symbol: str, date: str):
        url = self._base_url + (
            "function=HISTORICAL_OPTIONS"
            f"&apikey={self._api_key}"
            f"&symbol={symbol}"
            f"&date={date}"
        )
        return self.make_request(url)
    
    def get_symbols(self, company: str) -> Dict[str, Any]:
        url = self._base_url + (
            "function=SYMBOL_SEARCH"
            f"&keywords={company}"
            f"&apikey={self._api_key}"
        )
        return self.make_request(url)

    def get_company(self, symbol: str) -> Dict[str, Any]:
        url = self._base_url + (
            "function=OVERVIEW"
            f"&symbol={symbol}"
            f"&apikey={self._api_key}"
        )
        return self.make_request(url)