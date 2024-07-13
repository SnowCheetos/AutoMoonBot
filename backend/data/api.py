from yfinance import Tickers
from requests import Session
from typing import List, Dict, Any
from pyrate_limiter import Duration, RequestRate, Limiter
from requests_cache import CacheMixin, RedisCache, BaseCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket

from utils import Timing


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

    def _equity_intraday(
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

    def _equity_interday(
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
            f"function=NEWS_SENTIMENT"
            f"&apikey={self._api_key}"
            f"&tickers={symbol}"
            f"&time_from={start}"
            f"&time_to={end}"
            f"&sort=RELEVANCE"
            f"&limit=1000"
        )
        return self.make_request(url)





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
