"""A small, careful HTTP client for public data APIs.

* Requires an explicit ``User-Agent`` (SEC EDGAR, among others, rejects
  anonymous clients and asks for a contact). It is never invented.
* Enforces a per-host request rate with a token bucket shared by threads.
* Retries transient failures (429/5xx, connection resets) with jittered
  exponential backoff, honoring ``Retry-After``.
* Caches successful responses on disk by URL, so re-running a pipeline does
  not re-download, and records the fetch time alongside each body.
* Only HTTPS URLs to an allow-listed set of hosts are fetched.
* Credentials go in request headers, never in URLs, so they are not part
  of cache keys, cached files or error messages.
"""

import gzip
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_TRANSIENT = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Response:
    url: str
    status: int
    body: bytes
    fetched_at_ns: int
    from_cache: bool

    def json(self) -> object:
        return json.loads(self.body)


class RateLimiter:
    """Token bucket: at most ``rate`` requests per second, bursts of ``burst``."""

    def __init__(self, rate: float, burst: int = 1) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be > 0 and burst >= 1")
        self.rate, self.burst = rate, float(burst)
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self.rate
            time.sleep(wait)


class HttpClient:
    def __init__(
        self,
        user_agent: str,
        *,
        allowed_hosts: frozenset[str],
        rate_per_host: float,
        cache_dir: str | os.PathLike[str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 5,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        ua = user_agent.strip()
        if len(ua) < 8 or "\n" in ua or "\r" in ua:
            raise ValueError(
                "a descriptive single-line User-Agent with contact details is required"
            )
        self.user_agent = ua
        self.allowed_hosts = allowed_hosts
        self.timeout, self.max_retries, self.max_bytes = timeout, max_retries, max_bytes
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._limiters = {h: RateLimiter(rate_per_host) for h in allowed_hosts}

    def _cache_paths(self, url: str) -> tuple[Path, Path] | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256(url.encode()).hexdigest()
        base = self.cache_dir / h[:2] / h
        return base.with_suffix(".body"), base.with_suffix(".meta.json")

    def get(
        self, url: str, *, use_cache: bool = True, headers: dict[str, str] | None = None
    ) -> Response:
        """Fetch ``url``. ``headers`` (e.g. ``Authorization``) are sent but
        are not part of the cache key."""
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in self.allowed_hosts:
            raise ValueError(
                f"refusing to fetch {url!r}: only https to {sorted(self.allowed_hosts)}"
            )
        paths = self._cache_paths(url)
        if use_cache and paths and paths[0].exists() and paths[1].exists():
            meta = json.loads(paths[1].read_text())
            return Response(url, meta["status"], paths[0].read_bytes(), meta["fetched_at_ns"], True)

        limiter = self._limiters[parts.hostname]
        attempt = 0
        while True:
            limiter.acquire()
            req = urllib.request.Request(
                url,
                headers={
                    **(headers or {}),
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "gzip",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read(self.max_bytes + 1)
                    if len(body) > self.max_bytes:
                        raise ValueError(f"{url}: response exceeds {self.max_bytes} bytes")
                    if resp.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    status = resp.status
                break
            except urllib.error.HTTPError as e:
                if e.code in _TRANSIENT and attempt < self.max_retries:
                    attempt += 1
                    time.sleep(self._backoff(attempt, e.headers.get("Retry-After")))
                    continue
                raise
            except urllib.error.URLError, ConnectionError, TimeoutError:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(self._backoff(attempt, None))
                    continue
                raise
        fetched = time.time_ns()
        if paths:
            paths[0].parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(paths[0], body)
            _atomic_write(
                paths[1], json.dumps({"status": status, "fetched_at_ns": fetched}).encode()
            )
        return Response(url, status, body, fetched, False)

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            try:
                wait: float = min(float(retry_after), 120.0)
                return wait
            except ValueError:
                pass
        return min(60.0, 0.5 * 2.0**attempt) * (0.5 + random.random())


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
