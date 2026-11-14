"""SEC EDGAR provider (filings, XBRL facts, issuer identity).

Why EDGAR first: every filing carries an *acceptance timestamp*, the moment
the SEC accepted it for public dissemination. That is a genuine knowledge
time, which almost no commercial fundamentals feed provides; vendor
fundamentals are typically keyed by period end or "as restated", both of
which leak.

Time handling
-------------
``acceptanceDateTime`` in the submissions API is formatted with a trailing
``Z`` but is widely reported to be US/Eastern wall-clock time. We interpret it
as ``America/New_York`` by default (``acceptance_tz``). If the true zone were
UTC, the Eastern interpretation is 4-5 hours *later*, so the default can only
make ``known_time`` conservative, never leak. ``dissemination_delay`` adds a
further safety margin.

Facts whose accession is not found in the submissions history fall back to
22:00 ET on their ``filed`` date (EDGAR stops accepting at 22:00 ET), an upper
bound on acceptance.

Usage requires a User-Agent with contact details per SEC policy; it is read
from the constructor or the ``SEC_USER_AGENT`` environment variable and never
invented. Requests are limited to 8/s (SEC's limit is 10/s).
"""

import datetime as dt
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa

from automoonbot.data.http import HttpClient
from automoonbot.data.providers.base import KnowledgePolicy, Provider, ProviderInfo
from automoonbot.data.schemas import ENTITIES, FACTS, FILINGS, IDENTIFIERS

NAME = "sec_edgar"
_HOSTS = frozenset({"data.sec.gov", "www.sec.gov"})
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UTC = dt.UTC


def entity_id(cik: int | str) -> str:
    """Canonical entity id for a CIK: ``cik:0000320193``."""
    s = str(cik).strip()
    if s.lower().startswith("cik:"):
        s = s[4:]
    if not s.isdigit() or len(s) > 10 or int(s) <= 0:
        raise ValueError(f"invalid CIK {cik!r}")
    return f"cik:{int(s):010d}"


def _cik10(cik: int | str) -> str:
    return entity_id(cik)[4:]


def _ns(t: dt.datetime) -> int:
    t = t.astimezone(_UTC)
    delta = t - dt.datetime(1970, 1, 1, tzinfo=_UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _date(s: str | None) -> dt.date | None:
    if not s:
        return None
    if not _DATE.match(s):
        raise ValueError(f"malformed date {s!r}")
    return dt.date.fromisoformat(s)


class EdgarProvider(Provider):
    info = ProviderInfo(
        name=NAME,
        datasets=frozenset({ENTITIES.name, IDENTIFIERS.name, FILINGS.name, FACTS.name}),
        policy=KnowledgePolicy.CONSERVATIVE,
        description="SEC EDGAR submissions and XBRL company facts with acceptance-time knowledge",
    )

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        acceptance_tz: str = "America/New_York",
        dissemination_delay: dt.timedelta = dt.timedelta(minutes=1),
        client: HttpClient | None = None,
    ) -> None:
        if client is None:
            ua = user_agent or os.environ.get("SEC_USER_AGENT")
            if not ua:
                raise ValueError(
                    "SEC EDGAR requires a User-Agent with contact details: pass user_agent= "
                    "or set SEC_USER_AGENT (e.g. 'Your Name your@email')"
                )
            client = HttpClient(ua, allowed_hosts=_HOSTS, rate_per_host=8.0, cache_dir=cache_dir)
        self.client = client
        self.acceptance_tz = ZoneInfo(acceptance_tz)
        self.delay_ns = int(dissemination_delay.total_seconds() * 1e9)
        if self.delay_ns < 0:
            raise ValueError("dissemination_delay must be non-negative")
        self._fetched_ns: int | None = None

    # ------------------------------------------------------------ raw access
    def _json(self, url: str) -> dict[str, Any]:
        r = self.client.get(url)
        self._fetched_ns = r.fetched_at_ns
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError(f"{url}: expected a JSON object")
        return data

    def submissions(self, cik: int | str) -> dict[str, Any]:
        """Submissions JSON with all paginated history merged into ``recent``."""
        base = self._json(f"https://data.sec.gov/submissions/CIK{_cik10(cik)}.json")
        recent = base["filings"]["recent"]
        merged: dict[str, list[Any]] = {k: list(v) for k, v in recent.items()}
        for page in base["filings"].get("files", []):
            name = page["name"]
            if not re.fullmatch(r"CIK\d{10}-submissions-\d{3}\.json", name):
                raise ValueError(f"unexpected submissions page name {name!r}")
            extra = self._json(f"https://data.sec.gov/submissions/{name}")
            for k in merged:
                merged[k].extend(extra.get(k, [""] * len(extra["accessionNumber"])))
        base["filings"]["recent"] = merged
        return base

    def companyfacts(self, cik: int | str) -> dict[str, Any]:
        return self._json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_cik10(cik)}.json")

    # --------------------------------------------------------- normalization
    def acceptance_ns(self, value: str) -> int:
        """Parse ``acceptanceDateTime`` under the configured zone policy."""
        s = value.strip()
        if s.endswith(("Z", "z")):
            s = s[:-1]
        local = dt.datetime.fromisoformat(s)
        if local.tzinfo is not None:
            local = local.replace(tzinfo=None)
        return _ns(local.replace(tzinfo=self.acceptance_tz))

    def _filed_fallback_ns(self, filed: dt.date) -> int:
        return _ns(dt.datetime.combine(filed, dt.time(22, 0), tzinfo=ZoneInfo("America/New_York")))

    def filings_table(self, sub: Mapping[str, Any]) -> pa.Table:
        eid = entity_id(sub["cik"])
        r = sub["filings"]["recent"]
        n = len(r["accessionNumber"])
        for col in ("form", "filingDate", "acceptanceDateTime"):
            if len(r[col]) != n:
                raise ValueError(f"submissions column {col!r} has inconsistent length")
        acc, accepted, filed, period, docs, forms = [], [], [], [], [], []
        for i in range(n):
            a = r["accessionNumber"][i]
            if not _ACCESSION.match(a):
                raise ValueError(f"malformed accession {a!r}")
            acc.append(a)
            accepted.append(self.acceptance_ns(r["acceptanceDateTime"][i]))
            filed.append(_date(r["filingDate"][i]))
            period.append(_date((r.get("reportDate") or [""] * n)[i] or None))
            docs.append((r.get("primaryDocument") or [None] * n)[i] or None)
            forms.append(r["form"][i])
        acc_ns = np.asarray(accepted, dtype=np.int64)
        return pa.table(
            {
                "accession": acc,
                "entity_id": [eid] * n,
                "form": forms,
                "filing_date": pa.array(filed, pa.date32()),
                "report_period": pa.array(period, pa.date32()),
                "accepted_at": pa.array(acc_ns, pa.int64()).cast(
                    FILINGS.schema.field("accepted_at").type
                ),
                "primary_document": docs,
                "known_time": pa.array(acc_ns + self.delay_ns, pa.int64()).cast(
                    FILINGS.schema.field("known_time").type
                ),
                "source": [NAME] * n,
            }
        )

    def entities_table(self, sub: Mapping[str, Any]) -> pa.Table:
        """The issuer becomes known with its first filing."""
        r = sub["filings"]["recent"]
        if not r["acceptanceDateTime"]:
            raise ValueError("issuer has no filings")
        first = min(self.acceptance_ns(x) for x in r["acceptanceDateTime"]) + self.delay_ns
        ts = ENTITIES.schema.field("known_time").type
        return pa.table(
            {
                "entity_id": [entity_id(sub["cik"])],
                "kind": ["company"],
                "name": [sub.get("name")],
                "valid_from": pa.array([first], pa.int64()).cast(ts),
                "valid_to": pa.array([None], ts),
                "known_time": pa.array([first], pa.int64()).cast(ts),
                "source": [NAME],
            }
        )

    def identifiers_table(self, sub: Mapping[str, Any], fetched_ns: int) -> pa.Table:
        """Current tickers. EDGAR publishes no ticker history, so a ticker is
        only asserted from the moment it was observed (``fetched_ns``)."""
        tickers = [t for t in sub.get("tickers", []) if t]
        ts = IDENTIFIERS.schema.field("known_time").type
        n = len(tickers)
        return pa.table(
            {
                "entity_id": [entity_id(sub["cik"])] * n,
                "scheme": ["ticker"] * n,
                "value": tickers,
                "valid_from": pa.array([fetched_ns] * n, pa.int64()).cast(ts),
                "valid_to": pa.array([None] * n, ts),
                "known_time": pa.array([fetched_ns] * n, pa.int64()).cast(ts),
                "source": [NAME] * n,
            }
        )

    def facts_table(self, facts: Mapping[str, Any], filings: pa.Table) -> pa.Table:
        eid = entity_id(facts["cik"])
        known_by_acc = dict(
            zip(
                filings.column("accession").to_pylist(),
                filings.column("known_time").cast(pa.int64()).to_pylist(),
                strict=True,
            )
        )
        cols: dict[str, list[Any]] = {k: [] for k in FACTS.schema.names}
        for taxonomy, concepts in facts.get("facts", {}).items():
            for concept, body in concepts.items():
                for unit, rows in body.get("units", {}).items():
                    for row in rows:
                        accn = row["accn"]
                        if not _ACCESSION.match(accn):
                            raise ValueError(f"malformed accession {accn!r}")
                        val = row["val"]
                        if not isinstance(val, int | float) or isinstance(val, bool):
                            continue  # non-numeric facts are not features
                        known = known_by_acc.get(accn)
                        if known is None:
                            filed = _date(row.get("filed"))
                            if filed is None:
                                raise ValueError(
                                    f"fact in {accn} has neither acceptance nor filed date"
                                )
                            known = self._filed_fallback_ns(filed) + self.delay_ns
                        cols["entity_id"].append(eid)
                        cols["accession"].append(accn)
                        cols["taxonomy"].append(taxonomy)
                        cols["concept"].append(concept)
                        cols["unit"].append(unit)
                        cols["period_start"].append(_date(row.get("start")))
                        cols["period_end"].append(_date(row["end"]))
                        cols["value"].append(float(val))
                        cols["fiscal_year"].append(row.get("fy"))
                        cols["fiscal_period"].append(row.get("fp"))
                        cols["form"].append(row.get("form"))
                        cols["known_time"].append(known)
                        cols["source"].append(NAME)
        table = pa.table(
            {
                **{
                    k: v
                    for k, v in cols.items()
                    if k not in ("known_time", "period_start", "period_end")
                },
                "period_start": pa.array(cols["period_start"], pa.date32()),
                "period_end": pa.array(cols["period_end"], pa.date32()),
                "known_time": pa.array(cols["known_time"], pa.int64()).cast(
                    FACTS.schema.field("known_time").type
                ),
            }
        )
        return table.select(FACTS.schema.names)

    # -------------------------------------------------------------- Provider
    def fetch(self, dataset: str, entities: Sequence[str]) -> Iterator[pa.Table]:
        for e in entities:
            sub = self.submissions(e)
            fetched = self._fetched_ns or 0
            if dataset == ENTITIES.name:
                yield self.entities_table(sub)
            elif dataset == IDENTIFIERS.name:
                yield self.identifiers_table(sub, fetched)
            elif dataset == FILINGS.name:
                yield self.filings_table(sub)
            elif dataset == FACTS.name:
                yield self.facts_table(self.companyfacts(e), self.filings_table(sub))
            else:
                raise ValueError(f"{NAME} does not provide {dataset!r}")
