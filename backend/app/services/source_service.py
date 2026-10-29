from __future__ import annotations

import ipaddress
import math
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

SOURCE_PATH = Path(__file__).resolve().parents[2] / "data" / "publisher_history.csv"
SOURCE_METHOD = "Beta(2, 2)-smoothed historical REAL-label share"
UNAVAILABLE_MESSAGE = "Credibility score unavailable — insufficient source data."


def _minimum_sample_size() -> int:
    try:
        return max(1, int(os.environ.get("SOURCE_MINIMUM_SAMPLE_SIZE", "30")))
    except ValueError:
        return 30


def _normalize_domain(value: object) -> str | None:
    candidate = str(value).strip()
    if not candidate:
        return None
    parsed = urlsplit(candidate if "://" in candidate else f"//{candidate}")
    if "://" in candidate and parsed.scheme.lower() not in {"http", "https"}:
        return None
    if parsed.username or parsed.password:
        return None
    try:
        host = (parsed.hostname or "").rstrip(".").lower().removeprefix("www.")
        host = host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        return None
    if not host or " " in host or "/" in host or "@" in host:
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return None
    if "." not in host or not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", host):
        return None
    return host


def _count(value: object) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _registry_rows() -> tuple[list[dict[str, object]], str | None]:
    if not SOURCE_PATH.exists():
        return [], "Publisher history registry is not configured."
    try:
        df = pd.read_csv(SOURCE_PATH)
    except (OSError, pd.errors.ParserError, UnicodeError):
        return [], "Publisher history registry could not be read."
    required = {"publisher", "real_articles", "fake_articles"}
    if not required.issubset(df.columns):
        return [], "Publisher history registry has an invalid schema."

    rows = []
    for row in df.to_dict(orient="records"):
        publisher = str(row.get("publisher", "")).strip()
        domain = _normalize_domain(publisher)
        real = _count(row.get("real_articles"))
        fake = _count(row.get("fake_articles"))
        if not publisher or not domain or real is None or fake is None:
            continue
        total = real + fake
        rows.append({"publisher": publisher, "domain": domain, "real_articles": real,
                     "fake_articles": fake, "total_articles": total})
    return rows, None


def _score_row(row: dict[str, object]) -> dict[str, object]:
    real = int(row["real_articles"])
    fake = int(row["fake_articles"])
    total = int(row["total_articles"])
    return {
        **row,
        "credibility_score": round((real + 2) / (total + 4) * 100, 1),
        "method": SOURCE_METHOD,
        "minimum_sample_size": _minimum_sample_size(),
        "provenance_note": "Registry-provided historical labels; provenance is not independently verified by this application.",
    }


def leaderboard() -> dict[str, object]:
    rows, error = _registry_rows()
    if error:
        return {"available": False, "message": error, "sources": []}
    threshold = _minimum_sample_size()
    scored = [_score_row(row) for row in rows if int(row["total_articles"]) >= threshold]
    scored.sort(key=lambda row: (row["credibility_score"], row["total_articles"]), reverse=True)
    message = ("Beta(2,2)-smoothed historical REAL-label share; only sources with at least "
               f"{threshold} records are listed. Registry provenance is not independently verified.")
    if not scored:
        message = f"No publisher has the minimum {threshold} historical records required for a score."
    return {"available": bool(scored), "message": message, "minimum_sample_size": threshold,
            "sources": scored[:100]}


def score_for_url(url: str | None) -> dict[str, object]:
    if not url:
        return {"available": False, "status": "source_unavailable", "message": UNAVAILABLE_MESSAGE}
    host = _normalize_domain(url)
    if not host:
        return {"available": False, "status": "source_unavailable", "message": UNAVAILABLE_MESSAGE}
    rows, error = _registry_rows()
    if error:
        return {"available": False, "status": "registry_unavailable", "domain": host,
                "message": UNAVAILABLE_MESSAGE}
    matches = [row for row in rows if host == row["domain"] or host.endswith(f".{row['domain']}")]
    if not matches:
        return {"available": False, "status": "source_not_found", "domain": host,
                "message": UNAVAILABLE_MESSAGE}
    row = max(matches, key=lambda match: len(str(match["domain"])))
    if int(row["total_articles"]) < _minimum_sample_size():
        return {"available": False, "status": "insufficient_history", **row,
                "minimum_sample_size": _minimum_sample_size(), "method": SOURCE_METHOD,
                "message": UNAVAILABLE_MESSAGE}
    return {"available": True, "status": "available", **_score_row(row),
            "message": "Historical indicator only; it does not change or verify this article's model prediction."}
