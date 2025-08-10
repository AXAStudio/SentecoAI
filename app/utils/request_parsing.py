"""
Request Parsing Utilities
"""

import re

from typing import Union
from datetime import timezone
from urllib.parse import urlparse
from email.utils import parsedate_to_datetime


def norm_title(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip().lower())
    t = re.sub(r"[^a-z0-9 %$€£\-\.\,\:\;\?\!\(\)]", "", t)
    return t


def _canonical_host(u: str) -> str:
    try:
        d = urlparse(u).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""


def parse_domain(u: str) -> str:
    return _canonical_host(u)


def to_iso_utc(rfc822: str) -> Union[str, None]:
    try:
        dt = parsedate_to_datetime(rfc822)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None

