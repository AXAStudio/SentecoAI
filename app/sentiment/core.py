"""
Sentiment Analysis Core Module
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import io
import math
import aiohttp

import numpy as np
import yfinance as yf
import xml.etree.ElementTree as ET

from time import perf_counter
from urllib.parse import quote_plus
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

from app.models import TFModel
from app.utils.request_parsing import (
    norm_title,
    to_iso_utc,
    parse_domain
)


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

NEUTRAL_BAND = 0.2


class SentimentAnalysis:
    def __init__(
        self,
        ticker: str,
        model: Any,
        max_articles: int = 10,
        ensemble: bool = False,
    ):
        self.ticker = ticker.upper()
        self.model = TFModel(model, is_ensemble=ensemble)
        self.max_articles = max_articles
        self.ensemble = ensemble

    async def get_news_from_ticker_fast(
        self, ticker: str, days: int = 3, want: int = 60, timeout_s: float = 1.2
    ):
        """Ultra-low-latency single Google News RSS pull."""
        recent = f"+when:{days}d"
        rss_url = (
            f"https://news.google.com/rss/search?q={quote_plus(ticker)}{recent}"
            "&hl=en-US&gl=US&ceid=US:en"
        )

        try:
            session_params = dict(
                timeout=aiohttp.ClientTimeout(total=timeout_s),
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept-Encoding": "gzip, deflate, br"
                },
                connector=aiohttp.TCPConnector(ttl_dns_cache=300),
            )
            async with aiohttp.ClientSession(**session_params) as session:
                async with session.get(rss_url) as resp:
                    if resp.status != 200:
                        return []
                    xml_bytes = await resp.read()  # bytes, not str
        except Exception:
            return []

        # parse with iterparse (fast and streaming)
        out, seen = [], set()
        try:
            for _, elem in ET.iterparse(io.BytesIO(xml_bytes)):
                if elem.tag == "item":
                    title = (elem.findtext("title") or "").strip()
                    link = (elem.findtext("link") or "").strip()
                    date = (elem.findtext("pubDate") or "").strip()
                    if not (title and link):
                        continue

                    domain = parse_domain(link)
                    key = (norm_title(title), domain)
                    if key in seen:
                        continue
                    seen.add(key)

                    outlet = domain
                    src_el = elem.find("source")
                    if src_el is not None and (src_el.text or "").strip():
                        outlet = src_el.text.strip() or domain

                    out.append({
                        "headline": title,
                        "link": link,
                        "date": date,
                        "published_iso": to_iso_utc(date),
                        "outlet": outlet,
                        "source_domain": domain,
                    })
                    if len(out) >= want:
                        break
                    elem.clear()  # free memory early
        except Exception:
            return []

        # newest first
        out.sort(key=lambda a: a.get("published_iso") or "", reverse=True)
        return out

    def bucket_from_score(self, s: float, band: float = NEUTRAL_BAND) -> str:
        if abs(s) <= band:
            return "neutral"
        return "positive" if s > 0 else "negative"
    
    def weighted_avg_conf_time(self, items, lam=0.03):
        """
        Confidence × time-decay weighted average over items that have:
        - item["score"] (float in [-1, 1])
        - item["confidence"] (float in [0, 1])
        - item["age"]["age_hours"] (float or None)
        lam: decay rate per HOUR. Example 0.03 ≈ 0.49 weight after 24h.
        """
        num = 0.0
        den = 0.0
        for a in items:
            s = a["score"]
            c = a["confidence"]
            ah = 0.0
            try:
                ah = float(a.get("age", {}).get("age_hours") or 0.0)
            except Exception:
                ah = 0.0
            w = c * math.exp(-lam * ah)
            num += s * w
            den += w
        return (num / den) if den > 0 else 0.0

    def find_oldest_article(self, iso_dates: List[Union[str, None]]) -> Union[str, None]:
        try:
            parsed = []
            for d in iso_dates:
                if not d:
                    continue
                # handle trailing Z
                parsed.append(datetime.fromisoformat(d.replace("Z", "+00:00")))
            if not parsed:
                return None
            days = (datetime.now(timezone.utc) - min(parsed)).days
            return f"{days} Days Ago"
        except Exception:
            return None

    def compute_sentiment_metrics(self, articles, neutral_band: float = NEUTRAL_BAND):
        if not articles:
            return {
                "score_distribution": {},
                "avg_confidence": None,
                "top_positive": None,
                "top_negative": None,
                "trend": []
            }

        scores = [a["score"] for a in articles]
        confidences = [a["confidence"] for a in articles]

        # bucket counts
        labels = [self.bucket_from_score(s, neutral_band) for s in scores]
        pos = sum(lbl == "positive" for lbl in labels)
        neu = sum(lbl == "neutral" for lbl in labels)
        neg = sum(lbl == "negative" for lbl in labels)

        dist = {"positive": pos, "negative": neg, "neutral": neu}

        # (optional) percentages
        n = len(articles)
        dist_pct = {k: (v / n) for k, v in dist.items()}

        avg_confidence = sum(confidences) / n

        # pick top ± excluding neutrals (fallback to None)
        pos_idxs = [i for i, lbl in enumerate(labels) if lbl == "positive"]
        neg_idxs = [i for i, lbl in enumerate(labels) if lbl == "negative"]

        top_pos = max(pos_idxs, key=lambda i: scores[i], default=None)
        top_neg = min(neg_idxs, key=lambda i: scores[i], default=None)

        top_positive = (
            {"headline": articles[top_pos]["headline"], "score": scores[top_pos]}
            if top_pos is not None else None
        )
        top_negative = (
            {"headline": articles[top_neg]["headline"], "score": scores[top_neg]}
            if top_neg is not None else None
        )

        # chronological trend (already ISO strings in your data)
        trend = [
            {"date": a["published_iso"], "score": a["score"]}
            for a in sorted(articles, key=lambda x: x["published_iso"] or "")
        ]

        return {
            "score_distribution": dist,
            "score_distribution_pct": dist_pct,   # optional but handy
            "avg_confidence": avg_confidence,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "trend": trend,
            "neutral_band": neutral_band,         # expose for transparency
        }

    async def analyze_sentiment(self) -> Dict[str, Any]:
        """
        Fast end-to-end pipeline:
        1) resolve (name, ticker)
        2) fetch many headlines via Google News RSS (ticker-driven)
        3) score with TF model
        4) cosine-dedupe on raw titles
        5) return top `max_articles`
        """
        start_time = perf_counter()

        asset_name = yf.Ticker(self.ticker).info['shortName']
        if not asset_name:
            return {"error": f"Could not find asset: {self.asset}"}

        # pull plenty so that dedupe can still leave >= max_articles
        WANT_SIZE = max(self.max_articles * 2, self.max_articles + 4)

        try:
            t_fetch0 = perf_counter()
            raw = await self.get_news_from_ticker_fast(
                asset_name,
                days=3,
                want=WANT_SIZE,
                timeout_s=3.0,
            )
            t_fetch1 = perf_counter()
            news_retrieval_ms = round((t_fetch1 - t_fetch0) * 1000.0, 1)

        except Exception as e:
            return {"error": f"Failed to retrieve the news feed: {e.__class__.__name__}"}

        if not raw:
            return {"error": "Failed to retrieve the news feed."}

        # --- model inference on raw headlines ---
        titles = [it["headline"] for it in raw]
        probs  = np.asarray(self.model.predict(titles), dtype="float32").reshape(-1)
        scores = ((probs - 0.5) * 2.0).astype("float32")
        conf   = np.maximum(probs, 1.0 - probs).astype("float32")

        # --- enrich with metadata & ages ---
        now_utc = datetime.now(timezone.utc)
        enriched: List[Dict[str, Any]] = []
        for i, it in enumerate(raw):
            published_iso = it.get("published_iso")
            age_days = age_hours = None
            if published_iso:
                try:
                    dt_utc = datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
                    delta  = now_utc - dt_utc
                    age_days  = delta.days
                    age_hours = delta.total_seconds() / 3600.0
                except Exception:
                    pass

            enriched.append({
                "headline": it["headline"],                    # never modified
                "score": float(np.round(scores[i], 3)),
                "proba": float(np.round(probs[i], 6)),
                "confidence": float(np.round(conf[i], 6)),
                "date": it.get("date"),
                "published_iso": published_iso,
                "age": {
                    "age_days": age_days,
                    "age_hours": age_hours
                },
                "outlet": it.get("outlet"),
                "source_domain": it.get("source_domain"),
                "article_links": it.get("link"),
            })

        enriched.sort(key=lambda a: a.get("published_iso") or "", reverse=True)
        enriched = enriched[: self.max_articles]

        # --- summary + metrics ---
        fetched_at = datetime.now(timezone.utc).isoformat()
        data_map = {str(i): enriched[i] for i in range(len(enriched))}
        oldest = self.find_oldest_article([a.get("published_iso") for a in enriched])
        total_time = round((perf_counter() - start_time) * 1000.0, 1)

        metrics = self.compute_sentiment_metrics(
            enriched,
            neutral_band=NEUTRAL_BAND
        )

        avg_score_raw = float(
            np.mean([a["score"] for a in enriched]).round(3)
        ) if enriched else 0.0

        avg_score_weighted = float(
            round(self.weighted_avg_conf_time(enriched, lam=0.03), 3)
        ) if enriched else 0.0

        return {
            "asset_details": {
                "asset_name": asset_name,
                "asset_ticker": self.ticker
            },
            "response_ms": {
                "response_time_ms": total_time,
                "news_retrieval_ms": news_retrieval_ms
            },
            "model_variant": "ensemble" if self.ensemble else "single",
            "n_articles_found": len(enriched),
            "avg_score": avg_score_raw,
            "avg_score_weighted": avg_score_weighted,
            "weighting": {
                "time_decay_lambda_per_hour": 0.03
            },
            "fetched_at": fetched_at,
            "oldest_article_read": oldest,
            "metrics": metrics,
            "data": data_map,
        }
