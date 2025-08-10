import math
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import asyncio
from typing import Union, Any, Tuple, Dict, List
from urllib.parse import quote_plus, urlparse
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import aiohttp
import numpy as np
import tensorflow as tf
from yahooquery import search
from bs4 import BeautifulSoup
from time import perf_counter
from collections import Counter


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

NEUTRAL_BAND = 0.2

# ============================= Utils =============================


def norm_title(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip().lower())
    t = re.sub(r"[^a-z0-9 %$€£\-\.\,\:\;\?\!\(\)]", "", t)
    return t

def canonical_host(u: str) -> str:
    try:
        d = urlparse(u).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def _domain(u: str) -> str:
    return canonical_host(u)

def _to_iso_utc(rfc822: str) -> Union[str, None]:
    try:
        dt = parsedate_to_datetime(rfc822)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None

def build_rss_urls(asset_name: str, ticker: Union[str, None], days: int = 3) -> List[str]:
    q_name   = quote_plus(asset_name)
    q_exact  = quote_plus(f"\"{asset_name}\"")
    q_recent = f"+when:{days}d"
    base     = "https://news.google.com/rss/search?q="
    suffix   = "&hl=en-US&gl=US&ceid=US:en"

    queries = [
        q_name + q_recent,       
        q_exact + q_recent,         
    ]
    if ticker:
        q_ticker = quote_plus(ticker)
        queries += [
            f"{q_ticker}+OR+{q_name}" + q_recent,
            f"{q_exact}+OR+{q_ticker}" + q_recent,
        ]
        for site in ["reuters.com", "bloomberg.com", "wsj.com", "cnbc.com", "financialpost.com", "ft.com"]:
            queries.append(f"{q_ticker}+site:{site}" + q_recent)

    return [f"{base}{q}{suffix}" for q in queries]

class Http:
    def __init__(self, limit=10, total_timeout=20):
        timeout = aiohttp.ClientTimeout(total=total_timeout)
        self._conn = aiohttp.TCPConnector(limit=limit, ttl_dns_cache=300)
        self._session = aiohttp.ClientSession(
            connector=self._conn, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )

    async def close(self):
        await self._session.close()
        await self._conn.close()

    async def get_text(self, url: str, retries=2) -> Union[str, None]:
        for i in range(retries + 1):
            try:
                async with self._session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.text()
            except Exception:
                pass
            await asyncio.sleep(0.4 * (i + 1))
        return None


async def fetch_all_rss_quick(http, feed_urls, want=80, timeout_s=7.0, per_feed_cap=30):
    async def fetch_one(u):
        try:
            html = await asyncio.wait_for(http.get_text(u), timeout=timeout_s)
            if not html:
                return []
            soup = BeautifulSoup(html, "xml")
            out = []
            for item in soup.find_all("item")[:per_feed_cap]:
                title = (item.title.text or "").strip()
                link  = (item.link.text  or "").strip()
                date  = (item.pubDate.text or "").strip()
                src   = (item.source.text.strip() if item.source else canonical_host(link)) or "Unknown Outlet"
                if title and link:
                    out.append((title, link, date, src))
            return out
        except Exception:
            return []

    tasks = [asyncio.create_task(fetch_one(u)) for u in feed_urls]
    results = []

    try:
        for coro in asyncio.as_completed(tasks, timeout=timeout_s + 1):
            batch = await coro
            results.extend(batch)
            if len(results) >= want:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)

    return results[:want]




def dedupe_items(items: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
    seen = set()
    deduped = []
    for title, link, date, src in items:
        key = (norm_title(title), canonical_host(link))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((title, link, date, src))
    return deduped

def parse_dates_safe(dates: List[str]) -> List[str]:
    out = []
    for d in dates:
        try:
            dt = parsedate_to_datetime(d)
            out.append(dt.strftime("%a, %d %b %Y %H:%M:%S %Z") or d)
        except Exception:
            out.append(d)
    return out

class TFModel:
    def __init__(self, model: Any, is_ensemble: bool = False):
        self._model = model
        self.is_ensemble = is_ensemble

        self._predict_fn = self._model.predict

    def predict(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        t = tf.constant(texts, dtype=tf.string)

        if self.is_ensemble:
            t = tf.reshape(t, (-1, 1))  
        else:
            t = tf.reshape(t, (-1,)) 

        return self._predict_fn(t, verbose=0)


class SentimentAnalysis:
    def __init__(
        self,
        asset: str,
        model: Any,
        max_articles: int = 10,
        requires_ticker: bool = False,
        ensemble: bool = False,
    ):
        self.asset = asset
        self.model = TFModel(model, is_ensemble=ensemble)
        self.max_articles = max_articles
        self._requiresTicker = requires_ticker
        self.ensemble = ensemble

    async def fetch_company_info(self) -> Tuple[Union[None, str], Union[None, str]]:
        try:
            results = search(self.asset.upper())
            quotes = results['quotes'][0]
            name = quotes.get('longname') or quotes.get('shortname')
            return name, quotes.get('symbol')
        except Exception:
            return (None, None) if self._requiresTicker else (self.asset, None)
    
    
    def _tokenize(self, s: str) -> List[str]:
        s = (s or "").lower()
        return re.findall(r"[a-z0-9]+", s)

    
    def _tfidf_vectors(self, texts: List[str]):
        """
        Returns:
        - vecs: List[Dict[str, float]] sparse tf-idf per doc
        - idf:  Dict[str, float]
        """
        docs = [Counter(self._tokenize(x)) for x in texts]
        N = len(docs)
        df = Counter()
        for d in docs:
            for term in d.keys():
                df[term] += 1
        idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}

        vecs = []
        for d in docs:
            v = {}
            # L2 normalize tf-idf
            for t, cnt in d.items():
                v[t] = cnt * idf[t]
            # length
            norm = math.sqrt(sum(w*w for w in v.values())) or 1.0
            for t in list(v.keys()):
                v[t] /= norm
            vecs.append(v)
        return vecs, idf
    
    def dedupe_by_title_cosine(self, enriched: List[Dict[str, Any]], thresh: float = 0.90) -> List[Dict[str, Any]]:
        """
        Collapse near-duplicate stories by cosine similarity on TF-IDF of the *raw* headline text.
        - Keeps items with higher (confidence, recency) first.
        - Never mutates the headline.
        - `thresh` is the cosine similarity threshold for treating two titles as the same story.
        """
        if not enriched:
            return enriched

        def _ts(a: Dict[str, Any]) -> float:
            iso = a.get("published_iso") or ""
            try:
                # handle possible "Z"
                return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0

        # Prefer higher-confidence and newer items when collapsing
        ordered = sorted(
            enriched,
            key=lambda a: (a.get("confidence", 0.0), _ts(a)),
            reverse=True
        )

        # Build TF-IDF vectors from the *raw* headlines (lowercasing happens in _tokenize)
        vecs, _ = self._tfidf_vectors([a.get("headline") or "" for a in ordered])


        kept_idxs: List[int] = []
        kept_vecs: List[Dict[str, float]] = []

        for i, v in enumerate(vecs):
            is_dup = False
            for kv in kept_vecs:
                if self._cosine_sparse(v, kv) >= thresh:
                    is_dup = True
                    break
            if not is_dup:
                kept_idxs.append(i)
                kept_vecs.append(v)

        kept = [ordered[i] for i in kept_idxs]
        # final order: most recent first (optional)
        kept.sort(key=lambda a: a.get("published_iso") or "", reverse=True)
        return kept





    def _cosine_sparse(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        # iterate smaller dict
        if len(a) > len(b):
            a, b = b, a
        return sum(a[t] * b.get(t, 0.0) for t in a)
    
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
        start_time = perf_counter()

        asset_name, ticker = await self.fetch_company_info()
        if not asset_name:
            return {"error": f"Could not find asset: {self.asset}"}

        rss_urls = build_rss_urls(asset_name, ticker, days=3)

        http = Http(limit=12, total_timeout=25)
        KEEP_MULT = 16
        KEEP_TARGET = max(self.max_articles * KEEP_MULT, 120)
        try:
            raw_items = await fetch_all_rss_quick(
                http, rss_urls,
                want=KEEP_TARGET,          # <-- was min(100, ...)
                timeout_s=7.0
            )
        finally:
            await http.close()

        if not raw_items:
            return {"error": "Failed to retrieve the news feed."}

        # Deduplicate and trim
        raw_items = dedupe_items(raw_items)   # tune this cap

        def _ts(it):
            _, _, date, _ = it
            try:
                return parsedate_to_datetime(date).timestamp()
            except Exception:
                return 0.0

        # keep plenty so the later cosine dedupe can still leave you >= max_articles
          # try 12–20 if sources are very samey
        
        items = sorted(raw_items, key=_ts, reverse=True)[:KEEP_TARGET]

        
        if not items:
            return {"error": "No news articles found"}


        all_news      = [t for (t, _, _, _) in items]
        article_links = [l for (_, l, _, _) in items]
        all_time_raw  = [d for (_, _, d, _) in items]
        all_outlets   = [s for (_, _, _, s) in items]
        all_time_norm = parse_dates_safe(all_time_raw)

        probs = np.asarray(self.model.predict(all_news), dtype="float32").reshape(-1) 
        scores = ((probs - 0.5) * 2.0).astype("float32")                              
        conf = np.maximum(probs, 1.0 - probs).astype("float32")
 
        enriched = []
        now_utc = datetime.now(timezone.utc)
        for i, (title, link, date_str, outlet) in enumerate(zip(all_news, article_links, all_time_norm, all_outlets)):
            published_iso = _to_iso_utc(date_str)
            age_days = age_hours = None
            try:
                if published_iso:
                    dt_utc = parsedate_to_datetime(date_str).astimezone(timezone.utc)
                    delta  = now_utc - dt_utc
                    age_days  = delta.days
                    age_hours = delta.total_seconds() / 3600.0
            except Exception:
                pass

            enriched.append({
                "headline": title,
                "score": float(np.round(scores[i], 3)),
                "proba": float(np.round(probs[i], 6)),
                "confidence": float(np.round(conf[i], 6)),
                "date": date_str,
                "published_iso": published_iso,
                "age" : {"age_days": age_days, "age_hours": age_hours},
                "outlet": outlet,
                "source_domain": _domain(link),
                "article_links": link,
            })

        enriched.sort(key=lambda x: x["published_iso"] or "", reverse=True)

        enriched = self.dedupe_by_title_cosine(enriched, thresh=0.96)

        # If still short, relax once
        if len(deduped) < self.max_articles:
            deduped = self.dedupe_by_title_cosine(enriched, thresh=0.985)

        # Final top-up by recency with simple uniqueness (no cosine)
        if len(deduped) < self.max_articles:
            seen_ids = {id(a) for a in deduped}
            for a in sorted(enriched, key=lambda x: x.get("published_iso") or "", reverse=True):
                if id(a) not in seen_ids:
                    deduped.append(a); seen_ids.add(id(a))
                if len(deduped) >= self.max_articles:
                    break

        enriched = sorted(deduped, key=lambda a: a.get("published_iso") or "", reverse=True)[: self.max_articles]

        enriched.sort(key=lambda a: a.get("published_iso") or "", reverse=True)
        enriched = enriched[: self.max_articles]


        fetched_at = datetime.now(timezone.utc).isoformat()
        data_map = {str(i): enriched[i] for i in range(len(enriched))}
        oldest = self.find_oldest_article([it["published_iso"] for it in enriched])
        total_time = round((perf_counter() - start_time) * 1000.0, 1)



        metrics = self.compute_sentiment_metrics(enriched, neutral_band=NEUTRAL_BAND)
        avg_score_raw = float(np.mean([a["score"] for a in enriched]).round(3))
        avg_score_weighted = float(round(self.weighted_avg_conf_time(enriched, lam=0.03), 3))
         
        return {
            "asset_details": {"asset_name": asset_name, "asset_ticker": ticker},
            "response_time_ms": total_time,
            "model_variant": "ensemble" if self.ensemble else "single",
            "n_articles_found": len(enriched),
            "avg_score": avg_score_raw,
            "avg_score_weighted": avg_score_weighted,
            "weighting": {"time_decay_lambda_per_hour": 0.03},
            "fetched_at": fetched_at,
            "oldest_article_read": oldest,
            "metrics": metrics,
            "data": data_map,
        }
