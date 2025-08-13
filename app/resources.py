"""
Resources / Routes for Sentiment Analysis API
"""

import asyncio

from flask import request
from flask_restx import Resource, Namespace

from app import config
from app.models import load_model_variant
from app.sentiment import SentimentAnalysis

# Ensure custom layer is registered for deserialization (needed for ensemble.keras)
from app.utils.ensemble import ProbToLogit  # noqa: F401


def _build_service(
        ticker: str,
        max_articles: int,
        variant: str = "medium"
) -> SentimentAnalysis:
    model = load_model_variant(variant=variant)
    return SentimentAnalysis(
        ticker=ticker,
        model=model,
        max_articles=max_articles,
    )


# ---- Namespaces ----
api = Namespace("News Headline Sentiment Analysis")


# ---- Endpoints ----
# Supports:
#   /ticker/TSLA
#   /ticker/TSLA/25
@api.route(
    "/ticker/<string:ticker>",
    "/ticker/<string:ticker>/<int:max_articles>"
)
class TickerSentimentAPI(Resource):
    """Returns ticker-specific sentiment analysis (GET)"""

    @api.response(200, "Successfully Retrieved Sentiment Analysis")
    @api.response(400, "Failed to Retrieve Sentiment Analysis")
    def get(self, ticker, max_articles=None):
        """
        Get sentiment analysis for a given ticker symbol (Max-Articles: 100).
        """
        try:
            # Allow query-string overrides or defaults if not in the path
            if max_articles is None:
                max_articles = request.args.get(
                    "max_articles", config.DEFAULT_ARTICLE_COUNT, type=int
                )

            svc = _build_service(
                ticker=ticker,
                max_articles=max_articles
            )

            result = asyncio.run(svc.analyze_sentiment())
            return {"success": 1, "body": result}, 200

        except Exception as e:
            return {"success": 0, "body": str(e)}, 400
