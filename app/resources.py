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


def _build_service(ticker: str, max_articles: int, variant: str):
    model = load_model_variant(variant)
    return SentimentAnalysis(
        ticker=ticker,
        model=model,
        max_articles=max_articles,
        ensemble=(variant == "ensemble"),
    )


# ---- Namespaces ----
api = Namespace("News Headline Sentiment Analysis")


# ---- Endpoints ----
# Supports:
#   /ticker/TSLA
#   /ticker/TSLA?max_articles=25&model_variant=ensemble
#   /ticker/TSLA/25
#   /ticker/TSLA/25/ensemble
@api.route(
    "/ticker/<string:ticker>",
    "/ticker/<string:ticker>/<int:max_articles>",
    "/ticker/<string:ticker>/<int:max_articles>/<string:variant>",
)
class TickerSentimentAPI(Resource):
    """Returns ticker-specific sentiment analysis (GET)"""

    @api.response(200, "Successfully Retrieved Sentiment Analysis")
    @api.response(400, "Failed to Retrieve Sentiment Analysis")
    def get(self, ticker, max_articles=None, variant=None):
        try:
            # Allow query-string overrides or defaults if not in the path
            if max_articles is None:
                max_articles = request.args.get(
                    "max_articles", config.MAX_ARTICLE_COUNT, type=int
                )
            if variant is None:
                # accept both 'model_variant' (old name) and 'variant'
                variant = request.args.get("model_variant") or request.args.get("variant") or "medium"

            # Basic validation for variant
            if variant not in config.MODEL_VARIANTS:
                return {
                    "success": 0,
                    "body": f"Invalid model_variant '{variant}'. Choose one of {sorted(config.MODEL_VARIANTS)}."
                }, 400

            svc = _build_service(
                ticker=ticker,
                max_articles=max_articles,
                variant=variant
            )

            result = asyncio.run(svc.analyze_sentiment())
            return {"success": 1, "body": result}, 200

        except Exception as e:
            return {"success": 0, "body": str(e)}, 400
