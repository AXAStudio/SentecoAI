# app/resources.py

import os
import asyncio
from functools import lru_cache

import tensorflow as tf
from flask import request
from flask_restx import Resource, Namespace

from .sentiment import SentimentAnalysis
from .api_models import analyze_model, bias_model
from .bias import get_bias

# Ensure custom layer is registered for deserialization (needed for ensemble.keras)
from .ensemble_helpers.make_ensemble import ProbToLogit  # noqa: F401

DEFAULT_ARTICLE_COUNT = 10

# ---- Model selection (light | medium | ensemble) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "light":    os.path.join(BASE_DIR, "models", "model_light.keras"),
    "medium":   os.path.join(BASE_DIR, "models", "model_medium.keras"),
    "ensemble": os.path.join(BASE_DIR, "models", "model_ensemble.keras"),
}
# resources.py
@lru_cache(maxsize=3)
def load_model_variant(variant: str):
    path = MODEL_PATHS.get(variant, MODEL_PATHS["medium"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return tf.keras.models.load_model(path, compile=False)  # ← add compile=False


def build_service(asset: str, max_articles: int, variant: str, requires_ticker: bool):
    model = load_model_variant(variant)
    return SentimentAnalysis(
        asset=asset,
        model=model,
        max_articles=max_articles,
        requires_ticker=requires_ticker,
        ensemble=(variant == "ensemble"),
    )

# ---- Namespaces ----
api = Namespace("Sentiment Analysis")
apib = Namespace("Bias Analysis")

# ---- Endpoints ----
@api.route('/ticker')
class TickerSentimentAPI(Resource):
    """Returns ticker-specific sentiment analysis"""
    @api.expect(analyze_model)
    @api.response(200, "Successfully Retrieved Sentiment Analysis")
    @api.response(400, "Failed to Retrieve Sentiment Analysis")
    def post(self):
        try:
            data = request.json or {}
            asset = data.get('asset')
            max_articles = data.get('max_articles', DEFAULT_ARTICLE_COUNT)
            variant = data.get('model_variant', 'medium')  # "light" | "medium" | "ensemble"

            svc = build_service(asset, max_articles, variant, requires_ticker=True)
            result = asyncio.run(svc.analyze_sentiment())
            return {'success': 1, 'body': result}, 200
        except Exception as e:
            return {'success': 0, 'body': str(e)}, 400

@api.route('/company')
class CompanySentimentAPI(Resource):
    """Returns company-name sentiment analysis (no ticker required)"""
    @api.expect(analyze_model)
    @api.response(200, "Successfully Retrieved Sentiment Analysis")
    @api.response(400, "Failed to Retrieve Sentiment Analysis")
    def post(self):
        try:
            data = request.json or {}
            asset = data.get('asset')
            max_articles = data.get('max_articles', DEFAULT_ARTICLE_COUNT)
            variant = data.get('model_variant', 'medium')

            svc = build_service(asset, max_articles, variant, requires_ticker=False)
            result = asyncio.run(svc.analyze_sentiment())
            return {'success': 1, 'body': result}, 200
        except Exception as e:
            return {'success': 0, 'body': str(e)}, 400

@api.route('/general')
class GeneralSentimentAPI(Resource):
    """Returns general-topic sentiment analysis (no ticker required)"""
    @api.expect(analyze_model)
    @api.response(200, "Successfully Retrieved Sentiment Analysis")
    @api.response(400, "Failed to Retrieve Sentiment Analysis")
    def post(self):
        try:
            data = request.json or {}
            asset = data.get('asset')
            max_articles = data.get('max_articles', DEFAULT_ARTICLE_COUNT)
            variant = data.get('model_variant', 'medium')

            svc = build_service(asset, max_articles, variant, requires_ticker=False)
            result = asyncio.run(svc.analyze_sentiment())
            return {'success': 1, 'body': result}, 200
        except Exception as e:
            return {'success': 0, 'body': str(e)}, 400

@apib.route('')
class BiasAPI(Resource):
    """Returns outlet bias analysis"""
    @apib.expect(bias_model)
    @apib.response(200, "Successfully Retrieved Bias Analysis")
    @apib.response(400, "Failed to Retrieve Bias Analysis")
    def post(self):
        try:
            data = request.json or {}
            outlet = data.get('outlet')
            result = get_bias(outlet)
            return {'success': 1, 'body': result}, 200
        except Exception as e:
            return {'success': 0, 'body': str(e)}, 400
