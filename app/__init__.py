"""
Flask app init
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

from flask import Flask
from .extensions import api
from .resources import api as ticker_sentiment_analysis_ns


sem_ver = 1.0
api_prefix = f"/api/{sem_ver}"


app = Flask(__name__)
api.init_app(app, title="SentecAI | Headline Sentiment Analysis API")
api.add_namespace(ticker_sentiment_analysis_ns, path=f"{api_prefix}/sentiment")
