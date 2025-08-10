from flask_restx import fields

from .extensions import api

analyze_model = api.model("Analyze", {
    "asset": fields.String(required=True),
    "max_articles": fields.Integer(required=False, default=10),
    "model_variant": fields.String(
        required=False,
        default="medium",
        enum=["light", "medium", "ensemble"],
        description="Which model to use: light | medium | ensemble"
    ),
})

bias_model = api.model("Bias", {
    "outlet": fields.String(required=True)
})