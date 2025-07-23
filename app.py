# app.py

from flask import Flask
from flask_cors import CORS

# استيراد جميع Blueprints التي أنشأتها
from liveness_api import liveness_blueprint
from deepface.api.src.modules.core.routes import api_blueprint

app = Flask(__name__)
CORS(app)

# تسجيل الـ blueprints مع التطبيق (مع إمكانية تخصيص المسار الأساسي)
app.register_blueprint(liveness_blueprint, url_prefix="/api")
app.register_blueprint(api_blueprint, url_prefix="/api")

if __name__ == "__main__":
    # يمكنك تغيير المنفذ إذا أردت (8000 مثلاً)
    app.run(host="0.0.0.0", port=443, ssl_context=("mycert.pem", "mykey.pem"))
