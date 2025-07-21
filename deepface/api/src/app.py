# 3rd party dependencies
from flask import Flask
from flask_cors import CORS

# project dependencies
from deepface.api.src.modules.core.routes import api_blueprint  # تأكد من وجود api_blueprint في هذا الملف
from deepface.commons.logger import Logger
from deepface import DeepFace

logger = Logger()

def create_app():
    app = Flask(__name__)
    CORS(app)
    # تسجيل الـ blueprint مع المسار الأساسي /api
    app.register_blueprint(api_blueprint, url_prefix="/api")
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")
    return app

# فقط عند التشغيل المباشر (ليس عند استيراد الكود بواسطة gunicorn أو أي وركر)
if __name__ == "__main__":
    app = create_app()
    # يمكنك تغيير البورت لو أردت حسب إعدادات Render
    app.run(host="0.0.0.0", port=8000, debug=True)
else:
    # حتى تدعم منصات مثل gunicorn/uwsgi
    app = create_app()
