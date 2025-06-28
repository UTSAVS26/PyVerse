from flask import Flask
from config import Config
from .routes import bp

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Register Blueprint
    app.register_blueprint(bp)
    
    return app