from datetime import datetime
from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    from app.routes import bp as main_routes
    app.register_blueprint(main_routes)

    from app.dashboard import admin_bp
    app.register_blueprint(admin_bp)

    from app.auth import auth_bp
    app.register_blueprint(auth_bp)

    @app.template_filter('datetime')
    def datetime_filter(value):
        if isinstance(value, str):
            # If it's a string, parse it
            value = datetime.fromisoformat(value)
        # If it's already a datetime object or after parsing, format it
        return value.strftime('%B %d, %Y at %I:%M %p')


    return app
