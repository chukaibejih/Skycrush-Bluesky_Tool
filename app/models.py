from datetime import datetime
from app import db
import numpy as np

class AppMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_agent = db.Column(db.String(255))
    ip_address = db.Column(db.String(45))
    page_viewed = db.Column(db.String(50))
    
class CompatibilityCheck(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    handle1 = db.Column(db.String(100))
    handle2 = db.Column(db.String(100))
    compatibility_score = db.Column(db.Float)
    processing_time = db.Column(db.Float)  # in seconds
    success = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.Text, nullable=True)

    @staticmethod
    def _convert_to_python_float(value):
        """Convert NumPy types to Python float"""
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return value

    def __init__(self, **kwargs):
        # Convert NumPy types to Python native types
        if 'compatibility_score' in kwargs:
            kwargs['compatibility_score'] = self._convert_to_python_float(kwargs['compatibility_score'])
        if 'processing_time' in kwargs:
            kwargs['processing_time'] = self._convert_to_python_float(kwargs['processing_time'])
        super().__init__(**kwargs)
    
class ShareMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    result_id = db.Column(db.String(100))
    share_type = db.Column(db.String(50))  # 'download' or 'social'
    platform = db.Column(db.String(50), nullable=True)  # if shared to social media