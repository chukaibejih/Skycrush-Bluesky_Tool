from flask import Blueprint, render_template, request
from sqlalchemy import func
from datetime import datetime, timedelta
from app.models import db, AppMetrics, CompatibilityCheck, ShareMetrics
from app import db

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/dashboard')
def dashboard():
    # Get time range from query params
    time_range = request.args.get('range', '24h')
    
    # Calculate date range
    now = datetime.utcnow()
    if time_range == '24h':
        start_date = now - timedelta(days=1)
    elif time_range == '7d':
        start_date = now - timedelta(days=7)
    elif time_range == '30d':
        start_date = now - timedelta(days=30)
    else:
        start_date = datetime.min

    # Basic metrics
    total_checks = CompatibilityCheck.query.count()
    successful_checks = CompatibilityCheck.query.filter_by(success=True).count()
    
    # Recent metrics
    checks_24h = CompatibilityCheck.query.filter(
        CompatibilityCheck.timestamp > now - timedelta(days=1)
    ).count()

    # Average processing time
    avg_processing_time = db.session.query(
        func.avg(CompatibilityCheck.processing_time)
    ).filter(
        CompatibilityCheck.timestamp > start_date
    ).scalar() or 0

    # Total shares
    total_shares = ShareMetrics.query.count()

    # Activity timeline data
    if time_range == '24h':
        interval = timedelta(hours=1)
        format_str = '%H:00'
    elif time_range == '7d':
        interval = timedelta(days=1)
        format_str = '%Y-%m-%d'
    else:
        interval = timedelta(days=1)
        format_str = '%Y-%m-%d'

    activity_data = []
    activity_labels = []
    current = start_date
    while current <= now:
        next_period = current + interval
        count = CompatibilityCheck.query.filter(
            CompatibilityCheck.timestamp >= current,
            CompatibilityCheck.timestamp < next_period
        ).count()
        activity_data.append(count)
        activity_labels.append(current.strftime(format_str))
        current = next_period

    # Share distribution
    share_distribution = db.session.query(
        ShareMetrics.share_type,
        func.count(ShareMetrics.id)
    ).group_by(ShareMetrics.share_type).all()
    
    share_labels = [item[0] for item in share_distribution]
    share_data = [item[1] for item in share_distribution]

    # Popular handles
    popular_handles = db.session.query(
        CompatibilityCheck.handle2,
        func.count(CompatibilityCheck.id).label('count'),
        func.max(CompatibilityCheck.timestamp).label('last_check')
    ).group_by(
        CompatibilityCheck.handle1
    ).order_by(
        func.count(CompatibilityCheck.id).desc()
    ).limit(10).all()

    # Recent errors
    recent_errors = db.session.query(
        CompatibilityCheck.error_message.label('error_type'),
        func.count(CompatibilityCheck.id).label('count'),
        func.max(CompatibilityCheck.timestamp).label('timestamp')
    ).filter(
        CompatibilityCheck.success == False,
        CompatibilityCheck.timestamp > start_date
    ).group_by(
        CompatibilityCheck.error_message
    ).order_by(
        func.count(CompatibilityCheck.id).desc()
    ).limit(5).all()

    return render_template(
        'dashboard.html',
        total_checks=total_checks,
        successful_checks=successful_checks,
        avg_processing_time=avg_processing_time,
        checks_24h=checks_24h,
        total_shares=total_shares,
        activity_labels=activity_labels,
        activity_data=activity_data,
        share_labels=share_labels,
        share_data=share_data,
        popular_handles=popular_handles,
        recent_errors=recent_errors,
        selected_range=time_range
    )