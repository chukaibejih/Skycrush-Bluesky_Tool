from flask import Blueprint, render_template, request
from sqlalchemy import func
from datetime import datetime, timedelta
from app.auth import login_required
from app.models import db, AppMetrics, CompatibilityCheck, ShareMetrics
from app import db

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/dashboard')
@login_required
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


    # Popular handles
    popular_handles = db.session.query(
        CompatibilityCheck.handle2,
        func.count(CompatibilityCheck.id).label('count'),
        func.max(CompatibilityCheck.timestamp).label('last_check')
    ).group_by(
        CompatibilityCheck.handle2
    ).filter(
        CompatibilityCheck.handle2.isnot(None)  # Ensure handle2 is not null
    ).order_by(
        func.count(CompatibilityCheck.id).desc()
    ).limit(10).all()

    recent_checks = CompatibilityCheck.query.order_by(
    CompatibilityCheck.timestamp.desc()
    ).limit(10).all()

    # Calculate processing time distribution
    processing_time_buckets = [0, 1, 2, 3, 4, 5]
    processing_time_distribution = []

    for i in range(len(processing_time_buckets)):
        lower_bound = processing_time_buckets[i]
        upper_bound = processing_time_buckets[i + 1] if i < len(processing_time_buckets) - 1 else None
        
        query = db.session.query(func.count(CompatibilityCheck.id))
        
        if upper_bound is None:
            # For the last bucket (5+ seconds)
            count = query.filter(
                CompatibilityCheck.timestamp > start_date,
                CompatibilityCheck.processing_time >= lower_bound
            ).scalar()
        else:
            # For all other buckets
            count = query.filter(
                CompatibilityCheck.timestamp > start_date,
                CompatibilityCheck.processing_time >= lower_bound,
                CompatibilityCheck.processing_time < upper_bound
            ).scalar()
        
        processing_time_distribution.append(count or 0)


    return render_template(
        'dashboard.html',
        total_checks=total_checks,
        successful_checks=successful_checks,
        avg_processing_time=avg_processing_time,
        checks_24h=checks_24h,
        total_shares=total_shares,
        activity_labels=activity_labels,
        activity_data=activity_data,
        recent_checks=recent_checks,
        processing_time_distribution=processing_time_distribution,
        popular_handles=popular_handles,
        selected_range=time_range
    )