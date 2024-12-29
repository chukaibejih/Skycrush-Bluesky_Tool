from datetime import datetime
import os
from flask import Blueprint, flash, render_template, request, redirect, url_for, session, jsonify
from bluesky_api import BlueskyAPI
from compatibility import ImprovedCompatibilityAnalyzer
from threading import Thread
from uuid import uuid4
import time
from flask import request, jsonify
from app.models import db, AppMetrics, CompatibilityCheck, ShareMetrics
from flask import current_app, Flask
import base64
import requests

bp = Blueprint("main", __name__)

# Dictionary to store processing status
processing_jobs = {}
completed_results = {}

# Initialize BlueskyAPI with your credentials
bluesky = BlueskyAPI()
BLUESKY_USERNAME = os.getenv('BLUESKY_USERNAME')
BLUESKY_PASSWORD = os.getenv('BLUESKY_PASSWORD')

def process_compatibility(app: Flask, job_id: str, handle1: str, handle2: str):
    with app.app_context():
        start_time = time.time()
        # try:
        print(f"Starting process for job {job_id}")
        
        if not BLUESKY_USERNAME or not BLUESKY_PASSWORD:
            raise ValueError("Bluesky credentials not found in environment variables")
        
        # Login only if not already authenticated
        if not bluesky.auth_token:
            login_result = bluesky.login(BLUESKY_USERNAME, BLUESKY_PASSWORD)
            if not login_result['success']:
                raise Exception("Failed to authenticate with Bluesky")
        
        print(f"Fetching user data")
        processing_jobs[job_id]['status'] = 'fetching_data'
        user1_data = bluesky.fetch_user_data(handle1)
        user2_data = bluesky.fetch_user_data(handle2)
        
        print(f"Fetching posts")
        processing_jobs[job_id]['status'] = 'fetching_posts'
        user1_posts = bluesky.fetch_user_posts(handle1, limit=20)['feed']
        user2_posts = bluesky.fetch_user_posts(handle2, limit=20)['feed']
        
        print(f"Analyzing data")
        processing_jobs[job_id]['status'] = 'analyzing'
        analyzer = ImprovedCompatibilityAnalyzer()
        compatibility_result = analyzer.generate_compatibility_analysis(
            user1_data=user1_data,
            user2_data=user2_data,
            user1_posts=user1_posts,
            user2_posts=user2_posts
        )
        
        processing_time = time.time() - start_time
        
        check = CompatibilityCheck(
            handle1=handle1,
            handle2=handle2,
            compatibility_score=compatibility_result['metrics']['overall'],
            processing_time=processing_time,
            success=True
        )
        try:
            db.session.add(check)
            db.session.commit()
            print(f"Successfully saved check to database")
        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            db.session.rollback()
            processing_jobs[job_id]['error'] = f"Database error: {str(db_error)}"
            return
        
        processing_jobs[job_id]['status'] = 'complete'
        processing_jobs[job_id]['result'] = compatibility_result
            
        # except Exception as e:
        #     print(f"Error in process_compatibility: {str(e)}")
        #     processing_time = time.time() - start_time
            
        #     check = CompatibilityCheck(
        #         handle1=handle1,
        #         handle2=handle2,
        #         processing_time=processing_time,
        #         success=False,
        #         error_message=str(e)
        #     )
        #     try:
        #         db.session.add(check)
        #         db.session.commit()
        #         print(f"Successfully saved error check to database")
        #     except Exception as db_error:
        #         print(f"Database error while saving error: {str(db_error)}")
        #         db.session.rollback()
            
        #     processing_jobs[job_id]['status'] = 'error'
        #     processing_jobs[job_id]['error'] = str(e)

@bp.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        handle1 = request.form.get("handle1", "").strip()
        handle2 = request.form.get("handle2", "").strip()
        
        # Cleanup handles to ensure a single '@' prefix
        handle1 = f"@{handle1.lstrip('@')}" if handle1 else ""
        handle2 = f"@{handle2.lstrip('@')}" if handle2 else ""
        
        # Generate unique job ID
        job_id = str(uuid4())
        processing_jobs[job_id] = {'status': 'starting'}
        
        # Get the current application instance
        app = current_app._get_current_object()
        
        # Pass the app instance to the thread
        Thread(target=process_compatibility, args=(app, job_id, handle1, handle2)).start()
        
        return redirect(url_for("main.processing", job_id=job_id))
    return render_template("home.html")

@bp.route("/processing/<job_id>")
def processing(job_id):
    if job_id not in processing_jobs:
        flash("Invalid processing ID", "error")
        return redirect(url_for('main.home'))
    
    return render_template("processing.html", job_id=job_id)

@bp.route("/check-status/<job_id>")
def check_status(job_id):
    if job_id not in processing_jobs:
        return jsonify({'status': 'error', 'message': 'Invalid job ID'})
    
    job = processing_jobs[job_id]
    if job['status'] == 'complete':
        # Store the result before cleaning up the job
        result = job['result']
        result_id = str(uuid4())  # Generate a unique ID for the result
        completed_results[result_id] = {
            'result': result,
            'timestamp': datetime.now()
        }
        del processing_jobs[job_id]
        return jsonify({
            'status': 'complete',
            'redirect': url_for('main.results', result_id=result_id)
        })
    elif job['status'] == 'error':
        error = job['error']
        del processing_jobs[job_id]
        return jsonify({'status': 'error', 'message': error})
    
    return jsonify({'status': job['status']})

@bp.route("/results")
def results():
    try:
        result_id = request.args.get("result_id")
        
        if not result_id or result_id not in completed_results:
            flash("Invalid or expired result ID", "error")
            return redirect(url_for('main.home'))
        
        # Get the result and remove it from storage
        result_data = completed_results.pop(result_id)
        compatibility_result = result_data['result']
        
        # Convert avatars to base64
        def convert_image_to_base64(image_url):
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                return f"data:image/jpeg;base64,{base64.b64encode(response.content).decode('utf-8')}"
            except Exception as e:
                print(f"Error fetching image: {e}")
                return None
        
        compatibility_result['user1_avi'] = convert_image_to_base64(compatibility_result['user1_avi'])
        compatibility_result['user2_avi'] = convert_image_to_base64(compatibility_result['user2_avi'])

        category_icons = {
            'romantic': '❤️',
            'friendship': '🤝',
            'emotional': '💖',
            'communication': '💬',
            'values': '⚖️',
            'lifestyle': '🌍'
        }
        
        return render_template(
            "results.html",
            compatibility_result=compatibility_result,
            category_icons=category_icons
        )
        
    except Exception as e:
        flash(f"Error displaying results: {str(e)}", "error")
        return redirect(url_for('main.home'))