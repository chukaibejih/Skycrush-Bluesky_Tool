# In app/auth.py
from functools import wraps
import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from dotenv import load_dotenv

load_dotenv()

auth_bp = Blueprint('auth', __name__)

# Simple admin credentials - in production, use environment variables and password hashing
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('admin.dashboard'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))