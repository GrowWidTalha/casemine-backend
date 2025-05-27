from flask import Flask, request, jsonify, redirect
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import hashlib
import re
import jwt
from datetime import datetime, timedelta
from functools import wraps
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64
import uuid
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Database configuration
DATABASE_URL = "postgresql://saasdb_owner:hZm1Ql3RgJjs@ep-holy-voice-a2p4hd0z-pooler.eu-central-1.aws.neon.tech/saasdb?sslmode=require"

# Email configuration
EMAIL_CONFIG = {
    'EMAIL_HOST': 'smtp.gmail.com',
    'PORT': 587,
    'EMAIL_USER': 'mohsinshahid052@gmail.com',
    'EMAIL_PASS': 'migknvusxsbamoqm'
}

UPLOAD_FOLDER = './uploads/profile_images'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn

def setup_database():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            country VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            phone_number VARCHAR(20) NOT NULL,
            password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create blacklisted tokens table for logout functionality
    cur.execute('''
        CREATE TABLE IF NOT EXISTS blacklisted_tokens (
            id SERIAL PRIMARY KEY,
            token TEXT NOT NULL,
            blacklisted_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create contact messages table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS contact_messages (
            id SERIAL PRIMARY KEY,
            subject VARCHAR(255) NOT NULL,
            message TEXT NOT NULL,
            email VARCHAR(255),
            user_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id SERIAL PRIMARY KEY,
            user_id INTEGER UNIQUE NOT NULL,
            court_of_practice VARCHAR(255),
            organization VARCHAR(255),
            job_title VARCHAR(255),
            bio_graphy TEXT,
            profile_image_path VARCHAR(512),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            case_name VARCHAR(255) NOT NULL,
            client_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Create folders table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS folders (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            folder_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS case_documents (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            case_id INTEGER,
            folder_id INTEGER,
            document_name VARCHAR(255) NOT NULL,
            file_path VARCHAR(255) NOT NULL,
            file_type VARCHAR(100),
            file_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE,
            FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE CASCADE
    )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS attorney_profiles (
            id SERIAL PRIMARY KEY,
            user_id INTEGER UNIQUE NOT NULL,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            mobile_number VARCHAR(20) NOT NULL,
            show_mobile_number BOOLEAN DEFAULT FALSE,
            email VARCHAR(255) NOT NULL,
            show_email BOOLEAN DEFAULT FALSE,
            organization VARCHAR(255),
            website VARCHAR(255),
            total_experience FLOAT,
            consultation_fees FLOAT,
            job_title VARCHAR(255),
            education_degree VARCHAR(255),
            passing_year INTEGER,
            university_name VARCHAR(255),
            facebook_url VARCHAR(255),
            twitter_url VARCHAR(255),
            linkedin_url VARCHAR(255),
            address VARCHAR(255),
            city VARCHAR(100),
            province VARCHAR(100),
            postal_code VARCHAR(20),
            country VARCHAR(100),
            bio_graph TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Create table for practice areas
    cur.execute('''
        CREATE TABLE IF NOT EXISTS practice_areas (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            area_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Create table for courts of practice
    cur.execute('''
        CREATE TABLE IF NOT EXISTS practice_courts (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            court_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Create table for professional experience
    cur.execute('''
        CREATE TABLE IF NOT EXISTS professional_experience (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            job_title VARCHAR(255) NOT NULL,
            company_name VARCHAR(255) NOT NULL,
            from_year INTEGER NOT NULL,
            to_year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Create table for professional memberships
    cur.execute('''
        CREATE TABLE IF NOT EXISTS professional_memberships (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            membership_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Create table for languages known
    cur.execute('''
        CREATE TABLE IF NOT EXISTS languages_known (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            language_name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')


    conn.commit()
    cur.close()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def is_valid_phone(phone):
    pattern = r'^\+?[0-9]{10,15}$'
    return re.match(pattern, phone) is not None

def generate_token(user_id, email):
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(days=1)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def send_email(subject, message, from_email=None):
    # Create message container
    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG['EMAIL_USER']
    msg['To'] = EMAIL_CONFIG['EMAIL_USER']  # Sending to yourself
    msg['Subject'] = f"Contact Form: {subject}"

    # Add sender info to the message if available
    body = message
    if from_email:
        body = f"From: {from_email}\n\n{message}"

    # Attach the message to the email
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Setup the server
        server = smtplib.SMTP(EMAIL_CONFIG['EMAIL_HOST'], EMAIL_CONFIG['PORT'])
        server.starttls()  # Secure the connection
        server.login(EMAIL_CONFIG['EMAIL_USER'], EMAIL_CONFIG['EMAIL_PASS'])

        # Send the email
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Check if token is in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]

        if not token:
            return jsonify({'error': 'Token is missing'}), 401

        # Check if token is blacklisted
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM blacklisted_tokens WHERE token = %s", (token,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return jsonify({'error': 'Token has been revoked'}), 401

        try:
            # Decode token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            user_id = data['user_id']

            # Get user info
            cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
            current_user = cur.fetchone()
            cur.close()
            conn.close()

            if not current_user:
                return jsonify({'error': 'User not found'}), 401

        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(user_id, *args, **kwargs)

    return decorated

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_base64_image(base64_string, user_id):
    """Save a base64 image to the filesystem and return the path"""
    if not base64_string or ',' not in base64_string:
        return None

    # Extract the content type and the data
    format_info, base64_data = base64_string.split(',', 1)

    # Detect file extension from the format info
    if 'image/jpeg' in format_info:
        ext = 'jpg'
    elif 'image/png' in format_info:
        ext = 'png'
    elif 'image/gif' in format_info:
        ext = 'gif'
    else:
        # Default to jpg if we can't detect
        ext = 'jpg'

    try:
        # Decode the base64 data
        image_data = base64.b64decode(base64_data)

        # Generate a unique filename
        filename = f"profile_{user_id}_{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Write the image to file
        with open(filepath, 'wb') as f:
            f.write(image_data)

        return filepath
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

setup_database()

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json

    # Check if all required fields are present
    required_fields = ['first_name', 'last_name', 'country', 'email', 'phone_number', 'password']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    # Validate email format
    if not is_valid_email(data['email']):
        return jsonify({'error': 'Invalid email format'}), 400

    # Validate phone number format
    if not is_valid_phone(data['phone_number']):
        return jsonify({'error': 'Invalid phone number format'}), 400

    # Check password length
    if len(data['password']) < 8:
        return jsonify({'error': 'Password must be at least 8 characters long'}), 400

    # Hash the password
    hashed_password = hash_password(data['password'])

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check if email already exists
        cur.execute("SELECT id FROM users WHERE email = %s", (data['email'],))
        if cur.fetchone():
            return jsonify({'error': 'Email already registered'}), 409

        # Insert new user
        cur.execute('''
            INSERT INTO users (first_name, last_name, country, email, phone_number, password)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, email
        ''', (
            data['first_name'],
            data['last_name'],
            data['country'],
            data['email'],
            data['phone_number'],
            hashed_password
        ))

        new_user = cur.fetchone()
        cur.close()
        conn.close()

        # Generate auth token
        token = generate_token(new_user['id'], new_user['email'])

        return jsonify({
            'message': 'User registered successfully',
            'user_id': new_user['id'],
            'email': new_user['email'],
            'token': token
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json

    # Check if email and password are provided
    if 'email' not in data or 'password' not in data:
        return jsonify({'error': 'Email and password are required'}), 400

    hashed_password = hash_password(data['password'])

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute('''
            SELECT id, email, first_name, last_name
            FROM users
            WHERE email = %s AND password = %s
        ''', (data['email'], hashed_password))

        user = cur.fetchone()
        cur.close()
        conn.close()

        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401

        # Generate auth token
        token = generate_token(user['id'], user['email'])

        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name']
            },
            'token': token
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
@token_required
def logout(user_id):
    # Get token from request header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Invalid token format'}), 400

    token = auth_header.split(' ')[1]

    try:
        # Add token to blacklist
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('''
            INSERT INTO blacklisted_tokens (token)
            VALUES (%s)
        ''', (token,))

        conn.commit()
        cur.close()
        conn.close()

        # Return success and redirect information
        return jsonify({
            'message': 'Successfully logged out',
            'redirect_url': '/'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/change-password', methods=['POST'])
@token_required
def change_password(user_id):
    data = request.json

    # Check if current and new passwords are provided
    if 'current_password' not in data or 'new_password' not in data:
        return jsonify({'error': 'Current password and new password are required'}), 400

    # Validate new password length
    if len(data['new_password']) < 8:
        return jsonify({'error': 'New password must be at least 8 characters long'}), 400

    # Hash both passwords
    hashed_current_password = hash_password(data['current_password'])
    hashed_new_password = hash_password(data['new_password'])

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Verify current password
        cur.execute('''
            SELECT id
            FROM users
            WHERE id = %s AND password = %s
        ''', (user_id, hashed_current_password))

        user = cur.fetchone()

        if not user:
            cur.close()
            conn.close()
            return jsonify({'error': 'Current password is incorrect'}), 401

        # Update password
        cur.execute('''
            UPDATE users
            SET password = %s
            WHERE id = %s
        ''', (hashed_new_password, user_id))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Password updated successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/contact', methods=['POST'])
def contact():
    data = request.json

    # Check if subject and message are provided
    if 'subject' not in data or 'message' not in data:
        return jsonify({'error': 'Subject and message are required'}), 400

    # Get email from authenticated user or from request
    user_id = None
    user_email = None

    # Check if user is authenticated
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        try:
            # Decode token without checking if it's blacklisted (we allow contact even from logged out users)
            token_data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            user_id = token_data['user_id']
            user_email = token_data.get('email')
        except:
            pass  # Ignore token errors for contact form

    # Get email from request if not authenticated
    if not user_email and 'email' in data:
        user_email = data['email']

    try:
        # Save message to database
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('''
            INSERT INTO contact_messages (subject, message, email, user_id)
            VALUES (%s, %s, %s, %s)
        ''', (data['subject'], data['message'], user_email, user_id))

        conn.commit()
        cur.close()
        conn.close()

        # Send email
        email_sent = send_email(data['subject'], data['message'], user_email)

        if email_sent:
            return jsonify({
                'message': 'Your message has been sent successfully!'
            }), 200
        else:
            return jsonify({
                'message': 'Your message was saved but there was an issue sending the email. We will still process your request.',
                'warning': 'Email delivery issue'
            }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile', methods=['GET'])
@token_required
def get_profile(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get user basic info
        cur.execute('''
            SELECT id, email, first_name, last_name, created_at
            FROM users
            WHERE id = %s
        ''', (user_id,))

        user_data = cur.fetchone()

        if not user_data:
            cur.close()
            conn.close()
            return jsonify({'error': 'User not found'}), 404

        # Get profile info
        cur.execute('''
            SELECT court_of_practice, organization, job_title,
                   bio_graphy, profile_image_path, updated_at
            FROM profiles
            WHERE user_id = %s
        ''', (user_id,))

        profile_data = cur.fetchone()

        cur.close()
        conn.close()

        # Combine user and profile data
        result = dict(user_data)

        if profile_data:
            result.update(profile_data)

            # Convert profile image path to URL if exists
            if profile_data['profile_image_path']:
                # In a real app, you'd use a proper URL based on your hosting
                result['profile_image_url'] = f"/api/images/{os.path.basename(profile_data['profile_image_path'])}"

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile', methods=['PUT'])
@token_required
def update_profile(user_id):
    try:
        # Get form data from request
        # For a multi-part form with file uploads, use request.form and request.files
        # For a JSON request (without file uploads), use request.json

        form_data = {}

        # Check if content type is JSON
        if request.is_json:
            form_data = request.json
        else:
            # Handle form data
            form_data = request.form.to_dict()

        # Validate required fields
        required_fields = ['first_name', 'last_name']
        for field in required_fields:
            if field not in form_data or not form_data[field]:
                return jsonify({'error': f'{field} is required'}), 400

        # Optional fields with default empty values
        optional_fields = {
            'court_of_practice': '',
            'organization': '',
            'job_title': '',
            'bio_graphy': ''
        }

        # Set defaults for any missing optional fields
        for field, default in optional_fields.items():
            if field not in form_data:
                form_data[field] = default

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Update user basic info
        cur.execute('''
            UPDATE users
            SET first_name = %s, last_name = %s
            WHERE id = %s
            RETURNING id, email, first_name, last_name
        ''', (
            form_data['first_name'],
            form_data['last_name'],
            user_id
        ))

        updated_user = cur.fetchone()

        # Check if profile exists
        cur.execute('SELECT id FROM profiles WHERE user_id = %s', (user_id,))
        profile_exists = cur.fetchone() is not None

        # Process profile image if present
        profile_image_path = None

        # Check for base64 image data
        if 'profile_image' in form_data and form_data['profile_image'] and form_data['profile_image'].startswith('data:image/'):
            profile_image_path = save_base64_image(form_data['profile_image'], user_id)

        # Check for file upload (multipart/form-data)
        elif request.files and 'profile_image' in request.files:
            file = request.files['profile_image']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(f"profile_{user_id}_{uuid.uuid4().hex}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                profile_image_path = filepath

        # Update or insert profile
        if profile_exists:
            if profile_image_path:
                # Update with new image
                cur.execute('''
                    UPDATE profiles
                    SET court_of_practice = %s, organization = %s, job_title = %s,
                        bio_graphy = %s, profile_image_path = %s, updated_at = NOW()
                    WHERE user_id = %s
                    RETURNING id
                ''', (
                    form_data['court_of_practice'],
                    form_data['organization'],
                    form_data['job_title'],
                    form_data['bio_graphy'],
                    profile_image_path,
                    user_id
                ))
            else:
                # Update without changing image
                cur.execute('''
                    UPDATE profiles
                    SET court_of_practice = %s, organization = %s, job_title = %s,
                        bio_graphy = %s, updated_at = NOW()
                    WHERE user_id = %s
                    RETURNING id
                ''', (
                    form_data['court_of_practice'],
                    form_data['organization'],
                    form_data['job_title'],
                    form_data['bio_graphy'],
                    user_id
                ))
        else:
            # Insert new profile
            cur.execute('''
                INSERT INTO profiles (user_id, court_of_practice, organization, job_title,
                                     bio_graphy, profile_image_path)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (
                user_id,
                form_data['court_of_practice'],
                form_data['organization'],
                form_data['job_title'],
                form_data['bio_graphy'],
                profile_image_path
            ))

        # Get the updated profile data
        cur.execute('''
            SELECT court_of_practice, organization, job_title,
                   bio_graphy, profile_image_path, updated_at
            FROM profiles
            WHERE user_id = %s
        ''', (user_id,))

        profile_data = cur.fetchone()

        conn.commit()
        cur.close()
        conn.close()

        # Combine user and profile data for response
        result = dict(updated_user)
        if profile_data:
            result.update(profile_data)

            # Convert profile image path to URL if exists
            if profile_data['profile_image_path']:
                # In a real app, you'd use a proper URL based on your hosting
                result['profile_image_url'] = f"/api/images/{os.path.basename(profile_data['profile_image_path'])}"

        return jsonify({
            'message': 'Profile updated successfully',
            'user': result
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    return app.send_from_directory(app.config['UPLOAD_FOLDER'], filename)

############################## Attorney IQ Api #################################################
@app.route('/api/attorney/profile', methods=['POST'])
@token_required
def create_attorney_profile(user_id):
    data = request.json

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check if profile already exists
        cur.execute("SELECT id FROM attorney_profiles WHERE user_id = %s", (user_id,))
        if cur.fetchone():
            return jsonify({'error': 'Profile already exists for this user. Use PUT method to update.'}), 409

        # Insert main profile data
        cur.execute('''
            INSERT INTO attorney_profiles (
                user_id, first_name, last_name, mobile_number, show_mobile_number,
                email, show_email, organization, website, total_experience,
                consultation_fees, job_title, education_degree, passing_year,
                university_name, facebook_url, twitter_url, linkedin_url,
                address, city, province, postal_code, country, bio_graph
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        ''', (
            user_id,
            data.get('first_name', ''),
            data.get('last_name', ''),
            data.get('mobile_number', ''),
            data.get('show_mobile_number', False),
            data.get('email', ''),
            data.get('show_email', False),
            data.get('organization', ''),
            data.get('website', ''),
            data.get('total_experience', 0),
            data.get('consultation_fees', 0),
            data.get('job_title', ''),
            data.get('education_degree', ''),
            data.get('passing_year', None),
            data.get('university_name', ''),
            data.get('facebook_url', ''),
            data.get('twitter_url', ''),
            data.get('linkedin_url', ''),
            data.get('address', ''),
            data.get('city', ''),
            data.get('province', ''),
            data.get('postal_code', ''),
            data.get('country', ''),
            data.get('bio_graph', '')
        ))

        profile_id = cur.fetchone()['id']

        # Handle practice areas
        if 'practice_areas' in data and isinstance(data['practice_areas'], list):
            for area in data['practice_areas']:
                cur.execute(
                    "INSERT INTO practice_areas (user_id, area_name) VALUES (%s, %s)",
                    (user_id, area)
                )

        # Handle courts of practice
        if 'practice_courts' in data and isinstance(data['practice_courts'], list):
            for court in data['practice_courts']:
                cur.execute(
                    "INSERT INTO practice_courts (user_id, court_name) VALUES (%s, %s)",
                    (user_id, court)
                )

        # Handle professional experience
        if 'professional_experience' in data and isinstance(data['professional_experience'], list):
            for exp in data['professional_experience']:
                cur.execute(
                    "INSERT INTO professional_experience (user_id, job_title, company_name, from_year, to_year) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, exp.get('job_title', ''), exp.get('company_name', ''),
                     exp.get('from_year', 0), exp.get('to_year'))
                )

        # Handle professional memberships
        if 'professional_memberships' in data and isinstance(data['professional_memberships'], list):
            for membership in data['professional_memberships']:
                cur.execute(
                    "INSERT INTO professional_memberships (user_id, membership_name) VALUES (%s, %s)",
                    (user_id, membership)
                )

        # Handle languages known
        if 'languages_known' in data and isinstance(data['languages_known'], list):
            for language in data['languages_known']:
                cur.execute(
                    "INSERT INTO languages_known (user_id, language_name) VALUES (%s, %s)",
                    (user_id, language)
                )

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Attorney profile created successfully',
            'profile_id': profile_id
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attorney/profile', methods=['PUT'])
@token_required
def update_attorney_profile(user_id):
    data = request.json

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check if profile exists
        cur.execute("SELECT id FROM attorney_profiles WHERE user_id = %s", (user_id,))
        profile = cur.fetchone()

        if not profile:
            return jsonify({'error': 'Profile not found. Create one first.'}), 404

        # Update main profile data
        cur.execute('''
            UPDATE attorney_profiles SET
                first_name = %s,
                last_name = %s,
                mobile_number = %s,
                show_mobile_number = %s,
                email = %s,
                show_email = %s,
                organization = %s,
                website = %s,
                total_experience = %s,
                consultation_fees = %s,
                job_title = %s,
                education_degree = %s,
                passing_year = %s,
                university_name = %s,
                facebook_url = %s,
                twitter_url = %s,
                linkedin_url = %s,
                address = %s,
                city = %s,
                province = %s,
                postal_code = %s,
                country = %s,
                bio_graph = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s
        ''', (
            data.get('first_name', ''),
            data.get('last_name', ''),
            data.get('mobile_number', ''),
            data.get('show_mobile_number', False),
            data.get('email', ''),
            data.get('show_email', False),
            data.get('organization', ''),
            data.get('website', ''),
            data.get('total_experience', 0),
            data.get('consultation_fees', 0),
            data.get('job_title', ''),
            data.get('education_degree', ''),
            data.get('passing_year', None),
            data.get('university_name', ''),
            data.get('facebook_url', ''),
            data.get('twitter_url', ''),
            data.get('linkedin_url', ''),
            data.get('address', ''),
            data.get('city', ''),
            data.get('province', ''),
            data.get('postal_code', ''),
            data.get('country', ''),
            data.get('bio_graph', ''),
            user_id
        ))

        # Handle practice areas - delete and recreate
        if 'practice_areas' in data and isinstance(data['practice_areas'], list):
            cur.execute("DELETE FROM practice_areas WHERE user_id = %s", (user_id,))
            for area in data['practice_areas']:
                cur.execute(
                    "INSERT INTO practice_areas (user_id, area_name) VALUES (%s, %s)",
                    (user_id, area)
                )

        # Handle courts of practice - delete and recreate
        if 'practice_courts' in data and isinstance(data['practice_courts'], list):
            cur.execute("DELETE FROM practice_courts WHERE user_id = %s", (user_id,))
            for court in data['practice_courts']:
                cur.execute(
                    "INSERT INTO practice_courts (user_id, court_name) VALUES (%s, %s)",
                    (user_id, court)
                )

        # Handle professional experience - delete and recreate
        if 'professional_experience' in data and isinstance(data['professional_experience'], list):
            cur.execute("DELETE FROM professional_experience WHERE user_id = %s", (user_id,))
            for exp in data['professional_experience']:
                cur.execute(
                    "INSERT INTO professional_experience (user_id, job_title, company_name, from_year, to_year) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, exp.get('job_title', ''), exp.get('company_name', ''),
                     exp.get('from_year', 0), exp.get('to_year'))
                )

        # Handle professional memberships - delete and recreate
        if 'professional_memberships' in data and isinstance(data['professional_memberships'], list):
            cur.execute("DELETE FROM professional_memberships WHERE user_id = %s", (user_id,))
            for membership in data['professional_memberships']:
                cur.execute(
                    "INSERT INTO professional_memberships (user_id, membership_name) VALUES (%s, %s)",
                    (user_id, membership)
                )

        # Handle languages known - delete and recreate
        if 'languages_known' in data and isinstance(data['languages_known'], list):
            cur.execute("DELETE FROM languages_known WHERE user_id = %s", (user_id,))
            for language in data['languages_known']:
                cur.execute(
                    "INSERT INTO languages_known (user_id, language_name) VALUES (%s, %s)",
                    (user_id, language)
                )

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Attorney profile updated successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attorney/profile', methods=['GET'])
@token_required
def get_attorney_profile(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get main profile data
        cur.execute("SELECT * FROM attorney_profiles WHERE user_id = %s", (user_id,))
        profile = cur.fetchone()

        if not profile:
            return jsonify({'error': 'Profile not found'}), 404

        # Get practice areas
        cur.execute("SELECT area_name FROM practice_areas WHERE user_id = %s", (user_id,))
        practice_areas = [row['area_name'] for row in cur.fetchall()]

        # Get courts of practice
        cur.execute("SELECT court_name FROM practice_courts WHERE user_id = %s", (user_id,))
        practice_courts = [row['court_name'] for row in cur.fetchall()]

        # Get professional experience
        cur.execute('''
            SELECT job_title, company_name, from_year, to_year
            FROM professional_experience
            WHERE user_id = %s
            ORDER BY from_year DESC
        ''', (user_id,))
        professional_experience = cur.fetchall()

        # Get professional memberships
        cur.execute("SELECT membership_name FROM professional_memberships WHERE user_id = %s", (user_id,))
        memberships = [row['membership_name'] for row in cur.fetchall()]

        # Get languages known
        cur.execute("SELECT language_name FROM languages_known WHERE user_id = %s", (user_id,))
        languages = [row['language_name'] for row in cur.fetchall()]

        # Build complete profile
        complete_profile = dict(profile)
        complete_profile['practice_areas'] = practice_areas
        complete_profile['practice_courts'] = practice_courts
        complete_profile['professional_experience'] = professional_experience
        complete_profile['professional_memberships'] = memberships
        complete_profile['languages_known'] = languages

        cur.close()
        conn.close()

        return jsonify(complete_profile), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attorney/profile/<int:attorney_id>', methods=['GET'])
def get_public_attorney_profile(attorney_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute('''
            SELECT
                ap.id, ap.first_name, ap.last_name, ap.organization, ap.job_title,
                ap.total_experience, ap.consultation_fees, ap.city, ap.country,
                ap.bio_graph, ap.website,
                CASE WHEN ap.show_email THEN ap.email ELSE NULL END as email,
                CASE WHEN ap.show_mobile_number THEN ap.mobile_number ELSE NULL END as mobile_number
            FROM attorney_profiles ap
            WHERE ap.user_id = %s
        ''', (attorney_id,))

        profile = cur.fetchone()

        if not profile:
            return jsonify({'error': 'Profile not found'}), 404

        cur.execute("SELECT area_name FROM practice_areas WHERE user_id = %s", (attorney_id,))
        practice_areas = [row['area_name'] for row in cur.fetchall()]

        cur.execute("SELECT court_name FROM practice_courts WHERE user_id = %s", (attorney_id,))
        practice_courts = [row['court_name'] for row in cur.fetchall()]

        cur.execute('''
            SELECT job_title, company_name, from_year, to_year
            FROM professional_experience
            WHERE user_id = %s
            ORDER BY from_year DESC
        ''', (attorney_id,))
        professional_experience = cur.fetchall()

        cur.execute("SELECT language_name FROM languages_known WHERE user_id = %s", (attorney_id,))
        languages = [row['language_name'] for row in cur.fetchall()]

        # Build complete profile for public view
        complete_profile = dict(profile)
        complete_profile['practice_areas'] = practice_areas
        complete_profile['practice_courts'] = practice_courts
        complete_profile['professional_experience'] = professional_experience
        complete_profile['languages_known'] = languages

        # Add social media links if available
        if profile.get('facebook_url'):
            complete_profile['facebook_url'] = profile['facebook_url']
        if profile.get('twitter_url'):
            complete_profile['twitter_url'] = profile['twitter_url']
        if profile.get('linkedin_url'):
            complete_profile['linkedin_url'] = profile['linkedin_url']

        cur.close()
        conn.close()

        return jsonify(complete_profile), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attorney/search/document', methods=['POST'])
def search_attorneys_by_document():
    try:
        # Get the uploaded document and court name
        document = request.files.get('document')
        court_name = request.form.get('court_name', '')

        if not document:
            return jsonify({'error': 'No document provided'}), 400

        if not court_name:
            return jsonify({'error': 'Court name is required'}), 400

        # Process document (basic implementation - read text content)
        # In production, you'd use specific libraries for different file types
        try:
            document_text = document.read().decode('utf-8')
        except:
            document_text = ""  # If document can't be read as text

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Find attorneys whose practice courts match the given court name
        query = """
            SELECT DISTINCT
                ap.user_id, ap.first_name, ap.last_name, ap.organization,
                ap.job_title, ap.city, ap.country, ap.total_experience
            FROM attorney_profiles ap
            JOIN practice_courts pc ON ap.user_id = pc.user_id
            WHERE LOWER(pc.court_name) LIKE LOWER(%s)
        """

        cur.execute(query, (f"%{court_name}%",))
        results = cur.fetchall()

        # Get additional details for each attorney
        for attorney in results:
            # Get practice areas
            cur.execute("SELECT area_name FROM practice_areas WHERE user_id = %s", (attorney['user_id'],))
            attorney['practice_areas'] = [row['area_name'] for row in cur.fetchall()]

            # Get courts of practice
            cur.execute("SELECT court_name FROM practice_courts WHERE user_id = %s", (attorney['user_id'],))
            attorney['practice_courts'] = [row['court_name'] for row in cur.fetchall()]

        cur.close()
        conn.close()

        return jsonify({
            'attorneys': results,
            'count': len(results),
            'search_criteria': {
                'court_name': court_name,
                'document_name': document.filename if document else None
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attorney/search/text', methods=['POST'])
def search_attorneys_by_text():
    try:
        # Get text input and court name
        data = request.json

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        text = data.get('text', '')
        court_name = data.get('court_name', '')

        if not court_name:
            return jsonify({'error': 'Court name is required'}), 400

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Find attorneys whose practice courts match the given court name
        query = """
            SELECT DISTINCT
                ap.user_id, ap.first_name, ap.last_name, ap.organization,
                ap.job_title, ap.city, ap.country, ap.total_experience
            FROM attorney_profiles ap
            JOIN practice_courts pc ON ap.user_id = pc.user_id
            WHERE LOWER(pc.court_name) LIKE LOWER(%s)
        """

        cur.execute(query, (f"%{court_name}%",))
        results = cur.fetchall()

        # Get additional details for each attorney
        for attorney in results:
            # Get practice areas
            cur.execute("SELECT area_name FROM practice_areas WHERE user_id = %s", (attorney['user_id'],))
            attorney['practice_areas'] = [row['area_name'] for row in cur.fetchall()]

            # Get courts of practice
            cur.execute("SELECT court_name FROM practice_courts WHERE user_id = %s", (attorney['user_id'],))
            attorney['practice_courts'] = [row['court_name'] for row in cur.fetchall()]

        cur.close()
        conn.close()

        return jsonify({
            'attorneys': results,
            'count': len(results),
            'search_criteria': {
                'court_name': court_name,
                'text_provided': bool(text)
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

##########################################  Case and folder Api ##########################################

@app.route('/api/cases', methods=['POST'])
@token_required
def create_case(user_id):
    data = request.json

    # Check if required fields are present
    if 'case_name' not in data or 'client_name' not in data:
        return jsonify({'error': 'Case name and client name are required'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Insert new case
        cur.execute('''
            INSERT INTO cases (user_id, case_name, client_name)
            VALUES (%s, %s, %s)
            RETURNING id, case_name, client_name, created_at
        ''', (
            user_id,
            data['case_name'],
            data['client_name']
        ))

        new_case = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Case created successfully',
            'case': new_case
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cases', methods=['GET'])
@token_required
def get_cases(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get all cases for the user
        cur.execute('''
            SELECT id, case_name, client_name, created_at
            FROM cases
            WHERE user_id = %s
            ORDER BY created_at DESC
        ''', (user_id,))

        cases = cur.fetchall()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Cases retrieved successfully',
            'cases': cases
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cases/<int:case_id>', methods=['PUT'])
@token_required
def update_case(user_id, case_id):
    data = request.json

    # Check if required fields are present
    if 'case_name' not in data and 'client_name' not in data:
        return jsonify({'error': 'At least one field to update is required'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check if case exists and belongs to user
        cur.execute('''
            SELECT id FROM cases
            WHERE id = %s AND user_id = %s
        ''', (case_id, user_id))

        case = cur.fetchone()
        if not case:
            cur.close()
            conn.close()
            return jsonify({'error': 'Case not found or not authorized'}), 404

        # Build update query dynamically based on provided fields
        update_fields = []
        params = []

        if 'case_name' in data:
            update_fields.append("case_name = %s")
            params.append(data['case_name'])

        if 'client_name' in data:
            update_fields.append("client_name = %s")
            params.append(data['client_name'])

        # Complete the params and execute update
        params.append(case_id)
        params.append(user_id)

        query = f'''
            UPDATE cases
            SET {', '.join(update_fields)}
            WHERE id = %s AND user_id = %s
            RETURNING id, case_name, client_name, created_at
        '''

        cur.execute(query, params)
        updated_case = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Case updated successfully',
            'case': updated_case
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cases/<int:case_id>', methods=['DELETE'])
@token_required
def delete_case(user_id, case_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check if case exists and belongs to user
        cur.execute('''
            SELECT id FROM cases
            WHERE id = %s AND user_id = %s
        ''', (case_id, user_id))

        case = cur.fetchone()
        if not case:
            cur.close()
            conn.close()
            return jsonify({'error': 'Case not found or not authorized'}), 404

        # Delete the case
        cur.execute('''
            DELETE FROM cases
            WHERE id = %s AND user_id = %s
        ''', (case_id, user_id))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Case deleted successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cases/<int:case_id>', methods=['GET'])
@token_required
def get_case(user_id, case_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get case details
        cur.execute('''
            SELECT id, case_name, client_name, created_at
            FROM cases
            WHERE id = %s AND user_id = %s
        ''', (case_id, user_id))

        case = cur.fetchone()

        if not case:
            cur.close()
            conn.close()
            return jsonify({'error': 'Case not found or not authorized'}), 404

        # Get folders for this case
        cur.execute('''
            SELECT id, folder_name, created_at
            FROM folders
            WHERE case_id = %s
            ORDER BY created_at
        ''', (case_id,))

        folders = cur.fetchall()

        cur.close()
        conn.close()

        # Add folders to case data
        case_data = dict(case)
        case_data['folders'] = folders

        return jsonify({
            'message': 'Case retrieved successfully',
            'case': case_data
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/folders', methods=['POST'])
@token_required
def create_folder(user_id):
    data = request.json

    # Check if folder name is present
    if 'folder_name' not in data:
        return jsonify({'error': 'Folder name is required'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Insert new folder (with just a name and user_id)
        cur.execute('''
            INSERT INTO folders (user_id, folder_name)
            VALUES (%s, %s)
            RETURNING id, folder_name, created_at
        ''', (
            user_id,
            data['folder_name']
        ))

        new_folder = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Folder created successfully',
            'folder': new_folder
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/folders', methods=['GET'])
@token_required
def get_folders(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get all folders for the user (without case relationship)
        cur.execute('''
            SELECT id, folder_name, created_at
            FROM folders
            WHERE user_id = %s
            ORDER BY created_at DESC
        ''', (user_id,))

        folders = cur.fetchall()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Folders retrieved successfully',
            'folders': folders
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/folders/<int:folder_id>', methods=['PUT'])
@token_required
def update_folder(user_id, folder_id):
    data = request.json

    # Check if folder name is present
    if 'folder_name' not in data:
        return jsonify({'error': 'Folder name is required'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check if folder exists and belongs to user
        cur.execute('''
            SELECT id FROM folders
            WHERE id = %s AND user_id = %s
        ''', (folder_id, user_id))

        folder = cur.fetchone()
        if not folder:
            cur.close()
            conn.close()
            return jsonify({'error': 'Folder not found or not authorized'}), 404

        # Update the folder
        cur.execute('''
            UPDATE folders
            SET folder_name = %s
            WHERE id = %s AND user_id = %s
            RETURNING id, folder_name, created_at
        ''', (data['folder_name'], folder_id, user_id))

        updated_folder = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Folder updated successfully',
            'folder': updated_folder
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/folders/<int:folder_id>', methods=['DELETE'])
@token_required
def delete_folder(user_id, folder_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check if folder exists and belongs to user
        cur.execute('''
            SELECT id FROM folders
            WHERE id = %s AND user_id = %s
        ''', (folder_id, user_id))

        folder = cur.fetchone()
        if not folder:
            cur.close()
            conn.close()
            return jsonify({'error': 'Folder not found or not authorized'}), 404

        # Delete the folder
        cur.execute('''
            DELETE FROM folders
            WHERE id = %s AND user_id = %s
        ''', (folder_id, user_id))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Folder deleted successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

############### document upload in folder API ###########################################

@app.route('/api/documents/upload', methods=['POST'])
@token_required
def upload_document(user_id):
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if it's an allowed file type
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Get parent container details (case or folder)
    parent_type = request.form.get('parent_type')  # 'case' or 'folder'
    parent_id = request.form.get('parent_id')  # ID of the case or folder

    if not parent_type or not parent_id or parent_type not in ['case', 'folder']:
        return jsonify({'error': 'Valid parent_type (case or folder) and parent_id are required'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Verify the parent container belongs to the user
        if parent_type == 'case':
            cur.execute('''
                SELECT id FROM cases
                WHERE id = %s AND user_id = %s
            ''', (parent_id, user_id))
        else:  # parent_type == 'folder'
            cur.execute('''
                SELECT id FROM folders
                WHERE id = %s AND user_id = %s
            ''', (parent_id, user_id))

        parent = cur.fetchone()
        if not parent:
            cur.close()
            conn.close()
            return jsonify({'error': f'{parent_type.capitalize()} not found or not authorized'}), 404

        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{user_id}_{parent_type}_{parent_id}_{filename}")
        file.save(file_path)

        # Get file size
        file_size = os.path.getsize(file_path)

        # Get file type
        file_type = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

        # Insert document record in database
        case_id = parent_id if parent_type == 'case' else None
        folder_id = parent_id if parent_type == 'folder' else None

        cur.execute('''
            INSERT INTO case_documents
            (user_id, case_id, folder_id, document_name, file_path, file_type, file_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, document_name, file_path, file_type, file_size, created_at
        ''', (
            user_id,
            case_id,
            folder_id,
            filename,
            file_path,
            file_type,
            file_size
        ))

        document = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Document uploaded successfully',
            'document': document
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
@token_required
def get_documents(user_id):
    # Get query parameters
    parent_type = request.args.get('parent_type')  # 'case' or 'folder'
    parent_id = request.args.get('parent_id')

    if not parent_type or not parent_id or parent_type not in ['case', 'folder']:
        return jsonify({'error': 'Valid parent_type (case or folder) and parent_id are required'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Verify the parent container belongs to the user
        if parent_type == 'case':
            cur.execute('''
                SELECT id FROM cases
                WHERE id = %s AND user_id = %s
            ''', (parent_id, user_id))

            if cur.fetchone():
                # Get documents for this case
                cur.execute('''
                    SELECT id, document_name, file_path, file_type, file_size, created_at
                    FROM case_documents
                    WHERE case_id = %s AND user_id = %s
                    ORDER BY created_at DESC
                ''', (parent_id, user_id))
            else:
                cur.close()
                conn.close()
                return jsonify({'error': 'Case not found or not authorized'}), 404

        else:  # parent_type == 'folder'
            cur.execute('''
                SELECT id FROM folders
                WHERE id = %s AND user_id = %s
            ''', (parent_id, user_id))

            if cur.fetchone():
                # Get documents for this folder
                cur.execute('''
                    SELECT id, document_name, file_path, file_type, file_size, created_at
                    FROM case_documents
                    WHERE folder_id = %s AND user_id = %s
                    ORDER BY created_at DESC
                ''', (parent_id, user_id))
            else:
                cur.close()
                conn.close()
                return jsonify({'error': 'Folder not found or not authorized'}), 404

        documents = cur.fetchall()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Documents retrieved successfully',
            'documents': documents
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
