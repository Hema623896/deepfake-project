import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from PIL import Image

from model import ModelWrapper
from gradcam_utils import compute_saliency_overlay
from utils import extract_frames

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
DB_PATH = os.path.join(BASE_DIR, 'data', 'predictions.db')
ALLOWED_IMG = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ensure database directory exists and initialize
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
import sqlite3

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            label TEXT,
            score REAL,
            explanation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()


def save_prediction(filename, label, score, explanation):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'INSERT INTO predictions (filename, label, score, explanation) VALUES (?, ?, ?, ?)',
        (filename, label, score, explanation),
    )
    conn.commit()
    conn.close()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.secret_key = os.environ.get('FLASK_SECRET', 'change-me-to-a-secure-random-key')

model = ModelWrapper()


def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


@app.route('/')
def index():
    # simple landing page
    return render_template('index.html')


@app.route('/analyze')
def analyze():
    # separate page dedicated to media upload
    return render_template('analyze.html')


@app.route('/about')
def about():
    # show information and recent predictions from database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT filename, label, score, explanation, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 20'
    )
    preds = c.fetchall()
    conn.close()
    return render_template('about.html', predictions=preds)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        if not username:
            flash('Enter a username', 'error')
            return redirect(url_for('login'))
        session['user'] = username
        flash('Logged in as ' + username, 'success')
        return redirect(url_for('account'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out', 'info')
    return redirect(url_for('index'))


@app.route('/account')
def account():
    user = session.get('user')
    if not user:
        return redirect(url_for('login'))
    return render_template('account.html', user=user)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
        out_file = os.path.join(BASE_DIR, 'data', 'messages.txt')
        with open(out_file, 'a', encoding='utf-8') as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n---\n")
        flash('Thank you — your message was received.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if allowed_file(filename, ALLOWED_IMG):
        img = Image.open(filepath).convert('RGB')
        label, score, tensor = model.predict_with_tensor(img)
        # generate overlay at original image size
        overlay = compute_saliency_overlay(model.model, tensor, int(label), orig_size=img.size)
        out_name = f'result_{filename}'
        out_path = os.path.join(app.config['RESULT_FOLDER'], out_name)
        overlay.save(out_path)
        web_path = os.path.relpath(out_path, BASE_DIR).replace('\\', '/')
        # convert numeric label into human-readable class name
        # ModelWrapper.class_names was set to ['fake','real'] during initialization
        label_str = model.class_names[int(label)].capitalize()
        # include confidence in explanation so users can judge correctness
        if label_str == 'Fake':
            explanation = (
                f"The model predicted Fake (score {score:.4f}); "
                "it identified visual artifacts commonly present in manipulated media."
            )
        else:
            explanation = (
                f"The model predicted Real (score {score:.4f}); "
                "it detected features consistent with authentic content."
            )
        # persist to database
        save_prediction(filename, label_str, float(score), explanation)
        return render_template('result.html', image_url='/' + web_path,
                               label=label_str, score=round(float(score), 4), explanation=explanation)

    elif allowed_file(filename, ALLOWED_VIDEO):
        frames = extract_frames(filepath, max_frames=8, resize=None)
        # accumulate counts for majority voting; also track highest-confidence fake frame
        best_score = -1.0
        best_overlay_path = None
        best_frame = None
        label_counts = {0:0, 1:0}
        fake_score = -1.0
        fake_overlay_path = None
        fake_frame = None
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame).convert('RGB')
            label, score, tensor = model.predict_with_tensor(img)
            label_counts[label] = label_counts.get(label, 0) + 1
            # keep the highest-confidence fake frame for visualization
            if label == 0 and score > fake_score:
                overlay = compute_saliency_overlay(model.model, tensor, int(label), orig_size=img.size)
                out_name = f'result_{os.path.splitext(filename)[0]}_frame{i}.jpg'
                out_path = os.path.join(app.config['RESULT_FOLDER'], out_name)
                overlay.save(out_path)
                fake_score = score
                fake_overlay_path = out_path
                fake_frame = i
        # determine final label by majority (fallback to highest fake confidence)
        final_label = None
        final_score = None
        frame_index = None
        overlay_path = None
        if label_counts[0] > label_counts[1]:
            final_label = 0
            final_score = fake_score
            overlay_path = fake_overlay_path
            frame_index = fake_frame
        else:
            # all-real or tie -> choose best real frame by score
            best_score = -1.0
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame).convert('RGB')
                label, score, tensor = model.predict_with_tensor(img)
                if label == 1 and score > best_score:
                    overlay = compute_saliency_overlay(model.model, tensor, int(label), orig_size=img.size)
                    out_name = f'result_{os.path.splitext(filename)[0]}_frame{i}.jpg'
                    out_path = os.path.join(app.config['RESULT_FOLDER'], out_name)
                    overlay.save(out_path)
                    best_score = score
                    overlay_path = out_path
                    frame_index = i
            final_label = 1
            final_score = best_score
        if overlay_path is None:
            return redirect(url_for('index'))
        web_path = os.path.relpath(overlay_path, BASE_DIR).replace('\\', '/')
        label_text = model.class_names[final_label].capitalize()
        if label_text == 'Fake':
            explanation = (
                f"The model predicted Fake (score {final_score:.4f}); "
                "it identified visual artifacts commonly present in manipulated media."
            )
        else:
            explanation = (
                f"The model predicted Real (score {final_score:.4f}); "
                "it detected features consistent with authentic content."
            )
        # save video prediction as well
        save_prediction(filename, label_text, float(final_score), explanation)
        return render_template('result.html', image_url='/' + web_path,
                               label=label_text, score=round(float(final_score), 4), frame_index=frame_index, explanation=explanation)
    else:
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
