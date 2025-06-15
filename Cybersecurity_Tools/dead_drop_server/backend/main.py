from flask import Flask, request, send_file, abort, render_template_string, render_template
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
import base64
import time
import uuid
from io import BytesIO
from datetime import datetime, timedelta
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
import secrets
import logging

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
app.config['UPLOAD_FOLDER'] = 'drops/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max
app.config['TOKEN_TTL'] = 15 * 60  # 15 minutes expiry

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(filename='access.log', level=logging.INFO)

token_store = {}

BLOCKED_EXTENSIONS = {'.exe', '.bat', '.sh', '.msi', '.php', '.py'}

# all functions
def derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))

def encrypt_file(file_path: str, passphrase: str) -> str:
    with open(file_path, 'rb') as f:
        data = f.read()
    salt = os.urandom(16)
    key = derive_key(passphrase, salt)
    encrypted = Fernet(key).encrypt(data)
    token = str(uuid.uuid4())
    encrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], token + '.enc')
    with open(encrypted_path, 'wb') as f:
        f.write(salt + encrypted)
    os.remove(file_path)
    expiry_time = time.time() + app.config['TOKEN_TTL']
    token_store[token] = {'path': encrypted_path, 'expires': expiry_time}
    return token

def decrypt_file(file_path: str, passphrase: str) -> bytes:
    with open(file_path, 'rb') as f:
        content = f.read()
    salt = content[:16]
    encrypted_data = content[16:]
    key = derive_key(passphrase, salt)
    return Fernet(key).decrypt(encrypted_data)

def is_token_valid(token: str) -> bool:
    return token in token_store and token_store[token]['expires'] > time.time()

def cleanup_expired_tokens():
    now = time.time()
    expired = [token for token, data in token_store.items() if data['expires'] < now]
    for token in expired:
        try:
            os.remove(token_store[token]['path'])
        except:
            pass
        del token_store[token]

def log_access(action, filename, ip):
    logging.info(f"{datetime.now().isoformat()} | {action} | {filename} | IP: {ip}")

# ROUTES
@app.route('/', methods=['GET'])
def index():
    cleanup_expired_tokens()
    return render_template('index.html')  # This will now look in 'frontend/templates'

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    passphrase = request.form.get('passphrase')

    if not file or not passphrase:
        return "Missing fields", 400

    filename = secure_filename(file.filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() in BLOCKED_EXTENSIONS:
        return f"Files with extension '{ext}' are not allowed.", 400

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)

    token = encrypt_file(temp_path, passphrase)

    return render_template('success.html', token=token, ip=request.remote_addr)

@app.route('/download/<token>', methods=['GET', 'POST'])
def download(token):
    cleanup_expired_tokens()
    if not is_token_valid(token):
        return "⛔ Invalid or expired token.", 403

    if request.method == 'GET':
        return '''
            <form method="POST">
                <input type="text" name="passphrase" placeholder="Passphrase" required><br>
                <button type="submit">Download</button>
            </form>
        '''

    passphrase = request.form.get('passphrase')
    file_path = token_store[token]['path']

    try:
        decrypted = decrypt_file(file_path, passphrase)
        buffer = BytesIO(decrypted)
        buffer.seek(0)

        filename = os.path.basename(file_path).replace('.enc', '')
        log_access('DOWNLOAD', filename, request.remote_addr)

        os.remove(file_path)
        del token_store[token]

        return send_file(
            buffer,
            as_attachment=True,
            download_name="downloaded_file.bin",
            mimetype='application/octet-stream'
        )
    except Exception as e:
        return f"⛔ Error: {str(e)}", 400

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return "⛔ File too large. Max 10MB.", 413


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
##The encrypted file after download loses its extension so add the extension by adding ".extension"
