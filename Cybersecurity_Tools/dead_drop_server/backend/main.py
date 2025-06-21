from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
import base64
import time
import uuid
import tempfile
import contextlib
import logging
from io import BytesIO
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
app.config['UPLOAD_FOLDER'] = 'drops/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max
app.config['TOKEN_TTL'] = 15 * 60  # 15 mins expiry
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(filename='access.log', level=logging.INFO)

token_store = {}
BLOCKED_EXTENSIONS = {'.exe', '.bat', '.sh', '.msi', '.php', '.py'}


def derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))


def encrypt_file(file_path: str, passphrase: str, orig_name: str) -> str:
    with open(file_path, 'rb') as f:
        data = f.read()
    salt = os.urandom(16)
    key = derive_key(passphrase, salt)
    encrypted = Fernet(key).encrypt(data)
    token = str(uuid.uuid4())
    encrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{token}.enc")
    with open(encrypted_path, 'wb') as f:
        f.write(salt + encrypted)
    os.remove(file_path)
    expiry_time = time.time() + app.config['TOKEN_TTL']
    token_store[token] = {'path': encrypted_path, 'expires': expiry_time, 'filename': orig_name}
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
        with contextlib.suppress(Exception):
            os.remove(token_store[token]['path'])
        token_store.pop(token, None)


def log_access(action, filename, ip):
    logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {action} | {filename} | IP: {ip}")


@app.route('/', methods=['GET'])
def index():
    cleanup_expired_tokens()
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    passphrase = request.form.get('passphrase')
    if not file or not passphrase:
        return "Missing fields.", 400

    filename = secure_filename(file.filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() in BLOCKED_EXTENSIONS:
        return f"Files with extension '{ext}' are not allowed.", 400

    with tempfile.NamedTemporaryFile(dir=app.config['UPLOAD_FOLDER'], delete=False) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    token = encrypt_file(temp_path, passphrase, orig_name=filename)
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
    filename = token_store[token]['filename']

    try:
        decrypted_data = decrypt_file(file_path, passphrase)
        buffer = BytesIO(decrypted_data)
        buffer.seek(0)

        log_access('DOWNLOAD', filename, request.remote_addr)

        os.remove(file_path)
        del token_store[token]

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
    except Exception as e:
        return f"⛔ Error decrypting file: {e}", 400


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return "⛔ File too large. Max 10MB.", 413


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
