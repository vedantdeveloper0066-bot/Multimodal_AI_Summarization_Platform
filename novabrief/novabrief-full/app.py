import os, re, sys, json, logging, subprocess, threading, time, hashlib, sqlite3, uuid, math
from typing import Optional, Tuple
from functools import wraps
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

try:
    import config as cfg
except ImportError:
    cfg = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('novabrief')

perf_log = logging.getLogger('novabrief.performance')
perf_log.propagate = False
perf_log.setLevel(logging.INFO)
try:
    _perf_handler = logging.FileHandler(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'performance.log'), encoding='utf-8')
    _perf_handler.setFormatter(logging.Formatter('%(message)s'))
    perf_log.addHandler(_perf_handler)
except Exception as _e:
    log.warning(f'Could not open performance.log for writing ({_e}) — per-request profiling data will '
                f'still print to the console, just not be saved to that file.')

_STAGE_TAGS = {
    'upload': 'UPLOAD', 'extraction': 'EXTRACT', 'ocr': 'OCR', 'cleaning': 'CLEAN',
    'transcription': 'WHISPER', 'captioning': 'BLIP', 'summarization': 'BART',
    'language_detect': 'LANG', 'translation_to_en': 'TRANSLATE', 'translation_from_en': 'TRANSLATE',
    'tts': 'TTS', 'database': 'DATABASE',
}

class Timings:
    __slots__ = ('label', 'marks', '_t0', 'request_id', 'meta')

    def __init__(self, label: str = 'request', request_id: str = None):
        self.label = label
        self.marks = []
        self._t0 = time.perf_counter()
        self.request_id = request_id or uuid.uuid4().hex[:8]
        self.meta = {}

    def set_meta(self, **kw):
        self.meta.update(kw)

    def _tag_log(self, name: str, seconds: float):
        tag = _STAGE_TAGS.get(name, name.upper())
        log.info(f'[{tag}] req={self.request_id} elapsed={seconds*1000:.0f}ms')

    class _Stage:
        __slots__ = ('parent', 'name', '_t0')
        def __init__(self, parent, name):
            self.parent, self.name = parent, name
        def __enter__(self):
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, *exc):
            elapsed = time.perf_counter() - self._t0
            self.parent.marks.append((self.name, elapsed))
            self.parent._tag_log(self.name, elapsed)
            return False

    def stage(self, name: str):
        return Timings._Stage(self, name)

    def mark(self, name: str, seconds: float):
        self.marks.append((name, seconds))
        self._tag_log(name, seconds)

    def total(self) -> float:
        return time.perf_counter() - self._t0

    def as_dict(self) -> dict:
        d = {}
        for name, secs in self.marks:
            d[name] = d.get(name, 0) + round(secs * 1000)
        d['total_ms'] = round(self.total() * 1000)
        return d

    def log(self):
        parts = ', '.join(f'{name}={secs*1000:.0f}ms' for name, secs in self.marks)
        try:
            res = _resource_usage_str()
        except Exception:
            res = ''
        total_ms = self.total() * 1000
        log.info(f'[TOTAL] req={self.request_id} {parts}{", " if parts else ""}total={total_ms:.0f}ms'
                 f'{"  |  " + res if res else ""}')
        try:
            record = {
                'ts': datetime.now().isoformat(timespec='seconds'),
                'request_id': self.request_id,
                'label': self.label,
                **self.meta,
                'stages_ms': {name: round(secs * 1000) for name, secs in self.marks},
                'total_ms': round(total_ms),
                'thread_count': threading.active_count(),
                'torch_device': _device_kind,
            }
            try:
                if _psutil_mod is not None:
                    record['cpu_percent'] = _psutil_mod.cpu_percent(interval=None)
                    record['ram_mb'] = round(_psutil_mod.Process(os.getpid()).memory_info().rss / 1e6)
            except Exception:
                pass
            gpu = _nvidia_smi_stats()
            if gpu: record['gpu'] = gpu
            perf_log.info(json.dumps(record))
        except Exception as e:
            log.warning(f'performance.log write failed (non-fatal — the request itself already succeeded): {e}')

CACHE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'novabrief_cache.sqlite3')
ENABLE_CACHE  = bool(cfg is None or getattr(cfg, 'ENABLE_CACHE', True))
CACHE_EXPIRY_HOURS = float((cfg.CACHE_EXPIRY_HOURS if cfg and getattr(cfg, 'CACHE_EXPIRY_HOURS', None) else 0) or 0)
_cache_lock   = threading.Lock()

def _cache_init():
    if not ENABLE_CACHE: return
    try:
        with _cache_lock:
            conn = sqlite3.connect(CACHE_DB_PATH, timeout=10)
            conn.execute('CREATE TABLE IF NOT EXISTS cache '
                         '(key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)')
            conn.commit(); conn.close()
        log.info(f'✅ Result cache ready: {CACHE_DB_PATH}'
                 + (f' (entries expire after {CACHE_EXPIRY_HOURS:g}h)' if CACHE_EXPIRY_HOURS else ' (no expiry)'))
    except Exception as e:
        log.warning(f'Cache unavailable ({e}) — every request will be reprocessed from scratch.')

def _cache_get(key: str):
    if not ENABLE_CACHE: return None
    try:
        with _cache_lock:
            conn = sqlite3.connect(CACHE_DB_PATH, timeout=10)
            row = conn.execute('SELECT value, created_at FROM cache WHERE key = ?', (key,)).fetchone()
            if row and CACHE_EXPIRY_HOURS and (time.time() - row[1]) > CACHE_EXPIRY_HOURS * 3600:
                conn.execute('DELETE FROM cache WHERE key = ?', (key,)); conn.commit()
                row = None
            conn.close()
        return json.loads(row[0]) if row else None
    except Exception as e:
        log.warning(f'cache_get({key!r}) failed, treating as a miss: {e}')
        return None

def _cache_set(key: str, value) -> None:
    if not ENABLE_CACHE: return
    try:
        with _cache_lock:
            conn = sqlite3.connect(CACHE_DB_PATH, timeout=10)
            conn.execute('INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)',
                        (key, json.dumps(value), time.time()))
            conn.commit(); conn.close()
    except Exception as e:
        log.warning(f'cache_set({key!r}) failed — result still returned to the user, just not cached: {e}')

def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _hash_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

_cache_init()

def _pip(pkg, timeout=600):
    log.info(f'Installing {pkg} (first run only for this package — can take a '
             f'few minutes on a slower connection, especially for GPU-enabled '
             f'packages like ctranslate2/torch)...')
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                                 capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'Installing {pkg} timed out after {timeout}s. This usually means a slow '
                            f'or blocked connection to PyPI — check your network (or a proxy/firewall '
                            f'if you\'re on a corporate one), or install it yourself first: '
                            f'pip install {pkg}')
    if result.returncode != 0:
        output = (result.stderr or result.stdout or '').strip()
        lines = output.splitlines()
        error_lines = [l for l in lines if l.strip().lower().startswith('error:')]
        reason = error_lines[-1] if error_lines else (lines[-1] if lines else 'no output captured')
        if 'externally-managed-environment' in output:
            reason += ('  (Your Python installation blocks direct pip installs — common on Debian/Ubuntu '
                       'system Python and Homebrew Python on macOS. Fix: run this app from a virtual '
                       'environment — python3 -m venv venv && source venv/bin/activate — or install the '
                       f'missing package yourself first: pip install {pkg} --break-system-packages)')
        raise RuntimeError(f'pip install {pkg} failed: {reason}')
    log.info(f'✅ Installed {pkg}')

def _try_import(module, pip_name=None):
    try:
        return __import__(module)
    except ImportError:
        _pip(pip_name or module)
        return __import__(module)

try:
    from flask import (Flask, request, jsonify, send_from_directory,
                       session, redirect, send_file)
except ImportError:
    _pip('flask'); from flask import (Flask, request, jsonify, send_from_directory,
                                      session, redirect, send_file)
try:
    from flask_cors import CORS
except ImportError:
    _pip('flask-cors'); from flask_cors import CORS
from werkzeug.utils import secure_filename

def _ensure_nltk():
    try:
        try:
            import nltk
        except ImportError:
            _pip('nltk'); import nltk
        for r in ('punkt', 'punkt_tab', 'stopwords'):
            try: nltk.download(r, quiet=True)
            except: pass
    except Exception as e:
        log.warning(f'nltk setup failed ({e}) — the sumy extractive-summarization fallback may be '
                    f'degraded, but this will not stop the app from starting. The AI summarizer '
                    f'(DistilBART) is unaffected and does not use nltk at all.')

_ensure_nltk()

app = Flask(__name__, static_folder='static', template_folder='templates')
_cors_origins = (cfg.CORS_ORIGINS if cfg and getattr(cfg, 'CORS_ORIGINS', None)
                 else os.environ.get('NOVABRIEF_CORS_ORIGINS', 'http://localhost:5000,http://127.0.0.1:5000').split(','))
CORS(app, supports_credentials=True, origins=_cors_origins)
app.config['SECRET_KEY']           = (cfg.SECRET_KEY if cfg else None) or os.environ.get('SECRET_KEY', 'novabrief-change-this-in-production')
app.config['MAX_CONTENT_LENGTH']   = 750 * 1024 * 1024
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = bool(cfg and getattr(cfg, 'SESSION_COOKIE_SECURE', False))
if app.config['SECRET_KEY'] in ('novabrief-change-this-in-production', 'novabrief-secret-change-me-in-production'):
    log.warning('⚠ SECURITY: SECRET_KEY is still the shipped placeholder value — sessions can be forged by '
                'anyone who reads this source. Set SECRET_KEY (env var) or config.py\'s SECRET_KEY to a random '
                'value before exposing this beyond your own machine: '
                'python -c "import secrets; print(secrets.token_hex(32))"')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_UPLOAD_EXTENSIONS = {
    'pdf'  : {'pdf'},
    'image': {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'},
    'audio': {'mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac', 'wma', 'opus'},
    'video': {'mp4', 'mov', 'mkv', 'avi', 'webm', 'wmv', 'flv', 'm4v'},
}

def _safe_upload_path(filename: str, source_type: str):
    safe = secure_filename(os.path.basename(filename or ''))
    if not safe:
        return None, None, 'Invalid or unsupported filename.'
    ext = safe.rsplit('.', 1)[-1].lower() if '.' in safe else ''
    allowed = ALLOWED_UPLOAD_EXTENSIONS.get(source_type)
    if allowed is not None and ext not in allowed:
        return None, None, f'Unsupported file type ".{ext}" for {source_type} — allowed: {", ".join(sorted(allowed))}.'
    full_path = os.path.join(UPLOAD_FOLDER, f'{os.urandom(4).hex()}_{safe}')
    return full_path, safe, None

from werkzeug.exceptions import HTTPException

@app.errorhandler(413)
def _err_413(e):
    mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({'success': False, 'error': f'File is too large (limit is {mb} MB).'}), 413

@app.errorhandler(404)
def _err_404(e):
    return jsonify({'success': False, 'error': 'Not found.'}), 404

@app.errorhandler(Exception)
def _err_any(e):
    if isinstance(e, HTTPException):
        return jsonify({'success': False, 'error': e.description}), e.code
    log.error(f'Unhandled error: {e}', exc_info=True)
    client_msg = f'Server error: {e}' if app.debug else 'Server error — see the server log for details.'
    return jsonify({'success': False, 'error': client_msg}), 500

def _cfg_or_env(cfg_val, env_name, default):
    if cfg_val is not None and cfg_val != '':
        return cfg_val
    return os.environ.get(env_name, default)

DB_CFG = {
    'host'    : _cfg_or_env(cfg.DB_HOST     if cfg else None, 'MYSQL_HOST', 'localhost'),
    'port'    : int(_cfg_or_env(cfg.DB_PORT if cfg else None, 'MYSQL_PORT', 3306)),
    'user'    : _cfg_or_env(cfg.DB_USER     if cfg else None, 'MYSQL_USER', 'root'),
    'password': cfg.DB_PASSWORD if cfg else os.environ.get('MYSQL_PASSWORD', ''),
    'database': _cfg_or_env(cfg.DB_NAME     if cfg else None, 'MYSQL_DATABASE', 'novabrief'),
    'charset' : 'utf8mb4',
}
if DB_CFG['password'] == 'change-me':
    log.warning('⚠ SECURITY: DB_PASSWORD is still the shipped placeholder value — '
                'set MYSQL_PASSWORD (env var) or config.py\'s DB_PASSWORD to your real '
                'MySQL password before exposing this beyond your own machine.')
if not re.fullmatch(r'[A-Za-z0-9_]{1,64}', DB_CFG['database']):
    raise ValueError(f"Invalid DB_NAME {DB_CFG['database']!r} in config.py — must be 1-64 characters, "
                      f"letters/numbers/underscore only.")
_db_ready = False

_db_pool = None
_db_pool_lock = threading.Lock()

def _get_db():
    global _db_pool
    try:
        import mysql.connector
    except ImportError:
        _pip('mysql-connector-python')
        import mysql.connector
    if _db_pool is None:
        with _db_pool_lock:
            if _db_pool is None:
                try:
                    from mysql.connector import pooling
                    _db_pool = pooling.MySQLConnectionPool(
                        pool_name='novabrief_pool', pool_size=5, pool_reset_session=True, **DB_CFG)
                    log.info('MySQL connection pool ready (size 5)')
                except Exception as e:
                    log.warning(f'Could not create MySQL connection pool ({e}) — falling back to '
                                f'one-off connections (same behavior as before pooling was added).')
                    _db_pool = False
    if _db_pool:
        return _db_pool.get_connection()
    return mysql.connector.connect(**DB_CFG)

def _init_db_once():
    global _db_ready
    import mysql.connector
    base_cfg = {k: v for k, v in DB_CFG.items() if k != 'database'}
    conn = mysql.connector.connect(**base_cfg)
    cur  = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CFG['database']}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit(); cur.close(); conn.close()

    conn = _get_db(); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            name          VARCHAR(255) NOT NULL,
            email         VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            user_id             INT NOT NULL,
            title               VARCHAR(500),
            source_type         VARCHAR(50) NOT NULL,
            source_info         TEXT,
            original_language   VARCHAR(20)  DEFAULT 'en',
            original_word_count INT          DEFAULT 0,
            summary_text        MEDIUMTEXT,
            summary_language    VARCHAR(20)  DEFAULT 'en',
            method              VARCHAR(150),
            audio_filename      VARCHAR(255),
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """)
    try:
        cur.execute('CREATE INDEX idx_summaries_user_created ON summaries (user_id, created_at DESC)')
    except mysql.connector.Error as e:
        if e.errno != 1061:
            raise
    conn.commit(); cur.close(); conn.close()
    _db_ready = True

def init_db():
    delays = [0, 2, 3, 5, 8, 8, 8]
    last_err = None
    for i, d in enumerate(delays):
        if d: threading.Event().wait(d)
        try:
            _init_db_once()
            log.info(f'✅ MySQL ready ({DB_CFG["host"]}:{DB_CFG["port"]}/{DB_CFG["database"]})'
                      + (f' after {i} retr{"y" if i==1 else "ies"}' if i else ''))
            return
        except Exception as e:
            last_err = e
    log.warning(
        f'MySQL unavailable after {len(delays)} attempts ({last_err}). '
        f'Auth & history disabled — app still works for summarization.\n'
        f'  Fix: 1) Edit config.py with your MySQL credentials\n'
        f'       2) Run: python setup_db.py\n'
        f'       3) Restart: python app.py\n'
        f'  Or check GET /api/db-status any time for a live readout.'
    )

threading.Thread(target=init_db, daemon=True).start()

def _hash_pw(pw: str) -> str:
    try: import bcrypt
    except ImportError: _pip('bcrypt'); import bcrypt
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def _check_pw(pw: str, hashed: str) -> bool:
    try: import bcrypt
    except ImportError: _pip('bcrypt'); import bcrypt
    return bcrypt.checkpw(pw.encode(), hashed.encode())

def login_required(f):
    @wraps(f)
    def decorated(*a, **kw):
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Login required'}), 401
        return f(*a, **kw)
    return decorated

_device       = None
_device_kind  = 'cpu'
_device_label = 'CPU'
_use_fp16     = False

_model_lock = threading.Lock()

_device_resolve_lock = threading.Lock()

MODEL_LOCK_WAIT_SECONDS     = int((cfg.MODEL_LOCK_WAIT_SECONDS if cfg and getattr(cfg, 'MODEL_LOCK_WAIT_SECONDS', None) else 0) or 180)
INFERENCE_WATCHDOG_SECONDS  = int((cfg.INFERENCE_WATCHDOG_SECONDS if cfg and getattr(cfg, 'INFERENCE_WATCHDOG_SECONDS', None) else 0) or 900)
NETWORK_WATCHDOG_SECONDS    = int((cfg.NETWORK_WATCHDOG_SECONDS if cfg and getattr(cfg, 'NETWORK_WATCHDOG_SECONDS', None) else 0) or 120)

class ModelBusyError(RuntimeError):
    pass

@contextmanager
def _model_lock_or_busy(label='This request'):
    got = _model_lock.acquire(timeout=MODEL_LOCK_WAIT_SECONDS)
    if not got:
        raise ModelBusyError(
            f'{label} is queued behind another AI request that is still using the shared model — the '
            f'server intentionally runs only one GPU/CPU model call at a time so a 4GB GPU is never asked '
            f'to do two at once (see _model_lock). It has been waiting over {MODEL_LOCK_WAIT_SECONDS}s, '
            f'longer than normal queueing should ever take. Please try again shortly.'
        )
    try:
        yield
    finally:
        _model_lock.release()

def _run_watched(fn, timeout=INFERENCE_WATCHDOG_SECONDS, label='model call'):
    box = {}
    def _work():
        try:
            box['result'] = fn()
        except BaseException as e:
            box['error'] = e
    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(timeout=timeout)
    if th.is_alive():
        raise RuntimeError(
            f'{label} did not finish within {timeout}s, which is far longer than this normally takes — '
            f'something is genuinely stuck rather than just slow (most likely a native-level GPU/CUDA '
            f'stall; see _construct_whisper_model\'s docstring for why that can happen and why no ordinary '
            f'try/except can catch it). The server itself is fine and can still handle other requests. To '
            f'confirm this is GPU-related, set FORCE_CPU = True in config.py and try again.'
        )
    if 'error' in box:
        raise box['error']
    return box.get('result')

def _has_nvidia_gpu() -> bool:
    try:
        r = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=8)
        return r.returncode == 0 and 'GPU' in r.stdout
    except Exception:
        return False

_nvidia_smi_available = None

def _nvidia_smi_stats() -> Optional[dict]:
    global _nvidia_smi_available
    if _nvidia_smi_available is False:
        return None
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if r.returncode != 0 or not r.stdout.strip():
            _nvidia_smi_available = False
            return None
        _nvidia_smi_available = True
        util, mem_used, mem_total, temp = (x.strip() for x in r.stdout.strip().split(',')[:4])
        return {'gpu_util_pct': float(util), 'vram_used_mb': float(mem_used),
                'vram_total_mb': float(mem_total), 'gpu_temp_c': float(temp)}
    except Exception:
        _nvidia_smi_available = False
        return None

def _resolve_device():
    global _device, _device_kind, _device_label, _use_fp16
    if _device is not None:
        return _device
    with _device_resolve_lock:
        if _device is not None:
            return _device
        force_cpu = bool(cfg and getattr(cfg, 'FORCE_CPU', False))
        try:
            import torch
            if not force_cpu and torch.cuda.is_available():
                idx = int(getattr(cfg, 'GPU_DEVICE_INDEX', 0)) if cfg else 0
                if idx >= torch.cuda.device_count(): idx = 0
                gpu_name = torch.cuda.get_device_name(idx)
                try:
                    probe = torch.randn(8, 8, device=f'cuda:{idx}')
                    _ = (probe @ probe).sum().item()
                    torch.cuda.synchronize(idx)
                    cuda_verified = True
                except Exception as probe_err:
                    cuda_verified = False
                    log.warning(
                        f'⚠ GPU "{gpu_name}" was detected but failed a real CUDA operation ({probe_err}). '
                        f'This is the "cublas64_12.dll is not found or cannot be loaded" class of failure — '
                        f'the NVIDIA driver is present but the CUDA math libraries PyTorch needs are not '
                        f'(usually a PyTorch build/CUDA-toolkit version mismatch, or a partially-broken '
                        f'install). Every AI model will use CPU instead for the rest of this run, rather '
                        f'than letting this happen mid-request. Fix: run fix_torch.bat (Windows) / '
                        f'fix_torch.sh (Linux/macOS), or update your NVIDIA driver.'
                    )
                if cuda_verified:
                    _device       = torch.device(f'cuda:{idx}')
                    _device_kind  = 'cuda'
                    _device_label = gpu_name
                    _use_fp16     = True
                    vram_mb = torch.cuda.get_device_properties(idx).total_memory // (1024 * 1024)
                    log.info(f'🎮 GPU verified working: {_device_label} ({vram_mb} MB VRAM) — '
                             f'AI models will run here. Web app + preprocessing stay on CPU.')
                else:
                    _device, _device_kind, _device_label, _use_fp16 = torch.device('cpu'), 'cpu', 'CPU', False
            else:
                _device, _device_kind, _device_label, _use_fp16 = torch.device('cpu'), 'cpu', 'CPU', False
                log.info('FORCE_CPU is set in config.py — AI models will run on CPU.' if force_cpu else
                          'No usable GPU detected — AI models will run on CPU (works fine, just slower on long documents).')
        except ImportError:
            pass
        return _device

_psutil_checked = False
_psutil_mod     = None

def _resource_usage_str() -> str:
    global _psutil_checked, _psutil_mod
    if not _psutil_checked:
        _psutil_checked = True
        try:
            _try_import('psutil')
            import psutil
            _psutil_mod = psutil
        except Exception:
            _psutil_mod = None

    bits = [f'torch_device={_device_kind or "unresolved"}']
    if _whisper_ready:
        bits.append(f'whisper_device={"cuda" if "GPU" in (_whisper_device_label or "") else "cpu"}')
    if _psutil_mod is not None:
        try:
            rss_mb = _psutil_mod.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            bits.append(f'ram={rss_mb:.0f}MB')
        except Exception:
            pass
    if _device_kind == 'cuda':
        try:
            import torch
            idx = _device.index if _device is not None else 0
            alloc_mb = torch.cuda.memory_allocated(idx) / (1024 * 1024)
            reserv_mb = torch.cuda.memory_reserved(idx) / (1024 * 1024)
            bits.append(f'vram_alloc_torch_only={alloc_mb:.0f}MB')
            bits.append(f'vram_reserved_torch_only={reserv_mb:.0f}MB')
        except Exception:
            pass
    return ' '.join(bits)

DEFAULT_MODEL   = (cfg.AI_MODEL if cfg else None) or os.environ.get('NOVABRIEF_MODEL', 'sshleifer/distilbart-cnn-12-6')
_txt_pipeline   = None
_txt_model_name = None
_txt_ready      = False
_txt_error      = None
_txt_lock       = threading.Lock()

def _calc_length(word_count: int) -> Tuple[int, int]:
    if   word_count <  100: return  40,  120
    elif word_count <  300: return  70,  180
    elif word_count <  600: return 100,  250
    elif word_count < 1200: return 150,  350
    elif word_count < 2500: return 220,  480
    elif word_count < 5000: return 300,  620
    elif word_count <10000: return 400,  800
    else:                   return 500, 1000

_torch_install_lock = threading.Lock()

def _install_torch() -> bool:
    import importlib
    _torch_install_lock.acquire()
    try:
        return _install_torch_impl(importlib)
    finally:
        _torch_install_lock.release()

def _install_torch_impl(importlib) -> bool:
    try:
        importlib.invalidate_caches()
        import torch
        log.info(f'PyTorch already installed: {torch.__version__}')
        return True
    except ImportError:
        pass

    log.info('Installing PyTorch …')

    TRUSTED = [
        '--trusted-host', 'download.pytorch.org',
        '--trusted-host', 'files.pythonhosted.org',
        '--trusted-host', 'pypi.org',
    ]
    CPU_IDX = ['--index-url', 'https://download.pytorch.org/whl/cpu']

    attempts = []
    force_cpu = bool(cfg and getattr(cfg, 'FORCE_CPU', False))
    if force_cpu:
        log.info('FORCE_CPU is set in config.py — going straight to the CPU build.')
    elif _has_nvidia_gpu():
        log.info('🎮 NVIDIA GPU detected — trying a GPU-enabled PyTorch build first …')
        attempts.append([sys.executable, '-m', 'pip', 'install', 'torch'] + TRUSTED)
    else:
        log.info('No NVIDIA GPU detected — installing the CPU build.')

    attempts += [
        [sys.executable, '-m', 'pip', 'install', 'torch'] + CPU_IDX,
        [sys.executable, '-m', 'pip', 'install', 'torch'] + CPU_IDX + TRUSTED,
        [sys.executable, '-m', 'pip', 'install', 'torch'] + CPU_IDX + TRUSTED + ['--no-cache-dir'],
        [sys.executable, '-m', 'pip', 'install', 'torch'] + CPU_IDX + TRUSTED + ['--break-system-packages'],
    ]

    for i, cmd in enumerate(attempts, 1):
        try:
            log.info(f'  Method {i}/{len(attempts)}: {" ".join(cmd[4:])}')
            result = subprocess.run(
                cmd + ['-q'],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                log.warning(f'    Failed: {result.stderr.strip()[:200]}')
                continue
            importlib.invalidate_caches()
            import torch
            build = 'CUDA' if torch.cuda.is_available() else 'CPU'
            log.info(f'✅ PyTorch {torch.__version__} installed (method {i}, {build} build)')
            return True
        except subprocess.TimeoutExpired:
            log.warning(f'    Method {i} timed out (600 s)')
        except Exception as e:
            log.warning(f'    Method {i} error: {e}')

    log.error(
        f'All {len(attempts)} PyTorch install methods failed.\n'
        'Run fix_torch.bat (Windows) or fix_torch.sh (Linux/macOS) for a detailed fix.\n'
        'The app continues working using extractive summarization.'
    )
    return False

def _build_summarizer_pipeline(hfp, model_id, device_arg):
    import torch
    try:
        return hfp('summarization', model=model_id, device=device_arg, framework='pt',
                   torch_dtype=(torch.float16 if (_use_fp16 and device_arg == 0) else torch.float32))
    except Exception:
        return hfp('summarization', model=model_id, device=device_arg, framework='pt')

def _load_txt_model():
    global _txt_pipeline, _txt_model_name, _txt_ready, _txt_error
    with _txt_lock:
        if _txt_ready:
            return
        _t0 = time.perf_counter()

        _txt_error = None

        try:
            torch_ok = _install_torch()
        except Exception as e:
            log.exception('PyTorch install/check raised unexpectedly')
            torch_ok, _txt_error = False, f'PyTorch install check failed unexpectedly: {e}'
        if not torch_ok:
            _txt_error = _txt_error or (
                'PyTorch is not installed. '
                'Run:  pip install torch --index-url https://download.pytorch.org/whl/cpu'
            )
            log.error(_txt_error)
            return

        try:
            import importlib
            _try_import('transformers')
            _try_import('sentencepiece')
            importlib.invalidate_caches()
        except Exception as e:
            _txt_error = f'Cannot import transformers: {e}'
            log.error(_txt_error)
            return

        model_list = list(dict.fromkeys([DEFAULT_MODEL,
                                         'sshleifer/distilbart-cnn-12-6',
                                         't5-small']))
        last_err = ''
        for mid in model_list:
            try:
                from transformers import pipeline as hfp
                log.info(f'Loading AI model: {mid}  (downloads on first run, cached after)')
                _resolve_device()
                device_arg = 0 if _device_kind == 'cuda' else -1
                running_on = _device_label
                try:
                    pipe = _build_summarizer_pipeline(hfp, mid, device_arg)
                except Exception as gpu_err:
                    if device_arg != 0:
                        raise
                    log.warning(f'  GPU load of {mid} failed ({gpu_err}) — retrying on CPU for this model.')
                    device_arg = -1
                    running_on = 'CPU (GPU load failed for this model — see log)'
                    pipe = _build_summarizer_pipeline(hfp, mid, device_arg)
                _txt_pipeline   = pipe
                _txt_model_name = mid
                _txt_ready      = True
                log.info(f'✅ AI model ready: {mid}  ·  running on {running_on}  ·  loaded in {time.perf_counter()-_t0:.1f}s'
                         f'  ·  {_resource_usage_str()}')
                return
            except Exception as e:
                last_err = str(e)
                log.warning(f'  skip {mid}: {e}')

        _txt_error = (
            f'All models failed to load (after {time.perf_counter()-_t0:.1f}s). Last error: {last_err}\n'
            'Make sure you have an internet connection for the first-time model download. '
            'Summaries will use extractive fallback (sumy) until the model loads.'
        )
        log.error(_txt_error)

CHUNK_TOKEN_BUDGET = 900

def _count_tokens(text: str) -> int:
    if _txt_ready and _txt_pipeline is not None:
        try:
            return len(_txt_pipeline.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return int(len(text.split()) * 1.3)

_ARTIFACT_BRACKET_RE = re.compile(r'\[(?:music|applause|laughter|inaudible|silence|noise|crosstalk)\]', re.I)
_SPEAKER_LABEL_RE    = re.compile(r'^\s*(?:speaker\s*\d*|>>+|>|\[[A-Za-z][\w .]{0,20}\])\s*[:\-]?\s*', re.I | re.M)
_BARE_TIMESTAMP_RE   = re.compile(r'\[?\b\d{1,2}:\d{2}(?::\d{2})?\b\]?')
_FILLER_WORD_RE = re.compile(r'\b(?:um+|uh+|erm+|uhm+)\b[,.]?', re.I)
_MULTI_SPACE_RE      = re.compile(r'[ \t]+')
_MULTI_NEWLINE_RE    = re.compile(r'\n{3,}')

def _dedupe_repeated_phrases(text: str, max_overlap_words: int = 15, min_overlap_words: int = 4) -> str:
    if not text:
        return text
    sents = re.split(r'(?<=[.!?])\s+', text)
    deduped = []
    for s in sents:
        norm = s.strip().lower()
        if deduped and norm and norm == deduped[-1].strip().lower():
            continue
        deduped.append(s)
    text = ' '.join(deduped)

    words = text.split()
    words_lower = [w.lower() for w in words]
    out_lower = []
    out = []
    i, n = 0, len(words)
    while i < n:
        max_k = min(max_overlap_words, len(out), n - i)
        best_k = 0
        for k in range(max_k, min_overlap_words - 1, -1):
            if out_lower[-k:] == words_lower[i:i + k]:
                best_k = k
                break
        if best_k:
            i += best_k
        else:
            out.append(words[i])
            out_lower.append(words_lower[i])
            i += 1
    return ' '.join(out)

def _clean_text(text: str, source: str = 'generic') -> str:
    if not text:
        return text

    text = _ARTIFACT_BRACKET_RE.sub('', text)
    text = _SPEAKER_LABEL_RE.sub('', text)
    if source in ('audio', 'video', 'youtube'):
        text = _BARE_TIMESTAMP_RE.sub('', text)
        text = _FILLER_WORD_RE.sub('', text)
    text = _MULTI_SPACE_RE.sub(' ', text)
    text = _MULTI_NEWLINE_RE.sub('\n\n', text)
    text = _dedupe_repeated_phrases(text)
    return text.strip()

def _pack_sentences(sents: list, max_tokens: int) -> list:
    chunks, cur, cur_tokens = [], [], 0
    for s in sents:
        st = _count_tokens(s)
        if cur_tokens + st > max_tokens and cur:
            chunks.append(' '.join(cur)); cur, cur_tokens = [], 0
        if st > max_tokens:
            words = s.split()
            words_per_slice = max(1, int(len(words) * max_tokens / max(st, 1)))
            for i in range(0, len(words), words_per_slice):
                chunks.append(' '.join(words[i:i + words_per_slice]))
        else:
            cur.append(s); cur_tokens += st
    if cur: chunks.append(' '.join(cur))
    return chunks

def _chunk(text: str, max_tokens: int = CHUNK_TOKEN_BUDGET) -> list:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks = []
    min_viable = max_tokens * 0.3
    for para in paragraphs:
        sents = re.split(r'(?<=[.!?])\s+', para)
        para_chunks = _pack_sentences(sents, max_tokens)
        if (para_chunks and chunks and _count_tokens(para_chunks[0]) < min_viable
                and _count_tokens(chunks[-1]) + _count_tokens(para_chunks[0]) <= max_tokens):
            chunks[-1] = chunks[-1] + ' ' + para_chunks[0]
            chunks.extend(para_chunks[1:])
        else:
            chunks.extend(para_chunks)

    merged = []
    for c in chunks:
        if merged and _count_tokens(c) < min_viable and _count_tokens(merged[-1]) + _count_tokens(c) <= max_tokens:
            merged[-1] = merged[-1] + ' ' + c
        else:
            merged.append(c)
    return [c for c in merged if c.strip()]

def _summarize_texts(texts: list, min_t: int, max_t: int) -> list:
    import torch
    batch_size = 8 if _device_kind == 'cuda' else 2
    num_beams  = 4 if _device_kind == 'cuda' else 2

    def _do_summarize():
        with torch.inference_mode():
            return _txt_pipeline(texts, max_length=max_t, min_length=min(min_t, max_t - 5),
                                  do_sample=False, truncation=True,
                                  num_beams=num_beams, batch_size=batch_size,
                                  length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True)

    with _model_lock_or_busy('Text summarization'):
        results = _run_watched(_do_summarize, label='Text summarization')
    return [r['summary_text'].strip() for r in results]

def _summarize_chunk(chunk: str, min_t: int, max_t: int) -> str:
    wc = len(chunk.split())
    max_t = min(max_t, max(min_t + 10, wc // 2))
    return _summarize_texts([chunk], min_t, max_t)[0]

def _make_heading(sentence: str, max_words=6) -> str:
    words = re.sub(r'[^A-Za-z0-9\s]', '', sentence or '').split()
    stop = {'a','an','the','this','that','these','those','it','its','and','but','with','for','of','to'}
    kept = [w for w in words if w.lower() not in stop] or words
    return ' '.join(kept[:max_words]).title() or 'Overview'

MAX_SECTIONS = 8

def _merge_to_n(chunks: list, n: int) -> list:
    if len(chunks) <= n: return chunks
    step, i, merged = len(chunks) / n, 0.0, []
    for _ in range(n):
        j = max(int(i + step), int(i) + 1)
        merged.append(' '.join(chunks[int(i):j]))
        i = j
    return merged

_STAT_RE = re.compile(
    r'\$\s?[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|trillion|[kKmMbB]\b))?'
    r'|\b[\d,]+(?:\.\d+)?\s?%'
    r'|\b[\d,]+(?:\.\d+)?\s?(?:million|billion|trillion)\b'
)

_DEFINITION_RE = re.compile(
    r'\b([A-Z][a-zA-Z0-9\- ]{2,40}?)\s+(?:is defined as|refers to|is a term for|means|is known as)\s+',
    re.I)

def _extract_definitions(text: str, max_items: int = 5) -> list:
    sents = re.split(r'(?<=[.!?])\s+', text)
    found, seen = [], set()
    for s in sents:
        s = s.strip()
        if not s or len(s) > 300:
            continue
        if _DEFINITION_RE.search(s):
            key = s.lower()[:80]
            if key not in seen:
                seen.add(key)
                found.append(s)
        if len(found) >= max_items:
            break
    return found

def _extract_statistics(text: str, max_items: int = 6) -> list:
    sents = re.split(r'(?<=[.!?])\s+', text)
    found, seen = [], set()
    for s in sents:
        s = s.strip()
        if not s or len(s) > 280:
            continue
        if _STAT_RE.search(s):
            key = s.lower()[:80]
            if key not in seen:
                seen.add(key)
                found.append(s)
        if len(found) >= max_items:
            break
    return found

_CAP_PHRASE_RE = re.compile(
    r'\b[A-Z][a-zA-Z]+(?:\s+(?:(?:of|the|and|for|&)\s+)?[A-Z][a-zA-Z]+){0,3}\b'
)
_SENTENCE_STARTER_WORDS = {
    'the', 'this', 'these', 'those', 'it', 'in', 'on', 'at', 'for', 'to', 'a', 'an', 'we', 'i',
    'however', 'therefore', 'additionally', 'furthermore', 'moreover', 'meanwhile', 'overall',
    'first', 'second', 'third', 'finally', 'next', 'then', 'also', 'but', 'and', 'so',
}

def _extract_key_concepts(text: str, max_items: int = 8, min_occurrences: int = 2) -> list:
    from collections import Counter
    normalized = []
    for c in _CAP_PHRASE_RE.findall(text):
        words = c.split()
        while words and words[0].lower() in _SENTENCE_STARTER_WORDS:
            words = words[1:]
        if words:
            normalized.append(' '.join(words))
    counts = Counter(normalized)
    ranked = [(term, n) for term, n in counts.most_common(max_items * 3)
              if n >= (min_occurrences if ' ' in term else min_occurrences + 1)]
    return [term for term, _ in ranked[:max_items]]

def _bold_key_terms(text: str, terms: list, max_occurrences_per_term: int = 2) -> str:
    for term in terms:
        if not term.strip():
            continue
        pattern = re.compile(rf'\b{re.escape(term)}\b')
        count = 0
        def _sub(m):
            nonlocal count
            start = m.start()
            if text[max(0, start - 2):start] == '**':
                return m.group(0)
            if count >= max_occurrences_per_term:
                return m.group(0)
            count += 1
            return f'**{m.group(0)}**'
        text = pattern.sub(_sub, text)
    return text

def _append_dynamic_sections(out: str, source_text: str) -> str:
    concepts = _extract_key_concepts(source_text)
    if concepts:
        out = _bold_key_terms(out, concepts)
        out += '\n\n## Key Concepts\n' + '\n'.join(f'- {c}' for c in concepts)
    definitions = _extract_definitions(source_text)
    if definitions:
        out += '\n\n## Definitions\n' + '\n'.join(f'- {d}' for d in definitions)
    stats = _extract_statistics(source_text)
    if stats:
        out += '\n\n## Key Statistics\n' + '\n'.join(f'- {s}' for s in stats)
    return out

def _ai_summarize(text: str) -> Tuple[str, str]:
    if not _txt_ready:
        if _txt_error: raise RuntimeError(_txt_error)
        raise RuntimeError('The summarization model is still downloading/loading (first run only — '
                            'can take a few minutes, not just seconds). Check your server console for '
                            'progress, or GET /api/diagnose for live status.')
    wc = len(text.split())
    if wc <= 650:
        min_t, max_t = _calc_length(wc)
        body = _summarize_chunk(text, min_t, max_t)
        return _append_dynamic_sections(body, text), _txt_model_name

    chunks = _merge_to_n(_chunk(text), MAX_SECTIONS)
    wcs = [len(c.split()) for c in chunks]
    batch_min, batch_max = _calc_length(sum(wcs) // max(1, len(wcs)))
    bodies = _summarize_texts(chunks, batch_min, batch_max)
    sections = [(_make_heading(b), b) for b in bodies if b and b.strip()]

    if not sections:
        raise RuntimeError('Model produced no output.')
    if len(sections) == 1:
        return _append_dynamic_sections(sections[0][1], text), _txt_model_name

    overview = _summarize_chunk(' '.join(b for _, b in sections[:3]), 40, 110)
    takeaway_pool = re.split(r'(?<=[.!?])\s+', ' '.join(b for _, b in sections))
    takeaways = [s.strip() for s in takeaway_pool if len(s.strip()) > 25][:6]

    concepts = _extract_key_concepts(text)
    body_parts = [overview.strip()]
    for heading, body in sections:
        body_parts.append(f'## {heading}\n{body}')
    if takeaways:
        body_parts.append('## Key Takeaways\n' + '\n'.join(f'- {t}' for t in takeaways))
    out = '\n\n'.join(body_parts)

    if concepts:
        out = _bold_key_terms(out, concepts)
        overview_part, sep, rest = out.partition('\n\n')
        concepts_section = '## Key Concepts\n' + '\n'.join(f'- {c}' for c in concepts)
        out = overview_part + '\n\n' + concepts_section + (sep + rest if rest else '')

    definitions = _extract_definitions(text)
    if definitions:
        out += '\n\n## Definitions\n' + '\n'.join(f'- {d}' for d in definitions)
    stats = _extract_statistics(text)
    if stats:
        out += '\n\n## Key Statistics\n' + '\n'.join(f'- {s}' for s in stats)
    return out, _txt_model_name

def _sumy_fallback(text: str, n=8) -> Optional[str]:
    try:
        _try_import('sumy')
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers    import Tokenizer
        from sumy.summarizers.lsa   import LsaSummarizer
        from sumy.nlp.stemmers      import Stemmer
        from sumy.utils             import get_stop_words
        wc = len(text.split())
        n  = max(4, min(n, wc // 60))
        p  = PlaintextParser.from_string(text, Tokenizer('english'))
        s  = LsaSummarizer(Stemmer('english'))
        s.stop_words = get_stop_words('english')
        r = ' '.join(str(x) for x in s(p.document, n))
        return r or None
    except: return None

def summarize_text(text: str) -> Tuple[str, str]:
    text = text.strip()
    if not text: return 'No content to summarise.', 'none'
    words = text.split()
    if len(words) > 25000: text = ' '.join(words[:25000])
    try:
        s, m = _ai_summarize(text)
        if s: return s, f'AI · {m}'
    except ModelBusyError:
        raise
    except Exception as e:
        log.warning(f'AI failed: {e}')
    r = _sumy_fallback(text)
    if r: return r, 'sumy LSA (extractive)'
    sents = re.split(r'(?<=[.!?])\s+', text)
    return ' '.join(s.strip() for s in sents if len(s.strip()) > 20)[:800], 'sentence extraction'

IMAGE_MODEL_ID = ((cfg.IMAGE_MODEL if cfg and getattr(cfg, 'IMAGE_MODEL', None) else None)
                   or os.environ.get('NOVABRIEF_IMAGE_MODEL', 'Salesforce/blip-image-captioning-large'))
CONFIDENCE_LOW_THRESHOLD = float((cfg.CONFIDENCE_LOW_THRESHOLD if cfg and getattr(cfg, 'CONFIDENCE_LOW_THRESHOLD', None) else 0) or 0.4)
CONFIDENCE_VERY_LOW_THRESHOLD = float((cfg.CONFIDENCE_VERY_LOW_THRESHOLD if cfg and getattr(cfg, 'CONFIDENCE_VERY_LOW_THRESHOLD', None) else 0) or 0.15)
_img_processor = None
_img_model     = None
_img_ready     = False
_img_error     = None
_img_lock      = threading.Lock()
_img_device       = None
_img_device_label = 'CPU'

def _load_img_model():
    global _img_processor, _img_model, _img_ready, _img_error, _img_device, _img_device_label
    with _img_lock:
        if _img_ready: return
        _img_error = None
        _t0 = time.perf_counter()
        try:
            if not _install_torch(): return
            _try_import('transformers')
            try:
                from PIL import Image as _pil_test
            except ImportError:
                _pip('Pillow')
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            dev = _resolve_device()
            log.info(f'Loading image model: {IMAGE_MODEL_ID}')
            _img_processor = BlipProcessor.from_pretrained(IMAGE_MODEL_ID)
            dtype = torch.float16 if _use_fp16 else torch.float32

            def _place(target_device, use_dtype):
                try:
                    return BlipForConditionalGeneration.from_pretrained(
                        IMAGE_MODEL_ID, torch_dtype=use_dtype).to(target_device)
                except Exception:
                    return BlipForConditionalGeneration.from_pretrained(IMAGE_MODEL_ID).to(target_device)

            try:
                _img_model = _place(dev, dtype)
                _img_device, _img_device_label = dev, _device_label
            except Exception as gpu_err:
                if _device_kind != 'cuda':
                    raise
                log.warning(f'  GPU load of image model failed ({gpu_err}) — retrying on CPU.')
                _img_model = _place(torch.device('cpu'), torch.float32)
                _img_device, _img_device_label = torch.device('cpu'), 'CPU (GPU load failed — see log)'

            _img_ready = True
            log.info(f'✅ Image model ready: {IMAGE_MODEL_ID} · running on {_img_device_label}'
                     f'  ·  loaded in {time.perf_counter()-_t0:.1f}s  ·  {_resource_usage_str()}')
        except Exception as e:
            _img_error = str(e)
            log.error(f'Image model failed (after {time.perf_counter()-_t0:.1f}s): {e}')

_txt_load_thread     = threading.Thread(target=_load_txt_model, daemon=True)
_img_load_thread     = threading.Thread(target=_load_img_model, daemon=True)
_txt_load_thread.start()
_img_load_thread.start()

def _looks_degenerate(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 6:
        return False
    run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        if run >= 4:
            return True
    if len(words) >= 12 and len(set(words)) / len(words) < 0.4:
        return True
    return False

GUEST_MAX_IMAGES = 3
USER_MAX_IMAGES  = 7

def _caption_image(filepath: str) -> Tuple[Optional[dict], Optional[str]]:
    if not _img_ready:
        return None, ('The image AI model is still downloading/loading (first run only — can take '
                       'a few minutes, not just seconds, depending on your connection). Check your '
                       'server console for progress, or GET /api/diagnose for live status.')

    img_hash = None
    try:
        img_hash = _hash_file(filepath)
        cached = _cache_get(f'caption:{IMAGE_MODEL_ID}:{img_hash}')
        if cached is not None:
            cached = dict(cached)
            cached['filename'] = os.path.basename(filepath)
            return cached, None
    except Exception as e:
        log.warning(f'Image cache lookup skipped ({e}) — captioning normally.')

    try:
        import torch
        from PIL import Image as PILImage
        img = PILImage.open(filepath).convert('RGB')
        num_beams  = 4 if _img_device is not None and _img_device.type == 'cuda' else 2
        gen_kwargs = dict(num_beams=num_beams, repetition_penalty=1.4, no_repeat_ngram_size=3)

        def _do_caption():
            with torch.inference_mode():
                inp = _img_processor(img, return_tensors='pt').to(_img_device)
                out = _img_model.generate(**inp, max_new_tokens=60, **gen_kwargs,
                                           output_scores=True, return_dict_in_generate=True)
                base = _img_processor.decode(out.sequences[0], skip_special_tokens=True).strip()
                try:
                    confidence = math.exp(float(out.sequences_scores[0]))
                except Exception:
                    confidence = None

                inp2 = _img_processor(img, 'a photography of', return_tensors='pt').to(_img_device)
                out2 = _img_model.generate(**inp2, max_new_tokens=80, **gen_kwargs)
                detail = _img_processor.decode(out2[0], skip_special_tokens=True).strip()
                if _looks_degenerate(detail):
                    detail = ''

                inp3 = _img_processor(img, 'the setting is', return_tensors='pt').to(_img_device)
                out3 = _img_model.generate(**inp3, max_new_tokens=50, **gen_kwargs)
                setting = _img_processor.decode(out3[0], skip_special_tokens=True).strip()
                if _looks_degenerate(setting) or (setting and setting.lower() in (base + ' ' + detail).lower()):
                    setting = ''
                return base, detail, setting, confidence

        with _model_lock_or_busy('Image captioning'):
            base, detail, setting, confidence = _run_watched(_do_caption, label='Image captioning')

        is_low_confidence = confidence is not None and confidence < CONFIDENCE_LOW_THRESHOLD
        is_very_low_confidence = confidence is not None and confidence < CONFIDENCE_VERY_LOW_THRESHOLD
        if is_very_low_confidence or not base:
            overview = 'The image cannot be confidently identified.'
        elif is_low_confidence:
            overview = f'This image may show {base}'
        else:
            overview = (base[:1].upper() + base[1:])
        if not overview.endswith('.'): overview += '.'
        if detail and detail.lower() not in base.lower() and base.lower() not in detail.lower():
            detail = (detail[:1].upper() + detail[1:])
            if not detail.endswith('.'): detail += '.'
        else:
            detail = ''
        if setting:
            setting = (setting[:1].upper() + setting[1:])
            if not setting.endswith('.'): setting += '.'

        result = {'overview': overview, 'detail': detail, 'setting': setting,
                  'filename': os.path.basename(filepath),
                  'confidence': confidence, 'low_confidence': is_low_confidence,
                  'very_low_confidence': is_very_low_confidence,
                  'raw_caption': base}
        if img_hash:
            _cache_set(f'caption:{IMAGE_MODEL_ID}:{img_hash}', result)
        return result, None
    except Exception as e:
        return None, str(e)

_CAPTION_PREFIX_RE = re.compile(r'^(a|an|the)\s+(photo\w*|image|picture|close-?up|drawing|painting)\s+of\s+', re.I)

def _caption_subject(caption: str) -> str:
    s = _CAPTION_PREFIX_RE.sub('', (caption or '').strip())
    return s.rstrip('.').strip() or (caption or '').strip()

_STOPWORDS_IMG = {
    'a','an','the','this','that','these','those','is','are','was','were','of','in','on','with',
    'and','or','to','at','by','it','its','as','be','being','been','from','into','over','under',
    'photo','photograph','photography','image','picture','close','closeup','shows','showing','up',
}

def _keyword_bullets(*texts: str, max_n: int = 6) -> list:
    seen, out = set(), []
    for t in texts:
        for w in re.sub(r'[^A-Za-z0-9\s-]', ' ', t or '').split():
            wl = w.lower()
            if wl in _STOPWORDS_IMG or len(wl) < 3 or wl in seen:
                continue
            seen.add(wl); out.append(w)
    return out[:max_n]

_IMAGE_CATEGORY_KEYWORDS = {
    'Astronomy':    {'galaxy', 'nebula', 'planet', 'star', 'moon', 'comet', 'telescope', 'space', 'astronaut',
                      'satellite', 'solar', 'orbit', 'cosmic', 'mars', 'jupiter', 'saturn'},
    'Animals':      {'dog', 'cat', 'bird', 'animal', 'wildlife', 'fish', 'horse', 'lion', 'tiger', 'bear',
                      'elephant', 'insect', 'butterfly', 'reptile', 'mammal', 'zoo', 'pet'},
    'Plants':       {'plant', 'flower', 'tree', 'leaf', 'garden', 'forest', 'botanical', 'blossom', 'grass', 'fruit'},
    'Landmarks':    {'building', 'monument', 'tower', 'bridge', 'castle', 'statue', 'landmark', 'temple',
                      'cathedral', 'skyline', 'city', 'street', 'plaza'},
    'Vehicles':     {'car', 'truck', 'motorcycle', 'bicycle', 'airplane', 'boat', 'ship', 'train', 'vehicle', 'bus'},
    'Electronics':  {'computer', 'laptop', 'phone', 'smartphone', 'circuit', 'device', 'gadget', 'camera', 'monitor'},
    'Documents':    {'document', 'text', 'paper', 'letter', 'form', 'certificate', 'receipt', 'invoice', 'page'},
    'Medical':      {'medical', 'xray', 'x-ray', 'scan', 'mri', 'doctor', 'hospital', 'medicine', 'anatomy', 'surgery'},
    'Programming':  {'code', 'screenshot', 'terminal', 'programming', 'software', 'website', 'interface', 'app', 'ui'},
    'Architecture': {'architecture', 'interior', 'room', 'house', 'roof', 'wall', 'window', 'staircase', 'floor'},
    'Artwork':      {'painting', 'sculpture', 'artwork', 'canvas', 'artist', 'gallery', 'museum', 'illustration',
                      'drawing', 'portrait', 'mural', 'fresco', 'masterpiece'},
}
_ENRICHMENT_EXCLUDED_CATEGORIES = {'Documents', 'Programming', 'Medical'}

def _classify_image_category(caption_text: str) -> Optional[str]:
    words = set(re.findall(r'[a-z]+', (caption_text or '').lower()))
    scores = {cat: len(words & kws) for cat, kws in _IMAGE_CATEGORY_KEYWORDS.items()}
    best_cat, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_cat if best_score > 0 else None

def _wiki_lookup(query: str, timeout: int = 6) -> Optional[dict]:
    query = (query or '').strip()
    if not query: return None
    cache_key = f'wiki:{query.lower()}'
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached or None
    try:
        _try_import('requests')
        import requests
        headers = {'User-Agent': 'NovaBrief/1.0 (image-summary-enrichment)'}
        sr = requests.get('https://en.wikipedia.org/w/api.php', timeout=timeout, headers=headers,
                           params={'action': 'query', 'list': 'search', 'srsearch': query,
                                   'format': 'json', 'srlimit': 1})
        hits = ((sr.json().get('query') or {}).get('search')) or []
        if not hits:
            _cache_set(cache_key, {}); return None
        title = hits[0]['title']
        ex = requests.get(f'https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}',
                           timeout=timeout, headers=headers)
        if ex.status_code != 200:
            return None
        data = ex.json()
        if data.get('type') == 'disambiguation':
            _cache_set(cache_key, {}); return None
        extract = (data.get('extract') or '').strip()
        if not extract:
            _cache_set(cache_key, {}); return None
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', extract) if s.strip()]
        if not sentences:
            _cache_set(cache_key, {}); return None
        result = {'title': data.get('title', title), 'sentences': sentences}
        _cache_set(cache_key, result)
        return result
    except Exception as e:
        log.info(f'Wikipedia enrichment skipped ({query[:40]!r}): {e}')
        return None

def _build_image_sections(cap: dict, full: bool = True) -> Tuple[list, Optional[dict]]:
    sections = []
    if cap['detail']:
        sections.append(('Visual Description', cap['detail'], False))
    if cap.get('setting'):
        sections.append(('Setting & Composition', cap['setting'], False))

    elements = _keyword_bullets(cap['overview'], cap['detail'], cap.get('setting', ''))
    if elements:
        sections.append(('Detected Elements', '\n'.join(f'- {e}' for e in elements), True))

    raw = cap.get('raw_caption', cap['overview'])
    category = _classify_image_category(raw)
    wiki = None
    if cap.get('low_confidence'):
        pass
    elif category == 'Medical':
        sections.append(('Note', 'This appears to be medical imagery. NovaBrief does not attempt automated '
                                  'identification for medical content — for accurate interpretation, consult '
                                  'a qualified healthcare professional.', False))
    elif category in _ENRICHMENT_EXCLUDED_CATEGORIES:
        pass
    else:
        wiki = _wiki_lookup(_caption_subject(raw))
    if wiki:
        sents = wiki['sentences']
        split = max(1, round(len(sents) * 0.45))
        context_part, tech_part = sents[:split], sents[split:]
        if context_part:
            sections.append(('Context & Interpretation',
                              f"Likely related to **{wiki['title']}**. " + ' '.join(context_part), False))
        if tech_part:
            sections.append(('Technical & Scientific Background', ' '.join(tech_part), False))

    if full:
        obs = [cap['overview']] + ([cap['detail']] if cap['detail'] else []) + \
              ([cap['setting']] if cap.get('setting') else []) + \
              ([wiki['sentences'][0]] if wiki else [])
        obs = list(dict.fromkeys(o.rstrip('.') + '.' for o in obs if o))[:5]
        if obs:
            sections.append(('Key Observations', '\n'.join(f'- {o}' for o in obs), True))
        conclusion = cap['overview']
        if wiki:
            conclusion += f" Background research points to {wiki['title']} as the most likely subject."
        sections.append(('Conclusion', conclusion, False))

    return sections, wiki

def _assemble_sections(overview: str, sections: list) -> str:
    out = (overview or '').strip()
    for heading, body, _ in sections:
        out += f'\n\n## {heading}\n{body}'
    return out

def summarize_image(filepath: str) -> Tuple[Optional[str], Optional[str]]:
    cap, err = _caption_image(filepath)
    if err: return None, err
    sections, _ = _build_image_sections(cap, full=True)
    return _assemble_sections(cap['overview'], sections), None

def summarize_images(filepaths: list) -> Tuple[Optional[str], Optional[str]]:
    if len(filepaths) == 1:
        return summarize_image(filepaths[0])

    caps, errors = [], []
    for fp in filepaths:
        cap, err = _caption_image(fp)
        if cap: caps.append(cap)
        else: errors.append(f'{os.path.basename(fp)}: {err}')
    if not caps:
        return None, '; '.join(errors) or 'No images could be analyzed.'

    with ThreadPoolExecutor(max_workers=min(8, len(caps))) as ex:
        per_image = list(ex.map(lambda c: _build_image_sections(c, full=False), caps))

    out = f"{len(caps)} image{'s' if len(caps) != 1 else ''} analyzed."
    wiki_titles = []
    for i, (cap, (sub_sections, wiki)) in enumerate(zip(caps, per_image), 1):
        if wiki: wiki_titles.append(wiki['title'])
        body = cap['overview'] + (('\n\n' + cap['detail']) if cap['detail'] else '')
        if cap.get('setting'):
            body += '\n\n' + cap['setting']
        for heading, sec_body, _ in sub_sections:
            if heading in ('Context & Interpretation', 'Technical & Scientific Background'):
                body += f'\n\n{heading}: {sec_body}'
        out += f"\n\n## Image {i} \u2014 {cap['filename']}\n{body}"

    takeaways = list(dict.fromkeys(cap['overview'].rstrip('.') + '.' for cap in caps))[:7]
    out += '\n\n## Key Observations\n' + '\n'.join(f'- {t}' for t in takeaways)
    conclusion = f"Across the {len(caps)} images"
    conclusion += (', recurring subjects include ' + ', '.join(dict.fromkeys(wiki_titles)) + '.') if wiki_titles else '.'
    out += f'\n\n## Conclusion\n{conclusion}'
    if errors:
        out += '\n\n## Notes\n' + '\n'.join(f'- {e}' for e in errors)
    return out, None

WHISPER_MODEL_SIZE = ((cfg.WHISPER_MODEL_SIZE if cfg and getattr(cfg, 'WHISPER_MODEL_SIZE', None) else None)
                       or os.environ.get('NOVABRIEF_WHISPER_MODEL', 'small'))
_whisper_model       = None
_whisper_batched     = None
_whisper_ready       = False
_whisper_error       = None
_whisper_device_label = None
_whisper_lock        = threading.Lock()

def _construct_whisper_model(device: str, compute_type: str, timeout: int = 180):
    box = {}
    def _work():
        try:
            from faster_whisper import WhisperModel
            box['model'] = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
        except Exception as e:
            box['error'] = e
    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(timeout=timeout)
    if th.is_alive():
        raise RuntimeError(
            f'faster-whisper did not finish initializing on {device} within {timeout}s — it is still '
            f'running in the background, but something is stuck rather than erroring out cleanly. The '
            f'most common cause is a CUDA/cuDNN version ctranslate2 does not support: it needs its OWN '
            f'specific versions (recent releases need CUDA 12 + cuDNN 9), separate from whatever CUDA '
            f'version your PyTorch install uses — the two can genuinely disagree on the same machine. '
            f'To confirm: set FORCE_CPU = True in config.py and restart — if it loads fine on CPU, this '
            f'is a GPU/CUDA/cuDNN mismatch. Fix by matching ctranslate2 to your driver, e.g. '
            f'`pip install --force-reinstall ctranslate2==4.4.0` for CUDA 12 + cuDNN 8, or '
            f'`ctranslate2==3.24.0` for CUDA 11 + cuDNN 8 (run `nvidia-smi` to see your CUDA version).')
    if 'error' in box:
        raise box['error']
    return box['model']

def _load_whisper_model():
    global _whisper_model, _whisper_batched, _whisper_ready, _whisper_error, _whisper_device_label
    with _whisper_lock:
        if _whisper_ready: return
        _whisper_error = None
        _t0 = time.perf_counter()
        try:
            _try_import('faster_whisper', 'faster-whisper')
            force_cpu = bool(cfg and getattr(cfg, 'FORCE_CPU', False))
            use_cuda  = (not force_cpu) and _has_nvidia_gpu()
            device = 'cuda' if use_cuda else 'cpu'
            compute_type = 'int8_float16' if use_cuda else 'int8'
            log.info(f'Loading transcription model: faster-whisper "{WHISPER_MODEL_SIZE}" ({device}, {compute_type})')
            try:
                _whisper_model = _construct_whisper_model(device, compute_type)
                _whisper_device_label = f'GPU · {compute_type}' if use_cuda else f'CPU · {compute_type}'
            except Exception as e:
                if not use_cuda:
                    raise
                log.warning(f'GPU load of faster-whisper failed ({e}) — falling back to CPU/int8. '
                            f'Common cause: ctranslate2 has no build for this machine\'s CUDA/cuDNN version. '
                            f'See OPTIMIZATION_REPORT.md.')
                _whisper_model = _construct_whisper_model('cpu', 'int8')
                _whisper_device_label = 'CPU · int8 (GPU load failed — see log)'
            _whisper_batched = None
            if use_cuda and _whisper_model is not None:
                try:
                    from faster_whisper import BatchedInferencePipeline
                    _whisper_batched = BatchedInferencePipeline(model=_whisper_model)
                    log.info('  ✅ Batched GPU transcription pipeline ready (faster-whisper BatchedInferencePipeline)')
                except ImportError:
                    log.info('  (BatchedInferencePipeline not available in this faster-whisper version — '
                              'using the standard per-file transcription path; still fully functional, just '
                              'without the extra GPU batching speedup. pip install -U faster-whisper for it.)')
                except Exception as e:
                    log.warning(f'  Could not build BatchedInferencePipeline ({e}) — using the standard path.')
            _whisper_ready = True
            log.info(f'✅ Transcription model ready: faster-whisper "{WHISPER_MODEL_SIZE}" · {_whisper_device_label}'
                     f'  ·  loaded in {time.perf_counter()-_t0:.1f}s  ·  {_resource_usage_str()}')
        except Exception as e:
            _whisper_error = str(e)
            log.error(f'Whisper model failed to load (after {time.perf_counter()-_t0:.1f}s): {e}')

_whisper_load_thread = threading.Thread(target=_load_whisper_model, daemon=True)
_whisper_load_thread.start()

def _transcribe_audio_file(wav_path: str, language: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    if not _whisper_ready:
        if _whisper_error:
            return None, f'The transcription model failed to load: {_whisper_error}'
        return None, ('The transcription model is still downloading/loading (first run only — '
                       'installs faster-whisper and downloads the model, which can take a few '
                       'minutes depending on your connection, not just seconds). Check your server '
                       'console for progress, or GET /api/diagnose for live status. Try again shortly.')
    try:
        def _do_transcribe():
            beam_size = 5 if (_whisper_device_label or '').startswith('GPU') else 1
            common_kwargs = dict(
                language=language,
                vad_filter=True,
                beam_size=beam_size,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            if _whisper_batched is not None:
                segments, _info = _whisper_batched.transcribe(wav_path, batch_size=8, **common_kwargs)
            else:
                segments, _info = _whisper_model.transcribe(wav_path, **common_kwargs)
            return [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]

        watchdog_seconds = max(INFERENCE_WATCHDOG_SECONDS, _max_transcribe_seconds() * 3)
        with _model_lock_or_busy('Audio transcription'):
            parts = _run_watched(_do_transcribe, timeout=watchdog_seconds, label='Audio transcription')
        text = ' '.join(parts).strip()
        return (text, None) if text else (None, 'No speech could be detected in this file.')
    except Exception as e:
        log.exception(f'[whisper] transcription failed for {wav_path!r}')
        return None, str(e)

def detect_lang(text: str) -> str:
    try:
        _try_import('langdetect')
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text[:2000]) or 'en'
    except: return 'en'

def translate_text(text: str, source: str, target: str) -> str:
    if not text or source == target: return text
    try:
        _try_import('deep_translator', 'deep-translator')
        from deep_translator import GoogleTranslator
        MAX = 4500
        if len(text) <= MAX:
            return GoogleTranslator(source=source, target=target).translate(text) or text
        sents = re.split(r'(?<=[.!?])\s+', text)
        chunks, cur = [], ''
        for s in sents:
            if len(cur) + len(s) > MAX and cur:
                chunks.append(cur.strip()); cur = s
            else:
                cur += (' ' if cur else '') + s
        if cur: chunks.append(cur.strip())
        tr = GoogleTranslator(source=source, target=target)
        return ' '.join(tr.translate(c) or c for c in chunks)
    except Exception as e:
        log.warning(f'Translation error: {e}')
        return text

def _is_bullet_body(body: str) -> bool:
    lines = [l.strip() for l in (body or '').split('\n') if l.strip()]
    if not lines: return False
    marked = sum(1 for l in lines if l[:1] in ('-', '\u2022', '*'))
    return (marked / len(lines)) >= 0.6

def _parse_structured(raw: str) -> dict:
    raw = raw or ''
    parts = re.split(r'\n##\s+', raw)
    overview = parts[0].strip()
    sections = []
    for part in parts[1:]:
        nl = part.find('\n')
        heading = (part if nl == -1 else part[:nl]).strip()
        body    = '' if nl == -1 else part[nl+1:].strip()
        sections.append({'heading': heading, 'body': body, 'is_list': _is_bullet_body(body)})
    return {'overview': overview, 'sections': sections}

def _render_structured(parsed: dict) -> str:
    out = (parsed.get('overview') or '').strip()
    for sec in parsed.get('sections', []):
        out += f"\n\n## {sec['heading']}\n{sec['body']}"
    return out

def translate_structured(raw: str, source: str, target: str) -> str:
    if not raw or source == target: return raw
    try:
        parsed = _parse_structured(raw)
        parsed['overview'] = translate_text(parsed['overview'], source, target) if parsed['overview'] else ''
        for sec in parsed['sections']:
            sec['heading'] = translate_text(sec['heading'], source, target)
            if sec['is_list']:
                items = [l.strip().lstrip('-\u2022*').strip() for l in sec['body'].split('\n') if l.strip()]
                items = [translate_text(it, source, target) for it in items]
                sec['body'] = '\n'.join(f'- {it}' for it in items if it)
            else:
                sec['body'] = translate_text(sec['body'], source, target)
        return _render_structured(parsed)
    except Exception as e:
        log.warning(f'Structured translation error: {e}')
        return translate_text(raw, source, target)

_YT_ID_PATS = [r'(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})', r'^([A-Za-z0-9_-]{11})$']

def _youtube_video_id(url: str) -> Optional[str]:
    return next((m.group(1) for p in _YT_ID_PATS for m in [re.search(p, url)] if m), None)

def _youtube_transcript_api(vid: str) -> Tuple[Optional[str], Optional[str], str]:
    def _fetch():
        _try_import('youtube_transcript_api', 'youtube-transcript-api')
        from youtube_transcript_api import YouTubeTranscriptApi
        fetched = None
        if not hasattr(YouTubeTranscriptApi, 'get_transcript'):
            api = YouTubeTranscriptApi()
            for langs in (['en'], ['en-US', 'en-GB'], ['a.en']):
                try: fetched = api.fetch(vid, languages=langs); break
                except: pass
            if fetched is None:
                fetched = list(api.list(vid))[0].fetch()
        else:
            fetched = YouTubeTranscriptApi.get_transcript(vid)
        parts = []
        for s in fetched:
            seg = (s.text if hasattr(s, 'text') else s.get('text', '')).strip()
            if not seg:
                continue
            if not seg[-1] in '.!?':
                seg += '.'
            parts.append(seg)
        return ' '.join(parts).strip()

    try:
        text = _run_watched(_fetch, timeout=NETWORK_WATCHDOG_SECONDS, label='YouTube transcript fetch')
        return (text, None, '') if text else (None, 'Empty transcript.', 'EmptyTranscript')
    except Exception as e:
        return None, str(e), type(e).__name__

def _youtube_audio_fallback(vid: str, url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        _try_import('yt_dlp', 'yt-dlp')
        import yt_dlp
    except Exception as e:
        return None, f'This video has no captions, and the audio-download fallback is unavailable ({e}).'

    if not _ensure_ffmpeg():
        return None, _ffmpeg_hint('no working ffmpeg binary found (needed for the no-captions audio fallback)')

    max_minutes = int(cfg.MAX_YOUTUBE_FALLBACK_MINUTES) if cfg and getattr(cfg, 'MAX_YOUTUBE_FALLBACK_MINUTES', None) else 60
    out_base = os.path.join(UPLOAD_FOLDER, f'yt_fallback_{vid}_{os.urandom(3).hex()}')
    expected_wav = out_base + '.wav'
    ydl_opts = {
        'format'         : 'bestaudio/best',
        'outtmpl'        : out_base + '.%(ext)s',
        'ffmpeg_location': os.path.dirname(_ffmpeg_path),
        'postprocessors' : [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '0'}],
        'noplaylist'     : True,
        'quiet'          : True,
        'no_warnings'    : True,
        'socket_timeout' : 20,
        'retries'        : 3,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = _run_watched(lambda: ydl.extract_info(url, download=False),
                                 timeout=NETWORK_WATCHDOG_SECONDS, label='YouTube metadata fetch')
            duration = info.get('duration') or 0
            if duration and duration > max_minutes * 60:
                return None, (f'This video is about {duration // 60} minutes long, past the '
                               f'{max_minutes}-minute limit for the no-captions transcription fallback '
                               f'(MAX_YOUTUBE_FALLBACK_MINUTES in config.py). Try a shorter video.')
            download_timeout = max(NETWORK_WATCHDOG_SECONDS, max_minutes * 60 * 2)
            _run_watched(lambda: ydl.download([url]), timeout=download_timeout, label='YouTube audio download')
        if not os.path.isfile(expected_wav):
            return None, 'Audio download completed but the expected output file was not found.'
        return _transcribe_audio_file(expected_wav)
    except Exception as e:
        log.exception(f'[youtube-fallback] failed for {url!r}')
        return None, f'No captions were available, and the audio-download fallback also failed: {e}'
    finally:
        try: os.remove(expected_wav)
        except OSError: pass

def extract_youtube(url: str) -> Tuple[Optional[str], Optional[str]]:
    vid = _youtube_video_id(url)
    if not vid:
        return None, 'Cannot extract video ID from URL.'

    text, err_msg, err_type = _youtube_transcript_api(vid)
    if text:
        return text, None

    if 'VideoUnavailable' in err_type:
        return None, 'That video is unavailable, private, or region-restricted.'

    log.info(f'No transcript for video {vid} ({err_msg}) — falling back to audio download + Whisper.')
    fb_text, fb_err = _youtube_audio_fallback(vid, url)
    if fb_text:
        return fb_text, None

    hint = ('This video has no captions/transcript available.'
            if ('TranscriptsDisabled' in err_type or 'NoTranscriptFound' in err_type)
            else f'Could not fetch a transcript ({err_msg}).')
    return None, f'{hint} The audio-download fallback also failed: {fb_err}'

_LIST_MARKER_RE  = re.compile(r'^\s*(?:[-•*▪‣]|\d{1,3}[.)]|[a-zA-Z][.)])\s+')
_HEADING_LEN_MAX = 70

def _looks_like_heading(line: str, prev_blank: bool, next_blank: bool) -> bool:
    s = line.strip()
    if not s or len(s) > _HEADING_LEN_MAX:
        return False
    if _LIST_MARKER_RE.match(s):
        return False
    if s[-1] in '.,;:':
        return False
    if not (prev_blank or next_blank):
        return False
    word_count = len(s.split())
    if word_count > 12:
        return False
    return s.isupper() or s.istitle() or word_count <= 8

def _preserve_pdf_structure(raw_text: str) -> str:
    lines = raw_text.split('\n')
    out = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            out.append('')
            continue
        prev_blank = (i == 0) or not lines[i - 1].strip()
        next_blank = (i == len(lines) - 1) or not lines[i + 1].strip()
        if _looks_like_heading(s, prev_blank, next_blank):
            out.append('')
            out.append(f'## {s}')
            out.append('')
        elif _LIST_MARKER_RE.match(s):
            out.append(s)
        else:
            out.append(s)
    text = '\n'.join(out)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def extract_pdf(path: str) -> Tuple[Optional[str], Optional[str]]:
    def _read():
        _try_import('pypdf')
        from pypdf import PdfReader
        reader = PdfReader(path)
        return '\n'.join(p.extract_text() or '' for p in reader.pages).strip()
    try:
        text = _run_watched(_read, timeout=NETWORK_WATCHDOG_SECONDS, label='PDF extraction')
        if not text:
            return None, 'PDF has no extractable text (may be scanned).'
        return _preserve_pdf_structure(text), None
    except Exception as e: return None, str(e)

_ffmpeg_ready  = False
_ffmpeg_path   = None
_ffmpeg_source = None
_ffmpeg_error  = None

def _ffmpeg_binary_works(exe_path: str) -> bool:
    try:
        r = subprocess.run([exe_path, '-version'], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False

def _detect_decoy_ffmpeg_package():
    try:
        import importlib.metadata as _im
        return _im.version('ffmpeg')
    except Exception:
        return None

_pydub_mediainfo_json_original = None

def _patch_pydub_prober():
    global _pydub_mediainfo_json_original
    import shutil
    import pydub.audio_segment as _pas
    if _pydub_mediainfo_json_original is None:
        _pydub_mediainfo_json_original = _pas.mediainfo_json

    def _safe_mediainfo_json(filepath, read_ahead_limit=-1):
        if shutil.which('avprobe') or shutil.which('ffprobe'):
            return _pydub_mediainfo_json_original(filepath, read_ahead_limit=read_ahead_limit)
        return {}

    _pas.mediainfo_json = _safe_mediainfo_json

def _make_ffmpeg_discoverable(exe_path: str) -> bool:
    shim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ffmpeg_shim')
    try:
        os.makedirs(shim_dir, exist_ok=True)
        shim_name = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
        shim_path = os.path.join(shim_dir, shim_name)
        needs_refresh = True
        if os.path.exists(shim_path) or os.path.islink(shim_path):
            try:
                if os.path.realpath(shim_path) == os.path.realpath(exe_path) and _ffmpeg_binary_works(shim_path):
                    needs_refresh = False
                else:
                    os.remove(shim_path)
            except OSError:
                pass
        if needs_refresh:
            if os.name == 'nt':
                import shutil as _sh
                _sh.copy2(exe_path, shim_path)
            else:
                try:
                    os.symlink(exe_path, shim_path)
                except OSError:
                    import shutil as _sh
                    _sh.copy2(exe_path, shim_path)
            os.chmod(shim_path, 0o755)
        if shim_dir not in os.environ.get('PATH', '').split(os.pathsep):
            os.environ['PATH'] = shim_dir + os.pathsep + os.environ.get('PATH', '')
        return _ffmpeg_binary_works(shim_path)
    except Exception as e:
        log.warning(f'Could not create an ffmpeg PATH shim ({e}) — ffmpeg extraction will still work '
                    f'(AudioSegment.converter is set directly below regardless), pydub\'s own harmless '
                    f'startup warning about not finding ffmpeg on PATH just won\'t be suppressed.')
        return False

def _ensure_ffmpeg(force_recheck: bool = False) -> bool:
    global _ffmpeg_ready, _ffmpeg_path, _ffmpeg_source, _ffmpeg_error
    if _ffmpeg_ready and not force_recheck:
        return True
    _ffmpeg_error = None

    _pre_path_exe = None
    try:
        _try_import('imageio_ffmpeg', 'imageio-ffmpeg')
        import imageio_ffmpeg
        _candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.isfile(_candidate) and _ffmpeg_binary_works(_candidate):
            _pre_path_exe = _candidate
    except Exception:
        pass
    if _pre_path_exe is None:
        import shutil as _shutil_pre
        _sys_candidate = _shutil_pre.which('ffmpeg')
        if _sys_candidate and _ffmpeg_binary_works(_sys_candidate):
            _pre_path_exe = _sys_candidate
    if _pre_path_exe:
        _make_ffmpeg_discoverable(_pre_path_exe)

    try:
        _try_import('pydub')
        from pydub import AudioSegment
    except Exception as e:
        _ffmpeg_error = f'pydub is not installed/importable under {sys.executable}: {e}'
        log.warning(_ffmpeg_error)
        _ffmpeg_ready = False
        return False

    try:
        _try_import('imageio_ffmpeg', 'imageio-ffmpeg')
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.isfile(exe) and _ffmpeg_binary_works(exe):
            AudioSegment.converter = exe
            AudioSegment.ffmpeg    = exe
            _patch_pydub_prober()
            _ffmpeg_path, _ffmpeg_source, _ffmpeg_ready = exe, 'imageio-ffmpeg', True
            log.info(f'✅ ffmpeg ready (bundled via imageio-ffmpeg): {exe}')
            return True
        _ffmpeg_error = (
            f'imageio-ffmpeg is importable and returned a path ({exe}), but that file '
            f'{"does not exist" if not os.path.isfile(exe) else "would not run when tested"}. '
            f'On Windows this is almost always antivirus quarantining an unsigned .exe inside '
            f'site-packages — check Windows Defender / your antivirus quarantine/history for '
            f'imageio_ffmpeg or ffmpeg.exe, or exclude the Python site-packages folder from scanning.'
        )
        log.warning(_ffmpeg_error)
    except Exception as e:
        _ffmpeg_error = (
            f'imageio-ffmpeg could not be imported in THIS Python environment '
            f'({sys.executable}): {e}. '
            f'If you installed it via PyCharm, PyCharm may be using a DIFFERENT interpreter '
            f'than the one actually running this server — check PyCharm\'s Settings > Project '
            f'> Python Interpreter, or just run this exact command in the same terminal you '
            f'use to start the server: "{sys.executable}" -m pip install --upgrade imageio-ffmpeg'
        )
        log.warning(_ffmpeg_error)

    import shutil
    sys_exe = shutil.which('ffmpeg')
    if sys_exe and _ffmpeg_binary_works(sys_exe):
        AudioSegment.converter = sys_exe
        AudioSegment.ffmpeg    = sys_exe
        _patch_pydub_prober()
        _ffmpeg_path, _ffmpeg_source, _ffmpeg_ready = sys_exe, 'system', True
        log.info(f'✅ ffmpeg ready (system PATH): {sys_exe}')
        return True

    decoy_ver = _detect_decoy_ffmpeg_package()
    if decoy_ver:
        _ffmpeg_error = (
            (_ffmpeg_error + '\n\n') if _ffmpeg_error else ''
        ) + (
            f'⚠ Likely cause: the "ffmpeg" PyPI package (v{decoy_ver}) is installed. '
            f'Despite the name, this is a defunct 2018 wrapper — it does NOT contain or '
            f'provide any actual ffmpeg binary. It is harmless to leave installed but will '
            f'never fix this on its own. What actually works is a DIFFERENT package, '
            f'imageio-ffmpeg (already in requirements.txt) — it bundles a real ~60MB ffmpeg '
            f'binary. Run in the same terminal you use for "python app.py":\n'
            f'  "{sys.executable}" -m pip install --upgrade imageio-ffmpeg\n'
            f'(optional cleanup: "{sys.executable}" -m pip uninstall ffmpeg -y)'
        )
        log.warning(f'Detected decoy "ffmpeg" PyPI package v{decoy_ver} (not real ffmpeg) — see hint above.')

    _ffmpeg_ready = False
    return False

def _ffmpeg_hint(e) -> str:
    diagnosis = _ffmpeg_error or 'No specific diagnosis captured — see the server terminal log above this request.'
    return (
        f'Audio/video processing failed — no working ffmpeg was found.\n\n'
        f'Diagnosis: {diagnosis}\n\n'
        f'This server process is running under:\n  {sys.executable}\n'
        f'Whatever you pip-install needs to go into THAT exact interpreter — installing inside '
        f'PyCharm\'s package manager can target a different one than the terminal/venv actually '
        f'running app.py, which is the #1 cause of "already satisfied" but still failing.\n\n'
        f'Quickest fix — run this exact line in the same terminal you use for "python app.py":\n'
        f'  "{sys.executable}" -m pip install --upgrade imageio-ffmpeg\n'
        f'Then restart the server (or POST /api/reload-ffmpeg to retry without restarting).\n\n'
        f'Still stuck? Install ffmpeg manually and make sure it is on PATH:\n'
        f'  Ubuntu: sudo apt install ffmpeg\n  macOS: brew install ffmpeg\n'
        f'  Windows: https://ffmpeg.org/download.html\n\n'
        f'Original error: {e}'
    )

def _log_media_diag(stage: str, path: str):
    abspath = os.path.abspath(path)
    try:
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else -1
    except OSError:
        exists, size = False, -1
    log.info(
        f'[media:{stage}] input={abspath!r} exists={exists} size={size}B '
        f'ext={os.path.splitext(path)[1]!r} cwd={os.getcwd()!r} '
        f'ffmpeg_ready={_ffmpeg_ready} ffmpeg_path={_ffmpeg_path!r} ffmpeg_source={_ffmpeg_source!r}'
    )

def _ffmpeg_extract_wav(src_path: str, dst_wav_path: str, max_seconds: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    if not _ensure_ffmpeg():
        return False, _ffmpeg_hint('no working ffmpeg binary found')
    cmd = [_ffmpeg_path, '-y', '-i', src_path, '-vn', '-ac', '1', '-ar', '16000']
    if max_seconds:
        cmd += ['-t', str(max_seconds)]
    cmd += [dst_wav_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0 or not os.path.isfile(dst_wav_path):
            stderr_tail = (result.stderr or '').strip().splitlines()[-1:] or ['ffmpeg exited non-zero with no stderr output']
            log.warning(f'[ffmpeg] extraction failed for {src_path!r}: {(result.stderr or "")[-800:]}')
            return False, (f"ffmpeg couldn't process this file: {stderr_tail[0]}\n\n"
                            f"This usually means the file is corrupted, empty, or an unsupported/unusual "
                            f"codec — not a missing-ffmpeg problem (ffmpeg itself ran fine here). "
                            f"Try re-exporting the file or converting it with another tool first.")
        return True, None
    except subprocess.TimeoutExpired:
        return False, 'Audio extraction timed out (the file may be extremely long).'
    except Exception as e:
        log.exception(f'[ffmpeg] extraction crashed for {src_path!r}')
        return False, str(e)

def _max_transcribe_seconds() -> int:
    minutes = int(cfg.MAX_TRANSCRIBE_MINUTES) if cfg and getattr(cfg, 'MAX_TRANSCRIBE_MINUTES', None) else 120
    return minutes * 60

_SRT_INDEX_RE = re.compile(r'^\d+$')
_SRT_TIME_RE  = re.compile(r'-->')

def _parse_srt(srt_text: str) -> str:
    parts = []
    for line in srt_text.split('\n'):
        line = line.strip()
        if not line or _SRT_INDEX_RE.match(line) or _SRT_TIME_RE.search(line):
            continue
        line = re.sub(r'<[^>]+>', '', line)
        if line and line[-1] not in '.!?':
            line += '.'
        parts.append(line)
    return ' '.join(parts).strip()

def _extract_embedded_subtitles(path: str) -> Optional[str]:
    if not _ffmpeg_path:
        return None
    srt_path = path + '_embedded.srt'
    try:
        r = subprocess.run(
            [_ffmpeg_path, '-y', '-i', path, '-map', '0:s:0', '-f', 'srt', srt_path],
            capture_output=True, text=True, timeout=30)
        if r.returncode != 0 or not os.path.isfile(srt_path) or os.path.getsize(srt_path) == 0:
            return None
        with open(srt_path, encoding='utf-8', errors='replace') as f:
            text = _parse_srt(f.read())
        return text if len(text.split()) >= 5 else None
    except Exception as e:
        log.info(f'[media:extract_video] no usable embedded subtitle track ({e}) — using Whisper instead')
        return None
    finally:
        try: os.remove(srt_path)
        except OSError: pass

def _extract_and_transcribe(path: str, stage_label: str) -> Tuple[Optional[str], Optional[str]]:
    _log_media_diag(stage_label, path)
    wav_path = path + '_16k.wav'
    try:
        ok, err = _ffmpeg_extract_wav(path, wav_path, max_seconds=_max_transcribe_seconds())
        if not ok:
            return None, err
        return _transcribe_audio_file(wav_path)
    except Exception as e:
        log.exception(f'[media:{stage_label}] unexpected failure for {path!r}')
        return None, str(e)
    finally:
        try: os.remove(wav_path)
        except OSError: pass

def extract_audio(path: str) -> Tuple[Optional[str], Optional[str]]:
    return _extract_and_transcribe(path, 'extract_audio')

def extract_video(path: str) -> Tuple[Optional[str], Optional[str]]:
    embedded = _extract_embedded_subtitles(path)
    if embedded:
        log.info('[media:extract_video] using embedded subtitle track — skipping audio extraction + Whisper entirely')
        return embedded, None
    return _extract_and_transcribe(path, 'extract_video')

def text_to_speech(text: str, lang: str = 'en') -> Tuple[Optional[str], Optional[str]]:
    lang_code = lang.split('-')[0].lower()
    try:
        _try_import('gtts')
        from gtts import gTTS
        tts  = gTTS(text=text, lang=lang_code, slow=False)
        path = os.path.join(UPLOAD_FOLDER, f'audio_{lang_code}_{abs(hash(text[:30]))}.mp3')
        tts.save(path)
        return path, None
    except Exception as e:
        return None, f'TTS failed (gTTS needs internet): {e}'

def save_summary(user_id, source_type, source_info, orig_lang, word_count,
                 summary, sum_lang, method, audio_filename=None) -> Tuple[Optional[int], Optional[str]]:
    if not user_id:
        return None, None
    if not _db_ready:
        return None, 'Database not connected, so this summary was not saved to your history. Check GET /api/db-status.'
    try:
        title = f"{source_type.title()} · {(source_info or 'untitled')[:80]}"
        conn  = _get_db(); cur = conn.cursor()
        cur.execute("""INSERT INTO summaries
            (user_id,title,source_type,source_info,original_language,
             original_word_count,summary_text,summary_language,method,audio_filename)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (user_id, title, source_type, source_info, orig_lang,
             word_count, summary, sum_lang, method, audio_filename))
        sid = cur.lastrowid; conn.commit(); cur.close(); conn.close()
        return sid, None
    except Exception as e:
        log.error(f'save_summary: {e}')
        return None, f'Could not save to history: {e}'

@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    if not _db_ready: return jsonify({'success':False,'error':'Database unavailable'})
    d = request.get_json() or {}
    name, email, pw = d.get('name','').strip(), d.get('email','').strip().lower(), d.get('password','')
    if not name or not email or not pw:
        return jsonify({'success':False,'error':'Name, email and password are required'})
    if len(pw) < 6:
        return jsonify({'success':False,'error':'Password must be at least 6 characters'})
    try:
        conn = _get_db(); cur = conn.cursor()
        cur.execute("INSERT INTO users (name,email,password_hash) VALUES (%s,%s,%s)",
                    (name, email, _hash_pw(pw)))
        uid = cur.lastrowid; conn.commit(); cur.close(); conn.close()
        session['user_id'] = uid; session['user_name'] = name; session['user_email'] = email
        return jsonify({'success':True,'user':{'id':uid,'name':name,'email':email,'is_admin':_is_admin(email)}})
    except Exception as e:
        msg = 'Email already registered' if 'Duplicate' in str(e) else str(e)
        return jsonify({'success':False,'error':msg})

@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    if not _db_ready: return jsonify({'success':False,'error':'Database unavailable'})
    d = request.get_json() or {}
    email, pw = d.get('email','').strip().lower(), d.get('password','')
    if not email or not pw:
        return jsonify({'success':False,'error':'Email and password required'})
    try:
        conn = _get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone(); cur.close(); conn.close()
        if not user or not _check_pw(pw, user['password_hash']):
            return jsonify({'success':False,'error':'Invalid email or password'})
        session['user_id'] = user['id']; session['user_name'] = user['name']; session['user_email'] = user['email']
        return jsonify({'success':True,'user':{'id':user['id'],'name':user['name'],'email':user['email'],'is_admin':_is_admin(user['email'])}})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    session.clear()
    return jsonify({'success':True})

@app.route('/api/auth/me')
def auth_me():
    if 'user_id' not in session:
        return jsonify({'logged_in':False})
    email = session.get('user_email','')
    return jsonify({'logged_in':True,'user':{
        'id': session['user_id'], 'name': session['user_name'],
        'email': email, 'is_admin': _is_admin(email),
    }})

ADMIN_EMAILS = set(e.lower() for e in ((cfg.ADMIN_EMAILS if cfg and hasattr(cfg,'ADMIN_EMAILS') else None) or []))

def _is_admin(email: str) -> bool:
    return bool(email) and email.lower() in ADMIN_EMAILS

def admin_required(f):
    @wraps(f)
    def decorated(*a, **kw):
        if not _is_admin(session.get('user_email','')):
            return jsonify({'success': False, 'error': 'Admin access required'}), 403
        return f(*a, **kw)
    return decorated

@app.route('/api/admin/users')
@login_required
@admin_required
def admin_users():
    try:
        conn = _get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT u.id, u.name, u.email, u.created_at,
                   COUNT(s.id) AS summary_count,
                   MAX(s.created_at) AS last_active
            FROM users u LEFT JOIN summaries s ON s.user_id = u.id
            GROUP BY u.id ORDER BY u.created_at DESC
        """)
        rows = cur.fetchall(); cur.close(); conn.close()
        for r in rows:
            r['is_admin'] = _is_admin(r['email'])
            if r.get('created_at'):  r['created_at']  = r['created_at'].strftime('%d %b %Y, %H:%M')
            if r.get('last_active'): r['last_active'] = r['last_active'].strftime('%d %b %Y, %H:%M')
        return jsonify({'success': True, 'users': rows, 'total': len(rows)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/summarize', methods=['POST'])
def summarize():
    src  = request.form.get('source_type','').strip()
    out  = request.form.get('output_type','text').strip()
    lang = request.form.get('target_lang','auto').strip()
    t    = Timings(f'summarize:{src or "?"}')
    t.set_meta(input_type=src or 'unknown')

    raw, error = None, None
    cache_id       = None
    extract_cached = False

    if src == 'text':
        raw = request.form.get('text_content','').strip()
        if not raw: return jsonify({'success':False,'error':'No text provided.'})
        src_info = raw[:80]+'…'
        cache_id = _hash_bytes(raw.encode('utf-8'))

    elif src == 'youtube':
        url = request.form.get('youtube_url','').strip()
        if not url: return jsonify({'success':False,'error':'No URL provided.'})
        src_info = url
        cache_id = _youtube_video_id(url)
        cached_raw = _cache_get(f'extract:youtube:{cache_id}') if cache_id else None
        if cached_raw is not None:
            raw, error, extract_cached = cached_raw, None, True
            t.mark('extraction', 0.0)
        else:
            with t.stage('extraction'):
                raw, error = extract_youtube(url)
            if raw and not error and cache_id:
                _cache_set(f'extract:youtube:{cache_id}', raw)

    elif src == 'image':
        files = request.files.getlist('files') or ([request.files['file']] if 'file' in request.files else [])
        files = [f for f in files if f and f.filename]
        if not files:
            return jsonify({'success':False,'error':'No image uploaded.'})
        limit = USER_MAX_IMAGES if session.get('user_id') else GUEST_MAX_IMAGES
        if len(files) > limit:
            hint = '' if session.get('user_id') else f' Sign in for up to {USER_MAX_IMAGES}.'
            return jsonify({'success':False,'error':f'Too many images ({len(files)}) — the limit is {limit} at a time.{hint}'})
        fpaths, fnames = [], []
        with t.stage('upload'):
            for f in files:
                fpath, fname, up_err = _safe_upload_path(f.filename, 'image')
                if up_err: return jsonify({'success': False, 'error': up_err})
                f.save(fpath)
                fpaths.append(fpath); fnames.append(fname)
        t.set_meta(file_size_bytes=sum(os.path.getsize(p) for p in fpaths), file_count=len(fpaths))
        src_info = fnames[0] if len(fnames) == 1 else f"{len(fnames)} images: {', '.join(fnames)}"
        with t.stage('captioning'):
            img_desc, error = summarize_images(fpaths)
        if error and not img_desc: return jsonify({'success':False,'error':error})
        raw = img_desc

    elif src in ('pdf','audio','video'):
        if 'file' not in request.files:
            return jsonify({'success':False,'error':'No file uploaded.'})
        f = request.files['file']
        fpath, fname, up_err = _safe_upload_path(f.filename, src)
        if up_err: return jsonify({'success': False, 'error': up_err})
        with t.stage('upload'):
            f.save(fpath)
        t.set_meta(file_size_bytes=os.path.getsize(fpath))
        src_info   = fname
        cache_id   = _hash_file(fpath)
        stage_name = 'transcription' if src in ('audio', 'video') else 'extraction'
        cached_raw = _cache_get(f'extract:{src}:{cache_id}')
        log.info(f'[CACHE] req={t.request_id} extraction_lookup={"HIT" if cached_raw is not None else "MISS"}')
        if cached_raw is not None:
            raw, error, extract_cached = cached_raw, None, True
            t.mark(stage_name, 0.0)
        else:
            with t.stage(stage_name):
                if   src == 'pdf':   raw, error = extract_pdf(fpath)
                elif src == 'audio': raw, error = extract_audio(fpath)
                elif src == 'video': raw, error = extract_video(fpath)
            if raw and not error:
                _cache_set(f'extract:{src}:{cache_id}', raw)
    else:
        return jsonify({'success':False,'error':f'Unknown source: {src}'})

    if error or not raw:
        return jsonify({'success':False,'error':error or 'No text could be extracted.'})

    with t.stage('language_detect'):
        detected_lang = detect_lang(raw) if src != 'image' else (lang if lang != 'auto' else 'en')
    output_lang = detected_lang if lang == 'auto' else lang

    summary_key    = f'summary:{src}:{cache_id}:{DEFAULT_MODEL}:{output_lang}' if cache_id else None
    cached_summary = _cache_get(summary_key) if summary_key else None
    log.info(f'[CACHE] req={t.request_id} summary_lookup={"HIT" if cached_summary is not None else "MISS"}')

    if cached_summary is not None:
        summary, method = cached_summary['summary'], cached_summary['method']
        t.mark('summarization', 0.0)
    else:
        if src != 'image':
            with t.stage('cleaning'):
                raw = _clean_text(raw, source=src)
        text_for_model = raw
        if detected_lang != 'en' and src != 'image':
            with t.stage('translation_to_en'):
                text_for_model = translate_text(raw, detected_lang, 'en')

        try:
            with t.stage('summarization'):
                if src == 'image':
                    enriched = ('## Context & Interpretation' in text_for_model) or \
                               ('## Technical & Scientific Background' in text_for_model)
                    summary_en, method = text_for_model, ('BLIP captioning + Wikipedia' if enriched else 'BLIP image captioning')
                else:
                    summary_en, method = summarize_text(text_for_model)
        except ModelBusyError as e:
            return jsonify({'success': False, 'error': str(e)}), 503
        except Exception as e:
            return jsonify({'success':False,'error':f'Summarization failed: {e}'})

        summary = summary_en
        if output_lang != 'en':
            with t.stage('translation_from_en'):
                summary = translate_structured(summary_en, 'en', output_lang)

        if summary_key:
            _cache_set(summary_key, {'summary': summary, 'method': method})

    result = {
        'success'          : True,
        'summary'          : summary,
        'word_count'       : len(raw.split()),
        'method'           : method,
        'detected_language': detected_lang,
        'output_language'  : output_lang,
    }

    audio_filename = None
    if out in ('audio','both'):
        with t.stage('tts'):
            tts_key    = f'tts:{_hash_bytes(summary.encode("utf-8"))}:{output_lang}'
            cached_tts = _cache_get(tts_key)
            cached_fn  = (cached_tts or {}).get('filename', '')
            if cached_fn and os.path.isfile(os.path.join(UPLOAD_FOLDER, cached_fn)):
                audio_filename       = cached_fn
                result['audio_url']  = f'/uploads/{audio_filename}'
            else:
                apath, terr = text_to_speech(summary, output_lang)
                if terr:
                    result['audio_error'] = terr
                else:
                    audio_filename = os.path.basename(apath)
                    result['audio_url'] = f'/uploads/{audio_filename}'
                    _cache_set(tts_key, {'filename': audio_filename})

    user_id = session.get('user_id')
    if user_id:
        with t.stage('database'):
            sid, save_err = save_summary(user_id, src, src_info, detected_lang,
                               result['word_count'], summary, output_lang, method, audio_filename)
        result['history_id'] = sid
        if save_err: result['history_error'] = save_err

    result['cache']   = {'extraction': extract_cached, 'summary': cached_summary is not None}
    result['request_id'] = t.request_id
    result['timings'] = t.as_dict()
    t.log()
    try:
        generate_benchmark_log()
    except Exception as e:
        log.warning(f'benchmark.log generation failed (non-fatal): {e}')
    if src in ('audio', 'video', 'pdf'):
        import gc; gc.collect()
    return jsonify(result)

@app.route('/api/translate', methods=['POST'])
def translate_route():
    d    = request.get_json() or {}
    text = d.get('text','').strip()
    src  = d.get('source_lang','auto')
    tgt  = d.get('target_lang','en')
    if not text: return jsonify({'success':False,'error':'No text'})
    try:
        out = translate_structured(text, src, tgt)
        return jsonify({'success':True,'translated':out,'language':tgt})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/tts', methods=['POST'])
def tts_route():
    d    = request.get_json() or {}
    text = d.get('text','').strip()
    lang = d.get('lang','en')
    if not text: return jsonify({'success':False,'error':'No text'})
    path, err = text_to_speech(text, lang)
    if err: return jsonify({'success':False,'error':err})
    return jsonify({'success':True,'audio_url':f'/uploads/{os.path.basename(path)}'})

@app.route('/api/history')
@login_required
def history_list():
    try:
        conn = _get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""SELECT id,title,source_type,original_language,original_word_count,
                              summary_language,method,audio_filename,created_at,
                              SUBSTRING(summary_text,1,200) AS preview
                       FROM summaries WHERE user_id=%s ORDER BY created_at DESC LIMIT 100""",
                    (session['user_id'],))
        rows = cur.fetchall(); cur.close(); conn.close()
        for r in rows:
            if r.get('created_at'): r['created_at'] = r['created_at'].strftime('%d %b %Y, %H:%M')
        return jsonify({'success':True,'history':rows})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/history/<int:sid>')
@login_required
def history_get(sid):
    try:
        conn = _get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM summaries WHERE id=%s AND user_id=%s", (sid, session['user_id']))
        row = cur.fetchone(); cur.close(); conn.close()
        if not row: return jsonify({'success':False,'error':'Not found'}), 404
        if row.get('created_at'): row['created_at'] = row['created_at'].strftime('%d %b %Y, %H:%M')
        return jsonify({'success':True,'summary':row})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/history/<int:sid>', methods=['DELETE'])
@login_required
def history_delete(sid):
    try:
        conn = _get_db(); cur = conn.cursor()
        cur.execute("DELETE FROM summaries WHERE id=%s AND user_id=%s", (sid, session['user_id']))
        conn.commit(); cur.close(); conn.close()
        return jsonify({'success':True})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

_PDF_TRANSLITERATE = {
    '\u2014': '--', '\u2013': '-', '\u2018': "'", '\u2019': "'",
    '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u2022': '-',
    '\u00a0': ' ',
}

def _pdf_safe(s):
    s = s or ''
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    for ch, rep in _PDF_TRANSLITERATE.items():
        s = s.replace(ch, rep)
    try:
        s.encode('latin-1'); return s
    except UnicodeEncodeError:
        return s.encode('latin-1', 'replace').decode('latin-1')

def _render_summary_pdf(title, summary_text, meta: dict) -> bytes:
    _try_import('fpdf', 'fpdf2')
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=20); pdf.add_page()

    pdf.set_fill_color(14, 14, 17)
    pdf.rect(0, 0, 210, 26, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_xy(15, 8); pdf.cell(0, 10, 'NOVABRIEF', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(15, 30)

    pdf.set_text_color(15, 17, 32)
    pdf.set_font('Helvetica', 'B', 15)
    pdf.multi_cell(0, 8, _pdf_safe(title or 'Summary'), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(15)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(110, 110, 122)
    meta_bits = [
        str(meta.get('source_type', '')).upper(),
        str(meta.get('created_at', '')),
        f"{int(meta.get('original_word_count') or 0):,} words" if meta.get('original_word_count') else '',
        str(meta.get('summary_language', 'en') or 'en').upper(),
        str(meta.get('method', '')),
    ]
    pdf.cell(0, 6, _pdf_safe('  |  '.join(b for b in meta_bits if b)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.set_draw_color(224, 224, 230); pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    raw = summary_text or ''
    parts = re.split(r'\n##\s+', raw)
    overview = parts[0].strip()

    def heading(txt):
        pdf.set_x(15)
        pdf.set_font('Helvetica', 'B', 11.5); pdf.set_text_color(90, 70, 220)
        pdf.multi_cell(0, 7, _pdf_safe(txt.upper()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def body(txt):
        pdf.set_x(15)
        pdf.set_font('Helvetica', '', 10.5); pdf.set_text_color(25, 25, 30)
        pdf.multi_cell(0, 6, _pdf_safe(txt), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    if overview:
        heading('Overview'); body(overview)

    for part in parts[1:]:
        lines = part.split('\n', 1)
        sec_title = lines[0].strip()
        sec_body  = lines[1].strip() if len(lines) > 1 else ''
        heading(sec_title)
        if _is_bullet_body(sec_body):
            pdf.set_font('Helvetica', '', 10.5); pdf.set_text_color(25, 25, 30)
            for line in sec_body.split('\n'):
                line = line.strip().lstrip('-\u2022*').strip()
                if line:
                    pdf.set_x(15)
                    pdf.multi_cell(0, 6, _pdf_safe(f'\u2022  {line}'), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)
        else:
            body(sec_body)

    pdf.set_auto_page_break(False)
    pdf.set_y(-15); pdf.set_font('Helvetica', 'I', 8); pdf.set_text_color(150, 150, 158)
    pdf.cell(0, 10, _pdf_safe(f'Generated by NovaBrief  \u00b7  {datetime.now().strftime("%d %b %Y, %H:%M")}'), align='C')

    return bytes(pdf.output())

@app.route('/api/history/<int:sid>/pdf')
@login_required
def history_pdf(sid):
    try:
        conn = _get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM summaries WHERE id=%s AND user_id=%s", (sid, session['user_id']))
        row = cur.fetchone(); cur.close(); conn.close()
        if not row: return jsonify({'success':False,'error':'Not found'}), 404
        ts = row.get('created_at','')
        if hasattr(ts,'strftime'): row['created_at'] = ts.strftime('%d %b %Y, %H:%M')
        pdf_bytes = _render_summary_pdf(row.get('title'), row.get('summary_text'), row)
        out_path = os.path.join(UPLOAD_FOLDER, f'summary_{sid}.pdf')
        with open(out_path, 'wb') as f: f.write(pdf_bytes)
        return send_file(out_path, mimetype='application/pdf', as_attachment=True,
                         download_name=f'novabrief_{sid}.pdf')
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 500

@app.route('/api/export/pdf', methods=['POST'])
def export_pdf():
    try:
        d = request.get_json(force=True) or {}
        title = d.get('title') or 'NovaBrief Summary'
        summary = d.get('summary') or ''
        if not summary.strip():
            return jsonify({'success': False, 'error': 'Nothing to export yet.'}), 400
        meta = d.get('meta') or {}
        meta.setdefault('created_at', datetime.now().strftime('%d %b %Y, %H:%M'))
        pdf_bytes = _render_summary_pdf(title, summary, meta)
        fname = f"export_{os.urandom(4).hex()}.pdf"
        out_path = os.path.join(UPLOAD_FOLDER, fname)
        with open(out_path, 'wb') as f: f.write(pdf_bytes)
        return send_file(out_path, mimetype='application/pdf', as_attachment=True,
                         download_name=f"{re.sub(r'[^A-Za-z0-9 _-]','',title)[:50] or 'summary'}.pdf")
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def landing(): return send_from_directory('templates','landing.html')

@app.route('/app')
def app_page(): return send_from_directory('templates','app.html')

@app.route('/history')
def history_page():
    if 'user_id' not in session: return redirect('/?login=1')
    return send_from_directory('templates','history.html')

@app.route('/admin')
def admin_page():
    if 'user_id' not in session: return redirect('/?login=1')
    return send_from_directory('templates','admin.html')

@app.route('/static/<path:f>')
def static_files(f): return send_from_directory('static',f)

@app.route('/uploads/<path:f>')
def uploads(f): return send_from_directory(UPLOAD_FOLDER,f)

@app.route('/api/health')
def health():
    return jsonify({
        'status'         : 'ok',
        'model_ready'    : _txt_ready,
        'model_name'     : _txt_model_name,
        'model_error'    : _txt_error,
        'db_ready'       : _db_ready,
        'img_model_ready': _img_ready,
        'img_model_error': _img_error,
        'img_model_name' : IMAGE_MODEL_ID,
        'whisper_ready'  : _whisper_ready,
        'whisper_model'  : WHISPER_MODEL_SIZE,
        'whisper_device' : _whisper_device_label,
        'whisper_error'  : _whisper_error,
        'cache_enabled'  : ENABLE_CACHE,
        'fallback_active': not _txt_ready,
        'device'         : _device_kind,
        'device_name'    : _device_label,
        'fix_command'    : ('Run fix_torch.bat (Windows) or ./fix_torch.sh (Linux/macOS) — '
                             'it detects a GPU automatically if you have one.') if _txt_error else None,
    })

@app.route('/api/diagnose')
def diagnose():
    import importlib, platform
    checks = {}

    for pkg in ['torch', 'transformers', 'sentencepiece', 'sumy', 'langdetect',
                'deep_translator', 'pypdf', 'gtts', 'PIL', 'bcrypt',
                'mysql.connector', 'fpdf', 'faster_whisper', 'yt_dlp']:
        try:
            mod = importlib.import_module(pkg)
            checks[pkg] = getattr(mod, '__version__', 'installed')
        except ImportError:
            checks[pkg] = 'NOT INSTALLED'

    gpu_info = {'nvidia_smi_detected': _has_nvidia_gpu(), 'cuda_available': False, 'devices': []}
    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        gpu_info['torch_build']    = 'CUDA' if torch.cuda.is_available() else 'CPU-only'
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info['devices'].append({
                    'index': i, 'name': torch.cuda.get_device_name(i),
                    'vram_mb': torch.cuda.get_device_properties(i).total_memory // (1024*1024),
                })
    except ImportError:
        pass

    import shutil
    ffmpeg_info = {
        'ready'  : _ffmpeg_ready,
        'source' : _ffmpeg_source,
        'path'   : _ffmpeg_path,
        'error'  : _ffmpeg_error,
    }
    try:
        import imageio_ffmpeg
        ffmpeg_info['imageio_ffmpeg_installed'] = True
        ffmpeg_info['imageio_ffmpeg_version']    = getattr(imageio_ffmpeg, '__version__', '?')
        try:
            candidate = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_info['imageio_ffmpeg_exe_path']   = candidate
            ffmpeg_info['imageio_ffmpeg_exe_exists'] = os.path.isfile(candidate)
            ffmpeg_info['imageio_ffmpeg_exe_runs']   = _ffmpeg_binary_works(candidate) if os.path.isfile(candidate) else False
        except Exception as e:
            ffmpeg_info['imageio_ffmpeg_exe_error'] = str(e)
    except ImportError:
        ffmpeg_info['imageio_ffmpeg_installed'] = False
        ffmpeg_info['note'] = (f'imageio-ffmpeg not importable under {sys.executable}. '
                                f'If you installed it elsewhere (e.g. via PyCharm), that\'s the mismatch.')
    ffmpeg_info['system_ffmpeg_on_path'] = shutil.which('ffmpeg')
    ffmpeg_info['prober_on_path'] = shutil.which('ffprobe') or shutil.which('avprobe')
    ffmpeg_info['prober_patched'] = _pydub_mediainfo_json_original is not None
    decoy = _detect_decoy_ffmpeg_package()
    if decoy:
        ffmpeg_info['decoy_pypi_ffmpeg_installed'] = decoy
        ffmpeg_info['decoy_warning'] = (
            f'The "ffmpeg" PyPI package (v{decoy}) is installed — this is a commonly-confused '
            f'decoy that does NOT provide a real ffmpeg binary. Install imageio-ffmpeg instead.'
        )

    return jsonify({
        'python'            : sys.version,
        'python_executable' : sys.executable,
        'platform' : platform.platform(),
        'packages' : checks,
        'gpu'      : gpu_info,
        'ffmpeg'   : ffmpeg_info,
        'active_device'   : {'kind': _device_kind, 'label': _device_label, 'fp16': _use_fp16},
        'force_cpu_config': bool(cfg and getattr(cfg, 'FORCE_CPU', False)),
        'resource_usage'  : _resource_usage_str(),
        'gpu_nvidia_smi'  : _nvidia_smi_stats(),
        'model_ready' : _txt_ready,
        'model_name'  : _txt_model_name,
        'model_error' : _txt_error,
        'image_model' : {'ready': _img_ready, 'model': IMAGE_MODEL_ID,
                          'device': _img_device_label, 'error': _img_error},
        'whisper'     : {'ready': _whisper_ready, 'model': WHISPER_MODEL_SIZE,
                          'device': _whisper_device_label, 'error': _whisper_error},
        'cache'       : {'enabled': ENABLE_CACHE, 'path': CACHE_DB_PATH,
                          'expiry_hours': CACHE_EXPIRY_HOURS or 'never',
                          'size_bytes': os.path.getsize(CACHE_DB_PATH) if os.path.isfile(CACHE_DB_PATH) else 0},
        'db_ready'    : _db_ready,
        'db_pool_active': bool(_db_pool),
        'upload_dir'  : {'path': os.path.abspath(UPLOAD_FOLDER), 'writable': os.access(UPLOAD_FOLDER, os.W_OK)},
        'easyocr_note': 'EasyOCR is listed in some project notes as part of the stack but is not present in '
                         'this build — no OCR text-extraction step runs on images. See the Production '
                         'Readiness Report if you want this added.',
        'fix_script_windows': 'Run fix_torch.bat in your project folder',
        'fix_script_linux'  : 'Run chmod +x fix_torch.sh && ./fix_torch.sh',
    })

@app.route('/api/db-status')
def db_status():
    out = {
        'connected': _db_ready,
        'host'     : DB_CFG['host'],
        'port'     : DB_CFG['port'],
        'database' : DB_CFG['database'],
        'user'     : DB_CFG['user'],
    }
    if _db_ready:
        try:
            conn = _get_db(); cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM users'); out['users_count'] = cur.fetchone()[0]
            cur.execute('SELECT COUNT(*) FROM summaries'); out['summaries_count'] = cur.fetchone()[0]
            cur.close(); conn.close()
            out['message'] = (f"Connected — {out['users_count']} user(s), {out['summaries_count']} summarie(s) "
                               f"currently in `{DB_CFG['database']}` at {DB_CFG['host']}:{DB_CFG['port']}. "
                               f"If Workbench shows different numbers, it's pointed at a different server/schema.")
        except Exception as e:
            out['message'] = f'Connected, but the status query itself failed: {e}'
    else:
        out['message'] = ('Not connected — login/history disabled, but summarization still works. '
                           'Edit config.py with your MySQL credentials, then run: python setup_db.py')
    return jsonify(out)

@app.route('/api/reload-model', methods=['POST'])
def reload_model():
    global _txt_error
    if _txt_ready:
        return jsonify({'success': True, 'message': 'Model already loaded', 'model': _txt_model_name, 'device': _device_label})
    _txt_error = None
    t = threading.Thread(target=_load_txt_model, daemon=True)
    t.start()
    t.join(timeout=180)
    if _txt_ready:
        return jsonify({'success': True, 'message': 'Model loaded!', 'model': _txt_model_name, 'device': _device_label})
    return jsonify({
        'success': False,
        'error'  : _txt_error or 'Model still loading — check server logs.',
        'fix'    : 'Run fix_torch.bat (Windows) or ./fix_torch.sh (Linux/macOS) — detects a GPU automatically if present.',
    })

@app.route('/api/reload-ffmpeg', methods=['POST'])
def reload_ffmpeg():
    ok = _ensure_ffmpeg(force_recheck=True)
    return jsonify({
        'success': ok,
        'source' : _ffmpeg_source,
        'path'   : _ffmpeg_path,
        'error'  : None if ok else _ffmpeg_error,
        'python' : sys.executable,
    })

@app.route('/api/reload-whisper', methods=['POST'])
def reload_whisper():
    global _whisper_error
    if _whisper_ready:
        return jsonify({'success': True, 'message': 'Whisper already loaded',
                         'model': WHISPER_MODEL_SIZE, 'device': _whisper_device_label})
    _whisper_error = None
    th = threading.Thread(target=_load_whisper_model, daemon=True)
    th.start()
    th.join(timeout=180)
    if _whisper_ready:
        return jsonify({'success': True, 'message': 'Whisper loaded!',
                         'model': WHISPER_MODEL_SIZE, 'device': _whisper_device_label})
    return jsonify({'success': False, 'error': _whisper_error or 'Whisper still loading — check server logs.'})

@app.route('/api/cache/clear', methods=['POST'])
def cache_clear():
    try:
        with _cache_lock:
            conn = sqlite3.connect(CACHE_DB_PATH, timeout=10)
            n = conn.execute('SELECT COUNT(*) FROM cache').fetchone()[0]
            conn.execute('DELETE FROM cache')
            conn.commit(); conn.close()
        log.info(f'Cache cleared ({n} entries removed)')
        return jsonify({'success': True, 'entries_removed': n})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def _wait_for_startup_models():
    print('\n  Waiting for AI models to finish loading before accepting any requests...')
    print('  (first run downloads them — can take a few minutes; instant after that)\n')
    stages = [
        ('DistilBART (text)',             _txt_load_thread,     lambda: _txt_ready,     lambda: _txt_error),
        ('BLIP (image)',                  _img_load_thread,     lambda: _img_ready,     lambda: _img_error),
        ('Faster-Whisper (audio/video)',  _whisper_load_thread, lambda: _whisper_ready, lambda: _whisper_error),
    ]
    for name, th, is_ready, get_err in stages:
        th.join()
        if is_ready():
            print(f'  ✅ {name:<32} ready')
        else:
            err = (get_err() or 'see log above')
            print(f'  ⚠  {name:<32} NOT ready — {str(err)[:110]}')

    print('\n  Verifying models...')
    ready = [is_ready() for _, _, is_ready, _ in stages]
    if all(ready):
        print('  ✅ All models verified ready.')
    else:
        missing = sum(1 for r in ready if not r)
        print(f'  ⚠  {missing} of {len(stages)} model(s) did not load — those features will return a clear '
              f'error/fallback instead of hanging, and can be retried via /api/reload-model, '
              f'/api/reload-whisper, or the "Retry" button in the UI, without restarting the app.')
    print('\n  Application Ready.\n')

def get_model_manager_status() -> dict:
    return {
        'text':    {'ready': _txt_ready,     'name': _txt_model_name,  'device': _device_label if _txt_ready else None,  'error': _txt_error},
        'image':   {'ready': _img_ready,     'name': IMAGE_MODEL_ID,   'device': _img_device_label if _img_ready else None, 'error': _img_error},
        'whisper': {'ready': _whisper_ready, 'name': WHISPER_MODEL_SIZE, 'device': _whisper_device_label if _whisper_ready else None, 'error': _whisper_error},
    }

def get_resource_status() -> dict:
    torch_vram = None
    if _device_kind == 'cuda':
        try:
            import torch
            idx = _device.index if _device is not None else 0
            torch_vram = {'allocated_mb': round(torch.cuda.memory_allocated(idx) / 1e6),
                          'reserved_mb': round(torch.cuda.memory_reserved(idx) / 1e6),
                          'peak_allocated_mb': round(torch.cuda.max_memory_allocated(idx) / 1e6)}
        except Exception:
            pass
    return {
        'device'      : {'kind': _device_kind, 'label': _device_label, 'fp16': _use_fp16},
        'usage'       : _resource_usage_str(),
        'gpu_nvidia_smi': _nvidia_smi_stats(),
        'gpu_torch_only': torch_vram,
        'whisper_batched_pipeline': bool(_whisper_batched),
        'ffmpeg_ready': _ffmpeg_ready,
        'db_ready'    : _db_ready,
        'db_pool_active': bool(_db_pool),
        'cache'       : {'enabled': ENABLE_CACHE, 'path': CACHE_DB_PATH, 'expiry_hours': CACHE_EXPIRY_HOURS or 'never'},
        'upload_dir'  : {'path': os.path.abspath(UPLOAD_FOLDER), 'writable': os.access(UPLOAD_FOLDER, os.W_OK)},
    }

_PERF_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'performance.log')
_BENCHMARK_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark.log')
_last_benchmark_gen = 0.0
_benchmark_gen_lock = threading.Lock()

def generate_benchmark_log(min_interval_seconds: float = 60.0) -> Optional[str]:
    global _last_benchmark_gen
    now = time.time()
    if now - _last_benchmark_gen < min_interval_seconds:
        return None
    with _benchmark_gen_lock:
        if now - _last_benchmark_gen < min_interval_seconds:
            return None
        _last_benchmark_gen = now
        if not os.path.isfile(_PERF_LOG_PATH):
            return None
        by_type = {}
        try:
            with open(_PERF_LOG_PATH, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    itype = rec.get('input_type', 'unknown')
                    by_type.setdefault(itype, []).append(rec)
        except Exception as e:
            log.warning(f'generate_benchmark_log: could not read {_PERF_LOG_PATH}: {e}')
            return None
        if not by_type:
            return None

        lines = [f'NovaBrief Benchmark Report — generated {datetime.now().isoformat(timespec="seconds")}',
                 f'Source: {_PERF_LOG_PATH} ({sum(len(v) for v in by_type.values())} requests recorded)',
                 '(Real observed values only — nothing here is estimated or fabricated.)', '']
        for itype in sorted(by_type):
            recs = by_type[itype]
            totals = [r.get('total_ms', 0) for r in recs]
            cpu    = [r['cpu_percent'] for r in recs if r.get('cpu_percent') is not None]
            ram    = [r['ram_mb'] for r in recs if r.get('ram_mb') is not None]
            vram   = [r['gpu']['vram_used_mb'] for r in recs if r.get('gpu', {}).get('vram_used_mb') is not None]
            gutil  = [r['gpu']['gpu_util_pct'] for r in recs if r.get('gpu', {}).get('gpu_util_pct') is not None]
            lines.append(f'[{itype.upper()}]  n={len(recs)}')
            lines.append(f'  Total time (ms)  avg={sum(totals)/len(totals):.0f}  fastest={min(totals):.0f}  slowest={max(totals):.0f}')
            if cpu:   lines.append(f'  CPU %            avg={sum(cpu)/len(cpu):.1f}')
            if ram:   lines.append(f'  RAM (MB)         avg={sum(ram)/len(ram):.0f}')
            if vram:  lines.append(f'  VRAM used (MB)   avg={sum(vram)/len(vram):.0f}')
            if gutil: lines.append(f'  GPU util %       avg={sum(gutil)/len(gutil):.1f}')
            lines.append('')
        try:
            with open(_BENCHMARK_LOG_PATH, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return _BENCHMARK_LOG_PATH
        except Exception as e:
            log.warning(f'generate_benchmark_log: could not write {_BENCHMARK_LOG_PATH}: {e}')
            return None

def _startup_health_check():
    print('\n  Startup Health Check')
    print('  ────────────────────')
    rows = []

    try:
        import torch as _t
        rows.append(('PyTorch', True, _t.__version__))
    except Exception as e:
        rows.append(('PyTorch', False, str(e)[:70]))

    rows.append(('CUDA / GPU', _device_kind == 'cuda',
                 _device_label if _device_kind == 'cuda' else 'using CPU (see log above for why, if a GPU is present)'))
    rows.append(('FFmpeg', _ffmpeg_ready, _ffmpeg_source if _ffmpeg_ready else str(_ffmpeg_error)[:70]))
    rows.append(('DistilBART (text)', _txt_ready, _txt_model_name or str(_txt_error)[:70]))
    rows.append(('BLIP (image)', _img_ready, IMAGE_MODEL_ID if _img_ready else str(_img_error)[:70]))
    rows.append(('Faster-Whisper (audio/video)', _whisper_ready,
                 f'"{WHISPER_MODEL_SIZE}"' if _whisper_ready else str(_whisper_error)[:70]))
    rows.append(('MySQL', _db_ready,
                 'connected' if _db_ready else 'still connecting in background — see GET /api/db-status'))
    try:
        cache_ok = ENABLE_CACHE and os.path.isfile(CACHE_DB_PATH)
        rows.append(('SQLite result cache', cache_ok, CACHE_DB_PATH if cache_ok else 'disabled or not yet created'))
    except Exception as e:
        rows.append(('SQLite result cache', False, str(e)[:70]))
    try:
        probe = os.path.join(UPLOAD_FOLDER, f'.write_test_{os.urandom(4).hex()}')
        with open(probe, 'w') as f: f.write('ok')
        os.remove(probe)
        rows.append(('Upload directory', True, os.path.abspath(UPLOAD_FOLDER)))
    except Exception as e:
        rows.append(('Upload directory', False, f'{UPLOAD_FOLDER} is not writable: {e}'))

    for name, ok, detail in rows:
        print(f'  {"✔" if ok else "✘"} {name:<30} {detail}')
    failed = [name for name, ok, _ in rows if not ok]
    if failed:
        print(f'\n  ⚠  {len(failed)} check(s) did not pass: {", ".join(failed)}. The app still starts — see '
              f'GET /api/diagnose for full detail on any of these, or the log lines just above.')
    else:
        print('\n  All checks passed.')
    print()

if __name__ == '__main__':
    _host  = (cfg.HOST  if cfg else None) or '0.0.0.0'
    _port  = (cfg.PORT  if cfg else None) or 5000
    _debug = (cfg.DEBUG if cfg else False)

    print('\n╔══════════════════════════════════════════════╗')
    print('║   NovaBrief AI Summarizer — Full Edition     ║')
    print(f'║   http://localhost:{_port}{" " * (24 - len(str(_port)))}║')
    print('╚══════════════════════════════════════════════╝\n')
    print(f'  Config file : {"config.py (loaded)" if cfg else "NOT FOUND — using defaults / env vars"}')
    print(f'  Text model  : {DEFAULT_MODEL}')
    print(f'  Image model : {IMAGE_MODEL_ID}')
    print(f'  Whisper     : "{WHISPER_MODEL_SIZE}" (loading now — the server waits for this before it starts)')
    print(f'  Cache       : {"enabled — " + CACHE_DB_PATH if ENABLE_CACHE else "disabled (ENABLE_CACHE=False in config.py)"}')
    _gpu_hint = _has_nvidia_gpu()
    print(f'  Compute     : {"GPU detected (nvidia-smi) — AI models will use it" if _gpu_hint else "CPU only (no NVIDIA GPU detected)"}'
          f'{"  [FORCE_CPU set in config.py]" if (cfg and getattr(cfg, "FORCE_CPU", False)) else ""}')
    print('                web app + PDF/YouTube/audio extraction always run on CPU regardless')
    print(f'  MySQL       : {DB_CFG["host"]}:{DB_CFG["port"]} / db={DB_CFG["database"]} / user={DB_CFG["user"]}')
    _ffmpeg_ok = _ensure_ffmpeg()
    print(f'  ffmpeg      : {"ready (" + _ffmpeg_source + ")  " + _ffmpeg_path if _ffmpeg_ok else "NOT READY — audio/video will fail. See: " + str(_ffmpeg_error)[:90]}')
    print(f'  Running as  : {sys.executable}')
    if not cfg:
        print('\n  ⚠  config.py not found — create one or set env vars:')
        print('     MYSQL_HOST  MYSQL_PORT  MYSQL_USER  MYSQL_PASSWORD  MYSQL_DATABASE')
        print('     NOVABRIEF_MODEL  SECRET_KEY')
    print('\n  Run "python setup_db.py" first if MySQL connection fails.\n')

    _wait_for_startup_models()
    _startup_health_check()

    app.run(debug=_debug, host=_host, port=_port, threaded=True)
