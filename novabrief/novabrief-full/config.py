import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _env(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.strip().lower() in ('1', 'true', 'yes', 'on')
    if isinstance(default, int):
        try: return int(val)
        except ValueError: return default
    return val

def _env_list(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return [item.strip() for item in val.split(',') if item.strip()]

DB_HOST     = _env('MYSQL_HOST', 'localhost')
DB_PORT     = _env('MYSQL_PORT', 3306)
DB_USER     = _env('MYSQL_USER', 'root')
DB_PASSWORD = _env('MYSQL_PASSWORD', 'change-me')
DB_NAME     = _env('MYSQL_DATABASE', 'NovaAiSummary')

SECRET_KEY  = _env('SECRET_KEY', 'novabrief-secret-change-me-in-production')

ADMIN_EMAILS = _env_list('NOVABRIEF_ADMIN_EMAILS', [
])

AI_MODEL    = _env('NOVABRIEF_MODEL', 'sshleifer/distilbart-cnn-12-6')

IMAGE_MODEL = _env('NOVABRIEF_IMAGE_MODEL', 'Salesforce/blip-image-captioning-large')

WHISPER_MODEL_SIZE = _env('NOVABRIEF_WHISPER_MODEL', 'small')

FORCE_CPU = _env('NOVABRIEF_FORCE_CPU', False)

MAX_TRANSCRIBE_MINUTES = _env('NOVABRIEF_MAX_TRANSCRIBE_MINUTES', 120)

MAX_YOUTUBE_FALLBACK_MINUTES = _env('NOVABRIEF_MAX_YOUTUBE_MINUTES', 60)

ENABLE_CACHE = _env('NOVABRIEF_ENABLE_CACHE', True)

CACHE_EXPIRY_HOURS = _env('NOVABRIEF_CACHE_EXPIRY_HOURS', 0)

MODEL_LOCK_WAIT_SECONDS = _env('NOVABRIEF_MODEL_LOCK_WAIT_SECONDS', 180)

INFERENCE_WATCHDOG_SECONDS = _env('NOVABRIEF_INFERENCE_WATCHDOG_SECONDS', 900)

NETWORK_WATCHDOG_SECONDS = _env('NOVABRIEF_NETWORK_WATCHDOG_SECONDS', 120)

CONFIDENCE_LOW_THRESHOLD = _env('NOVABRIEF_CONFIDENCE_LOW_THRESHOLD', 0.4)
CONFIDENCE_VERY_LOW_THRESHOLD = _env('NOVABRIEF_CONFIDENCE_VERY_LOW_THRESHOLD', 0.15)

CORS_ORIGINS = _env_list('NOVABRIEF_CORS_ORIGINS', [
    'http://localhost:5000',
    'http://127.0.0.1:5000',
])

SESSION_COOKIE_SECURE = _env('NOVABRIEF_SESSION_COOKIE_SECURE', False)

HOST        = _env('NOVABRIEF_HOST', '0.0.0.0')
PORT        = _env('NOVABRIEF_PORT', 5000)
DEBUG       = _env('NOVABRIEF_DEBUG', False)
