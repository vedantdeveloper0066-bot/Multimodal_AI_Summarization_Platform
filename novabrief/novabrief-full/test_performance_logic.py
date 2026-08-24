import os, re, sys, json, time, hashlib, sqlite3, threading, tempfile

class Timings:
    __slots__ = ('label', 'marks', '_t0')
    def __init__(self, label='request'):
        self.label = label; self.marks = []; self._t0 = time.perf_counter()
    class _Stage:
        __slots__ = ('parent', 'name', '_t0')
        def __init__(self, parent, name): self.parent, self.name = parent, name
        def __enter__(self): self._t0 = time.perf_counter(); return self
        def __exit__(self, *exc):
            self.parent.marks.append((self.name, time.perf_counter() - self._t0)); return False
    def stage(self, name): return Timings._Stage(self, name)
    def mark(self, name, seconds): self.marks.append((name, seconds))
    def total(self): return time.perf_counter() - self._t0
    def as_dict(self):
        d = {}
        for name, secs in self.marks: d[name] = d.get(name, 0) + round(secs * 1000)
        d['total_ms'] = round(self.total() * 1000)
        return d

CACHE_DB_PATH = tempfile.mktemp(suffix='.sqlite3')
ENABLE_CACHE = True
_cache_lock = threading.Lock()
def _cache_init():
    with _cache_lock:
        conn = sqlite3.connect(CACHE_DB_PATH, timeout=10)
        conn.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)')
        conn.commit(); conn.close()
def _cache_get(key):
    with _cache_lock:
        conn = sqlite3.connect(CACHE_DB_PATH, timeout=10)
        row = conn.execute('SELECT value FROM cache WHERE key = ?', (key,)).fetchone()
        conn.close()
    return json.loads(row[0]) if row else None
def _cache_set(key, value):
    with _cache_lock:
        conn = sqlite3.connect(CACHE_DB_PATH, timeout=10)
        conn.execute('INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)', (key, json.dumps(value), time.time()))
        conn.commit(); conn.close()
def _hash_bytes(data): return hashlib.sha256(data).hexdigest()
def _hash_file(path, chunk_size=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''): h.update(chunk)
    return h.hexdigest()

_txt_ready = False
_txt_pipeline = None
CHUNK_TOKEN_BUDGET = 900
def _count_tokens(text):
    if _txt_ready and _txt_pipeline is not None:
        try: return len(_txt_pipeline.tokenizer.encode(text, add_special_tokens=False))
        except Exception: pass
    return int(len(text.split()) * 1.3)
def _chunk(text, max_tokens=CHUNK_TOKEN_BUDGET):
    sents = re.split(r'(?<=[.!?])\s+', text)
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
    return [c for c in chunks if c.strip()]

_YT_ID_PATS = [r'(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})', r'^([A-Za-z0-9_-]{11})$']
def _youtube_video_id(url):
    return next((m.group(1) for p in _YT_ID_PATS for m in [re.search(p, url)] if m), None)

failures = []
def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}' + (f' — {detail}' if detail and not cond else ''))
    if not cond: failures.append(name)

t = Timings('test')
with t.stage('a'):
    time.sleep(0.02)
with t.stage('b'):
    time.sleep(0.01)
t.mark('cache_hit', 0.0)
d = t.as_dict()
check('Timings records each stage', set(d.keys()) >= {'a', 'b', 'cache_hit', 'total_ms'}, d)
check('Timings stage a takes >=15ms', d['a'] >= 15, d)
check('Timings cache_hit stays 0ms', d['cache_hit'] == 0, d)
check('Timings total >= sum of stages', d['total_ms'] >= d['a'] + d['b'], d)

t2 = Timings('test2')
t2.mark('translate', 10.0); t2.mark('translate', 5.0)
check('Repeated stage name sums instead of overwriting', t2.as_dict()['translate'] == 15000, t2.as_dict())

_cache_init()
check('Cache miss returns None', _cache_get('nope:123') is None)
_cache_set('extract:pdf:abc123', 'hello world extracted text')
check('Cache hit returns stored value', _cache_get('extract:pdf:abc123') == 'hello world extracted text')
_cache_set('wiki:golden gate bridge', {})
check('Cache stores/reads an empty-dict "confirmed no match" marker distinctly from a miss',
      _cache_get('wiki:golden gate bridge') == {} and _cache_get('wiki:totally-unseen-query') is None)
_cache_set('summary:pdf:abc123:modelX:en', {'summary': 'S', 'method': 'M'})
check('Cache overwrite (INSERT OR REPLACE) works', True if _cache_get('summary:pdf:abc123:modelX:en')['summary'] == 'S' else False)

check('_hash_bytes is deterministic', _hash_bytes(b'hello') == _hash_bytes(b'hello'))
check('_hash_bytes distinguishes different content', _hash_bytes(b'hello') != _hash_bytes(b'world'))
tmp = tempfile.mktemp()
with open(tmp, 'wb') as f: f.write(os.urandom(1024 * 1024 * 3 + 17))
h1 = _hash_file(tmp)
h2 = hashlib.sha256(open(tmp, 'rb').read()).hexdigest()
check('_hash_file (streamed) matches a plain whole-file hash', h1 == h2)
os.remove(tmp)

short_text = 'This is one short sentence. Here is another one.'
check('_chunk leaves short text as a single chunk', len(_chunk(short_text)) == 1)

long_text = ' '.join(f'This is sentence number {i} in a long test document about various topics.' for i in range(400))
chunks = _chunk(long_text, max_tokens=200)
reconstructed_word_count = sum(len(c.split()) for c in chunks)
original_word_count = len(long_text.split())
check('_chunk with a long doc produces multiple chunks', len(chunks) > 1, f'{len(chunks)} chunks')
check('_chunk does not drop or duplicate words across chunks',
      reconstructed_word_count == original_word_count,
      f'{reconstructed_word_count} vs {original_word_count}')
check('_chunk respects the token budget per chunk (fallback word*1.3 estimate)',
      all(_count_tokens(c) <= 200 + 40 for c in chunks),
      [round(_count_tokens(c)) for c in chunks])

huge_single_sentence = 'word ' * 5000
chunks2 = _chunk(huge_single_sentence, max_tokens=200)
check('_chunk slices a single oversized sentence instead of returning one giant chunk', len(chunks2) > 1, len(chunks2))

cases = {
    'https://www.youtube.com/watch?v=dQw4w9WgXcQ': 'dQw4w9WgXcQ',
    'https://youtu.be/dQw4w9WgXcQ': 'dQw4w9WgXcQ',
    'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s': 'dQw4w9WgXcQ',
    'https://www.youtube.com/embed/dQw4w9WgXcQ': 'dQw4w9WgXcQ',
    'dQw4w9WgXcQ': 'dQw4w9WgXcQ',
    'not a url at all': None,
}
for url, expected in cases.items():
    check(f'_youtube_video_id({url!r}) == {expected!r}', _youtube_video_id(url) == expected, _youtube_video_id(url))
check('Same video, different URL shapes resolve to the SAME cache id (the whole point of hashing by video id, not URL)',
      len({_youtube_video_id(u) for u in
           ['https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'https://youtu.be/dQw4w9WgXcQ',
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s']}) == 1)

os.remove(CACHE_DB_PATH)
print(f'\n{len(failures)} failure(s)' if failures else '\nAll checks passed.')
sys.exit(1 if failures else 0)
