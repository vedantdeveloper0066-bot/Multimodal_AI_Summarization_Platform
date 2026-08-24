const API = 'http://localhost:5000';
const $ = id => document.getElementById(id);
const $$ = s => document.querySelectorAll(s);

let activeSource = 'youtube';
let selectedFiles = {};
let selectedImages = [];
let currentSummary = '';
let currentLang = 'en';
let currentUser = null;
let historyId = null;

const MAX_IMAGES_GUEST = 3;
const MAX_IMAGES_USER  = 7;
const maxImages = () => (currentUser ? MAX_IMAGES_USER : MAX_IMAGES_GUEST);

document.addEventListener('DOMContentLoaded', () => {
  checkAuth();
  checkHealth();
  initSidebar();
  initSourceNav();
  initDropZones();
  initFileInputs();
  initTextCounter();
  initOutputOptions();
  initSummarizeBtn();
  initResultActions();
  initAuthModal();
});

async function checkHealth() {
  const dot = () => $('modelStatus')?.querySelector('.status-dot');
  const txt = $('modelStatusText');
  try {
    const d = await apiFetch('/api/health');
    const deviceTag = d.device === 'cuda' ? `⚡ ${d.device_name}` : 'CPU';
    if (d.model_ready) {
      if (dot()) dot().className = 'status-dot ready';
      if (txt) txt.textContent = `AI Ready · ${d.model_name || ''} · ${deviceTag}`;
      if ($('modelStatus')) $('modelStatus').title = `Web app + preprocessing: CPU  ·  AI models: ${d.device === 'cuda' ? d.device_name : 'CPU'}`;
    } else if (d.model_error) {
      if (dot()) dot().className = 'status-dot error';
      if (txt) txt.textContent = 'Extractive mode';
      showModelBanner(d.model_error, d.fix_command);
    } else {
      if (dot()) dot().className = 'status-dot loading';
      if (txt) txt.textContent = 'Loading AI model…';
      setTimeout(checkHealth, 3000);
    }
  } catch {
    if (dot()) dot().className = 'status-dot error';
    if (txt) txt.textContent = 'Backend unreachable';
    showError('Backend not connected. Start app.py first:\n  python app.py');
  }
}

function showModelBanner(errorMsg, fixCmd) {
  let banner = $('modelErrorBanner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'modelErrorBanner';
    banner.className = 'model-error-banner';
    document.querySelector('.app-main')?.prepend(banner);
  }
  banner.style.display = 'flex';
  banner.innerHTML = `
    <div class="meb-icon">⚠️</div>
    <div class="meb-body">
      <strong>AI model failed to load — summaries still work using extractive mode.</strong>
      <span class="meb-detail">${errorMsg || 'Unknown error — see terminal for details'}</span>
      <span class="meb-detail" style="margin-top:4px">
        <b>Windows:</b> double-click <code style="background:rgba(0,0,0,.3);padding:1px 5px;border-radius:4px">fix_torch.bat</code> &nbsp;|&nbsp;
        <b>Linux/Mac:</b> run <code style="background:rgba(0,0,0,.3);padding:1px 5px;border-radius:4px">./fix_torch.sh</code>
      </span>
      ${fixCmd ? `<code class="meb-fix">$ ${fixCmd}</code>` : ''}
      <a href="/api/diagnose" target="_blank" style="font-size:.75rem;color:var(--cyan);margin-top:4px">
        View full diagnostic report →
      </a>
    </div>
    <div class="meb-actions">
      <button class="btn-ghost btn-sm" id="retryModelBtn">🔄 Retry</button>
      <button class="btn-ghost btn-sm" onclick="hideModelBanner()">✕</button>
    </div>`;
  $('retryModelBtn')?.addEventListener('click', retryModel);
}

function hideModelBanner() {
  const b = $('modelErrorBanner');
  if (b) b.style.display = 'none';
}

async function retryModel() {
  const btn = $('retryModelBtn');
  if (btn) { btn.textContent = 'Retrying…'; btn.disabled = true; }
  try {
    const d = await apiFetch('/api/reload-model', 'POST');
    if (d.success) {
      hideModelBanner();
      checkHealth();
    } else {
      if (btn) { btn.textContent = '🔄 Retry'; btn.disabled = false; }
      const banner = $('modelErrorBanner');
      const detail = banner?.querySelector('.meb-detail');
      if (detail) detail.textContent = d.error || 'Retry failed — check terminal logs.';
    }
  } catch {
    if (btn) { btn.textContent = '🔄 Retry'; btn.disabled = false; }
  }
}

async function checkAuth() {
  try {
    const d = await apiFetch('/api/auth/me');
    if (d.logged_in) {
      currentUser = d.user;
      updateUserUI(d.user);
    }
  } catch {}
}

function updateUserUI(user) {
  $('userArea').style.display = 'flex';
  $('authArea').style.display = 'none';
  $('userName').textContent = user.name;
}

function initAuthModal() {
  const overlay = $('authOverlay'), cl = $('authClose');
  const fi = $('formSignIn'), fu = $('formSignUp');
  const ti = $('tabSignIn'), tu = $('tabSignUp');

  function tab(t) {
    const s = t === 'si';
    fi.style.display = s ? 'block' : 'none';
    fu.style.display = s ? 'none' : 'block';
    ti.classList.toggle('active', s); tu.classList.toggle('active', !s);
  }
  function open(t) {
    tab(t || 'si'); overlay.style.display = 'flex'; document.body.style.overflow = 'hidden';
    checkDbStatusForModal();
  }

  async function checkDbStatusForModal() {
    const existing = document.getElementById('dbStatusBanner');
    if (existing) existing.remove();
    try {
      const d = await apiFetch('/api/db-status');
      if (!d.connected) {
        const banner = document.createElement('div');
        banner.id = 'dbStatusBanner';
        banner.style.cssText = 'background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3);border-radius:10px;padding:10px 14px;margin-bottom:16px;font-size:.78rem;color:var(--muted-hi);line-height:1.5;';
        banner.innerHTML = '⚠️ <strong style="color:var(--snow)">Database not connected.</strong> Sign in / Register won\'t work yet.<br>Run <code style="background:rgba(0,0,0,.3);padding:1px 5px;border-radius:4px;color:var(--cyan)">python setup_db.py</code> after editing <code style="background:rgba(0,0,0,.3);padding:1px 5px;border-radius:4px;color:var(--cyan)">config.py</code>.';
        overlay.querySelector('.modal-card')?.insertBefore(banner, overlay.querySelector('.auth-tabs-row'));
      }
    } catch {}
  }
  function close() { overlay.style.display = 'none'; document.body.style.overflow = ''; }

  $('appSignIn')?.addEventListener('click', () => open('si'));
  $('appSignUp')?.addEventListener('click', () => open('su'));
  cl?.addEventListener('click', close);
  overlay?.addEventListener('click', e => { if (e.target === overlay) close(); });
  ti?.addEventListener('click', () => tab('si'));
  tu?.addEventListener('click', () => tab('su'));
  $('goRegister')?.addEventListener('click', e => { e.preventDefault(); tab('su'); });
  $('goSignIn')?.addEventListener('click', e => { e.preventDefault(); tab('si'); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });

  $('doSignIn')?.addEventListener('click', async () => {
    const email = $('siEmail').value, pass = $('siPass').value, msg = $('siMsg');
    msg.textContent = 'Signing in…'; msg.style.color = '';
    const d = await apiFetch('/api/auth/login', 'POST', { email, password: pass });
    if (d.success) {
      currentUser = d.user; updateUserUI(d.user);
      msg.style.color = 'var(--cyan)'; msg.textContent = '✓ Signed in!';
      setTimeout(close, 900);
    } else { msg.style.color = '#EF4444'; msg.textContent = d.error; }
  });

  $('doSignUp')?.addEventListener('click', async () => {
    const name = $('suName').value, email = $('suEmail').value, pass = $('suPass').value, msg = $('suMsg');
    msg.textContent = 'Creating account…'; msg.style.color = '';
    const d = await apiFetch('/api/auth/register', 'POST', { name, email, password: pass });
    if (d.success) {
      currentUser = d.user; updateUserUI(d.user);
      msg.style.color = 'var(--cyan)'; msg.textContent = '✓ Account created!';
      setTimeout(close, 900);
    } else { msg.style.color = '#EF4444'; msg.textContent = d.error; }
  });

  $('logoutBtn')?.addEventListener('click', async () => {
    await apiFetch('/api/auth/logout', 'POST');
    currentUser = null;
    $('userArea').style.display = 'none';
    $('authArea').style.display = 'flex';
  });
}

function initSidebar() {
  const toggle  = $('sidebarToggle');
  const sidebar = $('sidebar');
  const overlay = $('sidebarOverlay');
  const layout  = document.querySelector('.app-layout');

  function openMobile()  {
    sidebar.classList.add('open');
    overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';
  }
  function closeMobile() {
    sidebar.classList.remove('open');
    overlay.classList.remove('visible');
    document.body.style.overflow = '';
  }

  // ── Desktop: collapse/expand sidebar width ────────────────────────
  function toggleDesktop() {
    const collapsed = sidebar.classList.toggle('desktop-collapsed');
    toggle.classList.toggle('sidebar-btn-active', collapsed);

    toggle.style.transform = collapsed ? 'rotate(90deg)' : '';
  }

  // ── Unified toggle ────────────────────────────────────────────────
  function handleToggle() {
    if (window.innerWidth > 900) {
      toggleDesktop();
    } else {
      sidebar.classList.contains('open') ? closeMobile() : openMobile();
    }
  }

  toggle?.addEventListener('click', handleToggle);
  overlay?.addEventListener('click', closeMobile);

  window.addEventListener('resize', () => {
    if (window.innerWidth > 900) {
      sidebar.classList.remove('open');
      overlay.classList.remove('visible');
      document.body.style.overflow = '';
    }
  });
}

// ── Source nav ─────────────────────────────────────────────────────
function initSourceNav() {
  $$('.snav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      const src = btn.dataset.source;
      const switching = src !== activeSource;
      $$('.snav-item').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      $$('.source-panel').forEach(p => p.classList.remove('active'));
      const panel = $(`panel-${src}`);
      if (panel) panel.classList.add('active');
      activeSource = src;

      if (switching) resetSummarizerState();

      if (window.innerWidth <= 900) {
        $('sidebar').classList.remove('open');
        $('sidebarOverlay').classList.remove('visible');
        document.body.style.overflow = '';
      }
    });
  });
}

// ── Output options ─────────────────────────────────────────────────
function initOutputOptions() {
  // Radio inputs in sidebar — already wired via HTML
}

// ── Drop zones ─────────────────────────────────────────────────────
function initDropZones() {
  ['pdf', 'audio', 'video', 'image'].forEach(type => {
    const zone = $(`dropZone-${type}`);
    if (!zone) return;
    ['dragenter', 'dragover'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('drag-over'); }));
    ['dragleave', 'drop'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('drag-over'); }));
    if (type === 'image') {
      zone.addEventListener('drop', e => addImageFiles(e.dataTransfer.files));
    } else {
      zone.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) handleFile(type, f); });
    }
  });
}

function initFileInputs() {
  ['pdf', 'audio', 'video'].forEach(type => {
    const inp = $(`fileInput-${type}`);
    if (!inp) return;
    inp.addEventListener('change', () => { if (inp.files[0]) handleFile(type, inp.files[0]); });
  });
  const imgInp = $('fileInput-image');
  imgInp?.addEventListener('change', () => {
    addImageFiles(imgInp.files);
    imgInp.value = '';  // allow re-picking the same file(s) later — browsers won't fire 'change' again otherwise
  });
}

function handleFile(type, file) {
  selectedFiles[type] = file;
  const zone = $(`dropZone-${type}`);
  zone?.classList.add('has-file');
  const prev = $(`preview-${type}`);
  if (prev) {
    const size = (file.size / 1024 / 1024).toFixed(2);
    prev.innerHTML = `
      <div class="file-selected-row">
        <div class="file-selected-info">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
          <span class="file-selected-name">${escHtml(file.name)}</span>
          <span class="file-selected-size">(${size} MB)</span>
        </div>
        <button type="button" class="change-file-btn" data-clear-type="${type}">Replace File</button>
      </div>`;
    prev.querySelector('[data-clear-type]')?.addEventListener('click', () => clearFile(type));
  }
}

function clearFile(type) {
  selectedFiles[type] = null;
  $(`dropZone-${type}`)?.classList.remove('has-file');
  const prev = $(`preview-${type}`);
  if (prev) prev.innerHTML = '';
  const inp = $(`fileInput-${type}`);
  if (inp) inp.value = '';
}

// ── Image gallery (multi-file) ────────────────────────────────────────
// Guest/logged-in caps mirror GUEST_MAX_IMAGES/USER_MAX_IMAGES in app.py —
// this is UX only (instant toast); the server enforces its own copy of the
// same limit regardless of what a client sends.
function addImageFiles(fileList) {
  const incoming = Array.from(fileList || []).filter(f => f.type?.startsWith('image/') || !f.type);
  if (!incoming.length) return;
  const room = maxImages() - selectedImages.length;
  if (room <= 0) { showImageLimitToast(); return; }
  selectedImages.push(...incoming.slice(0, room));
  renderImageGallery();
  if (incoming.length > room) showImageLimitToast();
}

function removeImageAt(index) {
  selectedImages.splice(index, 1);
  renderImageGallery();
}

function replaceImageAt(index, file) {
  if (!file) return;
  selectedImages[index] = file;
  renderImageGallery();
}

function renderImageGallery() {
  const grid = $('imageGallery');
  if (!grid) return;
  $('dropZone-image')?.classList.toggle('has-file', selectedImages.length > 0);

  const cards = selectedImages.map((file, i) => `
    <div class="img-card" id="img-card-${i}">
      <div class="img-card-thumb skel"></div>
      <div class="img-card-info">
        <span class="img-card-name" title="${escHtml(file.name)}">${escHtml(file.name)}</span>
        <span class="img-card-size">${(file.size / 1024 / 1024).toFixed(2)} MB</span>
      </div>
      <div class="img-card-actions">
        <button type="button" class="img-card-btn replace" data-action="replace" data-idx="${i}" title="Replace image">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.99"/></svg>
        </button>
        <button type="button" class="img-card-btn remove" data-action="remove" data-idx="${i}" title="Remove image">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
    </div>`).join('');

  const addCard = selectedImages.length < maxImages() ? `
    <button type="button" class="img-add-card" id="addImageCard">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      <span>Add Image</span>
    </button>` : '';

  grid.innerHTML = cards + addCard;

  selectedImages.forEach((file, i) => {
    const thumb = document.querySelector(`#img-card-${i} .img-card-thumb`);
    if (!thumb) return;
    const reader = new FileReader();
    reader.onload = e => { thumb.classList.remove('skel'); thumb.style.backgroundImage = `url(${e.target.result})`; };
    reader.readAsDataURL(file);
  });

  grid.querySelectorAll('[data-action="remove"]').forEach(btn =>
    btn.addEventListener('click', () => removeImageAt(parseInt(btn.dataset.idx))));
  grid.querySelectorAll('[data-action="replace"]').forEach(btn =>
    btn.addEventListener('click', () => {
      const picker = document.createElement('input');
      picker.type = 'file'; picker.accept = 'image/*'; picker.hidden = true;
      picker.addEventListener('change', () => replaceImageAt(parseInt(btn.dataset.idx), picker.files[0]));
      document.body.appendChild(picker); picker.click(); picker.remove();
    }));
  $('addImageCard')?.addEventListener('click', () => $('fileInput-image')?.click());
}

function showImageLimitToast() {
  if (currentUser) {
    showToast({ type: 'info', title: 'Upload limit reached', message: `You can add up to ${MAX_IMAGES_USER} images at a time.` });
  } else {
    showToast({
      type: 'info', title: 'Upload limit reached',
      message: `Sign in to upload more images and unlock a higher upload limit (up to ${MAX_IMAGES_USER}).`,
      actions: [{ label: 'Sign In', primary: true, onClick: () => $('appSignIn')?.click() }],
      duration: 9000,
    });
  }
}

function resetSummarizerState() {
  $('resultsSection') && ($('resultsSection').style.display = 'none');
  $('youtubeUrl') && ($('youtubeUrl').value = '');
  $('textContent') && ($('textContent').value = '');
  $('textMeta') && ($('textMeta').textContent = '0 words');

  selectedFiles = {};
  selectedImages = [];
  currentSummary = ''; currentLang = 'en'; historyId = null;

  ['pdf', 'audio', 'video'].forEach(t => {
    const p = $(`preview-${t}`); if (p) p.innerHTML = '';
    const i = $(`fileInput-${t}`); if (i) i.value = '';
    $(`dropZone-${t}`)?.classList.remove('has-file');
  });
  renderImageGallery();
  $('fileInput-image') && ($('fileInput-image').value = '');

  $('textResultCard') && ($('textResultCard').style.display = 'none');
  $('summaryText') && ($('summaryText').innerHTML = '');
  $('audioResultCard') && ($('audioResultCard').style.display = 'none');
  $('audioPlayer') && $('audioPlayer').removeAttribute('src');
  $('saveHistoryBtn') && ($('saveHistoryBtn').style.display = 'none');
  $('switchLang') && ($('switchLang').value = 'en');
  $('resultsMeta') && ($('resultsMeta').textContent = '');

  hideLoading();
  ['step1', 'step2', 'step3'].forEach(id => $(id)?.classList.remove('active', 'done'));
}

function initTextCounter() {
  const ta = $('textContent'), meta = $('textMeta');
  if (!ta || !meta) return;
  ta.addEventListener('input', () => {
    const w = ta.value.trim() ? ta.value.trim().split(/\s+/).length : 0;
    meta.textContent = `${w} word${w !== 1 ? 's' : ''}`;
  });
}

function initSummarizeBtn() {
  $('summarizeBtn')?.addEventListener('click', doSummarize);
}

async function doSummarize() {
  if (!validate()) return;
  const outputType = document.querySelector('input[name="output_type"]:checked')?.value || 'text';
  const targetLang = $('targetLang')?.value || 'auto';

  showLoading();
  animateSteps();
  historyId = null;

  try {
    const fd = buildFormData(outputType, targetLang);
    const res = await fetch(`${API}/api/summarize`, { method: 'POST', body: fd, credentials: 'include' });
    const data = await parseJsonSafe(res);
    hideLoading();
    if (!data.success) { showError(data.error || 'Summarization failed.'); return; }
    showResults(data, outputType);
  } catch (err) {
    hideLoading();
    if (err instanceof TypeError) {

      showError(`Cannot reach backend. Is app.py running?\n${err.message}`);
    } else {
      showError(err.message);
    }
  }
}

function validate() {
  if (activeSource === 'youtube') {
    const u = $('youtubeUrl')?.value?.trim();
    if (!u) { showError('Please enter a YouTube URL.'); return false; }
    if (!u.includes('youtube.com') && !u.includes('youtu.be')) { showError("That doesn't look like a YouTube URL."); return false; }
  } else if (activeSource === 'text') {
    const t = $('textContent')?.value?.trim();
    if (!t || t.length < 20) { showError('Please enter at least 20 characters.'); return false; }
  } else if (activeSource === 'image') {
    if (!selectedImages.length) { showError('Please select at least one image.'); return false; }
  } else {
    if (!selectedFiles[activeSource]) { showError(`Please select a ${activeSource} file.`); return false; }
  }
  return true;
}

function buildFormData(outputType, targetLang) {
  const fd = new FormData();
  fd.append('source_type', activeSource);
  fd.append('output_type', outputType);
  fd.append('target_lang', targetLang);
  if (activeSource === 'youtube') fd.append('youtube_url', $('youtubeUrl').value.trim());
  else if (activeSource === 'text') fd.append('text_content', $('textContent').value.trim());
  else if (activeSource === 'image') selectedImages.forEach(f => fd.append('files', f));
  else fd.append('file', selectedFiles[activeSource]);
  return fd;
}

function showResults(data, outputType) {
  currentSummary = data.summary || '';
  currentLang    = data.output_language || 'en';
  historyId      = data.history_id || null;

  const section = $('resultsSection');
  section.style.display = 'flex';
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const method = data.method ? ` · ${data.method}` : '';
  const lang   = data.output_language ? ` · ${data.output_language.toUpperCase()}` : '';
  $('resultsMeta').textContent = `${(data.word_count || 0).toLocaleString()} words${method}${lang}`;

  const sw = $('switchLang');
  if (sw) sw.value = currentLang;

  const tc = $('textResultCard'), tt = $('summaryText');
  if (outputType === 'text' || outputType === 'both') {
    tc.style.display = 'block';
    renderStructuredSummary(tt, currentSummary);
  } else { tc.style.display = 'none'; }

  const shb = $('saveHistoryBtn');
  if (shb) {
    if (historyId) {
      shb.style.display = 'inline-flex';
      shb.style.color = ''; shb.title = 'Saved to your history';
    } else if (data.history_error) {
      shb.style.display = 'inline-flex';
      shb.innerHTML = '⚠ Not saved';
      shb.style.color = '#F59E0B';
      shb.title = data.history_error;
    } else {
      shb.style.display = 'none';
    }
  }

  const ac = $('audioResultCard');
  if (outputType === 'audio' || outputType === 'both') {
    if (data.audio_url) {
      ac.style.display = 'block';
      $('audioPlayer').src       = `${API}${data.audio_url}?t=${Date.now()}`;
      $('downloadAudioBtn').href  = `${API}${data.audio_url}`;
    } else if (data.audio_error) {
      showError(`Audio: ${data.audio_error}`);
      if (outputType === 'both') { tc.style.display = 'block'; renderStructuredSummary(tt, currentSummary); }
    }
  } else { ac.style.display = 'none'; }
}

function initResultActions() {

  $('copyBtn')?.addEventListener('click', () => {
    if (!currentSummary) return;
    const clean = currentSummary.replace(/\n##\s+/g, '\n\n').trim();
    navigator.clipboard.writeText(clean).then(() => {
      const b = $('copyBtn');
      b.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>Copied!`;
      b.style.color = 'var(--cyan)';
      setTimeout(() => { b.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy`; b.style.color = ''; }, 2000);
    });
  });

  // Download PDF — works on any summary immediately, no save/login required
  $('downloadPdfBtn')?.addEventListener('click', async () => {
    if (!currentSummary) return;
    const btn = $('downloadPdfBtn');
    const original = btn.innerHTML;
    btn.disabled = true; btn.textContent = 'Preparing…';
    try {
      const meta = {
        source_type: activeSource,
        summary_language: currentLang,
        original_word_count: parseInt(($('resultsMeta')?.textContent || '').replace(/[^0-9]/g, '')) || 0,
      };
      const title = activeSource === 'youtube' ? ($('youtubeUrl')?.value || 'YouTube Summary')
                  : activeSource === 'text' ? 'Text Summary'
                  : activeSource === 'image' ? (selectedImages.length > 1 ? `${selectedImages.length} Images` : (selectedImages[0]?.name || 'Image Summary'))
                  : (selectedFiles[activeSource]?.name || 'NovaBrief Summary');
      const res = await fetch(`${API}/api/export/pdf`, {
        method: 'POST', credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, summary: currentSummary, meta }),
      });
      if (!res.ok) { const d = await parseJsonSafe(res); throw new Error(d.error || 'PDF export failed.'); }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `${title.replace(/[^A-Za-z0-9 _-]/g,'').slice(0,50) || 'summary'}.pdf`;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      showError(e.message || 'Could not generate PDF.');
    } finally {
      btn.disabled = false; btn.innerHTML = original;
    }
  });

  $('resetBtn')?.addEventListener('click', () => {
    resetSummarizerState();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  $('switchLangBtn')?.addEventListener('click', async () => {
    if (!currentSummary) return;
    const tgt = $('switchLang').value;
    if (tgt === currentLang) return;
    const btn = $('switchLangBtn');
    btn.textContent = 'Translating…';
    try {
      const d = await apiFetch('/api/translate', 'POST', { text: currentSummary, source_lang: currentLang, target_lang: tgt });
      if (d.success) {
        currentSummary = d.translated; currentLang = tgt;
        renderStructuredSummary($('summaryText'), currentSummary);
        $('textResultCard').style.display = 'block';
        $('resultsMeta').textContent += ` · translated → ${tgt.toUpperCase()}`;
      } else { showError(d.error); }
    } catch (e) { showError('Translation failed.'); }
    btn.textContent = 'Apply';
  });

  $('regenAudioBtn')?.addEventListener('click', async () => {
    if (!currentSummary) return;
    const lang = $('switchLang')?.value || currentLang;
    const btn = $('regenAudioBtn');
    btn.textContent = 'Generating…';
    try {
      const d = await apiFetch('/api/tts', 'POST', { text: currentSummary, lang });
      if (d.success) {
        $('audioResultCard').style.display = 'block';
        $('audioPlayer').src = `${API}${d.audio_url}?t=${Date.now()}`;
        $('downloadAudioBtn').href = `${API}${d.audio_url}`;
      } else { showError(d.error); }
    } catch { showError('Audio generation failed.'); }
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>Narrate`;
  });
}

function showLoading() {
  $('loadingOverlay').style.display = 'flex';
  const labels = {
    youtube: ['Fetching transcript', 'Running AI summarizer'],
    pdf:     ['Reading PDF pages',   'Running AI summarizer'],
    text:    ['Analysing text',      'Running AI summarizer'],
    image:   [`Analysing ${selectedImages.length > 1 ? selectedImages.length + ' images' : 'image'}`, 'Generating description'],
    audio:   ['Transcribing audio',  'Running AI summarizer'],
    video:   ['Extracting audio',    'Transcribing & summarizing'],
  };
  const [title, sub] = labels[activeSource] || ['Processing…', 'Please wait'];
  $('loadingTitle').textContent = title;
  $('loadingSub').textContent   = sub;
  ['step1','step2','step3'].forEach(id => $( id).classList.remove('active','done'));
  $('step1').classList.add('active');
}
function hideLoading() { $('loadingOverlay').style.display = 'none'; }
function animateSteps() {
  [1200, 3000, 5500].forEach((delay, i) => {
    setTimeout(() => {
      const ids = ['step1','step2','step3'];
      if (i > 0) $(ids[i-1]).classList.replace('active','done');
      $(ids[i]).classList.add('active');
    }, delay);
  });
}

const TOAST_ICONS = {
  error: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
  info:  '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
};

function showToast({ type = 'info', title = '', message = '', actions = [], duration = 8000 } = {}) {
  let stack = $('toastStack');
  if (!stack) {
    stack = document.createElement('div');
    stack.id = 'toastStack';
    stack.className = 'toast-stack';
    document.body.appendChild(stack);
  }
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `
    <div class="toast-icon">${TOAST_ICONS[type] || TOAST_ICONS.info}</div>
    <div class="toast-body">
      ${title ? `<div class="toast-title">${escHtml(title)}</div>` : ''}
      <div class="toast-msg">${escHtml(message)}</div>
      ${actions.length ? `<div class="toast-actions">${actions.map((a, i) => `<button type="button" class="toast-action-btn${a.primary ? ' primary' : ''}" data-i="${i}">${escHtml(a.label)}</button>`).join('')}</div>` : ''}
    </div>
    <button type="button" class="toast-close" aria-label="Close">✕</button>`;
  stack.appendChild(el);
  requestAnimationFrame(() => el.classList.add('toast-in'));

  const remove = () => { el.classList.remove('toast-in'); el.classList.add('toast-out'); setTimeout(() => el.remove(), 250); };
  el.querySelector('.toast-close').addEventListener('click', remove);
  actions.forEach((a, i) => el.querySelector(`[data-i="${i}"]`)?.addEventListener('click', () => { a.onClick?.(); remove(); }));
  if (duration) setTimeout(remove, duration);
  return remove;
}

function showError(msg) {
  showToast({ type: 'error', message: msg, duration: 8000 });
}

async function parseJsonSafe(res) {
  const raw = await res.text();
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error(
      res.ok
        ? 'The server sent back something that was not JSON. Check the app.py terminal for errors.'
        : `Server error (${res.status}). Check the app.py terminal for details.`
    );
  }
}

async function apiFetch(path, method = 'GET', body = null) {
  const opts = { method, credentials: 'include', headers: {} };
  if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const res = await fetch(`${API}${path}`, opts);
  return parseJsonSafe(res);
}
