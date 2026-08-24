const API = 'http://localhost:5000';
const $ = id => document.getElementById(id);

let allHistory = [];
let currentDetailRaw = ''; // raw '## Heading' summary text of whichever item is open in the modal

document.addEventListener('DOMContentLoaded', () => {
  checkAuth();
  loadHistory();
  initFilters();
  initDetailModal();
  initLogout();
});

async function apiFetch(path, method = 'GET', body = null) {
  const opts = { method, credentials: 'include', headers: {} };
  if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const res = await fetch(`${API}${path}`, opts);
  return res.json();
}

async function checkAuth() {
  try {
    const d = await apiFetch('/api/auth/me');
    if (d.logged_in) {
      $('userName').textContent = d.user.name;
    } else {
      window.location.href = '/?login=1';
    }
  } catch { window.location.href = '/'; }
}

async function loadHistory() {
  try {
    const d = await apiFetch('/api/history');
    if (!d.success) { showEmpty(); return; }
    allHistory = d.history || [];
    const sub = $('historySub');
    if (sub) sub.textContent = `${allHistory.length} saved summary${allHistory.length !== 1 ? 'ies' : 'y'}`;
    renderGrid(allHistory);
  } catch { showEmpty(); }
}

function renderGrid(items) {
  const grid = $('historyGrid');
  const empty = $('emptyState');
  if (!items.length) { grid.innerHTML = ''; showEmpty(); return; }
  empty && (empty.style.display = 'none');

  const TYPE_ICONS = {
    youtube: '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>',
    pdf:     '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    text:    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="17" y1="10" x2="3" y2="10"/><line x1="21" y1="6" x2="3" y2="6"/><line x1="21" y1="14" x2="3" y2="14"/></svg>',
    image:   '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
    audio:   '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/></svg>',
    video:   '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>',
  };

  grid.innerHTML = items.map(item => {
    const ico = TYPE_ICONS[item.source_type] || '';
    const audioTag = item.audio_filename
      ? `<span class="hc-audio-badge">🎵 Audio</span>` : '';
    return `
    <div class="hc" data-id="${item.id}">
      <div class="hc-top">
        <span class="hc-type ${item.source_type}">${ico} ${item.source_type}</span>
        <span class="hc-date">${item.created_at || ''}</span>
      </div>
      <div class="hc-title">${escHtml(item.title || 'Untitled')}</div>
      <div class="hc-preview">${escHtml(item.preview || '')}…</div>
      <div class="hc-footer">
        <div class="hc-stats">
          <span class="hc-stat">${(item.original_word_count || 0).toLocaleString()} words</span>
          <span class="hc-lang">${(item.summary_language || 'en').toUpperCase()}</span>
          ${audioTag}
        </div>
        <div class="hc-actions">
          <a class="hc-btn" href="/api/history/${item.id}/pdf" title="Download PDF" target="_blank" onclick="event.stopPropagation()">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          </a>
          <button class="hc-btn del" title="Delete" onclick="event.stopPropagation();deleteItem(${item.id},this)">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
          </button>
        </div>
      </div>
    </div>`;
  }).join('');

  // Click to open detail
  grid.querySelectorAll('.hc').forEach(card => {
    card.addEventListener('click', () => openDetail(parseInt(card.dataset.id)));
  });
}

function showEmpty() {
  $('historyGrid').innerHTML = '';
  $('emptyState') && ($('emptyState').style.display = 'block');
}

// ── Filters ────────────────────────────────────────────────────────
function initFilters() {
  ['filterType','filterLang','filterSearch'].forEach(id => {
    $(id)?.addEventListener('input', applyFilters);
    $(id)?.addEventListener('change', applyFilters);
  });
}

function applyFilters() {
  const type   = $('filterType')?.value   || '';
  const lang   = $('filterLang')?.value   || '';
  const search = ($('filterSearch')?.value || '').toLowerCase();
  const filtered = allHistory.filter(item => {
    if (type   && item.source_type     !== type)    return false;
    if (lang   && item.summary_language !== lang)    return false;
    if (search && !(item.title || '').toLowerCase().includes(search)) return false;
    return true;
  });
  renderGrid(filtered);
}

// ── Detail modal ───────────────────────────────────────────────────
function initDetailModal() {
  const overlay = $('detailOverlay'), cl = $('detailClose');
  cl?.addEventListener('click', () => { overlay.style.display = 'none'; });
  overlay?.addEventListener('click', e => { if (e.target === overlay) overlay.style.display = 'none'; });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') overlay.style.display = 'none'; });

  $('detailCopy')?.addEventListener('click', () => {
    const text = (currentDetailRaw || $('detailBody')?.textContent || '').replace(/\n##\s+/g, '\n\n').trim();
    if (text) navigator.clipboard.writeText(text).then(() => {
      const b = $('detailCopy'); b.textContent = '✓ Copied!'; b.style.color = 'var(--cyan)';
      setTimeout(() => { b.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy`; b.style.color = ''; }, 2000);
    });
  });
}

async function openDetail(id) {
  try {
    const d = await apiFetch(`/api/history/${id}`);
    if (!d.success) return;
    const s = d.summary;
    $('detailOverlay').style.display = 'flex';
    $('detailMeta').innerHTML = `
      <span class="hc-type ${s.source_type}" style="font-size:.7rem">${s.source_type}</span>
      <span>${s.created_at || ''}</span>
      <span class="hc-lang">${(s.summary_language || 'en').toUpperCase()}</span>
      <span style="color:var(--muted)">${(s.original_word_count || 0).toLocaleString()} words</span>`;
    $('detailTitle').textContent = s.title || 'Summary';
    currentDetailRaw = s.summary_text || '';
    renderStructuredSummary($('detailBody'), currentDetailRaw);
    $('detailPdf').href          = `/api/history/${id}/pdf`;

    const da = $('detailAudio');
    if (s.audio_filename) {
      da.style.display = 'block';
      $('detailPlayer').src    = `${API}/uploads/${s.audio_filename}`;
      $('detailAudioDl').href  = `${API}/uploads/${s.audio_filename}`;
    } else { da.style.display = 'none'; }
  } catch {}
}

async function deleteItem(id, btn) {
  if (!confirm('Delete this summary?')) return;
  btn.disabled = true;
  try {
    await apiFetch(`/api/history/${id}`, 'DELETE');
    allHistory = allHistory.filter(i => i.id !== id);
    applyFilters();
    const sub = $('historySub');
    if (sub) sub.textContent = `${allHistory.length} saved summar${allHistory.length !== 1 ? 'ies' : 'y'}`;
  } catch { btn.disabled = false; }
}

function initLogout() {
  $('logoutBtn')?.addEventListener('click', async () => {
    await apiFetch('/api/auth/logout', 'POST');
    window.location.href = '/';
  });
}
