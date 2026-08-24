const API = 'http://localhost:5000';
const $ = id => document.getElementById(id);

document.addEventListener('DOMContentLoaded', () => {
  init();
  initLogout();
});

async function parseJsonSafe(res) {
  const raw = await res.text();
  try { return JSON.parse(raw); }
  catch { throw new Error(`Server error (${res.status}).`); }
}
async function apiFetch(path, method = 'GET') {
  const res = await fetch(`${API}${path}`, { method, credentials: 'include' });
  return parseJsonSafe(res);
}

function deny(message) {
  $('adminSub').textContent = '';
  $('adminContent').style.display = 'none';
  $('adminDenied').style.display = 'block';
  if (message) $('adminDeniedMsg').textContent = message;
}

async function init() {
  let me;
  try {
    me = await apiFetch('/api/auth/me');
  } catch {
    deny('Could not reach the backend. Is app.py running?');
    return;
  }
  if (!me.logged_in) { deny('Sign in with an admin account to view this page.'); return; }
  $('userName').textContent = me.user.name;
  if (!me.user.is_admin) {
    deny(`${me.user.email} is not on the admin list. Add it to ADMIN_EMAILS in config.py to grant access.`);
    return;
  }

  try {
    const d = await apiFetch('/api/admin/users');
    if (!d.success) { deny(d.error || 'Access denied.'); return; }
    $('adminContent').style.display = 'block';
    $('adminSub').textContent = `${d.total} registered user${d.total !== 1 ? 's' : ''}`;
    const totalSummaries = d.users.reduce((sum, u) => sum + (u.summary_count || 0), 0);
    $('adminStats').innerHTML = `
      <div class="stat-card"><div class="stat-num">${d.total}</div><div class="stat-label">Total Users</div></div>
      <div class="stat-card"><div class="stat-num">${totalSummaries}</div><div class="stat-label">Total Summaries</div></div>
    `;
    $('usersBody').innerHTML = d.users.map(u => `
      <tr>
        <td>${escHtml(u.name)}${u.is_admin ? '<span class="admin-badge">ADMIN</span>' : ''}</td>
        <td>${escHtml(u.email)}</td>
        <td>${escHtml(u.created_at || '—')}</td>
        <td>${u.summary_count || 0}</td>
        <td>${escHtml(u.last_active || '—')}</td>
      </tr>`).join('') || `<tr><td colspan="5" style="text-align:center;color:var(--muted)">No users yet.</td></tr>`;
  } catch (e) {
    deny(e.message || 'Something went wrong loading the admin data.');
  }
}

function initLogout() {
  $('logoutBtn')?.addEventListener('click', async () => {
    await apiFetch('/api/auth/logout', 'POST');
    window.location.href = '/';
  });
}

function escHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
