function escHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Runs on ALREADY-escaped text, so the only '<' / '>' left in the string are
// the ones inserted right here — safe against injection via summary content.
function renderInline(escaped) {
  return escaped
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+?)`/g, '<code>$1</code>');
}

function isBulletBody(body) {
  const lines = (body || '').split('\n').map(l => l.trim()).filter(Boolean);
  if (!lines.length) return false;
  const marked = lines.filter(l => /^[-•*]/.test(l)).length;
  return marked / lines.length >= 0.6;
}

function isNumberedBody(body) {
  const lines = (body || '').split('\n').map(l => l.trim()).filter(Boolean);
  if (!lines.length) return false;
  const marked = lines.filter(l => /^\d+[.)]\s/.test(l)).length;
  return marked / lines.length >= 0.6;
}

function renderStructuredSummary(container, raw) {
  if (!container) return;
  raw = raw || '';
  const parts = raw.split(/\n##\s+/);
  const overview = parts[0].trim();
  let html = overview ? `<p class="sum-overview">${renderInline(escHtml(overview))}</p>` : '';

  for (let i = 1; i < parts.length; i++) {
    const nl = parts[i].indexOf('\n');
    const heading = (nl === -1 ? parts[i] : parts[i].slice(0, nl)).trim();
    const body    = (nl === -1 ? ''      : parts[i].slice(nl + 1)).trim();
    html += `<h4 class="sum-heading">${escHtml(heading)}</h4>`;

    if (isBulletBody(body)) {
      const items = body.split('\n').map(l => l.trim().replace(/^[-•*]\s*/, '')).filter(Boolean);
      html += `<ul class="sum-takeaways">${items.map(it => `<li>${renderInline(escHtml(it))}</li>`).join('')}</ul>`;
    } else if (isNumberedBody(body)) {
      const items = body.split('\n').map(l => l.trim().replace(/^\d+[.)]\s*/, '')).filter(Boolean);
      html += `<ol class="sum-numbered">${items.map(it => `<li>${renderInline(escHtml(it))}</li>`).join('')}</ol>`;
    } else {
      html += `<p class="sum-para">${renderInline(escHtml(body))}</p>`;
    }
  }
  container.innerHTML = html || `<p class="sum-para">${renderInline(escHtml(raw))}</p>`;
}
