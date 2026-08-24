(function () {
  var saved = localStorage.getItem('nb-theme') || 'dark';

  if (saved === 'system') { saved = 'dark'; localStorage.setItem('nb-theme', 'dark'); }
  document.documentElement.setAttribute('data-theme', saved);
})();

var NovaBriefTheme = (function () {

  var THEMES = ['dark', 'light'];
  var ICONS  = { dark: '🌙', light: '☀️' };
  var LABELS = { dark: 'Dark',  light: 'Light' };

  function current() {
    return localStorage.getItem('nb-theme') || 'dark';
  }

  function apply(theme) {
    if (!THEMES.includes(theme)) theme = 'dark';
    localStorage.setItem('nb-theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
    updateAllButtons(theme);
  }

  function cycle() {
    var idx = THEMES.indexOf(current());
    apply(THEMES[(idx + 1) % THEMES.length]);
  }

  function updateAllButtons(active) {

    document.querySelectorAll('[data-set-theme]').forEach(function (btn) {
      var isActive = btn.dataset.setTheme === active;
      btn.classList.toggle('theme-btn-active', isActive);
      btn.setAttribute('aria-pressed', isActive);
    });

    document.querySelectorAll('.theme-cycle-btn').forEach(function (btn) {
      btn.textContent = ICONS[active] || '🌙';
      btn.title = 'Theme: ' + (LABELS[active] || 'Dark') + ' (click to switch)';
    });
  }

  function buildSwitcher(container) {
    if (!container) return;
    container.innerHTML = '';
    container.className = 'theme-switcher';
    THEMES.forEach(function (t) {
      var btn = document.createElement('button');
      btn.className    = 'theme-btn';
      btn.dataset.setTheme = t;
      btn.title        = LABELS[t];
      btn.setAttribute('aria-pressed', t === current());
      btn.innerHTML    = ICONS[t] + '<span>' + LABELS[t] + '</span>';
      btn.addEventListener('click', function () { apply(t); });
      container.appendChild(btn);
    });
    updateAllButtons(current());
  }

  function init() {

    document.querySelectorAll('.theme-switcher-placeholder').forEach(buildSwitcher);

    document.querySelectorAll('[data-set-theme]').forEach(function (btn) {
      btn.addEventListener('click', function () { apply(btn.dataset.setTheme); });
    });

    document.querySelectorAll('.theme-cycle-btn').forEach(function (btn) {
      btn.addEventListener('click', cycle);
    });
    updateAllButtons(current());
  }

  document.addEventListener('DOMContentLoaded', init);

  return { apply: apply, cycle: cycle, current: current };
})();
