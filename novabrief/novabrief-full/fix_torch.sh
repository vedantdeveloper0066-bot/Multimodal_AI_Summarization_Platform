#!/bin/bash
echo "============================================"
echo " NovaBrief — PyTorch Fix Script (Linux/macOS)"
echo "============================================"
echo

PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then echo "[ERROR] Python not found"; exit 1; fi
PIP="$PY -m pip"
echo "Python: $($PY --version)"
echo "Pip:    $($PIP --version)"
echo

echo "[PRE] Upgrading pip..."
$PIP install --upgrade pip -q 2>/dev/null || true
echo

try_install() {
  echo -n "Trying: $* ... "
  if $PIP install "$@" -q 2>/dev/null; then
    echo "OK"
    return 0
  fi
  echo "FAILED"
  return 1
}

echo "[1/5] Standard CPU wheel..."
try_install torch --index-url https://download.pytorch.org/whl/cpu && goto_verify=1

if [ -z "$goto_verify" ]; then
  echo "[2/5] With trusted-host flags (SSL fix)..."
  try_install torch --index-url https://download.pytorch.org/whl/cpu \
    --trusted-host download.pytorch.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org && goto_verify=1
fi

if [ -z "$goto_verify" ]; then
  echo "[3/5] Plain PyPI..."
  try_install torch && goto_verify=1
fi

if [ -z "$goto_verify" ]; then
  echo "[4/5] No-cache + trusted-host..."
  try_install torch --index-url https://download.pytorch.org/whl/cpu \
    --trusted-host download.pytorch.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org \
    --no-cache-dir && goto_verify=1
fi

if [ -z "$goto_verify" ]; then
  echo "[5/5] Break-system-packages (Linux)..."
  try_install torch --index-url https://download.pytorch.org/whl/cpu \
    --trusted-host download.pytorch.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org \
    --break-system-packages 2>/dev/null && goto_verify=1
fi

if [ -z "$goto_verify" ]; then
  echo
  echo "============================================"
  echo " ALL METHODS FAILED"
  echo "============================================"
  echo
  echo "Possible causes:"
  echo "  * No internet / firewall"
  echo "  * Disk space < 2 GB"
  echo "  * Proxy required (set http_proxy env var)"
  echo
  echo "Manual fix:"
  echo "  Visit https://download.pytorch.org/whl/cpu/torch/"
  echo "  Download the .whl for your Python version, then:"
  echo "  python3 -m pip install downloaded_file.whl"
  echo
  echo "The app STILL WORKS without PyTorch (extractive mode)."
  exit 1
fi

echo
$PY -c "import torch; print('[OK] PyTorch', torch.__version__)" || {
  echo "[ERROR] Import failed after install — try restarting terminal"
  exit 1
}

$PIP install transformers sentencepiece -q 2>/dev/null || true

echo
echo "============================================"
echo " Done! Start the server: python3 app.py"
echo " OR click [Retry] in the NovaBrief UI"
echo "============================================"
