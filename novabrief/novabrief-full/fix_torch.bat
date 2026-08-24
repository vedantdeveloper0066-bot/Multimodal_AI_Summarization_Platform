@echo off
echo ============================================
echo  NovaBrief — PyTorch Fix Script (Windows)
echo ============================================
echo.
echo This script tries 5 different ways to install PyTorch.
echo.

python --version >nul 2>&1
if errorlevel 1 (echo [ERROR] Python not found & pause & exit /b 1)
echo Python: & python --version & echo.

echo [PRE] Upgrading pip...
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org -q
echo.

echo [1/5] Standard CPU wheel...
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q
if not errorlevel 1 goto verify

echo [2/5] With trusted-host flags (fixes SSL errors)...
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --trusted-host files.pythonhosted.org --trusted-host pypi.org -q
if not errorlevel 1 goto verify

echo [3/5] PyPI standard build...
python -m pip install torch --trusted-host files.pythonhosted.org --trusted-host pypi.org -q
if not errorlevel 1 goto verify

echo [4/5] No-cache + trusted-host...
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --no-cache-dir -q
if not errorlevel 1 goto verify

echo [5/5] System-level install...
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --break-system-packages -q 2>nul
if not errorlevel 1 goto verify

echo.
echo ============================================
echo  ALL 5 METHODS FAILED
echo ============================================
echo.
echo Possible causes:
echo   * No internet / firewall blocking downloads
echo   * Low disk space (PyTorch needs ~2 GB free)
echo   * Antivirus blocking pip
echo.
echo Manual fix:
echo   1. Visit https://download.pytorch.org/whl/cpu/torch/
echo   2. Download the .whl matching your Python version
echo   3. Run: python -m pip install downloaded_file.whl
echo.
echo The app STILL WORKS using extractive summarization.
echo You do NOT need PyTorch to use NovaBrief.
echo.
pause & exit /b 1

:verify
echo.
python -c "import torch; print('[OK] PyTorch', torch.__version__)"
if errorlevel 1 (echo [ERROR] Import failed - restart terminal & pause & exit /b 1)

python -m pip install transformers sentencepiece --trusted-host files.pythonhosted.org --trusted-host pypi.org -q
echo.
echo ============================================
echo  Done! Start the server: python app.py
echo  OR click [Retry] in the NovaBrief UI
echo ============================================
pause
