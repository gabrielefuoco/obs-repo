@echo off
setlocal enabledelayedexpansion

:: Imposta la directory corrente come base
cd /d "%~dp0"

:: Crea file .env se non esiste
if not exist api-keys.env (
    echo GROQ_API_KEY=your_api_key_here > .env
)

:: Crea file prompt.txt se non esiste
if not exist prompt.txt (
    echo Analizza il contenuto delle immagini, descrivendone gli elementi principali > prompt.txt
)

:: Crea cartella img se non esiste
if not exist img mkdir img

:: Crea cartella results se non esiste
if not exist results mkdir results

:: Attiva ambiente virtuale se esiste
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate
)

:: Installa dipendenze
pip install groq python-dotenv
pip install pytesseract Pillow opencv-python numpy
pip install pymupdf


:: Passa i percorsi come argomenti allo script Python
python script.py "%cd%\img" "%cd%\prompt.txt" "%cd%\results"

endlocal