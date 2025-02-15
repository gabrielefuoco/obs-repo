@echo off
REM Imposta ROOT_DIR come la directory in cui si trova questo script
set ROOT_DIR=%~dp0

REM Verifica che ROOT_DIR sia impostato correttamente
if not defined ROOT_DIR (
    echo Errore: Impossibile determinare ROOT_DIR.
    pause
    exit /b 1
)

REM Stampa ROOT_DIR per debug (opzionale)
echo ROOT_DIR: %ROOT_DIR%

REM Esegui lo script Python
python "%ROOT_DIR%src\main.py" "%ROOT_DIR%output\original" "%ROOT_DIR%output\formatted" --prompt_folder "%ROOT_DIR%src\Prompt" --split_method headers

REM Verifica il codice di uscita
IF ERRORLEVEL 1 (
    echo "Final formatting script failed. Exiting..."
    pause
    exit /b 1
)

echo "All scripts executed successfully."
pause
exit /b 0