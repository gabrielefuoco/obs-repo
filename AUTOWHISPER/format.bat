

REM Esegui lo script format.py
python "%ROOT_DIR%src\format\format.py" "%ROOT_DIR%output\original" "%ROOT_DIR%output\formatted" "%ROOT_DIR%src\format\prompt.txt"

REM Controlla se il secondo script è stato eseguito con successo
IF ERRORLEVEL 1 (
    echo "Formatting script failed. Exiting..."
    exit /b 1
)

echo "Both scripts executed successfully."
