

REM Esegui lo script format.py
python "%ROOT_DIR%src\format\trascrizione.py" "%ROOT_DIR%output\original" "%ROOT_DIR%output\original" "%ROOT_DIR%src\format\trascrivi.txt"

REM Controlla se il secondo script è stato eseguito con successo
IF ERRORLEVEL 1 (
    echo "Formatting script failed. Exiting..."
    exit /b 1
)

echo "Both scripts executed successfully."
