@echo off
REM Imposta la cartella di lavoro alla posizione attuale
SET ROOT_DIR=%~dp0

REM Elimina file .mp3 dalla cartella audio-file
echo Eliminazione file .mp3 dalla cartella audio-file...
del /q "%ROOT_DIR%audio-file\*.mp3"

REM Elimina file .mp3 dalla cartella tmp
echo Eliminazione file .mp3 dalla cartella tmp...
del /q "%ROOT_DIR%tmp\*.mp3"

REM Elimina file .txt dalla cartella output/original
echo Eliminazione file .txt dalla cartella output/original...
del /q "%ROOT_DIR%output\original\*.txt"

REM Elimina file .md dalla cartella output/formatted
echo Eliminazione file .md dalla cartella output/formatted...
del /q "%ROOT_DIR%output\formatted\*.md"

REM Controlla se l'operazione è riuscita
IF ERRORLEVEL 1 (
    echo "Error deleting files. Exiting..."
    exit /b 1
)

echo "File eliminati con successo."
