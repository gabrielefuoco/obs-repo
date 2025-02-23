@echo off
REM Imposta la cartella di lavoro alla posizione attuale
SET ROOT_DIR=%~dp0

REM Chiedi all'utente quale operazione eseguire
echo Seleziona quale attività eseguire:
echo 1 - Tutte le attività
echo 2 - Solo trascrizione
echo 3 - Solo miglioramento della trascrizione
echo 4 - Solo formattazione finale
echo 5 - Pulizia delle cartelle
echo 6 - Esporta in Obsidian
set /p task_choice="Inserisci il numero dell'operazione da eseguire: "

REM Verifica se il valore inserito è tra 1 e 4, in tal caso chiedi l'argomento della lezione
if "%task_choice%"=="1" goto :set_args
if "%task_choice%"=="2" goto :set_args
if "%task_choice%"=="3" goto :set_args
if "%task_choice%"=="4" goto :set_args
if "%task_choice%"=="5" goto :clean
if "%task_choice%"=="6" goto :export

REM Se l'utente inserisce un valore non valido, termina
echo Scelta non valida. Uscita dallo script...
exit /b 1

:set_args
REM Chiedi all'utente di inserire l'argomento della lezione
set /p lesson_topic="Inserisci l'argomento della lezione: "

REM Rimanda all'attività scelta in base al valore di task_choice
if "%task_choice%"=="1" goto :all_tasks
if "%task_choice%"=="2" goto :transcription_only
if "%task_choice%"=="3" goto :format_transcription_only
if "%task_choice%"=="4" goto :format_final_only

:clean
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
exit /b 0

:export
@echo off
REM Definisci le directory di origine e destinazione
set ROOT_DIR=%~dp0
set SOURCE_DIR=%ROOT_DIR%output\formatted
set DEST_DIR="C:\Users\gabri\Documents\Obsidian Vault\Repo\APPPUNTI\Imported"

REM Controlla se la cartella di origine esiste
if not exist "%SOURCE_DIR%" (
    echo La cartella di origine non esiste: %SOURCE_DIR%
    exit /b 1
)

REM Controlla se la cartella di destinazione esiste, altrimenti la crea
if not exist %DEST_DIR% (
    echo La cartella di destinazione non esiste, la sto creando: %DEST_DIR%
    mkdir %DEST_DIR%
)

REM Copia i file dalla cartella di origine a quella di destinazione
echo Copiando i file da %SOURCE_DIR% a %DEST_DIR%...
xcopy /s /y "%SOURCE_DIR%\*" %DEST_DIR%

echo Operazione completata!
exit /b 0

:all_tasks
REM Esegui tutte le attività in ordine
goto :transcription_only

:transcription_only
REM Esegui lo script transcribe.py e passa l'argomento
python "%ROOT_DIR%src\transcribe\transcribe.py" "%ROOT_DIR%audio-file" "%ROOT_DIR%output\original" "%lesson_topic%"

REM Controlla se il primo script è stato eseguito con successo
IF ERRORLEVEL 1 (
    echo "Transcription script failed. Exiting..."
    exit /b 1
)

if "%task_choice%"=="2" (
    echo "Transcription completed successfully."
    exit /b 0
)

goto :format_transcription_only

:format_transcription_only
REM Esegui lo script trascrizione.py e passa l'argomento
python "%ROOT_DIR%src\format\trascrizione.py" "%ROOT_DIR%output\original" "%ROOT_DIR%output\original" "%ROOT_DIR%src\format\trascrivi.txt" "%lesson_topic%"

REM Controlla se il secondo script è stato eseguito con successo
IF ERRORLEVEL 1 (
    echo "Formatting script failed. Exiting..."
    exit /b 1
)

if "%task_choice%"=="3" (
    echo "Transcription formatting completed successfully."
    exit /b 0
)

goto :format_final_only

:format_final_only
REM Esegui lo script format.py e passa l'argomento con il metodo di split predefinito (headers)
python "%ROOT_DIR%src\format\main-script.py" "%ROOT_DIR%output\original" "%ROOT_DIR%output\formatted" "%ROOT_DIR%src\format\prompt.txt" "%lesson_topic%" --split_method headers

REM Controlla se il terzo script è stato eseguito con successo
IF ERRORLEVEL 1 (
    echo "Final formatting script failed. Exiting..."
    exit /b 1
)
echo "All scripts executed successfully."
exit /b 0