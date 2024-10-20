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
