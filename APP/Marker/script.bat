@echo off
setlocal

REM Crea la cartella output se non esiste
if not exist output mkdir output

REM Processa ogni file nella cartella input
for %%F in (input\*) do (
    echo Processing %%F...
    powershell -Command "$env:GOOGLE_API_KEY='AIzaSyCXSO-sdZOixiwf7HwcDN0n-m1hoSPSw7Y'; marker_single '%%F' --output_dir '%CD%\output' --output_format markdown --languages it"
)

echo Done!
endlocal
