@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { 
    $env:GOOGLE_API_KEY = 'AIzaSyCXSO-sdZOixiwf7HwcDN0n-m1hoSPSw7Y'; 
    New-Item -ItemType Directory -Force -Path .\output | Out-Null; 
    marker \"$PWD\input\" --output_dir \"$PWD\output\" --output_format markdown --use_llm --force_ocr --languages it --workers 4 2> error.log; 
    Read-Host -Prompt 'Premi un tasto per uscire...' 
}"
