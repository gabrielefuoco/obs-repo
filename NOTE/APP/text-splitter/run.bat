python "%ROOT_DIR%src\main-script.py" "%ROOT_DIR%output\original" "%ROOT_DIR%output\formatted" "%ROOT_DIR%src\prompt.txt" --split_method headers

IF ERRORLEVEL 1 (
    echo "Final formatting script failed. Exiting..."
    exit /b 1
)
echo "All scripts executed successfully."
exit /b 0