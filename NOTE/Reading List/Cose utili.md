## Convertire ipynb in markdown

``` bash
@echo off
setlocal enabledelayedexpansion

REM Imposta la cartella contenente i file .ipynb (usa . per la cartella corrente)
set "source_folder=."

REM Itera attraverso tutti i file .ipynb nella cartella
for %%f in ("%source_folder%\*.ipynb") do (
    echo Convertendo "%%f" in Markdown...
    jupyter nbconvert --to markdown "%%f"
)

echo Conversione completata.
pause
```

## Rimozione newline

``` python
import os
import re

def remove_extra_newlines(content):
    # Sostituisce due o più newline consecutivi con un solo newline

    return re.sub(r'\n{2,}', '\n\n', content)

def process_markdown_files(folder_path):
    # Itera su tutti i file nella cartella

    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(folder_path, filename)
            print(f"Elaborazione del file: {filename}")

            # Legge il contenuto del file

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Rimuove i newline ridondanti

            new_content = remove_extra_newlines(content)

            # Sovrascrive il file con il nuovo contenuto

            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)

    print("Elaborazione completata.")

# Imposta la cartella contenente i file Markdown

folder_path = "./"  # Usa "./" per la cartella corrente

# Esegue lo script

process_markdown_files(folder_path)
```