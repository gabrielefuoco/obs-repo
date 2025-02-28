import os
import re
import sys

def normalize_whitespace_outside_formulas(text):
    """
    Normalizza gli spazi multipli in una stringa, applicando la sostituzione
    solo alle parti che NON sono formule (delimitate da $...$).
    """
    segments = re.split(r'(\$.*?\$)', text)
    for idx, seg in enumerate(segments):
        if not (seg.startswith("$") and seg.endswith("$")):
            seg = re.sub(r' {2,}', ' ', seg)
            segments[idx] = seg
    return "".join(segments)

def replace_underscore_with_asterisks(text):
    """
    Sostituisce l'uso di _frase_ con *frase* nelle parti che non sono formule.
    """
    segments = re.split(r'(\$.*?\$)', text)
    for idx, seg in enumerate(segments):
        if not (seg.startswith("$") and seg.endswith("$")):
            seg = re.sub(r'(?<!\S)_([^_]+)_(?!\S)', r'*\1*', seg)
            segments[idx] = seg
    return "".join(segments)

def process_markdown_file(file_path):
    """
    Legge il file Markdown e applica le seguenti trasformazioni, evitando di intervenire:
      - All'interno di blocchi di codice (delimitati da tripli backtick)
      - All'interno delle formule (delimitate da $)
      
    Le modifiche applicate sono:
      1. Riduzione di righe vuote multiple a una singola riga vuota.
      2. Aggiunta di una riga vuota dopo ogni header (linee che iniziano con '#' seguito da spazio).
      3. Normalizzazione degli spazi bianchi fuori dalle formule.
      4. Inserimento di una riga vuota dopo una riga contenente solo una frase in grassetto che termina con i due punti,
         gestendo la posizione dei due punti (sia all'interno che subito dopo il grassetto).
      5. Uniformazione della sintassi: _frase_ viene sostituito con *frase*.
      6. Se una riga contiene solo una formula espressa come $formula$, la trasforma in $$formula$$.
      7. Trasformazione delle righe che sono esclusivamente in grassetto in header h5,
         ovvero, **frase** diventa "##### frase". Se la riga contiene altro al di fuori del grassetto,
         essa non viene modificata.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    in_code_block = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.rstrip('\n')

        # Gestione dei blocchi di codice: inizio/fine blocco delimitato da ```
        if re.match(r'^\s*```', stripped_line):
            processed_lines.append(line)
            in_code_block = not in_code_block
            i += 1
            continue

        if in_code_block:
            processed_lines.append(line)
            i += 1
            continue

        # Applicazione delle trasformazioni fuori dai blocchi di codice:
        # 1. Normalizzazione degli spazi bianchi (escludendo le formule)
        modified_line = normalize_whitespace_outside_formulas(stripped_line)
        # 2. Sostituzione di _frase_ con *frase* (escludendo le formule)
        modified_line = replace_underscore_with_asterisks(modified_line)
        # 3. Se la riga contiene solo una formula espressa con un singolo paio di $,
        #    trasformala in $$formula$$ (solo se non è già stata trasformata)
        if not (modified_line.strip().startswith('$$') and modified_line.strip().endswith('$$')):
            formula_match = re.match(r'^\s*\$(.+?)\$\s*$', modified_line)
            if formula_match:
                formula_content = formula_match.group(1)
                modified_line = f'$${formula_content}$$'
        # 4. Se la riga è interamente in grassetto (e non contiene altro),
        #    trasformala in header h5 (rimuovendo i delimitatori **)
        h5_match = re.match(r'^\s*\*\*(.+?)\*\*\s*$', modified_line)
        if h5_match:
            content = h5_match.group(1)
            modified_line = f'##### {content}'

        # Ricostruisce la riga con newline
        modified_line_with_newline = modified_line + "\n"
        processed_lines.append(modified_line_with_newline)

        # Aggiungi una riga vuota dopo gli header (cioè, se la riga modificata inizia con '#' e uno spazio)
        if re.match(r'^\s*#+\s', modified_line):
            if i + 1 < len(lines):
                next_line = lines[i+1]
                if next_line.strip() != "":
                    processed_lines.append("\n")
            else:
                processed_lines.append("\n")

        # Aggiungi una riga vuota dopo una riga contenente solo una frase in grassetto che termina con i due punti.
        # Gestisce entrambe le varianti: **frase:** oppure **frase**:
        if (re.match(r'^\s*\*\*.+:\*\*\s*$', modified_line) or
            re.match(r'^\s*\*\*.+\*\*:\s*$', modified_line)):
            if i + 1 < len(lines):
                next_line = lines[i+1]
                if next_line.strip() != "":
                    processed_lines.append("\n")
            else:
                processed_lines.append("\n")
        i += 1

    # Riduci eventuali sequenze di righe vuote multiple a una singola riga vuota
    final_lines = []
    previous_empty = False
    for line in processed_lines:
        if line.strip() == "":
            if not previous_empty:
                final_lines.append("\n")
                previous_empty = True
        else:
            final_lines.append(line)
            previous_empty = False

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)

def process_directory(root_dir):
    """
    Itera ricorsivamente su tutte le sottocartelle a partire da root_dir e
    processa ogni file con estensione .md.
    """
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(subdir, file)
                print(f"Processing: {file_path}")
                process_markdown_file(file_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <root_directory>")
        sys.exit(1)
    root_directory = sys.argv[1]
    process_directory(root_directory)
