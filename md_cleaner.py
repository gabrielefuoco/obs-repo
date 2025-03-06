import os
import re
import sys
import shutil
import tempfile

# Precompile regular expressions for performance
FORMULA_PATTERN = re.compile(r'(?<!\\)(\$.*?(?<!\\)\$)')
MULTISPACE_PATTERN = re.compile(r' {2,}')
UNDERSCORE_PATTERN = re.compile(r'(?<!\S)_([^_]+?)_(?!\S)')
CODE_BLOCK_PATTERN = re.compile(r'^\s*```')
HEADER_PATTERN = re.compile(r'^\s*#+\s')
SINGLE_FORMULA_PATTERN = re.compile(r'^\s*\$([^$]+)\$\s*$')
BOLD_LINE_PATTERN = re.compile(r'^\s*\*\*(.+?)\*\*\s*$')
BOLD_COLON_PATTERN1 = re.compile(r'^\s*\*\*.+:\*\*\s*$')
BOLD_COLON_PATTERN2 = re.compile(r'^\s*\*\*.+\*\*:\s*$')
# Pattern per correggere il tag immagine errato
WRONG_IMAGE_TAG_PATTERN = re.compile(r'!\[\[\|(?P<caption>.*?)\]\((?P<filename>.*?)\)')

def process_segments(text, formula_pattern, transformations):
    segments = formula_pattern.split(text)
    for i in range(len(segments)):
        if i % 2 == 0:  # Indici pari: segmenti non-formula
            for pattern, replacement in transformations:
                segments[i] = pattern.sub(replacement, segments[i])
    return ''.join(segments)

def normalize_and_replace(text):
    transformations = [
        (MULTISPACE_PATTERN, ' '),
        (UNDERSCORE_PATTERN, r'*\1*')
    ]
    return process_segments(text, FORMULA_PATTERN, transformations)

def process_formula_line(line):
    line = line.strip()
    if not (line.startswith('$$') and line.endswith('$$')):
        match = SINGLE_FORMULA_PATTERN.match(line)
        if match:
            return f'$${match.group(1)}$$'
    return line

def process_bold_line(line):
    match = BOLD_LINE_PATTERN.match(line)
    if match:
        return f'##### {match.group(1)}'
    return line

def transform_h1_to_h2(line):
    """
    Trasforma gli header h1 (riga che inizia con un solo '#' seguito da spazio)
    in header h2 aggiungendo un ulteriore '#' all'inizio.
    """
    return re.sub(r'^(\s*)#(\s+)', r'\1##\2', line)

def correct_image_tag(line):
    """
    Corregge i tag delle immagini errati.
    Esempio:
      Tag errato: ![[|302](_page_8_Figure_6.jpeg)
      Tag corretto: ![[_page_8_Figure_6.jpeg|302]]
    """
    return WRONG_IMAGE_TAG_PATTERN.sub(lambda m: f'![[{m.group("filename")}|{m.group("caption")}]]', line)

def normalize_leading_enumeration(lines):
    """
    Normalizza le enumerazioni all'inizio di una riga.
    Riconosce numerazioni composte da 1 a 3 gruppi di 1 o 2 cifre separati da punti,
    con un eventuale punto finale opzionale, e rimuove eventuali spazi dopo i punti.
    Se il pattern non rispetta queste regole, la riga viene lasciata inalterata.
    
    Esempio:
      * **1.1. Mapper:**  ->  * **1.1 Mapper:**
    """
    numbering_pattern = re.compile(
        r'^(?P<prefix>[\*\#\s]*(?:\*\*)?)'
        r'(?P<number>\d{1,2}(?:\.\s*\d{1,2}(?:\.\s*\d{1,2})?)?)'
        r'(?:\.)?(?P<suffix>\s+.*)'
    )
    new_lines = []
    in_code_block = False
    for line in lines:
        if CODE_BLOCK_PATTERN.match(line):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
        if in_code_block:
            new_lines.append(line)
            continue
        match = numbering_pattern.match(line)
        if match:
            prefix = match.group('prefix')
            num = match.group('number')
            suffix = match.group('suffix')
            # Suddivide e unisce i gruppi, rimuovendo eventuali spazi attorno ai punti
            parts = re.split(r'\.\s*', num)
            # Verifica che ci siano al massimo 3 gruppi
            if 1 <= len(parts) <= 3:
                normalized_num = '.'.join(parts)
                new_line = f"{prefix}{normalized_num}{suffix}"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    return new_lines

def process_code_blocks(lines):
    processed = []
    in_code = False
    for line in lines:
        stripped = line.rstrip('\n')
        if CODE_BLOCK_PATTERN.match(stripped):
            in_code = not in_code
            processed.append(line)
            continue
        processed.append(line if in_code else None)
    return processed

def add_newlines_after_headers(lines):
    processed = []
    for i, line in enumerate(lines):
        processed.append(line)
        stripped = line.rstrip('\n')
        if HEADER_PATTERN.match(stripped):
            if i + 1 < len(lines) and lines[i + 1].strip() != '':
                processed.append('\n')
    return processed

def normalize_tabulations(lines):
    """
    Normalizza l'indentazione delle liste puntate Markdown.
    - Le righe di bullet header (es. "* **Titolo:**") vengono forzate a non avere indentazione.
    - Le righe bullet che seguono immediatamente un header vengono indentate con una tabulazione.
    - Le righe all'interno di blocchi di codice (``` ... ```) non vengono modificate.
    """
    new_lines = []
    in_code_block = False
    bullet_pattern = re.compile(r'^(\s*)\* (.*)$')
    header_bullet_pattern = re.compile(r'^\*\*.*\*\*$')
    
    expecting_child = False
    for line in lines:
        if CODE_BLOCK_PATTERN.match(line):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
        if in_code_block:
            new_lines.append(line)
            continue

        if line.strip() == "":
            new_lines.append(line)
            expecting_child = False
            continue

        m = bullet_pattern.match(line)
        if m:
            indent, content = m.group(1), m.group(2)
            content_stripped = content.strip()
            # Se il contenuto è un header bullet (testo in grassetto interamente),
            # forziamo l'assenza di indentazione e attiviamo il flag per i figli
            if header_bullet_pattern.match(content_stripped):
                new_line = "* " + content_stripped + "\n"
                new_lines.append(new_line)
                expecting_child = True
            else:
                if expecting_child:
                    new_line = "\t* " + content_stripped + "\n"
                    new_lines.append(new_line)
                else:
                    new_line = "* " + content_stripped + "\n"
                    new_lines.append(new_line)
        else:
            new_lines.append(line)
            expecting_child = False

    return new_lines

def normalize_numbered_lists(lines):
    """
    Trasforma tutti gli elenchi numerici in elenchi puntati.
    Ad esempio:
        1. Elemento -> - Elemento
    L'operazione viene applicata solo alle righe fuori dai blocchi di codice.
    """
    new_lines = []
    in_code_block = False
    numbered_pattern = re.compile(r'^(\s*)\d{1,2}\.\s+(.*)$')
    for line in lines:
        if CODE_BLOCK_PATTERN.match(line):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
        if in_code_block:
            new_lines.append(line)
            continue
        m = numbered_pattern.match(line)
        if m:
            indent = m.group(1)
            rest = m.group(2)
            new_line = f"{indent}- {rest}\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return new_lines

def remove_only_dashes(lines):
    """
    Rimuove tutte le righe che contengono esclusivamente '---' (eventualmente circondate da spazi).
    """
    new_lines = []
    dash_pattern = re.compile(r'^\s*---\s*$')
    for line in lines:
        if dash_pattern.match(line):
            continue
        new_lines.append(line)
    return new_lines

def process_markdown_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error reading {file_path}: {e}")
        return

    processed_lines = []
    in_code_block = False

    for line in lines:
        stripped_line = line.rstrip('\n')
        if CODE_BLOCK_PATTERN.match(stripped_line):
            in_code_block = not in_code_block
            processed_lines.append(line)
            continue

        if in_code_block:
            processed_lines.append(line)
            continue

        modified_line = normalize_and_replace(stripped_line)
        modified_line = process_formula_line(modified_line)
        modified_line = process_bold_line(modified_line)
        modified_line = transform_h1_to_h2(modified_line)   # Trasforma gli header h1 in h2
        modified_line = correct_image_tag(modified_line)    # Corregge i tag delle immagini errati
        processed_lines.append(modified_line + '\n')

    # Applica le trasformazioni
    processed_lines = normalize_tabulations(processed_lines)
    processed_lines = normalize_numbered_lists(processed_lines)
    processed_lines = normalize_leading_enumeration(processed_lines)
    processed_lines = add_newlines_after_headers(processed_lines)
    processed_lines = remove_only_dashes(processed_lines)
    
    # Rimuove righe vuote consecutive
    final_lines = []
    prev_empty = False
    for line in processed_lines:
        curr_empty = (line.strip() == '')
        if not (prev_empty and curr_empty):
            final_lines.append(line)
        prev_empty = curr_empty

    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.writelines(final_lines)
        shutil.move(tmp.name, file_path)
    except IOError as e:
        print(f"Error writing {file_path}: {e}")
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

def process_directory(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                process_markdown_file(file_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <root_directory>")
        sys.exit(1)
    process_directory(sys.argv[1])
