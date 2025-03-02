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

def process_segments(text, formula_pattern, transformations):
    segments = formula_pattern.split(text)
    for i in range(len(segments)):
        if i % 2 == 0:  # Even indices are non-formula segments
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
        processed_lines.append(modified_line + '\n')

    # Apply post-processing
    processed_lines = add_newlines_after_headers(processed_lines)
    
    # Remove consecutive empty lines
    final_lines = []
    prev_empty = False
    for line in processed_lines:
        curr_empty = line.strip() == ''
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