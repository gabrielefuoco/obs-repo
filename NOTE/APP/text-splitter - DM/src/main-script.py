import os
import sys
import subprocess
import re
import argparse
import time
from dotenv import load_dotenv

# Costanti
MIN_WORDS = 400
MAX_WORDS = 750
MAX_WORDS_2 = 500

# Carica variabili d'ambiente
load_dotenv(dotenv_path='api-keys.env')

# Verifica presenza della chiave API
if not os.getenv("Gemini"):
    raise ValueError("Chiave API Gemini non trovata nel file .env")

# Installazione e configurazione di google-generativeai


import google.generativeai as genai
genai.configure(api_key=os.getenv("Gemini"))


def split_text_by_headers(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """Split text into chunks using ## headers as primary split points."""
    words = text.split()
    chunks = []
    
    while words:
        current_chunk = words[:max_words]
        split_found = False
        
        # Look for ## headers or newlines between min_words and max_words
        for i in range(min_words - 1, len(current_chunk)):
            # Check if current word is '##' but not '###' or more
            if current_chunk[i].startswith('@@@') and not current_chunk[i].startswith('@@@@'):
                chunks.append(' '.join(current_chunk[:i+1]))
                words = words[i+1:]
                split_found = True
                break
            elif current_chunk[i] == '\n':
                chunks.append(' '.join(current_chunk[:i+1]))
                words = words[i+1:]
                split_found = True
                break
        
        if not split_found:
            # If no natural split point found, use sentence-based split
            chunk_text = ' '.join(current_chunk)
            result, remaining = split_by_sentence(chunk_text)
            chunks.append(result)
            words = remaining.split() + words[len(current_chunk):]
    
    return chunks

def split_text_natural(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """Split text into chunks using any headers or newlines as split points."""
    words = text.split()
    chunks = []
    
    while words:
        current_chunk = words[:max_words]
        split_found = False
        
        # Look for any headers or newlines between min_words and max_words
        for i in range(min_words - 1, len(current_chunk)):
            if current_chunk[i] == '\n' or current_chunk[i].startswith('#'):
                chunks.append(' '.join(current_chunk[:i+1]))
                words = words[i+1:]
                split_found = True
                break
        
        if not split_found:
            # If no natural split point found, use sentence-based split
            chunk_text = ' '.join(current_chunk)
            result, remaining = split_by_sentence(chunk_text)
            chunks.append(result)
            words = remaining.split() + words[len(current_chunk):]
    
    return chunks

def split_by_sentence(text, min_words=MIN_WORDS, max_words=MAX_WORDS_2):
    """Split text into chunks by sentence endings."""
    words = text.split()
    for i in range(min_words, min(len(words), max_words)):
        if '.' in words[i]:
            return ' '.join(words[:i+1]), ' '.join(words[i+1:])
    return ' '.join(words[:max_words]), ' '.join(words[max_words:])



def check_and_install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"{package} non trovato. Installazione in corso...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", package])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Errore durante l'installazione di {package}: {e}")

check_and_install_package("google-generativeai")

def read_prompt(prompt_file_path):

    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
            return prompt_text
    except FileNotFoundError:
        print(f"Prompt file non trovato: {prompt_file_path}")
        return None

def call_gemini_api(text_chunk, prompt, retries=3):
    """Call Gemini 1.5 Flash API with text chunk and provided prompt."""
    model = genai.GenerativeModel("gemini-1.5-flash")

    for attempt in range(retries):
        try:
            response = model.generate_content(
                f"{prompt}\n\n{str(text_chunk)}", 
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )

            if response:
                return f"{text_chunk}\n\n{response.text}\n\n"
            else:
                print("La chiamata API non ha restituito testo.")
                return None
        except Exception as e:
            if "429" in str(e):
                wait_time = 2 ** attempt
                print(f"Quota API superata. Aspetto {wait_time} secondi prima di riprovare...")
                time.sleep(wait_time)
            else:
                print(f"Errore durante la chiamata API: {e}")
                return None

def process_text_file(file_path, prompt, output_folder, split_method='headers'):
    """Process a text file by splitting it into chunks and sending them to the API."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Choose splitting method based on parameter
    if split_method == 'headers':
        chunks = split_text_by_headers(text)
    else:
        chunks = split_text_natural(text)

    # Prepare output file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(output_folder, f"{base_name}-processed.md")
    
    # Clear output file if it exists
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("")

    print(f"Processing {file_path} into {output_file_path}")
    
    # Process each chunk with the API
    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)}...")
        response = call_gemini_api(chunk, prompt)
        if response:
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(response + "\n\n")
        else:
            print(f"Failed to process chunk {i}")

def main():
    parser = argparse.ArgumentParser(description="Process text files and save results.")
    parser.add_argument('input_folder', type=str, help='Source folder with text files')
    parser.add_argument('output_folder', type=str, help='Destination folder for processed files')
    parser.add_argument('prompt_file', type=str, help='Path to prompt file')
    parser.add_argument('--split_method', type=str, choices=['headers', 'natural'], 
                       default='headers', help='Text splitting method to use')

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: source folder {args.input_folder} does not exist.")
        return
    if not os.path.isdir(args.output_folder):
        print(f"Error: destination folder {args.output_folder} does not exist.")
        return

    prompt = read_prompt(args.prompt_file)
    
    if not prompt:
        print("Missing prompt. Exiting.")
        return

    txt_files = [f for f in os.listdir(args.input_folder) 
                if (f.lower().endswith('.md')or f.lower().endswith('.txt'))  and f != os.path.basename(args.prompt_file)]
    
    if not txt_files:
        print("No text files found in source folder.")
        return

    for txt_file in txt_files:
        file_path = os.path.join(args.input_folder, txt_file)
        process_text_file(file_path, prompt, args.output_folder, args.split_method)

if __name__ == "__main__":
    main()
