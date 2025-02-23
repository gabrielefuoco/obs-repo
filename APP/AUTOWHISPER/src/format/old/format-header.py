import os
import sys
import subprocess
import re
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path='api-keys.env')  

# Funzione per controllare e installare pacchetti mancanti
def check_and_install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"{package} non trovato. Installazione in corso...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", package])

# Controlla se google-generativeai è installato, altrimenti lo installa
check_and_install_package("google-generativeai")

# Ora importa il pacchetto dopo l'installazione
import google.generativeai as genai

# Configurazione dell'API utilizzando la chiave API impostata come variabile d'ambiente
genai.configure(api_key=os.getenv("Gemini"))

MAX_WORDS = 750
MIN_WORDS = 400
MAX_WORDS_2 = 500


def read_prompt(prompt_file_path, subject):
    """Leggi il prompt dal file e aggiungi l'argomento della lezione universitaria."""
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
            return prompt_text
    except FileNotFoundError:
        print(f"Prompt file non trovato: {prompt_file_path}")
        return None


def split_text_into_chunks(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """Dividi il testo in chunk rispettando i nuovi criteri."""
    words = text.split()
    chunks = []
    
    while words:
        current_chunk = words[:max_words]
        split_found = False
        
        # Cerca una newline o ## tra min_words e max_words
        for i in range(min_words - 1, len(current_chunk)):
            # Controlla se l'elemento corrente è '##' ma non '###' o più
            if current_chunk[i].startswith('##') and not current_chunk[i].startswith('###'):
                chunks.append(' '.join(current_chunk[:i+1]))
                words = words[i+1:]
                split_found = True
                break
            # Mantieni anche il controllo per newline
            elif current_chunk[i] == '\n':
                chunks.append(' '.join(current_chunk[:i+1]))
                words = words[i+1:]
                split_found = True
                break
        
        if not split_found:
            # Se non troviamo un punto di split naturale, usiamo split_text_no_newLine dall'inizio del chunk corrente
            chunk_text = ' '.join(current_chunk)
            result, remaining = split_text_no_newLine(chunk_text)
            chunks.append(result)
            words = remaining.split() + words[len(current_chunk):]
    
    return chunks

def split_text_no_newLine(text, min_words=MIN_WORDS, max_words=MAX_WORDS_2):
    """Dividi il testo in chunk rispettando i limiti min_words e max_words."""
    words = text.split()
    for i in range(min_words, min(len(words), max_words)):
        if '.' in words[i]:
            return ' '.join(words[:i+1]), ' '.join(words[i+1:])
    return ' '.join(words[:max_words]), ' '.join(words[max_words:])
    

def call_gemini_api(text_chunk, prompt, retries=3):
    """Chiama l'API Gemini 1.5 Flash con il chunk di testo e il prompt fornito."""
    model = genai.GenerativeModel("gemini-1.5-flash")

    for attempt in range(retries):
        try:
            response = model.generate_content(f"{prompt}\n\n{str(text_chunk)}", generation_config=genai.types.GenerationConfig(
                temperature=0.0,
            ),)

            if response:
                return response.text
            else:
                print("La chiamata API non ha restituito testo.")
                return None
        except Exception as e:
            if "429" in str(e):  # Controlla se è un errore 429
                wait_time = 2 ** attempt  # Raddoppia il tempo di attesa
                print(f"Quota API superata. Aspetto {wait_time} secondi prima di riprovare...")
                time.sleep(wait_time)  # Aspetta prima di riprovare
            else:
                print(f"Errore durante la chiamata API: {e}")
                return None


def process_text_file(file_path, prompt, output_folder):
    """Processa un file di testo, dividendolo in chunk e inviandoli all'API."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Suddividi il testo in chunk
    chunks = split_text_into_chunks(text)

    # Prepara il file di output
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(output_folder, f"{base_name}-processed.md")
    
    # Cancella il file di output se esiste
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("")

    print(f"Processing {file_path} into {output_file_path}")
    
    # Processa ogni chunk con l'API
    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)}...")
        response = call_gemini_api(chunk, prompt)
        if response:
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(response + "\n\n")  # Aggiunge il risultato al file di output con due new line
        else:
            print(f"Failed to process chunk {i}")


def main():
    # Configura argparse per leggere i parametri da linea di comando
    parser = argparse.ArgumentParser(description="Processa file di testo e salva il risultato.")
    parser.add_argument('input_folder', type=str, help='Cartella sorgente con i file di testo')
    parser.add_argument('output_folder', type=str, help='Cartella di destinazione per i file elaborati')
    parser.add_argument('prompt_file', type=str, help='Percorso del file contenente il prompt')
    parser.add_argument('subject', type=str, help='Argomento della lezione universitaria')

    # Parse degli argomenti
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    prompt_file_path = args.prompt_file
    subject = args.subject

    # Verifica se le cartelle esistono
    if not os.path.isdir(input_folder):
        print(f"Errore: la cartella sorgente {input_folder} non esiste.")
        return
    if not os.path.isdir(output_folder):
        print(f"Errore: la cartella di destinazione {output_folder} non esiste.")
        return

    # Leggi il prompt dal file specificato
    prompt = read_prompt(prompt_file_path, subject)
    
    if not prompt:
        print("Prompt mancante. Esco.")
        return

    # Trova tutti i file txt nella cartella sorgente
    txt_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.md') and f != os.path.basename(prompt_file_path)]
    
    if not txt_files:
        print("Nessun file di testo trovato nella cartella sorgente.")
        return

    # Processa ogni file di testo
    for txt_file in txt_files:
        file_path = os.path.join(input_folder, txt_file)
        process_text_file(file_path, prompt, output_folder)


if __name__ == "__main__":
    main()
