# main.py
import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Dict, List

import google.generativeai as genai
from splitcriteria import MarkdownSplitter, NaturalTextSplitter
from dotenv import load_dotenv

# Costanti
MIN_WORDS = 400
MAX_WORDS = 750

# Configura il logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Carica le variabili d'ambiente da api-keys.env
dotenv_path = Path("api-keys.env")
if not dotenv_path.exists():
    logging.error("File api-keys.env non trovato")
    sys.exit(1)
load_dotenv(dotenv_path=dotenv_path)

# Verifica la presenza della chiave API
gemini_key = os.getenv("Gemini")
if not gemini_key:
    logging.error("Chiave API Gemini non trovata nel file .env")
    sys.exit(1)

# Configura google-generativeai
genai.configure(api_key=gemini_key)


def check_and_install_package(package: str) -> None:
    """Verifica se un pacchetto è installato e, in caso contrario, lo installa una volta sola."""
    try:
        __import__(package)
    except ImportError:
        logging.info(f"{package} non trovato. Installazione in corso...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", package])
        except subprocess.CalledProcessError as e:
            logging.error(f"Errore durante l'installazione di {package}: {e}")
            sys.exit(1)


check_and_install_package("google-generativeai")
check_and_install_package("tkinterdnd2")


def read_prompts(prompt_folder: Path) -> OrderedDict:
    """Legge tutti i file di prompt dalla cartella specificata."""
    prompts = OrderedDict()
    if not prompt_folder.is_dir():
        logging.error(f"Cartella dei prompt non trovata: {prompt_folder}")
        return prompts

    prompt_files = sorted(prompt_folder.glob("*.md")) + sorted(prompt_folder.glob("*.txt"))
    for prompt_file in prompt_files:
        try:
            content = prompt_file.read_text(encoding="utf-8").strip()
            prompts[prompt_file.name] = content
        except Exception as e:
            logging.warning(f"Errore nella lettura del file {prompt_file}: {e}")
    return prompts


def call_gemini_api(text_chunk: str, prompt: str, retries: int = 3) -> Optional[str]:
    """Chiama l'API di Gemini 1.5 Flash con un chunk di testo e un prompt."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    for attempt in range(1, retries + 1):
        try:
            response = model.generate_content(
                f"{prompt}\n\n{text_chunk}",
                generation_config=genai.types.GenerationConfig(temperature=0.2)
            )
            if response and response.text:
                return response.text
            else:
                logging.warning("La chiamata API non ha restituito testo.")
                return None
        except Exception as e:
            if "429" in str(e):
                wait_time = 2 ** attempt
                logging.warning(f"Quota API superata. Attesa di {wait_time} secondi (tentativo {attempt}/{retries})...")
                time.sleep(wait_time)
            else:
                logging.error(f"Errore durante la chiamata API: {e}")
                return None
    logging.error("Numero massimo di tentativi raggiunto senza successo.")
    return None


def process_text_file(file_path: Path, prompts: OrderedDict, output_folder: Path, split_method: str,
                      order_mode: str, output_mode: str) -> None:
    """
    Elabora un file di testo:
      - Suddivide il testo in chunk tramite lo splitter scelto.
      - Per ogni chunk e per ogni prompt applica l'API Gemini.
      - Salva l'output in un file Markdown.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Errore nella lettura di {file_path}: {e}")
        return

    splitter = (MarkdownSplitter(min_words=MIN_WORDS, max_words=MAX_WORDS)
                if split_method == "headers" else NaturalTextSplitter(min_words=MIN_WORDS, max_words=MAX_WORDS))
    chunks = splitter.split(text)
    logging.info(f"Il file {file_path.name} è stato diviso in {len(chunks)} chunk.")

    results: Dict[tuple, str] = {}
    for chunk_idx, chunk in enumerate(chunks, start=1):
        for prompt_name, prompt_text in prompts.items():
            logging.info(f"Processo chunk {chunk_idx}/{len(chunks)} con prompt '{prompt_name}'")
            response = call_gemini_api(chunk, prompt_text) or "[nessuna risposta dall'API]"
            results[(chunk_idx, prompt_name)] = response

    base_name = file_path.stem
    output_file = output_folder / f"{base_name}-processed.md"

    # Costruzione dell'output (esempio per modalità 'single')
    lines = [f"# Output processing per: {base_name}\n"]
    if output_mode == "single":
        if order_mode == "chunk":
            for chunk_idx in range(1, len(chunks) + 1):
                lines.append(f"## Chunk {chunk_idx}\n")
                for prompt_name in prompts.keys():
                    lines.append(f"### {prompt_name}\n")
                    lines.append(results[(chunk_idx, prompt_name)] + "\n")
        elif order_mode == "prompt":
            for prompt_name in prompts.keys():
                lines.append(f"## {prompt_name}\n")
                for chunk_idx in range(1, len(chunks) + 1):
                    lines.append(f"### Chunk {chunk_idx}\n")
                    lines.append(results[(chunk_idx, prompt_name)] + "\n")
    else:
        # Altre modalità (per_chunk, per_prompt, per_pair) possono essere implementate se necessario
        lines.append("Modalità di output non supportata al momento.\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"File elaborato salvato in {output_file}")


def ensure_folder(folder: Path) -> None:
    """Crea la cartella se non esiste."""
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)


def process_all_files(input_files: List[str], output_folder: str, selected_prompts: List[str],
                      prompt_folder: str, split_method: str, order_mode: str, output_mode: str) -> None:
    """Esegue l'elaborazione sui file selezionati dalla GUI."""
    output_folder_path = Path(output_folder)
    prompt_folder_path = Path(prompt_folder)
    ensure_folder(output_folder_path)
    prompts = read_prompts(prompt_folder_path)
    # Costruisce un dizionario dei prompt selezionati
    selected_prompts_dict = {p: prompts[p] for p in selected_prompts if p in prompts}
    for file in input_files:
        process_text_file(Path(file), selected_prompts_dict, output_folder_path, split_method, order_mode, output_mode)


if __name__ == "__main__":
    logging.error("Questo script deve essere eseguito tramite la GUI (gui.py)")
