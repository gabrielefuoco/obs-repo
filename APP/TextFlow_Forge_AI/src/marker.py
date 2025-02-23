#!/usr/bin/env python3
import argparse
import os
import sys
import json
import glob
from multiprocessing.pool import Pool

# Importa le classi e funzioni dal package marker
from marker.converters.pdf import PdfConverter
from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

def process_single_file(input_file, output_dir, args):
    """
    Elabora un singolo file (PDF o immagine) e salva l'output nel formato richiesto.
    """
    # Carica eventuale configurazione da file JSON, se specificata
    config = {}
    if args.config_json:
        try:
            with open(args.config_json, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Errore nel caricamento del file di configurazione: {e}")
            sys.exit(1)

    # Aggiorna la configurazione con i parametri passati da CLI
    if args.output_format:
        config["output_format"] = args.output_format
    if args.paginate_output:
        config["paginate_output"] = True
    if args.disable_image_extraction:
        config["disable_image_extraction"] = True
    if args.page_range:
        config["page_range"] = args.page_range
    if args.force_ocr:
        config["force_ocr"] = True
    if args.strip_existing_ocr:
        config["strip_existing_ocr"] = True
    if args.debug:
        config["debug"] = True
    if args.languages:
        config["languages"] = args.languages
    if args.processors:
        config["processors"] = args.processors  # In questo esempio il parsing specifico è delegato al ConfigParser

    # Crea il parser di configurazione per marker
    config_parser = ConfigParser(config)

    # Determina la classe converter da utilizzare
    converter_cls = PdfConverter
    if args.converter_cls.lower() == "marker.converters.table.tableconverter":
        converter_cls = TableConverter

    # Istanzia il converter con i parametri presi dal parser di configurazione
    converter = converter_cls(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service() if args.use_llm else None,
    )

    try:
        # Esegue la conversione del file
        rendered = converter(input_file)
        # Estrae il testo (e eventualmente le immagini) dal risultato renderizzato
        text, metadata, images = text_from_rendered(rendered)
    except Exception as e:
        print(f"Errore durante la conversione del file {input_file}: {e}")
        return

    # Determina il nome del file di output (usa l'estensione basata sul formato scelto)
    base_name = os.path.basename(input_file)
    name, _ = os.path.splitext(base_name)
    ext = "md" if args.output_format == "markdown" else args.output_format
    output_file = os.path.join(output_dir, f"{name}.{ext}")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"File convertito: {input_file} -> {output_file}")
    except Exception as e:
        print(f"Errore nel salvataggio del file {output_file}: {e}")

def process_multiple_files(input_dir, output_dir, args):
    """
    Elabora tutti i file supportati (PDF e immagini) presenti nella cartella di input.
    Se viene specificato un numero di worker > 1, usa un pool di processi.
    """
    # Cerca file PDF e immagini (png, jpg, jpeg)
    patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print("Nessun file supportato trovato nella cartella di input.")
        return

    def process_file(file):
        process_single_file(file, output_dir, args)

    if args.workers and args.workers > 1:
        with Pool(args.workers) as pool:
            pool.map(process_file, files)
    else:
        for file in files:
            process_file(file)

