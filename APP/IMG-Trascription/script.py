import os
import sys
import base64
import time
from google.generativeai import GenerativeModel  
import google.generativeai as genai  
from groq import Groq
from dotenv import load_dotenv
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import platform
import fitz  
import logging
import re


def extract_number(filename):
    # Estrae il primo numero trovato nel nome del file
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')




def set_tesseract_path():
    """Imposta il percorso corretto di Tesseract in base al sistema operativo."""
    system = platform.system().lower()
    
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Verifica se il percorso esiste
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return True
    
    # Se il percorso predefinito non esiste, cerca tesseract nel PATH
    from shutil import which
    system_tesseract = which('tesseract')
    if system_tesseract:
        pytesseract.pytesseract.tesseract_cmd = system_tesseract
        return True
    
    return False

def enhance_image(image):
    """Applica tecniche di miglioramento dell'immagine per l'OCR."""
    # Migliora il contrasto
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(2)
    
    # Applica filtro di nitidezza
    image_filtered = image_enhanced.filter(ImageFilter.SHARPEN)
    
    # Converti in scala di grigi
    image_gray = image_filtered.convert('L')
    
    # Converti in array NumPy per il thresholding adattivo
    image_cv = np.array(image_gray)
    
    # Applica thresholding adattivo
    image_threshold = cv2.adaptiveThreshold(
        image_cv, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return Image.fromarray(image_threshold)

def perform_ocr(image_path):
    """Esegue OCR sull'immagine con varie tecniche di preprocessing."""
    # Carica l'immagine
    image = Image.open(image_path)
    
    # Lista per salvare i risultati di diversi metodi OCR
    ocr_results = []
    
    try:
        # OCR sull'immagine originale
       # original_text = pytesseract.image_to_string(image)
        #if original_text.strip():
          #  ocr_results.append(original_text)
        
        # OCR sull'immagine migliorata
        enhanced_image = enhance_image(image)
        enhanced_text = pytesseract.image_to_string(enhanced_image)
        if enhanced_text.strip():
            ocr_results.append(enhanced_text)
        
        # Segmentazione dell'immagine (dividi in due metà)
        left_half = image.crop((0, 0, image.width // 2, image.height))
        right_half = image.crop((image.width // 2, 0, image.width, image.height))
        
        # OCR sulle metà
        #left_text = pytesseract.image_to_string(enhance_image(left_half))
        #right_text = pytesseract.image_to_string(enhance_image(right_half))
        
       # if left_text.strip() or right_text.strip():
        #    ocr_results.append(left_text + "\n" + right_text)
        
    except Exception as e:
        print(f"Errore durante l'OCR: {e}")
        return "Errore nell'estrazione del testo"
    
    # Combina i risultati, rimuovi duplicati e spazi vuoti
    combined_text = "\n".join(set(filter(None, ocr_results)))
    return combined_text.strip() if combined_text.strip() else "Nessun testo rilevato"

def encode_image(image_path):
    """Codifica un'immagine in base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        print(f"Errore durante la lettura dell'immagine {image_path}: {e}")
        raise


def set_gemini():
    # Carica variabili ambiente
    load_dotenv()

    # Configura Gemini API
    api_key = "AIzaSyC0PSSPPADlvq41xB3lM2TF6H0FllUlO1M"
    if not api_key:
        print("Errore: API key di Google non configurata correttamente nel file .env")
        return

    # Inizializza Gemini
    genai.configure(api_key=api_key)

def call_gemini(enhanced_prompt,image_path,model_name="gemini-exp-1121"):
    model = GenerativeModel(model_name)
    image = Image.open(image_path)
    return model.generate_content([enhanced_prompt, image]).text
                

def set_groq():
    load_dotenv()

    # Ottieni API key
    api_key = "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"
    if not api_key or api_key == "your_api_key_here":
        print("Errore: API key non configurata correttamente nel file .env")
        return
    # Inizializza il client Groq
    return Groq(api_key=api_key)


def call_groq(client,enhanced_prompt,image_path, model_name="llama-3.2-90b-vision-preview"):

    base64_image = encode_image(image_path)
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": enhanced_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model=model_name,
    temperature=0.2
    )
    return chat_completion.choices[0].message.content

def process_images(image_dir, prompt_file, output_dir, retries=3, model="gemini"):
    
    client=None
    if model=="gemini":
        set_gemini()
    elif model=="groq":
        client=set_groq()

    # Verifica e imposta il percorso di Tesseract
    if not set_tesseract_path():
        print("ERRORE: Tesseract non trovato. Verifica l'installazione e il percorso.")
        return
    
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Leggi il prompt
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    except IOError as e:
        print(f"Errore durante la lettura del file prompt: {e}")
        return

    # Crea/apri il file result.md
    output_filename = os.path.join(output_dir, "result.md")
    
    # Elabora ogni immagine
    image_files = [filename for filename in os.listdir(image_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    sorted_image_files = sorted(image_files, key=extract_number)

    processed_images = set()  # Traccia le immagini già elaborate
    print (f"Utilizzo il modello {model} per processare le immagini: \n\n")
    for filename in sorted_image_files:
        if filename in processed_images:
            continue  # Salta l'immagine se già elaborata
        
        image_path = os.path.join(image_dir, filename)
        success = False
        
        for attempt in range(retries):
            try:
                # Esegui OCR e prepara l'immagine
                ocr_text = perform_ocr(image_path)
                enhanced_prompt = f"{prompt}\n\nRisultato OCR per l'immagine {filename}:\n{ocr_text}"
                
                # Scegli il modello per processare le immagini:
                if model=="gemini":
                    response = call_gemini(image_path=image_path,enhanced_prompt=enhanced_prompt)
                elif model=="groq":
                    response=call_groq(client=client,enhanced_prompt=enhanced_prompt,image_path=image_path)

        
                # Scrivi il risultato nel file
                with open(output_filename, 'a', encoding='utf-8') as outfile:
                    outfile.write(response)
                    outfile.write("\n\n---\n\n")
                
                processed_images.add(filename)
                success = True
                print(f"Elaborazione completata per {filename}")
                break  # Esci dal loop di retry se l'elaborazione ha successo
            
            except Exception as e:
                print(f"Tentativo {attempt + 1} fallito per {filename}: {e}")
                wait_time =3 ** attempt  # Exponential backoff
                print(f"Riprovo tra {wait_time} secondi" )
                time.sleep(wait_time)
        
        if not success:
            print(f"Impossibile elaborare {filename} dopo {retries} tentativi")

    print(f"Elaborazione completata. Immagini elaborate: {len(processed_images)}")
    
def pdf_to_images(pdf_path, output_folder):


    try:
        # Assicurati che la cartella di output esista
        os.makedirs(output_folder, exist_ok=True)
        
        # Apri il PDF
        pdf_document = fitz.open(pdf_path)
        output_files = []

        logging.info(f"Elaborazione PDF: {pdf_path}")
        logging.info(f"Numero di pagine nel PDF: {len(pdf_document)}")

        for page_number in range(len(pdf_document)):
            try:
                # Accedi alla pagina
                page = pdf_document[page_number]
                
                # Renderizza la pagina come immagine con risoluzione più alta
                pix = page.get_pixmap(dpi=300)  # Alta risoluzione
                
                # Percorso di salvataggio dell'immagine
                output_file = os.path.join(output_folder, f"page_{page_number + 1}.png")
                
                # Salva l'immagine in PNG
                pix.save(output_file)
                output_files.append(output_file)
                
                logging.info(f"Salvata immagine: {output_file}")
            except Exception as page_error:
                logging.error(f"Errore nell'elaborazione della pagina {page_number + 1}: {page_error}")
        
        pdf_document.close()
        
        if not output_files:
            logging.warning("Nessuna immagine generata dal PDF")
        
        return output_files
    except Exception as e:
        logging.error(f"Errore fatale nella conversione PDF: {e}")
        return []

def main():
    if len(sys.argv) != 4:
        print("Uso: script.py <cartella_immagini> <file_prompt> <cartella_risultati>")
        sys.exit(1)

    image_directory = sys.argv[1]
    prompt_filepath = sys.argv[2]
    output_directory = sys.argv[3]

    # Verifica esistenza directory e file
    if not os.path.isdir(image_directory):
        logging.error(f"La directory {image_directory} non esiste")
        sys.exit(1)
    
    if not os.path.isfile(prompt_filepath):
        logging.error(f"Il file prompt {prompt_filepath} non esiste")
        sys.exit(1)
    
    # Sottocartella per immagini da PDF
    pdf_images_dir = os.path.join(image_directory, "pdf_images")
    os.makedirs(pdf_images_dir, exist_ok=True)
    

    
    # Converti PDF in immagini
    for filename in os.listdir(image_directory):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(image_directory, filename)
            pdf_images = pdf_to_images(pdf_path, pdf_images_dir)


    process_images(image_directory, prompt_filepath, output_directory)
    process_images(pdf_images_dir, prompt_filepath, output_directory)

if __name__ == "__main__":
    main()











