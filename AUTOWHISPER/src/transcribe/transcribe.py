import os
import sys
import math
import argparse
from pydub import AudioSegment
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv(dotenv_path='api-keys.env')  

try:
    from groq import Groq
except ImportError as e:
    print(f"Error importing module: {e}")
    sys.exit(1)

# Configura l'API Groq
client = Groq(api_key=os.getenv("Groq"))

MAX_CHUNK_SIZE = 25 * 1024 * 1024  # 25 MB in bytes


def create_tmp_folder():
    tmp_folder = os.path.join(os.getcwd(), "tmp")
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    return tmp_folder


def split_audio(file_path, max_size_mb=25):
    audio = AudioSegment.from_file(file_path)
    file_size = os.path.getsize(file_path)
    duration_ms = len(audio)

    tmp_folder = create_tmp_folder()
    num_chunks = math.ceil(file_size / (max_size_mb * 1024 * 1024))+1
    chunk_duration_ms = math.ceil(duration_ms / num_chunks) 
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration_ms
        end = min((i + 1) * chunk_duration_ms, duration_ms)
        chunk = audio[start:end]

        chunk_path = os.path.join(tmp_folder, f"{os.path.basename(file_path)[:-4]}_chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3", bitrate="192k")
        chunks.append(chunk_path)
        
        print(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
    
    return chunks


def transcribe_audio(audio_file_path, chunk_index):
    """Trascrive un singolo file audio e ritorna l'indice e la trascrizione."""
    try:
        with open(audio_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model="whisper-large-v3-turbo",
                prompt="""Questa è la registrazione di una lezione universitaria. """,
                response_format="text",
                language="it",
            )
        return chunk_index, transcription
    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        return chunk_index, None


def save_transcription_to_file(transcriptions, output_file):
    """Salva le trascrizioni in ordine."""
    transcriptions.sort(key=lambda x: x[0])  # Ordina per indice
    with open(output_file, "a", encoding="utf-8") as f:
        for _, transcription in transcriptions:
            if transcription:
                f.write(transcription + "\n\n")


def process_audio_file(audio_file_path, output_folder):
    file_size = os.path.getsize(audio_file_path)
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}-transcripted.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")  # Resetta il file di output

    # Se il file è più piccolo del massimo chunk size, trascrivi direttamente
    if file_size <= MAX_CHUNK_SIZE:
        transcription = transcribe_audio(audio_file_path, 0)
        if transcription[1]:  # transcription is a tuple (index, result)
            save_transcription_to_file([transcription], output_file)
    else:
        print(f"File {audio_file_path} is larger than 25 MB. Splitting into chunks...")
        chunks = split_audio(audio_file_path)

        # Uso di ThreadPoolExecutor per trascrivere in parallelo
        transcriptions = []
        with ThreadPoolExecutor(max_workers=4) as executor:  # Puoi regolare max_workers in base al carico
            futures = [executor.submit(transcribe_audio, chunk, i) for i, chunk in enumerate(chunks)]
            
            for future in as_completed(futures):
                transcriptions.append(future.result())

        # Salva le trascrizioni nel file, assicurandosi che siano nell'ordine giusto
        save_transcription_to_file(transcriptions, output_file)

        # Rimuovi i file temporanei
        for chunk in chunks:
            os.remove(chunk)


def main():
    parser = argparse.ArgumentParser(description="Processa file audio e salva trascrizioni.")
    parser.add_argument('input_folder', type=str, help='Cartella di partenza con i file audio')
    parser.add_argument('output_folder', type=str, help='Cartella di destinazione per le trascrizioni')

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.isdir(input_folder):
        print(f"Errore: la cartella di partenza {input_folder} non esiste.")
        return
    if not os.path.isdir(output_folder):
        print(f"Errore: la cartella di destinazione {output_folder} non esiste.")
        return
    
    print(f"Processing audio files in: {input_folder}")
    
    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.wav', '.mp3'))]
    
    if not audio_files:
        print("Nessun file WAV o MP3 trovato nella cartella di partenza.")
        return

    for filename in audio_files:
        audio_file_path = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")
        process_audio_file(audio_file_path, output_folder)
        print(f"Transcription for {filename} saved to {output_folder}/{filename[:-4]}-transcripted.txt")

    print("Finished processing all audio files.")


if __name__ == "__main__":
    main()
