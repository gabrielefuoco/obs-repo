import os
import sys
import math
import argparse
from pydub import AudioSegment
from dotenv import load_dotenv




def check_and_install_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"{package} not found. Installing...")
            os.system(f"{sys.executable} -m pip install {package}")

required_packages = ['pyautogui', 'pyperclip', 'groq', 'pydub', 'python-dotenv']
check_and_install_packages(required_packages)

try:
    from groq import Groq
except ImportError as e:
    print(f"Error importing module: {e}")
    print("Please install the missing module manually using:")
    print(f"pip install {str(e).split()[-1]}")
    sys.exit(1)

load_dotenv(dotenv_path='api-keys.env')  

# TODO rimuovere la chiave
client = Groq(api_key=os.getenv("Groq"))

MAX_CHUNK_SIZE = 25 * 1024 * 1024  # 25 MB in bytes

def create_tmp_folder():
    """
    Create a temporary folder named 'tmp' in the script root directory if it doesn't exist.
    """
    tmp_folder = os.path.join(os.getcwd(), "tmp")
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    return tmp_folder

def split_audio(file_path, max_size_mb=25):
    """
    Split audio file into chunks based on file size and audio duration.
    Save chunks in the 'tmp' folder.
    """
    audio = AudioSegment.from_file(file_path)
    file_size = os.path.getsize(file_path)
    duration_ms = len(audio)

    # Create the tmp folder for temporary chunks
    tmp_folder = create_tmp_folder()

    # Calculate the number of chunks needed
    num_chunks = math.ceil(file_size / (max_size_mb * 1024 * 1024))
    
    # Calculate the duration of each chunk
    chunk_duration_ms = math.ceil(duration_ms / num_chunks)
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration_ms
        end = min((i + 1) * chunk_duration_ms, duration_ms)
        chunk = audio[start:end]

        # Save each chunk in the 'tmp' folder
        chunk_path = os.path.join(tmp_folder, f"{os.path.basename(file_path)[:-4]}_chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3", bitrate="192k")
        chunks.append(chunk_path)
        
        print(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
    
    return chunks


def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model="whisper-large-v3-turbo",
                prompt="""Questa è la registrazione di una lezione universitaria. """,
                response_format="text",
                language="it",
            )
        return transcription
    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        return None



def save_transcription_to_file(transcription, output_file):
    """
    Save transcription to a text file, ensuring order and proper formatting.
    """
    with open(output_file, "a", encoding="utf-8") as f:  # "a" to append to the file
        f.write(transcription + "\n\n")  # Adds two new lines after each chunk

def process_audio_file(audio_file_path, output_folder):
    """
    Process a single audio file, splitting if necessary, and saving the result to a specific text file.
    """
    file_size = os.path.getsize(audio_file_path)
    
    # Generate output file name based on audio file name and output folder
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}-transcripted.txt")
    
    # Clear the content of the output file if it exists
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")

    if file_size <= MAX_CHUNK_SIZE:
        transcription = transcribe_audio(audio_file_path)
        if transcription:
            save_transcription_to_file(transcription, output_file)
    else:
        print(f"File {audio_file_path} is larger than 25 MB. Splitting into chunks...")
        chunks = split_audio(audio_file_path)
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}: {chunk}")
            chunk_transcription = transcribe_audio(chunk)
            if chunk_transcription:
                save_transcription_to_file(chunk_transcription, output_file)
            os.remove(chunk)  # Remove the temporary chunk file

def main():
    # Configura argparse per leggere i parametri da linea di comando
    parser = argparse.ArgumentParser(description="Processa file audio e salva trascrizioni.")
    parser.add_argument('input_folder', type=str, help='Cartella di partenza con i file audio')
    parser.add_argument('output_folder', type=str, help='Cartella di destinazione per le trascrizioni')

    # Parse degli argomenti
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    # Verifica se le cartelle esistono
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
