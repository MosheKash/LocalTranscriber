# Script by Moshe Kashlinsky
# kashlinskymoshe@gmail.com

import os
from pathlib import Path
import yt_dlp
import whisper
import json
import torch
import sys
import textwrap

# General Overview:

# 1. Ask for a YouTube Link, or for a local file (give path)
# 2. Ask for the accuracy, and say the requirements that are needed + estimated time
# 3. Transcribe and save into desired json formats
# 4. Offer to summarize with some sort of GPT

def intro():
    valid = {'1': fetch_file_youtube, '2': fetch_file_local}
    print("Welcome! This script is brought to you by Moshe Kashlinsky.\n")
    print("Would you like to transcribe a YouTube video or a local file?\n")
    print("1. YouTube Video\n")
    print("2. Local File\n")
    while True:
        result = input("Selection: ")
        if result in valid:
            return valid[result]()
        print("Invalid input. Please type 1 for a YouTube Video, and 2 for a Local File")
        

def fetch_file_youtube():
    if not os.path.exists("YouTubeAudioFiles"):
        print("Creating folder YouTubeAudioFiles in current directory...")
        os.makedirs("YouTubeAudioFiles")
        
    print("Please paste your URL below:")
    url = input("URL: ")
    print(f"Fetched from YT at url {url}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],

        'outtmpl': os.path.join("YouTubeAudioFiles", '%(title)s.%(ext)s'),
        'quiet': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        original_path = Path(ydl.prepare_filename(info))
        mp3_path = original_path.with_suffix('.mp3')
    
    
    print(f"Successfully downloaded YouTube video to path {mp3_path}\n")
    print("Preparing to transcribe...\n")
    
    transcribe(mp3_path)
    
def fetch_file_local():
    print(f"Please paste your path to the .mp3 file")
    while True:
        user_input = input("Path: ").strip()

        path = Path(user_input.replace('"', '').replace("'", ""))
        
        # Validation
        if path.is_absolute() and path.exists() and path.is_file():
            if path.suffix.lower() == '.mp3':
                print(f"Success: Found {path.name}\n")
                print("Preparing to transcribe...\n")
                transcribe(path)
                break
            else:
                print(f"Error: Extension is {path.suffix}, but .mp3 is required.")
        else:
            # if not path.is_absolute():
            #     print(f"Error: Path must be absolute (starting with {root_example}).")
            if not path.exists():
                print("Error: The path provided does not exist.")
            elif path.is_dir():
                print("Error: You provided a directory. Please include the file name.")

        print("Please try again.\n")
    
    
def transcribe(path: Path):
    print("Select a model to use. Faster models are often less accurate. For more information on models, especially on required VRAM, see this link: https://github.com/openai/whisper\n")
    print("If the audio is english only, I would recommend using an English only model for higher accuracy\n")
    print("1. Tiny")
    print("2. Tiny (English Only)")
    print("3. Base")
    print("4. Base (English Only)")
    print("5. Small")
    print("6. Small (English Only)")
    print("7. Medium")
    print("8. Medium (English Only)")
    print("9. Large")
    print("0. Turbo\n")
    
    models = {'1': "tiny", '2': 'tiny.en', '3': 'base', '4': 'base.en', '5': 'small', '6': 'small.en', '7': 'medium', '8': 'medium.en', '9': 'large', '0': 'turbo'}
    
    while True:
        selected_model = input("Desired Model: ")
        
        if selected_model in models:
            model_name = models[selected_model]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\nSelected Model {model_name} on {device.upper()}. Loading model...")
            
            model = whisper.load_model(model_name, device=device)
            
            print("-" * 30)
            print("TRANSCRIPTION STARTING")
            print("Progress will be printed below as it processes:")
            print("-" * 30)
            
            result = model.transcribe(str(path), verbose=True)

            print("-" * 30)
            print("Transcription complete, saving file...")
            
            if not os.path.exists("TranscribedAudioFiles"):
                print("Creating folder TranscribedAudioFiles in current directory...")
                os.makedirs("TranscribedAudioFiles")
            print("Transcription complete, ouputting to .json\n")
            
            with open(os.path.join("TranscribedAudioFiles", path.name + "_ALL_DATA.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
            
            with open(os.path.join("TranscribedAudioFiles", path.name + "_TEXT_ONLY.json"), "w", encoding="utf-8") as f:
                json.dump({"text": result["text"]}, f, indent=2)
            
            print("For outputting to .txt, would you like to make it readable? Y/N")
            text = result["text"]
            while True:
                selection = input("Selection: ")
                if selection == "Y":
                    text = textwrap.fill(result["text"], width=80)
                    break
                elif selection == "N":
                    break
                print("Invalid selection, please type Y or N")
                
            with open(os.path.join("TranscribedAudioFiles", path.name + "_TEXT_ONLY.txt"), "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Saved to: \n\tTranscribedAudioFiles/{path.name}_ALL_DATA.json\n\tTranscribedAudioFiles/{path.name}_TEXT_ONLY.json\n\tTranscribedAudioFiles/{path.name}_TEXT_ONLY.txt\n")
            
            print("Would you like to delete the audio file used in transcription? Y/N")
            while True:
                selection = input("Selection: ")
                if selection == "Y":
                    path.unlink()
                    break
                elif selection == "N":
                    break
                print("Invalid selection, please type Y or N")
            
            print("Would you like to automatically summarize the text used here? Y/N")
            while True:
                selection = input("Selection: ")
                if selection == "Y":
                    summarize(result["text"])
                    print("You should not see this message - transcribe")
                elif selection == "N":
                    print("Have a nice day!")
                    sys.exit()
                print("Invalid selection, please type Y or N")
            
        else:
            print("Please input a digit 0-9 to select a model")

def summarize(text):
    print("-------------------------")
    print("UNIMPLEMENTED, EXITING!!!")
    print("-------------------------")
    sys.exit()
    
    

if __name__ == "__main__":
    intro()