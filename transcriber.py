# Script by Moshe Kashlinsky
# kashlinskymoshe@gmail.com

import os
from pathlib import Path
import yt_dlp
import json
import torch
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import time

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import sys
import textwrap
import whisperx
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from whisperx.diarize import DiarizationPipeline

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
    print("1. Tiny (English Only)")
    print("2. Tiny")
    print("3. Base (English Only)")
    print("4. Base")
    print("5. Small (English Only)")
    print("6. Small")
    print("7. Distil Small (English Only)")
    print("8. Medium (English Only)")
    print("9. Medium")
    print("10. Distil Medium (English Only)")
    print("11. Large v1")
    print("12. Large v2")
    print("13. Large v3")
    print("14. Large (alias v3)")
    print("15. Distil Large v2")
    print("16. Distil Large v3")
    print("17. Distil Large v3.5")
    print("18. Large v3 Turbo")
    print("19. Turbo\n")

    models = {
        '1': 'tiny.en',
        '2': 'tiny',
        '3': 'base.en',
        '4': 'base',
        '5': 'small.en',
        '6': 'small',
        '7': 'distil-small.en',
        '8': 'medium.en',
        '9': 'medium',
        '10': 'distil-medium.en',
        '11': 'large-v1',
        '12': 'large-v2',
        '13': 'large-v3',
        '14': 'large',
        '15': 'distil-large-v2',
        '16': 'distil-large-v3',
        '17': 'distil-large-v3.5',
        '18': 'large-v3-turbo',
        '19': 'turbo',
    }

    
    while True:
        selected_model = input("Desired Model: ")
        
        if selected_model in models:
            model_name = models[selected_model]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\nAttempting to load model {model_name} on {device.upper()}...")

            model = whisperx.load_model(model_name, device=device, compute_type="float16", download_root="LocalWhisperModels")
            batch_size = 16 
            audio = whisperx.load_audio(path)
            print("-" * 30)
            print("TRANSCRIPTION STARTING")
            print("Progress will be printed below as it processes:")
            print("-" * 30)

            try:
                result = model.transcribe(audio, batch_size=batch_size, verbose=True)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device == "cuda":
                    print("CUDA OOM during transcription. Retrying with batch_size=1...")
                    torch.cuda.empty_cache()
                    
                    try:
                        # Try again on GPU but with minimum memory footprint
                        result = model.transcribe(audio, batch_size=1, verbose=True)
                    except RuntimeError:
                        print("Still OOM. Falling back to CPU...")
                        torch.cuda.empty_cache()
                        # Reload model on CPU
                        model = whisperx.load_model(model_name, device="cpu", download_root="LocalWhisperModels")
                        result = model.transcribe(audio, batch_size=1, verbose=True)
                else:
                    raise e
            
            print("Base Transcription Done, Aligning...\n")
            
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            torch.cuda.empty_cache()
            result["text"] = "".join([segment["text"] for segment in result["segments"]])

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
                    summarize(int(selected_model), result["text"])
                    print("You should not see this message - transcribe")
                elif selection == "N":
                    print("Have a nice day!")
                    sys.exit()
                print("Invalid selection, please type Y or N")
            
        else:
            print("Please input a number 1-9 to select a model")

def summarize(accuracy, text):
    
    template = """

    You are an expert writer and document analyzer. You are given a document of critical importance by a client who is paying you top-dollar to read it, and then compose a summary of the document.

    The requested format of the summary should be as follows, with the <CUSTOMIZABLE> section allowing you to put whatever you feel is necessary in that section. The <CUSTOMIZABLE> section should not be too long, and is optional. The template ends at <END>.

    You should not include the headers themselves that start with <> in the template below. They exist solely to tell you about the formatting. The template is shown below:

    YOU MUST STRICTLY FOLLOW THE FORMAT BELOW DURING YOUR RESPONSE. ANY DEVIANCE FROM THE FORMAT BELOW WILL RESULT IN YOU LOSING YOUR JOB.

    Brief Overview:

    Important Ideas:

    - Bullet points

    Most Important Parts:

    - Excerpts in the form of bullet points

    <CUSTOMIZABLE>

    <END>

    The text may contain some spelling errors, and it is your job to account for them and use judgement to figure out what could be meant by them.

    The document is a transcription of a video. The transcription may have errors in the words themselves. This is measured by the the accuracy index, which is given to you along with the document. It is an integer ranging from 1 to 19, where 19 is the most accurate and 1 is the least accurate.

    The accuracy index only makes a claim about the accuracy of the transcription. It says nothing about the accuracy of the content of the document.

    The user may ask you followup questions, or ask you to provide more details about the document after your initial report is submitted. Please respond only with the report that you are requested to make, in the format specified above.

    Here is the accuracy index: {accuracy_index}

    Here is the document: {document}

    """

    followup_template = """
    
    You are an expert writer and document analyzer. You previously analyzed this document: {document}
    
    The document is a transcription of a video. The transcription may have errors in the words themselves. This is measured by the the accuracy index, which is given to you along with the document. It is an integer ranging from 1 to 19, where 19 is the most accurate and 1 is the least accurate.

    The accuracy index only makes a claim about the accuracy of the transcription. It says nothing about the accuracy of the content of the document.
    
    The accuracy index of this document was: {accuracy_index}
    
    Your previous summary of the document was: {previous_summary}
    
    The user, a client who is paying you top-dollar to read the document and understand and convey the information clearly, now has a followup question on this document: {question}
    
    Please answer their question based on the document. Be specific and reference relevant parts of the document. Make sure to use clear language.
    
    """
    
    showReasoning = False

    print("The model will think for a bit to ensure a good answer. Would you like to show the thinking (May clog up terminal)? Y/N")
    while True:
        selection = input("Selection: ")
        if selection == "Y":
            showReasoning = True
            break
        elif selection == "N":
            break
        print("Invalid selection, please type Y or N")
    
    
    print(f"Document length: {len(text)} characters")
    
    t1 = time.time()
    model = ChatOllama(model="deepseek-r1", streaming=True, reasoning=True)
    print(f"Model init took: {time.time() - t1:.2f}s")
    
    t2 = time.time()
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    print(f"Chain creation took: {time.time() - t2:.2f}s")
    
    inputs = {"accuracy_index": accuracy, "document": text}
    
    print("Waiting for first token...")
    t3 = time.time()

    summary_text = ""
    
    print("Initializing Response (May take a bit to get started)...\n")
    first_token_received = False  # Flag to print only once
    
    for chunk in chain.stream(inputs):
        if not first_token_received:
            print(f"First token received after: {time.time() - t3:.2f}s")
            first_token_received = True

        # Check for reasoning (thinking) tokens
        # These are usually in additional_kwargs when reasoning=True
        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning and showReasoning:
            print(f"\033[90m{reasoning}\033[0m", end="", flush=True)
        
        # Check for the actual answer tokens
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            summary_text += content
    
    followup_prompt = ChatPromptTemplate.from_template(followup_template)
    followup_chain = followup_prompt | model
    
    while True: # fix formatting here, we want it to match previous messages
        user_input = input("\nAsk a followup question (or type 'exit' to quit) >>> ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q', '']:
            print("Exiting followup mode. Goodbye!")
            break
        
        followup_inputs = {
            "accuracy_index": accuracy,
            "document": text,
            "previous_summary": summary_text,
            "question": user_input
        }
        
        for chunk in followup_chain.stream(followup_inputs):
            # Check for reasoning (thinking) tokens
            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            if reasoning and showReasoning:
                print(f"\033[90m{reasoning}\033[0m", end="", flush=True)
            
            # Check for the actual answer tokens
            content = chunk.content
            if content:
                print(content, end="", flush=True)
    
    sys.exit()
    
    

if __name__ == "__main__":
    intro()