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
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

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
                    summarize(int(selected_model), result["text"])
                    print("You should not see this message - transcribe")
                elif selection == "N":
                    print("Have a nice day!")
                    sys.exit()
                print("Invalid selection, please type Y or N")
            
        else:
            print("Please input a digit 0-9 to select a model")

def summarize(accuracy, text): # adjust the template to remove the tags, and the introductions. Also adjust the notion of the accuracy index to strictly mean the accuracy of the transcription, not of the factual information
    
    accuracy -= 1
    if accuracy == -1: # 0 corresponds to turbo, which is roughly as accurate as large, so we are going to set it to be of the highest accuracy for simplicity's sake.
        accuracy = 9
    
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

The document is a transcription of a video. The transcription may have errors in the words themselves. This is measured by the the accuracy index, which is given to you along with the document. It is an integer ranging from 0 to 9, where 9 is the most accurate and 0 is the least accurate.

The accuracy index only makes a claim about the accuracy of the transcription. It says nothing about the accuracy of the content of the document.

The user may ask you followup questions, or ask you to provide more details about the document after your initial report is submitted. Please respond only with the report that you are requested to make, in the format specified above.

Here is the accuracy index: {accuracy_index}

Here is the document: {document}

"""

    # clarify what the accuracy index means and make sure its only the accuracy of transcription

    followup_template = """
    
    You are an expert writer and document analyzer. You previously analyzed this document: {document}
    
    The document is a transcription of a video. The transcription may have errors in the words themselves. This is measured by the the accuracy index, which is given to you along with the document. It is an integer ranging from 0 to 9, where 9 is the most accurate and 0 is the least accurate.

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
    
    
    model = ChatOllama(model="deepseek-r1", streaming=True, reasoning=True)
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    inputs = {"accuracy_index": accuracy, "document": text}
    
    summary_text = ""
    
    print("Initializing Response (May take a bit to get started)...\n")
    
    for chunk in chain.stream(inputs):
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