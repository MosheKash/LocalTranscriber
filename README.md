# 🎙️ LocalTranscriber

A **fully local**, privacy‑friendly video transcription and summarization tool powered by **OpenAI Whisper**, **PyTorch**, **LangChain**, and **DeepSeek R1**.

No cloud uploads. No subscriptions. Just fast, high‑quality transcription and summaries—right on your machine.

> Built and maintained by **Moshe Kashlinsky**

---

## ✨ Features

- 🎧 **High‑accuracy transcription** using Whisper
- 🧠 **Automatic summarization** with DeepSeek R1
- 📼 **YouTube video support** (audio downloaded automatically)
- 🔒 **Runs 100% locally** — your data never leaves your computer
- 🗂️ **Clean file organization** for models, audio, and transcripts
- 🖥️ **Simple interactive UI** — no config headaches

---

## 🛠️ Requirements

- **Conda** (required)
- Python 3.x (handled by Conda)
- A machine capable of running PyTorch (GPU recommended, but not required)

---

## 🚀 Installation

Clone the repository and create the Conda environment:

```bash
conda env create -f environment.yml
conda activate localtranscriber
```

That’s it! You’re ready to go.

---

## ▶️ Usage

Once the environment is activated, run:

```bash
python3 transcriber.py
```

The script is interactive and self‑explanatory; just follow the on‑screen prompts to:

- Download YouTube audio
- Transcribe videos
- Generate summaries

---

## 📁 Project Structure

As you use LocalTranscriber, several folders will be created automatically:

```
LocalWhisperModels/      # Downloaded Whisper models
YouTubeAudioFiles/       # Audio pulled from YouTube videos
TranscribedAudioFiles/   # Final transcriptions and summaries
```

No manual setup required—everything is handled for you.

---

## 🐛 Bugs & Feature Requests

Found a bug? Have an idea for a new feature?

📧 Email me at **[kashlinskymoshe@gmail.com](mailto\:kashlinskymoshe@gmail.com)** — I’d love to hear from you.

---

## ⭐ Tips

- Larger Whisper models = better accuracy (but slower)
- A GPU will significantly speed up transcription
- Long videos may take time—grab a coffee ☕

---

## 📜 License

Use it, tweak it, break it, improve it. Attribution appreciated.

Hope this is helpful!