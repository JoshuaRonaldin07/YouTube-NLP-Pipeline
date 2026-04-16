# YouTube-NLP-Pipeline
A Python project that takes any YouTube URL and returns a summary and sentiment analysis of the video's transcript.

## What it does
 
1. **Fetches the transcript** — pulls captions directly from YouTube, or falls back to transcribing the audio with OpenAI Whisper if captions are unavailable
2. **Summarizes the content** — extracts the most important sentences using the LSA algorithm (no model downloads, no API key)
3. **Analyzes sentiment** — scores every segment of the transcript from -1.0 (negative) to +1.0 (positive) using VADER, and shows how tone shifts across the video
4. **Saves results** — exports everything to a JSON file for further use
---
 
## Project structure
 
```
Summ/
├── main.py          # Run this — combines all three steps into one pipeline
├── transcript.py    # Fetches transcript via captions or Whisper
├── summarizer.py    # Summarizes transcript using sumy (extractive)
├── sentiment.py     # Analyzes sentiment per segment and overall (VADER)
├── cookies.txt      # Your YouTube cookies (required to bypass rate limiting)
└── README.md
```
 
---
 
## Setup
 
### 1. Make sure Python 3.13 is installed
 
Download from [python.org](https://python.org/downloads). During installation, tick **"Add Python to PATH"**.
 
### 2. Install dependencies
 
```powershell
python -m pip install youtube-transcript-api yt-dlp sumy nltk requests
```
 
> If `python` is not recognized, use the full path:
> ```powershell
> C:\Users\<you>\AppData\Local\Programs\Python\Python313\python.exe -m pip install youtube-transcript-api yt-dlp sumy nltk requests
> ```
 
### 3. Set up cookies.txt (required)
 
YouTube rate-limits unauthenticated transcript requests. To bypass this:
 
1. Install the **"Get cookies.txt LOCALLY"** extension in Chrome
2. Log into [youtube.com](https://youtube.com)
3. Click the extension → Export → save as `cookies.txt`
4. Place `cookies.txt` in the `Summ/` folder
---
 
## Usage
 
Open `main.py` and change the URL at the bottom:
 
```python
TEST_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```
 
Then run:
 
```powershell
python main.py
```
 
### Options
 
```python
run_pipeline(
    url="https://www.youtube.com/watch?v=...",
    summary_sentences=5,        # number of sentences in the summary
    summary_algorithm="lsa",    # "lsa", "luhn", or "lexrank"
    show_all_segments=False,    # True = show every segment, False = non-neutral only
    save_json=True,             # saves results to results_<video_id>.json
)
```
 
---
 
## Example output
 
```
============================================================
  YouTube NLP Pipeline
============================================================
 
[1/3] Fetching transcript...
      Video ID  : qp0HIF3SfI4
      Source    : captions
      Language  : en
      Segments  : 382
      Words     : 3015
 
[2/3] Summarizing transcript...
      Method    : sumy-lsa
      Sentences : 5
 
[3/3] Analyzing sentiment...
 
--- SUMMARY ---
As it turns out, all the great inspiring leaders and organizations
in the world, whether it's Apple or Martin Luther King or the Wright
brothers, they all think, act and communicate the exact same way...
 
=== OVERALL SENTIMENT ===
Label         : POSITIVE ✅
Average score : 0.0863
Positive      : 24.1%
Negative      : 6.8%
Neutral       : 69.1%
 
--- SENTIMENT ARC ---
  Beginning [0:16-6:13]   +0.134  neutral
  Middle    [6:15-12:01]  +0.077  neutral
  End       [12:03-17:57] +0.049  neutral
 
💾 Full results saved to results_qp0HIF3SfI4.json
```
 
---
 
## How it works
 
### Transcript (`transcript.py`)
 
Tries YouTube's built-in captions first via `youtube-transcript-api`. If the video has no captions or they are disabled, it downloads the audio with `yt-dlp` and transcribes it locally using OpenAI Whisper. The result is a list of timestamped segments.
 
### Summarizer (`summarizer.py`)
 
Uses **extractive summarization** — rather than generating new text, it picks the most important sentences directly from the transcript. The default algorithm is LSA (Latent Semantic Analysis), which finds sentences that cover the most distinct topics. Two alternatives are available: Luhn (keyword frequency) and LexRank (graph-based, similar to PageRank).
 
### Sentiment (`sentiment.py`)
 
Uses **VADER** (Valence Aware Dictionary and sEntiment Reasoner) — a rule-based analyzer built for conversational text. It scores each transcript segment individually and aggregates them into an overall score. The sentiment arc splits the video into thirds to show how tone evolves from start to finish.
 
---
 
## Dependencies
 
| Package | Purpose |
|---|---|
| `youtube-transcript-api` | Fetch YouTube captions |
| `yt-dlp` | Download audio for Whisper fallback |
| `openai-whisper` | Transcribe audio locally |
| `sumy` | Extractive summarization |
| `nltk` | Sentence tokenization + VADER sentiment |
| `requests` | Pass cookies to bypass rate limiting |
 
---
 
## Known limitations
 
- **Music videos** produce poor summaries and sentiment scores because lyrics repeat and lack normal sentence structure
- **Auto-generated captions** sometimes cut sentences mid-way, causing minor artifacts in the summary
- **Whisper** requires `ffmpeg` to be installed and is slow on CPU — use a small model (`"tiny"` or `"base"`) for faster results
- **VADER** is optimized for English — results on non-English transcripts that have been auto-translated may be less accurate
---
