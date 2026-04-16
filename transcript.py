
import re
import os
import tempfile
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TranscriptResult:
    video_id: str
    text: str                      # Full plain-text transcript
    segments: list[dict]           # List of {start, duration, text} dicts
    source: str                    # "captions" | "whisper"
    language: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_video_id(url_or_id: str) -> str:

    patterns = [
        r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    # Assume it's already a bare video ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    raise ValueError(f"Could not extract a valid video ID from: {url_or_id!r}")


def segments_to_text(segments: list[dict]) -> str:
    """Joins segment texts into a single clean string."""
    return " ".join(seg["text"].strip() for seg in segments)


# ---------------------------------------------------------------------------
# Method 1 – YouTube captions
# ---------------------------------------------------------------------------
def fetch_captions(video_id: str, preferred_languages: list[str] = None) -> TranscriptResult | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
        import requests
    except ImportError:
        raise ImportError("Run: pip install youtube-transcript-api requests")

    preferred_languages = preferred_languages or ["en"]

    try:
        session = requests.Session()
        cookies = {}
        with open("cookies.txt", "r") as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    parts = line.strip().split("\t")
                    if len(parts) >= 7:
                        cookies[parts[5]] = parts[6]
        session.cookies.update(cookies)

        ytt_api = YouTubeTranscriptApi(http_client=session)
        transcript_list = ytt_api.list(video_id)

        try:
            transcript = transcript_list.find_transcript(preferred_languages)
        except NoTranscriptFound:
            transcript = next(iter(transcript_list))
            print(f"[captions] No transcript in {preferred_languages}. Translating to English.")
            transcript = transcript.translate("en")

        raw_segments = transcript.fetch()

        segments = [
            {"start": snip.start, "duration": snip.duration, "text": snip.text}
            for snip in raw_segments.snippets
        ]

        return TranscriptResult(
            video_id=video_id,
            text=segments_to_text(segments),
            segments=segments,
            source="captions",
            language=transcript.language_code,
        )

    except (NoTranscriptFound, TranscriptsDisabled) as e:
        print(f"[captions] Not available: {e}")
        return None

# ---------------------------------------------------------------------------
# Method 2 – Whisper (audio transcription)
# ---------------------------------------------------------------------------

def fetch_whisper(
    video_id: str,
    whisper_model: str = "base",
    audio_format: str = "mp3",
) -> TranscriptResult:
   
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("Run: pip install yt-dlp")

    try:
        import whisper
    except ImportError:
        raise ImportError("Run: pip install openai-whisper")

    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "128",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        print(f"[whisper] Downloading audio for {video_id} …")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded file
        downloaded = [f for f in os.listdir(tmpdir) if f.startswith("audio")]
        if not downloaded:
            raise FileNotFoundError("Audio download failed — no file found in temp dir.")
        final_audio = os.path.join(tmpdir, downloaded[0])

        print(f"[whisper] Loading model '{whisper_model}' …")
        model = whisper.load_model(whisper_model)

        print("[whisper] Transcribing …")
        result = model.transcribe(final_audio)

    segments = [
        {
            "start": seg["start"],
            "duration": seg["end"] - seg["start"],
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
    ]

    return TranscriptResult(
        video_id=video_id,
        text=result["text"].strip(),
        segments=segments,
        source="whisper",
        language=result.get("language"),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_transcript(
    url_or_id: str,
    preferred_languages: list[str] = None,
    whisper_model: str = "base",
    force_whisper: bool = False,
) -> TranscriptResult:
    
    video_id = extract_video_id(url_or_id)
    print(f"[transcript] Video ID: {video_id}")

    if not force_whisper:
        result = fetch_captions(video_id, preferred_languages)
        if result:
            print(f"[transcript] ✓ Got captions ({len(result.segments)} segments, lang={result.language})")
            return result
        print("[transcript] Captions unavailable — falling back to Whisper …")

    result = fetch_whisper(video_id, whisper_model=whisper_model)
    print(f"[transcript] ✓ Whisper transcription done ({len(result.segments)} segments, lang={result.language})")
    return result


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your URL

    transcript = get_transcript(TEST_URL)

    print("\n--- TRANSCRIPT PREVIEW (first 500 chars) ---")
    print(transcript.text[:500])
    print(f"\nSource : {transcript.source}")
    print(f"Language: {transcript.language}")
    print(f"Segments: {len(transcript.segments)}")

    # Save to JSON for inspection
    with open("transcript_output.json", "w") as f:
        json.dump({
            "video_id": transcript.video_id,
            "source": transcript.source,
            "language": transcript.language,
            "text": transcript.text,
            "segments": transcript.segments,
        }, f, indent=2)
    print("\nSaved full output to transcript_output.json")