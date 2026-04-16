

import re
import textwrap
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SummaryResult:
    summary: str          # The final summary paragraph
    method: str           # The algorithm used
    sentence_count: int   # How many sentences are in the summary


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_transcript(text: str) -> str:
  
    # Remove everything between ♪ symbols (song lyrics)
    text = re.sub(r'♪[^♪]*♪', '', text)

    # Remove any remaining ♪ symbols
    text = re.sub(r'♪', '', text)

    # Remove [♪♪♪], [Applause], [Laughter] style tags
    text = re.sub(r'\[.*?\]', '', text)

    # Remove (Ooh), (Yeah) style expressions
    text = re.sub(r'\(.*?\)', '', text)

    # Remove >> speaker indicators often found in auto-captions
    text = re.sub(r'>>', '', text)

    # Collapse multiple spaces and newlines into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def summarize_with_sumy(
    text: str,
    sentence_count: int = 5,
    algorithm: str = "lsa",
) -> SummaryResult:
  
    try:
        import nltk
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer
        from sumy.summarizers.luhn import LuhnSummarizer
        from sumy.summarizers.lex_rank import LexRankSummarizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words
    except ImportError:
        raise ImportError("Run: pip install sumy nltk")

    # Download required nltk data on first run (small, ~1MB)
    print("[sumy] Checking nltk data...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    # Clean the transcript before parsing
    print("[sumy] Cleaning transcript...")
    text = clean_transcript(text)
    print(f"[sumy] Cleaned transcript length: {len(text.split())} words")

    if not text.strip():
        raise ValueError(
            "Transcript is empty after cleaning — "
            "this video may be music-only with no spoken content."
        )

    LANGUAGE = "english"

    # Parse the transcript text into sentences
    # PlaintextParser splits the text into sentences and words
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    # Pick the algorithm
    # A stemmer reduces words to their root form (e.g. "running" → "run")
    # so the algorithm can match related words more accurately
    algorithms = {
        "lsa":     LsaSummarizer(stemmer),
        "luhn":    LuhnSummarizer(stemmer),
        "lexrank": LexRankSummarizer(stemmer),
    }

    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: {list(algorithms.keys())}"
        )

    summarizer = algorithms[algorithm]

    # Stop words are common words like "the", "is", "at" that don't
    # carry meaning and should be ignored when scoring sentences
    summarizer.stop_words = get_stop_words(LANGUAGE)

    print(f"[sumy] Summarizing with {algorithm.upper()} algorithm...")

    # Run the summarizer — returns the top N most important sentences
    summary_sentences = summarizer(parser.document, sentence_count)

    # Join the sentences into a single paragraph
    summary = " ".join(str(sentence) for sentence in summary_sentences)

    return SummaryResult(
        summary=summary,
        method=f"sumy-{algorithm}",
        sentence_count=len(summary_sentences),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def summarize(
    text: str,
    sentence_count: int = 5,
    algorithm: str = "lsa",
) -> SummaryResult:
   
    if not text or not text.strip():
        raise ValueError("Transcript text is empty — nothing to summarize.")

    return summarize_with_sumy(text, sentence_count=sentence_count, algorithm=algorithm)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transcript import get_transcript

    # Swap this URL for any YouTube video you want to test
    # TED talk - How great leaders inspire action (Simon Sinek)
    TEST_URL = "https://www.youtube.com/watch?v=qp0HIF3SfI4"

    print("--- Fetching transcript ---")
    transcript = get_transcript(TEST_URL)
    print(f"Transcript length: {len(transcript.text.split())} words\n")

    print("--- Summarizing with Sumy ---")
    result = summarize(transcript.text, sentence_count=5, algorithm="lsa")

    print("\n=== SUMMARY ===")
    print(textwrap.fill(result.summary, width=80))
    print(f"\nMethod    : {result.method}")
    print(f"Sentences : {result.sentence_count}")