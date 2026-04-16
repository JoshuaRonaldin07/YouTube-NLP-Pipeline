
import textwrap
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SegmentSentiment:
    start: float        # Timestamp in seconds
    text: str           # The segment text
    label: str          # "positive", "negative", or "neutral"
    score: float        # Compound score from -1.0 (most negative) to 1.0 (most positive)


@dataclass
class SentimentResult:
    segments: list[SegmentSentiment]   # Per-segment sentiment
    overall: dict                       # Overall sentiment summary
    label: str                          # Overall label: "positive", "negative", "neutral"
    score: float                        # Overall compound score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """
    Converts seconds to MM:SS format for readable output.
    e.g. 90.5 → "1:30"
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def label_from_score(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

def analyze_segments(segments: list[dict]) -> list[SegmentSentiment]:
   
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError:
        raise ImportError("Run: pip install nltk")

    # Download VADER lexicon on first run (~1MB)
    print("[sentiment] Downloading VADER lexicon if needed...")
    nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()
    results = []

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        # scores = {"neg": 0.0, "neu": 0.8, "pos": 0.2, "compound": 0.34}
        scores = sia.polarity_scores(text)
        compound = scores["compound"]

        results.append(SegmentSentiment(
            start=seg["start"],
            text=text,
            label=label_from_score(compound),
            score=round(compound, 4),
        ))

    return results


def compute_overall(segment_results: list[SegmentSentiment]) -> tuple[dict, str, float]:
   
    if not segment_results:
        return {}, "neutral", 0.0

    scores = [seg.score for seg in segment_results]
    avg_score = round(sum(scores) / len(scores), 4)

    positive = sum(1 for s in segment_results if s.label == "positive")
    negative = sum(1 for s in segment_results if s.label == "negative")
    neutral  = sum(1 for s in segment_results if s.label == "neutral")
    total    = len(segment_results)

    summary = {
        "average_score" : avg_score,
        "positive_pct"  : round(positive / total * 100, 1),
        "negative_pct"  : round(negative / total * 100, 1),
        "neutral_pct"   : round(neutral  / total * 100, 1),
        "total_segments": total,
    }

    return summary, label_from_score(avg_score), avg_score


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze(segments: list[dict]) -> SentimentResult:
   
    if not segments:
        raise ValueError("Segments list is empty — nothing to analyze.")

    print(f"[sentiment] Analyzing {len(segments)} segments...")
    segment_results = analyze_segments(segments)

    print("[sentiment] Computing overall sentiment...")
    overall, label, score = compute_overall(segment_results)

    return SentimentResult(
        segments=segment_results,
        overall=overall,
        label=label,
        score=score,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_results(result: SentimentResult, show_all_segments: bool = False):
   
    # Emoji map for labels
    icons = {"positive": "✅", "negative": "❌", "neutral": "⬜"}

    print("\n=== OVERALL SENTIMENT ===")
    print(f"Label         : {result.label.upper()} {icons[result.label]}")
    print(f"Average score : {result.score} (range: -1.0 to 1.0)")
    print(f"Positive      : {result.overall['positive_pct']}%")
    print(f"Negative      : {result.overall['negative_pct']}%")
    print(f"Neutral       : {result.overall['neutral_pct']}%")
    print(f"Total segments: {result.overall['total_segments']}")

    print("\n=== SEGMENT BREAKDOWN ===")
    print("(showing non-neutral segments — pass show_all_segments=True to see all)\n")

    for seg in result.segments:
        if not show_all_segments and seg.label == "neutral":
            continue

        timestamp = format_timestamp(seg.start)
        icon = icons[seg.label]
        preview = textwrap.shorten(seg.text, width=60, placeholder="...")
        print(f"[{timestamp:>5}] {icon} {seg.score:+.3f}  {preview}")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transcript import get_transcript

    # Simon Sinek TED talk — good spoken content for sentiment analysis
    TEST_URL = "https://www.youtube.com/watch?v=qp0HIF3SfI4"

    print("--- Fetching transcript ---")
    transcript = get_transcript(TEST_URL)
    print(f"Segments: {len(transcript.segments)}\n")

    print("--- Analyzing sentiment ---")
    result = analyze(transcript.segments)

    # Print results — change to True to see every segment
    print_results(result, show_all_segments=False)