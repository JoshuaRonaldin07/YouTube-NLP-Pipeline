
import json
import textwrap
from dataclasses import asdict

from transcript import get_transcript
from summarizer import summarize
from sentiment import analyze, print_results, format_timestamp


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    url: str,
    summary_sentences: int = 5,
    summary_algorithm: str = "lsa",
    show_all_segments: bool = False,
    save_json: bool = True,
) -> dict:
   
    print("=" * 60)
    print("  YouTube NLP Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Transcript
    # ------------------------------------------------------------------
    print("\n[1/3] Fetching transcript...")
    transcript = get_transcript(url)

    print(f"      Video ID  : {transcript.video_id}")
    print(f"      Source    : {transcript.source}")
    print(f"      Language  : {transcript.language}")
    print(f"      Segments  : {len(transcript.segments)}")
    print(f"      Words     : {len(transcript.text.split())}")

    # ------------------------------------------------------------------
    # Step 2 — Summary
    # ------------------------------------------------------------------
    print("\n[2/3] Summarizing transcript...")
    summary_result = summarize(
        transcript.text,
        sentence_count=summary_sentences,
        algorithm=summary_algorithm,
    )

    print(f"      Method    : {summary_result.method}")
    print(f"      Sentences : {summary_result.sentence_count}")

    # ------------------------------------------------------------------
    # Step 3 — Sentiment
    # ------------------------------------------------------------------
    print("\n[3/3] Analyzing sentiment...")
    sentiment_result = analyze(transcript.segments)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    # Summary
    print("\n--- SUMMARY ---")
    print(textwrap.fill(summary_result.summary, width=60))

    # Sentiment
    print_results(sentiment_result, show_all_segments=show_all_segments)

    # Sentiment arc — show how tone shifts over the video in thirds
    print("\n--- SENTIMENT ARC ---")
    print_sentiment_arc(sentiment_result)

    # ------------------------------------------------------------------
    # Save to JSON
    # ------------------------------------------------------------------
    output = {
        "video_id"  : transcript.video_id,
        "source"    : transcript.source,
        "language"  : transcript.language,
        "word_count": len(transcript.text.split()),
        "summary"   : {
            "text"      : summary_result.summary,
            "method"    : summary_result.method,
            "sentences" : summary_result.sentence_count,
        },
        "sentiment" : {
            "label"   : sentiment_result.label,
            "score"   : sentiment_result.score,
            "overall" : sentiment_result.overall,
            "segments": [
                {
                    "timestamp": format_timestamp(seg.start),
                    "start_sec": seg.start,
                    "label"    : seg.label,
                    "score"    : seg.score,
                    "text"     : seg.text,
                }
                for seg in sentiment_result.segments
            ],
        },
    }

    if save_json:
        filename = f"results_{transcript.video_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full results saved to {filename}")

    return output


# ---------------------------------------------------------------------------
# Sentiment arc helper
# ---------------------------------------------------------------------------

def print_sentiment_arc(sentiment_result) -> None:
    """
    Splits the video into three equal thirds and shows the average
    sentiment score for each third — giving you a sense of how the
    tone evolves from beginning to end.

    For example a talk might start neutral, get positive in the middle
    as examples are given, then end on a high note.
    """
    segments = sentiment_result.segments
    if not segments:
        return

    third = len(segments) // 3
    thirds = [
        segments[:third],
        segments[third:third * 2],
        segments[third * 2:],
    ]
    labels = ["Beginning", "Middle  ", "End     "]
    bar_chars = {
        "positive": "█",
        "negative": "▓",
        "neutral" : "░",
    }

    for label, group in zip(labels, thirds):
        if not group:
            continue
        avg = sum(s.score for s in group) / len(group)
        dominant = max(
            ["positive", "negative", "neutral"],
            key=lambda l: sum(1 for s in group if s.label == l)
        )
        # Build a simple bar (scale -1..1 to 0..20 chars)
        bar_length = int((avg + 1) / 2 * 20)
        bar = bar_chars[dominant] * bar_length

        start_ts = format_timestamp(group[0].start)
        end_ts   = format_timestamp(group[-1].start)
        print(f"  {label} [{start_ts}-{end_ts}]  {bar:<20}  {avg:+.3f}  {dominant}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Swap this URL for any YouTube video you want to analyze
    TEST_URL = "https://www.youtube.com/watch?v=qp0HIF3SfI4"

    run_pipeline(
        url=TEST_URL,
        summary_sentences=5,
        summary_algorithm="lsa",
        show_all_segments=False,
        save_json=True,
    )