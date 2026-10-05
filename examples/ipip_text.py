"""Score example narratives using IPIP-50 and a local embedding model.

Run: python examples/ipip_text.py --model /path/to/local/embedding-model
The inventory is real; these narratives are illustrative, not participant data.
"""
import argparse
import xpsych as xp
from xpsych.text import EmbeddingScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local SentenceTransformer model directory")
    args = parser.parse_args()
    bank = xp.instruments.ipip50()
    scorer = EmbeddingScorer.from_pretrained(args.model)
    texts = [
        "I spent the evening chatting with people I had just met and introducing friends to each other.",
        "I prefer a quiet evening alone. At the gathering I mostly listened and stayed in the background.",
        "I made a plan for the week, finished my chores, and checked the details before handing in my work.",
    ]
    scores = xp.compass(texts, bank, scorer=scorer)
    for row in range(len(texts)):
        print(texts[row])
        print({name: round(float(values[row]), 3) for name, values in scores.items()})
    print("These are signed text-activity scores, not questionnaire ratings or personality diagnoses.")


if __name__ == "__main__":
    main()
