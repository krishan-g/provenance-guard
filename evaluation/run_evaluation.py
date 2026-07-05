"""
Larger-scale calibration test: 50 genuinely human texts (pulled from diverse
public-domain sources) and 50 genuinely AI-generated texts (spanning the
platform's target content types: blog posts, short stories, poems, essays,
informational writing), run through the actual production pipeline
(signals.py + scoring.py) to measure real accuracy, not just performance on
the 4 examples from the assignment.

Run from the repo root with the venv active:
    python evaluation/run_evaluation.py

Note: AI text generation uses temperature=0.9 and Groq does not guarantee
seeded reproducibility, so re-running this script will draw a fresh sample
of AI texts each time (same prompts, different completions). The human text
sample is deterministic (fixed random seed) modulo which Gutenberg sources
are reachable at fetch time. See results.md for the actual results this
script produced when it was run to validate design decisions in planning.md.
"""

import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from scoring import classify, compute_confidence
from signals import GROQ_MODEL, _get_client, get_llm_score, get_style_score

random.seed(42)

GUTENBERG_SOURCES = {
    "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",    # formal narrative/dialogue
    "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",        # dialogue-heavy fiction
    "meditations": "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",       # formal philosophy
    "roundabout_papers": "https://www.gutenberg.org/cache/epub/1462/pg1462.txt",  # personal essay
    "shropshire_lad": "https://www.gutenberg.org/cache/epub/5720/pg5720.txt",    # poetry
}

AI_PROMPT_TOPICS = {
    "blog": [
        "learning to cook", "starting a garden", "adopting a rescue dog", "training for a 5k",
        "moving to a new city", "trying meditation", "learning guitar", "decluttering my apartment",
        "a recent hiking trip", "getting into pottery",
    ],
    "story": [
        "a mystery involving a missing painting", "a sci-fi story about a lonely space station",
        "a quiet small-town drama", "an adventure in a jungle ruin", "a romance at a train station",
        "a ghost story in an old house", "a heist gone wrong", "a coming-of-age summer",
        "a detective's first big case", "a survival story after a storm",
    ],
    "poem": [
        "autumn leaves falling", "a city at night", "losing a childhood friend", "the ocean at dawn",
        "a grandmother's kitchen", "waiting for a letter", "a forest after rain", "growing older",
        "a train journey", "the first snow",
    ],
    "essay": [
        "the impact of social media on attention spans", "whether remote work improves quality of life",
        "the ethics of gene editing", "the future of public transportation",
        "whether standardized testing is fair", "the role of art in society",
        "the tradeoffs of nuclear energy", "how cities can adapt to climate change",
        "the value of a liberal arts education", "whether privacy is dead in the digital age",
    ],
    "info": [
        "how photosynthesis works", "the history of the printing press", "how vaccines work",
        "the causes of the fall of Rome", "how black holes form", "the basics of compound interest",
        "how the immune system fights infection", "the history of jazz music",
        "how tectonic plates move", "the basics of how the internet routes data",
    ],
}

PROMPT_TEMPLATES = {
    "blog": "Write a short, casual first-person blog post paragraph (3-4 sentences) about {topic}.",
    "story": "Write the opening paragraph (3-4 sentences) of a short story about {topic}.",
    "poem": "Write a short poem (4-6 lines) about {topic}.",
    "essay": "Write a short formal essay paragraph (3-4 sentences) arguing a position on {topic}.",
    "info": "Write a short informational paragraph (3-4 sentences) explaining {topic}.",
}

DECOMPOSED_PROMPT = """Assess two independent properties of this text.

1. formality: how formal/careful the register is (0.0 = very casual/colloquial, 1.0 = very formal/academic).
2. genericness: how generic and non-distinctive the writing is, independent of formality (0.0 = highly distinctive voice, specific idiosyncratic detail, or irregular phrasing; 1.0 = generic phrasing, no distinct voice, could have been written by anyone about anything).

A text can be formal AND distinctive (e.g. a specific, well-argued academic paragraph), or formal AND generic (e.g. corporate boilerplate). Judge them independently.

Respond with ONLY a JSON object in this exact format, no other text:
{"formality": <float 0.0-1.0>, "genericness": <float 0.0-1.0>, "rationale": "<one sentence>"}
"""


def fetch(url, retries=2):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception:
            if attempt == retries - 1:
                raise


def extract_paragraphs(text, min_words=30, max_words=150):
    start = text.find("*** START")
    end = text.find("*** END")
    if start != -1:
        text = text[text.find("\n", start) + 1:]
    if end != -1:
        text = text[:end]
    paras = re.split(r"\n\s*\n", text)
    out = []
    for p in paras:
        p = " ".join(p.split())
        wc = len(p.split())
        if min_words <= wc <= max_words and not p.isupper():
            out.append(p)
    return out


def collect_human_texts(n=50):
    pool = []
    for name, url in GUTENBERG_SOURCES.items():
        try:
            raw = fetch(url)
            paras = extract_paragraphs(raw)
            random.shuffle(paras)
            for p in paras[:15]:
                pool.append({"text": p, "source": name})
        except Exception as e:
            print(f"WARN: failed to fetch {name}: {e}", file=sys.stderr)
    random.shuffle(pool)
    return pool[:n]


def collect_ai_texts(client):
    texts = []
    for genre, topics in AI_PROMPT_TOPICS.items():
        for topic in topics:
            prompt = PROMPT_TEMPLATES[genre].format(topic=topic)
            try:
                completion = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                )
                texts.append({"text": completion.choices[0].message.content.strip(), "genre": genre})
            except Exception as e:
                print(f"WARN: generation failed ({genre}/{topic}): {e}", file=sys.stderr)
    return texts


def get_genericness(client, text):
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": DECOMPOSED_PROMPT}, {"role": "user", "content": text}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return float(json.loads(completion.choices[0].message.content)["genericness"])


def run_both(client, text, actual_label, genre):
    llm = get_llm_score(text)
    style = get_style_score(text)
    generic = get_genericness(client, text)

    conf_orig = compute_confidence(llm["ai_score"], style["style_score"])
    conf_generic = compute_confidence(generic, style["style_score"])

    return {
        "actual": actual_label,
        "genre": genre,
        "llm_score": llm["ai_score"],
        "genericness": generic,
        "style_score": style["style_score"],
        "conf_orig": conf_orig,
        "pred_orig": classify(conf_orig),
        "conf_generic": conf_generic,
        "pred_generic": classify(conf_generic),
        "text": text,
    }


def report(results, label, pred_key):
    print(f"\n=== {label} ===")
    matrix = Counter((r["actual"], r[pred_key]) for r in results)
    for actual in ["human", "ai"]:
        total = sum(v for k, v in matrix.items() if k[0] == actual)
        print(f"  {actual:6s} (n={total}):", end=" ")
        for pred in ["likely_human", "uncertain", "likely_ai"]:
            print(f"{pred}={matrix.get((actual, pred), 0)}", end="  ")
        print()
    fp = matrix.get(("human", "likely_ai"), 0)
    fn = matrix.get(("ai", "likely_human"), 0)
    human_total = sum(v for k, v in matrix.items() if k[0] == "human") or 1
    ai_total = sum(v for k, v in matrix.items() if k[0] == "ai") or 1
    print(f"  FP rate: {fp}/{human_total} = {fp/human_total*100:.1f}%   FN rate: {fn}/{ai_total} = {fn/ai_total*100:.1f}%")
    correct = matrix.get(("human", "likely_human"), 0) + matrix.get(("ai", "likely_ai"), 0)
    print(f"  Confident+correct: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")


def main():
    client = _get_client()

    human_texts = collect_human_texts(50)
    print(f"Collected {len(human_texts)} human texts")

    ai_texts = collect_ai_texts(client)
    print(f"Generated {len(ai_texts)} AI texts")

    results = []
    for item in human_texts:
        try:
            results.append(run_both(client, item["text"], "human", item["source"]))
        except Exception as e:
            print(f"WARN: classify failed (human): {e}", file=sys.stderr)
    for item in ai_texts:
        try:
            results.append(run_both(client, item["text"], "ai", item["genre"]))
        except Exception as e:
            print(f"WARN: classify failed (ai): {e}", file=sys.stderr)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_raw.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal classified: {len(results)}")
    report(results, "ORIGINAL (llm_score)", "pred_orig")
    report(results, "GENERICNESS (decomposed, tested and rejected)", "pred_generic")

    print("\n=== AI results by genre, both methods ===")
    ai_results = [r for r in results if r["actual"] == "ai"]
    by_genre = {}
    for r in ai_results:
        by_genre.setdefault(r["genre"], {"orig": [], "generic": []})
        by_genre[r["genre"]]["orig"].append(r["pred_orig"])
        by_genre[r["genre"]]["generic"].append(r["pred_generic"])
    for genre, d in by_genre.items():
        print(f"  {genre:8s} orig={Counter(d['orig'])}  generic={Counter(d['generic'])}")


if __name__ == "__main__":
    main()
