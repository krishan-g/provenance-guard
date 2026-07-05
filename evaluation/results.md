# Evaluation Results — 100-Example Test

`planning.md`'s design decisions were validated against 4 texts (the assignment's own calibration set). This test checks whether those decisions hold up at a larger scale, across content types the 4 original examples didn't cover, and against text the system has never seen tuned against it.

**Method**: 50 genuinely human texts pulled from 5 public-domain sources (Jane Austen, Sherlock Holmes, Marcus Aurelius, a 19th-century personal essay collection, and A.E. Housman's poetry — spanning formal narrative, dialogue-heavy fiction, formal philosophy, personal voice, and poetry). 50 genuinely AI-generated texts, freshly generated via Groq across the platform's actual target content types: 10 each of blog posts, short story openings, poems, formal essays, and informational writing. Every text was run through the real production pipeline (`signals.py` + `scoring.py`), not a simulation. Raw data: `results_raw.json`. Script: `run_evaluation.py`.

## Headline results (current production pipeline — log-odds combination, 0.7/0.3 weights)

| | likely_human | uncertain | likely_ai |
|---|---|---|---|
| **Human** (n=50) | 48 | 2 | 0 |
| **AI** (n=50) | 16 | 12 | 22 |

- **False positive rate: 0%** — not one of 50 human texts, across 5 different genres and eras, was ever misclassified as `likely_ai`.
- **False negative rate: 32%** (16/50 AI texts wrongly read as `likely_human`).
- **Confident and correct: 70/100 (70%).**

## The false negatives are not evenly distributed

| AI content type | likely_ai | uncertain | likely_human (missed) |
|---|---|---|---|
| Poems | 8 | 2 | 0 |
| Formal essays | 5 | 5 | 0 |
| Informational | 8 | 2 | 0 |
| Short story openings | 1 | 2 | 7 |
| Blog posts | 0 | 1 | 9 |

The system is excellent at poems, essays, and informational writing — 21/30 caught outright, **zero** missed across all three categories combined. It is specifically and severely weak on first-person, casual writing: blog posts (9/10 missed) and personal-narrative story openings (7/10 missed). Every single miss in the entire 100-example test starts with "I." This is not a generic weakness — it's one specific, identifiable failure mode, and it lands on two of the three content types the assignment explicitly names as this platform's target audience ("a poem, a short story excerpt, a blog post").

## A tested-and-rejected fix: genericness decomposition

Given how concentrated this gap is, we tested an alternative to Signal 1: instead of asking the LLM for one holistic `ai_score`, ask it for two independent scores — `formality` and `genericness` — and use only `genericness` as the AI signal. The idea: AI text faking a casual voice still tends to lean on generic, cliché phrasing ("wild ride," "completely absorbed," "amazing journey") that a genericness-specific judgment might catch even when the holistic judgment is fooled by the personal tone.

| | likely_human | uncertain | likely_ai |
|---|---|---|---|
| **Human** (n=50) | 39 | 11 | 0 |
| **AI** (n=50) | 7 | 38 | 5 |

FP rate stayed at 0%, and FN rate dropped from 32% to 14% — but confident-and-correct dropped from 70% to 44%. By genre:

| Genre | Original (caught/uncertain/missed) | Genericness (caught/uncertain/missed) |
|---|---|---|
| Blog | 0 / 1 / 9 | 0 / **10** / 0 |
| Story | 1 / 2 / 7 | 0 / 3 / 7 |
| Poem | **8** / 2 / 0 | 2 / 8 / 0 |
| Essay | **5** / 5 / 0 | 0 / 10 / 0 |
| Info | **8** / 2 / 0 | 3 / 7 / 0 |

Genericness completely fixes blog posts (9 missed → 0 missed, all pushed safely to `uncertain`) — but doesn't touch story misses at all, and badly degrades three categories that were already working well (poems 8→2 confidently caught, essays 5→0, info 8→3). An earlier, smaller test (8 examples, all from the blog genre, before a Groq rate limit cut the run short) looked like a clean win — every example that got through happened to be from the one genre genericness helps. The full test reverses that conclusion: it trades a fix for one narrow, specific problem for broad degradation everywhere else. **Rejected** — the production pipeline keeps the original single `ai_score` from Signal 1, and the blog-post/personal-narrative gap is documented as an accepted, quantified limitation rather than something this particular fix is worth adopting for.

## How to reproduce

```
source .venv/bin/activate
python evaluation/run_evaluation.py
```

Note: AI generation uses `temperature=0.9` and Groq does not guarantee seeded reproducibility, so a fresh run will draw different specific AI texts each time (same prompts, different completions) — the exact numbers above will vary slightly run to run, but the genre pattern (strong on poems/essays/info, weak on blog/story) has reproduced consistently across two independent full runs.
