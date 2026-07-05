# Provenance Guard

An AI content provenance backend built with Flask, featuring multi-signal detection (LLM + stylometrics), calibrated confidence scoring, transparency labeling, and an appeals workflow.

**Demo**: https://www.loom.com/share/856bebba84cb445db263f78e0e5065ef

Full design rationale lives in [`planning.md`](planning.md); this README covers the implementation, the evidence behind the design decisions, and what was learned building it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the repo root (see `.env.example`):
```
GROQ_API_KEY=your-groq-api-key-here
```

Run the server:
```bash
python app.py
```

The API is served at `http://127.0.0.1:5000`.
> **Note:** Use `127.0.0.1`, not `localhost`; on macOS, `localhost` can resolve to `::1` and hit the AirPlay Receiver service, which also listens on port 5000.

## Architecture

A submission is rate-limited, assigned a `content_id`, then scored independently by two detection signals: a Groq LLM classification and a set of stylometric heuristics. The two scores are combined via log-odds into a single confidence value, which a threshold lookup converts into one of three transparency labels. Everything (both signal scores, the combined confidence, the label, and the classification status) is written to a SQLite audit log before the response returns. An appeal, submitted against an existing `content_id`, looks up that same audit record, flips its status to `under_review`, and appends the creator's reasoning (no re-scoring occurs). The full diagram is in `planning.md`'s Architecture section.

## Detection Signals

**Signal 1 — Groq LLM classification (semantic)**, in `signals.py`. Asks `llama-3.3-70b-versatile` to judge tone, idiom, and stylistic naturalness. Generic phrasing and an overly balanced, non-committal structure read as AI-like; a distinctive voice and specific idiosyncratic detail read as human. Chosen because it captures holistic, meaning-level judgment that no amount of surface statistics can. However, it's biased against non-native English speakers and formal academic writers, whose careful, hedge-heavy prose can resemble LLM output. This is documented in real research (Liang et al., Stanford, 2023) and is a factor that the rest of the system's design (asymmetric thresholds, the log-odds clip) is built around.

**Signal 2 — stylometric heuristics (structural)**, also in `signals.py`. Three sub-metrics, each pure Python with no external libraries: sentence-length variance, punctuation density, and connector-phrase density (a curated list of commonly-cited LLM phrases like "furthermore" and "it is important to note," counted mechanically rather than judged semantically). A fourth candidate metric, type-token ratio, was in the original design but was dropped after calibration testing showed it stayed flat (0.10–0.14) regardless of authorship across all four of the calibration texts; it was contributing noise, not signal. This signal's blind spot: it's unreliable on very short submissions (too few sentences for variance to mean anything), and it penalizes writers with a genuinely disciplined, low-variance style (dialogue-heavy fiction being the clearest example, since clipped, repetitive dialogue lines read as suspiciously uniform even though that's a normal feature of conversational writing).

These two signals fail in different places by construction. One is limited by what the model has learned to associate with AI phrasing, the other by what raw text statistics can capture, which is what the combination logic below is built around.

## Confidence Scoring

`confidence` is a single float in `[0, 1]`: `0.0` means confidently human-written, `1.0` means confidently AI-generated, `0.5` is maximum uncertainty. The two signals are combined via **log-odds**, in `scoring.py`:

```python
combined_logit = 0.7 * logit(llm_score) + 0.3 * logit(style_score)
confidence = sigmoid(combined_logit)
```

with each probability clipped to `[0.15, 0.85]` before the logit. LLM is weighted higher (0.7) because it's the more sophisticated, semantically-aware signal; stylometrics (0.3) acts as a corroborating check rather than an equal partner. Without the clip, a single signal reporting near-certainty can dominate the combination regardless of what the other signal says: testing found this let a formal human text (`llm_score` pushed to 0.92 in a hypothetical variant) cross into a false `likely_ai`, even though the stylometric signal correctly read it as human the whole time. With the clip, that same text's confidence plateaus at 0.667 and cannot cross the 0.70 threshold no matter how confident Signal 1 gets.

Thresholds are asymmetric on purpose: `confidence ≥ 0.70` → likely AI, `≤ 0.35` → likely human, otherwise uncertain. Reaching "likely AI" requires a larger swing from center (0.20) than reaching "likely human" does (0.15) given clearing someone should take less evidence than accusing them, since a false accusation costs more than a missed detection.

**Two example submissions, showing the actual scores** (from a live run against this exact codebase):

- **Higher confidence** — a clearly AI-generated paragraph about AI ethics (`llm_score = 0.9`, `style_score = 0.494`) → **`confidence = 0.770`** → `likely_ai`.
- **Lower confidence** — a genuinely human, formally-written paragraph about monetary policy (`llm_score = 0.8`, `style_score = 0.081`) → **`confidence = 0.611`** → `uncertain`. Signal 1 alone would have read this as fairly AI-like; the combination correctly pulls back from a confident verdict rather than accusing a formal human writer.

**Calibration testing.** The assignment provides four deliberately-chosen calibration texts spanning the confidence range. Full text included below. Results are from a live run against the current codebase.

1. **Clearly AI-generated** (expected: should score high)
   > "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment."

   `llm_score = 0.90`, `style_score = 0.494` → **`confidence = 0.770`** → `likely_ai` ✅

2. **Clearly human-written** (expected: should score low)
   > "ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after. my friend got the spicy version and said it was better. probably wont go back unless someone drags me there"

   `llm_score = 0.10`, `style_score = 0.000` → **`confidence = 0.150`** → `likely_human` ✅

3. **Borderline: formal human writing** (expected: should be borderline)
   > "The relationship between monetary policy and asset price inflation has been extensively studied in the literature. Central banks face a fundamental tension between their mandate for price stability and the unintended consequences of prolonged low interest rates on equity and real estate valuations."

   `llm_score = 0.80`, `style_score = 0.081` → **`confidence = 0.611`** → `uncertain` ✅ (Signal 1 alone reads this as fairly AI-like; the combination correctly pulls back rather than accusing a formal human writer)

4. **Borderline: lightly-edited AI** (expected: should be mid-range)
   > "I have been thinking a lot about remote work lately. There are genuine tradeoffs, flexibility and no commute on one side, isolation and blurred work-life boundaries on the other. Studies show productivity varies widely by individual and role type."

   `llm_score = 0.60`, `style_score = 0.222` → **`confidence = 0.477`** → `uncertain` ✅

**How this was validated at scale**: beyond these 4 texts, a 100-example evaluation (`evaluation/`) tested 50 real human texts (pulled from 5 public-domain sources spanning different eras/genres) against 50 fresh AI-generated texts (spanning the platform's actual target content types: blog posts, short stories, poems, essays, informational writing). Results: **0% false positive rate** across all 50 human texts, **32% false negative rate** on AI texts. However, that miss rate is not evenly distributed: see Known Limitations below, and `evaluation/results.md` for the full breakdown.

## Transparency Label

Exact text returned by the API (`labels.py`), keyed by confidence tier:

| Tier | Label text |
|---|---|
| **High-confidence AI** (`confidence ≥ 0.70`) | "⚠️ Likely AI-Generated — Our analysis found strong, consistent signals associated with AI-generated text. This is an automated assessment, not a certainty. If you wrote this yourself, you can appeal this classification." |
| **High-confidence human** (`confidence ≤ 0.35`) | "✅ Likely Human-Written — Our analysis found writing patterns consistent with human authorship, with no strong signals of AI generation." |
| **Uncertain** (`0.36 – 0.69`) | "❓ Uncertain — Our analysis could not confidently determine whether this content is AI-generated or human-written; the signals were mixed. This is not a mark against your content, and no action is taken automatically." |

## Appeals Workflow

`POST /appeal` accepts `{content_id, creator_reasoning}`, sets that submission's `status` to `under_review`, and logs the reasoning alongside the original decision. No automated re-classification occurs. Example, appealing the "uncertain" monetary-policy submission above:

```
POST /appeal
{"content_id": "5d71e043-...", "creator_reasoning": "I wrote this myself as part of an economics research paper. My academic writing style is naturally formal, but this is entirely my own analysis."}

→ {"content_id": "5d71e043-...", "status": "under_review", "message": "Appeal received and logged for review."}
```

## Rate Limiting

`6 per minute; 40 per day` on `POST /submit` (`app.py`, via Flask-Limiter). Reasoning:

- **6/minute** bounds realistic single-session usage (a writer submitting a piece, reading the result, revising, and resubmitting a couple of times) while making rapid scripted flooding impractical.
- **40/day** bounds how much of the *shared* Groq free-tier daily token quota one user can consume. During development, heavy testing exhausted Groq's entire 100,000 tokens/day limit for the account, taking the whole app offline for everyone until the quota reset. A generous-but-bounded per-user daily cap protects against one user (malicious or just enthusiastic) doing that to everyone else.

Evidence: 8 rapid requests against the live server, status codes in order:
```
200
200
200
429
429
429
429
429
```
(Only 3 succeeded here because 3 rate-limit slots were already used by demo submissions earlier in the same minute, which is consistent with the configured 6/minute limit.)

## Audit Log

Every submission and appeal is recorded in SQLite (`storage.py`), retrievable via `GET /log`. Three real entries from a live run, unedited except line-wrapping the `text` field for readability:

```json
{
  "content_id": "524fe0b9-52cb-4dc8-b788-b62ee98678de",
  "creator_id": "demo-user-1",
  "text": "Artificial intelligence represents a transformative paradigm shift in modern
           society. It is important to note that while the benefits of AI are numerous,
           it is equally essential to consider the ethical implications. Furthermore,
           stakeholders across various sectors must collaborate to ensure responsible
           deployment.",
  "llm_score": 0.9, "style_score": 0.494, "confidence": 0.770,
  "attribution": "likely_ai", "status": "classified",
  "appeal_reasoning": null
}
{
  "content_id": "5d71e043-e334-4ed3-ab05-fa7e3324fef2",
  "creator_id": "demo-user-2",
  "text": "The relationship between monetary policy and asset price inflation has been
           extensively studied in the literature. Central banks face a fundamental
           tension between their mandate for price stability and the unintended
           consequences of prolonged low interest rates on equity and real estate
           valuations.",
  "llm_score": 0.8, "style_score": 0.081, "confidence": 0.611,
  "attribution": "uncertain", "status": "under_review",
  "appeal_reasoning": "I wrote this myself as part of an economics research paper. My academic writing style is naturally formal, but this is entirely my own analysis."
}
{
  "content_id": "7000cfa9-6a8e-4f02-91d7-ef4781720801",
  "creator_id": "demo-user-3",
  "text": "ok so i finally tried that new ramen place downtown and honestly? underwhelming.
           the broth was fine but they put WAY too much sodium in it and i was thirsty
           for like three hours after.",
  "llm_score": 0.1, "style_score": 0.0, "confidence": 0.150,
  "attribution": "likely_human", "status": "classified",
  "appeal_reasoning": null
}
```

## Analytics Dashboard

`GET /analytics` renders a simple server-side HTML dashboard, computed directly from the existing SQLite audit log. Shows: total submissions, detection patterns (counts by `likely_ai` / `uncertain` / `likely_human`), appeal rate, average confidence, and average signal disagreement (`AVG(ABS(llm_score - style_score))`). It's read-only and admin-facing rather than creator-facing, so it isn't rate-limited, matching `GET /log`.

Example output after 3 submissions and 1 appeal:
```
Total submissions: 3
Appeals filed: 1 (33.3%)
Average confidence: 0.529
Average signal disagreement: 0.341

Detection patterns:
  likely_ai: 1
  uncertain: 1
  likely_human: 1
```

## Known Limitations

**AI-generated text written in a casual, first-person voice evades detection.** The 100-example evaluation confirmed this at scale. 32% of AI texts were missed overall, but that miss rate was almost entirely concentrated in two genres: blog posts (9/10 missed) and personal-narrative story openings (7/10 missed), while poems, formal essays, and informational writing had **zero** misses across 30 examples combined. Every single miss in the entire test started with "I." This lands directly on the platform's actual target content, so it's a real, quantified gap, not an edge case in the abstract sense.

A real example, which was fully AI-generated, freshly produced from the prompt "write a short, casual first-person blog post paragraph about learning to cook":

> "I'm still pretty new to cooking, but I've been trying to learn as much as I can in the kitchen. It's been a bit of a trial by fire (literally, in some cases - I've had my fair share of burnt dishes), but I'm slowly starting to get the hang of it. I've been experimenting with different recipes and techniques, and it's amazing how much more confident I feel with each successful meal. Now, I'm actually starting to enjoy cooking, and I love the feeling of being able to whip up a delicious meal from scratch."

`llm_score = 0.20`, `style_score = 0.344` → `confidence = 0.238` → **`likely_human`**: wrong; both signals confidently agreed on the wrong answer.

We tested a fix: decomposing Signal 1's judgment into separate `formality` and `genericness` scores, using only `genericness` as the AI signal (the theory being that AI text faking a casual voice still leans on generic, cliché phrasing that a genericness-specific judgment might catch). It completely fixed the blog-post cases (9 missed → 0), but it badly degraded three categories that were already working well (poems dropped from 8/10 confidently caught to 2/10, essays from 5/10 to 0/10, informational from 8/10 to 3/10). Net effect: worse overall. This fix is **not** in the shipped pipeline: see `evaluation/results.md` for the full numbers.

If deploying this for real, the next thing worth trying is a signal that specifically targets the personal-voice failure mode without the collateral damage genericness caused elsewhere; genericness conflates "sounds casual" with "sounds generic," and those aren't quite the same thing.

## Spec Reflection

**Where the spec helped**: writing out the exact three label variants and the confidence thresholds in `planning.md` *before* any code existed forced concrete answers to questions that would otherwise have stayed vague until much later. Specifically, deciding upfront that reaching "likely AI" should require more evidence than reaching "likely human" shaped the threshold design, the log-odds clip, and the eventual rejection of the genericness fix, all of which trace back to that one early asymmetry decision.

**Where implementation diverged**: `planning.md`'s original design used three stylometric sub-metrics (sentence variance, type-token ratio, punctuation density) combined via a dampening formula that pulled disagreeing signals toward 0.5. Both choices changed after empirical testing: TTR was dropped for contributing no discriminating signal, a connector-phrase sub-metric was added after discovering a formal-AI/formal-human confusion the original three metrics couldn't resolve, and the combination method was replaced with log-odds after dampening left two clearly-AI texts stuck at "uncertain" that log-odds correctly classified.

## AI Usage

This project was built with Claude Code. Specific instances of directing it, testing its output, and overriding it based on evidence:

1. **Building a large test and reversed supposed fix.** After confirming (via a small test) that AI-generated text mimicking a personal blog-post voice evaded detection, I directed Claude to design a fix, where we decomposed the LLM signal into separate formality/genericness judgments. An 8-example partial test (cut short by a Groq rate limit) showed a clean win and was initially recommended for adoption. I directed a full 100-example test across all five content genres before accepting that recommendation, which reversed it: the fix degraded three previously-strong categories (poems, essays, informational writing) by a wide margin to fix one narrow one. The initial recommendation was overridden once tested properly, and the fix was not shipped: see Known Limitations above.

2. **A confidence-scoring calibration.** I directed Claude to test log-odds combination as a replacement for the original dampening formula. An initial test showed a clean win: it correctly classified two AI texts that had been stuck at "uncertain," with no downside on any human text. Before shipping it, I asked for more adversarial testing, which found a real flaw: unclipped log-odds let a single overconfident signal dominate regardless of what the other signal said, so a formal human text could cross into a false `likely_ai` if Signal 1 happened to read just slightly more confident (0.92 instead of 0.80). Claude's fix, clipping probabilities to `[0.15, 0.85]` before the logit, was verified to close that gap before it was accepted, not assumed to work.
