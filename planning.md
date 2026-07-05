# Provenance Guard — Planning

## 1. Detection Signals

Two independent signals, one semantic and one structural, are combined into a single confidence score.

### Signal 1 — Groq LLM classification (semantic)

- **What it measures**: holistic coherence and stylistic naturalness. The model is prompted to assess whether the text reads as human- or AI-written, drawing on tone, idiom, generic phrasing, and an overly balanced, non-committal structure.
- **Output format**: a float `llm_score` in `[0, 1]`, where `1.0` = confidently AI-generated, `0.0` = confidently human-written. Requested from the model as structured JSON (`{"ai_score": 0.0-1.0, "rationale": "..."}`).
- **Blind spot**: biased against non-native English speakers and very formal human writers, whose careful, hedge-heavy prose resembles LLM output (see [Edge Cases](#5-anticipated-edge-cases)). Also weak against lightly-edited AI text, since a few human rewrites break the fluency "tell" the model is looking for.

> **Update during calibration testing**: the prompt originally listed specific hedge-phrase examples ("it is important to note," "on the other hand") as things to look for. These were removed once the connector-phrase sub-metric below was added to Signal 2; counting fixed phrases is mechanical, not semantic, so it belongs in the structural signal, not duplicated in the LLM prompt.

### Signal 2 — Stylometric heuristics (structural)

- **What it measures**: statistical regularity, computed with no semantic understanding, from three sub-metrics:
  1. **Sentence length variance** — population variance of word-count per sentence. Consistent sentence lengths reduce the natural rhythm shifts of unscripted writing; human writing tends to alternate between short and long sentences within the same passage.
  2. **Punctuation density** — count of commas/semicolons/colons per sentence. Higher, more uniform punctuation density is treated as more AI-like due to the formulaic clause structure.
  3. **Connector-phrase density** — occurrences per 100 words of a curated list of commonly-cited LLM connector/hedge phrases ("furthermore," "it is important to note," "delve into," etc.). This is mechanical substring matching, not semantic judgment, which is why it lives here rather than in Signal 1's prompt (see note above). Not exhaustive — a text avoiding this exact vocabulary evades it, which is a known limitation of this sub-metric specifically.

  **Calibration history** (full writeup in README): the original pre-implementation design used sentence length variance, type-token ratio (TTR), and punctuation density. Testing against the assignment's four calibration texts surfaced two findings: (1) a clearly AI-generated, formally-written paragraph produced nearly the same signal signature as a genuinely human, formally-written paragraph — high `llm_score`, low `style_score` on both — which motivated adding connector-phrase density; (2) TTR's sub-score stayed flat (0.10-0.14) across all four texts regardless of authorship, contributing no discriminating information, so it was dropped rather than kept as dead weight.
- **Output format**: each sub-metric is normalized to `[0, 1]` "AI-likeness" and averaged into a single float `style_score`:
  - `variance_subscore = 1 - min(variance, 40) / 40`
  - `punct_subscore = min(commas_per_sentence, 3) / 3`
  - `connector_subscore = min(connectors_per_100_words, 3) / 3`
  - `style_score = mean(variance_subscore, punct_subscore, connector_subscore)`
  - These normalization constants were validated against the assignment's four calibration texts; see README for the full results, including the formal-AI-vs-formal-human case that no amount of constant-tuning fully resolved (it's a property of the underlying data, not the normalization).
- **Blind spot**: unreliable on very short submissions (too few sentences for variance to mean anything), and penalizes writers with a genuinely disciplined, low-variance style — e.g. dialogue-heavy writing (see [Edge Cases](#5-anticipated-edge-cases)). The connector-phrase sub-metric only catches AI text that happens to use its specific phrase list.

## 2. Uncertainty Representation

`confidence` is a single float in `[0, 1]` on one continuous axis: `0.0` = confidently human-written, `1.0` = confidently AI-generated, `0.5` = maximum uncertainty (no lean either way).

**Combination — log-odds.** Each signal is converted to log-odds (logit), combined as a weighted sum, then converted back to a probability via the sigmoid function. This is a standard technique for combining independent probabilistic signals, more principled than a hand-picked weighted average when the signals are assumed independent.

```
logit(p)      = ln(p / (1 - p)), with p clipped to [0.15, 0.85] first
combined_logit = 0.7 * logit(llm_score) + 0.3 * logit(style_score)
confidence     = 1 / (1 + e^-combined_logit)
```

- LLM is weighted higher (0.7) because it's a more sophisticated, semantically-aware classifier; stylometrics (0.3) acts as a corroborating structural check rather than an equal partner.
- The `0.15` clip is load-bearing, not decorative: without it, a single signal reporting near-certainty (e.g. `llm_score = 0.99`) can dominate the combination regardless of what the other signal says — the opposite of the intended behavior. With the clip, a signal's contribution saturates once its raw score passes ~0.85, capping how much it alone can move the result.

**Calibration history** (full writeup in README): this replaced an earlier "agreement-gated dampening" formula (a weighted average pulled toward 0.5 in proportion to signal disagreement). Dampening was replaced after testing against 4 fresh, previously-unused texts (2 genuinely AI, 2 genuinely human, sourced independently of the original 4 calibration texts) showed log-odds correctly classified two clearly-AI texts that dampening left stuck at "uncertain," with no new false positives introduced on any human text. Plain log-odds (no clip) was tested first and rejected — it let `calib-borderline-formal-human` cross into a false "likely_ai" if `llm_score` varied just slightly higher (0.92 vs 0.80), because unclipped log-odds lets one confident signal dominate. The `0.15` clip closes that gap: verified that `calib-borderline-formal-human`'s confidence plateaus at 0.667 and cannot cross 0.70 regardless of how high `llm_score` goes.

**Thresholds — asymmetric, conservative on the "likely AI" side:**

| Confidence range | Label tier | Distance from center (0.5) |
|---|---|---|
| `≥ 0.70` | Likely AI | 0.20 |
| `0.36 – 0.69` | Uncertain | — |
| `≤ 0.35` | Likely human | 0.15 |

Reaching "likely AI" requires a larger swing from center (0.20) than reaching "likely human" does (0.15) (since clearing someone should take less evidence than accusing them).

## 3. Transparency Label Design

Exact text returned by the API, keyed by confidence tier. Note that the labels should not assert certainty, since the scores are probabilistic estimates.

| Tier | Label text |
|---|---|
| **High-confidence AI** (`confidence ≥ 0.70`) | "⚠️ Likely AI-Generated — Our analysis found strong, consistent signals associated with AI-generated text. This is an automated assessment, not a certainty. If you wrote this yourself, you can appeal this classification." |
| **High-confidence human** (`confidence ≤ 0.35`) | "✅ Likely Human-Written — Our analysis found writing patterns consistent with human authorship, with no strong signals of AI generation." |
| **Uncertain** (`0.36 – 0.69`) | "❓ Uncertain — Our analysis could not confidently determine whether this content is AI-generated or human-written; the signals were mixed. This is not a mark against your content, and no action is taken automatically." |

## 4. Appeals Workflow

- **Who can appeal**: the original creator, identified by submitting the `content_id` from their `/submit` response.
- **What they provide**: `content_id` and free-text `creator_reasoning` explaining why they believe the classification is wrong.
- **What the system does**: looks up the existing record by `content_id`, sets its `status` to `"under_review"`, and appends the appeal (`creator_reasoning`, appeal timestamp) to that record's audit log entry. No automated re-classification is triggered.
- **What a human reviewer would see**: querying the log/DB for `status = "under_review"` surfaces, per appeal: the original submitted text, `llm_score`, `style_score`, combined `confidence`, the label that was shown, and the creator's `creator_reasoning`.

## 5. Anticipated Edge Cases

1. **Formal writing by non-native English speakers.** Signal 1 (LLM) is prone to reading careful, hedge-heavy, grammatically conservative prose as AI-generated — a documented bias in perplexity/LLM-based detectors (Liang et al., Stanford, 2023, found significantly higher false-positive rates on non-native English writing). This is the primary scenario the asymmetric thresholds and the log-odds clip (see [Uncertainty Representation](#2-uncertainty-representation)) are designed to protect against.
2. **Dialogue-heavy fiction or interview transcripts.** Signal 2 (stylometrics) reads short, clipped dialogue lines and heavy quotation-mark usage as anomalies — sentence length variance collapses (many short lines back to back) and punctuation density spikes — even though this is a normal structural feature of conversational writing, not a marker of AI generation.
3. **Very short submissions** (roughly under 40–50 words). Sentence-length variance is statistically unstable with only one or two sentences to measure — the stylometric signal effectively becomes noise, so the combined score leans more heavily on the LLM signal for short text.
4. **AI-generated text written in a casual, first-person voice.** Confirmed at scale, not just anticipated: a 100-example evaluation (50 real human texts, 50 fresh AI texts spanning the platform's target content types) found a 32% false-negative rate concentrated almost entirely in two genres — blog posts (9/10 missed) and personal-narrative story openings (7/10 missed) — while poems, essays, and informational writing had zero misses across 30 examples. Every miss in the entire test starts with "I." This lands squarely on the platform's actual target content, so it's a meaningful, quantified gap, not a hypothetical edge case. A tested fix (decomposing Signal 1's judgment into separate `formality`/`genericness` scores) fully resolved the blog-post cases but badly degraded the previously-strong poem/essay/info categories — net negative, and rejected. See `evaluation/results.md` for the full methodology and numbers.

## Architecture

### Diagram

```
SUBMISSION FLOW
────────────────
Client
  │  POST /submit {text, creator_id}
  ▼
Flask-Limiter ──(over limit)──► 429 response
  │ (ok)
  ▼
content_id = uuid()
  │
  ├──► Signal 1: Groq LLM ──────► llm_score (0-1)
  │
  └──► Signal 2: Stylometrics ──► style_score (0-1)
  │
  ▼
Confidence Scorer (agreement-gated combination)
  │
  ▼  confidence (0-1)
Label Generator (threshold lookup) ──► label text
  │
  ▼
Audit Logger (SQLite) ──► writes row (content_id, scores, label, status="classified")
  │
  ▼
Response {content_id, attribution, confidence, label} ──► Client


APPEAL FLOW
───────────
Client
  │  POST /appeal {content_id, creator_reasoning}
  ▼
Lookup content_id in SQLite
  │
  ▼
Status Updater ──► status = "under_review"
  │
  ▼
Audit Logger ──► updates row with creator_reasoning + status
  │
  ▼
Response {content_id, status: "under_review"} ──► Client
```

### Narrative

A submission is rate-limited, assigned a `content_id`, then scored independently by the Groq LLM signal and the stylometric signal; the two scores are combined through agreement-gated dampening into one confidence value, which a threshold lookup converts into one of three transparency labels, all recorded in a SQLite audit log before the response returns. An appeal, submitted against an existing `content_id`, looks up that same audit record, flips its status to `under_review`, and appends the creator's reasoning to the log — no re-scoring occurs.

## AI Tool Plan

- **submission endpoint + first signal**: Provide the AI tool the [Detection Signals](#1-detection-signals) section (Signal 1 only) and the Architecture diagram. Ask for: a Flask app skeleton with the `POST /submit` route stub, plus the Groq signal function matching the `{"ai_score": float, "rationale": str}` output format. Verify by calling the signal function directly against 2–3 test inputs and inspecting that scores fall in `[0, 1]` and move in the right direction before wiring it into the route.

- **second signal + confidence scoring**: Provide the [Detection Signals](#1-detection-signals) section (Signal 2) and the [Uncertainty Representation](#2-uncertainty-representation) section, plus the diagram. Ask for: the stylometric signal function (the three sub-metrics + normalization formulas above) and the confidence-scoring function implementing the exact `raw_combined` / `disagreement` / `confidence` formulas. Verify by diffing the generated constants against this document (0.6/0.4 weights, the exact dampening formula), then run the four calibration texts and confirm each confidence value lands in the threshold band predicted here, not just that the ordering looks roughly right.

- **production layer**: Provide the [Transparency Label Design](#3-transparency-label-design) and [Appeals Workflow](#4-appeals-workflow) sections, plus the diagram. Ask for: a label-generation function mapping `confidence` to the exact label text above, and the `POST /appeal` endpoint. Verify by calling the label function with confidence values in all three tiers and diffing the output against this document's exact strings, and by testing that an appeal flips `status` to `under_review` and shows up correctly in `GET /log`.
