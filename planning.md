# Provenance Guard — Planning

## 1. Detection Signals

Two independent signals, one semantic and one structural, are combined into a single confidence score.

### Signal 1 — Groq LLM classification (semantic)

- **What it measures**: holistic coherence and stylistic naturalness. The model is prompted to assess whether the text reads as human- or AI-written, drawing on tone, idiom, and the kind of generic-fluency/hedging patterns ("it is important to note," balanced "on the other hand" framing) that are common in LLM output.
- **Output format**: a float `llm_score` in `[0, 1]`, where `1.0` = confidently AI-generated, `0.0` = confidently human-written. Requested from the model as structured JSON (`{"ai_score": 0.0-1.0, "rationale": "..."}`).
- **Blind spot**: biased against non-native English speakers and very formal human writers, whose careful, hedge-heavy prose resembles LLM output (see [Edge Cases](#5-anticipated-edge-cases)). Also weak against lightly-edited AI text, since a few human rewrites break the fluency "tell" the model is looking for.

### Signal 2 — Stylometric heuristics (structural)

- **What it measures**: statistical regularity, computed with no semantic understanding, from three sub-metrics:
  1. **Sentence length variance** — population variance of word-count per sentence. Consistent sentence lengths reduce the natural rhythm shifts of unscripted writing; human writing tends to alternate between short and long sentences within the same passage.
  2. **Type-token ratio (TTR)** — unique words / total words (case-insensitive, punctuation stripped). Lower TTR (more repetitive vocabulary) is treated as more AI-like, on the assumption that generated text draws on a smaller working vocabulary across a passage than unscripted human writing does.
  3. **Punctuation density** — count of commas/semicolons/colons per sentence. Higher, more uniform punctuation density is treated as more AI-like due to the formulaic clause structure.
- **Output format**: each sub-metric is normalized to `[0, 1]` "AI-likeness" and averaged into a single float `style_score`:
  - `variance_subscore = 1 - min(variance, 40) / 40`
  - `ttr_subscore = 1 - ttr`
  - `punct_subscore = min(commas_per_sentence, 3) / 3`
  - `style_score = mean(variance_subscore, ttr_subscore, punct_subscore)`
  - These normalization constants are starting points, which will later be validated and adjusted against calibration texts.
- **Blind spot**: unreliable on very short submissions (too few sentences for variance/TTR to mean anything), and penalizes writers with a genuinely disciplined, low-variance style — e.g. dialogue-heavy writing (see [Edge Cases](#5-anticipated-edge-cases)).

## 2. Uncertainty Representation

`confidence` is a single float in `[0, 1]` on one continuous axis: `0.0` = confidently human-written, `1.0` = confidently AI-generated, `0.5` = maximum uncertainty (no lean either way).

**Combination — agreement-gated dampening.** A weighted average is computed first, then pulled toward `0.5` in proportion to how much the two signals disagree. A single strong signal can't push the result to a confident verdict on its own; both signals have to point the same way, which matters because a false accusation (human work flagged as AI) costs more than a missed detection.

```
raw_combined  = 0.6 * llm_score + 0.4 * style_score
disagreement  = abs(llm_score - style_score)
confidence    = raw_combined + (0.5 - raw_combined) * disagreement
```

- When the signals agree (`disagreement ≈ 0`), `confidence ≈ raw_combined` — the weighted average is trusted.
- When the signals sharply disagree (`disagreement → 1`), `confidence → 0.5` regardless of what either signal said alone.
- LLM is weighted higher (0.6) because it captures more of the semantic signal humans actually focus on; stylometrics (0.4) acts as a structural check on it.

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

1. **Formal writing by non-native English speakers.** Signal 1 (LLM) is prone to reading careful, hedge-heavy, grammatically conservative prose as AI-generated — a documented bias in perplexity/LLM-based detectors (Liang et al., Stanford, 2023, found significantly higher false-positive rates on non-native English writing). This is the primary scenario the asymmetric thresholds and dampening logic are designed to protect against.
2. **Dialogue-heavy fiction or interview transcripts.** Signal 2 (stylometrics) reads short, clipped dialogue lines and heavy quotation-mark usage as anomalies — sentence length variance collapses (many short lines back to back) and punctuation density spikes — even though this is a normal structural feature of conversational writing, not a marker of AI generation.
3. **Very short submissions** (roughly under 40–50 words). Sentence-length variance and TTR are statistically unstable with only one or two sentences to measure — the stylometric signal effectively becomes noise, so the combined score leans entirely on the LLM signal for short text.

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
