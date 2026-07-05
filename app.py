import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

from labels import get_label
from scoring import classify, compute_confidence
from signals import get_llm_score, get_style_score
from storage import (
    get_analytics,
    get_recent_entries,
    get_submission,
    init_db,
    insert_submission,
    record_appeal,
)

app = Flask(__name__)
init_db()

# See README's Rate Limiting section for the full reasoning. Short version:
# 6/minute bounds realistic single-session iteration (submit, read, revise,
# resubmit); 40/day bounds how much of the shared Groq free-tier daily token
# quota one user can consume.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.route("/submit", methods=["POST"])
@limiter.limit("6 per minute;40 per day")
def submit():
    """Runs both detection signals, combines them into a confidence score, and logs the result.

    No error handling around get_llm_score yet; a Groq outage or rate limit
    currently surfaces as a raw 500 (server error).
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    creator_id = data.get("creator_id")
    if not text or not creator_id:
        return jsonify({"error": "text and creator_id are required"}), 400

    content_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    llm_result = get_llm_score(text)
    llm_score = llm_result["ai_score"]

    style_result = get_style_score(text)
    style_score = style_result["style_score"]

    confidence = compute_confidence(llm_score, style_score)
    attribution = classify(confidence)
    label = get_label(attribution)

    insert_submission(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "text": text,
            "timestamp": timestamp,
            "llm_score": llm_score,
            "style_score": style_score,
            "confidence": confidence,
            "attribution": attribution,
            "label": label,
            "status": "classified",
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "confidence": confidence,
            "label": label,
        }
    )


@app.route("/appeal", methods=["POST"])
def appeal():
    """Records a creator's appeal against an existing classification.

    Sets status to under_review and logs the reasoning alongside the
    original decision; does not trigger re-classification.
    """
    data = request.get_json(silent=True) or {}
    content_id = data.get("content_id")
    creator_reasoning = data.get("creator_reasoning")
    if not content_id or not creator_reasoning:
        return jsonify({"error": "content_id and creator_reasoning are required"}), 400

    if get_submission(content_id) is None:
        return jsonify({"error": "content_id not found"}), 404

    timestamp = datetime.now(timezone.utc).isoformat()
    record_appeal(content_id, creator_reasoning, timestamp)

    return jsonify(
        {
            "content_id": content_id,
            "status": "under_review",
            "message": "Appeal received and logged for review.",
        }
    )


@app.route("/log", methods=["GET"])
def get_log():
    """Returns the audit log's most recent entries as JSON."""
    return jsonify({"entries": get_recent_entries()})


ANALYTICS_TEMPLATE = """
<!doctype html>
<title>Provenance Guard — Analytics</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 3rem auto; color: #222; }
  h1 { font-size: 1.4rem; }
  .stat { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #eee; }
  .stat span:last-child { font-weight: 600; }
  table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
  td { padding: 0.4rem 0; border-bottom: 1px solid #eee; }
  td:last-child { text-align: right; font-weight: 600; }
</style>
<h1>Provenance Guard — Analytics</h1>

<div class="stat"><span>Total submissions</span><span>{{ a.total_submissions }}</span></div>
<div class="stat"><span>Appeals filed</span><span>{{ a.appeal_count }} ({{ "%.1f"|format(a.appeal_rate * 100) }}%)</span></div>
<div class="stat"><span>Average confidence</span><span>{{ "%.3f"|format(a.avg_confidence) }}</span></div>
<div class="stat"><span>Average signal disagreement</span><span>{{ "%.3f"|format(a.avg_signal_disagreement) }}</span></div>

<h2>Detection patterns</h2>
<table>
  {% for tier in ["likely_ai", "uncertain", "likely_human"] %}
  <tr><td>{{ tier }}</td><td>{{ a.by_attribution.get(tier, 0) }}</td></tr>
  {% endfor %}
</table>
"""


@app.route("/analytics", methods=["GET"])
def analytics():
    """Simple server-rendered dashboard: detection patterns, appeal rate, and
    average signal disagreement (see planning.md's Stretch Feature section)."""
    return render_template_string(ANALYTICS_TEMPLATE, a=get_analytics())


if __name__ == "__main__":
    app.run(debug=True)
