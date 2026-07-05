import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

from signals import get_llm_score
from storage import get_recent_entries, init_db, insert_submission

app = Flask(__name__)
init_db()


@app.route("/submit", methods=["POST"])
def submit():
    """Runs Groq LLM classification on submitted text and logs the result.

    Stylometric signal and the real agreement-gated confidence formula from 
    planning.md need to be implemented; confidence/attribution/label below are placeholders
    until then. No error handling around get_llm_score yet — a Groq outage or rate limit 
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

    confidence = llm_score
    attribution = "likely_ai" if confidence >= 0.5 else "likely_human"
    label = "Placeholder label — confidence scoring not yet implemented (Milestone 4)."

    insert_submission(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "text": text,
            "timestamp": timestamp,
            "llm_score": llm_score,
            "style_score": None,
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


@app.route("/log", methods=["GET"])
def get_log():
    """Returns the audit log's most recent entries as JSON."""
    return jsonify({"entries": get_recent_entries()})


if __name__ == "__main__":
    app.run(debug=True)
