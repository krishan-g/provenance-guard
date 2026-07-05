import math

# Clip probabilities before taking the logit so no single signal's own
# confidence (e.g. llm_score=1.0) can dominate the combination unbounded.
# 0.15 was chosen because it caps calib-borderline-formal-human's confidence
# at 0.667 regardless of how high llm_score goes — see planning.md section 2
# for the calibration testing that motivated this specific value.
_LOGIT_CLIP = 0.15


def _logit(p):
    p = min(max(p, _LOGIT_CLIP), 1 - _LOGIT_CLIP)
    return math.log(p / (1 - p))


def compute_confidence(llm_score, style_score):
    """Log-odds combination from planning.md section 2.

    Each signal is converted to log-odds, combined as a weighted sum, then
    converted back to a probability. LLM is weighted higher (0.7) than
    stylometrics (0.3) since Signal 1 is a more sophisticated,
    semantically-aware classifier; stylometrics acts as a corroborating
    structural check rather than an equal partner. Clipping before the logit
    (see _LOGIT_CLIP) bounds how much either signal can dominate alone.
    """
    combined_logit = 0.7 * _logit(llm_score) + 0.3 * _logit(style_score)
    return 1 / (1 + math.exp(-combined_logit))


def classify(confidence):
    """Asymmetric thresholds from planning.md section 2."""
    if confidence >= 0.70:
        return "likely_ai"
    if confidence <= 0.35:
        return "likely_human"
    return "uncertain"
