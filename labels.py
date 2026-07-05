LABELS = {
    "likely_ai": (
        "⚠️ Likely AI-Generated — Our analysis found strong, consistent signals "
        "associated with AI-generated text. This is an automated assessment, "
        "not a certainty. If you wrote this yourself, you can appeal this "
        "classification."
    ),
    "likely_human": (
        "✅ Likely Human-Written — Our analysis found writing patterns "
        "consistent with human authorship, with no strong signals of AI "
        "generation."
    ),
    "uncertain": (
        "❓ Uncertain — Our analysis could not confidently determine whether "
        "this content is AI-generated or human-written; the signals were "
        "mixed. This is not a mark against your content, and no action is "
        "taken automatically."
    ),
}


def get_label(attribution):
    """Maps a classification tier (scoring.classify's output) to its exact
    transparency label text."""
    return LABELS[attribution]
