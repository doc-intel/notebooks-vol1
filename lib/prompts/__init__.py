"""Catalogue centralisé des prompts système (T1.20).

Chaque prompt clé du package est exposé ici sous une constante string +
la liste des variables attendues. Les commandes (qa, classify, redact,
translate, summarize) lisent leur prompt via `resolve_prompt(project_id,
prompt_id)` qui consulte d'abord les overrides projet puis retombe sur
le défaut.

L'éditeur Settings → Prompts de l'app Electron lit ce catalogue pour
afficher le diff défaut/override.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptDefinition:
    """Métadonnées d'un prompt système."""
    prompt_id: str
    description: str
    default: str
    required_variables: tuple[str, ...]


# Catalogue par défaut. Chaque entrée doit matcher un appel LLM réel
# côté `lib.generation.*` ou les bricks appelantes.
DEFAULT_PROMPTS: dict[str, PromptDefinition] = {
    "qa_answer_system": PromptDefinition(
        prompt_id="qa_answer_system",
        description="Q&A : système prompt qui force la réponse depuis les lignes fournies.",
        default=(
            "Answer using ONLY the provided lines. Return JSON only. "
            "If the answer is not in the lines, set fields to null and "
            "explain why in caveats."
        ),
        required_variables=(),
    ),
    "qa_fields_system": PromptDefinition(
        prompt_id="qa_fields_system",
        description="Q&A multi-champs : extraction de plusieurs champs en un appel, evidence par champ (un FieldItem par champ).",
        default=(
            "You extract structured fields from the provided lines. Return JSON only.\n"
            "\n"
            "For EACH requested field, populate a FieldItem with :\n"
            "- value : the extracted value as a string, or null when the field is "
            "not present in the lines.\n"
            "- spans : the SPECIFIC page+line ranges that support this field's value. "
            "Each span carries page_start, page_end, line_start, line_end (and "
            "optionally the verbatim quote). Different fields should have DIFFERENT "
            "spans — never share one global span across fields.\n"
            "- justification : a short sentence explaining why these lines support "
            "this field's value (or why the field could not be located).\n"
            "- confidence : 0.0–1.0.\n"
            "- answer_found : false when the value could not be located ; in that "
            "case set value=null, spans=[], confidence low.\n"
            "\n"
            "Output ONE FieldItem per requested field name. Use the exact field "
            "names the user lists."
        ),
        required_variables=(),
    ),
    "classify_one_system": PromptDefinition(
        prompt_id="classify_one_system",
        description="Classification : pick a type from the allowed hierarchy.",
        default=(
            "Pick the best matching type for the document. Use only types "
            "from the provided list. Return JSON with type, sub_type, "
            "confidence, reasoning, alternative_type."
        ),
        required_variables=("available_types",),
    ),
    "redact_detect_system": PromptDefinition(
        prompt_id="redact_detect_system",
        description="Anonymisation : detect sensitive spans line by line.",
        default=(
            "Identify sensitive spans (PII or commercial) in the provided "
            "line. Return JSON with category + (start, end) char offsets."
        ),
        required_variables=("categories",),
    ),
    "translate_system": PromptDefinition(
        prompt_id="translate_system",
        description="Translation : preserve technical terms + numbers, tone parameterised.",
        default=(
            "You translate the user text to {target_lang}. Tone: {tone}. "
            "Preserve technical terms, numbers, and proper nouns verbatim. "
            "Keep paragraph breaks. Return only the translated text, no "
            "preamble."
        ),
        required_variables=("target_lang", "tone"),
    ),
    "summarize_system": PromptDefinition(
        prompt_id="summarize_system",
        description="Summarization : length-controlled, focus-aware.",
        default=(
            "Summarize the document at length {length}, focused on {focus}. "
            "Return JSON with sections (title + key_points + page references)."
        ),
        required_variables=("length", "focus"),
    ),
    "question_keywords_system": PromptDefinition(
        prompt_id="question_keywords_system",
        description="Q&A keyword extraction (Volume 1, brique 2).",
        default=(
            "Extract the search keywords from the question. Return JSON "
            "with corrected_question (typo-fixed version) and keywords "
            "(list of lowercased terms suitable for full-text matching)."
        ),
        required_variables=(),
    ),
    "metadata_extract_system": PromptDefinition(
        prompt_id="metadata_extract_system",
        description="Document metadata extraction (T1.14).",
        default=(
            "Extract document metadata. Return JSON with title, author "
            "(list), effective_date (ISO 8601), language (ISO 639-1), "
            "reference_number. Use null when not present."
        ),
        required_variables=(),
    ),
    "entities_extract_system": PromptDefinition(
        prompt_id="entities_extract_system",
        description="Named-entity recognition (T1.17).",
        default=(
            "Identify named entities (person, organization, location, "
            "date, amount, email, phone, id, reference) in the line. "
            "Return JSON list of {type, surface, char_start, char_end}."
        ),
        required_variables=(),
    ),
    "sections_detect_system": PromptDefinition(
        prompt_id="sections_detect_system",
        description="Semantic sections detection (T1.19).",
        default=(
            "Split the document into semantic sections (intro / body / "
            "exclusions / appendix / ...). Return JSON list of "
            "{type, label, start_page, end_page, confidence}."
        ),
        required_variables=(),
    ),
    "tags_suggest_system": PromptDefinition(
        prompt_id="tags_suggest_system",
        description="Tag suggestion (T1.18).",
        default=(
            "Suggest up to 5 tags characterising the document : topic, "
            "urgency, status, period. Return JSON list of "
            "{label, category, confidence}."
        ),
        required_variables=(),
    ),
    "toc_reconstruct_system": PromptDefinition(
        prompt_id="toc_reconstruct_system",
        description="TOC reconstruction when no native TOC (T1.13).",
        default=(
            "Reconstruct a table of contents from the document headings. "
            "Return JSON list of {level, title, page, confidence}."
        ),
        required_variables=(),
    ),
    "document_summary": PromptDefinition(
        prompt_id="document_summary",
        description=(
            "Parsing : le résumé factuel court d'un document (type + sujet + "
            "champs/sections), construit par build_summary à partir des premières "
            "pages + TOC. Surchargeable par projet (ex. « présente sous forme de "
            "tableau Markdown »). Canonical default pour "
            "lib.parsing.pdf.doc_context."
        ),
        default=(
            "You write a short factual description of a document for a retrieval "
            "system. You are given the first pages of the document, and its table "
            "of contents when one exists. Return three to four plain sentences "
            "naming the document type, the main subject, and the kinds of fields "
            "or sections it carries. Open with the document type and the main "
            "subject (for example 'One-page resume of Jane Doe, a Data "
            "Analyst ...'). State only facts visible in the input. No marketing "
            "language, no praise, no speculation about content you cannot see."
        ),
        required_variables=(),
    ),
    "image_description": PromptDefinition(
        prompt_id="image_description",
        description=(
            "Figure reading (Article 5_7) : the vision-LLM prompt the image "
            "cascade uses to describe a chart / diagram / photo into searchable "
            "text. Canonical default for lib.parsing.pdf.images."
        ),
        default=(
            "Describe this figure in plain, searchable words: say what it shows "
            "and its kind (chart, diagram, photo, map, logo, table-as-image), "
            "then transcribe every readable label, axis, legend entry, and data "
            "value. Someone should be able to find this figure later by "
            "searching your description."
        ),
        required_variables=(),
    ),
    "fraud_review_system": PromptDefinition(
        prompt_id="fraud_review_system",
        description=(
            "Fraud review (Special edition) : the LLM reads a bundle of evidence "
            "about one document (its text, its file metadata, and the signals the "
            "deterministic detectors already raised) and assesses it for tampering. "
            "It is a copilot, not a judge : it names suspicious elements and points "
            "where to look, never returns a guilty/innocent verdict."
        ),
        default=(
            "You review a single business document for signs of tampering or "
            "fabrication, as a claims-fraud analyst would. You are a copilot, not a "
            "judge: never conclude 'fraud' or 'genuine'. List concrete suspicious "
            "elements a human should check, each with where it is and why it is odd, "
            "and rate overall suspicion low / medium / high with a calibrated "
            "confidence. Weigh the deterministic signals already provided, look for "
            "inconsistencies they missed (a tone, a layout, a value that does not fit), "
            "and say plainly when nothing looks wrong. Return JSON only."
        ),
        required_variables=(),
    ),
    "risk_synthesis_system": PromptDefinition(
        prompt_id="risk_synthesis_system",
        description=(
            "Vision 360 : the LLM EXPLAINS a risk assessment that was already "
            "computed deterministically (lib.risk) using the client's "
            "extracted facts. It explains, it never decides : no new risk level, "
            "no invented facts, cites only what it was given."
        ),
        default=(
            "You explain a company risk assessment that has ALREADY been computed "
            "by deterministic code. You do NOT decide and you do NOT change it: never "
            "invent a new risk level, never add facts that were not provided. Use ONLY "
            "the given assessment (level, rationale, flags) and the given client facts, "
            "each of which carries its own source. Return JSON: a short underwriting "
            "summary that explains the level in plain language using those facts; the "
            "drivers you relied on (drawn from the provided flags/facts); and caveats "
            "(missing data, things to confirm). You put words on a decision someone "
            "else made; you are not the underwriter."
        ),
        required_variables=(),
    ),
}


def list_defaults() -> dict[str, PromptDefinition]:
    return dict(DEFAULT_PROMPTS)


def get_default(prompt_id: str) -> PromptDefinition:
    if prompt_id not in DEFAULT_PROMPTS:
        raise KeyError(f"Unknown prompt_id: {prompt_id}")
    return DEFAULT_PROMPTS[prompt_id]


def validate_override(prompt_id: str, override: str) -> list[str]:
    """Return a list of warnings (empty if OK).

    Checks :
    - prompt_id is known
    - override doesn't exceed 4000 chars
    - required variables are still mentioned (warning if not, NOT an error)
    """
    warnings: list[str] = []
    if prompt_id not in DEFAULT_PROMPTS:
        warnings.append(f"unknown_prompt_id:{prompt_id}")
        return warnings
    if len(override) > 4000:
        warnings.append("override_too_long")
    pdef = DEFAULT_PROMPTS[prompt_id]
    for var in pdef.required_variables:
        token = "{" + var + "}"
        if token not in override:
            warnings.append(f"missing_variable:{var}")
    return warnings


def resolve_prompt(project_id: str | None, prompt_id: str) -> str:
    """Resolve the system prompt to use : override > default.

    Project overrides live in the SQLite side-car DB (table
    ``prompt_overrides``). If no override is recorded (or no DB
    available), returns the default.
    """
    pdef = get_default(prompt_id)
    if project_id is None:
        return pdef.default
    try:
        from lib.storage.sqlite_index import get_connection

        conn = get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_overrides (
                    project_id    TEXT NOT NULL,
                    prompt_id     TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    version       INTEGER NOT NULL DEFAULT 1,
                    created_at    TEXT NOT NULL,
                    created_by    TEXT,
                    PRIMARY KEY (project_id, prompt_id)
                )
                """
            )
            row = conn.execute(
                "SELECT system_prompt FROM prompt_overrides "
                "WHERE project_id = ? AND prompt_id = ?",
                (project_id, prompt_id),
            ).fetchone()
        finally:
            conn.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return pdef.default


__all__ = [
    "PromptDefinition",
    "DEFAULT_PROMPTS",
    "list_defaults",
    "get_default",
    "validate_override",
    "resolve_prompt",
]
