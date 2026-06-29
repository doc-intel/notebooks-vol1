"""docintel — Document Intelligence: enterprise RAG building blocks.

Modules:
    core         — Pydantic models, types, LLM/embedding clients.
    parsing      — Document parsing: PDF → line_df / page_df / toc_df. (Article 5, 10)
    question     — Question parsing: question string → structured object. (Article 6)
    retrieval    — Retrieval as scope selection: keywords, TOC, embeddings. (Article 7, 9, 11, 12)
    generation   — Generation as controlled execution with verifiable output. (Article 8)
    extraction   — Structured field extraction at ingestion. (Article 14)
    pipeline     — Composite orchestrator + dispatcher + feedback loops. (Article 13, 15)
    corpus       — Corpus index, classification, versioning, filtering, SQL agent. (Article 16-21)
    storage      — Long-format tables, repositories, replayable artefacts. (Article 19)
    annotation   — PDF highlighting and annotated outputs. (Article 1, 9)
"""

__version__ = "0.1.0"
