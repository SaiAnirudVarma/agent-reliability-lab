"""Phase 7C: reranking foundation, evaluated as a distinct second stage
after retrieval (see app.retrieval) -- candidate generation and reranking
remain independently measurable.

Contains contracts, the Reranker interface, a deterministic pass-through
baseline, before/after metrics, a provenance schema for a future real
experiment, and a safe loader for reusing a preserved retrieval artifact's
frozen candidate lists. No real reranker model, no network calls, and no
production reranking experiment exist yet -- this is foundation only.
"""
