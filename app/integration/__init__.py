"""Phase 8A: the integration layer wiring a frozen, preserved reranking
result to the existing LLM agent and evaluation pipeline.

Frozen reranking result -> Top-K reranked evidence -> existing LLMAgent
-> existing tracing/evaluation -> end-to-end quality metrics.

Every module here REUSES existing contracts, provider abstractions,
tracing, evaluation, metrics, and artifact-safety patterns from
app.agent, app.evaluation, app.observability, app.retrieval, and
app.reranking -- nothing here redefines a shape that already exists
elsewhere in the project.
"""
