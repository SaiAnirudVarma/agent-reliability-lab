"""OpenAI implementation of the EmbeddingProvider seam.

NOT INVOKED ANYWHERE in this repository as of Phase 7B -- no test, script,
or runner constructs or calls this class. It exists so the architecture is
ready for a future, explicitly authorized real-embedding run (see
docs/RETRIEVAL_DESIGN.md), matching how ``OpenAIProvider`` itself
(app.providers.openai_provider) was built ahead of its own first
authorized real call.

``openai`` is imported lazily inside ``__init__`` (same convention as
``OpenAIProvider``), so importing this module never requires the package
to be installed unless this class is actually instantiated.

``model_name`` has NO default value: it is a required, configuration-driven
argument, never guessed or hardcoded, since no embedding model has been
selected yet. No pricing/cost estimation is implemented here -- inventing
a per-token embeddings price without a confirmed pricing table would
misrepresent a guess as a fact, and no free-token program is assumed to
apply.

Served-model provenance: the OpenAI Python SDK's embeddings response
object does carry a ``.model`` attribute in current SDK versions, so it is
read and used as ``EmbeddedVector.served_model_name`` when present and
non-empty. If a future SDK/API response omits it, this deliberately falls
back to ``None`` rather than substituting the requested ``model_name`` --
unlike ``OpenAIProvider.complete_structured``'s own fallback-to-requested
behavior for chat completions, embeddings provenance here is held to the
stricter rule this hardening pass calls for: never claim served-model
identity was independently confirmed when it wasn't.
"""

from __future__ import annotations

from app.models.contracts import ModelProvider
from app.retrieval.embedding_provider import EmbeddedVector, validate_vector


class OpenAIEmbeddingProvider:
    """EmbeddingProvider backed by OpenAI's embeddings API."""

    provider = ModelProvider.OPENAI

    def __init__(self, api_key: str, model_name: str, embedding_dimension: int):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIEmbeddingProvider requires the 'openai' package. Install it with: "
                "pip install -e '.[openai]'"
            ) from exc

        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        self._client = OpenAI(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[EmbeddedVector]:
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        served_model_name = getattr(response, "model", None) or None
        return [
            EmbeddedVector(
                vector=validate_vector(item.embedding, expected_dimension=self.embedding_dimension),
                served_model_name=served_model_name,
            )
            for item in response.data
        ]

    def embed_query(self, text: str) -> EmbeddedVector:
        return self.embed_documents([text])[0]
