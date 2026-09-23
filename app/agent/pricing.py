"""Explicit, documented cost estimation for LLM runs.

``estimate_cost`` returns ``None`` unless a complete pricing configuration is
supplied — never a hardcoded, silently-stale guess baked into source. Prices
are read from environment variables so the *assumption* (price per 1K
tokens, and the date it was recorded) is visible, swappable, and auditable
from run configuration rather than buried in code that could go stale
without anyone noticing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PricingConfig:
    """One price point for one model, with the date it was recorded.

    Every field must be supplied explicitly (see ``load_pricing_config``) —
    there is no hardcoded table to fall back on — so any cost estimate this
    produces always carries its own documented assumptions.
    """

    model_name: str
    input_price_per_1k_tokens: float
    output_price_per_1k_tokens: float
    pricing_as_of: str


def load_pricing_config(model_name: str) -> Optional[PricingConfig]:
    """Build a PricingConfig from environment variables, or None if pricing
    is not fully configured.

    Requires ALL of ``LLM_INPUT_PRICE_PER_1K``, ``LLM_OUTPUT_PRICE_PER_1K``,
    and ``LLM_PRICING_AS_OF`` to be set. A partially-configured environment
    is treated the same as an unconfigured one — estimated_cost stays
    ``None`` — rather than guessing at whichever piece is missing.
    """

    input_price = os.environ.get("LLM_INPUT_PRICE_PER_1K")
    output_price = os.environ.get("LLM_OUTPUT_PRICE_PER_1K")
    pricing_as_of = os.environ.get("LLM_PRICING_AS_OF")

    if not input_price or not output_price or not pricing_as_of:
        return None

    return PricingConfig(
        model_name=model_name,
        input_price_per_1k_tokens=float(input_price),
        output_price_per_1k_tokens=float(output_price),
        pricing_as_of=pricing_as_of,
    )


def estimate_cost(
    input_tokens: Optional[int], output_tokens: Optional[int], pricing: Optional[PricingConfig]
) -> Optional[float]:
    """Estimated cost in whatever currency/unit the configured price points
    use, or None if pricing is unconfigured or token counts are unavailable.
    """

    if pricing is None or input_tokens is None or output_tokens is None:
        return None
    return (
        (input_tokens / 1000) * pricing.input_price_per_1k_tokens
        + (output_tokens / 1000) * pricing.output_price_per_1k_tokens
    )
