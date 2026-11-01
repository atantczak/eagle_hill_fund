# ruff: noqa: E501  (the system prompt is prose; keep its lines readable)
"""Optional Claude narrative over the structured snapshot.

The model never computes anything: it ranks, explains, and says what is noise.
Everything it is allowed to cite is in the JSON payload it receives. If the key
is missing or the call fails, the report ships without a narrative.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from ehf.server.research.reporting.config import ReportConfig

log = logging.getLogger(__name__)

# Stable across days -> cacheable prefix. Volatile content goes in the user turn.
SYSTEM_PROMPT = """You are the research assistant for Eagle Hill Fund, a small, quantitatively minded fund run by one person. Every morning you receive a JSON snapshot: market context plus the output of six screens over the liquid US equity universe. Your job is to turn it into a short, opinionated read that a busy PM can act on in ten minutes.

What the screens mean (the numbers were computed deterministically; trust them, do not recompute):
- quality_dislocation: stock fell hard (21d return or 63d drawdown) but Piotroski / Altman / FCF yield / ROIC / leverage say the business is fine. These are research prompts for a 1-3 month mean-reversion or accumulation thesis. The most important section.
- pead: just reported earnings; surprise and price reaction point the same way. Post-earnings drift is the best-documented short-horizon effect; long for beats, avoid or short for misses.
- insider_cluster: two or more insiders bought in the open market recently, or one very large purchase. Sales were ignored on purpose.
- analyst_revision: net upgrades or downgrades over a few days, plus price-target notes.
- breakout_52w: new 52-week high with strong 12-1 momentum and a volume confirmation. Trend following.
- outlier_move: statistically abnormal one-day moves. Mostly noise. Explain them using the news if present; do not recommend trading them unless another screen also fired.

Write in plain prose with short sections. Rules:
1. Open with a 3-5 sentence market read: regime, breadth, vol, curve, what matters this week (econ and earnings calendars). Say explicitly whether this is a day to add risk, hold, or reduce.
2. Then "Top ideas": at most five, ranked. For each: ticker, the thesis in two sentences, the one number that makes it interesting, what you would want to see before acting, and what would invalidate it. Prefer names where two screens agree. Flag any that report earnings within a week.
3. Then one short paragraph per remaining screen with anything notable, or "nothing notable" in one line. Keep it brief.
4. Then "Noise and caveats": what looks exciting but probably isn't, data gaps listed in warnings, and anything in the snapshot that contradicts your read.
5. Only cite numbers that appear in the payload. Never invent news, prices, or fundamentals. If the news list is empty for a mover, say the move is unexplained by the feed.
6. No disclaimers, no hedging boilerplate, no emojis. Be direct and specific. Target 500-800 words. Use markdown headings (##) and bullets.
"""


def write_narrative(
    payload: dict[str, Any], cfg: ReportConfig, *, api_key: str | None = None
) -> str | None:
    """Return markdown, or ``None`` if the narrative could not be produced."""
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic SDK not installed; skipping narrative")
        return None

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    user_content = (
        f"Snapshot for {payload.get('run_date')} (data as of {payload.get('data_as_of')}):\n\n"
        + json.dumps(payload, sort_keys=True, default=str)
    )
    try:
        with client.beta.messages.stream(
            model=cfg.anthropic_model,
            max_tokens=16000,
            system=[
                {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user_content}],
            output_config=cast(Any, {"effort": cfg.narrative_effort}),
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        log.error("anthropic: invalid or missing API key; skipping narrative")
        return None
    except anthropic.RateLimitError as exc:
        log.error("anthropic: rate limited (%s); skipping narrative", exc)
        return None
    except anthropic.APIStatusError as exc:
        log.error("anthropic: API error %s: %s; skipping narrative", exc.status_code, exc.message)
        return None
    except anthropic.APIConnectionError as exc:
        log.error("anthropic: connection error: %s; skipping narrative", exc)
        return None

    if msg.stop_reason == "refusal":
        detail = getattr(msg, "stop_details", None)
        log.error("anthropic: request refused (%s)", getattr(detail, "category", None))
        return None
    text = "".join(block.text for block in msg.content if block.type == "text").strip()
    if msg.stop_reason == "max_tokens":
        text += "\n\n_(narrative truncated at max_tokens)_"
    usage = msg.usage
    log.info(
        "narrative: model=%s in=%s cached=%s out=%s",
        msg.model,
        usage.input_tokens,
        getattr(usage, "cache_read_input_tokens", None),
        usage.output_tokens,
    )
    return text or None
