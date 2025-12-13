"""
LLM-based intent parsing for natural language queries.
Uses OpenAI ChatCompletion (function-free) to return a compact JSON intent.
Falls back to rule-based if no API key.
"""
import json
import logging
from typing import Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    logger.warning("OpenAI SDK not available - LLM intent parsing disabled")


INTENT_SCHEMA = """{
  "intent": "compare_days | spikes | detect | simulate | plan | other",
  "days": ["Saturday","Monday"] | [],
  "window_hours": number | null,
  "top_n": number | null,
  "render_hint": "table | chart | summary | table+chart",
  "notes": "short rationale"
}"""


async def parse_intent_llm(query: str) -> Optional[Dict[str, Any]]:
    """Parse intent via OpenAI. Returns None if unavailable/fails."""
    if not OPENAI_AVAILABLE or not settings.OPENAI_API_KEY:
        return None

    try:
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        prompt = f"""You are an ED ops intent parser. Return ONLY JSON matching this schema:
{INTENT_SCHEMA}

Rules:
- If user asks to compare days (e.g., Saturday vs Monday), set intent=compare_days and fill days[].
- If user asks about spikes/surges, intent=spikes.
- If user asks about bottlenecks/current issues, intent=detect.
- If user asks to simulate/what-if, intent=simulate.
- If unclear, intent=other.

Query: \"{query}\"
Return JSON only."""

        resp = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "ED ops intent parser. JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(content) if content else None
    except Exception as e:
        logger.warning(f"LLM intent parsing failed: {e}")
        return None

LLM-based intent parsing for natural language queries.
Uses OpenAI ChatCompletion (function-free) to return a compact JSON intent.
Falls back to rule-based if no API key.
"""
import json
import logging
from typing import Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    logger.warning("OpenAI SDK not available - LLM intent parsing disabled")


INTENT_SCHEMA = """{
  "intent": "compare_days | spikes | detect | simulate | plan | other",
  "days": ["Saturday","Monday"] | [],
  "window_hours": number | null,
  "top_n": number | null,
  "render_hint": "table | chart | summary | table+chart",
  "notes": "short rationale"
}"""


async def parse_intent_llm(query: str) -> Optional[Dict[str, Any]]:
    """Parse intent via OpenAI. Returns None if unavailable/fails."""
    if not OPENAI_AVAILABLE or not settings.OPENAI_API_KEY:
        return None

    try:
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        prompt = f"""You are an ED ops intent parser. Return ONLY JSON matching this schema:
{INTENT_SCHEMA}

Rules:
- If user asks to compare days (e.g., Saturday vs Monday), set intent=compare_days and fill days[].
- If user asks about spikes/surges, intent=spikes.
- If user asks about bottlenecks/current issues, intent=detect.
- If user asks to simulate/what-if, intent=simulate.
- If unclear, intent=other.

Query: \"{query}\"
Return JSON only."""

        resp = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "ED ops intent parser. JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(content) if content else None
    except Exception as e:
        logger.warning(f"LLM intent parsing failed: {e}")
        return None

