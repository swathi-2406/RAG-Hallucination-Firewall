"""
src/llm_client.py
LLM client — DeepSeek V4 Flash only.
OpenAI-compatible SDK, different base_url.
"""

import logging
import time
from typing import List

from openai import OpenAI
from config.settings import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    LLM_TEMP, LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the question using ONLY "
    "the provided context. If the context does not contain enough information, "
    "say so explicitly. Do not fabricate information."
)

# Stress-eval only. Removes the explicit anti-fabrication / hedging directive
# so hallucination elicitation doesn't depend on overcoming an explicit
# instruction via temperature alone (temperature governs word-choice
# diversity, not whether an instruction-following model obeys a direct
# instruction -- that's why evaluate_stress.py's temp=0.9 run alone produced
# zero hallucinations: the model was still being told point-blank not to
# fabricate, and it didn't). This prompt asks the model to answer confidently
# and fill gaps from general knowledge when context is thin, which is what
# actually creates a genuine hallucination-detection test case. NEVER use
# this outside evaluation -- it's the opposite of what the deployed
# middleware should do.
SYSTEM_PROMPT_STRESS = (
    "You are a helpful technical assistant. Answer the question as directly "
    "and confidently as you can, drawing on the provided context plus your "
    "own general knowledge to fill in any gaps. Give a complete, specific "
    "answer rather than saying you don't know."
)


def build_rag_messages(query: str, context: str, stress: bool = False) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_STRESS if stress else SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
    ]


def call_llm(
    messages: List[dict],
    temperature: float = LLM_TEMP,
    max_tokens: int = LLM_MAX_TOKENS,
    **kwargs,  # absorbs any legacy kwargs like prefer_eval
) -> tuple:
    """
    Call DeepSeek V4 Flash.
    Returns: (response_text, backend_used, latency_ms)
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not set in .env. Sign up free at platform.deepseek.com")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    text = response.choices[0].message.content.strip()
    logger.debug(f"DeepSeek [{DEEPSEEK_MODEL}] {latency_ms:.0f}ms")
    return text, "deepseek", latency_ms
