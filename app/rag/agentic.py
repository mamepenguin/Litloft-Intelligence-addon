"""Agentic Ask loop (Phase 1.C).

Orchestrates the LLM and the 4 Phase 1.B tools into a multi-turn
conversation: LLM decides → tool runs → result returned → LLM decides
→ … until the LLM emits a final answer or a stop condition fires.

Stop conditions (in priority order):

* ``finish_reason="stop"`` — LLM emitted a final answer; parse and
  return.
* ``max_tool_calls`` reached — inject a "force final answer" message
  and ask the LLM once more for a non-tool response.
* Cumulative token budget exhausted — same forced-answer flow.
* Tool dispatch failed (unknown tool, JSON parse failure, fail-loud
  envelope) — abort with ``{answer: "情報を取得できませんでした",
  citations: []}`` (hako ``TtOsHILDUcbcyCghciY-9``).

Citation-strict verification: every citation in the final answer must
reference a ``file_id`` that a tool actually returned. The verifier
permits one retry — if the LLM cites an invalid ID, we resubmit the
last turn with a system reminder; if the second attempt still fails,
the answer is dropped on the floor and the loop fails loud with a
fixed "情報なし" envelope.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.config import AgenticModelEntry, LLMConfig
from app.credentials import CallerCredential
from app.llm import ChatTurnResult, LLMClient
from app.prompt_loader import render
from app.rag.agentic_types import AgenticTelemetry
from app.rag.parser import ParsedAnswer, parse_answer
from app.rag.tools.budget import estimate_payload_tokens, estimate_tokens
from app.rag.tools.context import ToolContext, ToolResultEnvelope
from app.rag.tools.get_file_chunks import get_file_chunks
from app.rag.tools.get_file_detail import get_file_detail
from app.rag.tools.get_related_files import get_related_files
from app.rag.tools.schemas import TOOL_NAMES, TOOL_SCHEMAS
from app.rag.tools.search_files import search_files

logger = logging.getLogger(__name__)

# Default ceilings; the spec recommends 10 / 0.7 × context window.
DEFAULT_MAX_TOOL_CALLS = 10
BUDGET_RATIO = 0.7

FAIL_LOUD_ANSWER = "情報を取得できませんでした。"


@dataclass(frozen=True)
class AgenticAnswer:
    """Final shape returned by ``run_agentic_loop``."""

    answer: str
    citations: list[dict[str, str]]
    telemetry: AgenticTelemetry


def agentic_capability_supported(model: str, config: LLMConfig) -> bool:
    """True iff the active model is on the operator's agentic allowlist."""
    if config.agentic_mode == "off":
        return False
    if not model:
        return False
    return any(m.name == model for m in config.agentic_models)


def get_agentic_model_entry(
    model: str, config: LLMConfig
) -> AgenticModelEntry | None:
    for m in config.agentic_models:
        if m.name == model:
            return m
    return None


def compute_token_budget(context_window: int) -> int:
    """``context_window * 0.7`` floored to a sane minimum."""
    return max(1024, int(context_window * BUDGET_RATIO))


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


async def _dispatch_tool(
    name: str, arguments: dict, *, context: ToolContext
) -> ToolResultEnvelope:
    """Route ``name`` to the matching wrapper, passing through args.

    Unknown tool names return a fail-loud envelope so the loop can
    abort. The dispatcher does NOT silently swallow errors — every
    tool wrapper already returns a well-formed envelope on its
    domain failure modes (404, fail-loud secrets, etc.).
    """
    logger.info("agentic dispatch: %s args=%s", name, arguments)
    if name == "search_files":
        env = await search_files(
            context=context,
            query=str(arguments.get("query", "")),
            top_k=int(arguments.get("top_k", 10) or 10),
        )
    elif name == "get_file_detail":
        env = await get_file_detail(
            context=context,
            file_id=str(arguments.get("file_id", "")),
        )
    elif name == "get_file_chunks":
        env = await get_file_chunks(
            context=context,
            file_id=str(arguments.get("file_id", "")),
            type=str(arguments.get("type", "transcript")),
            mode=str(arguments.get("mode", "summary")),
            range=arguments.get("range"),
        )
    elif name == "get_related_files":
        env = await get_related_files(
            context=context,
            file_id=str(arguments.get("file_id", "")),
            kind=arguments.get("kind"),
        )
    else:
        env = ToolResultEnvelope(
            payload={"error": f"unknown tool: {name}"},
            token_estimate=0,
            truncated=False,
            warning="unknown_tool",
        )
    # Log a compact preview so operators can trace what the LLM saw.
    payload_str = _serialise_tool_payload(env.payload)
    logger.info(
        "agentic result: %s tokens=%d truncated=%s payload[:600]=%s",
        name,
        env.token_estimate,
        env.truncated,
        payload_str[:600],
    )
    return env


# ---------------------------------------------------------------------------
# Loop helpers
# ---------------------------------------------------------------------------


def _build_system_prompt(
    *, max_tool_calls: int, language_instruction: str
) -> str:
    return render(
        "rag/agentic_system.jinja2",
        language_instruction=language_instruction,
        max_tool_calls=max_tool_calls,
    )


def _format_messages(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _serialise_tool_payload(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(payload)


def _parse_tool_arguments(raw: str) -> dict | None:
    """Tolerant JSON parser for the LLM's tool-call arguments.

    Returns None on parse failure; the caller treats that as a
    fail-loud event because the LLM violated the schema.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _parse_final_answer(
    content: str | None,
    allowed_file_ids: frozenset[str],
) -> tuple[ParsedAnswer | None, list[dict[str, Any]]]:
    """Pull JSON out of the LLM's final ``content`` and validate it.

    Returns (parsed, raw_citations) so the caller can distinguish
    "no citations" from "citations were dropped by strict check".

    Fallback: if ``content`` is non-empty prose that does not parse
    as JSON, treat it as ``{"answer": content, "citations": []}``.
    That way a model that ignored the JSON-only instruction still
    delivers something useful instead of a fail-loud envelope. The
    citation-strict guarantee is preserved because the synthetic
    payload carries an empty citations list.
    """
    if not content:
        return None, []
    text = content.strip()
    if text.startswith("```"):
        # Strip fenced code block ``` ... ``` if the LLM wrapped JSON.
        text = text.strip("`")
        # Remove leading ``json`` language tag if present.
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.info(
            "agentic loop: final content was not JSON; "
            "falling back to prose-with-no-citations"
        )
        synthetic = {"answer": content, "citations": []}
        return parse_answer(synthetic, allowed_file_ids), []
    raw_citations: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        citations_raw = payload.get("citations")
        if isinstance(citations_raw, list):
            raw_citations = [c for c in citations_raw if isinstance(c, dict)]
    return parse_answer(payload, allowed_file_ids), raw_citations


def _fail_loud_answer(telemetry: AgenticTelemetry) -> AgenticAnswer:
    return AgenticAnswer(
        answer=FAIL_LOUD_ANSWER,
        citations=[],
        telemetry=telemetry,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_agentic_loop(
    *,
    query: str,
    llm_client: LLMClient,
    drive: str | None = None,
    viewer_id: str | None = None,
    credential: CallerCredential | None = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_total_tokens: int,
    language_instruction: str = "",
    temperature: float | None = None,
) -> AgenticAnswer:
    """Drive the agentic loop end-to-end and return the final answer.

    The function never raises. On any unrecoverable error it returns
    a ``FAIL_LOUD_ANSWER`` envelope with empty citations; the
    telemetry reflects whatever progress was made.
    """
    context = ToolContext(
        drive=drive, viewer_id=viewer_id, credential=credential
    )

    system_prompt = _build_system_prompt(
        max_tool_calls=max_tool_calls,
        language_instruction=language_instruction,
    )
    messages: list[dict] = _format_messages(system_prompt, query)
    context.cumulative_tokens += estimate_tokens(system_prompt) + estimate_tokens(
        query
    )

    forced_answer = False
    citation_retries = 0
    tool_calls_so_far = 0

    while tool_calls_so_far <= max_tool_calls:
        # Budget check BEFORE another LLM turn. We allow a forced
        # final answer even when over budget; that's the whole point
        # of the forced-answer escape.
        over_budget = context.cumulative_tokens >= max_total_tokens
        force_now = (
            over_budget or tool_calls_so_far >= max_tool_calls
        ) and not forced_answer

        # When forcing the final answer, append a system reminder so
        # the LLM stops calling tools.
        if force_now:
            force_msg = (
                "You have used the allotted tool budget. Do not call "
                "any more tools. Produce the final JSON answer now, "
                "using only the information gathered so far. If you "
                "have insufficient information, say so honestly."
            )
            messages.append({"role": "system", "content": force_msg})
            # Account for the injected reminder against the budget
            # so the next iteration's budget check stays honest.
            context.cumulative_tokens += estimate_tokens(force_msg)
            forced_answer = True

        # ``response_format=json_object`` enforces JSON output on the
        # final-answer turn. OpenAI (and ollama 0.4+'s /v1 layer)
        # honour it alongside ``tools``; backends that ignore the
        # field fall back to the prompt-level "Return JSON only"
        # instruction, which the system prompt already includes.
        result: ChatTurnResult | None = await llm_client.chat_with_tools(
            messages,
            tools=list(TOOL_SCHEMAS) if not forced_answer else None,
            temperature=temperature,
            tool_choice="auto" if not forced_answer else "none",
            response_format={"type": "json_object"},
        )
        if result is None:
            telemetry = AgenticTelemetry(
                tool_calls=tuple(context.tool_calls),
                max_context_tokens_used=context.cumulative_tokens,
                forced_answer=forced_answer,
                citation_retries=citation_retries,
            )
            return _fail_loud_answer(telemetry)

        logger.info(
            "agentic turn: finish=%s content_len=%d tool_calls=%d forced=%s",
            result.finish_reason,
            len(result.content or ""),
            len(result.tool_calls),
            forced_answer,
        )
        if result.content:
            logger.info("agentic content[:400]: %s", result.content[:400])

        # Account for the assistant content the LLM just produced;
        # otherwise long reasoning text would not press the budget
        # check, while it still inflates the real prompt sent next
        # turn. Tool-result tokens are already booked at dispatch.
        context.cumulative_tokens += estimate_tokens(result.content or "")

        # Append the LLM's reply so the next turn sees the full
        # transcript. OpenAI's tool message contract requires the
        # assistant turn that produced the tool_calls to precede each
        # ``role: "tool"`` reply. We deliberately strip ``tool_calls``
        # in the forced-answer branch — when ``tool_choice="none"`` is
        # ignored by the LLM, leaving the orphans would either dispatch
        # them (we don't) or leave them dangling without matching
        # ``role: "tool"`` replies, which provokes a 400 on the next
        # turn and trips fail-loud for an avoidable reason.
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": result.content or "",
        }
        if result.tool_calls and not forced_answer:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments_raw,
                    },
                }
                for tc in result.tool_calls
            ]
        messages.append(assistant_msg)

        # If the model elected to stop, parse the final answer.
        if not result.tool_calls or forced_answer:
            allowed = frozenset(context.tool_returned_file_ids)
            parsed, raw_citations = _parse_final_answer(result.content, allowed)

            # Citation strict: if any cited file_id is outside the
            # allowed set and we have retries left, ask the LLM once
            # more. The retry happens in-line; the loop does not go
            # back to issuing tool calls.
            cited_ids = {
                str(c.get("file_id"))
                for c in raw_citations
                if isinstance(c, dict) and c.get("file_id")
            }
            invalid = cited_ids - set(allowed)
            if (
                invalid
                and citation_retries < 1
                and not forced_answer
            ):
                citation_retries += 1
                retry_msg = (
                    "Citation check failed: the following file_id "
                    "values are not in the tool results — "
                    f"{sorted(invalid)}. Re-emit the JSON answer "
                    "using only file_ids that a tool actually "
                    "returned. Empty citations [] is preferred over "
                    "fabricated ones."
                )
                messages.append({"role": "system", "content": retry_msg})
                context.cumulative_tokens += estimate_tokens(retry_msg)
                # Retry without tools so the model focuses on the answer.
                retry = await llm_client.chat_with_tools(
                    messages,
                    tools=None,
                    temperature=temperature,
                    tool_choice="none",
                    response_format={"type": "json_object"},
                )
                if retry is None:
                    logger.warning(
                        "agentic loop: citation-strict retry failed "
                        "(LLM client returned None); failing loud"
                    )
                    telemetry = AgenticTelemetry(
                        tool_calls=tuple(context.tool_calls),
                        max_context_tokens_used=context.cumulative_tokens,
                        forced_answer=forced_answer,
                        citation_retries=citation_retries,
                    )
                    return _fail_loud_answer(telemetry)
                context.cumulative_tokens += estimate_tokens(retry.content or "")
                messages.append(
                    {"role": "assistant", "content": retry.content or ""}
                )
                parsed, raw_citations = _parse_final_answer(
                    retry.content, allowed
                )
                cited_ids = {
                    str(c.get("file_id"))
                    for c in raw_citations
                    if isinstance(c, dict) and c.get("file_id")
                }
                invalid = cited_ids - set(allowed)

            telemetry = AgenticTelemetry(
                tool_calls=tuple(context.tool_calls),
                max_context_tokens_used=context.cumulative_tokens,
                forced_answer=forced_answer,
                citation_retries=citation_retries,
            )

            if parsed is None or invalid:
                # Either malformed JSON or citation_strict still
                # failing — fail loud (hako TtOsHILDUcbcyCghciY-9).
                logger.warning(
                    "agentic loop produced no usable answer "
                    "(parsed=%s, invalid_citations=%s)",
                    parsed is not None,
                    sorted(invalid),
                )
                return _fail_loud_answer(telemetry)

            return AgenticAnswer(
                answer=parsed.answer,
                citations=[
                    {"file_id": c.file_id, "location": c.location}
                    for c in parsed.citations
                ],
                telemetry=telemetry,
            )

        # Tool dispatch path. Run each tool the LLM asked for,
        # serialise the result, append a tool message per call.
        for tc in result.tool_calls:
            if tool_calls_so_far >= max_tool_calls:
                # Drop overflow tool calls; next iteration will force
                # the final answer.
                break
            tool_calls_so_far += 1
            args = _parse_tool_arguments(tc.arguments_raw)
            if args is None:
                # JSON-mangled arguments — fail loud per spec
                # (TtOsHILDUcbcyCghciY-9).
                telemetry = AgenticTelemetry(
                    tool_calls=tuple(context.tool_calls),
                    max_context_tokens_used=context.cumulative_tokens,
                    forced_answer=forced_answer,
                    citation_retries=citation_retries,
                )
                logger.warning(
                    "agentic loop: malformed tool arguments for %s",
                    tc.name,
                )
                return _fail_loud_answer(telemetry)
            envelope = await _dispatch_tool(
                tc.name, args, context=context
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": _serialise_tool_payload(envelope.payload),
                }
            )

    # We fell out of the while loop without returning — should not
    # happen because the budget check above forces a final answer,
    # but defend against an off-by-one by surfacing fail-loud.
    telemetry = AgenticTelemetry(
        tool_calls=tuple(context.tool_calls),
        max_context_tokens_used=context.cumulative_tokens,
        forced_answer=forced_answer,
        citation_retries=citation_retries,
    )
    return _fail_loud_answer(telemetry)


__all__ = [
    "AgenticAnswer",
    "DEFAULT_MAX_TOOL_CALLS",
    "FAIL_LOUD_ANSWER",
    "agentic_capability_supported",
    "compute_token_budget",
    "get_agentic_model_entry",
    "run_agentic_loop",
]
