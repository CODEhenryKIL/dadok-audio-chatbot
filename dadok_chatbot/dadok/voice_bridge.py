"""LiveKit LLM bridge with a commit barrier for speculative answers.

Confirm the selected user item from AgentSession's conversation_item_added event.
The on_user_turn_completed hook has a different ID for reused speculative turns.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import logging
import time
from typing import Callable

from livekit.agents import llm
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from .voice_adapter import AlreadyCommittedError, StaleTurnError, VoiceAdapter


logger = logging.getLogger(__name__)


class DadokLLM(llm.LLM):
    """Run the existing chatbot early, publishing only a selected committed turn."""

    def __init__(self, adapter: VoiceAdapter, emit: Callable | None = None, *, allow_speculation: bool = True):
        super().__init__()
        self.adapter = adapter
        self.allow_speculation = allow_speculation
        self._emit_callback = emit
        self.last_committed_id: str | None = None
        self._confirmed: set[str] = set()
        self._committed: set[str] = set()
        self._gates: dict[str, asyncio.Event] = {}
        self._streams: set[_DadokStream] = set()
        self._latest: _DadokStream | None = None
        self._closed = False

    @property
    def model(self) -> str:
        return "dadok-existing-pipeline"

    @property
    def provider(self) -> str:
        return "dadok"

    def confirm(self, user_message_id: str) -> None:
        """Release the exact user message selected by LiveKit for this turn."""
        if self._closed:
            return
        self._confirmed.add(user_message_id)
        if gate := self._gates.get(user_message_id):
            gate.set()

    def _event(self, event: str, turn_id: str, **fields) -> None:
        # Conversation text goes only to the explicitly supplied UI callback.
        # Logs never receive answer text, provider errors, or credentials.
        logger.info("voice_bridge event=%s turn_id=%s duration_ms=%s",
                    event, turn_id, fields.get("duration_ms"))
        if self._emit_callback is not None:
            try:
                self._emit_callback(event, turn_id, **fields)
            except Exception:
                logger.warning("voice_bridge callback_failed event=%s", event)

    def chat(self, *, chat_ctx: llm.ChatContext, tools=None,
             conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS, **kwargs):
        if self._closed:
            raise RuntimeError("Voice bridge is closed")
        user = next((item for item in reversed(chat_ctx.items)
                     if isinstance(item, llm.ChatMessage) and item.role == "user"), None)
        if user is None or not user.text_content or not user.text_content.strip():
            raise ValueError("Voice bridge requires a user message")
        if self._latest is not None and not self._latest.finished:
            self._latest.cancel("superseded")
        stream = _DadokStream(
            self, user=user, chat_ctx=chat_ctx, tools=tools or [],
            # SDK retries must not repeat a potentially committed chatbot turn.
            conn_options=replace(conn_options, max_retry=0),
        )
        self._latest = stream
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        streams = tuple(self._streams)
        for stream in streams:
            stream.cancel("closed")
        try:
            await asyncio.gather(*(stream.aclose() for stream in streams))
        finally:
            await self.adapter.close()
            await super().aclose()
            self._gates.clear()
            self._confirmed.clear()


class _DadokStream(llm.LLMStream):
    def __init__(self, bridge: DadokLLM, *, user: llm.ChatMessage, **kwargs):
        self.bridge = bridge
        self.turn_id = user.id
        self.question = user.text_content.strip()
        self.prepared = None
        self.finished = False
        self.committed = False
        self._cancelled = False
        self._cancel_reported = False
        # The SDK cancels its inference consumer before closing its LLMStream.
        # Observing that owner closes the short propagation window at the gate.
        self._owner = asyncio.current_task()
        self._gate = bridge._gates.setdefault(self.turn_id, asyncio.Event())
        if self.turn_id in bridge._confirmed:
            self._gate.set()
        super().__init__(bridge, **kwargs)

    def _check_current(self) -> None:
        if (self._cancelled or self.bridge._closed or self.bridge._latest is not self
                or (self._owner is not None and self._owner.cancelling())):
            raise asyncio.CancelledError

    def _discard(self, reason: str) -> None:
        if self.committed or self._cancel_reported:
            return
        self._cancel_reported = True
        if self.prepared is not None:
            self.bridge.adapter.cancel(self.prepared)
        elif self.bridge._latest is self:
            self.bridge.adapter.cancel()
        self.bridge._event("speculative_cancel" if self.bridge.allow_speculation else "turn_cancelled",
                           self.turn_id, reason=reason)

    def cancel(self, reason: str) -> None:
        self._cancelled = True
        self._discard(reason)
        self._task.cancel()

    async def _run(self) -> None:
        started = time.perf_counter()
        try:
            self._check_current()
            if self.turn_id in self.bridge._committed:
                return
            if self.bridge.allow_speculation and not self._gate.is_set():
                self.bridge._event("speculative_start", self.turn_id)
            for attempt in range(2):
                self._check_current()
                self.bridge._event("chatbot_start", self.turn_id, attempt=attempt + 1)
                self.prepared = await self.bridge.adapter.prepare(self.question)
                self._check_current()
                self.bridge._event("chatbot_complete", self.turn_id, attempt=attempt + 1,
                                   duration_ms=(time.perf_counter() - started) * 1000)
                await self._gate.wait()
                self._check_current()
                try:
                    result = await self.bridge.adapter.commit(self.prepared)
                    break
                except StaleTurnError:
                    self._check_current()
                    # A maintenance/session revision conflict can be refreshed, but
                    # cancelled or superseded preparations must never be resurrected.
                    if self.prepared._status != "stale":
                        raise
                    if attempt == 1:
                        raise RuntimeError("Conversation changed during voice generation") from None
                    self.bridge.adapter.cancel(self.prepared)
                    self.prepared = None
            self.committed = True
            self.bridge._committed.add(self.turn_id)
            self.bridge.last_committed_id = self.turn_id
            self._check_current()
            self.bridge._event("turn_committed", self.turn_id,
                               question=self.question, answer=result["answer"], result=result,
                               duration_ms=(time.perf_counter() - started) * 1000)
            self._event_ch.send_nowait(llm.ChatChunk(
                id=self.turn_id,
                delta=llm.ChoiceDelta(role="assistant", content=result["answer"]),
            ))
        except asyncio.CancelledError:
            self._discard("cancelled")
            raise
        except (StaleTurnError, AlreadyCommittedError):
            self._discard("stale")
        except Exception:
            self._discard("failed")
            self.bridge._event("error", self.turn_id, code="chatbot_generation_failed")
            # Do not pass provider exception text into LiveKit error logs/traces.
            raise RuntimeError("Chatbot voice generation failed") from None
        finally:
            self.finished = True

    async def aclose(self) -> None:
        if not self.finished:
            self.cancel("cancelled")
        try:
            await super().aclose()
        finally:
            self.bridge._streams.discard(self)
            if self.bridge._gates.get(self.turn_id) is self._gate:
                self.bridge._gates.pop(self.turn_id, None)
