"""Prepare voice answers without committing speculative conversational state.

The existing Pipeline.chat remains the sole answer path. Its store, log sink,
and query-vector cache are staged per generation; paid provider usage is not.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any

from .config import sibling


class StaleTurnError(RuntimeError):
    """The transcript or durable conversation changed after preparation."""


class AlreadyCommittedError(RuntimeError):
    """A prepared answer can be committed only once."""


@dataclass(eq=False)
class PreparedTurn:
    question: str
    result: dict[str, Any]
    _owner: object = field(repr=False)
    _generation: int = field(repr=False)
    _baseline: dict = field(repr=False)
    _session: dict = field(repr=False)
    _logs: list = field(repr=False)
    _budget: Any = field(default=None, repr=False)
    _status: str = field(default="prepared", repr=False)

    @property
    def answer(self) -> str:
        return self.result["answer"]

    @property
    def metrics(self) -> dict:
        return self.result["metrics"]


class _StagedStore:
    def __init__(self, original, snapshot):
        self.original = original
        self.session = copy.deepcopy(snapshot)
        self._lock = asyncio.Lock()
        self.appended = False

    def lock(self, session_id):
        return self._lock

    def get_session(self, session_id, user_id, child_id):
        expected = (self.session["session_id"], self.session["user_id"], self.session["child_id"])
        if (session_id, user_id, child_id) != expected:
            raise PermissionError("Session belongs to another user or child")
        return self.session

    def get_memories(self, user_id, child_id, question):
        return self.original.get_memories(user_id, child_id, question)

    def append_turn(self, session, question, answer):
        self.session = copy.deepcopy(session)
        self.appended = True


class _StagedBudget:
    def __init__(self, original):
        self.original = original
        self.vectors = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def query_vector(self, question, vector=None):
        if vector is not None:
            self.vectors[question] = copy.deepcopy(vector)
            return vector
        if question in self.vectors:
            return copy.deepcopy(self.vectors[question])
        return self.original.query_vector(question)

    def commit(self):
        for question, vector in self.vectors.items():
            self.original.query_vector(question, vector)
        self.vectors.clear()


class _Provider:
    """Let already-started provider work settle cost after its consumer cancels."""

    def __init__(self, original, pending):
        self.inner = copy.copy(original)
        self.pending = pending
        self.budget = None
        if hasattr(original, "budget"):
            self.budget = _StagedBudget(original.budget)
            self.inner.budget = self.budget

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def _call(self, awaitable):
        task = asyncio.create_task(awaitable)
        self.pending.add(task)

        def finished(done):
            self.pending.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def generate(self, stage, messages, schema=None):
        return await self._call(self.inner.generate(stage, messages, schema=schema))

    async def embed(self, question):
        return await self._call(self.inner.embed(question))


class VoiceAdapter:
    """One voice session, with only the newest prepared transcript committable.

    Construction validates/creates the session at voice join, before speculation.
    ``cancel`` is synchronous; ``close`` also drains outstanding paid operations.
    The caller schedules normal pipeline maintenance after output, not prepare.
    """

    def __init__(self, pipeline, user_id: str, child_id: str | None, session_id: str):
        self.pipeline = pipeline
        self.user_id, self.child_id, self.session_id = user_id, child_id, session_id
        pipeline.store.get_session(session_id, user_id, child_id)
        self._owner = object()
        self._generation = 0
        self._preparing = set()
        self._providers = set()
        self._closed = False
        self._voice_instructions = (
            sibling("정책 및 질문 문서") / "09_음성_답변_가이드.md"
        ).read_text()

    def cancel(self, prepared: PreparedTurn | None = None) -> None:
        if prepared is not None:
            if prepared._owner is not self._owner:
                raise PermissionError("Prepared turn belongs to another voice session")
            if prepared._status == "committed":
                return
            prepared._status = "cancelled"
            if prepared._generation != self._generation:
                return
        self._generation += 1
        for task in tuple(self._preparing):
            task.cancel()

    async def prepare(self, question: str) -> PreparedTurn:
        if self._closed:
            raise RuntimeError("Voice session is closed")
        self.cancel()
        generation = self._generation
        task = asyncio.current_task()
        self._preparing.add(task)
        try:
            async with self.pipeline.store.lock(self.session_id):
                snapshot = copy.deepcopy(self.pipeline.store.get_session(
                    self.session_id, self.user_id, self.child_id))
            store = _StagedStore(self.pipeline.store, snapshot)
            staged = copy.copy(self.pipeline)
            staged.voice_mode = True
            # The shared text pipeline and its base policies stay unchanged.
            staged.answer_policy = self.pipeline.answer_policy + "\n\n" + self._voice_instructions
            staged.store = store
            staged.llm = _Provider(self.pipeline.llm, self._providers)
            staged.rag = copy.copy(self.pipeline.rag)
            if getattr(self.pipeline.rag, "embed", None) is not None:
                staged.rag.embed = staged.llm.embed
            logs = []
            staged._log = lambda entry: logs.append(copy.deepcopy(entry))
            result = await staged.chat(self.user_id, self.child_id, self.session_id, question)
            if self._closed or generation != self._generation:
                raise StaleTurnError("Prepared transcript was superseded")
            if not store.appended:
                raise RuntimeError("Pipeline did not produce a complete turn")
            return PreparedTurn(question, result, self._owner, generation, snapshot,
                                store.session, logs, staged.llm.budget)
        finally:
            self._preparing.discard(task)

    async def commit(self, prepared: PreparedTurn) -> dict:
        if prepared._owner is not self._owner:
            raise PermissionError("Prepared turn belongs to another voice session")
        async with self.pipeline.store.lock(self.session_id):
            if prepared._status == "committed":
                raise AlreadyCommittedError("Prepared turn is already committed")
            if self._closed or prepared._status != "prepared" or prepared._generation != self._generation:
                raise StaleTurnError("Prepared transcript was cancelled or superseded")
            current = self.pipeline.store.get_session(self.session_id, self.user_id, self.child_id)
            if any(current[key] != prepared._baseline[key] for key in ("total_turns", "_revision")):
                prepared._status = "stale"
                raise StaleTurnError("Conversation changed while the answer was being prepared")
            self.pipeline.store.append_turn(prepared._session, prepared.question, prepared.answer)
            # Mark immediately after append so a later log/cache failure cannot duplicate a turn.
            prepared._status = "committed"
            if prepared._budget is not None:
                prepared._budget.commit()
            for entry in prepared._logs:
                self.pipeline._log(entry)
            return prepared.result

    async def close(self) -> None:
        self._closed = True
        self.cancel()
        preparing = tuple(self._preparing)
        if preparing:
            await asyncio.gather(*preparing, return_exceptions=True)
        providers = tuple(self._providers)
        if providers:
            # Cancellation of a caller must not cancel the cost-settling operations.
            await asyncio.shield(asyncio.gather(*providers, return_exceptions=True))
