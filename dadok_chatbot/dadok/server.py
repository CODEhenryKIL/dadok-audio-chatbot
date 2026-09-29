"""Loopback-only demo web API. Production authentication must be upstream."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import PROJECT, Settings
from .pipeline import build_pipeline

def create_app(pipeline=None):
    tasks = set()

    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline = pipeline or build_pipeline()
        yield
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title='다독 RAG 멀티턴 챗봇', lifespan=lifespan)
    if pipeline: app.state.pipeline = pipeline

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        from fastapi.responses import JSONResponse
        if request.url.hostname not in ('127.0.0.1', 'localhost', '::1', 'testserver'):
            return JSONResponse({'detail': '로컬 전용 서버입니다.'}, status_code=403)
        origin = request.headers.get('origin')
        if request.method != 'GET' and origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': '다른 출처의 요청을 허용하지 않습니다.'}, status_code=403)
        return await call_next(request)

    async def schedule(user, child, session, force=False):
        task = asyncio.create_task(app.state.pipeline.maintain(user, child, session, force_memory=force))
        tasks.add(task)
        def done(t):
            tasks.discard(t)
            if not t.cancelled(): t.exception()
        task.add_done_callback(done)

    @app.get('/')
    async def index():
        return FileResponse(PROJECT / 'static' / 'index.html')

    @app.get('/health')
    async def health():
        p = app.state.pipeline
        return {'status': 'ok', 'model': p.settings.model,
                'requested_reasoning': p.settings.requested_reasoning,
                'effective_reasoning': p.settings.effective_reasoning,
                'default_child_id': os.getenv('DADOK_CHILD_ID', 'demo-child'),
                'records_mode': 'snapshot' if p.settings.snapshot_path else ('server_api' if p.settings.server_token else 'unavailable')}

    @app.post('/api/chat')
    async def chat(body: ChatInput, background_tasks: BackgroundTasks):
        user = os.getenv('DADOK_AUTH_USER_ID', 'local-demo')
        session = body.session_id or uuid.uuid4().hex
        try:
            result = await app.state.pipeline.chat(user, body.child_id, session, body.question)
        except PermissionError:
            raise HTTPException(403, '이 세션에 접근할 수 없습니다.') from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        background_tasks.add_task(schedule, user, body.child_id, session)
        return {**result, 'session_id': session}

    @app.post('/api/chat/stream')
    async def stream(body: ChatInput):
        user = os.getenv('DADOK_AUTH_USER_ID', 'local-demo')
        session = body.session_id or uuid.uuid4().hex
        try:
            app.state.pipeline.store.get_session(session, user, body.child_id)
        except PermissionError:
            raise HTTPException(403, '이 세션에 접근할 수 없습니다.') from None
        async def events():
            queue = asyncio.Queue()
            sent_text = False
            result = None

            def emit(text):
                nonlocal sent_text
                if text:
                    sent_text = True
                    queue.put_nowait(('delta', {'text': text}))

            async def run():
                nonlocal result
                try:
                    result = await app.state.pipeline.chat(user, body.child_id, session, body.question,
                                                           on_delta=emit)
                    if not sent_text:
                        emit(result['answer'])
                    queue.put_nowait(('done', {**result, 'session_id': session}))
                except Exception:
                    queue.put_nowait(('error', {
                        'message': '답변 전송이 중단됐어요. 다시 시도해 주세요.' if sent_text else '요청 처리 중 오류가 발생했습니다.',
                        'interrupted': sent_text}))

            yield 'event: status\ndata: ' + json.dumps({'session_id': session, 'message': '확인 중'}) + '\n\n'
            task = asyncio.create_task(run())
            try:
                while True:
                    event, payload = await queue.get()
                    yield 'event: ' + event + '\ndata: ' + json.dumps(payload, ensure_ascii=False) + '\n\n'
                    if event in ('done', 'error'):
                        break
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                if result is not None:
                    await schedule(user, body.child_id, session)
        return StreamingResponse(events(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.post('/api/client-timing')
    async def client_timing(body: ClientTimingInput):
        user = os.getenv('DADOK_AUTH_USER_ID', 'local-demo')
        try:
            app.state.pipeline.store.get_session(body.session_id, user, body.child_id)
        except PermissionError:
            raise HTTPException(403, '이 세션에 접근할 수 없습니다.') from None
        if body.first_visible_ms is not None and body.first_visible_ms > body.completion_ms:
            raise HTTPException(400, '첫 표시 시간은 완료 시간보다 늦을 수 없습니다.')
        path = app.state.pipeline.settings.log_path.with_name('client_timings.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {'timestamp': datetime.now(timezone.utc).isoformat(), **body.model_dump()}
        with path.open('a') as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
        return {'status': 'recorded'}

    @app.post('/api/session/close')
    async def close(body: CloseInput):
        user = os.getenv('DADOK_AUTH_USER_ID', 'local-demo')
        try:
            app.state.pipeline.store.get_session(body.session_id, user, body.child_id)
        except PermissionError:
            raise HTTPException(403, '이 세션에 접근할 수 없습니다.') from None
        await schedule(user, body.child_id, body.session_id, True)
        return {'status': 'maintenance_scheduled'}

    @app.get('/api/cost')
    async def cost():
        return app.state.pipeline.llm.budget.report()

    return app

class ChatInput(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    child_id: str = Field(default='demo-child', min_length=1, max_length=100)
    session_id: str | None = Field(default=None, min_length=1, max_length=100)

class CloseInput(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    child_id: str = Field(default='demo-child', min_length=1, max_length=100)

class ClientTimingInput(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    child_id: str = Field(default='demo-child', min_length=1, max_length=100)
    request_id: str | None = Field(default=None, min_length=1, max_length=100)
    first_visible_ms: float | None = Field(default=None, ge=0, le=3600000, allow_inf_nan=False)
    completion_ms: float = Field(ge=0, le=3600000, allow_inf_nan=False)
    status: Literal['completed', 'interrupted', 'failed']

app = create_app()
