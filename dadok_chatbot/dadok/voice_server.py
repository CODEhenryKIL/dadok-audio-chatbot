"""Local voice demo with the same answer style for keyboard and microphone input."""
from contextlib import asynccontextmanager
import copy
from datetime import timedelta
import json
import os
import uuid

from fastapi import HTTPException
from fastapi.responses import FileResponse
from livekit import api
from pydantic import BaseModel, Field

from .config import BASE, PROJECT, sibling
from .server import create_app as create_text_app
from .voice_config import VoiceSettings

AGENT_NAME = 'dadok-voice'


class VoiceInput(BaseModel):
    child_id: str = Field(default='demo-child', min_length=1, max_length=100)
    session_id: str | None = Field(default=None, min_length=1, max_length=100)
    tts_voice: str | None = Field(default=None, min_length=1, max_length=100)


def create_app(pipeline=None, settings=None):
    app = create_text_app(pipeline)
    settings = settings or VoiceSettings.from_env()
    text_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with text_lifespan(app):
            # Scope the voice style to this demo, including both chat endpoints.
            app.state.pipeline = copy.copy(app.state.pipeline)
            app.state.pipeline.voice_mode = True
            guide = (sibling('정책 및 질문 문서') / '09_음성_답변_가이드.md').read_text()
            app.state.pipeline.answer_policy += (
                '\n\n이 웹 데모의 키보드 입력에도 아래 음성 대화 모드 규칙을 동일하게 적용한다.\n' + guide)
            yield

    app.router.lifespan_context = lifespan

    @app.get('/api/voice/config')
    async def config():
        return settings.public()

    @app.post('/api/voice/token')
    async def token(body: VoiceInput):
        if not settings.enabled:
            raise HTTPException(409, '음성대화가 꺼져 있습니다.')
        try:
            settings.validate()
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from None
        try:
            selected = settings.select_voice(body.tts_voice)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        user = os.getenv('DADOK_AUTH_USER_ID') or 'local-demo'
        session = body.session_id or uuid.uuid4().hex
        try:
            app.state.pipeline.store.get_session(session, user, body.child_id)
        except PermissionError:
            raise HTTPException(403, '이 세션에 접근할 수 없습니다.') from None
        room = 'dadok-' + uuid.uuid4().hex
        identity = 'parent-' + uuid.uuid4().hex
        metadata = json.dumps({'user_id': user, 'child_id': body.child_id,
                               'session_id': session, 'identity': identity, 'tts_voice': selected.tts_voice})
        jwt = (api.AccessToken(settings.livekit_key, settings.livekit_secret)
               .with_identity(identity).with_name('보호자').with_ttl(timedelta(minutes=10))
               .with_grants(api.VideoGrants(room_join=True, room=room,
                                            can_publish=True, can_subscribe=True, can_publish_data=True))
               .with_room_config(api.RoomConfiguration(agents=[
                   api.RoomAgentDispatch(agent_name=AGENT_NAME, metadata=metadata)]))
               .to_jwt())
        return {'url': settings.livekit_url, 'token': jwt, 'room': room,
                'session_id': session, 'identity': identity, 'tts_voice': selected.tts_voice}

    @app.get('/voice-assets/{filename}')
    async def asset(filename: str):
        allowed = {'voice.js': PROJECT / 'static' / 'voice.js',
                   'voice.css': PROJECT / 'static' / 'voice.css',
                   'livekit-client.umd.js': BASE / 'node_modules' / 'livekit-client' / 'dist' / 'livekit-client.umd.js'}
        path = allowed.get(filename)
        if path is None or not path.is_file():
            raise HTTPException(404, '파일을 찾을 수 없습니다.')
        return FileResponse(path)

    return app


app = create_app()
