"""Local voice configuration; never expose credential values in diagnostics."""
from dataclasses import dataclass, field, replace
import json
import os
import re
from urllib.parse import urlparse

from dotenv import load_dotenv
from .config import BASE


def load_environment():
    load_dotenv(BASE / '.env', override=False)
    # Keep downloaded models and caches inside this experiment.
    os.environ.setdefault('HF_HOME', str(BASE / '.cache' / 'huggingface'))
    os.environ.setdefault('XDG_CACHE_HOME', str(BASE / '.cache'))
    os.environ.setdefault('LIVEKIT_URL', 'ws://127.0.0.1:7880')
    os.environ.setdefault('LIVEKIT_API_KEY', 'devkey')
    os.environ.setdefault('LIVEKIT_API_SECRET', 'secret')


def boolean(name, default):
    value = os.getenv(name, str(default)).strip().lower()
    if value not in ('true', 'false', '1', '0'):
        raise ValueError(f'{name}은 true 또는 false여야 합니다.')
    return value in ('true', '1')


def voice_options(value):
    try:
        options = json.loads(value)
        if not isinstance(options, list) or len(options) > 20:
            raise ValueError
        result = []
        for option in options:
            if not isinstance(option, dict) or set(option) != {'id', 'name'}:
                raise ValueError
            identifier, name = option['id'], option['name']
            if (not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', identifier)
                    or not isinstance(name, str) or not name.strip() or len(name) > 80):
                raise ValueError
            if identifier in {item[0] for item in result}:
                raise ValueError
            result.append((identifier, name.strip()))
        return tuple(result)
    except (ValueError, TypeError):
        raise ValueError('TTS_VOICE_OPTIONS는 id와 name을 가진 JSON 배열이어야 합니다 (중복 없이 최대 20개).') from None


@dataclass(frozen=True)
class VoiceSettings:
    enabled: bool = True
    vad_enabled: bool = True
    turn_detection_enabled: bool = True
    barge_in_enabled: bool = True
    preemptive_generation_enabled: bool = True
    preemptive_tts_enabled: bool = False
    elevenlabs_key: str = field(default='', repr=False)
    stt_provider: str = 'elevenlabs'
    stt_model: str = 'scribe_v2_realtime'
    stt_language: str = 'ko'
    stt_keyterms: tuple[str, ...] = ('분유', '수유', '이유식', '배변', '낮잠', '밤잠', '수면')
    tts_provider: str = 'elevenlabs'
    tts_model: str = 'eleven_flash_v2_5'
    tts_voice: str = ''
    tts_voice_options: tuple[tuple[str, str], ...] = ()
    livekit_url: str = 'ws://127.0.0.1:7880'
    livekit_key: str = field(default='devkey', repr=False)
    livekit_secret: str = field(default='secret', repr=False)

    @classmethod
    def from_env(cls):
        load_environment()
        return cls(
            **{attr: boolean(env, default) for attr, env, default in (
                ('enabled', 'VOICE_ENABLED', True), ('vad_enabled', 'VAD_ENABLED', True),
                ('turn_detection_enabled', 'TURN_DETECTION_ENABLED', True),
                ('barge_in_enabled', 'BARGE_IN_ENABLED', True),
                ('preemptive_generation_enabled', 'PREEMPTIVE_GENERATION_ENABLED', True),
                ('preemptive_tts_enabled', 'PREEMPTIVE_TTS_ENABLED', False))},
            elevenlabs_key=os.getenv('ELEVENLABS_API_KEY', ''),
            stt_provider=os.getenv('STT_PROVIDER', 'elevenlabs'),
            stt_model=os.getenv('STT_MODEL', 'scribe_v2_realtime'),
            stt_language=os.getenv('STT_LANGUAGE', 'ko'),
            stt_keyterms=tuple(t.strip() for t in os.getenv('STT_KEYTERMS', '분유,수유,이유식,배변,낮잠,밤잠,수면').split(',') if t.strip()),
            tts_provider=os.getenv('TTS_PROVIDER', 'elevenlabs'),
            tts_model=os.getenv('TTS_MODEL', 'eleven_flash_v2_5'),
            tts_voice=os.getenv('TTS_VOICE', ''),
            tts_voice_options=voice_options(os.getenv('TTS_VOICE_OPTIONS', '[]')),
            livekit_url=os.environ['LIVEKIT_URL'], livekit_key=os.environ['LIVEKIT_API_KEY'],
            livekit_secret=os.environ['LIVEKIT_API_SECRET'],
        )

    def voices(self):
        choices = dict(self.tts_voice_options)
        if self.tts_voice and self.tts_voice not in choices:
            choices = {self.tts_voice: '기본 목소리', **choices}
        return [{'id': identifier, 'name': name} for identifier, name in choices.items()]

    def select_voice(self, identifier=None):
        if identifier is None:
            identifier = self.tts_voice
        if not isinstance(identifier, str) or identifier not in {voice['id'] for voice in self.voices()}:
            raise ValueError('사용할 수 없는 목소리입니다. 목록에서 다시 선택해 주세요.')
        return replace(self, tts_voice=identifier)

    def missing(self):
        return [name for name, value in (
            ('ELEVENLABS_API_KEY', self.elevenlabs_key),
            ('TTS_VOICE', self.tts_voice), ('LIVEKIT_API_KEY', self.livekit_key),
            ('LIVEKIT_API_SECRET', self.livekit_secret)) if not value.strip()]

    def validate(self):
        if self.turn_detection_enabled and not self.vad_enabled:
            raise ValueError('LiveKit Turn Detector는 VAD가 필요합니다. VAD_ENABLED=true로 설정하거나 TURN_DETECTION_ENABLED=false로 함께 비교하세요.')
        if self.stt_provider != 'elevenlabs' or self.tts_provider != 'elevenlabs':
            raise ValueError('현재 음성 MVP는 STT와 TTS 모두 ElevenLabs를 사용합니다.')
        if self.stt_model != 'scribe_v2_realtime':
            raise ValueError('실시간 음성대화에는 STT_MODEL=scribe_v2_realtime이 필요합니다.')
        if len(self.stt_keyterms) > 50 or any(len(term) > 20 for term in self.stt_keyterms):
            raise ValueError('STT_KEYTERMS는 최대 50개이며 각 항목은 20자 이하여야 합니다.')
        parsed = urlparse(self.livekit_url)
        if parsed.scheme not in ('ws', 'wss') or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('LIVEKIT_URL은 로컬 self-host 서버 주소여야 합니다.')
        if self.missing():
            raise ValueError('환경설정이 필요합니다: ' + ', '.join(self.missing()))

    def public(self):
        error = ''
        try:
            self.validate()
        except ValueError as exc:
            error = str(exc)
        return {'enabled': self.enabled, 'ready': self.enabled and not error, 'error': error,
                'tts_voice': self.tts_voice, 'voices': self.voices(),
                'missing': self.missing(), 'features': {
                    'vad': self.vad_enabled, 'turn_detection': self.turn_detection_enabled,
                    'barge_in': self.barge_in_enabled,
                    'preemptive_generation': self.preemptive_generation_enabled,
                    'preemptive_tts': self.preemptive_tts_enabled}}
