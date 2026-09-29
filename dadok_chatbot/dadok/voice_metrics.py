"""Monotonic stage timings. Never persist transcript, response or credentials."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time


class VoiceMetrics:
    def __init__(self, path: Path, session_id: str, features: dict):
        self.path, self.session_id, self.features = path, session_id, features
        self.input_times = {}
        self.turns = {}

    def record(self, event, turn_id='', **fields):
        now = time.perf_counter()
        if event == 'user_speech_start':
            self.input_times = {'user_speech_start': now}
        elif event in ('user_speech_end', 'stt_final'):
            self.input_times[event] = now
        if turn_id:
            times = self.turns.setdefault(turn_id, dict(self.input_times))
            if event == 'turn_confirmed':
                times.update(self.input_times)
            times.setdefault(event, now)
        allowed = {k: v for k, v in fields.items() if k in (
            'state', 'error_type', 'recoverable', 'probability', 'threshold', 'duration_ms',
            'ttfb_ms', 'interrupted', 'audio_duration_ms', 'stage', 'speech_id')}
        row = {'timestamp': datetime.now(timezone.utc).isoformat(), 'monotonic_s': now,
               'session_id': self.session_id, 'turn_id': turn_id, 'event': event,
               'features': self.features, **allowed}
        if event == 'tts_end' and turn_id:
            row['latency_ms'] = self.latency(turn_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return row

    def latency(self, turn_id):
        times = self.turns.get(turn_id, {})
        pairs = {
            'end_to_turn': ('user_speech_end', 'turn_confirmed'),
            'end_to_stt_final': ('user_speech_end', 'stt_final'),
            'chatbot': ('chatbot_start', 'chatbot_complete'),
            'tts_ttfa': ('tts_request', 'tts_first_frame'),
            'end_to_audio_output': ('user_speech_end', 'audio_output_start'),
            'total_turn': ('user_speech_start', 'tts_end'),
        }
        return {key: round((times[end] - times[start]) * 1000, 2)
                for key, (start, end) in pairs.items() if start in times and end in times}
