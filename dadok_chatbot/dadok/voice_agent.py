"""Official LiveKit audio pipeline around the existing guarded Dadok pipeline."""
import asyncio
import json
import logging
import os

from .voice_config import VoiceSettings, load_environment
load_environment()  # Set local model cache before loading native inference libraries.

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference, room_io, llm
from livekit.agents.voice.events import CloseReason
from livekit.plugins import elevenlabs, silero

from .config import PROJECT
from .pipeline import build_pipeline
from .voice_adapter import VoiceAdapter
from .voice_bridge import DadokLLM
from .voice_metrics import VoiceMetrics

AGENT_NAME = 'dadok-voice'
server = AgentServer(num_idle_processes=0, host='127.0.0.1', initialize_process_timeout=60,
                     shutdown_process_timeout=60, log_level='INFO')


def build_session(settings, bridge):
    settings.validate()
    detector = inference.TurnDetector(version='v1-mini') if settings.turn_detection_enabled else (
        'vad' if settings.vad_enabled else 'stt')
    return AgentSession(
        llm=bridge,
        # Server VAD commits transcript segments; local Silero/TurnDetector still
        # decide conversation turns. Manual commit mode would need explicit flushes.
        stt=elevenlabs.STT(model=settings.stt_model, language_code=settings.stt_language,
                          api_key=settings.elevenlabs_key, keyterms=list(settings.stt_keyterms),
                          server_vad={}),
        tts=elevenlabs.TTS(model=settings.tts_model, voice_id=settings.tts_voice,
                          language='ko', api_key=settings.elevenlabs_key),
        vad=silero.VAD.load() if settings.vad_enabled else None,
        turn_handling={
            'turn_detection': detector,
            'interruption': {'enabled': settings.barge_in_enabled,
                             'mode': 'vad'},
            'preemptive_generation': {'enabled': settings.preemptive_generation_enabled,
                                      'preemptive_tts': settings.preemptive_tts_enabled}},
        # No unguarded text cleanup or separate response LLM.
        tts_text_transforms=None,
        # The browser captures with echo cancellation. The SDK's initial 3-second
        # STT mute otherwise drops the beginning of a user's first interruption.
        aec_warmup_duration=0,
    )


class DadokAgent(Agent):
    def __init__(self, bridge, metrics):
        super().__init__(instructions='기존 다독 파이프라인의 검증된 답변을 전달합니다.')
        self.bridge, self.metrics = bridge, metrics

    async def tts_node(self, text, model_settings):
        # The bridge yields only after final safety checks and confirmed-turn commit.
        first = True
        turn_id = self.bridge.last_committed_id or ''
        self.metrics.record('tts_request', turn_id)
        async for frame in Agent.default.tts_node(self, text, model_settings):
            if first:
                self.metrics.record('tts_first_frame', turn_id)
                first = False
            yield frame


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    settings = VoiceSettings.from_env()
    if not settings.enabled:
        ctx.shutdown(reason='voice_disabled')
        return
    metadata = json.loads(ctx.job.metadata or '{}')
    expected_user = os.getenv('DADOK_AUTH_USER_ID') or 'local-demo'
    if metadata.get('user_id') != expected_user or not all(metadata.get(k) for k in ('child_id', 'session_id', 'identity')):
        ctx.shutdown(reason='invalid_session_metadata')
        return
    try:
        settings = settings.select_voice(metadata.get('tts_voice'))
    except ValueError:
        ctx.shutdown(reason='invalid_voice_selection')
        return
    pipeline = build_pipeline()
    adapter = VoiceAdapter(pipeline, expected_user, metadata['child_id'], metadata['session_id'])
    metrics = VoiceMetrics(PROJECT / 'runtime' / 'voice_events.jsonl', metadata['session_id'], settings.public()['features'])
    pending = set()

    def spawn(coro):
        task = asyncio.create_task(coro)
        pending.add(task)
        def done(t):
            pending.discard(t)
            if not t.cancelled():
                t.exception()  # Transport failures must not become unhandled tasks.
        task.add_done_callback(done)
        return task

    async def publish(event, turn_id='', **fields):
        if ctx.room.isconnected():
            await ctx.room.local_participant.publish_data(
                json.dumps({'event': event, 'turn_id': turn_id, **fields}, ensure_ascii=False).encode(),
                reliable=True, topic='dadok.voice', destination_identities=[metadata['identity']])

    def emit(event, turn_id='', **fields):
        metrics.record(event, turn_id, **fields)
        if event == 'turn_committed':
            result = fields.get('result', {})
            spawn(publish('answer', turn_id, text=fields.get('answer', result.get('answer', ''))))
            spawn(publish(event, turn_id))
        elif event in ('speculative_cancel', 'turn_cancelled'):
            spawn(publish('turn_cancelled', turn_id))
        elif event == 'error':
            spawn(publish('error', turn_id, message='답변 처리 중 문제가 발생했습니다. 다시 말씀해 주세요.'))

    bridge = DadokLLM(adapter, emit=emit, allow_speculation=settings.preemptive_generation_enabled)
    session = build_session(settings, bridge)
    job_shutdown_requested = False
    cleanup_started = False

    def shutdown_job(reason):
        nonlocal job_shutdown_requested
        if job_shutdown_requested or cleanup_started:
            return
        job_shutdown_requested = True
        ctx.shutdown(reason=reason)

    @session.on('close')
    def session_closed(ev):
        # A closed AgentSession does not disconnect its room by default. Let the
        # job runtime do that, including after the SDK exhausts provider retries.
        if ev.reason != CloseReason.JOB_SHUTDOWN:
            shutdown_job('voice_session_' + ev.reason.value)

    @session.on('conversation_item_added')
    def conversation(ev):
        if not isinstance(ev.item, llm.ChatMessage):
            return
        if ev.item.role == 'user':
            bridge.confirm(ev.item.id)
            metrics.record('turn_confirmed', ev.item.id)
            spawn(publish('user_final', ev.item.id, question=ev.item.text_content or ''))
        elif ev.item.role == 'assistant':
            turn_id = bridge.last_committed_id or ''
            metrics.record('tts_end', turn_id, interrupted=ev.item.interrupted)
            spawn(publish('turn_finished', turn_id, interrupted=ev.item.interrupted))
            if ev.item.interrupted:
                metrics.record('barge_in', turn_id)
            spawn(pipeline.maintain(expected_user, metadata['child_id'], metadata['session_id']))

    @session.on('user_state_changed')
    def user_state(ev):
        if ev.new_state == 'speaking':
            metrics.record('user_speech_start')
        elif ev.old_state == 'speaking':
            metrics.record('user_speech_end')

    @session.on('user_input_transcribed')
    def transcription(ev):
        if ev.is_final:
            metrics.record('stt_final')

    @session.on('agent_state_changed')
    def state(ev):
        turn_id = bridge.last_committed_id or ''
        metrics.record('agent_state', turn_id, state=ev.new_state)
        if ev.new_state == 'speaking':
            metrics.record('audio_output_start', turn_id)
        spawn(publish('state', turn_id, state=ev.new_state))

    @session.on('eot_prediction')
    def prediction(ev):
        metrics.record('eot_prediction', probability=ev.probability, threshold=ev.threshold,
                       duration_ms=ev.inference_duration * 1000)

    @session.on('metrics_collected')
    def provider_metrics(ev):
        metric = ev.metrics
        if metric.type == 'eou_metrics':
            metrics.record('sdk_stt_latency', speech_id=metric.speech_id,
                           stage='stt', duration_ms=metric.transcription_delay * 1000)
            metrics.record('sdk_turn_latency', speech_id=metric.speech_id,
                           stage='turn', duration_ms=metric.end_of_utterance_delay * 1000)
        elif metric.type == 'tts_metrics':
            metrics.record('sdk_tts_latency', bridge.last_committed_id or '',
                           speech_id=metric.speech_id, ttfb_ms=metric.ttfb * 1000,
                           duration_ms=metric.duration * 1000, audio_duration_ms=metric.audio_duration * 1000)

    @session.on('error')
    def error(ev):
        error_type = getattr(ev.error, 'type', 'voice_error')
        status = getattr(getattr(ev.error, 'error', None), 'status_code', None)
        metrics.record('error', bridge.last_committed_id or '', error_type=error_type,
                       recoverable=getattr(ev.error, 'recoverable', False))
        code = ('stt_payment_required' if error_type == 'stt_error' and status == 402 else
                'provider_auth_required' if status in (401, 403) else
                'provider_config_error' if status in (400, 404) else 'voice_service_error')
        async def notify_error():
            try:
                await publish('error', code=code)
            finally:
                if status in (400, 401, 402, 403, 404):
                    shutdown_job(code)
        spawn(notify_error())

    async def shutdown():
        nonlocal cleanup_started
        if cleanup_started:
            return
        cleanup_started = True

        async def finish(stage, operation):
            try:
                await operation
            except Exception:
                # A failed cleanup step must not skip later resource releases or
                # leak provider exception bodies into the application's logs.
                logging.getLogger(__name__).warning('voice_cleanup_failed stage=%s', stage)

        # SDK callbacks run concurrently. Finish all session events before closing storage.
        try:
            await finish('session', session.aclose())
            await finish('bridge', bridge.aclose())
            if pending:
                await asyncio.gather(*list(pending), return_exceptions=True)
            await finish('memory', pipeline.maintain(
                expected_user, metadata['child_id'], metadata['session_id'], force_memory=True))
        finally:
            pipeline.store.close()

    ctx.add_shutdown_callback(shutdown)
    await ctx.connect()
    await session.start(agent=DadokAgent(bridge, metrics), room=ctx.room, record=False,
                        room_options=room_io.RoomOptions(participant_identity=metadata['identity'], text_input=False))


if __name__ == '__main__':
    cli.run_app(server)
