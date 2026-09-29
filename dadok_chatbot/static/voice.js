/* The browser receives a room token only; provider API keys stay on the server. */
(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const chat = window.dadokChat;
  const startButton = el('voice-start');
  const stopButton = el('voice-stop');
  const voiceChoice = el('voice-choice');
  let voicesAvailable = false;
  const status = el('voice-status');
  const errorBox = el('voice-error');
  const audioContainer = el('voice-audio');
  const attachedAudio = new WeakMap();
  const turns = new Map();
  let room = null;
  let phase = 'idle';
  let generation = 0;
  let request = null;
  let lastTranscriptSequence = 0;
  let transcriptSequence = 0;
  const labels = {
    listening: '듣는 중', thinking: '처리 중', processing: '처리 중',
    speaking: '말하는 중', initializing: '음성 준비 중…', idle: '대기',
  };

  function showError(message = '') {
    errorBox.textContent = message;
    errorBox.hidden = !message;
  }

  function setPhase(value, label) {
    phase = value;
    chat.setVoiceLocked(value !== 'idle');
    voiceChoice.disabled = value !== 'idle' || !voicesAvailable;
    stopButton.disabled = value === 'idle' || value === 'stopping';
    status.textContent = label;
  }

  function turnFor(id) {
    if (!id || typeof id !== 'string') return null;
    if (!turns.has(id)) {
      turns.set(id, {});
      if (turns.size > 32) turns.delete(turns.keys().next().value);
    }
    return turns.get(id);
  }

  function markInterrupted(turn) {
    if (!turn || turn.playbackFinished || turn.cancelled) return;
    turn.cancelled = true;
    if (turn.answer) {
      const note = document.createElement('div');
      note.className = 'meta voice-interrupted';
      note.textContent = '답변이 중단되었어요.';
      turn.answer.append(note);
    }
  }

  function handleEvent(data) {
    if (!data || typeof data !== 'object') return;
    const turn = turnFor(data.turn_id);
    switch (data.event) {
      case 'state':
        if (phase === 'connected' && labels[data.state]) status.textContent = labels[data.state];
        break;
      case 'user_final':
        if (typeof data.question !== 'string') break;
        el('voice-final').textContent = data.question;
        el('voice-transcript').textContent = data.question;
        el('voice-turn').textContent = '질문 확정 · 답변 처리 중';
        if (turn && !turn.question) turn.question = chat.addMessage(data.question, 'user');
        break;
      case 'answer':
        if (typeof data.text !== 'string' || (turn && turn.cancelled)) break;
        el('voice-answer').textContent = data.text;
        el('voice-turn').textContent = '답변 생성 · 음성 출력 중';
        if (turn && !turn.answer) turn.answer = chat.addMessage(data.text);
        break;
      case 'turn_committed':
        if (turn) turn.committed = true;
        el('voice-turn').textContent = '대화 기록 저장 · 음성 출력 중';
        break;
      case 'turn_finished':
        if (data.interrupted) {
          markInterrupted(turn);
          el('voice-turn').textContent = '음성 출력 중단 · 다음 질문 대기';
        } else {
          if (turn) turn.playbackFinished = true;
          el('voice-turn').textContent = '턴 완료 · 음성 출력 종료';
        }
        break;
      case 'turn_cancelled':
        markInterrupted(turn);
        el('voice-turn').textContent = '턴 중단 · 다음 질문 대기';
        break;
      case 'error':
        showError(data.code === 'stt_payment_required'
          ? 'ElevenLabs 사용 한도 또는 크레딧을 확인한 후 다시 시작해 주세요.'
          : data.code === 'provider_auth_required'
            ? '음성 서비스 인증에 실패했습니다. API 키와 사용 권한을 확인해 주세요.'
            : data.code === 'provider_config_error'
              ? '음성 서비스 설정을 확인해 주세요. API 키, 모델 또는 Voice ID가 올바르지 않습니다.'
            : '음성 처리 중 문제가 발생했어요. 잠시 후 다시 말하거나 음성 대화를 다시 시작해 주세요.');
        el('voice-turn').textContent = '처리 오류';
        if (['stt_payment_required', 'provider_auth_required', 'provider_config_error'].includes(data.code)) {
          void stop('음성 설정 확인 필요');
        }
        break;
    }
  }

  async function readConfiguration(signal) {
    const response = await fetch('/api/voice/config', { signal, cache: 'no-store' });
    if (!response.ok) throw new Error('configuration');
    return response.json();
  }

  function updateVoices(config) {
    const previous = voiceChoice.value;
    const voices = Array.isArray(config.voices) ? config.voices.filter(
      voice => voice && typeof voice.id === 'string' && typeof voice.name === 'string') : [];
    voiceChoice.replaceChildren();
    for (const voice of voices) {
      const option = document.createElement('option');
      option.value = voice.id;
      option.textContent = voice.name;
      voiceChoice.append(option);
    }
    voiceChoice.value = voices.some(voice => voice.id === previous) ? previous : config.tts_voice || '';
    voicesAvailable = voices.length > 0;
    voiceChoice.disabled = phase !== 'idle' || !voicesAvailable;
  }

  function configurationMessage(config) {
    if (!config.enabled) return '음성 기능이 꺼져 있어요. 서버의 음성 설정을 확인해 주세요.';
    // Only setting names are displayed, never a raw provider response or token.
    const missing = Array.isArray(config.missing)
      ? config.missing.filter(name => typeof name === 'string' && /^[A-Z][A-Z0-9_]{0,63}$/.test(name)) : [];
    return missing.length
      ? `음성 설정이 필요해요: ${missing.join(', ')}. 설정 후 다시 시작해 주세요.`
      : '음성 서버가 아직 준비되지 않았어요. 서버 설정을 확인한 후 다시 시작해 주세요.';
  }

  async function cleanup(target) {
    if (!target) return;
    // Stop capture immediately, including when connect/disconnect is still pending.
    for (const publication of target.localParticipant.trackPublications.values()) {
      try { publication.track?.stop(); } catch (_) { /* Continue releasing other tracks. */ }
    }
    try { await target.disconnect(true); } catch (_) { /* Local tracks are already stopped. */ }
    for (const audio of attachedAudio.get(target) || []) {
      audio.pause();
      audio.srcObject = null;
      audio.remove();
    }
    attachedAudio.delete(target);
  }

  async function stop(message = '음성 대화가 중지되었어요.') {
    const stopGeneration = ++generation;
    request?.abort();
    request = null;
    const oldRoom = room;
    room = null;
    setPhase('stopping', '마이크 종료 중…');
    await cleanup(oldRoom);
    if (generation !== stopGeneration) return;
    for (const turn of turns.values()) markInterrupted(turn);
    setPhase('idle', message);
    el('voice-turn').textContent = '대기';
  }

  async function start() {
    if (phase !== 'idle' || chat.isBusy()) return;
    showError();
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      showError('마이크를 사용하려면 localhost 또는 HTTPS 주소로 접속해 주세요.');
      return;
    }
    const sdk = window.LivekitClient;
    if (!sdk?.Room) {
      showError('음성 연결 모듈을 불러오지 못했어요. 서버 설치 상태를 확인한 뒤 새로고침해 주세요.');
      return;
    }
    const ownGeneration = ++generation;
    const candidate = new sdk.Room({ adaptiveStream: true, dynacast: true,
      audioCaptureDefaults: { echoCancellation: true } });
    attachedAudio.set(candidate, new Set());
    room = candidate;
    request = new AbortController();
    const signal = request.signal;
    const isCurrent = () => generation === ownGeneration && room === candidate;
    const guard = async () => {
      if (isCurrent()) return true;
      await cleanup(candidate);
      return false;
    };
    setPhase('connecting', '음성 연결 중…');
    turns.clear();
    el('voice-transcript').textContent = '—';
    el('voice-final').textContent = '—';
    el('voice-answer').textContent = '—';
    el('voice-turn').textContent = '연결 중';

    candidate.on(sdk.RoomEvent.TrackSubscribed, (track, _publication, participant) => {
      if (!isCurrent() || !participant?.isAgent || track.kind !== sdk.Track.Kind.Audio) return;
      const audio = track.attach();
      attachedAudio.get(candidate)?.add(audio);
      audioContainer.append(audio);
      audio.play().catch(() => {
        if (isCurrent()) showError('브라우저가 소리 재생을 막았어요. 음성 대화를 중지한 뒤 다시 시작해 주세요.');
      });
    });
    candidate.on(sdk.RoomEvent.TrackUnsubscribed, track => {
      for (const element of track.detach()) {
        attachedAudio.get(candidate)?.delete(element);
        element.remove();
      }
    });
    candidate.on(sdk.RoomEvent.DataReceived, (payload, participant, _kind, topic) => {
      if (!isCurrent() || !participant?.isAgent || topic !== 'dadok.voice' || payload.byteLength > 65536) return;
      try { handleEvent(JSON.parse(new TextDecoder().decode(payload))); } catch (_) { /* Ignore invalid data. */ }
    });
    candidate.on(sdk.RoomEvent.ParticipantAttributesChanged, (attributes, participant) => {
      if (!isCurrent() || !participant.isAgent || phase !== 'connected') return;
      const agentState = attributes['lk.agent.state'];
      if (labels[agentState]) status.textContent = labels[agentState];
    });
    candidate.on(sdk.RoomEvent.ParticipantDisconnected, participant => {
      if (isCurrent() && participant.isAgent) {
        if (errorBox.hidden) showError('음성 도우미 연결이 종료되었어요. 다시 시작해 주세요.');
        void stop('음성 연결 종료');
      }
    });
    candidate.on(sdk.RoomEvent.Reconnecting, () => {
      if (isCurrent()) status.textContent = '연결 복구 중…';
    });
    candidate.on(sdk.RoomEvent.Reconnected, () => {
      if (isCurrent()) status.textContent = '듣는 중';
    });
    candidate.on(sdk.RoomEvent.Disconnected, () => {
      if (isCurrent()) {
        showError('음성 연결이 종료되었어요. 다시 시작할 수 있어요.');
        void stop('음성 연결 종료');
      }
    });
    candidate.registerTextStreamHandler('lk.transcription', async (reader, participantInfo) => {
      const sequence = ++transcriptSequence;
      try {
        const text = await reader.readAll();
        if (!isCurrent() || participantInfo.identity !== candidate.localParticipant.identity || sequence < lastTranscriptSequence) return;
        lastTranscriptSequence = sequence;
        el('voice-transcript').textContent = text;
        if (reader.info.attributes?.['lk.transcription_final'] === 'true') {
          el('voice-final').textContent = text;
        }
      } catch (_) { /* A cancelled transcription is expected on disconnect. */ }
    });

    try {
      // Unlock audio from this explicit click; microphone access happens only in this handler.
      await candidate.startAudio();
      if (!await guard()) return;
      const config = await readConfiguration(signal);
      if (!await guard()) return;
      updateVoices(config);
      if (!config.enabled || !config.ready) {
        showError(configurationMessage(config));
        await stop('음성 설정 필요');
        return;
      }
      const response = await fetch('/api/voice/token', {
        method: 'POST', signal, headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ child_id: chat.getChild(), session_id: chat.getSession(),
          tts_voice: voiceChoice.value || undefined }),
      });
      if (!response.ok) throw new Error('token');
      const connection = await response.json();
      if (!await guard()) return;
      if (!connection.url || !connection.token || !connection.session_id) throw new Error('token');
      chat.setSession(connection.session_id);
      await candidate.connect(connection.url, connection.token);
      if (!await guard()) return;
      await candidate.localParticipant.setMicrophoneEnabled(true);
      if (!await guard()) return;
      request = null;
      setPhase('connected', '듣는 중');
      el('voice-turn').textContent = '질문 대기';
    } catch (error) {
      if (!isCurrent()) { await cleanup(candidate); return; }
      const permissionDenied = ['NotAllowedError', 'PermissionDeniedError'].includes(error.name);
      const missingMicrophone = ['NotFoundError', 'DevicesNotFoundError'].includes(error.name);
      showError(permissionDenied
        ? '마이크 권한이 필요해요. 브라우저에서 마이크를 허용한 후 다시 시작해 주세요.'
        : missingMicrophone
          ? '마이크를 찾지 못했어요. 마이크를 연결한 후 다시 시작해 주세요.'
          : '음성 연결에 실패했어요. 음성 서버와 마이크 연결을 확인한 후 다시 시작해 주세요.');
      await stop('음성 연결 실패');
    }
  }

  startButton.addEventListener('click', () => { void start(); });
  stopButton.addEventListener('click', () => { void stop(); });
  window.addEventListener('pagehide', () => { void stop(); });
  readConfiguration().then(config => {
    if (phase !== 'idle') return;
    updateVoices(config);
    status.textContent = config.enabled && config.ready ? '음성 대화 준비 완료' : '음성 설정 필요';
    if (!config.enabled || !config.ready) showError(configurationMessage(config));
  }).catch(() => {
    if (phase === 'idle') status.textContent = '음성 서버 연결 확인 필요';
  });
})();
