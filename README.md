# Dadok Audio Chatbot

Korean voice chatbot using LiveKit and ElevenLabs. Public source snapshot of the audio chatbot project only.

- STT: ElevenLabs Scribe v2 Realtime
- TTS: ElevenLabs Flash v2.5 streaming
- Silero VAD, LiveKit Turn Detection, barge-in
- Voice response instructions also apply to keyboard chat in the voice demo.

## Setup

Requires Python 3.12 and Node.js.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r dadok_chatbot/requirements.txt -r requirements-voice.txt
npm ci
cp .env.example .env
python scripts/install_livekit.py
(cd dadok_chatbot && python -m dadok.voice_agent download-files)
python run_voice.py --demo
```

Configure your own API keys and Voice ID in `.env`. Local URL: http://127.0.0.1:8766 . Provider API usage can incur charges.

## Public export scope

Runtime source, required prompts and configuration examples are included. Development notes and test/verification tools are omitted. Secrets, conversation/record databases, logs, certificates, installed dependencies, binaries and historical reports are excluded. Synthetic test records are included.

Third-party reference documents, RAG corpus/vector data and extracted reference_context.json passages are excluded pending redistribution permission. The reference_context.json file contains only an empty structure. Supply your lawfully obtained professional RAG bundle and reference context separately to enable grounded reference retrieval. This export is not a self-contained RAG dataset.

The original local project is unchanged. No new full runtime verification was performed for this public export.
