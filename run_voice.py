"""Run the local LiveKit server, voice worker and existing web chatbot together."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'dadok_chatbot'
sys.path.insert(0, str(PROJECT))
from dadok.voice_config import VoiceSettings, load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='명시적 합성 육아기록 (API는 실제 호출)')
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--lan', metavar='PRIVATE_IP', help='같은 네트워크용 HTTPS 주소 (이 Mac의 사설 IPv4)')
    parser.add_argument('--https-port', type=int, default=8443)
    parser.add_argument('--setup-port', type=int, default=8767)
    args = parser.parse_args()
    load_environment()
    if args.demo:
        path = PROJECT / 'tests' / 'fixtures' / 'babylog_snapshot.synthetic.json'
        data = json.loads(path.read_text())
        os.environ.update(DADOK_RECORD_SNAPSHOT=str(path), DADOK_AUTH_USER_ID=data['user_id'], DADOK_CHILD_ID=data['children'][0]['id'])
    settings = VoiceSettings.from_env()
    port = args.port or int(os.getenv('VOICE_WEB_PORT', '8766'))
    children = []
    logs = []
    runtime = PROJECT / 'runtime'
    runtime.mkdir(exist_ok=True)
    lan_config = None
    livekit_config = ROOT / 'livekit.yaml'
    if args.lan:
        from dadok.voice_lan_setup import prepare_lan, private_ipv4
        lan_ip = private_ipv4(args.lan)
        # Fail before changing any running service if this address is not ours.
        with socket.socket() as sock:
            sock.bind((lan_ip, 0))
        for candidate in (args.https_port, args.setup_port):
            with socket.socket() as sock:
                sock.bind((lan_ip, candidate))
        lan_path, livekit_config, lan_config, fingerprint = prepare_lan(
            ROOT, lan_ip, port, args.https_port, args.setup_port)
        # LAN participants must not be able to forge room tokens with devkey/secret.
        os.environ.update(LIVEKIT_API_KEY='lan-' + secrets.token_hex(8),
                          LIVEKIT_API_SECRET=secrets.token_urlsafe(48))
        settings = VoiceSettings.from_env()
    def launch(name, command, cwd, env=None):
        log = (runtime / (name + '.log')).open('a')
        logs.append(log)
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        return child
    try:
        if settings.enabled:
            settings.validate()
            binary = ROOT / 'bin' / 'livekit-server'
            if not binary.is_file():
                raise SystemExit('bin/livekit-server가 없습니다. README의 설치 단계를 실행하세요.')
            # All services are local; avoid interrupting an existing process on these ports.
            for candidate in (7880, port):
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1', candidate)) == 0:
                        raise SystemExit(f'{candidate} 포트가 사용 중입니다. 기존 서버를 확인하세요.')
            env = {**os.environ, 'LIVEKIT_KEYS': f'{settings.livekit_key}: {settings.livekit_secret}'}
            launch('livekit', [str(binary), '--config', str(livekit_config)], ROOT, env)
            for _ in range(100):
                if children[0].poll() is not None:
                    raise RuntimeError('LiveKit 서버가 종료되었습니다.')
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1', 7880)) == 0:
                        break
                time.sleep(.1)
            else:
                raise RuntimeError('LiveKit 서버 시작 시간 초과')
            launch('voice-agent', [sys.executable, '-m', 'dadok.voice_agent', 'start'], PROJECT)
        launch('voice-web', [sys.executable, '-m', 'uvicorn', 'dadok.voice_server:app',
                            '--host', '127.0.0.1', '--port', str(port)], PROJECT)
        if lan_config:
            launch('voice-lan', [sys.executable, '-m', 'dadok.voice_lan', '--config', str(lan_path)], PROJECT)
        (runtime / 'voice-pids.json').write_text(json.dumps({'launcher': os.getpid(), 'children': [c.pid for c in children]}))
        print(f'다독 음성 실험: http://127.0.0.1:{port} (종료: Ctrl+C)', flush=True)
        if lan_config:
            print(f'같은 네트워크: https://{lan_ip}:{args.https_port}', flush=True)
            print(f'인증서 설치 안내: http://{lan_ip}:{args.setup_port}', flush=True)
            print(f'접속 코드: {lan_config["access_code"]}', flush=True)
            print(f'인증서 SHA-256: {fingerprint}', flush=True)
        print(f'프로세스 로그: {runtime}', flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise RuntimeError('구성 프로세스가 종료되어 실험 서버를 정리합니다.')
    except KeyboardInterrupt:
        pass
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
        for child in reversed(children):
            try:
                child.wait(timeout=65)
            except subprocess.TimeoutExpired:
                child.terminate()
        for log in logs:
            log.close()
        (runtime / 'voice-pids.json').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
