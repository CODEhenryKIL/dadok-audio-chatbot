"""Share the running voice demo through an authenticated temporary HTTPS entrance."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "dadok_chatbot"
QUICK_ORIGIN = re.compile(r"https://[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com", re.ASCII)


class StopRequested(Exception):
    pass


def extract_public_origin(log: str) -> str | None:
    """Accept only a complete HTTPS Quick Tunnel origin, never a suffix or path."""
    for candidate in re.findall(r"https://[^\s|\"'<>]+", log):
        if QUICK_ORIGIN.fullmatch(candidate):
            return candidate
    return None


def private_file(path: Path, *, truncate: bool = False):
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    flags |= os.O_TRUNC if truncate else os.O_APPEND
    fd = os.open(path, flags, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, "w" if truncate else "a", encoding="utf-8")


def save_private_json(path: Path, data: dict):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".share-", delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            json.dump(data, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fchmod(temporary.fileno(), 0o600)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


def assert_free_port(port: int):
    if not 1024 <= port <= 65535:
        raise ValueError("포트는 1024~65535 범위여야 합니다.")
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    except OSError:
        raise RuntimeError(f"127.0.0.1:{port} 포트가 사용 중입니다. 기존 프로세스는 종료하지 않습니다.") from None


def local_request(port: int, path: str, *, host: str | None = None):
    # Do not send local health checks or cookies through a configured HTTP proxy.
    request = Request(f"http://127.0.0.1:{port}{path}", headers={"Host": host} if host else {})
    return build_opener(ProxyHandler({})).open(request, timeout=2)


def verify_existing_demo(config: dict):
    port = config.get("web_port", 8766)
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("LAN 설정의 web_port가 올바르지 않습니다.")
    try:
        with local_request(port, "/health") as response:
            health = json.load(response)
        with local_request(port, "/api/voice/config") as response:
            voice = json.load(response)
    except (URLError, OSError, ValueError):
        raise RuntimeError("기존 로컬 음성 데모에 연결할 수 없습니다. run_voice.py 서버를 먼저 실행하세요.") from None
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise RuntimeError("로컬 웹 서버가 준비되지 않았습니다.")
    if not isinstance(voice, dict) or voice.get("ready") is not True:
        raise RuntimeError("로컬 음성 설정이 준비되지 않았습니다. 로컬 데모 설정을 확인하세요.")


def wait_for_origin(child: subprocess.Popen, log_path: Path, timeout: float = 45) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("Cloudflare 연결 프로세스가 종료되었습니다. runtime/tunnel/cloudflared.log를 확인하세요.")
        origin = extract_public_origin(log_path.read_text(encoding="utf-8", errors="replace"))
        if origin:
            return origin
        time.sleep(.2)
    raise RuntimeError("Cloudflare HTTPS 주소 발급 시간이 초과되었습니다.")


def wait_for_gateway(children: list[subprocess.Popen], port: int, origin: str, timeout: float = 15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(child.poll() is not None for child in children):
            raise RuntimeError("공유 연결 구성 프로세스가 종료되었습니다. runtime/tunnel 로그를 확인하세요.")
        try:
            with local_request(port, "/", host=urlsplit(origin).netloc) as response:
                if response.status == 200:
                    return
        except (URLError, OSError):
            pass
        time.sleep(.2)
    raise RuntimeError("접속 코드 게이트웨이 시작 시간이 초과되었습니다.")


def stop_children(children: list[subprocess.Popen]):
    """Only signal our own child handles; reap them within a shared ten-second budget."""
    for sig, seconds in ((signal.SIGINT, 6), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        active = [child for child in reversed(children) if child.poll() is None]
        if not active:
            break
        deadline = time.monotonic() + seconds
        for child in active:
            try:
                child.send_signal(sig)
            except ProcessLookupError:
                pass
        for child in active:
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tunnel-port", type=int, default=8768)
    parser.add_argument("--metrics-port", type=int, default=20241)
    args = parser.parse_args(argv)
    binary = ROOT / "bin" / "cloudflared"
    config_path = PROJECT / "runtime" / "lan" / "config.json"
    runtime = PROJECT / "runtime" / "tunnel"
    children = []
    logs = []
    lock = None
    locked = False
    previous_signals = {}

    def stop(_signum, _frame):
        raise StopRequested()

    try:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError("bin/cloudflared 실행 파일이 없습니다. 검증된 공식 바이너리를 먼저 설치하세요.")
        if not config_path.is_file():
            raise RuntimeError("runtime/lan/config.json이 없습니다. 기존 LAN 데모를 먼저 준비하세요.")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise RuntimeError("LAN 설정 파일을 읽을 수 없습니다.") from None
        if not isinstance(config, dict) or not re.fullmatch(r"[0-9]{8}", str(config.get("access_code", ""))):
            raise RuntimeError("LAN 설정에 올바른 8자리 접속 코드가 필요합니다.")
        verify_existing_demo(config)
        if args.tunnel_port == args.metrics_port:
            raise ValueError("터널과 상태 확인 포트는 달라야 합니다.")
        for port in (args.tunnel_port, args.metrics_port):
            assert_free_port(port)
        runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        runtime.chmod(0o700)
        lock = private_file(runtime / "share.lock")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError:
            raise RuntimeError("이미 HTTPS 공유가 실행 중입니다. 기존 공유 프로세스는 그대로 유지합니다.") from None
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_signals[sig] = signal.signal(sig, stop)
        log_path = runtime / "cloudflared.log"
        tunnel_log = private_file(log_path, truncate=True)
        logs.append(tunnel_log)
        tunnel = subprocess.Popen([str(binary), "tunnel", "--no-autoupdate", "--protocol", "http2",
                                   "--url", f"http://127.0.0.1:{args.tunnel_port}",
                                   "--metrics", f"127.0.0.1:{args.metrics_port}"],
                                  cwd=ROOT, stdout=tunnel_log, stderr=subprocess.STDOUT)
        children.append(tunnel)
        origin = wait_for_origin(tunnel, log_path)
        gateway_log = private_file(runtime / "gateway.log", truncate=True)
        logs.append(gateway_log)
        gateway = subprocess.Popen([sys.executable, "-m", "dadok.voice_lan", "--config", str(config_path),
                                    "--public-origin", origin, "--tunnel-port", str(args.tunnel_port)],
                                   cwd=PROJECT, stdout=gateway_log, stderr=subprocess.STDOUT)
        children.append(gateway)
        wait_for_gateway(children, args.tunnel_port, origin)
        pids = {"launcher": os.getpid(), "children": [child.pid for child in children]}
        save_private_json(runtime / "share-pids.json", pids)
        save_private_json(runtime / "connection.json", {"url": origin, "tunnel_port": args.tunnel_port,
                          "metrics_port": args.metrics_port, **pids})
        print(f"인증서 설치 없이 접속: {origin}", flush=True)
        print(f"접속 코드: {config['access_code']}", flush=True)
        print("임시 공유 주소입니다. 이 실행을 종료하면 닫히며, 다시 실행하면 주소가 달라집니다. (종료: Ctrl+C)", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise RuntimeError("공유 연결 프로세스가 종료되어 공개 터널도 닫습니다.")
    except (StopRequested, KeyboardInterrupt):
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"공유 시작 실패: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for sig in previous_signals:
            signal.signal(sig, signal.SIG_IGN)
        stop_children(children)
        for log in logs:
            log.close()
        if locked:
            (runtime / "share-pids.json").unlink(missing_ok=True)
            (runtime / "connection.json").unlink(missing_ok=True)
        if lock:
            lock.close()
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
