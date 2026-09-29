"""Authenticated, TLS-only LAN entrance to the loopback voice demo."""
from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field, replace
import hashlib
import hmac
import ipaddress
import json
from pathlib import Path
import re
import secrets
import signal
import ssl
import time

from aiohttp import ClientError, ClientSession, ClientTimeout, DummyCookieJar, WSMsgType, web
from yarl import URL


COOKIE = "__Host-dadok-lan"
CERTIFICATES = {
    "dadok-local-ca.crt": "application/x-x509-ca-cert",
    "dadok-local-ca.cer": "application/pkix-cert",
    "dadok-local-ca.mobileconfig": "application/x-apple-aspen-config",
}
SIGNAL_PATHS = {"/livekit/rtc", "/livekit/rtc/v1", "/livekit/rtc/validate",
                "/livekit/rtc/v1/validate"}
REQUEST_HEADERS = {"accept", "accept-language", "content-type", "last-event-id",
                   "range", "if-range", "if-none-match", "if-modified-since"}
RESPONSE_HEADERS = {"content-type", "content-encoding", "cache-control", "etag", "last-modified", "vary",
                    "content-range", "accept-ranges", "content-disposition"}
SESSION = web.AppKey("upstream_session", ClientSession)
SOCKETS = web.AppKey("active_sockets", set)


@dataclass(frozen=True)
class GatewayConfig:
    lan_ip: str
    cert_dir: Path
    access_code: str = field(repr=False)
    cookie_token: str = field(repr=False)
    https_port: int = 8443
    setup_port: int = 8767
    web_port: int = 8766
    public_origin: str | None = None

    def __post_init__(self):
        address = ipaddress.ip_address(self.lan_ip)
        networks = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
        if address.version != 4 or not any(address in ipaddress.ip_network(net) for net in networks):
            raise ValueError("lan_ip must be a private LAN IPv4 address")
        if not Path(self.cert_dir).is_absolute():
            raise ValueError("cert_dir must be absolute")
        object.__setattr__(self, "cert_dir", Path(self.cert_dir))
        if not re.fullmatch(r"[0-9]{8}", self.access_code):
            raise ValueError("access_code must contain eight digits")
        if len(self.cookie_token) < 32 or not re.fullmatch(r"[A-Za-z0-9_-]+", self.cookie_token):
            raise ValueError("cookie_token must be a strong URL-safe random token")
        for port in (self.https_port, self.setup_port, self.web_port):
            if type(port) is not int or not 1024 <= port <= 65535:
                raise ValueError("ports must be integers between 1024 and 65535")
        if len({self.https_port, self.setup_port, self.web_port}) != 3:
            raise ValueError("ports must be distinct")
        if self.public_origin is not None and (not isinstance(self.public_origin, str) or not re.fullmatch(
            r"https://[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com", self.public_origin
        )):
            raise ValueError("public_origin must be the exact HTTPS Quick Tunnel origin")

    @classmethod
    def from_file(cls, path: str | Path) -> GatewayConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def origin(self) -> str:
        return self.public_origin or f"https://{self.lan_ip}:{self.https_port}"


class LoginThrottle:
    """Bound both guessing rate and memory usage; never trust forwarding headers."""

    def __init__(self):
        self.failures: OrderedDict[str, tuple[float, int]] = OrderedDict()

    def blocked(self, peer: str) -> bool:
        now = time.monotonic()
        for key, (start, _) in list(self.failures.items()):
            if now - start >= 60:
                del self.failures[key]
        return self.failures.get(peer, (now, 0))[1] >= 5

    def failed(self, peer: str):
        start, count = self.failures.get(peer, (time.monotonic(), 0))
        self.failures[peer] = (start, count + 1)
        self.failures.move_to_end(peer)
        if len(self.failures) > 1024:
            self.failures.popitem(last=False)


def _host_matches(request: web.Request, expected: str) -> bool:
    return request.headers.getall("Host", []) == [expected]


def _same_origin(request: web.Request, expected: str) -> bool:
    return request.headers.getall("Origin", []) == [expected]


def _authenticated(request: web.Request, config: GatewayConfig) -> bool:
    value = request.cookies.get(COOKIE, "")
    return hmac.compare_digest(value.encode(), config.cookie_token.encode())


def _page(title: str, content: str) -> web.Response:
    return web.Response(text=f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:17px/1.6 system-ui;max-width:680px;margin:48px auto;padding:0 20px}}
input,button{{font:inherit;padding:10px;margin:8px 0}}code{{overflow-wrap:anywhere}}
</style><h1>{title}</h1>{content}</html>""", content_type="text/html", headers={
        "Cache-Control": "no-store", "Referrer-Policy": "same-origin",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'",
    })


def _login_page() -> web.Response:
    return _page("다독 음성 데모", """<p>Mac에 표시된 8자리 접속 코드를 입력하세요.</p>
<form method="post" action="/login"><label>접속 코드<br>
<input name="code" type="password" inputmode="numeric" pattern="[0-9]{8}"
minlength="8" maxlength="8" autocomplete="off" required autofocus></label><br>
<button type="submit">데모 열기</button></form>""")


def create_setup_app(config: GatewayConfig) -> web.Application:
    """HTTP can deliver public CA files only, never the demo or login."""
    async def setup(request: web.Request):
        if not _host_matches(request, f"{config.lan_ip}:{config.setup_port}"):
            raise web.HTTPForbidden(text="허용되지 않은 접속 주소입니다.")
        if request.method not in {"GET", "HEAD"}:
            raise web.HTTPMethodNotAllowed(request.method, ["GET", "HEAD"])
        name = request.path.removeprefix("/")
        if name in CERTIFICATES:
            path = config.cert_dir / name
            if not path.is_file():
                raise web.HTTPNotFound()
            return web.FileResponse(path, headers={
                "Content-Type": CERTIFICATES[name], "Cache-Control": "no-store",
                "Content-Disposition": f'attachment; filename="{name}"',
                "X-Content-Type-Options": "nosniff",
            })
        if request.path != "/":
            raise web.HTTPNotFound()
        fingerprint = hashlib.sha256((config.cert_dir / "dadok-local-ca.cer").read_bytes()).hexdigest()
        fingerprint = ":".join(fingerprint[i:i + 2].upper() for i in range(0, 64, 2))
        return _page("같은 네트워크에서 음성 데모 열기", f"""
<p>마이크 사용을 위해 이 Mac의 로컬 인증서를 한 번 설치하고 신뢰해야 합니다.
아래 SHA-256 지문이 Mac에 표시된 값과 일치하는지 확인하세요.</p><code>{fingerprint}</code>
<ul><li>Mac: <a href="/dadok-local-ca.cer">CA 인증서</a>를 다운로드하고 키체인 접근의 <b>로그인</b> 키체인에 추가하세요.
<b>Dadok Local Demo CA</b>를 두 번 클릭해 <b>신뢰</b>를 펼친 뒤,
<b>SSL(Secure Sockets Layer) → 항상 신뢰</b>로 설정하고 창을 닫아 저장하세요.</li>
<li>Windows: <a href="/dadok-local-ca.cer">CA 인증서</a>를 열고 인증서 설치 → <b>현재 사용자</b>를 선택하세요.
모든 인증서를 다음 저장소에 저장 → 찾아보기 → <b>신뢰할 수 있는 루트 인증 기관</b>을 선택하고 설치를 마치세요.</li>
<li>iPhone / iPad: Safari에서 <a href="/dadok-local-ca.mobileconfig">설정 프로파일</a>을 받으세요.
설정 → 일반 → VPN 및 기기 관리에서 설치한 뒤, 일반 → 정보 → 인증서 신뢰 설정에서 이 인증서의 신뢰를 켜세요.</li>
<li>Android: <a href="/dadok-local-ca.crt">CA 인증서</a>를 받고 설정의 인증서 설치 → CA 인증서에서 설치하세요.
기종마다 메뉴 이름이 다를 수 있습니다.</li>
</ul>
<p>설치 후 <a href="{config.origin}/">HTTPS 음성 데모 열기</a>에서 Mac의 접속 코드를 입력하고 마이크를 허용하세요.
인증서 경고가 남아 있다면 신뢰 설정을 확인하세요.</p>
<p>실험이 끝나면 설치한 다독 로컬 CA 인증서 또는 프로파일을 삭제할 수 있습니다.</p>""")

    app = web.Application(client_max_size=1024)
    app.router.add_route("*", "/{path:.*}", setup)
    return app


def create_gateway(config: GatewayConfig) -> web.Application:
    throttle = LoginThrottle()

    @web.middleware
    async def boundary(request: web.Request, handler):
        if not _host_matches(request, config.origin.removeprefix("https://")):
            raise web.HTTPForbidden(text="허용되지 않은 접속 주소입니다.")
        is_websocket = request.headers.get("Upgrade", "").lower() == "websocket"
        if (request.method not in {"GET", "HEAD"} or is_websocket) and not _same_origin(request, config.origin):
            raise web.HTTPForbidden(text="같은 주소의 데모 페이지에서 다시 접속하세요.")
        if request.path != "/login" and not _authenticated(request, config):
            if request.method == "GET" and request.path == "/":
                return _login_page()
            raise web.HTTPUnauthorized(text="접속 코드를 입력해 주세요.")
        try:
            return await handler(request)
        except (ClientError, TimeoutError):
            raise web.HTTPBadGateway(text="Mac의 음성 서버에 연결할 수 없습니다.") from None

    async def login(request: web.Request):
        if request.method == "GET":
            return _login_page()
        if request.method != "POST":
            raise web.HTTPMethodNotAllowed(request.method, ["GET", "POST"])
        peer = request.remote or "unknown"
        if throttle.blocked(peer):
            raise web.HTTPTooManyRequests(text="잠시 후 다시 시도하세요.", headers={"Retry-After": "60"})
        if request.content_length is not None and request.content_length > 1024:
            raise web.HTTPRequestEntityTooLarge(max_size=1024, actual_size=request.content_length)
        raw = bytearray()
        async for chunk in request.content.iter_chunked(1025):
            raw.extend(chunk)
            if len(raw) > 1024:
                raise web.HTTPRequestEntityTooLarge(max_size=1024, actual_size=len(raw))
        from urllib.parse import parse_qs
        try:
            form = parse_qs(raw.decode("utf-8", errors="replace"), max_num_fields=8)
        except ValueError:
            raise web.HTTPBadRequest(text="올바른 접속 코드를 입력하세요.") from None
        submitted = form.get("code", [""])
        code = submitted[0] if len(submitted) == 1 else ""
        if not hmac.compare_digest(code.encode(), config.access_code.encode()):
            throttle.failed(peer)
            raise web.HTTPUnauthorized(text="접속 코드가 일치하지 않습니다.")
        throttle.failures.pop(peer, None)
        response = web.Response(status=303, headers={"Location": "/", "Cache-Control": "no-store"})
        response.set_cookie(COOKIE, config.cookie_token, secure=True, httponly=True,
                            samesite="Strict", path="/", max_age=12 * 60 * 60)
        return response

    async def proxy(request: web.Request):
        is_signal = request.path in SIGNAL_PATHS
        if request.path.startswith("/livekit") and not is_signal:
            raise web.HTTPNotFound()
        if is_signal and request.method != "GET":
            raise web.HTTPMethodNotAllowed(request.method, ["GET"])
        if is_signal:
            path = request.path.removeprefix("/livekit")
            upstream = "http://127.0.0.1:7880"
        else:
            path = request.rel_url.raw_path
            upstream = f"http://127.0.0.1:{config.web_port}"
        # Use the validated canonical signal path, not a potentially escaped raw path.
        url = URL(upstream + path + ("?" + request.rel_url.raw_query_string if request.query_string else ""), encoded=True)
        headers = {key: value for key, value in request.headers.items() if key.lower() in REQUEST_HEADERS}
        headers["Origin"] = upstream
        headers["Host"] = upstream.removeprefix("http://")
        session = request.app[SESSION]
        if request.headers.get("Upgrade", "").lower() == "websocket":
            if not is_signal or request.path.endswith("/validate"):
                raise web.HTTPNotFound()
            return await _relay_websocket(request, session, url, headers)
        body = await request.read()
        async with session.request(request.method, url, headers=headers,
                                   data=body, allow_redirects=False) as response:
            response_headers = {key: value for key, value in response.headers.items()
                                if key.lower() in RESPONSE_HEADERS}
            response_headers.update({"Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff",
                                     "Cache-Control": "no-store"})
            if request.path == "/api/voice/token" and response.status == 200:
                data = await response.json()
                data["url"] = config.origin.replace("https://", "wss://", 1) + "/livekit"
                return web.json_response(data, headers={"Cache-Control": "no-store"})
            outgoing = web.StreamResponse(status=response.status, headers=response_headers)
            await outgoing.prepare(request)
            async for chunk in response.content.iter_chunked(64 * 1024):
                await outgoing.write(chunk)
            await outgoing.write_eof()
            return outgoing

    async def client_context(app: web.Application):
        async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=5),
                                 skip_auto_headers={"Accept-Encoding"},
                                 auto_decompress=False, cookie_jar=DummyCookieJar()) as session:
            app[SESSION] = session
            yield

    async def close_sockets(app: web.Application):
        await asyncio.gather(*(socket.close(code=1001) for socket in list(app[SOCKETS])), return_exceptions=True)

    app = web.Application(middlewares=[boundary], client_max_size=1024 * 1024)
    app[SOCKETS] = set()
    app.cleanup_ctx.append(client_context)
    app.on_shutdown.append(close_sockets)
    app.router.add_route("*", "/login", login)
    app.router.add_route("*", "/{path:.*}", proxy)
    return app


async def _relay_websocket(request: web.Request, session: ClientSession, url: URL, headers: dict) -> web.WebSocketResponse:
    async with session.ws_connect(url, headers=headers, heartbeat=30, max_msg_size=4 * 1024 * 1024) as upstream:
        downstream = web.WebSocketResponse(heartbeat=30, max_msg_size=4 * 1024 * 1024)
        await downstream.prepare(request)
        request.app[SOCKETS].add(downstream)

        async def relay(source, destination):
            async for message in source:
                if message.type == WSMsgType.TEXT:
                    await destination.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await destination.send_bytes(message.data)
                elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                    break

        tasks = [asyncio.create_task(relay(upstream, downstream)),
                 asyncio.create_task(relay(downstream, upstream))]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await downstream.close()
            request.app[SOCKETS].discard(downstream)
        return downstream


async def serve(config: GatewayConfig, tunnel_port: int = 8768):
    if not 1024 <= tunnel_port <= 65535:
        raise ValueError("tunnel_port must be between 1024 and 65535")
    runners = [web.AppRunner(create_gateway(config), access_log=None, handler_cancellation=True)]
    if not config.public_origin:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.minimum_version = ssl.TLSVersion.TLSv1_2
        tls.load_cert_chain(config.cert_dir / "server.crt", config.cert_dir / "server.key")
        runners.append(web.AppRunner(create_setup_app(config), access_log=None, handler_cancellation=True))
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopped.set)
    try:
        for runner in runners:
            await runner.setup()
        if config.public_origin:
            # Cloudflare terminates public HTTPS; the local hop never leaves this Mac.
            await web.TCPSite(runners[0], "127.0.0.1", tunnel_port).start()
        else:
            await web.TCPSite(runners[0], config.lan_ip, config.https_port, ssl_context=tls).start()
            await web.TCPSite(runners[1], config.lan_ip, config.setup_port).start()
        await stopped.wait()
    finally:
        for runner in reversed(runners):
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--public-origin", help="Assigned Cloudflare Quick Tunnel HTTPS origin")
    parser.add_argument("--tunnel-port", type=int, default=8768)
    args = parser.parse_args()
    config = GatewayConfig.from_file(args.config)
    if args.public_origin:
        config = replace(config, public_origin=args.public_origin, cookie_token=secrets.token_urlsafe(32))
    asyncio.run(serve(config, args.tunnel_port))


if __name__ == "__main__":
    main()
