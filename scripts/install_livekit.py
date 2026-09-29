"""Install a verified LiveKit 1.13.7 Homebrew bottle into this experiment only."""
import hashlib
import io
import json
from pathlib import Path
import platform
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DIGESTS = {
    '26': '0302c5e72f2a2a56f0194efd1949e416ea61807701635d54fe3e80b09f7744e1',
    '15': 'a1999329af64094572cf75ee21d9c352973b0e08d8120927027b56394369f1bb',
}


def main():
    version = platform.mac_ver()[0].split('.')[0]
    if platform.machine() != 'arm64' or version not in DIGESTS:
        raise SystemExit('이 설치기는 macOS 15/26 Apple Silicon용입니다. 공식 self-host 설치 문서를 확인하세요.')
    digest = DIGESTS[version]
    with urllib.request.urlopen('https://ghcr.io/token?service=ghcr.io&scope=repository:homebrew/core/livekit:pull', timeout=30) as response:
        token = json.load(response)['token']
    request = urllib.request.Request('https://ghcr.io/v2/homebrew/core/livekit/blobs/sha256:' + digest,
                                     headers={'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise SystemExit('다운로드 SHA-256이 일치하지 않습니다.')
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        members = [item for item in archive.getmembers() if item.isfile() and item.name.endswith('/bin/livekit-server')]
        if len(members) != 1:
            raise SystemExit('실행파일을 확인하지 못했습니다.')
        binary = archive.extractfile(members[0]).read()
    path = ROOT / 'bin' / 'livekit-server'
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(binary)
    path.chmod(0o755)
    (path.parent / 'provenance.json').write_text(json.dumps({
        'version': '1.13.7', 'source': 'https://formulae.brew.sh/api/formula/livekit.json',
        'bottle_sha256': digest, 'binary_sha256': hashlib.sha256(binary).hexdigest()}, indent=2) + '\n')
    print('LiveKit 1.13.7 설치 및 SHA-256 검증 완료:', path)


if __name__ == '__main__':
    main()
