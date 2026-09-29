"""Prepare an opt-in LAN demo without changing the system trust store."""
import hashlib
import ipaddress
import json
from pathlib import Path
import plistlib
import secrets
import ssl
import subprocess
import uuid

import yaml


def private_ipv4(value):
    address = ipaddress.IPv4Address(value)
    ranges = ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16')
    if not any(address in ipaddress.IPv4Network(network) for network in ranges):
        raise ValueError('LAN 주소는 이 Mac의 사설 IPv4 주소여야 합니다.')
    return str(address)


def _openssl(*args):
    result = subprocess.run(['openssl', *map(str, args)], capture_output=True)
    if result.returncode:
        raise RuntimeError('로컬 HTTPS 인증서 준비에 실패했습니다. openssl 설치를 확인하세요.')
    return result.stdout


def _private_write(path, content):
    # Create with restrictive permissions, including during the initial write.
    import os
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as output:
        output.write(content)
    path.chmod(0o600)


def prepare_lan(root, lan_ip, web_port, https_port=8443, setup_port=8767):
    root = Path(root)
    lan_ip = private_ipv4(lan_ip)
    ports = (web_port, https_port, setup_port)
    if any(not 1024 <= port <= 65535 for port in ports) or len(set(ports + (7880, 7882, 8081))) != 6:
        raise ValueError('웹·HTTPS·인증서 안내 포트는 서로 다른 1024–65535 포트여야 합니다.')
    directory = root / 'dadok_chatbot' / 'runtime' / 'lan'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    ca_key, ca_cert = directory / 'ca.key', directory / 'dadok-local-ca.crt'
    if ca_key.exists() != ca_cert.exists():
        raise RuntimeError('로컬 CA 파일 일부가 없습니다. 기존 인증서 구성을 확인하세요.')
    if not ca_cert.exists():
        ca_config = directory / 'ca.cnf'
        ca_config.write_text('[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=ca\n'
                             '[dn]\nCN=Dadok Local Demo CA\n[ca]\n'
                             'basicConstraints=critical,CA:TRUE,pathlen:0\n'
                             'keyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\n')
        _openssl('genrsa', '-out', ca_key, '2048')
        ca_key.chmod(0o600)
        _openssl('req', '-new', '-x509', '-sha256', '-days', '365', '-key', ca_key,
                 '-out', ca_cert, '-config', ca_config)
    else:
        _openssl('x509', '-in', ca_cert, '-checkend', '86400', '-noout')
    ca_key.chmod(0o600)
    server_key, server_cert = directory / 'server.key', directory / 'server.crt'
    # Reissue the leaf on startup; the same CA remains valid after a LAN IP change.
    server_config = directory / 'server.cnf'
    server_config.write_text('[req]\nprompt=no\ndistinguished_name=dn\nreq_extensions=server\n'
                             f'[dn]\nCN={lan_ip}\n[server]\nbasicConstraints=critical,CA:FALSE\n'
                             'keyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n'
                             f'subjectAltName=IP:{lan_ip},IP:127.0.0.1,DNS:localhost\n')
    _openssl('genrsa', '-out', server_key, '2048')
    server_key.chmod(0o600)
    request = directory / 'server.csr'
    _openssl('req', '-new', '-key', server_key, '-out', request, '-config', server_config)
    _openssl('x509', '-req', '-in', request, '-CA', ca_cert, '-CAkey', ca_key,
             '-set_serial', str(secrets.randbits(128) or 1), '-out', server_cert,
             '-days', '90', '-sha256', '-extfile', server_config, '-extensions', 'server')
    der = _openssl('x509', '-in', ca_cert, '-outform', 'DER')
    (directory / 'dadok-local-ca.cer').write_bytes(der)
    fingerprint = hashlib.sha256(der).hexdigest().upper()
    identifier = 'local.dadok.demo.' + fingerprint[:16].lower()
    profile = {'PayloadType': 'Configuration', 'PayloadVersion': 1,
               'PayloadIdentifier': identifier, 'PayloadUUID': str(uuid.uuid4()),
               'PayloadDisplayName': '다독 로컬 음성 데모 인증서',
               'PayloadDescription': '같은 네트워크의 다독 음성 실험 HTTPS 접속용입니다.',
               'PayloadContent': [{'PayloadType': 'com.apple.security.root', 'PayloadVersion': 1,
                                   'PayloadIdentifier': identifier + '.ca', 'PayloadUUID': str(uuid.uuid4()),
                                   'PayloadDisplayName': 'Dadok Local Demo CA', 'PayloadContent': der}]}
    (directory / 'dadok-local-ca.mobileconfig').write_bytes(plistlib.dumps(profile))
    # Validate the generated pair before any listener is started.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(server_cert, server_key)
    config_path = directory / 'config.json'
    access_code = None
    if config_path.exists():
        try:
            previous = json.loads(config_path.read_text())
            candidate = previous.get('access_code') if isinstance(previous, dict) else None
            if isinstance(candidate, str) and len(candidate) == 8 and candidate.isascii() and candidate.isdecimal():
                access_code = candidate
        except (OSError, ValueError):
            pass
    config = {'lan_ip': lan_ip, 'web_port': web_port, 'https_port': https_port,
              'setup_port': setup_port, 'cert_dir': str(directory),
              'access_code': access_code or ''.join(secrets.choice('0123456789') for _ in range(8)),
              'cookie_token': secrets.token_urlsafe(32)}
    _private_write(config_path, json.dumps(config))
    livekit = yaml.safe_load((root / 'livekit.yaml').read_text())
    livekit['rtc'].pop('node_ip', None)
    livekit['rtc'].update(use_external_ip=False, enable_loopback_candidate=True,
                          ips={'includes': ['127.0.0.1/32', lan_ip + '/32']})
    livekit_path = directory / 'livekit-lan.yaml'
    livekit_path.write_text(yaml.safe_dump(livekit))
    return config_path, livekit_path, config, ':'.join(fingerprint[i:i + 2] for i in range(0, 64, 2))
