"""Paths and explicit runtime configuration; never inherit the old app database."""
from dataclasses import dataclass
from pathlib import Path
import os
import unicodedata
from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT.parent

def sibling(name: str) -> Path:
    return next((p for p in BASE.iterdir() if unicodedata.normalize('NFC', p.name) == name), BASE / name)

@dataclass
class Settings:
    state_path: Path = PROJECT / 'runtime' / 'state.sqlite3'
    budget_path: Path = PROJECT / 'runtime' / 'cost.sqlite3'
    log_path: Path = PROJECT / 'runtime' / 'requests.jsonl'
    rag_path: Path = sibling('전문자료_RAG')
    model: str = 'gpt-5-nano'
    requested_reasoning: str = 'none'
    effective_reasoning: str = 'minimal'
    api_timeout: float = 45.0
    budget_usd: float = 10.0
    snapshot_path: str = ''
    server_url: str = ''
    server_token: str = ''
    server_user_id: str = ''

    @classmethod
    def from_env(cls):
        load_dotenv(BASE / '.env', override=False)
        return cls(snapshot_path=os.getenv('DADOK_RECORD_SNAPSHOT', ''),
                   server_url=os.getenv('DADOK_SERVER_URL', ''),
                   server_token=os.getenv('DADOK_SERVER_TOKEN', ''),
                   server_user_id=os.getenv('DADOK_AUTH_USER_ID', ''))

def read_policy(prefix: str) -> str:
    return next(sibling('정책 및 질문 문서').glob(prefix + '_*.md')).read_text()
