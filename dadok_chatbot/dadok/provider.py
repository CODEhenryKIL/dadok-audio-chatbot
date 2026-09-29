"""Small Responses/Voyage HTTP adapters with durable, conservative cost accounting."""
import asyncio
import contextvars
import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

request_metrics = contextvars.ContextVar('request_metrics', default=None)

class BudgetExceeded(RuntimeError):
    pass

class ProviderError(RuntimeError):
    pass

class Budget:
    def __init__(self, path, limit=10.0):
        self.path = str(path)
        self.limit = min(float(limit), 10.0)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS costs (id TEXT PRIMARY KEY, stage TEXT, usd REAL, estimated INTEGER, usage TEXT, created REAL)')
            db.execute('CREATE TABLE IF NOT EXISTS query_vectors (hash TEXT PRIMARY KEY, vector TEXT NOT NULL)')

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def reserve(self, stage, ceiling):
        ident = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            total = db.execute('SELECT COALESCE(SUM(usd),0) FROM costs').fetchone()[0]
            if total + ceiling > self.limit:
                raise BudgetExceeded('실험 API 비용 상한에 도달했습니다.')
            db.execute('INSERT INTO costs VALUES (?,?,?,?,?,?)', (ident, stage, ceiling, 1, '{}', time.time()))
        return ident

    def settle(self, ident, usd, usage):
        with self.connect() as db:
            db.execute('UPDATE costs SET usd=?,estimated=0,usage=? WHERE id=?', (usd, json.dumps(usage), ident))

    def report(self):
        with self.connect() as db:
            rows = db.execute('SELECT stage,SUM(usd),COUNT(*),SUM(estimated) FROM costs GROUP BY stage').fetchall()
        return {'total_usd': sum(r[1] for r in rows), 'limit_usd': self.limit,
                'stages': [{'stage': r[0], 'usd': r[1], 'calls': r[2], 'estimated_calls': r[3]} for r in rows]}

    def query_vector(self, question, vector=None):
        key = hashlib.sha256(('voyage-4:1024:query:' + question).encode()).hexdigest()
        with self.connect() as db:
            if vector is not None:
                db.execute('INSERT OR REPLACE INTO query_vectors VALUES (?,?)', (key, json.dumps(vector)))
                return vector
            row = db.execute('SELECT vector FROM query_vectors WHERE hash=?', (key,)).fetchone()
            return json.loads(row[0]) if row else None

def validate(value, schema):
    """Validate the small supported schema vocabulary; reject extra output fields."""
    typ = schema.get('type')
    if isinstance(typ, list):
        if value is None and 'null' in typ:
            return
        typ = next(t for t in typ if t != 'null')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('schema enum')
    if typ == 'object':
        if not isinstance(value, dict): raise ValueError('schema object')
        props = schema.get('properties', {})
        if not set(schema.get('required', [])) <= value.keys(): raise ValueError('schema required')
        if schema.get('additionalProperties') is False and value.keys() - props.keys(): raise ValueError('schema extra')
        for key, child in value.items():
            if key in props: validate(child, props[key])
    elif typ == 'array':
        if not isinstance(value, list): raise ValueError('schema array')
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', 10000): raise ValueError('schema length')
        for child in value: validate(child, schema['items'])
    elif typ == 'string':
        if not isinstance(value, str): raise ValueError('schema string')
        if 'pattern' in schema and re.fullmatch(schema['pattern'], value) is None:
            raise ValueError('schema string pattern')
    elif typ == 'integer' and (isinstance(value, bool) or not isinstance(value, int)): raise ValueError('schema integer')
    elif typ == 'boolean' and not isinstance(value, bool): raise ValueError('schema boolean')
    if isinstance(value, (int, float)):
        if value < schema.get('minimum', float('-inf')) or value > schema.get('maximum', float('inf')):
            raise ValueError('schema range')

class LLM:
    def __init__(self, settings, budget):
        self.settings, self.budget = settings, budget
        self.key = os.getenv('OPENAI_API_KEY', '')
        self.voyage_key = os.getenv('VOYAGE_API_KEY', '')

    def _post(self, url, key, payload):
        req = urllib.request.Request(url, json.dumps(payload).encode(),
              headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=self.settings.api_timeout)

    async def generate(self, stage, messages, schema=None, on_delta=None):
        if not self.key:
            raise ProviderError('OPENAI_API_KEY가 없습니다.')
        metrics = request_metrics.get()
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        delivered = False
        accepting = True
        callback_error = None

        def deliver(text):
            nonlocal delivered, callback_error
            if not accepting or callback_error is not None or not text:
                return
            delivered = True
            try:
                on_delta(text)
            except Exception as exc:
                callback_error = exc

        def forward(text):
            # urllib runs in a worker; all consumer callbacks belong to the event loop.
            if accepting:
                loop.call_soon_threadsafe(deliver, text)

        for attempt in range(2):
            payload = {'model': self.settings.model, 'reasoning': {'effort': self.settings.effective_reasoning},
                       'store': False, 'input': messages, 'max_output_tokens': 4096}
            if schema:
                payload['text'] = {'format': {'type': 'json_schema', 'name': stage, 'strict': True, 'schema': schema}}
                if stage == 'answer': payload['stream'] = True
            else:
                payload['stream'] = True
                payload['text'] = {'verbosity': 'low'}
            ceiling = (len(json.dumps(payload).encode()) * .05 + 4096 * .4) / 1e6 + .0001
            reservation = self.budget.reserve(stage, ceiling)
            if metrics is not None:
                metrics['gpt_calls'] += 1
                metrics['api_cost_usd'] += ceiling
                if attempt: metrics['retries'].append(stage)
            try:
                try:
                    if on_delta is not None and schema is None:
                        result, ttft = await asyncio.to_thread(self._response, payload, forward)
                    else:
                        result, ttft = await asyncio.to_thread(self._response, payload)
                finally:
                    # Drain callbacks scheduled by the worker before checking retry eligibility.
                    await asyncio.sleep(0)
                if callback_error is not None:
                    raise ProviderError('스트림 전달 실패') from callback_error
                usage = result.get('usage', {})
                cost = self._cost(usage)
                if usage: self.budget.settle(reservation, cost, usage)
                if metrics is not None:
                    metrics['api_cost_usd'] += cost - ceiling if usage else 0
                    if stage == 'answer' and ttft is not None:
                        metrics['latency_ms']['answer_ttft'] = ttft * 1000
                if result.get('status') != 'completed': raise ProviderError('응답이 불완전합니다.')
                answer = ''.join(c.get('text', '') for item in result.get('output', [])
                                 for c in item.get('content', []) if c.get('type') == 'output_text')
                if not answer.strip(): raise ProviderError('빈 응답 또는 거절 응답입니다.')
                if schema: validate(json.loads(answer), schema)
                if metrics is not None:
                    metrics['latency_ms'][stage + '_llm'] = (time.perf_counter() - started) * 1000
                accepting = False
                return answer
            except asyncio.CancelledError:
                accepting = False
                raise
            except (urllib.error.URLError, TimeoutError, ValueError, ProviderError, OSError) as exc:
                if metrics is not None: metrics['errors'].append({'stage': stage, 'error': type(exc).__name__, 'http_status': getattr(exc, 'code', None)})
                if attempt or delivered:
                    accepting = False
                    if metrics is not None and delivered:
                        metrics['stream_interrupted'] = True
                    raise ProviderError(f'{stage} 호출 실패 ({type(exc).__name__})') from None
        raise ProviderError(stage)

    def _response(self, payload, on_delta=None):
        start = time.perf_counter()
        with self._post('https://api.openai.com/v1/responses', self.key, payload) as response:
            if not payload.get('stream'): return json.load(response), None
            completed, ttft = None, None
            for line in response:
                if not line.startswith(b'data: '): continue
                raw = line[6:].strip()
                if raw == b'[DONE]': break
                event = json.loads(raw)
                if event.get('type') == 'response.output_text.delta':
                    delta = event.get('delta', '')
                    if delta:
                        if ttft is None: ttft = time.perf_counter() - start
                        if on_delta is not None: on_delta(delta)
                if event.get('type') == 'response.completed': completed = event['response']
                if event.get('type') in ('response.failed', 'response.incomplete', 'error'):
                    failed = event.get('response')
                    if failed: return failed, ttft
                    raise ProviderError('stream failed')
            if not completed: raise ProviderError('stream interrupted')
            return completed, ttft

    @staticmethod
    def _cost(usage):
        cached = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
        return ((usage.get('input_tokens', 0) - cached) * .05 + cached * .005 + usage.get('output_tokens', 0) * .4) / 1e6

    async def embed(self, question):
        return (await self.embed_many([question]))[0]

    async def embed_many(self, questions):
        """Batch only query embeddings; reuse identical model/dimension inputs."""
        import math
        cached = {q: self.budget.query_vector(q) for q in dict.fromkeys(questions)}
        missing = [q for q, vector in cached.items() if vector is None]
        if not missing:
            metrics = request_metrics.get()
            if metrics is not None: metrics['embedding_cache_hits'] = metrics.get('embedding_cache_hits', 0) + len(questions)
            return [cached[q] for q in questions]
        if len(missing) > 128:
            raise ValueError('한 번에 최대 128개 검색 질문만 임베딩합니다.')
        if not self.voyage_key: raise ProviderError('VOYAGE_API_KEY가 없습니다.')
        metrics = request_metrics.get()
        payload = {'input': missing, 'model': 'voyage-4', 'input_type': 'query',
                   'output_dimension': 1024, 'output_dtype': 'float', 'truncation': False}
        for attempt in range(2):
            ceiling = sum(len(q.encode()) for q in missing) * .06 / 1e6 + .00001
            reserve = self.budget.reserve('embedding', ceiling)
            if metrics is not None:
                metrics['api_cost_usd'] += ceiling
                if attempt: metrics['retries'].append('embedding')
            try:
                def call():
                    with self._post('https://api.voyageai.com/v1/embeddings', self.voyage_key, payload) as r:
                        return json.load(r)
                result = await asyncio.to_thread(call)
                usage = result['usage']
                cost = usage['total_tokens'] * .06 / 1e6
                self.budget.settle(reserve, cost, usage)
                if metrics is not None:
                    metrics['api_cost_usd'] += cost - ceiling
                entries = result['data']
                if len(entries) != len(missing): raise ValueError('embedding batch length')
                indexed = {entry['index']: entry['embedding'] for entry in entries}
                for i, q in enumerate(missing):
                    vector = indexed[i]
                    if len(vector) != 1024 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in vector):
                        raise ValueError('embedding vector')
                    cached[q] = self.budget.query_vector(q, vector)
                return [cached[q] for q in questions]
            except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
                if metrics is not None: metrics['errors'].append({'stage': 'embedding', 'error': type(exc).__name__, 'http_status': getattr(exc, 'code', None)})
                if attempt: raise ProviderError('질문 임베딩 호출 실패') from None
