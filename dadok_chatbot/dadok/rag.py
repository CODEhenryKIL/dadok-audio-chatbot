"""New hybrid retriever over the supplied, immutable Voyage-4 index."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np


class RagError(RuntimeError):
    pass


_STOP = {"알려줘", "어떻게", "무엇", "뭐야", "있어", "있을", "때는", "어떤", "대한", "하면", "좋아", "해줘", "그럼", "아까", "조금", "하나요"}
CONTEXT_VERSION = 4
_CONTEXT_FIELDS = {'context_text', 'context_start_offset', 'context_end_offset',
                   'context_warnings', 'context_sections', 'context_page_heading', 'context_layout_id'}


def _concepts(question):
    concepts = []
    for expression, terms in [
        (r"수면|잠|재우|재울|재워", ["수면", "잠", "재우", "재울"]),
        (r"이유식", ["이유식", "보충식"]),
        (r"분유|수유|모유|안 먹|먹는 양", ["분유", "수유", "모유"]),
        (r"기침", ["기침"]), (r"체온|발열|미열|열이|열까지", ["체온", "발열", "열"]),
        (r"약|해열제", ["약", "해열"]), (r"설사|묽은 변", ["설사", "대변"]),
        (r"구토|토했|토하|토한", ["구토", "토하", "토할"]),
        (r"알레르기|붉은|반점", ["알레르기", "발진"]),
        (r"성장|발달|몸무게|신장", ["성장", "발달", "몸무게"]),
        (r"물은|물을|물 언제", ["물", "수분"]),
    ]:
        if re.search(expression, question):
            concepts.extend(terms)
    return concepts


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[가-힣]+|[a-z]+|\d+", unicodedata.normalize("NFKC", text).lower())
    result = []
    for word in words:
        if word in _STOP or len(word) < 2:
            continue
        result.append(word)
        if re.fullmatch(r"[가-힣]{3,}", word):
            result.extend(word[i:i+2] for i in range(len(word) - 1))
    return result


def _page_age_scopes(pages):
    """Use headings/numbered-section introductions, never random body ages.

    An approximate section age is a conservative retrieval exclusion for younger
    requests, not a medical lower-age recommendation. Its original qualifier is
    preserved. Scope ends at the next numbered section or explicit age heading.
    """
    result, scope, document = {}, None, None
    for page in pages:
        if page["document_id"] != document:
            document, scope = page["document_id"], None
        # A standalone page age heading is not proof of the next page's scope.
        # Only numbered sections have a detectable continuation boundary.
        if scope and scope["kind"] == "explicit_age_heading":
            scope = None
        text = unicodedata.normalize("NFKC", page["text"])
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        first = lines[0] if lines else ""
        numbered_section = bool(re.match(r"\d{1,3}\n\d{2}\n(?:\d{2}\n)?[^\d\n]", "\n".join(lines[:5])))
        if numbered_section:
            scope = None
        heading = re.match(r"(?:생후\s*|만\s*)?(\d{1,3})\s*[~∼–-]\s*(\d{1,3})\s*(개월|세)", first)
        if heading:
            factor = 12 if heading.group(3) == "세" else 1
            scope = {"kind": "explicit_age_heading", "min_months": int(heading.group(1)) * factor,
                     "max_months": int(heading.group(2)) * factor, "evidence": first[:90],
                     "source_page": page["page"]}
        elif numbered_section:
            # The lead comes before the section's main paragraphs. Plain numbered
            # lists in the body do not qualify as a new age-scoped introduction.
            lead = " ".join(lines[3:])[:240]
            match = re.search(r"만\s*(\d{1,2})\s*세\s*(즈음|전후|무렵|이상|이후|부터)", lead)
            if match:
                scope = {"kind": "section_age_reference", "reference_months": int(match.group(1)) * 12,
                         "qualifier": match.group(2), "evidence": match.group(0), "source_page": page["page"],
                         "limitation": "주변 연령 범위는 명시되지 않음. 더 어린 질문에는 보수적으로 제외하며 의료적 연령 하한을 뜻하지 않음."}
        result[(document, page["page"])] = dict(scope) if scope else None
    return result


def _question_ages(question):
    return [int(match.group(1)) * (12 if match.group(2) == "세" else 1)
            for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*(개월|세)(?!\s*(?:미만|이하|이상|이후))", question)]


def _age_eligible(row, ages):
    if not ages:
        return True
    scope = row.get("age_scope")
    for months in ages:
        if not row["age_min_months"] <= months <= row["age_max_months"]:
            continue
        if not scope:
            return True
        if scope["kind"] == "explicit_age_heading" and scope["min_months"] <= months <= scope["max_months"]:
            return True
        if scope["kind"] == "section_age_reference" and months >= scope["reference_months"]:
            return True
    return False


def _measurement_intent(question):
    return bool(re.search(r"체온|열을|열은|온도계", question) and
                re.search(r"측정|재는|재야|재면|재기|재줘|재나요|어디|어떻게", question))


def _measurement_evidence(text):
    # Require the actual requested procedure, not temperature in a burns article,
    # mercury in a poison list, or an armpit mentioned in a bathing/rash passage.
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text))
    return bool(re.search(r"체온(?:계)?.{0,35}(?:측정|재는|재야|잽니다|재고|재어|재면|재도록)", text) or
                re.search(r"(?:측정|재는|재야).{0,25}(?:체온|체온계)", text) or
                re.search(r"(?:귀|고막|겨드랑이|직장|이마).{0,18}(?:체온계|체온\s*측정)", text))


def _specific_intent(question):
    if _measurement_intent(question):
        return "temperature_measurement"
    if (re.search(r"(?:예방접종|접종|백신).{0,15}(?:후|뒤|맞고|맞은)", question) and
            re.search(r"관찰|기록|미열|발열|열이|살펴|확인", question)):
        return "post_vaccination_observation"
    if (re.search(r"(?<![가-힣])물(?:은|을|도|만|\s)", question) and
            re.search(r"언제|얼마나|양|먹여|먹이|줘|주어|주면|마시", question) and
            not re.search(r"분유.{0,8}(?:타는|타야|타려)|조제|씻|목욕", question)):
        return "drinking_water"
    if (re.search(r"매일|하루|육아\s*일지|육아\s*기록", question) and
            re.search(r"기록|적어|남겨|써야", question) and
            re.search(r"무엇|뭘|어떤|항목|처음|시작|부터", question)):
        return "daily_care_recording"
    return None


def _intent_sentences(text):
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text))
    return re.split(r"(?<=[.!?])\s+", text)


def _water_evidence(text):
    for sentence in _intent_sentences(text):
        if not re.search(r"(?<![가-힣])물(?:을|은|도|만|이나|이 아닌)", sentence):
            continue
        if re.search(r"마시|먹이|먹여|권장|제공", sentence) and re.search(r"개월|세용|영아|영유아|유아|모유|이유식", sentence):
            # A recipe, illness, or poisoning passage is not advice about routine
            # drinking water. Other topics still retrieve those passages normally.
            if not re.search(r"설사|탈수|구토|경련|니코틴|담배|입덧|산모|산통|가스|좌욕|분유.{0,12}(?:타|붓)|쌀미음", sentence):
                return True
    return False


def _daily_record_evidence(text):
    for sentence in _intent_sentences(text):
        if not re.search(r"기록|일지|메모", sentence):
            continue
        # A diary checklist must actually describe routine care observations.
        # Staff-health records, play photos and preparation labels do not qualify.
        observations = sum(bool(re.search(term, sentence)) for term in
                           (r"수유|먹는\s*양|먹인\s*양", r"수면|잠든|수면시간", r"체온", r"대소변|기저귀|배변", r"몸무게|체중"))
        if observations >= 2 and re.search(r"매일|하루|육아|아기|아이|영아", sentence):
            return True
    return False


def _post_vaccine_evidence(text):
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text))
    return bool(re.search(r"(?:예방)?접종\s*(?:후|뒤|을\s*마치고).{0,100}(?:발열|컨디션|관찰|기록|증상|의료기관)", text))


def _intent_evidence(text, intent):
    checks = {"temperature_measurement": _measurement_evidence,
              "drinking_water": _water_evidence,
              "daily_care_recording": _daily_record_evidence,
              "post_vaccination_observation": _post_vaccine_evidence}
    return checks[intent](text) if intent else True


def _complete_context(page_text, start, end):
    """Restore cut sentence edges using only the verified source page.

    These offsets describe presentation context, not new retrieval chunks.
    Do not guess a missing condition or cross into another page's age scope.
    """
    boundaries = [0, *(match.end() for match in re.finditer(r'[.!?。](?=\s|$)|\n[ \t]*\n', page_text)), len(page_text)]
    left = max(point for point in boundaries if point <= start)
    right = min(point for point in boundaries if point >= end)
    warnings = []
    # Long HTML articles may lack reliable sentence/paragraph boundaries.
    # Keep the original window in that case and expose the missing context.
    if start - left > 2000:
        left = start
        warnings.append('앞 문장의 완결 경계를 확인하지 못함')
    if right - end > 2000:
        right = end
        warnings.append('뒤 문장의 완결 경계를 확인하지 못함')
    while left < right and page_text[left].isspace():
        left += 1
    while right > left and page_text[right - 1].isspace():
        right -= 1
    return {'context_text': page_text[left:right], 'context_start_offset': left,
            'context_end_offset': right, 'context_warnings': warnings}


class RagStore:
    def __init__(self, path: str | Path, embed=None, *, top_k: int = 4):
        self.path = Path(path)
        self.embed = embed
        self.top_k = max(2, min(5, top_k))
        self._loaded = False
        self._lock = asyncio.Lock()

    def _load(self):
        index = self.path / "index" / "voyage-4"
        try:
            manifest = json.loads((index / "index_manifest.json").read_text())
            ids = json.loads((index / "chunk_ids.json").read_text())
            vectors = np.load(index / "vectors.npy", allow_pickle=False)
            if manifest["embedding_model"] != "voyage-4" or manifest["output_dimension"] != 1024:
                raise ValueError("embedding contract")
            if vectors.shape != (manifest["chunk_count"], 1024) or len(ids) != len(vectors):
                raise ValueError("vector shape")
            if len(ids) != len(set(ids)) or not np.isfinite(vectors).all():
                raise ValueError("invalid vectors")
            digest = "sha256:" + hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            if digest != manifest["chunk_ids_hash"]:
                raise ValueError("chunk IDs fingerprint")
            digest = "sha256:" + hashlib.sha256((index / "vectors.npy").read_bytes()).hexdigest()
            if digest != manifest["vectors_hash"]:
                raise ValueError("vectors fingerprint")
            db = self.path / "corpus" / "professional_corpus.sqlite3"
            connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            with connection:
                rows = connection.execute("""SELECT c.*, d.title, d.organization, d.source_page_url,
                    d.age_min_months, d.age_max_months, d.published_at, d.topics_json,
                    d.external_processing_approved FROM professional_chunks c
                    JOIN professional_documents d ON d.document_id=c.document_id""").fetchall()
                pages = connection.execute("SELECT document_id,page,text FROM professional_pages ORDER BY document_id,page").fetchall()
            connection.close()
            by_id = {r["chunk_id"]: dict(r) for r in rows}
            scopes = _page_age_scopes(pages)
            page_texts = {(page['document_id'], page['page']): page['text'] for page in pages}
            # Reviewed layout is presentation metadata, never a replacement
            # corpus/index. Bind it to both the actual PDF and parsed page.
            layouts = {}
            source_hashes = {}
            layout_file = Path(__file__).with_name('reference_context.json')
            for layout in json.loads(layout_file.read_text())['pages']:
                key = (layout['document_id'], layout['page'])
                if key not in page_texts:
                    continue
                filename = layout['source_filename']
                if Path(filename).name != filename:
                    raise ValueError('layout source filename')
                if filename not in source_hashes:
                    source_hashes[filename] = 'sha256:' + hashlib.sha256((self.path / 'sources' / filename).read_bytes()).hexdigest()
                page_hash = 'sha256:' + hashlib.sha256(page_texts[key].encode()).hexdigest()
                if source_hashes[filename] != layout['source_hash'] or page_hash != layout['page_text_hash']:
                    raise ValueError('layout source fingerprint')
                layouts[key] = layout
            for row in by_id.values():
                row["age_scope"] = scopes.get((row["document_id"], row["page"]))
                page_text = page_texts[(row['document_id'], row['page'])]
                if page_text[row['start_offset']:row['end_offset']] != row['text']:
                    raise ValueError('chunk page offsets mismatch')
                layout = layouts.get((row['document_id'], row['page']))
                if layout:
                    indices = layout.get('chunk_sections', {}).get(
                        row['chunk_id'], list(range(len(layout['sections']))))
                    if not indices or any(type(i) is not int or not 0 <= i < len(layout['sections'])
                                          for i in indices):
                        raise ValueError('layout section indices')
                    row.update({'context_layout_id': f"{row['document_id']}:{row['page']}",
                                'context_page_heading': layout['page_heading'],
                                'context_sections': [layout['sections'][i] for i in indices],
                                'context_warnings': layout['warnings']})
                else:
                    row.update(_complete_context(page_text, row['start_offset'], row['end_offset']))
            if set(ids) != set(by_id):
                raise ValueError("corpus index mismatch")
            for row in by_id.values():
                if "sha256:" + hashlib.sha256(row["text"].encode()).hexdigest() != row["text_hash"]:
                    raise ValueError("chunk fingerprint")
            norms = np.linalg.norm(vectors, axis=1)
            if np.any(norms <= 0):
                raise ValueError("zero vector")
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            raise RagError("rag_index_invalid") from exc
        self.ids, self.chunks, self.manifest = ids, [by_id[i] for i in ids], manifest
        self.by_id = by_id
        self.vectors = vectors / norms[:, None]
        self.counts = [Counter(_tokens(row["text"])) for row in self.chunks]
        self.lengths = np.asarray([sum(c.values()) for c in self.counts], dtype=float)
        self.average_length = float(self.lengths.mean()) or 1.0
        frequency = Counter(term for count in self.counts for term in count)
        self.idf = {term: math.log(1 + (len(ids) - n + .5) / (n + .5)) for term, n in frequency.items()}
        self._loaded = True

    async def refresh_context(self, result):
        """Upgrade saved evidence locally, without a new search/embedding call."""
        if result.get('status') != 'ok':
            return result
        async with self._lock:
            if not self._loaded:
                await asyncio.to_thread(self._load)
        chunks = []
        for cached in result.get('chunks', []):
            row = self.by_id.get(cached.get('chunk_id'))
            if row is None or cached.get('text') != row['text']:
                raise RagError('cached_source_mismatch')
            chunk = {k: v for k, v in cached.items() if k not in _CONTEXT_FIELDS}
            chunk.update({k: row[k] for k in _CONTEXT_FIELDS if k in row})
            chunks.append(chunk)
        return {**result, 'chunks': chunks,
                'coverage': {**result.get('coverage', {}), 'context_version': CONTEXT_VERSION}}

    def _lexical(self, query):
        scores = np.zeros(len(self.ids))
        for token in set(_tokens(query)):
            if token not in self.idf:
                continue
            frequency = np.asarray([c.get(token, 0) for c in self.counts], dtype=float)
            denom = frequency + 1.5 * (.25 + .75 * self.lengths / self.average_length)
            scores += self.idf[token] * frequency * 2.5 / denom
        return scores

    async def search(self, question: str, previous: dict | None = None, delta: bool = False) -> dict:
        async with self._lock:
            if not self._loaded:
                await asyncio.to_thread(self._load)
        query = question
        if delta and previous:
            # Preserve the prior subject and age for ellipsis. Replace an old
            # age only when the follow-up actually supplies a new one.
            old = previous.get("query", "")
            if _question_ages(question):
                old = re.sub(r"(?:만\s*)?\d+\s*(?:개월|세)", "", old)
            query = old[-250:] + " " + question
        ages = _question_ages(query)
        age_eligible = np.asarray([_age_eligible(row, ages) for row in self.chunks])
        intent = _specific_intent(query)
        # These hand-written patterns can recognize helpful passages, but a
        # mismatch cannot prove that a differently worded passage lacks evidence.
        # Keep them as positive ranking hints, never as pre-embedding gates.
        intent_matches = np.asarray([bool(intent) and _intent_evidence(row["text"], intent)
                                     for row in self.chunks])
        eligible = age_eligible.copy()
        eligible &= np.asarray([bool(r["external_processing_approved"]) for r in self.chunks])
        concepts = _concepts(query)
        lexical = self._lexical(query + " " + " ".join(concepts))
        cosine = np.zeros(len(self.ids))
        method = "lexical_bm25"
        fallback = None
        if self.embed and eligible.any():
            try:
                query_vector = np.asarray(await self.embed(query), dtype=np.float32)
                if query_vector.shape != (1024,) or not np.isfinite(query_vector).all() or np.linalg.norm(query_vector) <= 0:
                    raise RagError("rag_query_vector_invalid")
                cosine = np.einsum("ij,j->i", self.vectors, query_vector / np.linalg.norm(query_vector))
                if not np.isfinite(cosine).all():
                    raise RagError("rag_query_vector_invalid")
                method = "hybrid_bm25_cosine_rrf"
            except RagError:
                raise
            except Exception:
                # Available lexical evidence remains useful after an embedding outage.
                fallback = "query_embedding_unavailable"
        # Never fill the requested K with zero-evidence chunks.
        originals = [w for w in re.findall(r"[가-힣a-zA-Z]{3,}", query) if w not in _STOP]
        lexical_evidence = np.asarray([any(term in r["text"] for term in (concepts or originals)) for r in self.chunks])
        evidence = ((lexical > 0) & lexical_evidence) | ((cosine >= .35) if method.startswith("hybrid") else False)
        candidates = set()
        score = np.zeros(len(self.ids))
        for ranker in ([lexical, cosine] if method.startswith("hybrid") else [lexical]):
            ranked = [int(i) for i in np.argsort(-ranker) if eligible[i] and evidence[i] and ranker[i] > 0][:20]
            for rank, i in enumerate(ranked, 1):
                candidates.add(i)
                score[i] += 1 / (60 + rank)
        # An explicit procedure match is a useful extra relevance signal even
        # when BM25 tokenization misses its wording. Other semantic matches
        # remain eligible; absence of this hint must never imply no evidence.
        for i in np.flatnonzero(eligible & intent_matches):
            candidates.add(int(i))
            score[i] += 2 / 61
        chosen = []
        for i in sorted(candidates, key=lambda i: (-score[i], -lexical[i], self.ids[i])):
            row = self.chunks[i]
            if _is_duplicate(row, chosen):
                continue
            # Suppress a weak lexical tail when strong topical matches are present.
            if chosen and not intent_matches[i] and cosine[i] < .35 and lexical[i] < max(1.0, float(lexical.max()) * .18):
                continue
            chosen.append({**row, "score": round(float(score[i]), 7),
                           "cosine": round(float(cosine[i]), 5) if method.startswith("hybrid") else None,
                           "lexical_score": round(float(lexical[i]), 5)})
            if len(chosen) == self.top_k:
                break
        if delta and previous:
            # The resolved query already ranks old and new passages together.
            # Mark still-relevant hits rather than appending stale old passages.
            previous_ids = {row.get("chunk_id") for row in previous.get("chunks", [])}
            for row in chosen:
                if row["chunk_id"] in previous_ids:
                    row["reused"] = True
        fields = {"chunk_id", "document_id", "text", "page", "title", "organization", "source_page_url", "published_at", "score", "cosine", "lexical_score", "reused", "age_scope", "age_min_months", "age_max_months"} | _CONTEXT_FIELDS
        chunks = [{k: v for k, v in r.items() if k in fields} for r in chosen]
        for chunk in chunks:
            if not chunk["age_scope"]:
                chunk["age_applicability"] = "문서 전체 연령만 확인됨. 본문의 다른 연령별 설명을 질문 아이에게 그대로 적용하지 마세요."
            elif not ages:
                chunk["age_applicability"] = "질문 아이의 연령이 확인되지 않음. 명시된 자료 연령 범위의 일반 설명으로만 사용하세요."
        return {"status": "ok", "source": "professional_rag", "query": query,
                "chunks": chunks, "source_ids": [r["chunk_id"] for r in chunks],
                "coverage": {"method": method, "fallback": fallback, "delta": delta,
                             "index_version": self.manifest["index_version"], "embedding_model": "voyage-4",
                             "context_version": CONTEXT_VERSION,
                             "dimensions": 1024, "corpus_chunks": len(self.ids),
                             "documents_reembedded": 0, "low_evidence": not chunks,
                             "question_age_months": ages, "age_scope_filtered_chunks": int((~age_eligible).sum()),
                             "specific_intent": intent,
                             "intent_hint_matches": int(intent_matches.sum()) if intent else None,
                             "missing_evidence_reason": "no_relevant_evidence_retrieved" if not chunks else None,
                             "evidence_scope": "retrieved_passages_only",
                             "facet_coverage": "not_evaluated",
                             "partial_evidence": None, "missing_facets": []}}


def _is_duplicate(row, chosen):
    for old in chosen:
        if row.get('context_layout_id') and row['context_layout_id'] == old.get('context_layout_id'):
            # A reviewed page can supply different complete sections. A shared
            # page or overlapping original window alone must not hide a section.
            if all(section in old['context_sections'] for section in row['context_sections']):
                return True
            continue
        if row["text_hash"] == old["text_hash"]:
            return True
        if row["document_id"] == old["document_id"] and row["page"] == old["page"]:
            overlap = max(0, min(row["end_offset"], old["end_offset"]) - max(row["start_offset"], old["start_offset"]))
            if overlap / min(len(row["text"]), len(old["text"])) > .45:
                return True
        # Catch substantially repeated passages across different source documents.
        a, b = set(_tokens(row["text"])), set(_tokens(old["text"]))
        if a and b and len(a & b) / len(a | b) > .8:
            return True
    return False
