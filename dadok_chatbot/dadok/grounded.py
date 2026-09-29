"""Evidence-only answer candidates: code computes records; RAG stays extractive.

The answer model selects indices in one call. It cannot supply replacement
numbers, prose, diagnoses, or citations to the user-facing renderer.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from .context import NAMES, period_text, value_text

SELECTOR_SCHEMA = {
    "type": "object", "properties": {"indices": {"type": "array", "items": {"type": "integer"}, "maxItems": 6}},
    "required": ["indices"], "additionalProperties": False,
}
selector_schema = SELECTOR_SCHEMA


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _candidate(kind, text, ids):
    return {"kind": kind, "text": text.strip(), "source_ids": list(dict.fromkeys(str(i) for i in ids if i))}


def _stamp(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(ZoneInfo("Asia/Seoul"))
        return stamp.strftime("%Y-%m-%d %H:%M") + " (한국 시간)"
    except (ValueError, TypeError, AttributeError):
        return str(value or "시각 미기록")


def _name(key):
    kind, sub, _ = key.split(":", 2)
    return NAMES.get(sub, sub) if sub != "all" else NAMES.get(kind, kind)


def _row_text(row):
    name = NAMES.get(row.get("subtype"), row.get("subtype")) or NAMES.get(row.get("type"), "기록")
    text = f"{_stamp(row.get('occurred_at'))}: {name}"
    if _number(row.get("amount")):
        text += " " + value_text(row["amount"], row.get("unit") or "unknown")
    if row.get("interval_end"):
        text += " / 종료 " + _stamp(row["interval_end"])
    if row.get("note"):
        text += " / 기록 메모: 「" + re.sub(r"\s+", " ", str(row["note"])) + "」"
    return text


def _record_candidates(record, question):
    if not record or record.get("status") != "ok":
        return []
    stats, coverage = record.get("statistics", {}), record.get("coverage", {})
    rows = sorted(record.get("records", []), key=lambda r: r.get("occurred_at", ""))
    ids = record.get("source_ids", [])
    count = stats.get("count")
    period = period_text(stats.get("period") or coverage)
    prefix = (period + "의 ") if period else "조회 기간의 "
    result = []
    def add(text, source_ids=ids):
        result.append(_candidate("record", text, source_ids))
    if count == 0:
        add(prefix + "해당 기록은 없습니다. 기록이 없다는 것만으로 증상이나 활동이 없었다고 판단하지 않습니다.")
        return result
    groups = stats.get("totals_by_type_and_unit") or stats.get("groups", {})
    comparison = stats.get("comparison")
    compares = bool(comparison and re.search(r"비교|이전|차이|늘|줄|변화|감소|증가|대비|보다", question))
    lists = bool(re.search(r"보여|목록|나열|시간순|일별|날짜별|일자별|하나씩|전부|모든", question))
    means = bool(re.search(r"평균|일평균|하루|요약", question))
    maximum = bool(re.search(r"최대|최고|가장.{0,8}(?:높|많|길|오래)|제일.{0,8}(?:높|많|길|오래)|최장", question))
    minimum = bool(re.search(r"최소|최저|가장.{0,8}(?:낮|적|짧)|제일.{0,8}(?:낮|적|짧)|최단", question))
    counts = bool(re.search(r"몇\s*(?:번|회)|횟수|건수", question))
    kinds = {r.get("type") for r in rows}

    if compares:
        current = comparison.get("current", stats)
        old = comparison.get("previous", {})
        now_groups = current.get("totals_by_type_and_unit") or current.get("groups", {})
        old_groups = old.get("totals_by_type_and_unit") or old.get("groups", {})
        parts = []
        for key, group in now_groups.items():
            previous = old_groups.get(key)
            unit = key.split(":", 2)[-1]
            if previous and _number(group.get("daily_mean")) and _number(previous.get("daily_mean")):
                change = group["daily_mean"] - previous["daily_mean"]
                direction = "같습니다" if change == 0 else value_text(abs(change), unit) + (" 줄었습니다" if change < 0 else " 늘었습니다")
                parts.append(f"{_name(key)}의 기록 있는 날 하루 평균은 최근 기간 {value_text(group['daily_mean'], unit)}, 이전 기간 {value_text(previous['daily_mean'], unit)}이며, {direction}.")
        if not parts:
            parts.append(f"조회된 기록 횟수는 최근 기간 {current.get('count', 0)}회, 이전 기간 {old.get('count', 0)}회입니다.")
        add("최근 기간은 " + period_text(current.get("period")) + ", 이전 기간은 " + period_text(old.get("period")) + "입니다. " + " ".join(parts) + " 기록이 없는 날은 0으로 계산하지 않았습니다.")
        return result  # A single indivisible comparison prevents omitting its baseline.

    if counts and isinstance(count, int):
        label = "대변 기저귀" if re.search(r"대변|똥", question) else "소변 기저귀" if re.search(r"소변|오줌", question) else "해당"
        mixed = " 대변·소변이 함께 기록된 항목도 포함했습니다." if label != "해당" and any(r.get("subtype") == "both" for r in rows) else ""
        add(f"{prefix}{label} 기록은 전체 {count}회입니다." + mixed)
        return result

    if maximum or minimum:
        extreme = "max" if maximum else "min"
        for key, group in groups.items():
            unit = key.split(":", 2)[-1]
            daily = group.get("daily", {})
            if daily and re.search(r"날|날짜|요일|언제", question) and key.startswith("sleep:"):
                amount = (max if maximum else min)(daily.values())
                dates = [day for day, value in daily.items() if value == amount]
                label = ", ".join(dates[:7]) + (f" 외 {len(dates) - 7}일" if len(dates) > 7 else "")
                add(f"{prefix}{_name(key)} 일별 합계가 가장 {'긴' if maximum else '짧은'} 날은 {label}이며, 하루 {value_text(amount, unit)}입니다.")
            elif _number(group.get(extreme)):
                amount = group[extreme]
                found = group.get(extreme + "_row") or group.get(extreme + "row")
                if not isinstance(found, dict):
                    found = next((r for r in rows if r.get("type") == key.split(":")[0] and r.get("amount") == amount), None)
                timing = " 기록 시각은 " + _stamp(found.get("occurred_at")) + "입니다." if found else ""
                add(f"{prefix}{_name(key)} {'최댓값' if maximum else '최솟값'}은 {value_text(amount, unit)}입니다." + timing)
        if result:
            return result

    if (lists and not means) or kinds & {"medicine", "symptom"}:
        shown = rows[-120:]
        text = prefix + f"시간순 기록은 다음과 같습니다 ({len(shown)}건).\n" + "\n".join("- " + _row_text(row) for row in shown)
        if coverage.get("records_truncated") or len(rows) > 120:
            text += f"\n전체 {coverage.get('records_total', count)}건 중 최근 {len(shown)}건만 표시했습니다."
        if "medicine" in kinds:
            text += "\n위 수치는 과거 복약 기록이며 현재 복용량이나 복용 간격의 권고가 아닙니다."
        add(text)
        return result

    if "growth" in kinds:
        for key, group in stats.get("groups", {}).items():
            if key.startswith("growth:") and isinstance(group.get("last"), dict):
                add("가장 최근 " + _row_text(group["last"]) + "입니다.")
        if result:
            return result

    for key, group in groups.items():
        kind, _, unit = key.split(":", 2)
        name = _name(key)
        parts = [f"{prefix}{name} 기록은 {group.get('count', 0)}회입니다."]
        if kind not in {"temperature", "growth", "medicine"} and _number(group.get("total")):
            parts.append("기록된 합계는 " + value_text(group["total"], unit) + "입니다.")
        if _number(group.get("daily_mean")) and (means or kind == "sleep"):
            parts.append(f"기록이 있는 {group.get('recorded_days', 0)}일의 하루 평균은 {value_text(group['daily_mean'], unit)}입니다.")
            parts.append("기록이 없는 날은 0으로 계산하지 않았습니다.")
        if kind == "sleep" and _number(group.get("min")) and _number(group.get("max")):
            parts.append(f"수면 기록 1회당 최솟값은 {value_text(group['min'], unit)}, 최댓값은 {value_text(group['max'], unit)}입니다.")
        elif kind == "temperature" and isinstance(group.get("last"), dict):
            parts.append("가장 최근 기록은 " + _row_text(group["last"]) + "입니다.")
        add(" ".join(parts))
    return result


def _terms(text):
    stop = {"알려줘", "어떻게", "무엇", "해주세요", "해줘", "그럼", "아기", "아이", "개월", "대한", "있는", "우리", "좋아", "어디에서", "어디", "언제", "얼마나", "무엇을", "무엇부터", "아기에게", "아기를", "아이가", "어떤", "확인하는", "확인해야", "관찰하면", "있는데", "해", "돼", "줘", "점은", "지켜야", "꼭", "하면", "평소보다", "가끔", "매일"}
    words = set(re.findall(r"[가-힣A-Za-z0-9]+", text.lower())) - stop
    result = set(words)
    for word in words:
        if len(word) > 2:
            result.update(word[i:i + 2] for i in range(len(word) - 1))
    for pattern, aliases in ((r"수면|잠|재우|재울|재워", ["수면", "잠", "재우", "재울"]), (r"먹|수유|분유|모유", ["수유", "모유", "분유"]), (r"똥|대변|변비", ["대변", "변비"]), (r"언어|말문|말을", ["언어", "말"]), (r"열|체온", ["열", "체온"])):
        if re.search(pattern, text):
            result.update(aliases)
    return result


def _anchors(question):
    # Topic matching only; no medical conclusion or new fact is encoded here.
    anchors = set()
    for pattern, terms in (
        (r"수면|잠|재우|재울|재워", "수면 잠 낮잠 밤잠 재우 재울 재워 잠자 잠들 잠드 눕혀 침대 침구 베개 이불"),
        (r"이유식|음식|먹이|먹여", "이유식 보충식 음식 식품 먹기 먹이 먹여 재료"),
        (r"분유|수유|모유|안 먹|먹지 못", "분유 수유 모유 먹지 먹는 먹기 먹이 먹고 섭취 대소변 소변량"),
        (r"(?:^|\s)물(?:은|을|이|에|도)?(?:\s|$)|수분", "물 수분 탈수 음료"),
        (r"기침|쌕쌕|호흡|숨쉬|숨을", "기침 호흡 쌕쌕 숨쉬 숨을"),
        (r"구토|토했|토한|한 번 토", "구토 토하 토한 토할"),
        (r"설사|대변|묽은 변|변비|똥", "설사 대변 묽은 변비"),
        (r"체온|온도|미열|발열", "체온 온도 발열 미열 열"),
        (r"예방접종|백신", "예방접종 백신 접종"),
        (r"성장|키|몸무게|체중", "성장 몸무게 체중 신장"),
        (r"발달|언어|말문", "발달 언어"),
        (r"반점|알레르기|두드러기", "반점 알레르기 두드러기 음식 증상"),
    ):
        if re.search(pattern, question):
            anchors.update(terms.split())
    return anchors


def _mentions(term, sentence):
    if len(term) > 1:
        return term in sentence
    return bool(re.search(r"(?<![가-힣])" + re.escape(term) + r"(?:은|는|을|를|이|가|도|만|에)?(?![가-힣])", sentence))


def _question_focus(question):
    """Require the requested action/relation, rather than topic words alone."""
    if re.search(r"억지|강제로|강요", question):
        return r"억지|강요|강제|먹기를?\s*거부|먹기\s*싫"
    if re.search(r"새로운?\s*(?:음식|식품|재료)|새\s*(?:음식|식품|재료)|이유식\s*재료", question) and re.search(r"시작|추가|도입|먹|줘", question):
        return r"새로운?\s*(?:음식|식품|재료)|새\s*(?:음식|식품|재료)|한\s*번에\s*한\s*가지|한\s*가지씩\s*추가"
    if re.search(r"약|약물|의약품|약품|세제|중독", question) and re.search(r"보관|예방|못\s*먹|안전", question):
        return r"(?:약|의약품|약품|약물|세제|살충제|화학물질).{0,80}(?:보관|손.{0,8}닿|잠금|잠그|두지|두어|높은)|(?:보관|잠금).{0,40}(?:약|의약품|약품|약물|세제|살충제)"
    if re.search(r"(?:얼마나\s*)?자주|빈도|주기|며칠마다|몇\s*(?:일|주|달).*마다", question):
        return r"(?:매일|매주|매달|정기적|주기적|\d+\s*(?:일|주|개월|달)\s*(?:마다|간격)|\d+\s*회)"
    if re.search(r"구토|토했|토한|한\s*번\s*토", question) and re.search(r"수유|먹|분유|모유", question):
        return r"(?:구토|토했|토한|토하면|토하더라도).{0,160}(?:수유|먹|분유|모유)|(?:수유|먹|분유|모유).{0,160}(?:구토\s*(?:후|뒤)|토한\s*(?:후|뒤))"
    if re.search(r"루틴|생활\s*리듬|수면\s*습관|잠들기\s*전|자기\s*전", question):
        return r"(?:매일|규칙적|일정한|동일한|똑같이|반복적).{0,100}(?:시간|잠|환경|자장가|동화)|(?:잠|수면|자기\s*전).{0,100}(?:매일|규칙적|일정한|동일한|똑같이|반복적)"
    if re.search(r"안전", question) and re.search(r"수면|잠|재우|재울", question):
        return r"눕|엎드|옆으로|침대|침구|이불|베개|속싸개|카시트|바운서"
    if re.search(r"낮잠", question) and re.search(r"기준|시간|횟수|몇|달라|개월|돌", question):
        return r"낮잠.{0,100}(?:\d+\s*(?:회|시간|분)|횟수|지속\s*시간)"
    if re.search(r"기록(?:해|하|할|해야)|무엇.{0,12}기록|어떤.{0,12}기록", question):
        return r"기록|메모|적어\s*두|기재"
    if re.search(r"확인|관찰|살펴|체크|뭘\s*봐|무엇을\s*봐", question):
        if re.search(r"분유량|수유량|먹는\s*양|안\s*먹|먹지\s*못|잘\s*먹", question):
            return r"대소변|소변량|처지는|체중.{0,16}(?:증가|변화).{0,40}(?:확인|관찰|추세)|(?:잘|충분히)\s*먹지\s*못|(?:먹는\s*양|수유량).{0,60}(?:확인|관찰|살펴)"
        return r"확인|관찰|살펴|주의해야|대소변|소변량|처지는|체중.{0,16}(?:증가|변화)|(?:잘|충분히)\s*먹지\s*못"
    if re.search(r"열|발열|미열", question) and re.search(r"기다|내일|병원|진료", question):
        return r"(?:발열|열|체온).{0,150}(?:의사|의료기관|진료|병원|진찰|응급)|(?:의사|의료기관|진료|병원|진찰|응급).{0,100}(?:발열|열|체온)"
    if re.search(r"왜|원인|이유가|이유는|이유일|이유.{0,10}뭐|어째서", question):
        if re.search(r"수면|잠", question):
            return r"(?:수면\s*(?:패턴|시간|거부)|잠(?:을|이|자는)).{0,80}(?:따라|다양|차이|편차|개인차|결정|변하|줄|늘)|배고픔|분리불안"
        if re.search(r"분유|수유|모유|먹", question):
            return r"(?:먹는|먹이는|먹은)\s*양|수유\s*량|(?:수유|먹는).{0,30}(?:간격|횟수|패턴)|(?:먹지|적게\s*먹|섭취).{0,30}(?:못|원인|부족)|양육\s*환경|식사\s*습관"
        return r"원인|때문|영향|요인|따라|달라|관련"
    if re.search(r"발열|열[이가과도은]|기침|호흡|숨쉬|구토|설사|반점|두드러기", question) and re.search(r"나타났|생겼|나고|있어|심해|달라|보여|하는데", question):
        return r"관찰|확인|살펴|기록|(?:호흡곤란|청색증|의식|처지).{0,80}(?:의료진|의사|진료|병원|119)"
    return None


_AGE_EXPR = re.compile(r"(?:생후\s*|만\s*)?(\d+)\s*(?:[~∼–-]\s*(\d+)\s*)?(개월|세)\s*(미만|이하|이내|까지|이상|이후|부터|무렵|즈음|경|시기|때|에는)?")


def _age_constraints(text, age):
    """Return only age conditions, excluding elapsed durations such as delays."""
    result = []
    for match in _AGE_EXPR.finditer(text):
        after = re.sub(r"\s+", "", text[match.end():match.end() + 28])
        if re.match(r"\s*(?:내외로?\s*)?(?:로\s*)?(?:지연|늦|빠르|차이|동안|걸|지속|간격)", after):
            continue
        low, high, unit, qualifier = match.groups()
        factor = 12 if unit == "세" else 1
        low, high = int(low) * factor, int(high or low) * factor
        if qualifier == "미만":
            valid = age < high
        elif qualifier in {"이하", "이내", "까지"}:
            valid = age <= high
        elif qualifier in {"이상", "이후", "부터"}:
            valid = age >= high
        else:
            valid = low <= age <= high
        result.append(valid)
    return result


def _age_eligible(text, age):
    # Comma-separated ages can be alternatives (3개월까지..., 1세까지...).
    # Bounds within one clause (3개월 이후부터 1세까지) must all hold.
    clauses = [_age_constraints(clause, age) for clause in text.split(",")]
    clauses = [checks for checks in clauses if checks]
    return not clauses or any(all(checks) for checks in clauses)


def _sentence_eligible(sentence, question, age, chunk):
    if re.search(r"QR\s*코드|QR코드|서비스입니다|누리집|앱을\s*설치|다운로드", sentence, re.I):
        return False
    if re.search(r"긴급도\s*\d|자세한\s*내용은.{0,30}(?:상담|참고)", sentence):
        return False  # Triage legends and generic footers lack a patient condition.
    if re.search(r"다음과\s*같습니다|다음은.{0,30}안내|아래\s*(?:표|그림)|이정표로\s*살펴보세요|알려드립니다|알아보겠습니다", sentence):
        return False  # An introduction without the promised facts is not an answer.
    visiting = bool(re.search(r"기다|내일|병원|진료", question) and re.search(r"열|발열|미열", question))
    if visiting and re.search(r"복용|투약|교차|다른\s*계열|해열제.{0,20}(?:먹|사용|성분)", sentence):
        return False  # Dosing conditions cannot answer whether it is safe to wait.
    if not re.search(r"진찰|검사|청진|엑스레이|영상", question) and re.search(r"진찰\s*소견|청진|폐가?\s*과도하게\s*팽창|흉부\s*(?:X|엑스)|검사\s*소견", sentence):
        return False
    # Heading conditions and sentence conditions are both operative; age groups
    # within a sentence may be alternatives, without changing the original text.
    if age is None and re.search(r"수면|잠", question) and re.search(r"왜|원인|이유가|이유는|이유일|이유.{0,10}뭐|어째서", question):
        age_context = str(chunk.get("passage_scope") or "") + " " + str((chunk.get("age_scope") or {}).get("evidence") or "") + " " + sentence
        if _AGE_EXPR.search(age_context) or "신생아" in age_context:
            return False  # An unspecified age cannot select a stage-specific cause.
    if age is not None:
        if not _age_eligible(str(chunk.get("passage_scope") or ""), age) or not _age_eligible(sentence, age):
            return False
        if age > 1 and "신생아" in str(chunk.get("passage_scope") or "") + sentence and "신생아" not in question:
            return False
    disease_words = re.findall(r"세기관지염|기관지염|폐렴|천식|크룹|장염|수족구|수두|홍역|백일해|아토피|인플루엔자", str(chunk.get("title", "")) + " " + str(chunk.get("passage_scope", "")))
    if disease_words and not any(word in question for word in disease_words):
        if re.search(r"호전|회복|예후|사라지기|지속될|며칠\s*안에|치료|입원|진찰|진단|환자관리|격리|투약|항바이러스", sentence):
            return False
    return True


def _passage_sentences(text):
    """Keep numbered section and age-table headings attached to their body."""
    text = re.sub(r"(습니|합니|입니|됩니|잡니|립니)\s*\n\s*(다[.!?])", r"\1\2", str(text))
    scope, parent_scope, lines = "", "", []
    def body():
        age_context = ""
        for sentence in _sentences("\n".join(lines)):
            if _age_constraints(sentence, 0):
                age_context = ""
            elif not re.match(r"일반적으로|그러나|따라서|예를|아이의|수면|그런데|이\s*시기", sentence):
                age_context = ""
            yield sentence, " / ".join(part for part in (scope, age_context) if part)
            # Preserve a preceding age subject for its following explanation.
            match = re.match(r"(?:또\s*)?((?:생후\s*|만\s*)?\d+\s*(?:[~∼–-]\s*\d+\s*)?(?:개월|세)\s*(?:이내|까지|이후|부터|무렵|즈음|경|시기|때|에는)(?:의\s*신생아)?)", sentence)
            if match:
                age_context = match.group(1)
            elif sentence.startswith("신생아는"):
                age_context = "신생아"
    for line in text.splitlines():
        stripped = line.strip()
        heading = (len(stripped) <= 40 and not re.search(r"[!?]|[다요]\.$", stripped) and (
            re.match(r"^[가-힣][.)]\s+\S", stripped)
            or re.match(r"^\d+[)]\s+\S", stripped)
            or re.match(r"^(?:생후\s*|만\s*)?\d+\s*[~∼–-]\s*\d+\s*(?:개월|세)\s*[^.!?]*$", stripped)))
        if heading:
            yield from body()
            if re.match(r"^[가-힣]\.\s", stripped):
                parent_scope = stripped
            scope = " / ".join(dict.fromkeys(part for part in (parent_scope, stripped) if part))
            lines = []
        else:
            lines.append(line)
    yield from body()


def _sentences(text):
    # Page numbers and bullet glyphs are layout, not part of the source claim.
    lines = [line.strip() for line in str(text).splitlines() if line.strip()
             and not re.fullmatch(r"\s*\d+\s*", line)
             and not re.fullmatch(r"(?:#+\s*)?[^.!?]{1,40}(?:안내|가이드|주의사항|설명서|수칙|방법|환경)", line.strip())]
    text = "\n".join(lines)
    text = re.sub(r"(?:^|\n)[•●□◦▪\-+]\s*", "\n• ", text)
    text = re.sub(r"(습니|합니|입니|됩니|잡니|립니)\s*\n\s*(다[.!?])", r"\1\2", text)
    # Preserve sentence content: only whitespace is collapsed, never rewritten.
    text = re.sub(r"[ \t]+", " ", text)
    for sentence in re.split(r"(?<=[.!?。])\s+|\n(?=[•●□◦▪])", text):
        sentence = re.sub(r"\s+", " ", sentence).strip(" •")
        complete_short = len(sentence) >= 8 and bool(re.search(r"(?:않습니다|마세요|주세요|하세요|해야\s*합니다)[.!]$", sentence))
        fragmented = re.match(r"^(?:수 있습니다|없을\s*때|있을\s*때|어있는|글에서는|이\s*시기|그\s*시기|이후\s*며칠|를 |을 |니다[.!]|\d+\.\s|s\d+_R\d+|슈퍼클래스|콘텐츠명)", sentence)
        if (len(sentence) >= 20 or complete_short) and len(sentence) <= 900 and not fragmented and sentence.count(")") <= sentence.count("("):
            if sentence.endswith("?"):
                continue  # A source's question heading is not an answer assertion.
            if re.search(r"(?:다|요|시오|음|함|기)[.!?。]?$", sentence):
                yield sentence


def build_candidates(data: dict, question: str) -> list[dict]:
    record = _record_candidates(data.get("child_record", {}), question)
    rag = data.get("professional_rag", {})
    if rag.get("status") != "ok":
        return record[:16]
    # Records use the current request above. RAG also needs the resolved topic
    # for ellipsis; a new explicit age always supersedes the cached query's age.
    age_pattern = r"(?:생후\s*|만\s*)?\d{1,3}\s*(?:개월|세)|(?<![가-힣])(?:첫\s*돌|돌)(?=은|이면|이|\s|$)"
    prior_query = str(rag.get("query") or "")
    if re.search(age_pattern, question):
        prior_query = re.sub(age_pattern, "", prior_query)
    resolved_question = question + " " + prior_query
    age_match = re.search(r"(?:만\s*)?(\d{1,3})\s*(개월|세)", resolved_question)
    age = int(age_match.group(1)) * (12 if age_match.group(2) == "세" else 1) if age_match else None
    if age is None and re.search(r"(?<![가-힣])(?:첫\s*돌|돌)(?:은|이면|이|\s|$)", resolved_question):
        age = 12
    query_terms = _terms(resolved_question)
    anchors = _anchors(resolved_question)
    focus = _question_focus(question) or _question_focus(resolved_question)
    age_specific_development = age is not None and bool(re.search(r"발달|이정표", resolved_question)) and not re.search(r"검사|장애|진단|의심", resolved_question)
    ranked, seen = [], set()
    for chunk_index, chunk in enumerate(rag.get("chunks", [])):
        if age is not None and (age < chunk.get("age_min_months", 0) or age > chunk.get("age_max_months", 1200)):
            continue
        scope = chunk.get("age_scope") or {}
        if age is not None and scope.get("kind") == "explicit_age_heading" and not scope.get("min_months", 0) <= age <= scope.get("max_months", 1200):
            continue
        if age is not None and scope.get("kind") == "section_age_reference" and age < scope.get("reference_months", 0):
            continue
        source_id = chunk.get("chunk_id")
        if not source_id:
            continue  # Never manufacture provenance for unidentifiable text.
        for position, (sentence, passage_scope) in enumerate(_passage_sentences(chunk.get("text", ""))):
            if focus and not re.search(focus, sentence):
                continue
            if not _sentence_eligible(sentence, resolved_question, age, {**chunk, "passage_scope": passage_scope}):
                continue
            explicit_age_evidence = bool(_age_constraints(passage_scope + sentence, age or 0))
            if age_specific_development and not explicit_age_evidence:
                continue
            scope_text = re.sub(r"\s+", " ", str(scope.get("evidence") or "")).strip()
            normalized = re.sub(r"\s", "", scope_text + sentence)
            if normalized in seen:
                continue
            topic_matches = sum(1 for term in anchors if _mentions(term, sentence))
            if anchors and not topic_matches and not (age_specific_development and explicit_age_evidence):
                continue
            score = (20 if focus else 0) + (30 if age_specific_development and explicit_age_evidence else 0) + 6 * topic_matches + sum(2 if len(term) > 2 else 1 for term in query_terms if len(term) >= 2 and term in sentence.lower())
            if score == 0:
                continue
            seen.add(normalized)
            title = str(chunk.get("title") or "").strip()
            if title:
                sentence = "자료 「" + title + "」: " + sentence
            if passage_scope:
                sentence = "원문 범위 「" + passage_scope + "」: " + sentence
            if scope_text:
                sentence = "자료의 연령 조건 「" + scope_text + "」: " + sentence
            elif chunk.get("age_applicability"):
                sentence = "연령별 적용 조건을 확인해야 하는 일반 자료: " + sentence
            ranked.append((-score, chunk_index, position, _candidate("rag", sentence, [source_id])))
    ranked.sort(key=lambda item: item[:3])
    return record[:8] + [item[3] for item in ranked[:max(0, 16 - min(8, len(record)))]]


def conversation_candidates(previous_answer: str) -> list[dict]:
    """Select original response sentences for a rewrite without new assertions."""
    result = []
    synthetic = "테스트용 합성 기록을 기준으로 안내합니다."
    is_synthetic = synthetic in previous_answer
    for text in re.split(r"(?<=[.!?。])\s+|\n", previous_answer):
        text = text.strip()
        if not text or text == synthetic:
            continue
        result.append(_candidate("conversation", text, []))
    if is_synthetic:
        for candidate in result:
            candidate["text"] = synthetic + " " + candidate["text"]
    return result[:24]


def effective_selection(candidates: list[dict], selected, safety: dict) -> list[dict]:
    """The renderer and provenance logger must use this same selection."""
    if isinstance(selected, dict):
        if set(selected) != {"indices"}:
            raise ValueError("Selection must contain only indices")
        selected = selected["indices"]
    if not isinstance(selected, list) or len(selected) > 6 or any(type(i) is not int or not 0 <= i < len(candidates) for i in selected):
        raise ValueError("Invalid grounded answer index")
    indices = list(dict.fromkeys(selected))
    records = [i for i, c in enumerate(candidates) if c["kind"] == "record"][:3]
    indices = records + [i for i in indices if i not in records]
    action = safety.get("safety_action", safety.get("action", "GUIDE"))
    if action == "EMERGENCY":
        return []
    chosen, counts = [], {}
    for index in indices:
        item = candidates[index]
        kind = item["kind"]
        if action == "BLOCK" and kind != "record":
            continue
        if kind not in {"record", "rag", "conversation"} or counts.get(kind, 0) >= 3:
            continue
        chosen.append(item)
        counts[kind] = counts.get(kind, 0) + 1
    return chosen


def render_selection(candidates: list[dict], selected, safety: dict, data: dict) -> str:
    chosen = effective_selection(candidates, selected, safety)
    record = [c["text"] for c in chosen if c["kind"] == "record"]
    rag = [c["text"] for c in chosen if c["kind"] == "rag"]
    conversation = [c["text"] for c in chosen if c["kind"] == "conversation"]
    action = safety.get("safety_action", safety.get("action", "GUIDE"))
    types = {str(t) for t in safety.get("safety_types", safety.get("types", []))}
    parts = []
    if data.get("child_record", {}).get("coverage", {}).get("synthetic") and record:
        parts.append("테스트용 합성 기록을 기준으로 안내합니다.")
    if action == "EMERGENCY":
        return "긴급한 안전 확인이 필요한 상황입니다. 즉시 119에 연락해 현장 상황을 알리고 안내를 따르세요."
    if action == "BLOCK":
        rag = []
        conversation = []
        if types & {"4", "10"}:
            parts.append("아이의 복용량·간격·추가 투약이나 처방 변경은 정해드릴 수 없습니다. 약 이름, 먹인 양과 시각을 정리해 의료진이나 약사에게 확인해 주세요.")
        elif types & {"1", "2", "12"}:
            parts.append("개인의 질병·발달 진단이나 확률을 판단해드릴 수 없습니다. 관찰한 증상과 기록을 정리해 의료진과 상담해 주세요.")
        else:
            parts.append("개인에게 필요한 치료나 의료적 결정을 대신 내려드릴 수 없습니다. 관찰한 내용과 기록을 정리해 의료진에게 확인해 주세요.")
    elif action == "GUIDE":
        if "3" in types:
            parts.append("진료를 미루거나 내일까지 기다려도 안전한지는 확정해 드릴 수 없습니다. 증상이 시작된 시각, 체온, 먹고 마신 양과 복약 기록을 정리해 의료진에게 확인해 주세요. 호흡곤란·청색증·의식저하가 있으면 즉시 119에 연락하세요.")
        else:
            parts.append("아이 개인의 원인·진단·치료 여부는 단정할 수 없어, 확인된 기록과 전문자료의 일반 안내를 구분해 드립니다.")
    if record:
        parts.append("육아기록 기준: " + "\n".join(record))
        if data.get("child_record", {}).get("coverage", {}).get("partial_today"):
            parts.append("오늘 기록은 현재 시점까지 입력된 일부 기록입니다.")
    if rag:
        parts.append("전문자료의 일반 안내: " + "\n".join(rag))
    if conversation:
        parts.append("\n".join(conversation))
    for source, label in (("child_record", "육아기록"), ("professional_rag", "전문자료")):
        source_data = data.get(source, {})
        if source_data.get("partial") or source_data.get("delta_error"):
            parts.append(f"추가 {label} 조회가 완료되지 않아, 새로 요청한 범위는 확인하지 못했고 기존에 확인한 근거만 사용했습니다.")
    if action != "BLOCK":
        missing = []
        rag_coverage = data.get("professional_rag", {}).get("coverage", {})
        facets = rag_coverage.get("missing_facets", [])
        facet_labels = {"water_intake_amount": "물의 권장량", "drinking_water_start_age": "물을 제공하기 시작할 시기"}
        labels = [facet_labels[key] for key in facets if key in facet_labels]
        if labels:
            missing.append("현재 전문자료에는 " + "·".join(labels) + "를 직접 설명하는 근거가 없어 해당 범위는 안내하기 어렵습니다.")
        if any(key not in facet_labels for key in facets) or (rag_coverage.get("partial_evidence") and not facets):
            missing.append("현재 전문자료는 질문의 일부만 다루므로 확인되지 않은 범위는 추정하지 않겠습니다.")
        if "child_record" in data and data["child_record"].get("status") != "ok":
            missing.append("육아기록 조회에 문제가 있어 실제 기록 수치나 개인 상태를 확인할 수 없습니다.")
        if "professional_rag" in data and (data["professional_rag"].get("status") != "ok" or not rag):
            reason_text = {"no_temperature_measurement_in_corpus": "현재 전문자료에는 체온 측정 부위와 측정 방법을 직접 설명하는 근거가 없습니다.",
                           "no_daily_care_recording_in_corpus": "현재 전문자료에는 매일 어떤 육아 항목을 기록할지 정한 체크리스트 근거가 없습니다."}
            missing.append(reason_text.get(rag_coverage.get("missing_evidence_reason"), "현재 질문에 답할 전문자료 근거가 충분하지 않아 일반 의학 정보를 추정하지 않겠습니다."))
        parts.extend(missing)
        if not record and not rag and not conversation and not missing:
            parts.append("현재 질문에 답할 수 있는 기록이나 전문자료 근거가 없습니다.")
    return "\n".join(parts)
