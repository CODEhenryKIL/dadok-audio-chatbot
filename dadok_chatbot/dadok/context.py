"""Small presentation projection; IDs and retrieval diagnostics stay in logs."""
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import re


def rewrite_schema(question):
    """Constrain only an explicitly requested line count, not the prose."""
    # Only simple, positive, exact-count requests are mechanically constrained.
    # Corrections, negations and upper bounds retain semantic prose generation.
    match = re.fullmatch(
        r'\s*(?:(?:방금|이전|앞선|위|이)\s*)?'
        r'(?:(?:답변|내용|설명|안내)(?:을|를)?\s*)?'
        r'(\d{1,2}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*줄(?:로|만으로|만)\s*'
        r'(?:요약|정리)(?:해\s*줘요?|해\s*주세요|해)?[.!?]?\s*', question)
    if not match:
        return None
    word = match.group(1)
    count = int(word) if word.isdigit() else ['한','두','세','네','다섯','여섯','일곱','여덟','아홉','열'].index(word) + 1
    if not 1 <= count <= 20:
        return None
    return {'type': 'object', 'additionalProperties': False, 'required': ['lines'],
            'properties': {'lines': {'type': 'array', 'minItems': count, 'maxItems': count,
                'description': '요약한 답변의 각 줄. 기존 답변의 핵심과 필요한 적용 조건을 유지하고, 새 조언이나 서론을 추가하지 않는다.',
                'items': {'type': 'string', 'pattern': r'^[^\S\r\n]*\S[^\r\n]*$'}}}}

NAMES = {'sleep': '수면', 'night': '밤잠', 'nap': '낮잠', 'feeding': '수유', 'formula': '분유',
         'breast': '모유', 'solid': '이유식', 'temperature': '체온', 'diaper': '기저귀',
         'poop': '대변', 'pee': '소변', 'both': '대변·소변', 'growth': '성장',
         'weight': '몸무게', 'height': '키', 'medicine': '복약', 'fever_reducer': '해열제',
         'symptom': '증상', 'cough': '기침', 'vomit': '구토', 'all': ''}

def value_text(value, unit):
    if value is None: return '수치 기록 없음'
    number = f'{value:g}' if isinstance(value, (int, float)) else str(value)
    if unit == '분':
        hour, minute = divmod(float(value), 60)
        equivalent = (f'{int(hour)}시간' if hour else '') + (f' {minute:g}분' if minute else '')
        return f'{number}분 ({equivalent.strip() or "0분"})'
    return number + (' ' + unit if unit != 'unknown' else '')


def _calculated_value(value, unit):
    """Readable derived duration; stored measurements/statistics stay exact."""
    if value is None or unit != '분':
        return value_text(value, unit)
    minutes = Decimal(str(value))
    rounded = abs(minutes).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    if minutes and not rounded:
        return ('-' if minutes < 0 else '') + '1분 미만'
    hours, remainder = divmod(int(rounded), 60)
    text = ' '.join(part for part in (f'{hours}시간' if hours else '',
                                      f'{remainder}분' if remainder else '') if part) or '0분'
    if minutes < 0 and rounded:
        text = '-' + text
    return ('약 ' if rounded != abs(minutes) else '') + text


def _calculated_change(value, unit):
    if value == 0:
        return '변화 없음'
    return _calculated_value(abs(value), unit) + (' 감소' if value < 0 else ' 증가')

def period_text(period):
    if not period: return ''
    try:
        begin = datetime.fromisoformat(period['start'])
        last = datetime.fromisoformat(period['end_exclusive']) - timedelta(microseconds=1)
        return begin.strftime('%Y-%m-%d') + '부터 ' + last.strftime('%Y-%m-%d') + '까지 (한국 시간)'
    except (KeyError, ValueError): return ''

def statistics_view(stats):
    groups = []
    for key, value in stats.get('groups', {}).items():
        kind, sub, unit = key.split(':')
        name = NAMES.get(sub, sub) if sub != 'all' else NAMES.get(kind, kind)
        groups.append({'항목': name, '횟수': value['count'],
                       '합계': value_text(value.get('total'), unit),
                       '기록당 평균': value_text(value.get('mean'), unit),
                       '기록 있는 날의 하루 평균': value_text(value.get('daily_mean'), unit),
                       '최댓값': value_text(value.get('max'), unit), '최솟값': value_text(value.get('min'), unit),
                       '일별 합계': {date: value_text(v, unit) for date, v in value.get('daily', {}).items()},
                       '최대인 날짜': value.get('max_day'), '최소인 날짜': value.get('min_day'),
                       '기록 있는 날 수': value.get('recorded_days')})
    result = {'기간': period_text(stats.get('period')), '전체 횟수': stats.get('count'), '항목별 계산': groups,
              '주의': '기록이 없는 날은 0으로 추정하지 않았습니다. 합계는 관측기록의 합이며, 체온·체중·키의 합계는 해석하지 마세요.'}
    combined = stats.get('totals_by_type_and_unit', {})
    if combined: result['종류별 통합 계산'] = statistics_view({'groups': combined})['항목별 계산']
    comparison = stats.get('comparison')
    if comparison:
        result['이전 기간'] = statistics_view(comparison['previous'])
        result['변화 (같은 원 단위)'] = comparison['changes']
    return result


def _requested_groups(stats, plan):
    operation = plan.get('operation', 'summary')
    requested = set(plan.get('types', []))
    groups = dict(stats.get('groups', {}))
    if operation not in ('list', 'latest'):
        combined = stats.get('totals_by_type_and_unit', {})
        for kind in ('sleep', 'feeding'):
            totals = {key: value for key, value in combined.items() if key.split(':')[0] == kind}
            if totals and (kind in requested or 'all' in requested):
                groups = {key: value for key, value in groups.items() if key.split(':')[0] != kind}
                groups.update(totals)
    return groups


def _record_fact(row):
    return {'항목': NAMES.get(row.get('subtype'), row.get('subtype')) or NAMES.get(row.get('type'), '기록'),
            '시각': row.get('occurred_at'), '값': value_text(row.get('amount'), row.get('unit') or 'unknown'),
            '메모': row.get('note') or ''}


def _requested_statistics(stats, plan, *, include_recorded_dates=False):
    """Project the requested calculations with explicit units and denominators."""
    operation = plan.get('operation', 'summary')
    groups = _requested_groups(stats, plan)
    projected = []
    for key, value in groups.items():
        kind, sub, unit = key.split(':')
        name = NAMES.get(sub, sub) if sub != 'all' else NAMES.get(kind, kind)
        item = {'항목': name, '횟수': value['count']}
        cumulative = kind in ('sleep', 'feeding', 'diaper')
        if operation in ('average', 'summary', 'compare'):
            if cumulative:
                item['기록 있는 날의 하루 평균'] = _calculated_value(value.get('daily_mean'), unit)
                item['평균 계산에 사용한 날 수'] = value.get('recorded_days')
                item['평균의 대상'] = f"기록이 있는 {value.get('recorded_days', 0)}일"
                if value.get('period_days') is not None:
                    item['평균의 대상'] = f"조회한 {value['period_days']}일 중 " + item['평균의 대상']
                if operation == 'summary' or include_recorded_dates:
                    item['기록 있는 날짜'] = sorted(value.get('daily', {}))
                item['미기록일 처리'] = '평균에서 제외; 0으로 간주하지 않음'
            else:
                item['측정 기록당 평균'] = _calculated_value(value.get('mean'), unit)
                item['평균 계산에 사용한 수치 기록 수'] = value.get('numeric_count')
                item['수치 기록이 있는 날 수'] = value.get('recorded_days')
        if operation in ('total', 'summary', 'compare'):
            if kind not in ('temperature', 'growth'):
                item['합계'] = _calculated_value(value.get('total'), unit)
            elif operation == 'total':
                item['계산 제한'] = '체온·몸무게·키의 측정값 합계는 제공하지 않음'
        if operation in ('min', 'max', 'select'):
            extremes = ('min', 'max') if operation == 'select' else (operation,)
            for extreme_name in extremes:
                label = '최솟값' if extreme_name == 'min' else '최댓값'
                item['기록 1회의 ' + label] = {
                    '값': _calculated_value(value.get(extreme_name), unit),
                    '해당 시각 (동률 포함)': value.get(extreme_name + '_at', [])}
                if cumulative:
                    daily = value.get('daily', {})
                    if daily:
                        extreme = (min if extreme_name == 'min' else max)(daily.values())
                        item['하루 합계의 ' + label] = {
                            '값': _calculated_value(extreme, unit),
                            '해당 날짜 (동률 포함)': sorted(date for date, amount in daily.items() if amount == extreme)}
            if operation == 'select':
                for name, label in (('first', '가장 처음 기록'), ('last', '가장 최근 기록')):
                    if value.get(name):
                        item[label] = _record_fact(value[name])
        projected.append(item)
    result = {'기간': period_text(stats.get('period')), '전체 횟수': stats.get('count'), '항목별 계산': projected}
    comparison = stats.get('comparison')
    if comparison:
        result['이전 기간'] = _requested_statistics(comparison['previous'], plan, include_recorded_dates=include_recorded_dates)
        # Compare matching selected groups, never reuse a subtype's difference
        # for a broader category or ask the language model to subtract values.
        previous = _requested_groups(comparison['previous'], plan)
        changes = []
        for key, value in groups.items():
            old = previous.get(key)
            if old is None:
                continue
            kind, sub, unit = key.split(':')
            change = {'항목': NAMES.get(sub, sub) if sub != 'all' else NAMES.get(kind, kind),
                      '수치 단위': unit, '기록 횟수 차이': value['count'] - old['count']}
            mean_key = 'daily_mean' if kind in ('sleep', 'feeding', 'diaper') else 'mean'
            if value.get(mean_key) is not None and old.get(mean_key) is not None:
                change['하루 평균 차이' if mean_key == 'daily_mean' else '측정 기록당 평균 차이'] = _calculated_change(round(value[mean_key] - old[mean_key], 4), unit)
            if kind not in ('temperature', 'growth') and value.get('total') is not None and old.get('total') is not None:
                change['합계 차이'] = _calculated_change(round(value['total'] - old['total'], 4), unit)
            changes.append(change)
        if changes:
            result['변화 (현재 기간 − 이전 기간)'] = changes
    return result


def present_records(data, question):
    if data.get('status') != 'ok': return data
    coverage = data.get('coverage', {})
    plan = data.get('record_query') or {}
    operation = plan.get('operation', 'summary')
    stats = _requested_statistics(data.get('statistics', {}), plan,
                                  include_recorded_dates=bool(re.search(r'날짜|어느 날|언제', question)))
    view = {'status': 'ok', '데이터 종류': '테스트용 합성 기록' if coverage.get('synthetic') else '서버 조회 기록',
            '아이': {k: v for k, v in data.get('child', {}).items() if k != 'id'},
            '제공된 통계 묶음': {'average': '평균', 'total': '합계', 'count': '횟수',
                'select': '최대·최소·최초·최근', 'compare': '기간 비교', 'list': '기록 목록',
                'summary': '요약'}.get(operation, operation), '코드 계산 결과': stats,
            '조회 상태': '성공'}
    if plan.get('types'):
        view['조회한 항목'] = [NAMES.get(kind.split(':')[-1], kind) for kind in plan['types']]
        if 'all' not in plan['types']:
            view['조회 범위의 한계'] = '요청한 항목만 조회했습니다. 다른 항목의 기록 유무는 확인하지 않았습니다.'
    if coverage.get('partial_today'):
        view['오늘 기록 범위'] = '오늘 현재까지 입력된 기록. 이후 기록은 아직 없음; 조회 실패가 아님.'
    if coverage.get('records_truncated'):
        view['일부 기록 표시'] = True
    if data.get('delta_error'):
        view['추가 조회 실패'] = data['delta_error']
    # Aggregation questions need calculated facts, not every historical event.
    if operation not in ('average', 'total', 'count', 'compare', 'select'):
        rows = data.get('records', [])
        if operation == 'latest':
            latest = {key: group['last'] for key, group in data.get('statistics', {}).get('groups', {}).items()
                      if group.get('last')}
            if not latest:
                for row in rows:
                    latest[(row.get('type'), row.get('subtype'), row.get('unit'))] = row
            rows = sorted(latest.values(), key=lambda row: row.get('occurred_at', ''))
        view['시간순 기록 (과거→최근)'] = [_record_fact(row) for row in rows]
    return view


def _spoken_value(value, unit):
    spoken_unit = {'ml': '밀리리터', 'mL': '밀리리터', '°C': '도', 'kg': '킬로그램', 'cm': '센티미터'}.get(unit)
    text = f'{value:g}{spoken_unit}' if spoken_unit else _calculated_value(value, unit)
    last = ord(text[-1])
    return text + ('이에요' if 0xAC00 <= last <= 0xD7A3 and (last - 0xAC00) % 28 else '예요')


def without_voice_offers(answer):
    """Remove standalone optional assistant offers, never conditional care advice."""
    offer = re.compile(
        r'^(?:원하시면|필요하시면|필요하면|필요한 경우|더\s*(?:자세히|확인|설명|알려|정리))'
        r'.*(?:드릴게요|드릴까요|드리겠습니다|해볼까요|해드릴 수 있어요)[.!?。]*$')
    parts = re.split(r'((?<=[.!?。])\s+|\n+)', answer.strip())
    return ''.join(parts[index] + (parts[index + 1] if index + 1 < len(parts) else '')
                   for index in range(0, len(parts), 2)
                   if not offer.fullmatch(parts[index].strip())).strip()


def voice_calculation(data, question, safety, retrieval):
    """Speak small, already-computed record results without reinterpreting them.

    Explanations, medical judgments, comparisons and mixed RAG requests continue
    through the answer model. No supplied measurement is recomputed here.
    """
    if (safety['safety_action'] != 'ALLOW' or set(data) != {'child_record'}
            or retrieval.get('answer_task') == 'rewrite'
            or re.search(r'날짜|언제|어느 날|계산|근거|왜|이유|어떻게|자세|상세|기준|제외|미기록|목록|비교', question)):
        return None
    record = data['child_record']
    plan = record.get('record_query') or {}
    operation = plan.get('operation')
    if (record.get('status') != 'ok' or record.get('partial') or record.get('delta_error')
            or operation not in ('average', 'total', 'count') or plan.get('compare_previous')):
        return None
    groups = _requested_groups(record.get('statistics', {}), plan)
    if not 1 <= len(groups) <= 2:
        return None
    start, end = (datetime.fromisoformat(plan[key]) for key in ('start_date', 'end_date'))
    days = (end - start).days + 1
    sentences = []
    for key, group in groups.items():
        kind, sub, unit = key.split(':')
        name = NAMES.get(sub, sub) if sub != 'all' else NAMES.get(kind, kind)
        if operation == 'count':
            sentences.append(f"조회한 {days}일 동안 {name} 기록은 {group['count']}회예요.")
        elif operation == 'total':
            if kind not in ('sleep', 'feeding', 'diaper') or group.get('total') is None:
                return None
            sentences.append(f"조회한 {days}일 동안 기록된 {name} 합계는 {_spoken_value(group['total'], unit)}.")
        elif kind in ('sleep', 'feeding', 'diaper'):
            if group.get('daily_mean') is None or not group.get('recorded_days'):
                return None
            sentences.append(f"조회한 {days}일 중 {name} 기록이 있는 {group['recorded_days']}일의 "
                             f"하루 평균은 {_spoken_value(group['daily_mean'], unit)}.")
        else:
            if group.get('mean') is None or not group.get('numeric_count'):
                return None
            sentences.append(f"조회한 {days}일 동안 {name} 측정 기록 {group['numeric_count']}건의 "
                             f"평균은 {_spoken_value(group['mean'], unit)}.")
    return ' '.join(sentences)


def present_sources(data, question):
    result = {}
    if 'child_record' in data: result['육아기록'] = present_records(data['child_record'], question)
    if 'professional_rag' in data:
        rag = data['professional_rag']
        result['전문자료'] = {'상태': rag.get('status'),
            '근거': [{'자료명': c.get('title', '전문자료'), '페이지': c.get('page'),
                      '원문 연령 범위': c.get('age_scope'),
                      '적용 범위 주의': c.get('age_applicability', '') if c.get('age_scope') else
                          '본문 전체에 공통으로 적용되는 연령 범위는 확인되지 않았습니다. '
                          '범위가 없다는 것은 모든 연령에 적용된다는 뜻이 아닙니다. '
                          '개별 문단의 연령·대상·적용 조건을 확인하세요.',
                      **({'페이지 제목': c.get('context_page_heading', ''),
                          '본문': [{'제목': section['heading'], '문단': section['paragraphs']}
                                   for section in c['context_sections']]}
                         if c.get('context_sections') else
                         {'본문': c.get('context_text', c.get('text', ''))}),
                      **({'문맥 주의': c['context_warnings']} if c.get('context_warnings') else {})}
                     for c in rag.get('chunks', [])],
            '근거 범위': '현재 검색된 원문만 확인됨. 자료 전체의 부재를 뜻하지 않음.'}
        if rag.get('delta_error'):
            result['전문자료']['추가 조회 실패'] = rag['delta_error']
    return result


def present_restatement(answer, question):
    """Change display/register only; preserve selected claims and numeric facts."""
    import re
    text = answer.removeprefix('전문자료의 일반 안내: ').strip()
    # Source IDs remain in logs. Remove repetitive bibliographic display, never
    # the original age/disease scope that controls the meaning of the claim.
    text = re.sub(r'연령별 적용 조건을 확인해야 하는 일반 자료: ', '', text)
    text = re.sub(r'(?<!원문 범위 )자료 「[^」]+」: ', '', text)
    if re.search(r'쉽게|쉬운|간단히', question):
        for old, new in [('이유기 보충식', '이유식'), ('이유기보충식', '이유식'), ('조제유', '분유'), ('양육자', '보호자'), ('태내', '엄마 뱃속'), ('권장하는', '권하는'), ('가능성이 높아지고', '가능성이 커지고'), ('때문입니다', '때문이에요'), ('권장하고 있습니다', '권장해요'), ('않습니다', '않아요'), ('있습니다', '있어요'), ('없습니다', '없어요'),
                         ('좋습니다', '좋아요'), ('것입니다', '거예요'), ('합니다', '해요'), ('됩니다', '돼요')]:
            text = text.replace(old, new)
        text = '쉽게 말하면, ' + text
    items = [item.strip(' -•') for item in re.split(r'(?<=[.!?])\s+|\n', text) if item.strip()]
    if '체크리스트' in question:
        text = '앞선 안내를 확인 항목으로 정리했어요.\n' + '\n'.join('□ ' + item for item in items)
    elif re.search(r'(?:두|2)\s*(?:개|가지)', question):
        selected = items[:2]
        text = '\n'.join(f'{i + 1}. {item}' for i, item in enumerate(selected))
        if len(selected) < 2:
            text = f'앞선 답변에서 확인된 기준은 {len(selected)}개여서 그 범위만 정리해요.\n' + text
    elif re.search(r'(?:세|3)\s*줄', question):
        # Physical line wrapping preserves the selected source wording exactly.
        # It does not invent extra "facts" to meet a requested number of lines.
        words = text.split()
        lines = []
        while len(lines) < 2 and len(words) > 3 - len(lines):
            target = len(' '.join(words)) / (3 - len(lines))
            count, size = 0, 0
            while count < len(words) - (2 - len(lines)):
                next_size = size + len(words[count]) + (1 if count else 0)
                if count and abs(size - target) <= abs(next_size - target):
                    break
                size = next_size
                count += 1
            lines.append(' '.join(words[:count]))
            words = words[count:]
        lines.append(' '.join(words))
        text = '\n'.join(lines)
    return text
