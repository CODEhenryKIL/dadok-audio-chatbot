"""Read-only adapter for the verified babylog sync API and explicit exports.

The export is an API snapshot, not an invented PostgreSQL schema. Server-side
authorization determines the family scope; local filtering determines the child.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
RECORD_QUERY_TYPES = frozenset({
    "sleep", "sleep:night", "sleep:nap", "feeding", "feeding:formula",
    "feeding:breast", "feeding:solid", "temperature", "diaper", "diaper:poop",
    "diaper:pee", "medicine", "growth", "growth:weight", "growth:height",
    "symptom", "symptom:cough", "symptom:vomit", "all",
})
RECORD_QUERY_OPERATIONS = frozenset({
    "summary", "average", "total", "count", "select", "min", "max", "list", "latest", "compare",
})


class RecordError(RuntimeError):
    """A safe error code without record contents or credentials."""


def normalize_record_query(plan: dict, now: datetime) -> dict:
    """Validate the semantic resolver's query, never reinterpret its user text.

    Dates are inclusive KST calendar dates. A resolver may request ``all`` only
    when the user explicitly requests overall records; an unknown type is not
    an alias for that scope. No database/schema change is involved.
    """
    fields = {"types", "start_date", "end_date", "operation", "compare_previous"}
    if not isinstance(plan, dict) or set(plan) != fields:
        raise RecordError("record_query_invalid")
    types = plan["types"]
    if (not isinstance(types, list) or not types or len(types) > len(RECORD_QUERY_TYPES)
            or any(not isinstance(value, str) or value not in RECORD_QUERY_TYPES for value in types)
            or "all" in types and len(types) != 1):
        raise RecordError("record_query_invalid")
    operation, compare = plan["operation"], plan["compare_previous"]
    if (not isinstance(operation, str) or operation not in RECORD_QUERY_OPERATIONS
            or type(compare) is not bool or operation == "compare" and not compare):
        raise RecordError("record_query_invalid")
    dates = []
    for key in ("start_date", "end_date"):
        value = plan[key]
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise RecordError("record_query_invalid")
        try:
            dates.append(datetime.strptime(value, "%Y-%m-%d").date())
        except ValueError:
            raise RecordError("record_query_invalid") from None
    start, end = dates
    days = (end - start).days + 1
    if days < 1 or days > 366 or end > now.astimezone(KST).date():
        raise RecordError("record_query_invalid")
    if compare:
        try:
            start - timedelta(days=days)
        except OverflowError:
            raise RecordError("record_query_invalid") from None
    # A general category subsumes its subtypes; eliminate redundant filters.
    types = list(dict.fromkeys(value for value in types
                             if ":" not in value or value.split(":", 1)[0] not in types))
    return {"types": types, "start_date": start.isoformat(), "end_date": end.isoformat(),
            "operation": operation, "compare_previous": compare}


def _query_filters(plan):
    return [tuple(value.split(":", 1)) if ":" in value else (value, None)
            for value in plan["types"]]


def _instant(value: str) -> datetime:
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _id(value):
    return str(value or "").lower()


def _filters(question: str) -> list[tuple[str, str | None]]:
    rules = [
        (r"밤잠", "sleep", "night"), (r"낮잠", "sleep", "nap"),
        (r"수면|잠", "sleep", None), (r"분유", "feeding", "formula"),
        (r"이유식", "feeding", "solid"), (r"모유|직수", "feeding", "breast"),
        (r"수유|먹는 양|먹은 양", "feeding", None), (r"체온|온도|몇 도", "temperature", None),
        (r"대변|똥", "diaper", "poop"), (r"소변|오줌", "diaper", "pee"),
        (r"기저귀", "diaper", None), (r"복약|투약|약 기록|먹인 약|약을", "medicine", None),
        (r"몸무게|체중", "growth", "weight"), (r"키|신장", "growth", "height"),
        (r"성장", "growth", None), (r"기침", "symptom", "cough"),
        (r"구토|토한|토했", "symptom", "vomit"), (r"증상", "symptom", None),
    ]
    result = []
    for pattern, kind, subtype in rules:
        if re.search(pattern, question):
            if subtype is None and any(k == kind for k, _ in result):
                continue
            result.append((kind, subtype))
    return result


def _matches(record: dict, filters: list) -> bool:
    for kind, subtype in filters:
        if kind == "all":
            return True
        if record.get("type") == kind and (subtype is None or record.get("subtype") == subtype
                or kind == "diaper" and record.get("subtype") == "both" and subtype in {"poop", "pee"}):
            return True
    return False


class RecordStore:
    def __init__(self, snapshot_path: str | Path | None = None, *, base_url: str = "",
                 token: str = "", authenticated_user_id: str = "", transport=None,
                 now: Callable[[], datetime] | None = None):
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.base_url = (base_url or "").rstrip("/")
        self.token = token
        self.authenticated_user_id = authenticated_user_id
        self.transport = transport
        self.now = now or (lambda: datetime.now(KST))
        self._records: dict[str, dict] = {}
        self._cursor = 0
        self._lock = asyncio.Lock()

    async def _get(self, path: str) -> dict:
        headers = {"Authorization": f"Bearer {self.token}", "Accept-Language": "ko"}
        if self.transport:
            result = await self.transport(self.base_url + path, headers)
        else:
            def get():
                request = urllib.request.Request(self.base_url + path, headers=headers)
                # Never forward a bearer token through an HTTP redirect.
                class NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
                        return None
                try:
                    with urllib.request.build_opener(NoRedirect).open(request, timeout=15) as response:
                        return json.load(response)
                except urllib.error.HTTPError as exc:
                    raise RecordError("record_auth_failed" if exc.code in {401, 403} else "record_http_failed") from None
                except (OSError, ValueError):
                    raise RecordError("record_unavailable") from None
            result = await asyncio.to_thread(get)
        if not isinstance(result, dict):
            raise RecordError("record_invalid_response")
        return result

    async def _load(self, user_id: str):
        if self.snapshot_path:
            try:
                data = await asyncio.to_thread(lambda: json.loads(self.snapshot_path.read_text()))
            except (OSError, ValueError):
                raise RecordError("record_snapshot_unavailable") from None
            if data.get("schema_version") != "babylog-api-snapshot.v1" or data.get("user_id") != user_id:
                raise RecordError("record_scope_denied")
            children, records = data.get("children"), data.get("records")
            if not isinstance(children, list) or not isinstance(records, list):
                raise RecordError("record_invalid_snapshot")
            return children, records, bool(data.get("synthetic", False)), data.get("captured_at")
        if not self.base_url or not self.token or not self.authenticated_user_id:
            raise RecordError("record_auth_not_configured")
        if user_id != self.authenticated_user_id:
            raise RecordError("record_scope_denied")
        if urllib.parse.urlsplit(self.base_url).scheme != "https":
            raise RecordError("record_https_required")
        children = (await self._get("/v1/children")).get("children")
        if not isinstance(children, list):
            raise RecordError("record_invalid_response")
        # Stage changes: a failed page must not publish a partial restore.
        staged, cursor = dict(self._records), self._cursor
        for _ in range(1000):
            page = await self._get(f"/v1/sync/pull?after={cursor}&limit=500")
            rows, next_after, more = page.get("records"), page.get("next_after"), page.get("has_more")
            if not isinstance(rows, list) or type(next_after) is not int or type(more) is not bool:
                raise RecordError("record_invalid_response")
            for row in rows:
                if not isinstance(row, dict) or not row.get("id") or type(row.get("seq")) is not int:
                    raise RecordError("record_invalid_response")
                if row["seq"] <= cursor or row["seq"] > next_after:
                    raise RecordError("record_invalid_cursor")
                staged[_id(row["id"])] = row
            if next_after < cursor or more and next_after <= cursor:
                raise RecordError("record_invalid_cursor")
            cursor = next_after
            if not more:
                self._records, self._cursor = staged, cursor
                return children, list(staged.values()), False, self.now().isoformat()
        raise RecordError("record_pagination_limit")

    async def fetch(self, user_id: str, child_id: str, question: str,
                    previous: dict | None = None, delta: bool = False, *,
                    record_query: dict | None = None) -> dict:
        async with self._lock:
            children, raw, synthetic, captured = await self._load(user_id)
        known = {_id(c.get("id")): c for c in children if isinstance(c, dict)}
        if _id(child_id) not in known:
            raise RecordError("record_child_scope_denied")
        child = known[_id(child_id)]
        now = self.now().astimezone(KST)
        latest = {}
        invalid = unassigned = unconfirmed = 0
        for row in raw:
            if not isinstance(row, dict) or not row.get("id"):
                invalid += 1
                continue
            old = latest.get(_id(row["id"]))
            # A sync snapshot may contain repeated UUID spellings; last sequence wins.
            if old is None or row.get("seq", 0) >= old.get("seq", 0):
                latest[_id(row["id"])] = row
        rows = []
        for row in latest.values():
            if row.get("deleted") is True:
                continue
            # Do not silently assign legacy/invalid child IDs to the first child.
            if _id(row.get("child_id")) not in known:
                unassigned += 1
                continue
            if _id(row.get("child_id")) != _id(child_id):
                continue
            if row.get("needs_confirmation") is True:
                unconfirmed += 1
                continue
            try:
                occurred = _instant(row["occurred_at"]).astimezone(KST)
                if occurred > now:
                    continue
            except (KeyError, TypeError, ValueError):
                invalid += 1
                continue
            clean = {key: row.get(key) for key in ("id", "type", "subtype", "action", "amount", "unit", "note", "ended_at")}
            clean["occurred_at"] = occurred.isoformat()
            rows.append(clean)
        rows.sort(key=lambda row: row["occurred_at"])
        rows, incomplete_sleep = _sleep_intervals(rows)
        query = question
        if record_query is not None:
            plan = normalize_record_query(record_query, now)
            start = datetime.combine(datetime.strptime(plan["start_date"], "%Y-%m-%d").date(), time.min, KST)
            end = datetime.combine(datetime.strptime(plan["end_date"], "%Y-%m-%d").date(), time.min, KST) + timedelta(days=1)
            days, compare = (end - start).days, plan["compare_previous"]
            filters = _query_filters(plan)
        else:
            # Compatibility for explicit legacy callers. Production semantic
            # plans above are authoritative and do not scan question keywords.
            filters = _filters(question)
            if delta and previous and not filters:
                filters = [tuple(f) for f in previous.get("coverage", {}).get("filters", [])]
                query = previous.get("query", "") + " / " + question
            if not filters and _overall_records_requested(question):
                filters = [("all", None)]
            if not filters:
                raise RecordError("record_scope_ambiguous")
            start, end, days, compare = _period(question, now, rows, previous if delta else None)
            plan = normalize_record_query({
                "types": [kind + (":" + subtype if subtype else "") for kind, subtype in filters],
                "start_date": start.date().isoformat(),
                "end_date": (end - timedelta(days=1)).date().isoformat(),
                "operation": _legacy_operation(question, compare), "compare_previous": compare,
            }, now)
            filters = _query_filters(plan)
        selected = [r for r in rows if _matches(r, filters)]
        current = _within(selected, start, end)
        statistics = _statistics(current, start, end, days)
        comparison = None
        if compare:
            before_start = start - timedelta(days=days)
            before = _statistics(_within(selected, before_start, start), before_start, start, days)
            comparison = {"current": statistics, "previous": before, "changes": {}}
            for key, group in statistics["groups"].items():
                old = before["groups"].get(key)
                if old and group["total"] is not None and old["total"] is not None:
                    comparison["changes"][key] = {
                        "total_difference": round(group["total"] - old["total"], 4),
                        "recorded_daily_mean_difference": round(group["daily_mean"] - old["daily_mean"], 4),
                        "count_difference": group["count"] - old["count"],
                    }
        if comparison:
            statistics["comparison"] = comparison
            # Avoid a self-referential dict in the JSON context.
            comparison["current"] = {k: v for k, v in statistics.items() if k != "comparison"}
        all_source_rows = _within(selected, start - timedelta(days=days) if compare else start, end)
        return {
            "status": "ok", "source": "child_record", "query": query, "record_query": plan,
            "child": {k: child.get(k) for k in ("id", "name", "birthday", "gender")},
            "records": current[-120:], "statistics": statistics,
            "coverage": {"start": start.isoformat(), "end_exclusive": end.isoformat(), "days": days,
                "timezone": "Asia/Seoul", "filters": filters, "synthetic": synthetic,
                "captured_at": captured, "partial_today": end > now, "missing_days_are_unknown": True,
                "records_total": len(current), "records_truncated": len(current) > 120,
                "invalid_records": invalid, "unassigned_child_records": unassigned,
                "unconfirmed_records": unconfirmed, "incomplete_sleep_events": incomplete_sleep,
                "delta": delta, "source_snapshot_complete": True,
                "source_id_count": len(all_source_rows), "source_ids_truncated": len(all_source_rows) > 250},
            "source_ids": ["record:" + r["id"] for r in all_source_rows[-250:]],
        }


def _overall_records_requested(question):
    # Broad record requests remain supported, but a negated reference to the
    # previous record output must never mean "fetch every category".
    if re.search(r"기록\s*(?:말고|대신|빼|제외|아니)", question):
        return False
    return bool(re.search(r"(?:전체|모든|육아)\s*기록|기록\s*전체", question)
                or re.fullmatch(r"\s*(?:(?:오늘|어제|최근\s*\d+일)\s*)?기록(?:을|만)?\s*(?:보여줘|알려줘)?[.!?]?\s*", question))


def _legacy_operation(question, compare):
    if compare:
        return "compare"
    for pattern, operation in ((r"평균", "average"), (r"총|합계", "total"),
                               (r"몇 번|횟수", "count"), (r"최고|가장 높은", "max"),
                               (r"최저|가장 낮은", "min"), (r"마지막|최신", "latest"),
                               (r"목록|시간순", "list")):
        if re.search(pattern, question):
            return operation
    return "summary"


def _period(question, now, rows, previous):
    midnight = datetime.combine(now.date(), time.min, KST)
    end = midnight + timedelta(days=1)
    match = re.search(r"(?:최근\s*)?(\d{1,3})\s*일", question)
    days = max(1, min(366, int(match.group(1)))) if match else 14
    compare = bool(re.search(r"이전|비교|지난주.*(?:보다|대비)", question))
    if not match and previous:
        days = previous.get("coverage", {}).get("days", 14)
    if "오늘" in question:
        days = 1
    elif "어제" in question:
        days, end = 1, midnight
    elif "이번 주" in question or "이번주" in question:
        days = now.weekday() + 1
    elif "지난주" in question or "지난 주" in question:
        days = 7
        end = midnight - timedelta(days=now.weekday())
    elif not match and re.search(r"전체|모든|기록된 체온 중", question) and rows:
        days = max(1, (now.date() - _instant(rows[0]["occurred_at"]).astimezone(KST).date()).days + 1)
    start = end - timedelta(days=days)
    return start, end, days, compare


def _sleep_intervals(rows):
    """Pair explicit start/end events; never invent the end of an open sleep."""
    result, pending, incomplete = [], {}, 0
    for row in rows:
        if row.get("type") != "sleep":
            result.append(row)
            continue
        kind = row.get("subtype")
        if row.get("ended_at"):
            try:
                end = _instant(row["ended_at"])
                duration = (end - _instant(row["occurred_at"])).total_seconds() / 60
                if duration < 0:
                    raise ValueError()
                result.append({**row, "amount": duration, "unit": "분", "interval_end": end.astimezone(KST).isoformat()})
            except (ValueError, TypeError):
                incomplete += 1
        elif row.get("action") == "start":
            if kind in pending:
                incomplete += 1
            pending[kind] = row
        elif row.get("action") == "end":
            first = pending.pop(kind, None)
            if first is None and len(pending) == 1 and (kind is None or None in pending):
                first = pending.pop(next(iter(pending)))
            if first:
                duration = (_instant(row["occurred_at"]) - _instant(first["occurred_at"])).total_seconds() / 60
                result.append({**first, "id": first["id"] + "+" + row["id"], "amount": duration,
                               "unit": "분", "interval_end": row["occurred_at"]})
            else:
                incomplete += 1
        elif _number(row.get("amount")) and row.get("unit") in {"시간", "분", "min", "h"}:
            duration = row["amount"] * (60 if row["unit"] in {"시간", "h"} else 1)
            result.append({**row, "amount": duration, "unit": "분"})
        else:
            incomplete += 1
    result.sort(key=lambda row: row["occurred_at"])
    return result, incomplete + len(pending)


def _within(rows, start, end):
    found = []
    for row in rows:
        stamp = _instant(row["occurred_at"]).astimezone(KST)
        if row.get("interval_end"):
            finish = _instant(row["interval_end"]).astimezone(KST)
            if stamp < end and finish > start:
                left, right = max(start, stamp), min(end, finish)
                found.append({**row, "occurred_at": left.isoformat(), "interval_end": right.isoformat(),
                              "amount": (right - left).total_seconds() / 60})
        elif start <= stamp < end:
            found.append(row)
    return found


def _statistics(rows, start, end, days, aggregate=True):
    groups = {}
    for row in rows:
        unit = row.get("unit") or "unknown"
        subtype = (row.get("subtype") or "all") if aggregate else "all"
        key = ":".join((row.get("type") or "unknown", subtype, unit))
        group = groups.setdefault(key, {"count": 0, "values": [], "daily": {}, "first": row, "last": None})
        group["count"] += 1
        group["last"] = row
        value = row.get("amount")
        if not _number(value):
            continue
        group["values"].append(float(value))
        # Keep the times of all tied extrema before the display-row limit.
        # The answer context can then distinguish event time from daily totals.
        extremes = group.setdefault("extremes", {})
        for name, better in (("min", min), ("max", max)):
            prior = extremes.get(name)
            if prior is None or value != prior["value"] and better(value, prior["value"]) == value:
                extremes[name] = {"value": value, "times": [row["occurred_at"]]}
            elif value == prior["value"]:
                prior["times"].append(row["occurred_at"])
        stamp = _instant(row["occurred_at"]).astimezone(KST)
        if row.get("interval_end"):
            finish = _instant(row["interval_end"]).astimezone(KST)
            while stamp < finish:
                boundary = min(finish, datetime.combine(stamp.date() + timedelta(days=1), time.min, KST))
                date = stamp.date().isoformat()
                group["daily"][date] = group["daily"].get(date, 0) + (boundary - stamp).total_seconds() / 60
                stamp = boundary
        else:
            date = stamp.date().isoformat()
            group["daily"][date] = group["daily"].get(date, 0) + value
    for group in groups.values():
        values = group.pop("values")
        for name, extreme in group.pop("extremes", {}).items():
            group[name + "_at"] = sorted(set(extreme["times"]))
        total = sum(values) if values else None
        group.update({"numeric_count": len(values), "total": round(total, 4) if total is not None else None,
                      "mean": round(total / len(values), 4) if values else None,
                      "max": max(values) if values else None, "min": min(values) if values else None,
                      "daily_mean": round(total / len(group["daily"]), 4) if group["daily"] else None,
                      "daily": {k: round(v, 4) for k, v in group["daily"].items()},
                      "recorded_days": len(group["daily"]), "period_days": days,
                      "mean_basis": "days_with_numeric_records; missing days are not zero"})
        if group["daily"]:
            group["max_day"] = max(group["daily"], key=group["daily"].get)
            group["min_day"] = min(group["daily"], key=group["daily"].get)
    result = {"count": len(rows), "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()}, "groups": groups}
    if aggregate:
        result["totals_by_type_and_unit"] = _statistics(
            rows, start, end, days, aggregate=False)["groups"]
    return result
