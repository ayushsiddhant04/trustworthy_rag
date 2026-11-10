"""Conservative, context-aware number and date mismatch checks.

Compares values only when they refer to the same semantic field:
- Dates & deadlines against dates & deadlines (with canonical date normalization).
- Fees & currency amounts against fees & currency amounts.
- Attendance & percentages against percentages.
- Durations against durations.
- Counts/students against counts/students.

Four-digit calendar years (e.g., 2026) are preserved as date components and are
never misidentified as standalone numbers or fees.
"""
from __future__ import annotations

import re
from typing import Any

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

DATE_PATTERNS = [
    # YYYY-MM-DD
    re.compile(r"\b(?P<year>19\d\d|20\d\d)-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b"),
    # DD/MM/YYYY or DD-MM-YYYY
    re.compile(r"\b(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P<year>\d{2,4})\b"),
    # 15 November 2026 / 15th Nov 2026 / 15 November
    re.compile(r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month_name>[A-Za-z]+)(?:[,\s]+(?P<year>19\d\d|20\d\d))?\b", re.IGNORECASE),
    # November 15, 2026 / Nov 15 2026 / November 15
    re.compile(r"\b(?P<month_name>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:[,\s]+(?P<year>19\d\d|20\d\d))?\b", re.IGNORECASE),
]

CUES = [
    "deadline", "due date", "last date", "submission date", "exam date",
    "attendance", "percentage", "percent", "fee", "fees", "amount",
    "duration", "days", "weeks", "months", "marks", "score", "students",
]

VALUE_RE = re.compile(
    r"(?:\b\d{4}-\d{1,2}-\d{1,2}\b|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s*\d{0,4}\b|"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b|"
    r"₹\s*\d[\d,]*(?:\.\d+)?|"
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|days?|weeks?|months?|years?)?\b)",
    re.IGNORECASE,
)


def extract_dates(text: str) -> tuple[list[dict], list[tuple[int, int]]]:
    """Extract and normalize all calendar dates from text."""
    dates = []
    spans = []
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            start, end = m.start(), m.end()
            if any(s[0] <= start < s[1] or s[0] < end <= s[1] for s in spans):
                continue
            gd = m.groupdict()
            year = int(gd["year"]) if gd.get("year") else None
            if year and year < 100:
                year += 2000
            if "month_name" in gd and gd["month_name"]:
                m_str = gd["month_name"].lower()
                if m_str not in MONTH_MAP:
                    continue
                month = MONTH_MAP[m_str]
            else:
                month = int(gd["month"])
            day = int(gd["day"])
            if 1 <= month <= 12 and 1 <= day <= 31:
                spans.append((start, end))
                raw_clean = m.group(0).strip(" .,:;()")
                canonical = f"{year:04d}-{month:02d}-{day:02d}" if year else f"--{month:02d}-{day:02d}"
                dates.append({
                    "year": year,
                    "month": month,
                    "day": day,
                    "canonical": canonical,
                    "raw": raw_clean,
                    "span": (start, end),
                })
    return dates, spans


def dates_compatible(d1: dict, d2: dict) -> bool:
    """Check if two normalized dates match (handling optional year)."""
    if d1["month"] != d2["month"] or d1["day"] != d2["day"]:
        return False
    if d1["year"] is not None and d2["year"] is not None and d1["year"] != d2["year"]:
        return False
    return True


def extract_fees(text: str, date_spans: list[tuple[int, int]]) -> list[dict]:
    """Extract currency amounts, ignoring calendar years and dates."""
    curr_pat = re.compile(
        r"(?:₹|rs\.?|inr|\$)\s*(\d[\d,]*(?:\.\d+)?)|"
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:rupees?|inr|dollars?|rs\.?)|"
        r"\b(?:fee|fees|cost|charge|tuition|amount)\s+(?:is|of|was|totaling)?\s*[:\s]*(\d[\d,]*(?:\.\d+)?)",
        re.IGNORECASE,
    )
    fees = []
    for m in curr_pat.finditer(text):
        start, end = m.start(), m.end()
        # Skip if inside an extracted date span
        if any(s[0] <= start < s[1] or s[0] < end <= s[1] for s in date_spans):
            continue
        v_str = m.group(1) or m.group(2) or m.group(3)
        try:
            val = float(v_str.replace(",", ""))
        except (ValueError, TypeError):
            continue
        # Avoid treating standalone 4-digit years (1900-2099) as bare fees unless explicit currency symbol
        if 1900 <= val <= 2099 and not (m.group(1) or m.group(2)):
            continue
        fees.append({"val": val, "raw": m.group(0).strip(" .,:;()")})
    return fees


def extract_percentages(text: str) -> list[dict]:
    """Extract percentage figures from text."""
    pct_pat = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent|per\s*cent)\b", re.IGNORECASE)
    pcts = []
    for m in pct_pat.finditer(text):
        try:
            val = float(m.group(1).replace(",", ""))
            pcts.append({"val": val, "raw": m.group(0).strip(" .,:;()")})
        except (ValueError, TypeError):
            continue
    # Contextual attendance cue: e.g. "attendance ... 75"
    if not pcts:
        att_pat = re.compile(r"\battendance\b[^.!?\n]{0,35}\b(\d{1,3})\b", re.IGNORECASE)
        for m in att_pat.finditer(text):
            try:
                val = float(m.group(1))
                if 0 <= val <= 100:
                    pcts.append({"val": val, "raw": f"{int(val)}%"})
            except (ValueError, TypeError):
                continue
    return pcts


def extract_durations(text: str) -> list[dict]:
    """Extract duration expressions from text."""
    dur_pat = re.compile(r"\b(\d+)\s*(days?|weeks?|months?|years?)\b", re.IGNORECASE)
    durs = []
    for m in dur_pat.finditer(text):
        val = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        durs.append({"val": val, "unit": unit, "raw": m.group(0).strip(" .,:;()")})
    return durs


def extract_counts(text: str, cue: str) -> list[dict]:
    """Extract counts associated with a specific count cue (students, marks, etc.)."""
    cnt_pat = re.compile(rf"\b(\d+)\s*{re.escape(cue)}\b|\b{re.escape(cue)}\s*[:\s]*(\d+)\b", re.IGNORECASE)
    counts = []
    for m in cnt_pat.finditer(text):
        v_str = m.group(1) or m.group(2)
        try:
            val = int(v_str)
            # Four-digit years (1900-2099) should not accidentally match as student counts
            if 1900 <= val <= 2099 and not text[max(0, m.start()-10):m.end()+10].lower().startswith(cue.lower()):
                continue
            counts.append({"val": val, "raw": m.group(0).strip(" .,:;()")})
        except (ValueError, TypeError):
            continue
    return counts


def _values_near_cue(text: str, cue: str, radius: int = 90) -> list[str]:
    """Backward-compatible helper: extracts values near cue snapping to word boundaries."""
    values = []
    lower = text.lower()
    for match in re.finditer(re.escape(cue), lower):
        start = max(0, match.start() - 25)
        # Snap start backward to preceding word boundary to prevent cutting words (e.g., 2026 -> 026)
        while start > 0 and text[start].isalnum():
            start -= 1
        end = min(len(text), match.end() + radius)
        while end < len(text) and text[end - 1].isalnum():
            end += 1
        context = text[start:end]
        values.extend(v.strip(" .,:;()") for v in VALUE_RE.findall(context))
    return list(dict.fromkeys(v for v in values if v))


def check_numeric_conflict(claim: str, evidence_passages: Any) -> dict:
    """Context-aware comparison across distinct semantic fields.

    Only compares:
    - Date vs Date
    - Fee vs Fee
    - Percentage vs Percentage
    - Duration vs Duration
    - Count vs Count
    """
    if isinstance(evidence_passages, str):
        evidence_passages = [evidence_passages]
    evidence_passages = evidence_passages or []

    # Combine evidence texts
    ev_texts = [
        p.get("text", "") if isinstance(p, dict) else str(p)
        for p in evidence_passages
    ]
    all_ev_text = " \n ".join(ev_texts)

    claim_text = str(claim or "")
    conflicts = []

    # 1. FIELD: DATE / DEADLINE
    date_cues = ["deadline", "due date", "last date", "submission date", "exam date", "scheduled date", "date"]
    has_date_cue = any(c in claim_text.lower() for c in date_cues)
    claim_dates, claim_date_spans = extract_dates(claim_text)
    ev_dates, ev_date_spans = extract_dates(all_ev_text)

    if claim_dates and has_date_cue:
        ev_has_date_cue = any(c in all_ev_text.lower() for c in date_cues)
        if ev_dates and ev_has_date_cue:
            # Check if each claim date is supported by at least one compatible evidence date
            for cd in claim_dates:
                if not any(dates_compatible(cd, ed) for ed in ev_dates):
                    ev_rep = [ed["raw"] for ed in ev_dates]
                    conflicts.append({
                        "cue": "deadline",
                        "claim_values": [cd["raw"]],
                        "evidence_values": ev_rep,
                        "reason": f"Possible deadline mismatch: claim says {cd['raw']}; evidence says {', '.join(ev_rep)}.",
                    })

    # 2. FIELD: FEE / CURRENCY
    fee_cues = ["fee", "fees", "registration fee", "application fee", "tuition", "cost", "charge", "amount", "rupees", "inr", "rs", "₹"]
    has_fee_cue = any(c in claim_text.lower() for c in fee_cues)
    claim_fees = extract_fees(claim_text, claim_date_spans)
    ev_fees = extract_fees(all_ev_text, ev_date_spans)

    if claim_fees and has_fee_cue:
        ev_has_fee_cue = any(c in all_ev_text.lower() for c in fee_cues)
        if ev_fees and ev_has_fee_cue:
            for cf in claim_fees:
                if not any(abs(cf["val"] - ef["val"]) < 1e-4 for ef in ev_fees):
                    c_str = str(int(cf["val"])) if cf["val"].is_integer() else str(cf["val"])
                    e_strs = [str(int(ef["val"])) if ef["val"].is_integer() else str(ef["val"]) for ef in ev_fees]
                    conflicts.append({
                        "cue": "fee",
                        "claim_values": [c_str],
                        "evidence_values": e_strs,
                        "reason": f"Possible fee mismatch: claim says {c_str}; evidence says {', '.join(e_strs)}.",
                    })

    # 3. FIELD: PERCENTAGE / ATTENDANCE
    pct_cues = ["attendance", "percentage", "percent", "%"]
    has_pct_cue = any(c in claim_text.lower() for c in pct_cues)
    claim_pcts = extract_percentages(claim_text)
    ev_pcts = extract_percentages(all_ev_text)

    if claim_pcts and has_pct_cue:
        ev_has_pct_cue = any(c in all_ev_text.lower() for c in pct_cues)
        if ev_pcts and ev_has_pct_cue:
            for cp in claim_pcts:
                if not any(abs(cp["val"] - ep["val"]) < 1e-4 for ep in ev_pcts):
                    c_str = f"{int(cp['val'])}%" if cp["val"].is_integer() else f"{cp['val']}%"
                    e_strs = [f"{int(ep['val'])}%" if ep["val"].is_integer() else f"{ep['val']}%" for ep in ev_pcts]
                    conflicts.append({
                        "cue": "percentage",
                        "claim_values": [c_str],
                        "evidence_values": e_strs,
                        "reason": f"Possible percentage mismatch: claim says {c_str}; evidence says {', '.join(e_strs)}.",
                    })

    # 4. FIELD: DURATION
    dur_cues = ["duration", "validity", "period", "valid for"]
    has_dur_cue = any(c in claim_text.lower() for c in dur_cues)
    claim_durs = extract_durations(claim_text)
    ev_durs = extract_durations(all_ev_text)

    if claim_durs and has_dur_cue:
        if ev_durs:
            for cd in claim_durs:
                if not any(cd["val"] == ed["val"] and cd["unit"] == ed["unit"] for ed in ev_durs):
                    c_str = cd["raw"]
                    e_strs = [ed["raw"] for ed in ev_durs]
                    conflicts.append({
                        "cue": "duration",
                        "claim_values": [c_str],
                        "evidence_values": e_strs,
                        "reason": f"Possible duration mismatch: claim says {c_str}; evidence says {', '.join(e_strs)}.",
                    })

    # 5. FIELD: COUNTS (students, marks, score, seats)
    count_cues = ["students", "marks", "score", "seats"]
    for cue in count_cues:
        if cue in claim_text.lower() and cue in all_ev_text.lower():
            claim_cnts = extract_counts(claim_text, cue)
            ev_cnts = extract_counts(all_ev_text, cue)
            if claim_cnts and ev_cnts:
                for cc in claim_cnts:
                    if not any(cc["val"] == ec["val"] for ec in ev_cnts):
                        c_str = str(cc["val"])
                        e_strs = [str(ec["val"]) for ec in ev_cnts]
                        conflicts.append({
                            "cue": cue,
                            "claim_values": [c_str],
                            "evidence_values": e_strs,
                            "reason": f"Possible {cue} mismatch: claim says {c_str}; evidence says {', '.join(e_strs)}.",
                        })

    return {
        "numeric_conflict": bool(conflicts),
        "conflicts": conflicts,
        "reason": conflicts[0]["reason"] if conflicts else "No clear mismatch found by the numeric/date rule.",
    }
