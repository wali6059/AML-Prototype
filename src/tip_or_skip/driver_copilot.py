from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DriverContext:
    zone_risk: pd.DataFrame
    zone_names: tuple[str, ...]


ALIASES = {
    "jfk": "JFK Airport",
    "kennedy airport": "JFK Airport",
    "lga": "LaGuardia Airport",
    "laguardia": "LaGuardia Airport",
    "times square": "Times Sq/Theatre District",
    "midtown": "Midtown Center",
    "financial district": "Financial District North",
}


def build_driver_context(zone_risk: pd.DataFrame | None, zone_names: Iterable[str]) -> DriverContext:
    zones = zone_risk.copy() if zone_risk is not None else pd.DataFrame()
    unique_names = tuple(dict.fromkeys(str(zone) for zone in zone_names if str(zone).strip()))
    return DriverContext(zone_risk=zones, zone_names=unique_names)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9/& ]+", " ", text.lower())).strip()


def _dedupe_in_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _zone_positions(text: str, context: DriverContext) -> list[tuple[int, str]]:
    normalized = _normalize(text)
    positions: list[tuple[int, str]] = []
    for alias, zone in ALIASES.items():
        if zone not in context.zone_names:
            continue
        idx = normalized.find(_normalize(alias))
        if idx >= 0:
            positions.append((idx, zone))
    for zone in context.zone_names:
        idx = normalized.find(_normalize(zone))
        if idx >= 0:
            positions.append((idx, zone))
    return sorted(positions, key=lambda item: item[0])


def extract_zone_mentions(question: str, context: DriverContext) -> list[str]:
    positions = _zone_positions(question, context)
    if not positions:
        return []

    lower = _normalize(question)
    option_markers = ["option", "choose between", "between", "versus", " vs "]
    marker_positions = [lower.find(marker) for marker in option_markers if lower.find(marker) >= 0]
    if marker_positions:
        start = min(marker_positions)
        option_positions = [(pos, zone) for pos, zone in positions if pos >= start]
        if len(option_positions) >= 2:
            return _dedupe_in_order(zone for _, zone in option_positions)

    return _dedupe_in_order(zone for _, zone in positions)


def _filter_zone_rows(
    context: DriverContext,
    zone_names: list[str],
    taxi_type: str | None = None,
    min_rows: int = 50,
) -> pd.DataFrame:
    if context.zone_risk.empty:
        return pd.DataFrame()
    rows = context.zone_risk[context.zone_risk["pickup_zone"].isin(zone_names)].copy()
    if taxi_type and "taxi_type" in rows.columns:
        typed = rows[rows["taxi_type"] == taxi_type]
        if not typed.empty:
            rows = typed
    if "rows" in rows.columns:
        enough = rows[rows["rows"] >= min_rows]
        if not enough.empty:
            rows = enough
    return rows


def compare_zone_options(
    zone_names: list[str],
    context: DriverContext,
    taxi_type: str | None = None,
    min_rows: int = 50,
) -> pd.DataFrame:
    rows = _filter_zone_rows(context, zone_names, taxi_type=taxi_type, min_rows=min_rows)
    if rows.empty:
        return rows
    rows["score"] = rows["expected_tip"]
    return rows.sort_values(["score", "rows"], ascending=False).reset_index(drop=True)


def _fmt_money(value: object) -> str:
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "unknown"


def _fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "unknown"


def _taxi_type_from_question(question: str) -> str | None:
    lower = question.lower()
    if "green" in lower:
        return "green"
    if "yellow" in lower:
        return "yellow"
    return None


def _quality_label(row: pd.Series, context: DriverContext) -> str:
    rows = _filter_zone_rows(context, [str(row["pickup_zone"])], taxi_type=row.get("taxi_type"), min_rows=1)
    universe = context.zone_risk
    if "rows" in universe.columns:
        universe = universe[universe["rows"] >= 50]
    if "taxi_type" in row and "taxi_type" in universe.columns:
        typed = universe[universe["taxi_type"] == row["taxi_type"]]
        if not typed.empty:
            universe = typed
    if universe.empty or "expected_tip" not in universe.columns:
        return "unknown"
    value = float(row["expected_tip"])
    q75 = float(universe["expected_tip"].quantile(0.75))
    q40 = float(universe["expected_tip"].quantile(0.40))
    if value >= q75:
        return "strong"
    if value >= q40:
        return "solid"
    return "lower"


def _row_line(row: pd.Series) -> str:
    return (
        f"**{row['pickup_zone']}** ({row.get('taxi_type', 'taxi')}, {row.get('pickup_borough', 'unknown borough')}): "
        f"expected tip {_fmt_money(row.get('expected_tip'))}, downside Q10 {_fmt_money(row.get('q10_tip'))}, "
        f"tip probability {_fmt_pct(row.get('predicted_tip_probability'))}, "
        f"observed trips {_fmt_int(row.get('rows'))}"
    )


def _compare_answer(question: str, context: DriverContext, zones: list[str]) -> str:
    taxi_type = _taxi_type_from_question(question)
    ranked = compare_zone_options(zones, context, taxi_type=taxi_type)
    if ranked.empty:
        return (
            "I found the area names, but the final zone-risk table does not have enough held-out trips for those options. "
            "Try using exact TLC zone names such as JFK Airport, LaGuardia Airport, Midtown Center, or Battery Park City."
        )
    best = ranked.iloc[0]
    runner_up = ranked.iloc[1] if len(ranked) > 1 else None
    lines = [
        f"I recommend **{best['pickup_zone']}** for the better expected electronic tip signal.",
        "",
        "Comparison from the final model's zone-risk table:",
    ]
    for _, row in ranked.head(4).iterrows():
        lines.append(f"- {_row_line(row)}")
    if runner_up is not None:
        delta = float(best["expected_tip"]) - float(runner_up["expected_tip"])
        lines.append("")
        lines.append(
            f"The expected-tip gap between {best['pickup_zone']} and {runner_up['pickup_zone']} is {_fmt_money(delta)}. "
            "Use the Q10 number as the downside-risk check: higher Q10 means the lower-tail outcome is better."
        )
    lines.append("")
    lines.append(
        "This is decision support, not a guarantee. It predicts recorded electronic tips only; cash tips are not observed in TLC data."
    )
    return "\n".join(lines)


def _single_zone_answer(question: str, context: DriverContext, zone: str) -> str:
    taxi_type = _taxi_type_from_question(question)
    ranked = compare_zone_options([zone], context, taxi_type=taxi_type, min_rows=1)
    if ranked.empty:
        return f"I recognized **{zone}**, but I do not have enough final model evidence for that area."
    row = ranked.iloc[0]
    quality = _quality_label(row, context)
    label_text = {
        "strong": "a strong tip area relative to comparable zones",
        "solid": "a solid but not top-tier tip area",
        "lower": "a lower expected-tip area relative to comparable zones",
        "unknown": "an area with available model evidence",
    }[quality]
    return (
        f"**{zone}** looks like **{label_text}**.\n\n"
        f"- {_row_line(row)}\n\n"
        "For a ride decision, compare it against the other option's expected tip and Q10 downside estimate. "
        "If you give me two areas, I can recommend which one has the stronger model signal."
    )


def answer_driver_question(question: str, context: DriverContext) -> str:
    clean = (question or "").strip()
    if not clean:
        return (
            "Tell me your pickup area, ride option areas, taxi type, and rough time or fare. "
            "Example: I'm at Midtown Center at 6pm and can choose JFK Airport or LaGuardia Airport."
        )
    zones = extract_zone_mentions(clean, context)
    if len(zones) >= 2:
        return _compare_answer(clean, context, zones)
    if len(zones) == 1:
        return _single_zone_answer(clean, context, zones[0])
    return (
        "I need at least one exact NYC TLC zone or area name to ground the recommendation. "
        "Try: 'I am at Midtown Center and can choose JFK Airport or LaGuardia Airport. Which ride should I take?'"
    )
