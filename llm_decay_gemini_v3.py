#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


BASE_TTL_MIN_S = 0.1
BASE_TTL_MAX_S = 600.0


class TagUpdate(BaseModel):
    tag_key: str = Field(description="Exact compound tag from the current TTL policy table")
    ttl_base_s: float = Field(
        ge=BASE_TTL_MIN_S,
        le=BASE_TTL_MAX_S,
        description="Proposed tag-wise base TTL in seconds",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model-reported supervisory confidence for this proposal",
    )
    reason: str = Field(default="", description="Concise rationale grounded in supplied evidence")


class GeminiProposal(BaseModel):
    updates: List[TagUpdate] = Field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class EventCase:
    mission_id: str
    event_id: str
    tag_key: str
    detour_ratio: Optional[float] = None
    old_len: Optional[float] = None
    new_len: Optional[float] = None
    vlm_confidence: Optional[float] = None
    evidence: Optional[str] = None
    applied_ttl_s: Optional[float] = None
    timestamp: Optional[float] = None
    tag_group_id: Optional[str] = None
    repeat_count_in_mission: int = 1
    same_obstacle_reencountered: bool = False
    depth_ratio: Optional[float] = None
    depth_corrected_ttl_s: Optional[float] = None
    vlm_dt_used_s: Optional[float] = None
    ttl_base_s: Optional[float] = None
    stale_cost_evidence: Optional[bool] = None
    replanning_after_expiration: Optional[bool] = None
    approval_mode: Optional[str] = None
    approval_status: Optional[str] = None
    proposed_ttl_s: Optional[float] = None
    llm_proposal_confidence: Optional[float] = None
    proposal_reason: Optional[str] = None
    approval_timestamp: Optional[float] = None
    human_feedback: Optional[str] = None
    human_feedback_norm: Optional[str] = None
    human_feedback_present: bool = False


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def normalize_tag_key(tag_key: Any) -> str:
    s = str(tag_key or "").strip().lower().replace("|", ":")
    s = re.sub(r"\s+", "", s)
    aliases = {
        "human": "person",
        "pedestrian": "person",
        "worker": "person",
        "people": "person",
        "truck": "forklift",
        "vehicle": "forklift",
        "trolley": "cart",
        "dolly": "cart",
        "handtruck": "cart",
        "shoppingcart": "cart",
    }
    if ":" in s:
        base, motion = s.split(":", 1)
        base = aliases.get(base, base)
        if motion in {"slow", "fast"} and base:
            return f"{base}:{motion}"
        return ""
    return aliases.get(s, s)


def _read_json(path: str, default: Any) -> Any:
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: str, obj: Any) -> None:
    p = Path(os.path.expanduser(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def _read_key_file(path: str) -> Optional[str]:
    try:
        value = Path(os.path.expanduser(path)).read_text(encoding="utf-8").strip()
        return value or None
    except Exception:
        return None


def resolve_api_key(api_key_file: Optional[str] = None) -> Tuple[Optional[str], str]:
    value = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if value:
        return value.strip(), "environment"
    if api_key_file:
        value = _read_key_file(api_key_file)
        if value:
            return value, os.path.expanduser(api_key_file)
    for candidate in (
        "~/.config/policy_bridge/gemini_api_key.txt",
        "~/.config/gemini_api_key.txt",
        "~/.gemini_api_key",
        "~/STeP_Cost/.secrets/gemini_api_key.txt",
    ):
        value = _read_key_file(candidate)
        if value:
            return value, os.path.expanduser(candidate)
    return None, "not-found"


# -----------------------------------------------------------------------------
# Mission / archive parsing
# -----------------------------------------------------------------------------


def load_decay_table(path: str) -> Dict[str, Dict[str, float]]:
    raw = _read_json(path, {})
    table: Dict[str, Dict[str, float]] = {}
    if not isinstance(raw, dict):
        return table
    for key, value in raw.items():
        tag = normalize_tag_key(key)
        if not tag or ":" not in tag:
            continue
        try:
            ttl = float(value.get("ttl")) if isinstance(value, dict) else float(value)
        except Exception:
            continue
        table[tag] = {"ttl": min(BASE_TTL_MAX_S, max(BASE_TTL_MIN_S, ttl))}
    return table


def extract_current_cases(ms: Dict[str, Any]) -> List[EventCase]:
    mission_id = str(ms.get("mission_id") or "unknown_mission")
    out: List[EventCase] = []
    for ev in ms.get("events", []) or []:
        if not isinstance(ev, dict):
            continue
        tag = normalize_tag_key(ev.get("vlm_tag_key"))
        if not tag or ":" not in tag:
            continue
        vlm = ev.get("vlm") if isinstance(ev.get("vlm"), dict) else {}
        try:
            repeat_count = int(ev.get("tag_repeat_count_in_mission") or 1)
        except Exception:
            repeat_count = 1
        repeat_count = max(1, repeat_count)
        out.append(
            EventCase(
                mission_id=mission_id,
                event_id=str(ev.get("event") or ev.get("event_id") or ""),
                tag_key=tag,
                detour_ratio=_safe_float(ev.get("ratio")),
                old_len=_safe_float(ev.get("plan_prev_len")),
                new_len=_safe_float(ev.get("plan_new_len")),
                vlm_confidence=_safe_float(vlm.get("confidence")),
                evidence=str(vlm.get("evidence") or "") or None,
                applied_ttl_s=_safe_float(ev.get("applied_ttl_s")),
                timestamp=_safe_float(ev.get("timestamp_unix")),
                tag_group_id=str(ev.get("tag_group_id") or "") or None,
                repeat_count_in_mission=repeat_count,
                same_obstacle_reencountered=bool(repeat_count >= 2),
                depth_ratio=_safe_float(ev.get("depth_ratio")),
                depth_corrected_ttl_s=_safe_float(ev.get("depth_corrected_ttl_s")),
                vlm_dt_used_s=_safe_float(ev.get("vlm_dt_used_s")),
                ttl_base_s=_safe_float(ev.get("ttl_base_s")),
                stale_cost_evidence=(
                    bool(ev.get("stale_cost_evidence"))
                    if ev.get("stale_cost_evidence") is not None
                    else None
                ),
                replanning_after_expiration=(
                    bool(ev.get("replanning_after_expiration"))
                    if ev.get("replanning_after_expiration") is not None
                    else None
                ),
            )
        )
    return out


def load_archive(path: str) -> List[EventCase]:
    raw = _read_json(path, [])
    # Migrate the older {"cases": [...]} format if encountered.
    rows = raw.get("cases", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        return []
    out: List[EventCase] = []
    allowed = set(EventCase.__dataclass_fields__.keys())
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean = {k: v for k, v in row.items() if k in allowed}
        if "vlm_confidence" not in clean and "confidence" in row:
            clean["vlm_confidence"] = row.get("confidence")
        try:
            clean["tag_key"] = normalize_tag_key(clean.get("tag_key"))
            out.append(EventCase(**clean))
        except Exception:
            continue
    return out


def current_tag_counts(cases: List[EventCase]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for case in cases:
        counts[case.tag_key] = counts.get(case.tag_key, 0) + 1
    return counts


def recent_same_tag_cases(
    archive: List[EventCase],
    tag: str,
    max_cases: int,
    current_mission_id: str,
) -> List[EventCase]:
    rows = [
        c
        for c in archive
        if c.tag_key == tag and c.mission_id != current_mission_id
    ]
    rows.sort(key=lambda c: c.timestamp or c.approval_timestamp or 0.0)
    return rows[-max(1, max_cases) :]


def confirmed_insufficient_bound(cases: List[EventCase]) -> Optional[float]:
    """Maximum policy-before base TTL at which repeat>=2 was observed."""
    vals = [
        c.ttl_base_s
        for c in cases
        if c.repeat_count_in_mission >= 2 and c.ttl_base_s is not None
    ]
    return max(vals) if vals else None


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------


def _fmt(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    return "?" if value is None else f"{value:.{digits}f}{suffix}"


def build_prompt(
    current_cases: List[EventCase],
    decay_table: Dict[str, Dict[str, float]],
    archive: List[EventCase],
    max_recent_cases: int,
) -> str:
    tags = sorted({
        c.tag_key
        for c in current_cases
    })

    mission_id = (
        current_cases[0].mission_id
        if current_cases
        else "unknown_mission"
    )

    lines: List[str] = [
        "You are a TTL policy optimizer for a robot navigation system.",
        "depth_corrected_ttl = TTL_base * (1 - depth_ratio).",
        "",
        "Goal:",
        "Find the minimum TTL_base so the robot encounters each obstacle",
        "exactly once (repeat=1), avoids repeated detours/replanning,",
        "and minimizes stale residual cost.",
        "Always propose ttl_base_s, not depth_corrected_ttl.",
        "Bias toward smaller values.",
        "",
        "IMPORTANT:",
        f"You MUST use ONLY these exact tag_key values: {', '.join(tags)}.",
        "Do NOT invent or substitute other tag names",
        '(e.g., do not use "vehicle" if "forklift" is listed).',
        "",
        "Rules:",
        "R1 repeat>=2:",
        "TTL_base too short -> RAISE, but incrementally.",
        "Do NOT jump to a large value at once.",
        "Small consistent raises are preferred.",
        "",
        "R2 repeat=1, depth<0.5:",
        "depth_corrected_ttl was long -> HOLD or small decrease",
        "(not below confirmed_insufficient_bound).",
        "",
        "R3 repeat=1, depth>=0.5:",
        "depth_corrected_ttl was short -> RAISE proportionally.",
        "Treat as weak signal.",
        "",
        "R4 Minimum:",
        "Prefer lowest TTL_base with past repeat=1.",
        "",
        "R5 Converged:",
        "HOLD with confidence=0.9 if:",
        "- confirmed_insufficient_bound exists,",
        "- current TTL_base > bound, and",
        "- no stale-cost evidence is observed in the current",
        "or retrieved cases.",
        "",
        "confirmed_insufficient_bound:",
        "max TTL_base where repeat>=2 ever occurred.",
        "Do NOT propose at or below this.",
        "Confidence -0.25 if violated.",
        "",
        "Decay table (current):",
    ]

    for tag in tags:
        ttl = decay_table.get(
            tag,
            {},
        ).get("ttl")

        lines.append(
            f"{tag}: {_fmt(ttl, 3, 's')}"
        )

    lines += [
        "",
        "Current mission:",
    ]

    for tag in tags:
        cases = [
            c
            for c in current_cases
            if c.tag_key == tag
        ]

        max_repeat = max(
            (
                c.repeat_count_in_mission
                for c in cases
            ),
            default=1,
        )

        reencounters = sum(
            1
            for c in cases
            if c.repeat_count_in_mission >= 2
        )

        lines.append(
            f"{tag}: count={len(cases)} "
            f"max_repeat={max_repeat} "
            f"reencounters={reencounters}"
        )

        for c in cases:
            lines.append(
                f"tag={tag} "
                f"repeat={c.repeat_count_in_mission} "
                f"ttl_base={_fmt(c.ttl_base_s, 3, 's')} "
                f"depth={_fmt(c.depth_ratio, 3)} "
                f"applied={_fmt(c.applied_ttl_s, 3, 's')}"
            )

    lines += [
        "",
        "Past cases:",
    ]

    for tag in tags:
        recent = recent_same_tag_cases(
            archive,
            tag,
            max_recent_cases,
            mission_id,
        )

        bound = confirmed_insufficient_bound(
            recent
        )

        failures = [
            c
            for c in recent
            if c.repeat_count_in_mission >= 2
        ]

        successful = [
            c
            for c in recent
            if c.repeat_count_in_mission == 1
        ]

        lines.append(
            f"{tag}: confirmed_insufficient_bound="
            + (
                f"{bound:.3f}s"
                if bound is not None
                else "unknown"
            )
        )

        lines.append(
            f"(n={len(failures)} failures)"
        )

        lines.append(
            "repeat=1 cases "
            f"(oldest->newest, n={len(successful)}):"
        )

        for c in successful:
            lines.append(
                f"ttl={_fmt(c.ttl_base_s, 3, 's')} "
                f"depth={_fmt(c.depth_ratio, 3)} "
                f"applied={_fmt(c.applied_ttl_s, 3, 's')}"
            )

    lines += [
        "",
        "Output JSON only.",
        f"Use ONLY these tag_key values: {', '.join(tags)}",
        "{",
        '  "updates": [',
        "    {",
        '      "tag_key": "...",',
        '      "ttl_base_s": 0.0,',
        '      "confidence": 0.0,',
        '      "reason": "..."',
        "    }",
        "  ],",
        '  "notes": null',
        "}",
    ]

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Gemini call / parsing
# -----------------------------------------------------------------------------


def parse_proposal_text(text: str) -> GeminiProposal:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty Gemini response")

    candidates = [raw]
    if raw.startswith('"') and raw.endswith('"'):
        try:
            unquoted = json.loads(raw)
            if isinstance(unquoted, str):
                candidates.append(unquoted.strip())
        except Exception:
            pass

    stripped = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    candidates.append(stripped)
    m = re.search(r"\{[\s\S]*\}", stripped)
    if m:
        candidates.append(m.group(0))

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return GeminiProposal.model_validate_json(candidate)
        except Exception as e:
            last_error = e
    raise ValueError(f"could not parse structured proposal: {last_error}")


def call_gemini(
    model: str,
    prompt: str,
    api_key_file: Optional[str],
    max_attempts: int = 6,
) -> GeminiProposal:
    api_key, source = resolve_api_key(api_key_file)
    if not api_key:
        raise RuntimeError("Gemini API key not found")

    print(f"[INFO] Gemini API key source: {source}")

    http_options = None
    try:
        http_options = types.HttpOptions(timeout=30000)
    except Exception:
        http_options = None

    client = (
        genai.Client(api_key=api_key, http_options=http_options)
        if http_options is not None
        else genai.Client(api_key=api_key)
    )

    config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=0.9,
        response_mime_type="application/json",
        response_schema=GeminiProposal,
    )

    backoffs = [1, 2, 3, 4, 5]
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, GeminiProposal):
                return parsed
            if isinstance(parsed, dict):
                return GeminiProposal.model_validate(parsed)
            return parse_proposal_text(getattr(response, "text", "") or "")
        except Exception as e:
            last_error = e
            if attempt >= max_attempts - 1:
                break
            delay = backoffs[min(attempt, len(backoffs) - 1)]
            print(
                f"[WARN] Gemini attempt {attempt + 1}/{max_attempts} failed: "
                f"{type(e).__name__}: {e}; retrying in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(f"Gemini call failed after {max_attempts} attempts: {last_error}")


# -----------------------------------------------------------------------------
# Application / optional standalone archive append
# -----------------------------------------------------------------------------

def filter_and_guard_proposal(
    proposal: GeminiProposal,
    allowed_tags: List[str],
    archive: List[EventCase],
    current_mission_id: str,
) -> GeminiProposal:
    # These arguments are retained for CLI/API compatibility.
    del archive
    del current_mission_id

    allowed = set(allowed_tags)
    out: List[TagUpdate] = []

    for update in proposal.updates:
        tag = normalize_tag_key(
            update.tag_key
        )

        if tag not in allowed:
            continue

        ttl = min(
            BASE_TTL_MAX_S,
            max(
                BASE_TTL_MIN_S,
                float(update.ttl_base_s),
            ),
        )

        out.append(
            TagUpdate(
                tag_key=tag,
                ttl_base_s=ttl,
                confidence=min(
                    1.0,
                    max(
                        0.0,
                        float(update.confidence),
                    ),
                ),
                reason=str(
                    update.reason or ""
                ),
            )
        )

    return GeminiProposal(
        updates=out,
        notes=proposal.notes,
    )


def apply_updates(decay_table: Dict[str, Dict[str, float]], proposal: GeminiProposal) -> int:
    count = 0
    for update in proposal.updates:
        tag = normalize_tag_key(update.tag_key)
        if not tag:
            continue
        decay_table[tag] = {"ttl": float(update.ttl_base_s)}
        count += 1
    return count


def append_standalone_archive(
    path: str,
    current_cases: List[EventCase],
    proposal: GeminiProposal,
    max_cases: int,
) -> None:
    raw = _read_json(path, [])
    rows = raw.get("cases", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        rows = []
    proposal_map = {u.tag_key: u for u in proposal.updates}
    index = {
        (str(r.get("mission_id") or ""), str(r.get("event_id") or ""), normalize_tag_key(r.get("tag_key"))): i
        for i, r in enumerate(rows)
        if isinstance(r, dict)
    }
    for case in current_cases:
        update = proposal_map.get(case.tag_key)
        if update is None:
            continue
        row = asdict(case)
        row.update(
            {
                "approval_mode": "auto",
                "approval_status": "auto",
                "proposed_ttl_s": float(update.ttl_base_s),
                "llm_proposal_confidence": float(update.confidence),
                "proposal_reason": update.reason,
                "approval_timestamp": time.time(),
            }
        )
        key = (case.mission_id, case.event_id, case.tag_key)
        if key in index:
            rows[index[key]] = row
        else:
            index[key] = len(rows)
            rows.append(row)
    _write_json(path, rows[-max(1, max_cases) :])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="mission_summary.json")
    ap.add_argument("--output", required=True, help="output mission summary path")
    ap.add_argument("--decay_table_path", required=True)
    ap.add_argument("--init_table_if_missing", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--update_mission_summary", default=None)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--api_key_file", default=None)
    ap.add_argument("--retrieval_archive_path", default="~/.ros/llm_decay_rag_archive.json")
    ap.add_argument("--retrieval_max_repeat1_cases", type=int, default=30)
    ap.add_argument("--append_to_archive", action="store_true")
    ap.add_argument("--archive_max_cases", type=int, default=2000)
    ap.add_argument("--debug_prompt", default="")
    args = ap.parse_args()

    if args.init_table_if_missing and not Path(os.path.expanduser(args.decay_table_path)).exists():
        _write_json(args.decay_table_path, {})

    mission = _read_json(args.input, {})
    if not isinstance(mission, dict):
        raise RuntimeError("mission summary must be a JSON object")

    decay_table = load_decay_table(args.decay_table_path)
    current_cases = extract_current_cases(mission)
    counts = current_tag_counts(current_cases)

    print(
        f"[INFO] mission_id={mission.get('mission_id', 'unknown_mission')} "
        f"events_total={len(mission.get('events', []) or [])} "
        f"eventful_compound_tags={len(current_cases)} unique_tags={len(counts)}"
    )

    if not current_cases:
        _write_json(args.output, mission)
        if args.update_mission_summary:
            _write_json(args.update_mission_summary, mission)
        print("[INFO] No complete semantic-motion events; nothing to update.")
        return 0

    archive = load_archive(args.retrieval_archive_path)
    prompt = build_prompt(
        current_cases=current_cases,
        decay_table=decay_table,
        archive=archive,
        max_recent_cases=max(1, args.retrieval_max_repeat1_cases),
    )

    if args.debug_prompt:
        Path(os.path.expanduser(args.debug_prompt)).write_text(prompt, encoding="utf-8")

    proposal = call_gemini(
        model=args.model,
        prompt=prompt,
        api_key_file=args.api_key_file,
        max_attempts=6,
    )
    proposal = filter_and_guard_proposal(
        proposal,
        allowed_tags=sorted(counts.keys()),
        archive=archive,
        current_mission_id=str(mission.get("mission_id") or "unknown_mission"),
    )

    payload = {
        "updates": [u.model_dump() for u in proposal.updates],
        "notes": proposal.notes,
    }
    print("[PROPOSAL_JSON] " + json.dumps(payload, ensure_ascii=False))
    for update in proposal.updates:
        print(
            f"  - {update.tag_key}: ttl_base_s={update.ttl_base_s:.6f} "
            f"confidence={update.confidence:.3f} # {update.reason}"
        )

    if not proposal.updates:
        _write_json(args.output, mission)
        return 0

    apply_now = bool(args.yes)
    if not args.yes:
        try:
            answer = input("Apply these TTL_base updates? [y/N] ").strip().lower()
            apply_now = answer in {"y", "yes"}
        except EOFError:
            apply_now = False

    if args.dry_run:
        apply_now = False

    mission_out = json.loads(json.dumps(mission))
    mission_out["llm_policy_update"] = {
        "model": args.model,
        "timestamp": time.time(),
        "proposal": payload,
        "applied": bool(apply_now),
        "dry_run": bool(args.dry_run),
    }

    if apply_now:
        n = apply_updates(decay_table, proposal)
        _write_json(args.decay_table_path, decay_table)
        print(f"[INFO] Applied updates={n} -> {os.path.expanduser(args.decay_table_path)}")
        if args.append_to_archive:
            append_standalone_archive(
                args.retrieval_archive_path,
                current_cases,
                proposal,
                args.archive_max_cases,
            )
    else:
        print("[INFO] Proposal not applied.")

    _write_json(args.output, mission_out)
    if args.update_mission_summary:
        _write_json(args.update_mission_summary, mission_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
